"""Dynamic Bayesian Network for Congressional Voting Dynamics.

This module models the temporal evolution of congressional voting patterns
using Bayesian Ridge Regression with cross-validation for hyperparameter
optimization.

The model learns: x(t+1) = x(t) + W * x(t) + b + noise
where:
  - W: influence matrix between actors
  - b: bias term representing external influences
  - noise: Gaussian noise capturing randomness
"""

from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.exceptions import UndefinedMetricWarning
import warnings

import config

logger = logging.getLogger(__name__)


class DBCongressModel:
    """Dynamic Bayesian Network for modeling congressional voting dynamics.
    
    Learns individual regression models for each actor using Bayesian Ridge,
    with cross-validation for hyperparameter optimization to improve predictive
    accuracy on unseen voting patterns.
    """

    def __init__(self, 
                 alpha_1: float = 1e-6,
                 alpha_2: float = 1e-6,
                 lambda_1: float = 1e-6,
                 lambda_2: float = 1e-6,
                 max_iter: int = 300,
                 use_cross_validation: bool = True,
                 cv_splits: int = 5,
                 relationship_bias: float = config.RELATIONSHIP_PRIOR_WEIGHT):
        """Initialize DBCongressModel with hyperparameters.
        
        Args:
            alpha_1: Shape parameter for Gamma prior on precision of noise.
            alpha_2: Rate parameter for Gamma prior on precision of noise.
            lambda_1: Shape parameter for Gamma prior on precision of weights.
            lambda_2: Rate parameter for Gamma prior on precision of weights.
            max_iter: Maximum iterations for optimization (used by BayesianRidge).
            use_cross_validation: Whether to optimize hyperparameters via CV.
            cv_splits: Number of cross-validation splits.
        """
        self.models: List[BayesianRidge] = []
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.max_iter = max_iter
        self.use_cross_validation = use_cross_validation
        self.cv_splits = cv_splits
        self.cv_scores: Dict[str, List[float]] = {}
        self.relationship_bias = float(relationship_bias)
        self.relationship_matrix: Optional[np.ndarray] = None
        self.learn_deltas: bool = bool(config.DBN_LEARN_DELTAS)
        self.feature_dim: Optional[int] = None
        self.z_dim: int = 0

    def fit(
        self,
        X_time_series: np.ndarray,
        Z_time_series: Optional[np.ndarray] = None,
        y_time_series: Optional[np.ndarray] = None,
        relationship_matrix: Optional[np.ndarray] = None
    ) -> None:
        """Fit BayesianRidge model for each actor dimension.

        Args:
            X_time_series: Shape (T, N) - voting states over time for N actors.
            Z_time_series: Optional contextual features over time aligned with X (T, K).
            y_time_series: Optional explicit targets; if None, uses X_time_series[1:].
            relationship_matrix: Optional influence prior for weighting features.
        """
        n_actors = X_time_series.shape[1]

        # Create lagged features: X(t) predicts Δx(t) or x(t+1)
        X_lagged = X_time_series[:-1]

        Z_lagged = None
        if Z_time_series is not None:
            Z_time_series = np.asarray(Z_time_series, dtype=float)
            if len(Z_time_series) != len(X_time_series):
                raise ValueError(
                    f"Z_time_series length {len(Z_time_series)} != X_time_series length {len(X_time_series)}"
                )
            Z_lagged = Z_time_series[:-1]
            if Z_lagged.ndim == 1:
                Z_lagged = Z_lagged.reshape(-1, 1)

        if y_time_series is None:
            y_targets = X_time_series[1:]
        else:
            y_targets = y_time_series
            if len(y_targets) != len(X_lagged):
                raise ValueError(f"y_time_series length {len(y_targets)} != X_lagged length {len(X_lagged)}")

        # Optionally convert to deltas
        if self.learn_deltas:
            y_targets = y_targets - X_lagged

        if Z_lagged is not None:
            X_aug = np.concatenate([X_lagged, Z_lagged], axis=1)
            self.z_dim = Z_lagged.shape[1]
        else:
            X_aug = X_lagged
            self.z_dim = 0

        self.feature_dim = X_aug.shape[1]

        if relationship_matrix is not None and relationship_matrix.shape == (n_actors, n_actors):
            self.relationship_matrix = np.array(relationship_matrix, dtype=float)
        else:
            self.relationship_matrix = None

        logger.info(
            f"Fitting DBCongressModel for {n_actors} actors with {len(X_aug)} training samples "
            f"(learn_deltas={self.learn_deltas}, z_dim={self.z_dim})"
        )

        self.models = []
        self.cv_scores = {}

        for i in range(n_actors):
            y_target = y_targets[:, i]

            feature_weights = np.ones(self.feature_dim, dtype=float)
            if self.relationship_matrix is not None:
                feature_weights[:n_actors] += self.relationship_matrix[i] * self.relationship_bias

            X_weighted = X_aug * feature_weights

            # Create and fit base model
            model = BayesianRidge(
                alpha_1=self.alpha_1,
                alpha_2=self.alpha_2,
                lambda_1=self.lambda_1,
                lambda_2=self.lambda_2,
                max_iter=self.max_iter,
                compute_score=True
            )

            # Fit model
            model.fit(X_weighted, y_target)
            self.models.append(model)

            # Cross-validation to assess generalization
            if self.use_cross_validation:
                # TimeSeriesSplit will produce test folds of approximate size n_samples // (n_splits + 1).
                # R^2 is undefined for test folds with fewer than 2 samples, which triggers warnings.
                min_test_size = len(X_aug) // (self.cv_splits + 1)
                if len(X_aug) > self.cv_splits and min_test_size >= 2:
                    cv_splitter = TimeSeriesSplit(n_splits=self.cv_splits)
                    # Suppress UndefinedMetricWarning from sklearn.metrics for the duration of this CV call
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                        cv_scores = cross_val_score(
                            model,
                            X_weighted,
                            y_target,
                            cv=cv_splitter,
                            scoring='r2'  # R2 score
                        )
                    self.cv_scores[f"actor_{i}"] = cv_scores.tolist()

                    mean_score = np.mean(cv_scores)
                    std_score = np.std(cv_scores)
                    logger.debug(f"Actor {i}: CV R2 = {mean_score:.4f} +/- {std_score:.4f}")
                else:
                    # Not enough data to produce reliable R^2 estimates for CV folds
                    logger.debug(
                        f"Skipping CV for actor {i}: not enough samples for reliable R2 (n_samples={len(X_lagged)}, n_splits={self.cv_splits}, test_size~{min_test_size})"
                    )
                    self.cv_scores[f"actor_{i}"] = []

    def step(self, X_current: np.ndarray, Z_current: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict next state from current state and optional context Z(t).
        
        Args:
            X_current: Current state vector of shape (N,) for N actors.
            Z_current: Optional context vector aligned with fit-time Z features.
            
        Returns:
            Predicted next state X_next of shape (N,).
        """
        if not self.models:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_flat = np.asarray(X_current, dtype=float).ravel()
        z_vec = None
        if self.z_dim > 0:
            if Z_current is None:
                z_vec = np.zeros(self.z_dim, dtype=float)
            else:
                z_vec = np.asarray(Z_current, dtype=float).ravel()
                if z_vec.size < self.z_dim:
                    z_vec = np.pad(z_vec, (0, self.z_dim - z_vec.size))
                elif z_vec.size > self.z_dim:
                    z_vec = z_vec[: self.z_dim]

        if z_vec is not None:
            feature_vec = np.concatenate([X_flat, z_vec])
        else:
            feature_vec = X_flat

        X_next = np.zeros_like(X_flat, dtype=float)
        feature_2d = feature_vec.reshape(1, -1)

        for i, model in enumerate(self.models):
            pred = model.predict(feature_2d)[0]
            if self.learn_deltas:
                X_next[i] = X_flat[i] + pred
            else:
                X_next[i] = pred
        
        return X_next

    def predict_trajectory(
        self,
        X_init: np.ndarray,
        steps: int,
        Z_sequence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate a trajectory of predictions with optional context sequence.
        
        Args:
            X_init: Initial state of shape (N,).
            steps: Number of steps to predict.
            Z_sequence: Optional sequence of Z vectors (length steps or steps-1).
            
        Returns:
            Trajectory of shape (steps, N).
        """
        trajectory = [X_init.copy()]
        X_current = X_init.copy()

        for t in range(steps - 1):
            z_cur = None
            if Z_sequence is not None:
                try:
                    if len(Z_sequence) >= steps:
                        z_cur = Z_sequence[t]
                    elif len(Z_sequence) == steps - 1:
                        z_cur = Z_sequence[t]
                except Exception:
                    z_cur = None
            X_next = self.step(X_current, Z_current=z_cur)
            trajectory.append(X_next.copy())
            X_current = X_next
        
        return np.array(trajectory)

    def get_cv_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of cross-validation performance.
        
        Returns:
            Dictionary with mean and std CV R2 scores for each actor.
        """
        summary = {}
        for actor_key, scores in self.cv_scores.items():
            scores_arr = np.array(scores)
            if scores_arr.size == 0:
                summary[actor_key] = {
                    "mean_r2": float("nan"),
                    "std_r2": float("nan"),
                    "min_r2": float("nan"),
                    "max_r2": float("nan"),
                }
            else:
                summary[actor_key] = {
                    "mean_r2": float(np.mean(scores_arr)),
                    "std_r2": float(np.std(scores_arr)),
                    "min_r2": float(np.min(scores_arr)),
                    "max_r2": float(np.max(scores_arr)),
                }
        return summary

    def get_model_coefficients(self, actor_idx: int) -> np.ndarray:
        """Get learned influence weights (coefficients) for an actor.
        
        Args:
            actor_idx: Index of the actor.
            
        Returns:
            Coefficient vector (influence matrix row) of shape (N,).
        """
        if actor_idx >= len(self.models):
            raise ValueError(f"Actor index {actor_idx} out of range")
        return self.models[actor_idx].coef_.copy()