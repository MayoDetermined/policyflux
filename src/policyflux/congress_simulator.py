"""High-level Keras-like API for congressional simulations.

Exposes a Keras-style interface:
- compile(): configure hyperparameters and regime scenario
- fit(): run static training pipeline (IPM export + DBN training) and build Congress system
- simulate(): run Monte Carlo simulations via CongressEngine
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Callable, TYPE_CHECKING

import numpy as np

from policyflux import config
from policyflux.congress.actors import CongressMan
from policyflux.congress.congress import TheCongress
from policyflux.data_collectors.actors_architectural_bureau import CongressMenBuilder
from policyflux.engine.symulations import CongressEngine
from policyflux.models.dbn import DBCongressModel
from policyflux.public_opinion.regime import PublicRegime

if TYPE_CHECKING:
    from policyflux.core.model import Model

logger = logging.getLogger(__name__)


class CongressSimulator:
    """Facade object that exposes compile/fit/simulate entrypoints."""

    def __init__(
        self,
        scenario: Optional[str] = None,
        use_gpu: bool = True,
        use_hmm_state: bool = True,
        actor_model: Optional['Model'] = None,
    ) -> None:
        self.scenario = scenario or config.SCENARIO
        self.use_gpu = use_gpu
        self.use_hmm_state = use_hmm_state
        self.actor_model = actor_model

        self.regime = PublicRegime(self.scenario)
        self.builder: Optional[CongressMenBuilder] = None
        self.actors_data = None
        self.actors: list[CongressMan] = []
        self.influence_matrix: Optional[np.ndarray] = None
        self.cosponsorship_matrix: Optional[np.ndarray] = None
        self.committee_matrix: Optional[np.ndarray] = None
        self.ipm_params: Optional[Dict[str, np.ndarray]] = None
        self.dbn_model: Optional[DBCongressModel] = None
        self.congress: Optional[TheCongress] = None
        self.engine: Optional[CongressEngine] = None

        self.compiled = False
        self.fitted = False

        self.dqn_params: Dict[str, Any] = {}
        self.dbn_params: Dict[str, Any] = {}
        self.lstm_params: Dict[str, Any] = {}

    def compile(
        self,
        scenario: Optional[str] = None,
        dqn_params: Optional[Dict[str, Any]] = None,
        dbn_params: Optional[Dict[str, Any]] = None,
        lstm_params: Optional[Dict[str, Any]] = None,
        use_hmm_state: Optional[bool] = None,
        actor_model: Optional['Model'] = None,
    ) -> "CongressSimulator":
        """Configure hyperparameters and regime scenario."""
        if scenario:
            self.scenario = scenario
            self.regime = PublicRegime(scenario)
        if use_hmm_state is not None:
            self.use_hmm_state = use_hmm_state
        if actor_model is not None:
            self.actor_model = actor_model

        self.dqn_params = {
            "state_dim": config.DQN_STATE_DIM,
            "hidden_dims": config.DQN_HIDDEN_DIMS,
            "learning_rate": config.DQN_LEARNING_RATE,
            "gamma": config.DQN_GAMMA,
            "epsilon": config.DQN_EPSILON_START,
            "epsilon_decay": config.DQN_EPSILON_DECAY,
            "epsilon_min": config.DQN_EPSILON_MIN,
            "target_update_freq": config.DQN_TARGET_UPDATE_FREQ,
            **(dqn_params or {}),
        }

        self.dbn_params = {
            "alpha_1": config.DBN_ALPHA_1,
            "alpha_2": config.DBN_ALPHA_2,
            "lambda_1": config.DBN_LAMBDA_1,
            "lambda_2": config.DBN_LAMBDA_2,
            "max_iter": config.DBN_MAX_ITER,
            "use_cross_validation": config.DBN_USE_CV,
            "cv_splits": config.DBN_CV_SPLITS,
        }
        self.dbn_params.update(dbn_params or {})

        self.lstm_params = {
            "input_dim": config.LSTM_INPUT_DIM,
            "hidden_dim": config.LSTM_HIDDEN_DIM,
            "output_dim": config.LSTM_OUTPUT_DIM,
        }
        self.lstm_params.update(lstm_params or {})

        self.compiled = True
        return self

    def fit(self, use_cache: bool = True) -> "CongressSimulator":
        """Run static pipeline (IPM export + DBN training) and build Congress."""
        if not self.compiled:
            self.compile()

        if config.EXTERNAL_METRICS_AUTO_FETCH:
            try:
                from policyflux.data_collectors.external_signals import MacroSignalProvider

                macro_provider = MacroSignalProvider()
                metrics = macro_provider.fetch_latest_metrics()
                self.regime.update_external_metrics(
                    vix_level=metrics.get("vix"),
                    polarization_index=metrics.get("polarization"),
                    presidential_approval=metrics.get("approval"),
                )
                logger.info("Synced regime with live macro metrics: %s", self.regime.last_external_metrics)
            except Exception as exc:
                logger.warning("Live macro metric fetch failed; continuing with defaults: %s", exc)

        self.builder = CongressMenBuilder(use_cache=use_cache)
        actors_export, ideal_point_model, ipm_params = self.builder.export_actors_with_model()
        self.ipm_params = ipm_params
        self.actors_data = actors_export
        self.influence_matrix = getattr(self.builder, "influence_matrix", None)
        self.cosponsorship_matrix = getattr(self.builder, "cosponsorship_matrix", None)
        self.committee_matrix = getattr(self.builder, "committee_matrix", None)

        self.actors = [CongressMan(data["id"], data, ideal_point_model) for data in actors_export]

        adj_matrix = self._prepare_adjacency(self.influence_matrix, len(self.actors))
        self.congress = TheCongress(
            self.actors,
            adj_matrix,
            self.regime,
            cosponsorship=self.cosponsorship_matrix,
            committee_matrix=self.committee_matrix,
            cospon_alpha=config.COSPONSORSHIP_ALPHA,
            homophily_beta=config.HOMOPHILY_BETA,
            leader_boost=config.LEADER_BOOST,
        )

        for actor in self.actors:
            actor.init_rnn_model(**self.lstm_params)
            actor.init_dqn_agent(**self.dqn_params)

        self.dbn_model = self._train_dbn_on_history(adj_matrix)

        self.engine = CongressEngine(
            self.congress,
            use_gpu=self.use_gpu,
            ipm_params=self.ipm_params,
            dbn_model=self.dbn_model,
            use_hmm_state=self.use_hmm_state,
        )

        self.fitted = True
        return self

    def simulate(
        self,
        n_simulations: Optional[int] = None,
        steps: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        show_tqdm: bool = True,
        show_small_steps: bool = True,
        show_step_tqdm: bool = True,
        show_component_tqdm: bool = True,
    ) -> Dict[str, Any]:
        """Run dynamic Monte Carlo simulation via CongressEngine.

        New optional arguments:
            progress_callback: Optional callable that receives progress events from the engine.
            show_tqdm: If True and `tqdm` is available, show a console progress bar.
        """
        if not self.fitted or not self.engine:
            raise RuntimeError("Simulator not fitted. Call fit() before simulate().")

        n_runs = n_simulations or config.NUM_SIMULATIONS
        n_steps = steps or config.SIMULATION_STEPS
        return self.engine.run_monte_carlo(
            n_simulations=n_runs,
            steps=n_steps,
            progress_callback=progress_callback,
            show_tqdm=show_tqdm,
            show_small_steps=show_small_steps,
            show_step_tqdm=show_step_tqdm,
            show_component_tqdm=show_component_tqdm,
        )

    def _prepare_adjacency(self, matrix: Optional[np.ndarray], n: int) -> np.ndarray:
        """Ensure adjacency is dense, zero-diagonal, and sparsified."""
        if matrix is None:
            fallback = np.random.rand(n, n).astype(np.float32)
            np.fill_diagonal(fallback, 0.0)
            return fallback

        adj = np.array(matrix, dtype=np.float32)
        adj = np.nan_to_num(adj)
        adj = np.where(np.abs(adj) > config.SPARSITY_THRESHOLD, adj, 0.0)
        np.fill_diagonal(adj, 0.0)
        return adj

    def _train_dbn_on_history(self, adj_matrix: np.ndarray) -> Optional[DBCongressModel]:
        """Train DBN on a short synthetic trajectory using current adjacency."""
        try:
            base_state = np.array([a.get_ideological_position() for a in self.actors], dtype=np.float32)
            X_time = [base_state.copy()]
            steps = max(config.SIMULATION_STEPS, 8)
            X_current = base_state.copy()
            regime_context = PublicRegime(self.regime.scenario)
            regime_context.base_pressure = self.regime.base_pressure
            regime_context.volatility_multiplier = self.regime.volatility_multiplier
            regime_context.update_external_metrics(
                vix_level=self.regime.vix_level,
                polarization_index=self.regime.polarization_index,
                presidential_approval=self.regime.presidential_approval,
            )
            Z_time = [regime_context.get_context_vector()]
            for _ in range(steps):
                influence_term = (adj_matrix @ X_current) * 0.05
                noise_term = np.random.normal(0, 0.02, size=X_current.shape)
                X_next = np.clip(X_current + influence_term + noise_term, -1.0, 1.0)
                X_time.append(X_next.copy())
                regime_context.step()
                Z_time.append(regime_context.get_context_vector())
                X_current = X_next

            series = np.stack(X_time)
            dbn = DBCongressModel(**self.dbn_params)
            dbn.fit(series, Z_time_series=np.stack(Z_time), relationship_matrix=adj_matrix)
            logger.info("[✓] DBN trained on synthetic trajectory (learn_deltas=%s)", dbn.learn_deltas)
            return dbn
        except Exception as exc:
            logger.warning("DBN training skipped: %s", exc)
            return None


__all__ = ["CongressSimulator"]




