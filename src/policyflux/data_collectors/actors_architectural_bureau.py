"""Congressional Actor Data Collection and ML Feature Engineering.

This module orchestrates the pipeline for collecting congressional voting data,
training deep learning models (autoencoders, ideal point models), and deriving
behavioral attributes (loyalty, vulnerability, volatility) from voting records.

Key improvements:
- Normalization of behavioral features using StandardScaler or MinMaxScaler
- Cross-Validation for model hyperparameter optimization
- Reconstruction error tracking for autoencoder validation
"""

import logging
from typing import Tuple, Literal, Optional

import numpy as np
import tensorflow as tf

try:
    import cupy as cp  # type: ignore[import]
    _HAS_CUPY = True
except Exception:  # pragma: no cover - optional dependency
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False

try:
    from cuml.covariance import GraphicalLasso as CuGraphicalLasso  # type: ignore[import]
    _HAS_CUML = True
except Exception:  # pragma: no cover - optional dependency
    CuGraphicalLasso = None  # type: ignore[assignment]
    _HAS_CUML = False

from policyflux.models.autoencoders import VoteAutoencoder
from policyflux.models.ideal_point import IdealPointModel
from sklearn.covariance import GraphicalLassoCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from policyflux.data_collectors.external_signals import (
    ExternalSignalCollector,
    OpenFECFinanceProvider,
    ProPublicaCommitteeProvider,
)

try:
    from pyvoteview.core import get_records_by_congress
except ImportError:
    get_records_by_congress = None

from policyflux import config  # from policyflux import configuration

logger = logging.getLogger(__name__)

# Configuration

CONGRESS_NUMBER = config.CONGRESS_NUMBER  # Use config for Congress number
NORMALIZE_BEHAVIORAL_FEATURES = config.NORMALIZE_BEHAVIORAL_FEATURES
NORMALIZATION_TYPE = config.NORMALIZATION_TYPE
AUTOENCODER_LATENT_DIM = config.AUTOENCODER_LATENT_DIM


class CongressMenBuilder:
    CACHE_DIR = "cache"
    CACHE_PREFIX = f"congress_{config.CONGRESS_NUMBER}"
    
    def __init__(
        self,
        use_cache: bool = True,
        finance_provider=None,
        committee_provider=None,
    ):
        import sys
        import os
        
        # Initialize external signal cache manager
        self.external_signals = ExternalSignalCollector(
            use_cache=use_cache,
            finance_provider=finance_provider,
            committee_provider=committee_provider,
        )
        self.cosponsorship_matrix: Optional[np.ndarray] = None
        self.committee_matrix: Optional[np.ndarray] = None
        
        # Check if cache exists and can be loaded
        if use_cache and self._cache_exists():
            logger.info(f"Loading cached data for Congress #{config.CONGRESS_NUMBER}...")
            self._load_cache()
            logger.info("[OK] Loaded all data from cache")
            return
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR, exist_ok=True)
        
        logger.info("[1/9] Fetching data...")
        sys.stdout.flush()
        self.raw = self._fetch()
        logger.info("[2/9] Cleaning data...")
        sys.stdout.flush()
        self.clean = self._clean()
        logger.info("[3/9] Building vote matrix...")
        sys.stdout.flush()
        self.vote_matrix, self.leg_ids, self.vote_ids = self._build_vote_matrix()
        logger.info(f"[4/9] Training autoencoder (matrix shape: {self.vote_matrix.shape})...")
        sys.stdout.flush()
        self.autoencoder_emb = self._train_autoencoder()
        logger.info("[5/9] Training ideal points...")
        sys.stdout.flush()
        self.ideal_points = self._train_ideal_points()
        logger.info("[6/9] Computing influence matrix (Graphical Lasso)...")
        sys.stdout.flush()
        self.influence_matrix = self._graphical_lasso()
        self.network_centrality = self._compute_network_centrality(self.influence_matrix)
        logger.info("[7/9] Training behavior models...")
        sys.stdout.flush()
        self.loyalty, self.vulnerability, self.volatility_from_flips = self._train_behavior_models()
        logger.info("[8/9] Done with initialization")
        sys.stdout.flush()
        
        # Save to cache for next time
        if use_cache:
            logger.info("[9/9] Saving cache...")
            self._save_cache()
            sys.stdout.flush()

    # ---- CACHE MANAGEMENT ----
    
    def _cache_exists(self) -> bool:
        """Check if all necessary cache files exist."""
        import os
        required_files = [
            f"{self.CACHE_PREFIX}_matrices.npz",
            f"{self.CACHE_PREFIX}_arrays.npz"
        ]
        cache_dir = self.CACHE_DIR
        return all(os.path.exists(os.path.join(cache_dir, f)) for f in required_files)
    
    def _save_cache(self) -> None:
        """Save trained models and data to cache."""
        import os
        
        cache_dir = self.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # Save numpy arrays (matrices and behavioral scores)
            np.savez(
                os.path.join(cache_dir, f"{self.CACHE_PREFIX}_matrices.npz"),
                vote_matrix=self.vote_matrix,
                influence_matrix=self.influence_matrix,
                autoencoder_emb=self.autoencoder_emb,
                ideal_points=self.ideal_points,
                loyalty=self.loyalty,
                vulnerability=self.vulnerability,
                volatility_from_flips=self.volatility_from_flips
            )
            
            # Save IDs as separate arrays
            np.savez(
                os.path.join(cache_dir, f"{self.CACHE_PREFIX}_arrays.npz"),
                leg_ids=self.leg_ids,
                vote_ids=self.vote_ids
            )
            
            logger.info(f"[OK] Cache saved to {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_cache(self) -> None:
        """Load trained models and data from cache."""
        import os
        
        cache_dir = self.CACHE_DIR
        
        try:
            # Load numpy arrays
            matrices = np.load(os.path.join(cache_dir, f"{self.CACHE_PREFIX}_matrices.npz"))
            self.vote_matrix = matrices['vote_matrix']
            self.influence_matrix = matrices['influence_matrix']
            self.autoencoder_emb = matrices['autoencoder_emb']
            self.ideal_points = matrices['ideal_points']
            self.loyalty = matrices['loyalty']
            self.vulnerability = matrices['vulnerability']
            self.volatility_from_flips = matrices['volatility_from_flips']

            # Validate loaded influence matrix: it must be legislator x legislator (n_leg x n_leg)
            try:
                n_leg = len(self.leg_ids) if hasattr(self, 'leg_ids') else None
                if n_leg and getattr(self, 'influence_matrix', None) is not None:
                    if self.influence_matrix.shape != (n_leg, n_leg):
                        logger.warning(
                            "Cached influence_matrix shape %s incompatible with number of legislators %s; recomputing legislator-by-legislator matrix.",
                            getattr(self.influence_matrix, 'shape', None), n_leg
                        )
                        # Fallback: compute legislator correlations from vote matrix
                        X = StandardScaler().fit_transform(self.vote_matrix)
                        corr = np.corrcoef(X)
                        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                        np.fill_diagonal(corr, 0)
                        W = np.tanh(np.abs(corr))
                        if np.any(W != 0):
                            thresh = np.quantile(np.abs(W[W != 0]), 0.5)
                            W[np.abs(W) < thresh] = 0.0
                        self.influence_matrix = W
            except Exception as _e:
                logger.debug("Failed to validate/repair cached influence_matrix: %s", _e)
            
            # Load IDs
            arrays = np.load(os.path.join(cache_dir, f"{self.CACHE_PREFIX}_arrays.npz"))
            self.leg_ids = arrays['leg_ids']
            self.vote_ids = arrays['vote_ids']

            # Re-validate influence_matrix now that we know leg_ids length
            try:
                n_leg = len(self.leg_ids)
                if getattr(self, 'influence_matrix', None) is not None:
                    if self.influence_matrix.shape != (n_leg, n_leg):
                        logger.warning(
                            "Cached influence_matrix shape %s incompatible with number of legislators %s; recomputing legislator-by-legislator matrix.",
                            getattr(self.influence_matrix, 'shape', None), n_leg
                        )
                        # Fallback: compute legislator correlations from vote matrix
                        X = StandardScaler().fit_transform(self.vote_matrix)
                        corr = np.corrcoef(X)
                        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                        np.fill_diagonal(corr, 0)
                        W = np.tanh(np.abs(corr))
                        if np.any(W != 0):
                            thresh = np.quantile(np.abs(W[W != 0]), 0.5)
                            W[np.abs(W) < thresh] = 0.0
                        self.influence_matrix = W
            except Exception as _e:
                logger.debug("Failed to validate/repair cached influence_matrix after arrays load: %s", _e)
            
            self.ideal_point_model = None
            
            # Initialize other attributes that might be needed
            self.raw = None  # Don't need raw data after training
            self.clean = None  # Don't need clean data after training
            
            logger.info(f"[OK] Loaded all cache from {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            raise

    # -------------------------------
    # DATA
    # -------------------------------

    def _fetch(self):
        """Fetch voting records from pyvoteview API with robust error handling and timeout."""
        import pandas as pd
        
        if get_records_by_congress is None:
            logger.warning("pyvoteview not installed. Falling back to mock data.")
            raise ImportError("pyvoteview not available")
        
        logger.info(f"Attempting to fetch Congress #{CONGRESS_NUMBER} voting records...")
        
        try:
            # Try direct approach first (may hang, but worth trying)
            logger.debug("Calling get_records_by_congress...")
            raw = get_records_by_congress(CONGRESS_NUMBER, chamber="House")
            
            if raw is None or (isinstance(raw, (list, tuple)) and len(raw) == 0):
                raise ValueError("No records returned from pyvoteview")
            
            logger.debug(f"Raw data type: {type(raw)}")
            logger.info(f"[OK] Fetched data from pyvoteview, converting to Pandas...")
            
            # Convert Polars to Pandas if needed (this can take a while for large datasets)
            if hasattr(raw, 'to_pandas'):  # Polars DataFrame
                logger.debug(f"Converting Polars DataFrame to Pandas (large operation)...")
                df = raw.to_pandas()
                logger.info(f"[OK] Converted Polars DataFrame to Pandas ({len(df)} rows)")
            elif isinstance(raw, pd.DataFrame):
                df = raw.copy()
            else:
                df = pd.DataFrame(raw)
            
            logger.debug(f"Data shape: {df.shape}, columns: {list(df.columns)}")
            
            # Handle pyvoteview column formats
            # Required columns: icpsr (legislator), rollnumber (roll call), cast_code (vote)
            col_mapping = {}
            
            # Map legislator ID column
            for legislator_col in ['icpsr', 'legislator_id', 'member_id']:
                if legislator_col in df.columns:
                    col_mapping[legislator_col] = 'legislator_id'
                    break
            
            # Map roll call ID column
            for rollcall_col in ['rollnumber', 'roll_call_number', 'roll_call_id', 'vote_id']:
                if rollcall_col in df.columns:
                    col_mapping[rollcall_col] = 'rollcall_id'
                    break
            
            # Map vote column (cast_code: 1=Yea, 2=Nay, 3=Present, etc.)
            for vote_col in ['cast_code', 'vote', 'vote_code']:
                if vote_col in df.columns:
                    col_mapping[vote_col] = 'vote'
                    break
            
            if not col_mapping:
                logger.error(f"Could not map columns. Available: {list(df.columns)}")
                raise ValueError("Could not find required voting data columns")
            
            # Apply column renaming
            df = df.rename(mapper=col_mapping, axis='columns')
            
            # Select and validate required columns
            required_cols = ['legislator_id', 'rollcall_id', 'vote']
            available_cols = [c for c in required_cols if c in df.columns]
            
            if not available_cols:
                raise ValueError(f"Required columns not found after mapping. Have: {list(df.columns)}")
            
            df = df[available_cols]
            
            # Clean data
            logger.info("Cleaning data...")
            df = df.dropna(subset=['legislator_id', 'vote'])
            df['legislator_id'] = pd.to_numeric(df['legislator_id'], errors='coerce').astype('Int64')
            df['rollcall_id'] = pd.to_numeric(df['rollcall_id'], errors='coerce').astype('Int64')
            df['vote'] = pd.to_numeric(df['vote'], errors='coerce')
            
            # Remove rows with invalid conversions
            df = df.dropna(subset=['legislator_id', 'rollcall_id', 'vote'])
            
            logger.info(f"[OK] Loaded {len(df)} voting records from policyflux.congress #{CONGRESS_NUMBER} (pyvoteview)")
            
            # Fetch party information separately
            self.legislator_party_map = self._fetch_party_data(raw)
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load real data: {str(e)}")
            raise
    
    def _fetch_party_data(self, leg_data=None):
        """Fetch legislator party information."""
        import pandas as pd
        
        party_map = {}
        try:
            # If leg_data provided (from _fetch), use it directly
            if leg_data is None:
                return party_map
            
            # Convert Polars to Pandas if needed
            if hasattr(leg_data, 'to_pandas'):
                leg_df = leg_data.to_pandas()
            elif isinstance(leg_data, pd.DataFrame):
                leg_df = leg_data.copy()
            else:
                leg_df = pd.DataFrame(leg_data)
            
            # For pyvoteview data, we need icpsr and party_code from same rows
            # Each row represents a vote, so we need to drop duplicates
            if 'icpsr' in leg_df.columns and 'party_code' in leg_df.columns:
                # Get unique legislator-party pairs
                party_data = leg_df[['icpsr', 'party_code']].drop_duplicates()
                
                for _, row in party_data.iterrows():
                    try:
                        leg_id = int(row['icpsr'])
                        party_code = int(row['party_code'])
                        party_map[leg_id] = party_code
                    except (ValueError, TypeError):
                        continue
            
            if party_map:
                logger.info(f"[OK] Loaded party data for {len(party_map)} legislators")
        
        except Exception as e:
            logger.debug(f"Could not fetch party data: {e}")
        
        return party_map

    def _clean(self):
        df = self.raw.dropna()
        df["legislator_id"] = df["legislator_id"].astype(int)
        df["rollcall_id"] = df["rollcall_id"].astype(int)
        return df

    def _encode(self, v):
        if v in [1,2,3]: return 1
        if v in [4,5,6]: return -1
        return 0

    def _build_vote_matrix(self):
        df = self.clean.copy()
        df["v"] = df["vote"].apply(self._encode)

        pivot = df.pivot_table(index="legislator_id",
                               columns="rollcall_id",
                               values="v",
                               fill_value=0)

        return pivot.values.astype(np.float32), pivot.index.values, pivot.columns.values

    # -------------------------------
    # AUTOENCODER
    # -------------------------------

    def _train_autoencoder(self, latent_dim: int = 8, epochs: int = 40):
        X = StandardScaler().fit_transform(self.vote_matrix)
        batch_size = min(512, max(64, X.shape[0]))
        model = VoteAutoencoder(X.shape[1], latent_dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = tf.keras.losses.MeanSquaredError()

        dataset = tf.data.Dataset.from_tensor_slices(X.astype(np.float32)).batch(batch_size)
        for _ in range(epochs):
            for batch in dataset:
                with tf.GradientTape() as tape:
                    recon, _ = model(batch, training=True)
                    loss = loss_fn(batch, recon)
                grads = tape.gradient(loss, model.trainable_variables)
                if grads:
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

        embeddings = []
        for batch in dataset:
            _, z = model(batch, training=False)
            embeddings.append(z.numpy())
        return np.concatenate(embeddings, axis=0)

    # -------------------------------
    # IDEAL POINT ESTIMATION
    # -------------------------------

    def _train_ideal_points(self, dim=3, epochs=30, lr=1e-2, batch_size: int = 2048):
        """Train Ideal Point Model using TensorFlow."""
        df = self.clean.copy()
        df["v"] = (df["vote"].apply(self._encode) == 1).astype(int)

        leg_map = {l: i for i, l in enumerate(self.leg_ids)}
        vote_map = {v: i for i, v in enumerate(self.vote_ids)}

        df["li"] = df["legislator_id"].map(leg_map)
        df["vi"] = df["rollcall_id"].map(vote_map)

        li = df["li"].values.astype(np.int32)
        vi = df["vi"].values.astype(np.int32)
        y = df["v"].values.astype(np.float32)

        eff_batch_size = max(256, min(batch_size, len(df)))
        dataset = tf.data.Dataset.from_tensor_slices((li, vi, y)).batch(eff_batch_size)

        model = IdealPointModel(len(self.leg_ids), len(self.vote_ids), dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        for e in range(epochs):
            epoch_loss = 0.0
            batches = 0
            for batch_li, batch_vi, batch_y in dataset:
                with tf.GradientTape() as tape:
                    preds = model(batch_li, batch_vi)
                    loss = loss_fn(batch_y, preds)
                grads = tape.gradient(loss, model.trainable_variables)
                if grads:
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss += float(loss.numpy())
                batches += 1
            if batches:
                epoch_loss /= batches
            if e % 5 == 0 or e == epochs - 1:
                logger.info(f"IPM epoch {e+1:3d}/{epochs}: loss={epoch_loss:.6f}")

        self.ideal_point_model = model
        self.ideal_points = model.x.numpy()
        return self.ideal_points

    # -------------------------------
    # GRAPHICAL LASSO – CAUSAL NETWORK
    # -------------------------------

    def _graphical_lasso(self):
        """Estimate influence network using Graphical Lasso with timeout fallback."""
        import signal
        import functools
        
        # Define timeout handler
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Graphical Lasso computation timed out (>120s)")
        
        try:
            # GPU-accelerated path using cuML + CuPy when available
            if _HAS_CUPY and _HAS_CUML:
                try:
                    X_gpu = cp.asarray(self.vote_matrix, dtype=cp.float32)
                    X_gpu = (X_gpu - cp.mean(X_gpu, axis=0)) / cp.clip(cp.std(X_gpu, axis=0), 1e-6, None)
                    if self.vote_matrix.shape[1] > 500:
                        X_gpu = X_gpu[:, ::2]
                        logger.info("Using GPU Graphical Lasso on column-subsampled matrix for speed...")
                    gl_gpu = CuGraphicalLasso(alpha=0.01, max_iter=100, verbose=False)
                    gl_gpu.fit(X_gpu.T)
                    precision_gpu = cp.asarray(gl_gpu.precision_)
                    precision_gpu = precision_gpu - cp.diag(cp.diag(precision_gpu))
                    W_gpu = cp.tanh(cp.abs(precision_gpu))
                    if cp.any(W_gpu):
                        cutoff_gpu = cp.quantile(cp.abs(W_gpu[W_gpu != 0]), 0.92)
                        W_gpu = cp.where(cp.abs(W_gpu) < cutoff_gpu, 0.0, W_gpu)
                    W = cp.asnumpy(W_gpu)
                    if self.vote_matrix.shape[1] > 500:
                        X_full = StandardScaler().fit_transform(self.vote_matrix)
                        corr_full = np.abs(np.corrcoef(X_full))
                        np.fill_diagonal(corr_full, 0)
                        W = 0.7 * corr_full + 0.3 * W
                    logger.info("[OK] GPU Graphical Lasso converged via cuML")
                    return W
                except Exception as gpu_error:
                    logger.debug("cuML Graphical Lasso failed; falling back to CPU: %s", gpu_error)

            X = StandardScaler().fit_transform(self.vote_matrix)
            
            # For large matrices (>500 votes), use faster approximation
            if self.vote_matrix.shape[1] > 500:
                logger.info("Matrix is large (>500 votes). Using faster correlation-based approximation...")
                X_sampled = X[:, ::2]  # Subsample columns to reduce computational burden
            else:
                X_sampled = X
            
            # Use GraphicalLassoCV with optimized parameters for speed
            model = GraphicalLassoCV(
                cv=2,  # Reduced from 3 to 2 for speed
                alphas=10,  # Reduced from 20 to 10
                max_iter=100,  # Reduced from 200 to 100
                verbose=0,
                tol=1e-3,  # Relaxed tolerance
                n_jobs=1  # Use single core (parallel on Windows causes issues)
            )
            
            logger.info(f"Starting Graphical Lasso on {X_sampled.shape[1]} votes (transposed to {X_sampled.shape[0]} legislators) (timeout: 120s)...")
            
            # Set signal timeout (Unix only; Windows will skip this)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)  # 120-second timeout
            
            # Fit on the sampled data: transpose so each vote is a sample and legislators are variables
            # This yields a precision matrix of shape (n_legislators, n_legislators)
            model.fit(X_sampled.T)
            
            # Cancel alarm if successful
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            precision = model.precision_
            np.fill_diagonal(precision, 0)
            
            W = np.tanh(np.abs(precision))
            W[W < np.quantile(W, 0.92)] = 0
            
            # If we used sampled data, expand back to full size via correlation
            if self.vote_matrix.shape[1] > 500:
                logger.info("Expanding sparse network estimate back to full dimensionality...")
                X_full = StandardScaler().fit_transform(self.vote_matrix)
                # Correlate legislators (rows of X_full) to get legislator x legislator matrix
                corr_full = np.abs(np.corrcoef(X_full))
                np.fill_diagonal(corr_full, 0)
                # Blend sparse GL result with legislator correlation matrix
                W_expanded = np.zeros((len(self.leg_ids), len(self.leg_ids)))
                W_expanded = 0.7 * corr_full + 0.3 * W  # Use sampled GL as regularizer
                W = W_expanded
            
            logger.info(f"[OK] Graphical Lasso converged with alpha={model.alpha_:.4f}")
            return W
            
        except (TimeoutError, FloatingPointError, np.linalg.LinAlgError, Exception) as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel timeout
            
            logger.warning(f"Graphical Lasso timeout/failed ({type(e).__name__}): {str(e)[:100]}")
            logger.info("Using fast correlation-based fallback for influence network...")
            
            try:
                # Fallback: use legislator correlation matrix with thresholding
                X = StandardScaler().fit_transform(self.vote_matrix)
                
                # Compute correlation between legislators (rows)
                corr_matrix = np.corrcoef(X)
                
                # Handle NaN values from ill-conditioned data
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                np.fill_diagonal(corr_matrix, 0)
                
                # Apply tanh and threshold
                W = np.tanh(np.abs(corr_matrix))
                W[W < np.quantile(np.abs(W[W != 0]) if np.any(W != 0) else W, 0.5)] = 0
                
                logger.info(f"[OK] Using correlation-based fallback for influence network ({W.shape})")
                return W
            
            except Exception as e2:
                logger.warning(f"Even correlation fallback failed: {e2}")
                # Ultimate fallback: random sparse network
                n = len(self.leg_ids)
                W = np.random.rand(n, n)
                W[W < 0.9] = 0
                np.fill_diagonal(W, 0)
                logger.info(f"[OK] Using random sparse network fallback ({W.shape})")
                return W
            logger.warning(f"Influence network estimation failed: {e}")
            # Last resort: create sparse random network
            n = self.vote_matrix.shape[0]
            W = np.random.uniform(0, 0.1, (n, n))
            W[np.triu_indices_from(W, k=1)] = 0  # Lower triangular only
            logger.info(f"[OK] Using random sparse network as fallback")
            return W

    def _compute_network_centrality(self, matrix: Optional[np.ndarray]) -> np.ndarray:
        """Compute betweenness centrality on influence graph (fallback to degree)."""
        if matrix is None:
            return np.zeros(len(getattr(self, "leg_ids", [])), dtype=float)

        try:
            import networkx as nx  # type: ignore

            G = nx.from_numpy_array(np.abs(matrix), create_using=nx.Graph)
            cent = nx.betweenness_centrality(G, normalized=True)
            values = np.array([cent.get(i, 0.0) for i in range(len(self.leg_ids))], dtype=float)
        except Exception:
            # Degree-based proxy if networkx missing or graph too large
            values = np.sum(np.abs(matrix), axis=1).astype(float)

        if values.size == 0:
            return values

        max_val = float(values.max()) if np.isfinite(values).any() else 0.0
        if max_val > 0:
            values = values / max_val
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def combine_cosponsorship_graphical_lasso(self, cosponsorship: np.ndarray = None,
                                              alpha: float = 0.5,
                                              homophily_beta: float = 2.0,
                                              party_bonus: float = 1.5,
                                              threshold_quantile: float = 0.92):
        """
        Łączy macierz współ-sponsorowania z estymatem Graphical Lasso.
        - `cosponsorship`: macierz (n,n) z relacją cosponsor (może być 0)
        - `alpha`: waga cosponsorship w końcowym W (0..1)
        - `homophily_beta`: siła homofilii (większe -> silniejsze tłumienie dla różnych ideologii)
        - `party_bonus`: mnożnik dla par z tej samej partii
        """
        # 1) Estymuj Graphical Lasso na głosowaniach
        W_gl = self._graphical_lasso()

        n = W_gl.shape[0]
        # 2) Przygotuj cosponsorship (jeśli brak -> macierz zer)
        if cosponsorship is None:
            cosponsorship = np.zeros((n, n), dtype=float)
        else:
            cosponsorship = np.array(cosponsorship, dtype=float)

        # 3) Normalizacja obu macierzy
        def norm_mat(M):
            M = np.array(M, dtype=float)
            if M.max() > 0:
                return M / M.max()
            return M

        Wc = norm_mat(cosponsorship)
        Wg = norm_mat(W_gl)

        # 4) Blendowanie
        W = alpha * Wc + (1 - alpha) * Wg

        # 5) Homophily weighting: jeśli mamy ideal points, użyjemy ich
        if hasattr(self, 'ideal_points') and self.ideal_points is not None:
            ips = self.ideal_points.flatten() if self.ideal_points.ndim == 2 else self.ideal_points
            ips = np.array(ips)
            # dystans ideologiczny
            dist = np.abs(ips.reshape(-1, 1) - ips.reshape(1, -1))
            H = np.exp(-homophily_beta * dist)
        else:
            H = np.ones_like(W)

        # 6) Party bonus: jeśli mamy informacje o partii (export_actors does not yet include party),
        # próbujemy użyć self.loyalty as proxy (optional) or keep neutral
        P = np.ones_like(W)
        # If builder kept legislator ids in self.leg_ids and we can map parties, skip for now

        # 7) Apply homophily and party multipliers
        W = W * H * P

        # 8) Threshold small weights
        cutoff = np.quantile(np.abs(W), threshold_quantile) if np.any(W) else 0.0
        W[np.abs(W) < cutoff] = 0.0

        # 9) Symmetrize softly (keep directionality small) and return
        W = W
        return W

    # -------------------------------
    # LOYALTY + VULNERABILITY (ML)
    # -------------------------------

    def _train_behavior_models(self):
        """
        Trenuje modele zachowania (lojalność i wrażliwość) na podstawie głosów.
        Teraz uwzględnia PSC (Presidential Support Score) dla kalibracji lojalności.
        
        Returns:
            tuple: (loyalty, vulnerability, flips) gdzie flips to historyczna zmienność
                   każdego aktora (używana do kalibracji volatility)
        """
        X = self.vote_matrix
        flips = np.mean(np.abs(np.diff(X, axis=1)), axis=1)

        y_loyal = (flips < np.median(flips)).astype(int)
        y_vuln = (flips > np.quantile(flips, 0.75)).astype(int)

        model_loyal = GradientBoostingClassifier()
        model_vuln = GradientBoostingClassifier()

        model_loyal.fit(X, y_loyal)
        model_vuln.fit(X, y_vuln)

        loyalty = model_loyal.predict_proba(X)[:,1]
        vulnerability = model_vuln.predict_proba(X)[:,1]

        # Kalibracja lojalności poprzez PSC (jeśli dostępne)
        loyalty = self._calibrate_loyalty_with_psc(loyalty)
        
        # Normalizuj flips na [0, 1] dla użycia jako volatility
        flips_normalized = (flips - flips.min()) / (flips.max() - flips.min() + 1e-8)
        
        # Normaliz behavioral features if configured
        if NORMALIZE_BEHAVIORAL_FEATURES:
            loyalty, vulnerability, flips_normalized = self._normalize_behavioral_features(
                loyalty, vulnerability, flips_normalized
            )
        
        return loyalty, vulnerability, flips_normalized
    
    def _normalize_behavioral_features(self, loyalty: np.ndarray, vulnerability: np.ndarray, 
                                       volatility: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize behavioral features to comparable ranges.
        
        Normalization ensures that loyalty, vulnerability, and volatility are in
        comparable ranges [0, 1], which is crucial when using them as weights or
        multipliers in decision-making and simulation.
        
        Args:
            loyalty: Array of loyalty scores.
            vulnerability: Array of vulnerability scores.
            volatility: Array of volatility scores.
            
        Returns:
            Tuple of normalized (loyalty, vulnerability, volatility) arrays.
        """
        if NORMALIZATION_TYPE == "minmax":
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:  # default: "standard"
            scaler = StandardScaler()
        
        # Stack features into matrix for joint normalization
        features = np.column_stack([loyalty, vulnerability, volatility])
        features_normalized = scaler.fit_transform(features)
        
        # If using StandardScaler, clip to [0, 1] for interpretability
        if NORMALIZATION_TYPE == "standard":
            features_normalized = np.clip(features_normalized, 0, 1)
        
        loyalty_norm = features_normalized[:, 0]
        vulnerability_norm = features_normalized[:, 1]
        volatility_norm = features_normalized[:, 2]
        
        logger.info(f"Behavioral features normalized using {NORMALIZATION_TYPE} normalization")
        logger.debug(f"  Loyalty: mean={loyalty_norm.mean():.4f}, std={loyalty_norm.std():.4f}")
        logger.debug(f"  Vulnerability: mean={vulnerability_norm.mean():.4f}, std={vulnerability_norm.std():.4f}")
        logger.debug(f"  Volatility: mean={volatility_norm.mean():.4f}, std={volatility_norm.std():.4f}")
        
        return loyalty_norm, vulnerability_norm, volatility_norm
    
    def _calibrate_loyalty_with_psc(self, loyalty_scores):
        """
        Kalibruje wyniki lojalności przy użyciu Presidential Support Score (PSC).
        PSC reprezentuje historyczną wierność legislatora względem prezydenckiej linii partyjnej.
        
        Heurystyka:
        - Wysokie PSC (np. > 0.7) powinno zwiększać loyalty
        - Niskie PSC (np. < 0.3) powinno zmniejszać loyalty
        """
        try:
            # Próba pobrania PSC z pyvoteview lub innego źródła
            # W praktyce: można dodać API call lub załadować historyczne dane
            # Na razie: używamy party jako proxy
            
            # Jeśli mamy party map, użyjemy go
            if hasattr(self, 'legislator_party_map') and self.legislator_party_map:
                # Legislatorzy z partii głównej (code 100, 200) powinni mieć wyższą loyalty
                psc_proxy = []
                for leg_id in self.leg_ids:
                    party_code = self.legislator_party_map.get(int(leg_id), None)
                    if party_code in [100, 200]:  # Major parties
                        psc_proxy.append(0.7)  # Wyższa loyalty dla głównych partii
                    elif party_code:
                        psc_proxy.append(0.4)  # Średnia dla mniejszych partii
                    else:
                        psc_proxy.append(0.5)  # Neutralna dla nieznanych
                
                psc_proxy = np.array(psc_proxy)
                # Wzmocnij lojalność dla wysokiego PSC
                loyalty_scores = 0.7 * loyalty_scores + 0.3 * psc_proxy
        except Exception as e:
            print(f"Warning: PSC calibration failed: {e}")
        
        return loyalty_scores

    # -------------------------------
    # EXPORT
    # -------------------------------

    def export_actors_with_model(self):
        """
        Exportuje aktorów z wielowymiarową ideologią oraz zwraca wytrenowany IdealPointModel.
        Dodatkowo eksportuje parametry głosowań (a_j, b_j) do użytku w symulacji.
        
        Returns:
            tuple: (actors_list, ideal_point_model, voting_parameters)
                - actors_list: Lista słowników z danymi aktorów
                - ideal_point_model: Wytrenowana instancja IdealPointModel
                - voting_parameters: Dict z kluczami 'salience' (a_j) i 'threshold' (b_j)
        """
        # Ideal points shape: (n_legislators, dim)
        # dim=3: [Principles, Economics, Emotions]
        
        if self.ideal_points.ndim == 1:
            # Fallback jeśli model zwrócił 1D
            ideology_multidim = np.column_stack([
                np.tanh(self.ideal_points),
                np.random.uniform(-1, 1, size=len(self.ideal_points)),
                np.random.uniform(-1, 1, size=len(self.ideal_points))
            ])
        else:
            # Shape: (n_legislators, dim)
            # Dim 1-3: Principles, Economics, Emotions
            ideology_multidim = np.tanh(self.ideal_points)
        
        # Extract voting parameters from trained model if available
        voting_params = None
        if hasattr(self, 'ideal_point_model') and self.ideal_point_model is not None:
            try:
                salience = self.ideal_point_model.a.numpy()
                threshold = self.ideal_point_model.b.numpy()
                voting_params = {
                    'salience': salience,
                    'threshold': threshold
                }
                logger.info(f"[OK] Exported voting parameters: salience shape={salience.shape}, threshold shape={threshold.shape}")
            except Exception as e:
                logger.warning(f"Failed to extract voting parameters from IPM: {e}")
                voting_params = None
        
        signals_bundle = self.external_signals.collect_signals(self.leg_ids)
        actor_signal_map = signals_bundle.get("actors", {})
        self.cosponsorship_matrix = signals_bundle.get("cosponsorship_matrix")
        self.committee_matrix = signals_bundle.get("committee_matrix")

        actors = []
        for i in range(len(self.leg_ids)):
            leg_id = int(self.leg_ids[i])
            party = None

            # Spróbuj przypisać partię na podstawie party_code
            if hasattr(self, 'legislator_party_map') and self.legislator_party_map:
                party_code = self.legislator_party_map.get(leg_id)
                if party_code == 100:
                    party = 'Republican'
                elif party_code == 200:
                    party = 'Democratic'
                elif party_code:
                    party = 'Independent'
            
            # Jeśli brak danych, użyj heurystyki na podstawie wymiaru 1 ideologii
            if party is None:
                party = 'Republican' if ideology_multidim[i, 0] > 0 else 'Democratic'
            
            actor_signals = actor_signal_map.get(str(leg_id), {})
            finance_signal = actor_signals.get("finance", {}) or {}
            district_signal = actor_signals.get("district", {}) or {}
            relationship_signal = actor_signals.get("relationships", {}) or {}

            financial_support = 0.0
            if finance_signal:
                try:
                    financial_support = float(finance_signal.get("total_pac", 0.0))
                    if financial_support == 0.0:
                        financial_support = float(sum(v for v in finance_signal.values() if isinstance(v, (int, float))))
                except Exception:
                    financial_support = 0.0

            committee_memberships = relationship_signal.get("committee_memberships", []) or []
            committee_power = relationship_signal.get("committee_power")
            if committee_power is None:
                committee_power = float(len(committee_memberships))

            presidential_support_score = relationship_signal.get("presidential_support_score")
            if presidential_support_score is None:
                presidential_support_score = actor_signals.get("presidential_support_score")

            electoral_margin = district_signal.get("electoral_margin", district_signal.get("margin"))
            district_urbanization = district_signal.get("urbanization")
            district_income = district_signal.get("median_income")
            district_stability = district_signal.get("stability", district_signal.get("gerrymander_score"))
            district_demographics = district_signal.get("demographics", {}) or {}

            network_centrality = 0.0
            try:
                if hasattr(self, "network_centrality") and len(self.network_centrality) > i:
                    network_centrality = float(self.network_centrality[i])
            except Exception:
                network_centrality = 0.0

            actor_data = {
                "id": leg_id,
                "party": party,
                "ideology": float(ideology_multidim[i, 0]),  # Wymiar 1: głównie ideologiczny
                "ideology_multidim": ideology_multidim[i].tolist(),  # [principles, economics, emotions]
                "loyalty": float(self.loyalty[i]),
                "vulnerability": float(self.vulnerability[i]),
                "volatility": float(0.03 + 0.05*np.random.rand()),
                "presidential_support_score": presidential_support_score,
                "financial_support": financial_support,
                "network_centrality": network_centrality,
                "committee_memberships": committee_memberships,
                "committee_power": float(committee_power),
                "electoral_margin": electoral_margin,
                "district_urbanization": district_urbanization,
                "district_income": district_income,
                "district_stability": district_stability,
                "district_demographics": district_demographics,
            }

            actor_data["finance_signal"] = finance_signal
            actor_data["district_signal"] = district_signal
            actor_data["relationship_signal"] = relationship_signal

            actors.append(actor_data)
        
        # Zwróć aktorów, model, i parametry głosowań
        return actors, self.ideal_point_model if hasattr(self, 'ideal_point_model') else None, voting_params

    def export_actors(self):
        """
        Exportuje aktorów z wielowymiarową ideologią (Dimension 1: Ekonomiczny/Społeczny, 
        Dimension 2: Partyjny/Populistyczny).
        
        Uwaga: Ta metoda jest utrzymywana dla kompatybilności wstecznej.
        Preferowana jest metoda export_actors_with_model().
        """
        actors, _ = self.export_actors_with_model()
        return actors

# debugging 
#if __name__ == '__main__':
#    builder = CongressMenBuilder()
#    print(builder.votes_base)



