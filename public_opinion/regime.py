
"""Global Political Context and Regime Configuration.

This module manages macro-level variables Z(t) such as political pressure,
volatility, and contextual shocks affecting all congressional actors.
"""

from typing import Literal, Optional

import logging
import numpy as np

import config

# Optional HMM support (hmmlearn)
try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMMLEARN = True
except Exception:
    GaussianHMM = None
    _HAS_HMMLEARN = False


logger = logging.getLogger(__name__)


class PublicRegime:
    """Global political context affecting all actors.
    
    This class represents system-wide political conditions Z(t) that affect
    all congressional actors. Different scenarios (stable, polarized, crisis,
    and collapse) correspond to different levels of external pressure and volatility.
    """

    def __init__(self, scenario: Literal["stable", "polarized", "crisis", "collapse"] = "stable") -> None:
        """Initialize regime with given scenario.

        Args:
            scenario: Political scenario ('stable', 'polarized', 'crisis', 'collapse').
        """
        self.scenario = scenario

        # Stateful counters for feedback
        self.consecutive_failures: float = 0.0  # Allow fractional weighting by bill importance
        self.crisis_threshold: int = 5         # Number of weighted failures to trigger escalation

        # Base parameters (can be adjusted dynamically)
        self.base_pressure: float = 0.0
        self.volatility_multiplier: float = 1.0
        self.time_step: int = 0

        # External macro indicators (can be fed from real data pipelines)
        self.vix_level: Optional[float] = None  # Market volatility proxy
        self.polarization_index: Optional[float] = None  # Party distance (0-1)
        self.presidential_approval: Optional[float] = None  # Approval rating in [0,1]
        self.last_external_metrics: dict[str, Optional[float]] = {}

        # Initialize according to scenario
        self._configure_scenario()
        self.last_hmm_state: Optional[str] = None
        self.last_hmm_component: Optional[int] = None

    def _configure_scenario(self) -> None:
        """Configure pressure and volatility based on scenario."""
        if self.scenario == "stable":
            self.base_pressure = 0.1  # Low systemic pressure
            self.volatility_multiplier = 1.0  # Normal noise level
        elif self.scenario == "polarized":
            self.base_pressure = 0.5  # Strong party pressure
            self.volatility_multiplier = 1.5  # Increased volatility
        elif self.scenario == "crisis":
            self.base_pressure = 0.8  # Elevated systemic pressure
            self.volatility_multiplier = 3.0  # High chaos/noise
        elif self.scenario == "collapse":
            # Extreme, end-state regime
            self.base_pressure = 1.0
            self.volatility_multiplier = 5.0

    def get_current_pressure(self) -> float:
        """Get current system pressure Z(t) with potential shocks.

        Returns a scalar Z(t) in [0,1]. Shocks depend on scenario and the current
        volatility multiplier and may be asymmetric.
        """
        base_pressure = self.base_pressure

        # Data-driven scaling of pressure using polarization and approval
        if self.polarization_index is not None:
            pol = float(np.clip(self.polarization_index, 0.0, 1.0))
            base_pressure = max(base_pressure, 0.1 + 0.7 * pol)
        if self.presidential_approval is not None:
            approval = float(np.clip(self.presidential_approval, 0.0, 1.0))
            # Low approval raises systemic pressure; high approval stabilizes majority side
            base_pressure = base_pressure + 0.15 * (0.5 - approval)

        # Volatility driven by VIX-like indicator
        volatility_multiplier = self.volatility_multiplier
        if self.vix_level is not None:
            vix_scaled = float(np.clip(self.vix_level, 0.0, 80.0)) / 40.0  # normalize ~[0,2]
            volatility_multiplier = max(volatility_multiplier, 1.0 + 0.5 * vix_scaled)

        # Scenario-dependent shock magnitude
        if self.scenario == "stable":
            shock_range = 0.05
        elif self.scenario == "polarized":
            shock_range = 0.20
        else:  # crisis / collapse
            shock_range = 0.40

        # Shock sampled from Normal (mean=0, sd depending on shock range and volatility)
        shock = np.random.normal(0.0, shock_range * volatility_multiplier * 0.1)

        # Small escalation due to accumulated failures (soft effect)
        failure_trend = 0.0
        if self.consecutive_failures > 0:
            failure_trend = 0.01 * min(self.consecutive_failures, self.crisis_threshold)

        # Time trend for crisis states
        time_trend = 0.0
        if self.scenario == "crisis":
            time_trend = 0.01 * min(self.time_step, 10)

        pressure = base_pressure + shock + failure_trend + time_trend

        # Clip to [0, 1]
        return float(np.clip(pressure, 0.0, 1.0))
    
    def step(self) -> None:
        """Advance regime time step (call once per simulation step)."""
        self.time_step += 1
    
    def reset(self) -> None:
        """Reset regime to initial state (call before new simulation run)."""
        self.time_step = 0
        self.consecutive_failures = 0.0
        self._configure_scenario()

    def update_external_metrics(
        self,
        vix_level: Optional[float] = None,
        polarization_index: Optional[float] = None,
        presidential_approval: Optional[float] = None,
        auto_fetch: bool = False,
        provider: Optional[object] = None,
    ) -> dict[str, Optional[float]]:
        """Inject macro indicators that modulate pressure/volatility and homophily.

        Args:
            vix_level: Market/political volatility proxy (e.g., VIX), non-negative.
            polarization_index: Party distance metric in [0,1].
            presidential_approval: Approval rating in [0,1].
            auto_fetch: If True (or if config.EXTERNAL_METRICS_AUTO_FETCH is True),
                pull missing values from live providers.
            provider: Optional MacroSignalProvider instance to reuse across calls.
        """
        effective_vix = vix_level
        effective_pol = polarization_index
        effective_approval = presidential_approval

        should_fetch = auto_fetch or bool(getattr(config, "EXTERNAL_METRICS_AUTO_FETCH", False))
        if should_fetch and (effective_vix is None or effective_pol is None or effective_approval is None):
            macro_provider = provider or getattr(self, "_macro_provider", None)
            if macro_provider is None:
                try:
                    from data_collectors.external_signals import MacroSignalProvider  # type: ignore

                    macro_provider = MacroSignalProvider()
                except Exception as error:
                    logger.debug("Macro provider unavailable: %s", error)
                    macro_provider = None

            if macro_provider is not None:
                try:
                    fetched = macro_provider.fetch_latest_metrics()
                    effective_vix = effective_vix if effective_vix is not None else fetched.get("vix")
                    effective_pol = effective_pol if effective_pol is not None else fetched.get("polarization")
                    effective_approval = effective_approval if effective_approval is not None else fetched.get("approval")
                    self._macro_provider = macro_provider
                except Exception as error:
                    logger.debug("Macro fetch failed: %s", error)

        if effective_vix is not None:
            self.vix_level = float(effective_vix)
        if effective_pol is not None:
            self.polarization_index = float(effective_pol)
        if effective_approval is not None:
            self.presidential_approval = float(effective_approval)

        self.last_external_metrics = {
            "vix": self.vix_level,
            "polarization": self.polarization_index,
            "approval": self.presidential_approval,
        }
        return self.last_external_metrics

    def get_homophily_scale(self) -> float:
        """Return multiplier for homophily strength based on scenario and polarization."""
        scale = 1.0
        if self.scenario == "polarized":
            scale = 1.15
        elif self.scenario == "crisis":
            scale = 1.05
        if self.polarization_index is not None:
            scale += 0.35 * float(np.clip(self.polarization_index, 0.0, 1.0))
        return float(scale)

    def get_context_vector(self) -> np.ndarray:
        """Return fixed-length Z(t) vector for downstream models (DBN, etc.)."""
        vix = float(self.vix_level) if self.vix_level is not None else 0.0
        pol = float(self.polarization_index) if self.polarization_index is not None else 0.0
        approval = float(self.presidential_approval) if self.presidential_approval is not None else 0.0
        return np.array([
            float(self.base_pressure),
            float(self.volatility_multiplier),
            approval,
            vix,
            pol,
        ], dtype=float)

    def update_regime_state(self, bill_passed: bool, bill_importance: float = 1.0) -> None:
        """Update regime state using outcome feedback from a bill vote.

        Args:
            bill_passed: True if the bill passed.
            bill_importance: Weight of the bill (e.g., 3.0 for major bills).
        """
        # 1. Reaction to failure
        if not bill_passed:
            # Count failures weighted by importance
            self.consecutive_failures += 1.0 * float(bill_importance)

            # In crisis-like scenarios, failures increase base pressure and volatility
            if self.scenario in ("crisis", "polarized"):
                self.base_pressure = min(1.0, self.base_pressure + 0.05 * float(bill_importance))
                self.volatility_multiplier = min(5.0, self.volatility_multiplier + 0.1 * float(bill_importance))
        else:
            # 2. Reaction to success: reduce running failure count and slightly stabilize
            self.consecutive_failures = max(0.0, self.consecutive_failures - 0.5 * float(bill_importance))
            # Slight stabilization toward initial scenario baseline
            initial_pressure = self._get_initial_pressure()
            if self.base_pressure > initial_pressure:
                self.base_pressure = max(initial_pressure, self.base_pressure - 0.02 * float(bill_importance))
                self.volatility_multiplier = max(1.0, self.volatility_multiplier - 0.05 * float(bill_importance))

        # 3. Check for escalation to collapse
        if self.scenario != "collapse" and self.consecutive_failures >= self.crisis_threshold * 2:
            self.scenario = "collapse"
            self.base_pressure = 1.0
            self.volatility_multiplier = 5.0

    def update_regime_state_with_observation(self, acceptance_rate: float) -> None:
        """Helper that interprets a continuous acceptance rate as an observation
        and updates the regime accordingly.

        Uses 0.5 threshold by default to determine if bill effectively passed.
        """
        try:
            passed = float(acceptance_rate) > 0.5
        except Exception:
            passed = True
        # Treat marginal outcomes as partial failures (scaling by distance from 0.5)
        importance = 1.0
        margin = abs(acceptance_rate - 0.5)
        if margin < 0.05:
            # razor-thin outcome -> treat as slightly more important
            importance = 1.5
        self.update_regime_state(passed, bill_importance=importance)

    def _get_initial_pressure(self) -> float:
        """Return initial base pressure based on starting scenario (used for stabilization)."""
        if self.scenario == "stable":
            return 0.1
        elif self.scenario == "polarized":
            return 0.5
        elif self.scenario == "crisis":
            return 0.8
        else:  # collapse
            return 1.0

    # ========================================================================
    # HMM METHODS (optional, uses hmmlearn)
    # ========================================================================
    def init_hmm(self, n_components: int = 4, min_train_length: int = 20) -> None:
        """Initialize an HMM for regime inference.

        Args:
            n_components: Number of discrete hidden states (default 4: stable, polarized, crisis, collapse)
            min_train_length: Minimum number of observations before fitting the model
        """
        if not _HAS_HMMLEARN:
            # Defer requirement until used; provide a helpful message
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("hmmlearn not available; call `init_hmm()` after installing 'hmmlearn' to enable HMM features.")
            self.hmm_model = None
            self.hmm_states = []
            self.hmm_min_train_length = int(min_train_length)
            self.hmm_obs_buffer = []
            self.hmm_component_mapping = None
            return

        self.hmm_model = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=200, random_state=0)
        self.hmm_states = ['stable', 'polarized', 'crisis', 'collapse'][:n_components]
        self.hmm_min_train_length = int(min_train_length)
        self.hmm_obs_buffer = []  # list of observation vectors
        self.hmm_component_mapping = None  # maps component idx -> state name

    def fit_hmm(self, observations: np.ndarray) -> None:
        """Fit HMM to provided observation matrix (T x features).

        After fitting, map HMM components to regime state names by ordering
        the components according to their mean acceptance rate (assumed feature 0).
        """
        if not _HAS_HMMLEARN or self.hmm_model is None:
            return

        obs = np.asarray(observations, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)

        try:
            self.hmm_model.fit(obs)
            # Map components by ordering means on first feature
            try:
                means = np.squeeze(np.array(self.hmm_model.means_))
                # If multi-d, consider first feature (acceptance rate)
                primary_means = means[:, 0] if means.ndim > 1 else means
                order = np.argsort(primary_means)
                mapping = {}
                # Assign lowest mean -> 'stable', highest -> 'collapse'
                names_sorted = self.hmm_states[:len(order)]
                for rank, comp in enumerate(order):
                    mapping[int(comp)] = names_sorted[rank]
                self.hmm_component_mapping = mapping
            except Exception:
                self.hmm_component_mapping = {i: self.hmm_states[i] for i in range(len(self.hmm_states))}
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"HMM fit failed: {e}")

    def predict_hmm_state(self, observation: np.ndarray):
        """Predict HMM component for a single observation and return mapped state name.

        Returns:
            Tuple[state_name:str or None, component_index:int or None]
        """
        if not _HAS_HMMLEARN or self.hmm_model is None:
            return None, None

        obs = np.asarray(observation, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        try:
            comp = int(self.hmm_model.predict(obs)[0])
            state_name = None
            if self.hmm_component_mapping is not None:
                state_name = self.hmm_component_mapping.get(comp, None)
            return state_name, comp
        except Exception:
            return None, None

    def update_hmm_with_observation(self, observation: np.ndarray) -> None:
        """Append observation to buffer, (re)fit if enough data, and apply prediction.

        The observation should be a 1D array of features, e.g. [acceptance_rate, vote_volatility].
        If the predicted HMM state differs from current scenario, update scenario and call
        `_configure_scenario()` to apply new base parameters.
        """
        if not _HAS_HMMLEARN:
            return

        obs = np.asarray(observation, dtype=float).ravel()
        self.hmm_obs_buffer.append(obs)
        # Keep a rolling buffer
        if len(self.hmm_obs_buffer) > 500:
            self.hmm_obs_buffer = self.hmm_obs_buffer[-500:]

        # Fit if buffer large enough (throttled by hmm_fit_interval to avoid excessive re-fitting)
        if len(self.hmm_obs_buffer) >= self.hmm_min_train_length:
            try:
                interval = getattr(self, "hmm_fit_interval", 1)
                if interval <= 1 or (len(self.hmm_obs_buffer) % int(interval) == 0):
                    obs_mat = np.vstack(self.hmm_obs_buffer)
                    self.fit_hmm(obs_mat)
            except Exception:
                pass

        # Predict and apply if model available
        state_name, comp = self.predict_hmm_state(obs)
        self.last_hmm_state = state_name
        self.last_hmm_component = comp
        if state_name is not None and state_name != self.scenario:
            # Map predicted to scenario and adjust
            try:
                # Apply only if predicted state is more extreme or different
                self.scenario = state_name
                self._configure_scenario()
            except Exception:
                pass

    def get_last_hmm_state(self) -> Optional[str]:
        """Return last predicted HMM state name (if available)."""
        return self.last_hmm_state

    def get_last_hmm_component(self) -> Optional[int]:
        """Return last predicted HMM component index (if available)."""
        return self.last_hmm_component

    def get_scenario_name(self) -> str:
        """Return human-readable scenario name."""
        return self.scenario.upper()
    
    def get_state(self) -> dict:
        """Return serializable regime state."""
        return {
            "scenario": self.scenario,
            "current_pressure": self.get_current_pressure(),
            "base_pressure": self.base_pressure,
            "volatility_multiplier": self.volatility_multiplier,
            "time_step": self.time_step,
            "consecutive_failures": self.consecutive_failures,
            "crisis_threshold": self.crisis_threshold,
            "vix_level": self.vix_level,
            "polarization_index": self.polarization_index,
            "presidential_approval": self.presidential_approval,
        }
