"""Default configuration values for PolicyFlux.

These defaults are used when no explicit configuration is provided.
Override by passing parameters directly to classes or using a config dict.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Literal


@dataclass
class SimulationDefaults:
    """Default simulation parameters."""
    
    # Congress data
    CONGRESS_NUMBER: int = 118
    USE_REAL_DATA: bool = True
    NUM_MOCK_ACTORS: int = 100
    
    # Simulation runs
    NUM_SIMULATIONS: int = 100
    SIMULATION_STEPS: int = 10
    
    # Scenario
    SCENARIO: Literal["stable", "polarized", "crisis"] = "stable"
    
    # Network
    SPARSITY_THRESHOLD: float = 0.9
    COSPONSORSHIP_ALPHA: float = 0.5
    HOMOPHILY_BETA: float = 2.0
    LEADER_BOOST: float = 2.0
    COMMITTEE_WEIGHT: float = 0.35
    RELATIONSHIP_PRIOR_WEIGHT: float = 0.5
    
    # Behavioral normalization
    NORMALIZE_BEHAVIORAL_FEATURES: bool = True
    NORMALIZATION_TYPE: Literal["standard", "minmax"] = "standard"
    
    # External metrics
    EXTERNAL_METRICS_AUTO_FETCH: bool = False
    
    # Reproducibility
    RNG_SEED: int = 42


@dataclass
class ModelDefaults:
    """Default model hyperparameters."""
    
    # Autoencoder
    AUTOENCODER_LATENT_DIM: int = 8
    
    # Ideology
    IDEOLOGY_DIM: int = 3
    
    # DQN
    DQN_STATE_DIM: int = 11
    DQN_HIDDEN_DIMS: List[int] = field(default_factory=lambda: [128, 64])
    DQN_LEARNING_RATE: float = 1e-3
    DQN_GAMMA: float = 0.99
    DQN_EPSILON_START: float = 1.0
    DQN_EPSILON_DECAY: float = 0.995
    DQN_EPSILON_MIN: float = 0.01
    DQN_TARGET_UPDATE_FREQ: int = 1000
    DQN_GRAD_CLIP: float = 1.0
    
    # DBN
    DBN_ALPHA_1: float = 1e-6
    DBN_ALPHA_2: float = 1e-6
    DBN_LAMBDA_1: float = 1e-6
    DBN_LAMBDA_2: float = 1e-6
    DBN_MAX_ITER: int = 300
    DBN_USE_CV: bool = True
    DBN_CV_SPLITS: int = 5
    DBN_LEARN_DELTAS: bool = True
    
    # LSTM
    LSTM_INPUT_DIM: int = 3  # = IDEOLOGY_DIM
    LSTM_HIDDEN_DIM: int = 16
    LSTM_OUTPUT_DIM: int = 3  # = IDEOLOGY_DIM


@dataclass  
class PathDefaults:
    """Default paths for caching and output."""
    
    CACHE_DIR: str = "./cache"
    OUTPUT_DIR: str = "./results"
    EXTERNAL_SIGNAL_CACHE_DIR: str = "./cache/external_signals"
    EXTERNAL_METRICS_CACHE_PATH: str = "./cache/external_signals/macro_metrics.json"
    EXTERNAL_METRICS_VIX_URL: str = "https://stooq.pl/q/d/l/?s=^vix&d1={d1}&d2={d2}&i=d"
    EXTERNAL_METRICS_APPROVAL_URL: str = "https://projects.fivethirtyeight.com/biden-approval-data/approval_topline.csv"
    EXTERNAL_METRICS_POLARIZATION_URL: str = "https://voteview.com/static/data/out/party_mean/party_mean_house.csv"
    EXTERNAL_METRICS_MAX_AGE_DAYS: int = 1


# Global defaults instances (use these for backward compatibility)
SIMULATION = SimulationDefaults()
MODELS = ModelDefaults()
PATHS = PathDefaults()


# Backward compatibility: expose as module-level constants
# This allows `from policyflux.defaults import CONGRESS_NUMBER` to work
def __getattr__(name: str):
    """Allow access to defaults as module attributes for backward compatibility."""
    if hasattr(SIMULATION, name):
        return getattr(SIMULATION, name)
    if hasattr(MODELS, name):
        return getattr(MODELS, name)
    if hasattr(PATHS, name):
        return getattr(PATHS, name)
    raise AttributeError(f"module 'policyflux.defaults' has no attribute '{name}'")
