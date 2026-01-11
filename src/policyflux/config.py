"""Configuration module for Congressional Dynamics Simulation.

This module centralizes all configuration parameters, making it easy to:
1. Switch between different Congress numbers (116, 117, 118) for validation
2. Adjust simulation parameters without editing main.py
3. Configure behavioral feature normalization
4. Manage scenario and regime settings
"""

from typing import Dict, Literal

# ===========================
# CONGRESS DATA CONFIGURATION
# ===========================

# Congress number to use for data collection (116, 117, 118)
# Supports out-of-sample validation: train on Congress N, test on Congress N+1
CONGRESS_NUMBER: int = 118

# Fallback to mock data if real data cannot be loaded
# NOTE: pyvoteview is slow to fetch large congressional data sets
# Set to False for faster testing with mock data, True for real data
USE_REAL_DATA: bool = True  # Set to True or use CLI: python main.py --real

# Number of mock actors to generate if real data fails
NUM_MOCK_ACTORS: int = 100

# ===========================
# SIMULATION PARAMETERS
# ===========================

# Number of Monte Carlo simulations to run
NUM_SIMULATIONS: int = 100  # Reduced for faster testing (use 1000 for production)

# Number of evolution steps per simulation
SIMULATION_STEPS: int = 10  # Reduced for faster testing (use 50 for production)

# Network sparsity threshold for adjacency matrix
SPARSITY_THRESHOLD: float = 0.9

# ===========================
# SCENARIO AND REGIME
# ===========================

# Political scenario: 'stable', 'polarized', or 'crisis'
SCENARIO: Literal["stable", "polarized", "crisis"] = "crisis"

# Mixed logit toggle and parameters
USE_MIXED_LOGIT: bool = True
MIXED_LOGIT_PARAMS: Dict[str, list[float]] = {
    "gamma": [0.22, 0.18, 0.15, 0.1, 0.08, 0.05, 0.02],
    "delta": [0.4, 0.2, 0.15, 0.1, 0.05]
}

# External data cache for finance, district, and relationship signals
EXTERNAL_SIGNAL_USE_CACHE: bool = True
EXTERNAL_SIGNAL_CACHE_DIR: str = "./cache/external_signals"

# Live macro indicators (optional; off by default to avoid network calls in tests)
EXTERNAL_METRICS_AUTO_FETCH: bool = False
EXTERNAL_METRICS_CACHE_PATH: str = "./cache/external_signals/macro_metrics.json"
EXTERNAL_METRICS_VIX_URL: str = "https://stooq.pl/q/d/l/?s=^vix&d1={d1}&d2={d2}&i=d"
EXTERNAL_METRICS_APPROVAL_URL: str = "https://projects.fivethirtyeight.com/biden-approval-data/approval_topline.csv"
EXTERNAL_METRICS_POLARIZATION_URL: str = "https://voteview.com/static/data/out/party_mean/party_mean_house.csv"
EXTERNAL_METRICS_MAX_AGE_DAYS: int = 1

# ===========================
# BEHAVIORAL DATA NORMALIZATION
# ===========================

# Enable standardization of behavioral features (loyalty, vulnerability, volatility)
NORMALIZE_BEHAVIORAL_FEATURES: bool = True

# Type of normalization: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
NORMALIZATION_TYPE: Literal["standard", "minmax"] = "standard"

# ===========================
# NETWORK AND INFLUENCE
# ===========================

# Weight for cosponsorship blending in adjacency matrix [0, 1]
COSPONSORSHIP_ALPHA: float = 0.5

# Strength of homophily effects in network construction
HOMOPHILY_BETA: float = 2.0

# Amplification factor for leader influence (based on centrality)
LEADER_BOOST: float = 2.0

# Leader centrality and capital tuning
LEADER_CENTRALITY_METHOD: Literal["betweenness", "degree"] = "betweenness"
LEADER_CENTRALITY_THRESHOLD: float = 0.6
LEADER_CAPITAL_SENSITIVITY: float = 2.5
BASE_DEFECTION_COST: float = 0.08
BASE_LOYALTY_GAIN: float = 0.035
DEMOCRATIC_DEFECTOR_MULTIPLIER: float = 1.3
REPUBLICAN_DEFECTOR_MULTIPLIER: float = 1.1
DEMOCRATIC_LOYALTY_BONUS_MULTIPLIER: float = 1.1
REPUBLICAN_LOYALTY_BONUS_MULTIPLIER: float = 1.25
MEMORY_EWMA_ALPHA: float = 0.35
MEMORY_OPINION_WEIGHT: float = 0.45

# Weight for committee overlap when blending into the influence matrix
COMMITTEE_WEIGHT: float = 0.35
# Bonus multiplier for interactions inside the same community
COMMUNITY_REINFORCEMENT: float = 0.15
# Relationship prior weight that biases DBN toward cosponsorship/committee data
RELATIONSHIP_PRIOR_WEIGHT: float = 0.5
# ===========================
# DEEP LEARNING MODELS
# ===========================

# Latent dimension for autoencoders
AUTOENCODER_LATENT_DIM: int = 8

# Preferred TensorFlow device string ("auto", "gpu", "cpu")
TF_DEVICE: str = "auto"

# Ideological latent dimensionality (used across IPM/DQN/LSTM)
IDEOLOGY_DIM: int = 3

# ---------------------------
# DQN (Voting RL) Hyperparameters
# ---------------------------
DQN_STATE_DIM: int = 11
DQN_HIDDEN_DIMS: list[int] = [128, 64]
DQN_LEARNING_RATE: float = 1e-3
DQN_GAMMA: float = 0.99
DQN_EPSILON_START: float = 1.0
DQN_EPSILON_DECAY: float = 0.995
DQN_EPSILON_MIN: float = 0.01
DQN_TARGET_UPDATE_FREQ: int = 1000
DQN_GRAD_CLIP: float = 1.0
DQN_BATCH_SIZE: int = 32

# ---------------------------
# DBN (Ideology Dynamics) Hyperparameters
# ---------------------------
DBN_ALPHA_1: float = 1e-6
DBN_ALPHA_2: float = 1e-6
DBN_LAMBDA_1: float = 1e-6
DBN_LAMBDA_2: float = 1e-6
DBN_MAX_ITER: int = 300
DBN_USE_CV: bool = True
DBN_CV_SPLITS: int = 5
DBN_LEARN_DELTAS: bool = True  # Train on Δx(t) instead of x(t+1)

# ---------------------------
# LSTM (Actor memory) Hyperparameters
# ---------------------------
LSTM_INPUT_DIM: int = IDEOLOGY_DIM
LSTM_HIDDEN_DIM: int = 16
LSTM_OUTPUT_DIM: int = IDEOLOGY_DIM
LSTM_ONLINE_UPDATE_INTERVAL: int = 5  # steps between online updates during simulation

# ---------------------------
# Reward weights (used by RewardComputer)
# ---------------------------
REWARD_LOYALTY_WEIGHT: float = 1.0
REWARD_IPM_WEIGHT: float = 1.0
REWARD_ELECTORAL_WEIGHT: float = 0.5
REWARD_PARTY_ALIGNMENT_BONUS: float = 1.0

# ===========================
# SENSITIVITY ANALYSIS
# ===========================

# Parameters to test in sensitivity analysis
SENSITIVITY_PARAMS: dict = {
    "vulnerability_avg": [0.05, 0.1, 0.2, 0.3],
    "volatility_avg": [0.05, 0.1, 0.15, 0.2],
    "ideology_spread": [0.5, 1.0, 1.5, 2.0],
    # PublicRegime parameters
    "base_pressure": [0.1, 0.3, 0.5, 0.7],
    "volatility_multiplier": [0.5, 1.0, 1.5, 3.0],
}

# ===========================
# TIMING / PERFORMANCE
# ===========================

# Threshold (seconds) for reporting slow steps/sub-operations (0.0 disables)
SLOW_STEP_THRESHOLD_SECONDS: float = 0.05

# ===========================
# LOGGING
# ===========================

# Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
# Default to DEBUG for maximum verbosity across the CLI and API
LOGGING_LEVEL: str = "DEBUG"

# ===========================
# OUTPUT AND VISUALIZATION
# ===========================

# Directory for saving visualizations and results
OUTPUT_DIR: str = "./results"

# Format for saving network visualizations: 'png', 'pdf', 'svg'
VISUALIZATION_FORMAT: str = "png"

# DPI for output images
VISUALIZATION_DPI: int = 150




