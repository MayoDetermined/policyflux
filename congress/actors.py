"""Actor model for congressional simulation.

This file defines the `CongressMan` agent with state, decision logic,
and utility methods used by the simulation engine. Includes behavioral realism
features such as dynamic political capital coupled with network centrality,
party-asymmetric responses, and exponentially-weighted memory of recent votes.

Now includes:
- RNN/LSTM for ideology memory and evolution
- DQN for strategic voting decisions
- Online learning capabilities
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import config

from models.ideal_point import IdealPointModel
from models.rnn import ActorLSTM, ActorLSTMTrainer
from models.dqn import VoteDQN, DQNAgent

if TYPE_CHECKING:
    from congress.law import Law

class CongressMan:
    """Represents a legislator agent with realistic behavioral dynamics.

    Attributes are intentionally simple and typed to ease testing and
    serialization. Supports:
    - Dynamic political capital coupled with network influence (centrality).
    - Party-asymmetric responses to political pressure.
    - Exponentially-weighted moving average memory of voting history.
    - Multi-dimensional ideology representation.
    """

    def __init__(self, id: int, data: Dict, ideal_point_model: 'IdealPointModel' = None) -> None:
        #print("Initializing CongressMan with data:", data)
        self.id: int = int(id)
        # Ideology can be scalar or multi-dimensional.
        ideology_raw = data.get("ideology_multidim")
        #print("id", self.id, "ideology_raw:", ideology_raw, type(ideology_raw))
        if isinstance(ideology_raw, (list, tuple, np.ndarray)):
            self.ideology: np.ndarray = np.array(ideology_raw, dtype=np.float32)
        else:
            self.ideology = np.array([ideology_raw], dtype=np.float32)

        # Store reference to the trained IdealPointModel
        self.ideal_point_model: Optional['IdealPointModel'] = ideal_point_model

        # Behavioral parameters
        self.loyalty: float = float(data.get("loyalty", 0.5))
        self.vulnerability: float = float(data.get("vulnerability", 0.1))
        self.volatility: float = float(data.get("volatility", 0.1))
        self.finance_profile: Dict[str, float] = data.get("finance_signal", {})
        self.district_profile: Dict[str, float] = data.get("district_signal", {})
        self.relationship_profile: Dict[str, float] = data.get("relationship_signal", {})
        self.community_id: int = int(data.get("community_id", data.get("community", 0)))
        self.cosponsor_strength: float = float(self.relationship_profile.get("cosponsor_strength", 0.0))
        self.committee_overlap: float = float(self.relationship_profile.get("committee_overlap", 0.0))
        self.committee_memberships: List[str] = list(self.relationship_profile.get("committee_memberships", []))
        self.community_affinity: float = float(self.relationship_profile.get("community_affinity", 0.0))
        # External/structural features
        self.financial_support: float = float(data.get("financial_support", 0.0))
        self.network_centrality: float = float(data.get("network_centrality", 0.0))
        self.committee_power: float = float(data.get("committee_power", 0.0))
        self.electoral_margin: Optional[float] = data.get("electoral_margin")
        self.district_urbanization: Optional[float] = data.get("district_urbanization")
        self.district_income: Optional[float] = data.get("district_income")
        self.district_stability: Optional[float] = data.get("district_stability")
        self.district_demographics: Dict[str, float] = data.get("district_demographics", {})
        # Mixed Logit random coefficient (heterogeneity)
        # Drawn once per actor at initialization; can be re-drawn for experiments
        self.random_eta: float = float(data.get("random_eta", np.random.normal(0.0, 0.2)))

        # Optional metadata
        self.party: Optional[str] = data.get("party")
        self.role: str = data.get("role", "member")
        self.presidential_support_score: Optional[float] = data.get(
            "presidential_support_score"
        )

        # Dynamic state (scalars)
        self.current_opinion: float = self.get_ideological_position()
        self.vote_prob: float = 0.5
        self.vote_boost: float = 0.0  # Used for bargaining-induced shifts

        # Political capital and path-dependence memory
        base_pc = data.get("political_capital")
        if base_pc is None:
            if "leader" in self.role.lower():
                base_pc = np.random.uniform(5.0, 7.0)
            else:
                base_pc = np.random.uniform(2.0, 5.0)
        self.political_capital: float = float(base_pc)
        self.political_capital_max: float = float(data.get("political_capital_max", 10.0))
        
        # Leader position (centrality) - set by Congress during initialization
        self.centrality: float = float(self.network_centrality)  # Betweenness centrality [0, 1]
        self.is_leader: bool = False  # Flag if centrality exceeds threshold

        # Adjust behavioral knobs using external features
        if self.financial_support:
            support_term = 0.2 * np.tanh(self.financial_support / 1e6)
            self.vulnerability = float(np.clip(self.vulnerability - support_term, 0.0, 1.0))
        if self.electoral_margin is not None:
            margin_clipped = float(np.clip(self.electoral_margin, 0.0, 0.5))
            margin_effect = 0.3 * (0.5 - margin_clipped)
            self.vulnerability = float(np.clip(self.vulnerability + margin_effect, 0.0, 1.0))
        if self.district_stability is not None:
            stability_term = 0.15 * np.tanh(float(self.district_stability))
            self.vulnerability = float(np.clip(self.vulnerability - stability_term, 0.0, 1.0))

        # Recent votes history (list of tuples: (vote:int, against_party:bool))
        self.recent_votes: List[Tuple[int, bool]] = []
        self.vote_history_window: int = int(data.get("vote_history_window", 5))
        
        # Exponentially-weighted moving average for vote memory
        self.ewma_vote_memory: float = 0.0  # Running weighted average of recent votes
        self.ewma_alpha: float = float(data.get("ewma_alpha", config.MEMORY_EWMA_ALPHA))
        self.memory_influence_weight: float = float(data.get("memory_influence_weight", config.MEMORY_OPINION_WEIGHT))
        
        # ============================================================================
        # RNN/LSTM MODEL FOR IDEOLOGY MEMORY
        # ============================================================================
        self.rnn_model: Optional[ActorLSTM] = None  # Initialized later
        self.rnn_trainer: Optional[ActorLSTMTrainer] = None  # Trainer for RNN
        self.ideology_history: List[np.ndarray] = [self.ideology.copy()]  # Memory buffer
        self.max_history_len: int = 5  # Use only last 5 timesteps for prediction
        
        # ============================================================================
        # DQN MODEL FOR VOTING DECISIONS
        # ============================================================================
        self.dqn_agent: Optional[DQNAgent] = None  # Initialized later
        self.epsilon: float = 1.0  # Exploration parameter (starts high)
        self.use_dqn: bool = False  # Flag to enable/disable DQN voting
        self.dqn_state_dim: int = config.DQN_STATE_DIM  # Default state dimension (will adjust on first use)
        self.last_state: Optional[torch.Tensor] = None  # For storing state for learning
        self.last_action: Optional[int] = None  # For storing action for learning
        self.last_reward: Optional[float] = None  # For storing reward for learning

    def get_ideological_position(self) -> float:
        """Compute composite ideological position from all dimensions.
        
        Returns the mean across all ideological dimensions, providing a
        single scalar representation that incorporates all dimensions.
        """
        if self.ideology.size == 0:
            return 0.0
        return float(np.mean(self.ideology))

    def set_centrality(self, centrality: float, leader_threshold: Optional[float] = None) -> None:
        """Set actor's network centrality and update leader status.
        
        Args:
            centrality: Betweenness centrality value in [0, 1].
            leader_threshold: Threshold above which actor is considered a leader.
        """
        self.centrality = float(np.clip(centrality, 0.0, 1.0))
        self.network_centrality = self.centrality
        threshold = leader_threshold if leader_threshold is not None else config.LEADER_CENTRALITY_THRESHOLD
        self.is_leader = self.centrality >= threshold

    def record_vote(self, vote: int, party_line_vote: int) -> None:
        """Record a vote and update political capital with leader asymmetry.

        Political capital changes are now asymmetric based on:
        - Actor's network position (centrality): leaders lose/gain more.
        - Party affiliation: Democrats/Republicans may respond differently.

        Args:
            vote: 1 for "yea", 0 for "nay".
            party_line_vote: party recommended vote (1 or 0).
        """
        against_party = (vote != party_line_vote)
        self.recent_votes.append((int(vote), bool(against_party)))
        if len(self.recent_votes) > self.vote_history_window:
            self.recent_votes.pop(0)

        # Update EWMA vote memory
        vote_signal = float(vote)  # Convert to float in [0, 1]
        self.ewma_vote_memory = (self.ewma_alpha * vote_signal) + \
                                ((1.0 - self.ewma_alpha) * self.ewma_vote_memory)

        centrality_factor = 1.0 + (config.LEADER_CAPITAL_SENSITIVITY * self.centrality)

        if against_party:
            # Calculate party-asymmetric penalty that scales with leader weight
            if self.party == "Democratic":
                party_multiplier = config.DEMOCRATIC_DEFECTOR_MULTIPLIER
            elif self.party == "Republican":
                party_multiplier = config.REPUBLICAN_DEFECTOR_MULTIPLIER
            else:
                party_multiplier = 1.0

            cost = config.BASE_DEFECTION_COST * centrality_factor * party_multiplier
            self.political_capital -= cost
        else:
            # Loyal votes boost capital, leaders gain extra for keeping the bloc together
            if self.party == "Democratic":
                party_multiplier = config.DEMOCRATIC_LOYALTY_BONUS_MULTIPLIER
            elif self.party == "Republican":
                party_multiplier = config.REPUBLICAN_LOYALTY_BONUS_MULTIPLIER
            else:
                party_multiplier = 1.0

            leader_bonus = 1.0 + (0.15 * self.centrality) if self.is_leader else 1.0
            regen = config.BASE_LOYALTY_GAIN * centrality_factor * party_multiplier * leader_bonus
            self.political_capital += regen

        self.political_capital = float(np.clip(self.political_capital, 0.0, self.political_capital_max))
        self._refresh_opinion_from_memory(vote_signal)

    def _refresh_opinion_from_memory(self, vote_signal: float) -> None:
        """Blend the EWMA memory with the most recent vote to update opinion."""
        weight = float(np.clip(self.memory_influence_weight, 0.0, 1.0))
        if weight <= 0.0:
            return

        ewma_component = self.get_ewma_opinion()
        vote_component = (vote_signal * 2.0) - 1.0
        blended_target = (0.7 * ewma_component) + (0.3 * vote_component)
        updated_opinion = (1.0 - weight) * self.current_opinion + (weight * blended_target)
        self.current_opinion = float(np.clip(updated_opinion, -1.0, 1.0))

    def regenerate_political_capital(self, voted_with_party: bool, party_leader_boost: float = 0.5) -> None:
        """Regenerate political capital and loyalty based on recent conformity."""

        if voted_with_party:
            self.political_capital += 0.1 * self.loyalty
            self.loyalty = min(1.0, self.loyalty + 0.01)
        else:
            penalty = 0.5 * (1.0 - self.loyalty)
            self.political_capital -= penalty
            self.loyalty = max(0.0, self.loyalty - 0.02)

        if self.get_influence_score() > 0.8:
            self.political_capital += party_leader_boost

        self.political_capital = float(np.clip(self.political_capital, 0.0, self.political_capital_max))

    def get_political_capital_factor(self) -> float:
        """Return normalized political capital factor in [0,1]."""
        if self.political_capital_max <= 0:
            return 0.0
        return self.political_capital / self.political_capital_max

    def get_recent_defection_rate(self) -> float:
        """Compute fraction of recent votes cast against the party line."""
        if not self.recent_votes:
            return 0.0
        defections = sum(1 for _, ag in self.recent_votes if ag)
        return defections / len(self.recent_votes)

    def get_ewma_opinion(self) -> float:
        """Get opinion based on exponentially-weighted voting memory.
        
        Returns a value in [-1, 1] where:
        - 1.0 = consistently voting for (yea)
        - -1.0 = consistently voting against (nay)
        - 0.0 = neutral/mixed voting pattern
        """
        return (2.0 * self.ewma_vote_memory) - 1.0

    def calculate_decision_probability(
        self, network_pressure: float, global_context: float, party_line_vote: float = 0.5,
        use_ewma_memory: bool = True
    ) -> float:
        """Compute vote probability based on internal, network and global factors.

        Enhanced with exponentially-weighted memory where recent, high-salience votes
        have greater influence on current opinion than historical votes.

        The function updates `self.vote_prob` and `self.current_opinion` as side effects.
        
        Args:
            network_pressure: Pressure from network/colleagues.
            global_context: Global political context (regime pressure).
            party_line_vote: Party's recommended position (0 or 1).
            use_ewma_memory: Whether to incorporate EWMA voting memory.
        """
        # Internal ideological factor (composite from all dimensions)
        internal_factor = self.get_ideological_position()

        # Network factor scaled by vulnerability
        network_factor = float(network_pressure) * self.vulnerability

        # System/global factor: amplified for low-loyalty actors
        loyalty_factor = 1.0 - self.loyalty
        nonlinear_multiplier = loyalty_factor ** 2
        system_factor = float(global_context) * self.loyalty * (1.0 + nonlinear_multiplier)

        # Interaction term
        network_system_interaction = network_factor * system_factor * (1.0 - self.loyalty)

        u_t = internal_factor + network_factor + system_factor + 0.1 * network_system_interaction

        # Political capital pulls towards party line when capital is low
        capital_factor = self.get_political_capital_factor()
        party_pressure = (1.0 - capital_factor) * 0.3
        party_line_signal = (party_line_vote * 2.0 - 1.0)
        u_t += party_pressure * party_line_signal

        # Incorporate exponentially-weighted voting memory if enabled
        memory_weight = float(np.clip(self.memory_influence_weight, 0.0, 1.0))
        if use_ewma_memory and memory_weight > 0.0:
            ewma_opinion = self.get_ewma_opinion()  # In [-1, 1]
            # Recent vote pattern influences current decision, scaled by memory weight
            u_t += 0.15 * memory_weight * ewma_opinion

        # Add stochastic noise according to volatility
        noise = np.random.normal(0.0, self.volatility)
        u_t = float(u_t) + float(noise)

        # Sigmoid activation -> probability in [0,1]
        self.vote_prob = float(1.0 / (1.0 + np.exp(-u_t)))

        # Update opinion: blend current opinion with EWMA memory and decision
        # EWMA voting pattern has 30% weight, current decision has 20% weight
        if use_ewma_memory:
            ewma_component = 0.3 * memory_weight * self.get_ewma_opinion()
            decision_component = 0.2 * (self.vote_prob * 2.0 - 1.0)
            inertia = 0.5  # Existing opinion inertia
            self.current_opinion = (inertia * self.current_opinion) + \
                                  ewma_component + decision_component
        else:
            # Original behavior when EWMA disabled
            self.current_opinion = (self.current_opinion * 0.8) + \
                                  ((self.vote_prob * 2.0 - 1.0) * 0.2)

        return self.vote_prob

    def cast_vote(self) -> int:
        """Sample a binary vote from the current vote probability."""
        return 1 if np.random.random() < self.vote_prob else 0

    def get_influence_score(self) -> float:
        """Influence proxy combining loyalty, resilience, and network centrality."""
        # Base influence from loyalty and robustness
        base_influence = (self.loyalty + (1.0 - self.vulnerability)) / 2.0
        # Amplify by network centrality (leaders have more influence)
        centrality_factor = 1.0 + self.centrality
        return float(base_influence * centrality_factor)

    def get_ideological_distance(self, other: "CongressMan") -> float:
        """Euclidean distance between ideological vectors (supports multi-dim)."""
        if isinstance(self.ideology, np.ndarray) and isinstance(other.ideology, np.ndarray):
            return float(np.linalg.norm(self.ideology - other.ideology))
        if isinstance(self.ideology, np.ndarray):
            return float(abs(self.ideology[0] - float(other.ideology)))
        return float(abs(float(self.ideology) - float(other.ideology)))

    def flip_vote(self, pressure_threshold: float = 0.7) -> bool:
        """Randomly flip `vote_prob` under pressure; returns whether flipped."""
        if np.random.random() < (self.vulnerability * pressure_threshold):
            self.vote_prob = 1.0 - self.vote_prob
            return True
        return False
    
    # ============================================================================
    # RNN/LSTM METHODS FOR IDEOLOGY EVOLUTION
    # ============================================================================
    
    def init_rnn_model(
        self,
        input_dim: int = config.LSTM_INPUT_DIM,
        hidden_dim: int = config.LSTM_HIDDEN_DIM,
        output_dim: int = config.LSTM_OUTPUT_DIM
    ) -> None:
        """Initialize RNN model for this actor.
        
        Args:
            input_dim: Dimensionality of ideology (e.g., 3 for 3D political space).
            hidden_dim: Size of LSTM hidden state.
            output_dim: Output ideology dimensionality.
        """
        self.rnn_model = ActorLSTM(input_dim, hidden_dim, output_dim)
        self.rnn_trainer = ActorLSTMTrainer(self.rnn_model, learning_rate=1e-4)
    
    def _get_rnn_input_data(self, current_pressure: float) -> torch.Tensor:
        """Format ideology history and pressure into RNN input tensor.
        
        Creates a sequence of (ideology, pressure) pairs where each ideology
        vector is augmented with the current pressure value.
        
        Args:
            current_pressure: Current regime pressure value.
        
        Returns:
            Tensor of shape (1, max_history_len, input_dim + 1) where input_dim + 1
            accounts for ideology dimensions plus pressure.
        """
        if not self.rnn_model:
            raise RuntimeError("RNN model not initialized. Call init_rnn_model() first.")
        
        # Augment each ideology in history with pressure
        history_with_pressure = []
        for ideology_vec in self.ideology_history:
            # Ensure ideology is a numpy array
            if isinstance(ideology_vec, torch.Tensor):
                ideology_vec = ideology_vec.numpy()
            elif not isinstance(ideology_vec, np.ndarray):
                ideology_vec = np.array([ideology_vec])
            
            # Append pressure to ideology vector
            augmented = np.append(ideology_vec, current_pressure)
            history_with_pressure.append(augmented)
        
        # Pad history if too short
        while len(history_with_pressure) < self.max_history_len:
            padding = np.zeros_like(history_with_pressure[0])
            history_with_pressure.insert(0, padding)
        
        # Keep only last max_history_len entries
        history_with_pressure = history_with_pressure[-self.max_history_len:]
        
        # Convert to tensor efficiently: (1, max_history_len, input_dim + 1)
        history_array = np.asarray(history_with_pressure, dtype=np.float32)
        history_tensor = torch.from_numpy(history_array).unsqueeze(0)
        
        return history_tensor
    
    def train_rnn_step(self, target_ideology: np.ndarray, current_pressure: float) -> Optional[float]:
        """Perform one RNN training step (online learning).
        
        Args:
            target_ideology: Target ideology vector for this step.
            current_pressure: Current pressure value.
        
        Returns:
            Loss value, or None if training failed.
        """
        if not self.rnn_model or not self.rnn_trainer:
            return None
        
        try:
            # Prepare input data
            history_data = self._get_rnn_input_data(current_pressure)
            
            # Prepare target
            target_tensor = torch.tensor(target_ideology, dtype=torch.float32).unsqueeze(0)
            
            # Training step
            loss = self.rnn_trainer.train_step(history_data, target_tensor)
            
            return loss
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"RNN training failed for actor {self.id}: {e}")
            return None
    
    def predict_ideology_rnn(self, current_pressure: float) -> np.ndarray:
        """Predict next ideology state using RNN.
        
        Args:
            current_pressure: Current regime pressure.
        
        Returns:
            Predicted ideology vector as numpy array.
        """
        if not self.rnn_model or not self.rnn_trainer:
            return self.ideology.copy()
        
        try:
            history_data = self._get_rnn_input_data(current_pressure)
            predicted = self.rnn_trainer.predict(history_data)
            return predicted.squeeze(0).detach().numpy()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"RNN prediction failed for actor {self.id}: {e}")
            return self.ideology.copy()
    
    def update_ideology_history(self, new_ideology: Optional[np.ndarray] = None) -> None:
        """Record current ideology in history buffer.
        
        Args:
            new_ideology: Ideology to record. If None, uses self.ideology.
        """
        if new_ideology is None:
            new_ideology = self.ideology.copy()
        else:
            new_ideology = np.array(new_ideology, dtype=np.float32)
        
        self.ideology_history.append(new_ideology)
        
        # Keep only last max_history_len entries
        if len(self.ideology_history) > self.max_history_len:
            self.ideology_history = self.ideology_history[-self.max_history_len:]
    
    # ============================================================================
    # DQN METHODS FOR VOTING DECISIONS
    # ============================================================================
    
    def init_dqn_agent(self, state_dim: int = config.DQN_STATE_DIM, hidden_dims: List[int] = None, **dqn_kwargs) -> None:
        """Initialize DQN agent for voting decisions.
        
        Accepts the common DQN hyperparameters and forwards any additional
        keyword args to the underlying `DQNAgent` constructor. This makes it
        compatible with `CongressSimulator.dqn_params` and other call sites.

        Args:
            state_dim: Dimensionality of state vector (default 11).
            hidden_dims: Hidden layer dimensions for DQN network.
            **dqn_kwargs: Additional keyword args forwarded to `DQNAgent`.
        """
        self.dqn_state_dim = state_dim
        # Merge defaults with provided kwargs to keep backward compatibility
        agent_kwargs = dict(
            state_dim=state_dim,
            hidden_dims=hidden_dims or config.DQN_HIDDEN_DIMS,
            learning_rate=config.DQN_LEARNING_RATE,
            gamma=config.DQN_GAMMA,
            epsilon=config.DQN_EPSILON_START,
            epsilon_decay=config.DQN_EPSILON_DECAY,
            epsilon_min=config.DQN_EPSILON_MIN,
            target_update_freq=config.DQN_TARGET_UPDATE_FREQ,
        )
        agent_kwargs.update(dqn_kwargs or {})

        self.dqn_agent = DQNAgent(**agent_kwargs)
        self.use_dqn = True
    
    def _get_dqn_state_vector(
        self,
        law: "Law",
        influence_net: float,
        regime_pressure: float
    ) -> torch.Tensor:
        """Build DQN state vector from current context.
        
        State includes:
        - Actor ideology x_i(t) [3 dimensions]
        - Loyalty [1 dimension]
        - Vulnerability [1 dimension]
        - Current pressure Z(t) [1 dimension]
        - Law salience a_law [3 dimensions]
        - Law threshold b_law [1 dimension]
        - Network influence [1 dimension]
        
        Total: 11 dimensions
        
        Args:
            law: Current law/bill.
            influence_net: Network influence value.
            regime_pressure: Regime pressure value.
        
        Returns:
            State tensor of shape (11,).
        """
        # Ensure ideology is proper shape
        ideology_vec = np.array(self.ideology, dtype=np.float32)
        if ideology_vec.ndim == 0:
            ideology_vec = ideology_vec.reshape(1)
        
        # Ensure law salience is proper shape
        law_salience = np.array(law.salience, dtype=np.float32)
        if law_salience.ndim == 0:
            law_salience = law_salience.reshape(1)
        
        # Pad or trim to 3 dimensions for consistency
        if len(ideology_vec) < 3:
            ideology_vec = np.pad(ideology_vec, (0, 3 - len(ideology_vec)), mode='constant')
        else:
            ideology_vec = ideology_vec[:3]
        
        if len(law_salience) < 3:
            law_salience = np.pad(law_salience, (0, 3 - len(law_salience)), mode='constant')
        else:
            law_salience = law_salience[:3]
        
        # Build state vector
        state = np.concatenate([
            ideology_vec,  # [3]
            [self.loyalty],  # [1]
            [self.vulnerability],  # [1]
            [regime_pressure],  # [1]
            law_salience,  # [3]
            [law.threshold],  # [1]
            [influence_net]  # [1]
        ], dtype=np.float32)
        
        return torch.tensor(state, dtype=torch.float32)
    
    def decide_vote_dqn(
        self,
        law: "Law",
        influence_net: float,
        regime_pressure: float,
        use_exploration: bool = True
    ) -> bool:
        """Make voting decision using DQN.
        
        Args:
            law: Current law/bill.
            influence_net: Network influence value.
            regime_pressure: Regime pressure value.
            use_exploration: Whether to use epsilon-greedy exploration.
        
        Returns:
            Boolean vote (True=yes, False=no).
        """
        if not self.dqn_agent:
            # Fallback to IPM if DQN not initialized
            return self.decide_vote(law, influence_net, regime_pressure)
        
        try:
            # Build state
            state = self._get_dqn_state_vector(law, influence_net, regime_pressure)
            
            # Select action (0=against, 1=for)
            action = self.dqn_agent.select_action(state, use_exploration=use_exploration)
            
            # Store for learning
            self.last_state = state
            self.last_action = action
            
            return bool(action)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"DQN decision failed for actor {self.id}: {e}")
            return self.decide_vote(law, influence_net, regime_pressure)
    
    def record_dqn_reward(
        self,
        next_state: Optional[torch.Tensor] = None,
        done: bool = False
    ) -> None:
        """Record reward and learn from experience (RL training step).
        
        Args:
            next_state: Next state after action. If None, creates zero state.
            done: Whether episode terminated.
        """
        if not self.dqn_agent or self.last_state is None:
            return
        
        if next_state is None:
            next_state = torch.zeros_like(self.last_state)
        
        try:
            # Add to replay buffer
            self.dqn_agent.remember(
                state=self.last_state,
                action=self.last_action,
                reward=self.last_reward or 0.0,
                next_state=next_state,
                done=done
            )
            
            # Perform training step
            self.dqn_agent.train_step(batch_size=config.DQN_BATCH_SIZE)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"DQN reward recording failed for actor {self.id}: {e}")

    def get_state_vector(self) -> Dict:
        """Return serializable state dictionary for the actor."""
        ideology_val = self.get_ideological_position()
        return {
            "id": self.id,
            "ideology": ideology_val,
            "ideology_multidim": self.ideology.tolist() if isinstance(self.ideology, np.ndarray) else [ideology_val],
            "loyalty": self.loyalty,
            "vulnerability": self.vulnerability,
            "volatility": self.volatility,
            "party": self.party,
            "presidential_support_score": self.presidential_support_score,
            "current_opinion": self.current_opinion,
            "vote_prob": self.vote_prob,
            "centrality": self.centrality,
            "is_leader": self.is_leader,
            "ewma_vote_memory": self.ewma_vote_memory,
        }

    def calculate_base_vote_prob(self, law: "Law") -> float:
        """Calculate base voting probability using Ideal Point Model (IPM).
        
        Implements the core IPM formula:
        P(YES) = sigmoid(a_law · x_i - b_law)
        
        Where:
        - a_law: Law's salience vector (discriminant across ideological dimensions)
        - x_i: Legislator's multi-dimensional ideal point
        - b_law: Law's threshold (difficulty of passing)
        - sigmoid: Logistic function to convert linear predictor to probability [0, 1]
        
        Args:
            law: Law object with salience vector and threshold parameter.
                Expected to have attributes:
                - law.salience: numpy array of shape (dim,)
                - law.threshold: float scalar
        
        Returns:
            Base voting probability in [0, 1] based purely on ideology vs law position.
        
        Raises:
            TypeError: If law doesn't have required attributes.
            ValueError: If dimension mismatch between legislator ideology and law salience.
        """
        if not hasattr(law, 'salience') or not hasattr(law, 'threshold'):
            raise TypeError("law must be a Law object with 'salience' and 'threshold' attributes")
        
        # Ensure legislator ideology is multi-dimensional
        legislator_position = np.asarray(self.ideology, dtype=np.float32)
        law_salience = np.asarray(law.salience, dtype=np.float32)
        
        # Validate dimensions match
        if legislator_position.ndim == 0:
            legislator_position = legislator_position.reshape(1)
        if law_salience.ndim == 0:
            law_salience = law_salience.reshape(1)
        
        if len(legislator_position) != len(law_salience):
            raise ValueError(
                f"Dimension mismatch: legislator ideology has {len(legislator_position)} dimensions "
                f"but law salience has {len(law_salience)} dimensions"
            )
        
        # Calculate linear predictor: a_law · x_i - b_law
        linear_predictor = np.dot(law_salience, legislator_position) - law.threshold
        # Mixed Logit heterogeneity: actor-specific random coefficient perturbs utility
        linear_predictor = linear_predictor + self.random_eta * (1.0 - self.loyalty)
        
        # Apply sigmoid (logistic function) to convert to probability
        # Clip to prevent numerical overflow in exp
        linear_predictor_clipped = np.clip(linear_predictor, -10, 10)
        P_yes = 1.0 / (1.0 + np.exp(-linear_predictor_clipped))
        
        return float(P_yes)

    def _build_context_features(self, local_regime_pressure: float) -> np.ndarray:
        """Build district/finance feature vector for Mixed Logit context Z."""
        local_pressure = float(np.clip(local_regime_pressure, 0.0, 1.0))
        months_to_election = float(self.finance_profile.get("months_to_election", 24))
        months_norm = np.clip(months_to_election / 24.0, 0.0, 1.0)
        pac_share = float(np.clip(self.finance_profile.get("pac_share", 0.0), 0.0, 1.0))
        median_income = float(self.district_profile.get("median_income", 70000.0))
        income_norm = np.clip((median_income - 40000.0) / 100000.0, 0.0, 1.0)
        unemployment = float(self.district_profile.get("unemployment_rate", 0.05))
        unemployment_norm = np.clip(unemployment / 0.2, 0.0, 1.0)
        pres_margin = abs(float(self.district_profile.get("presidential_margin", 0.0)))
        diversity = float(np.clip(self.district_profile.get("racial_diversity", 0.5), 0.0, 1.0))

        features = [
            local_pressure,
            months_norm,
            pac_share,
            income_norm,
            unemployment_norm,
            pres_margin,
            diversity,
        ]

        return np.clip(np.array(features, dtype=float), -1.0, 1.0)

    def _build_loyalty_vector(self) -> np.ndarray:
        """Build loyalty/relationship feature vector L that mixes with delta and eta."""
        cosponsor_strength = np.clip(self.cosponsor_strength, 0.0, 1.0)
        committee_overlap = np.clip(self.committee_overlap, 0.0, 1.0)
        community_affinity = np.clip(self.community_affinity, 0.0, 1.0)
        return np.array([
            self.loyalty,
            self.centrality,
            cosponsor_strength,
            committee_overlap,
            community_affinity,
        ], dtype=float)

    def calculate_mixed_logit_prob(self, law: "Law", local_regime_pressure: float,
                                   context_features: Optional[np.ndarray] = None) -> float:
        """Mixed Logit probability that incorporates hidden heterogeneity and district context.

        Implements:
        P(yes) = sigma(a_j · x_i - b_j + Z_i · gamma + L_i · (delta + eta_i))
        where Z_i contains district/regime features, L_i contains loyalty-based signals,
        and eta_i is actor-specific heterogeneity drawn at initialization.
        """
        try:
            law_salience = np.asarray(law.salience, dtype=np.float32)
            legislator_position = np.asarray(self.ideology, dtype=np.float32)
            if legislator_position.ndim == 0:
                legislator_position = legislator_position.reshape(1)
            if len(legislator_position) != len(law_salience):
                return self.calculate_base_vote_prob(law)

            base_term = float(np.dot(law_salience, legislator_position)) - float(law.threshold)

            if context_features is None:
                Z = self._build_context_features(local_regime_pressure)
            else:
                Z = np.asarray(context_features, dtype=float).ravel()

            gamma = np.array(config.MIXED_LOGIT_PARAMS.get("gamma", [0.25, 0.15, 0.15]), dtype=float)
            if Z.size < gamma.size:
                Z = np.pad(Z, (0, gamma.size - Z.size), mode='constant')
            else:
                Z = Z[:gamma.size]

            L = self._build_loyalty_vector()
            delta = np.array(config.MIXED_LOGIT_PARAMS.get("delta", [0.4, 0.1]), dtype=float)
            if delta.size < L.size:
                delta = np.pad(delta, (0, L.size - delta.size), mode='constant')
            elif delta.size > L.size:
                L = L[:delta.size]

            hetero = delta + self.random_eta
            logistic_input = base_term + float(np.dot(Z, gamma)) + float(np.dot(L, hetero))
            logistic_input = np.clip(logistic_input, -10.0, 10.0)
            return float(1.0 / (1.0 + np.exp(-logistic_input)))
        except Exception:
            return self.calculate_base_vote_prob(law)

    def decide_vote(self, law: "Law", influence_effect: float = 0.0, regime_pressure: float = 0.0) -> bool:
        """Make a voting decision combining IPM, DBN influence, and regime pressure.

        Enhancements:
        - Localized regime pressure: scaled by district/presidential support
        - Asymmetric party response already in place
        - Heterogeneity (random_eta) is used in base IPM utility
        """
        # Localize regime pressure by district-level presidential support (lower support -> higher local pressure)
        pres_support = float(self.presidential_support_score) if self.presidential_support_score is not None else 0.0
        local_regime_pressure = float(regime_pressure) * (1.0 - pres_support)

        # Step 1: Base probability from IPM or Mixed Logit if enabled
        try:
            if config.USE_MIXED_LOGIT:
                base_prob = self.calculate_mixed_logit_prob(law, local_regime_pressure)
            else:
                base_prob = self.calculate_base_vote_prob(law)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"IPM/Mixed Logit calculation failed for actor {self.id}: {e}. Using random baseline.")
            base_prob = 0.5

        # Step 2: Network influence modification
        susceptibility = 1.0 - self.loyalty
        influence_modification = influence_effect * susceptibility

        modified_prob = np.clip(base_prob + influence_modification, 0.0, 1.0)

        # Apply bargaining-induced boost if present
        if self.vote_boost > 0.0:
            modified_prob = np.clip(modified_prob + self.vote_boost, 0.0, 1.0)
            self.vote_boost = 0.0

        # Step 2.5: Asymmetric regime influence based on party and law direction
        try:
            is_right_wing_law = bool(law.salience[0] > 0)
        except Exception:
            is_right_wing_law = True

        regime_factor = 0.0
        if (self.party == 'Republican' and is_right_wing_law) or (self.party == 'Democratic' and not is_right_wing_law):
            # Party-aligned: pressure increases conformity (loyalty helps)
            regime_factor = self.loyalty * local_regime_pressure * 0.1
        else:
            # Opposition or cross-pressured: pressure may increase resistance
            regime_factor = -self.vulnerability * local_regime_pressure * 0.05

        # Apply regime factor before adding stochastic volatility
        modified_prob = np.clip(modified_prob + regime_factor, 0.0, 1.0)

        # Step 3: Regime pressure adds volatility (stochastic)
        regime_volatility = self.volatility * local_regime_pressure
        noise = np.random.normal(0.0, regime_volatility)
        final_prob = np.clip(modified_prob + noise, 0.0, 1.0)

        # Step 4: Make binary decision
        vote_result = bool(np.random.random() < final_prob)

        # Store for reference
        self.vote_prob = final_prob
        return vote_result

    def decide_vote_legacy(self, law: "Law") -> float:
        """Compute voting probability for a law using ideal point model.
        
        Uses the ideal point model formula:
        P(vote=1) = sigmoid(salience_magnitude * (x_i · normalized_salience - threshold))
        
        This integrates with the Law class to compute voting decisions based on
        the legislator's multi-dimensional ideology and the law's parameters.
        
        Args:
            law: Law object with salience vector and threshold parameter.
        
        Returns:
            Voting probability in [0, 1].
        
        Requires:
            - self.ideology: multi-dimensional ideal point (numpy array)
            - law.salience: multi-dimensional salience vector
            - law.threshold: scalar threshold parameter
        """
        if not hasattr(law, 'get_voting_probability'):
            raise TypeError("law must be a Law object with get_voting_probability method")
        
        # Use the law's method to compute voting probability
        # This ensures consistency with the IPM definition
        try:
            vote_prob = law.get_voting_probability(self.ideology)
            self.vote_prob = float(vote_prob)
            return self.vote_prob
        except Exception as e:
            # Fallback if dimension mismatch or other error
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Law voting computation failed for actor {self.id}: {e}. Using random decision.")
            self.vote_prob = np.random.uniform(0.0, 1.0)
            return self.vote_prob
