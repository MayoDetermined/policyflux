"""Deep Q-Network for congressional voting decisions.

This module provides:
- VoteDQN: A neural network for learning voting strategies (action: 0=Against, 1=For)
- ReplayBuffer: Experience replay memory for stable off-policy learning
- DQNAgent: Trainer with epsilon-greedy exploration and target networks
"""

from typing import Tuple, List, Optional
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from gpu_utils import get_torch_device


class VoteDQN(nn.Module):
    """Deep Q-Network for voting decisions.
    
    Maps state vectors to Q-values for two actions:
    - Action 0: Vote Against (Nay)
    - Action 1: Vote For (Yea)
    
    State vector S(t) includes:
    - Actor ideology x_i(t) [3 dimensions]
    - Loyalty [1 dimension]
    - Vulnerability [1 dimension]
    - Current pressure Z(t) [1 dimension]
    - Law salience a_law [3 dimensions]
    - Law threshold b_law [1 dimension]
    - Network influence [1 dimension]
    
    Total input: 11 dimensions
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = None,
        output_dim: int = 2
    ):
        """Initialize VoteDQN.
        
        Args:
            state_dim: Dimensionality of state vector.
            hidden_dims: List of hidden layer dimensions. Default: [128, 64].
            output_dim: Number of actions (default: 2 for vote against/for).
        """
        super(VoteDQN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = config.DQN_HIDDEN_DIMS
        
        self.state_dim = state_dim
        self.output_dim = output_dim
        
        # Build multi-layer feedforward network
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer: Q-values for each action
        layers.append(nn.Linear(input_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for state.
        
        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).
        
        Returns:
            Tensor of shape (batch_size, output_dim) or (output_dim,) with Q-values.
        """
        return self.network(state)
    
    def select_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0,
        device: Optional[torch.device] = None
    ) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: State tensor of shape (state_dim,).
            epsilon: Exploration probability [0, 1].
            device: Device for computation.
        
        Returns:
            Action (0 or 1).
        """
        if device is None:
            device = get_torch_device()
        
        # Exploration: random action
        if random.random() < epsilon:
            return random.randint(0, self.output_dim - 1)
        
        # Exploitation: greedy action from Q-values
        with torch.no_grad():
            state_tensor = state.to(device) if isinstance(state, torch.Tensor) else \
                          torch.tensor(state, dtype=torch.float32, device=device)
            
            # Add batch dimension if needed
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            q_values = self.forward(state_tensor)
            action = q_values.argmax(dim=1).item() if q_values.dim() > 1 else q_values.argmax().item()
        
        return action


class ReplayBuffer:
    """Experience replay buffer for stable DQN training.
    
    Stores transitions (state, action, reward, next_state, done) and samples
    random minibatches for training.
    """
    
    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ) -> None:
        """Add a transition to the buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            done: Whether episode terminated.
        """
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch of transitions.
        
        Args:
            batch_size: Size of batch to sample.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            all as torch tensors.
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        transitions = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # Convert to tensors
        states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) 
                             for s in states])
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) 
                                  for s in next_states])
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)


class DQNAgent:
    """Agent for training VoteDQN with experience replay and target networks.
    
    Implements standard DQN algorithm with:
    - Experience replay for decorrelation
    - Target network for stability
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        state_dim: int = config.DQN_STATE_DIM,
        hidden_dims: List[int] = None,
        learning_rate: float = config.DQN_LEARNING_RATE,
        gamma: float = config.DQN_GAMMA,
        epsilon: float = config.DQN_EPSILON_START,
        epsilon_decay: float = config.DQN_EPSILON_DECAY,
        epsilon_min: float = config.DQN_EPSILON_MIN,
        target_update_freq: int = config.DQN_TARGET_UPDATE_FREQ,
        device: Optional[torch.device] = None
    ):
        """Initialize DQN agent.
        
        Args:
            state_dim: Dimensionality of state.
            hidden_dims: Hidden layer dimensions for network.
            learning_rate: Learning rate for optimizer.
            gamma: Discount factor for future rewards.
            epsilon: Initial exploration probability.
            epsilon_decay: Multiplicative decay per training step.
            epsilon_min: Minimum epsilon (exploration floor).
            target_update_freq: Steps between target network updates.
            device: Computation device (CPU or GPU).
        """
        self.device = device or get_torch_device()
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.step_count = 0
        
        # Policy and target networks
        self.policy_net = VoteDQN(state_dim, hidden_dims or config.DQN_HIDDEN_DIMS).to(self.device)
        self.target_net = VoteDQN(state_dim, hidden_dims or config.DQN_HIDDEN_DIMS).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net in eval mode
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss for DQN
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=10000)
    
    def select_action(
        self,
        state: torch.Tensor,
        use_exploration: bool = True
    ) -> int:
        """Select action for given state.
        
        Args:
            state: State tensor.
            use_exploration: Whether to use epsilon-greedy (True) or greedy (False).
        
        Returns:
            Action (0 or 1).
        """
        eps = self.epsilon if use_exploration else 0.0
        return self.policy_net.select_action(state, epsilon=eps, device=self.device)
    
    def remember(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ) -> None:
        """Add transition to replay buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            done: Whether episode terminated.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Perform one training step from replay buffer.
        
        Args:
            batch_size: Size of minibatch for training.
        
        Returns:
            Loss value, or None if buffer too small.
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample minibatch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute Q-values for current state
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions)
        
        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.max(1)[0].unsqueeze(1)
            
            # Bellman target: r + gamma * max Q(s', a') if not done, else r
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss and update
        loss = self.criterion(q_values, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=config.DQN_GRAD_CLIP)
        self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
