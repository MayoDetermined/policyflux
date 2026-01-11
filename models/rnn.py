"""RNN/LSTM model for congressional actor ideology memory.

This module provides individual LSTM networks for each congressional actor,
enabling them to learn and predict their own ideological evolution based on
historical ideology vectors and external pressure dynamics.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from gpu_utils import get_torch_device


class ActorLSTM(nn.Module):
    """LSTM-based ideology memory model for a single congressional actor.
    
    Each actor has an individual LSTM network that learns to predict their
    ideological position at time t+1 based on:
    - Historical ideology vectors: x_i(t-n), ..., x_i(t)
    - Current pressure: Z(t)
    
    The model compresses this temporal and contextual information through
    hidden states and produces a continuous ideology vector constrained to [-1, 1].
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        """Initialize ActorLSTM.
        
        Args:
            input_dim: Dimensionality of ideology vector (e.g., 3 for 3D politics).
            hidden_dim: Size of LSTM hidden state.
            output_dim: Dimensionality of output ideology vector.
            num_layers: Number of stacked LSTM layers (default: 1).
            dropout: Dropout probability between LSTM layers (default: 0.0).
        """
        super(ActorLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # LSTM input includes ideology (input_dim) + pressure (1)
        self.lstm = nn.LSTM(
            input_size=input_dim + 1,  # ideology + pressure
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output linear layer: hidden_dim -> output_dim (ideology)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim + 1)
               where input_dim + 1 accounts for ideology + pressure
        
        Returns:
            Output tensor of shape (batch_size, output_dim) representing
            predicted ideology, constrained to [-1, 1] via tanh activation.
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden_dim)
        
        # Use only the last output from the sequence
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Linear transformation to ideology space
        out = self.linear(last_out)  # (batch_size, output_dim)
        
        # Constrain to [-1, 1] using tanh activation
        return torch.tanh(out)
    
    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states for LSTM.
        
        Args:
            batch_size: Batch size for initialization.
            device: Device to place tensors on (CPU or GPU).
        
        Returns:
            Tuple of (hidden_state, cell_state) both of shape
            (num_layers, batch_size, hidden_dim).
        """
        if device is None:
            device = get_torch_device()
        
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        
        return h, c


class ActorLSTMTrainer:
    """Trainer class for ActorLSTM with online learning support.
    
    Handles forward passes, loss computation, backpropagation, and
    single-step (online) training updates.
    """
    
    def __init__(
        self, 
        model: ActorLSTM,
        learning_rate: float = 1e-4,
        device: Optional[torch.device] = None
    ):
        """Initialize trainer.
        
        Args:
            model: ActorLSTM model instance.
            learning_rate: Learning rate for optimizer.
            device: Device for computation (CPU or GPU).
        """
        self.model = model
        self.device = device or get_torch_device()
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_step(
        self,
        history_data: torch.Tensor,
        target_ideology: torch.Tensor
    ) -> float:
        """Perform one training step (online learning).
        
        Args:
            history_data: Tensor of shape (1, seq_len, input_dim + 1) containing
                         historical ideology + pressure data.
            target_ideology: Tensor of shape (1, output_dim) with target ideology.
        
        Returns:
            Loss value (scalar).
        """
        # Move tensors to device
        history_data = history_data.to(self.device)
        target_ideology = target_ideology.to(self.device)
        
        # Forward pass
        predicted = self.model(history_data)
        
        # Compute loss
        loss = self.criterion(predicted, target_ideology)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, history_data: torch.Tensor) -> torch.Tensor:
        """Make a prediction without training.
        
        Args:
            history_data: Tensor of shape (1, seq_len, input_dim + 1).
        
        Returns:
            Predicted ideology tensor of shape (1, output_dim).
        """
        history_data = history_data.to(self.device)
        with torch.no_grad():
            output = self.model(history_data)
        return output
