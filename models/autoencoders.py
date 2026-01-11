"""Autoencoder models for Congressional voting pattern compression.

This module provides autoencoders for dimensionality reduction of voting records
and reconstruction error metrics for validating learned latent representations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List


class Encoder(nn.Sequential):
    """Encoder network for dimensionality reduction."""
    
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        """Initialize encoder with specified input and latent dimensions.
        
        Args:
            input_dim: Dimension of input voting records (number of votes).
            latent_dim: Dimension of latent representation.
        """
        super().__init__(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )


class Decoder(nn.Sequential):
    """Decoder network for reconstruction from latent representation."""
    
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        """Initialize decoder with specified latent and output dimensions.
        
        Args:
            input_dim: Dimension of output (reconstructed votes).
            latent_dim: Dimension of latent input.
        """
        super().__init__(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Tanh()
        )


class VoteAutoencoder(nn.Module):
    """Autoencoder for voting records with reconstruction error metrics.
    
    This autoencoder compresses voting patterns into a latent space while
    tracking reconstruction error to ensure learned representations faithfully
    capture voting behavior.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 8) -> None:
        """Initialize autoencoder with encoder and decoder.
        
        Args:
            input_dim: Dimension of voting records.
            latent_dim: Dimension of latent space.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)
        
        # Track reconstruction errors for validation
        self.reconstruction_errors: List[float] = []
        self.bce_losses: List[float] = []
        self.mse_losses: List[float] = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode then decode.
        
        Args:
            x: Input voting records tensor of shape (batch_size, input_dim).
            
        Returns:
            Tuple of (reconstructed_output, latent_representation).
        """
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

    def compute_bce_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute Binary Cross-Entropy loss.
        
        Appropriate when voting data is binary {0, 1} after normalization.
        
        Args:
            original: Original voting records in range [0, 1].
            reconstructed: Reconstructed voting records (decoder output with Tanh).
            
        Returns:
            BCE loss as scalar tensor.
        """
        # Map Tanh output [-1, 1] to [0, 1] for BCE
        reconstructed_prob = (reconstructed + 1.0) / 2.0
        bce_loss = nn.BCELoss()(reconstructed_prob, original)
        return bce_loss

    def compute_mse_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute Mean Squared Error loss.
        
        Works well for continuous-valued or probabilistic voting data.
        
        Args:
            original: Original voting records.
            reconstructed: Reconstructed voting records.
            
        Returns:
            MSE loss as scalar tensor.
        """
        mse_loss = nn.MSELoss()(reconstructed, original)
        return mse_loss

    def compute_reconstruction_error(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute per-sample mean absolute reconstruction error.
        
        Useful for identifying which voting patterns are well/poorly reconstructed.
        
        Args:
            original: Original voting records of shape (batch_size, input_dim).
            reconstructed: Reconstructed voting records of same shape.
            
        Returns:
            Mean absolute error across all dimensions.
        """
        mae = torch.abs(original - reconstructed).mean().detach().cpu().item()
        return float(mae)

    def track_errors(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """Track and record reconstruction error metrics for validation.
        
        Args:
            original: Original voting records.
            reconstructed: Reconstructed voting records.
            
        Returns:
            Dictionary with 'mae', 'bce', 'mse' error metrics.
        """
        mae = self.compute_reconstruction_error(original, reconstructed)
        bce = self.compute_bce_loss(original, reconstructed).detach().cpu().item()
        mse = self.compute_mse_loss(original, reconstructed).detach().cpu().item()
        
        self.reconstruction_errors.append(mae)
        self.bce_losses.append(bce)
        self.mse_losses.append(mse)
        
        return {
            "mae": float(mae),
            "bce": float(bce),
            "mse": float(mse),
        }

    def get_error_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of tracked reconstruction errors.
        
        Returns:
            Dictionary with error statistics (mean, std, min, max) for each metric.
        """
        def compute_stats(errors: List[float]) -> Dict[str, float]:
            if not errors:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            arr = np.array(errors)
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        
        return {
            "mae": compute_stats(self.reconstruction_errors),
            "bce": compute_stats(self.bce_losses),
            "mse": compute_stats(self.mse_losses),
        }