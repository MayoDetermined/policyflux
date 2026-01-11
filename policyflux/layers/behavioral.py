"""Behavioral layers for political modeling.

This module provides domain-specific layers for modeling congressional
behavior, network influence, voting decisions, and regime context.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from policyflux.core.layer import Layer


class ActorLayer(Layer):
    """Dense layer with ideology-aware features for actor behavior.
    
    This layer implements a dense transformation with optional ideology
    features for modeling political actor behavior.
    
    Args:
        units: Number of output units
        activation: Activation function ('tanh', 'relu', 'sigmoid', 'linear')
        use_ideology: Whether to incorporate ideology features
        ideology_dim: Dimension of ideology features
        
    Example:
        >>> layer = ActorLayer(units=64, activation='tanh', use_ideology=True)
        >>> output = layer(inputs)
    """
    
    def __init__(
        self,
        units: int,
        activation: str = 'tanh',
        use_ideology: bool = False,
        ideology_dim: int = 1,
        **kwargs
    ):
        """Initialize ActorLayer.
        
        Args:
            units: Number of output units
            activation: Activation function name
            use_ideology: Whether to use ideology features
            ideology_dim: Dimension of ideology features
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.use_ideology = use_ideology
        self.ideology_dim = ideology_dim
        
        self.kernel = None
        self.bias = None
        self.ideology_kernel = None
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build layer weights.
        
        Args:
            input_shape: Shape of input tensor
        """
        input_dim = input_shape[-1]
        
        # Main transformation weights (Xavier/Glorot initialization)
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        
        # Optional ideology weights
        if self.use_ideology:
            self.ideology_kernel = self.add_weight(
                shape=(self.ideology_dim, self.units),
                initializer='glorot_uniform',
                trainable=True,
                name='ideology_kernel'
            )
    
    def call(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor after transformation and activation
        """
        # Dense transformation
        output = inputs @ self.kernel + self.bias
        
        # Add ideology features if enabled
        if self.use_ideology and self.ideology_kernel is not None:
            # Extract ideology features (assuming they are the first ideology_dim features)
            ideology_features = inputs[:, :self.ideology_dim]
            output += ideology_features @ self.ideology_kernel
        
        # Apply activation
        output = self._apply_activation(output)
        
        return output
    
    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function.
        
        Args:
            x: Input array
            
        Returns:
            Activated array
        """
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'linear':
            return x
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.
        
        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'use_ideology': self.use_ideology,
            'ideology_dim': self.ideology_dim,
        })
        return config


class NetworkInfluenceLayer(Layer):
    """Layer for incorporating network influence on actor states.
    
    This layer implements network diffusion where actor states are influenced
    by their neighbors in a social/political network.
    
    Args:
        influence_strength: Strength of network influence (0 to 1)
        normalization: Type of adjacency normalization ('symmetric', 'row', 'none')
        
    Note:
        The adjacency matrix must match the batch size during training.
        When using this layer, set batch_size equal to the total number of actors
        in your dataset, or use batch_size=len(dataset) during fit().
        
    Example:
        >>> layer = NetworkInfluenceLayer(influence_strength=0.5)
        >>> layer.set_adjacency(adjacency_matrix)
        >>> output = layer(inputs)
    """
    
    def __init__(
        self,
        influence_strength: float = 0.5,
        normalization: str = 'symmetric',
        **kwargs
    ):
        """Initialize NetworkInfluenceLayer.
        
        Args:
            influence_strength: Influence strength (0 to 1)
            normalization: Normalization type
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.influence_strength = np.clip(influence_strength, 0.0, 1.0)
        self.normalization = normalization
        self.adjacency_matrix: Optional[np.ndarray] = None
        self.normalized_adjacency: Optional[np.ndarray] = None
    
    def set_adjacency(self, adjacency_matrix: np.ndarray) -> None:
        """Set and normalize the adjacency matrix.
        
        Args:
            adjacency_matrix: Network adjacency matrix
        """
        self.adjacency_matrix = adjacency_matrix.astype(np.float32)
        self.normalized_adjacency = self._normalize_adjacency(self.adjacency_matrix)
    
    def _normalize_adjacency(self, adj: np.ndarray) -> np.ndarray:
        """Normalize adjacency matrix.
        
        Args:
            adj: Adjacency matrix
            
        Returns:
            Normalized adjacency matrix
        """
        if self.normalization == 'none':
            return adj
        elif self.normalization == 'row':
            # Row normalization
            row_sums = adj.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            return adj / row_sums
        elif self.normalization == 'symmetric':
            # Symmetric normalization: D^(-1/2) A D^(-1/2)
            degree = adj.sum(axis=1)
            degree[degree == 0] = 1  # Avoid division by zero
            d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
            return d_inv_sqrt @ adj @ d_inv_sqrt
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build layer (no trainable weights).
        
        Args:
            input_shape: Shape of input tensor
        """
        pass
    
    def call(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass with network influence.
        
        Implements: output = (1-α) * input + α * (normalized_adjacency @ input)
        
        Args:
            inputs: Input tensor (n_actors, features)
            training: Whether in training mode
            
        Returns:
            Output tensor with network influence applied
        """
        if self.normalized_adjacency is None:
            # No adjacency matrix set, return inputs unchanged
            return inputs
        
        # Apply network influence
        neighbor_influence = self.normalized_adjacency @ inputs
        output = (1 - self.influence_strength) * inputs + self.influence_strength * neighbor_influence
        
        return output
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.
        
        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'influence_strength': self.influence_strength,
            'normalization': self.normalization,
        })
        return config


class VotingLayer(Layer):
    """Layer for converting utility scores to voting decisions.
    
    This layer converts continuous utility/preference scores into binary
    voting decisions (yes/no).
    
    Args:
        temperature: Temperature for softmax (lower = more deterministic)
        stochastic: Whether to sample stochastically or use threshold
        
    Example:
        >>> layer = VotingLayer(temperature=1.0, stochastic=False)
        >>> votes = layer(utility_scores)
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        stochastic: bool = False,
        **kwargs
    ):
        """Initialize VotingLayer.
        
        Args:
            temperature: Temperature parameter
            stochastic: Whether to use stochastic sampling
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.temperature = temperature
        self.stochastic = stochastic
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build layer (no trainable weights).
        
        Args:
            input_shape: Shape of input tensor
        """
        pass
    
    def call(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass for voting decision.
        
        Args:
            inputs: Utility scores
            training: Whether in training mode
            
        Returns:
            Binary voting decisions
        """
        if self.stochastic:
            # Stochastic: sample from sigmoid probabilities
            probabilities = self._sigmoid(inputs / self.temperature)
            if training:
                # During training, sample
                votes = (np.random.random(inputs.shape) < probabilities).astype(np.float32)
            else:
                # During inference, use probabilities
                votes = probabilities
        else:
            # Deterministic: threshold at 0
            votes = (inputs > 0).astype(np.float32)
        
        return votes
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation.
        
        Args:
            x: Input array
            
        Returns:
            Sigmoid output
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.
        
        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'stochastic': self.stochastic,
        })
        return config


class RegimeContextLayer(Layer):
    """Layer for fusing external context/regime information with actor states.
    
    This layer incorporates external context (e.g., public opinion, economic
    conditions, regime pressure) into actor states using different fusion strategies.
    
    Args:
        context_dim: Dimension of context vector
        fusion_mode: Fusion strategy ('concat', 'add', 'multiply')
        
    Example:
        >>> layer = RegimeContextLayer(context_dim=10, fusion_mode='concat')
        >>> output = layer(inputs)
    """
    
    def __init__(
        self,
        context_dim: int,
        fusion_mode: str = 'concat',
        **kwargs
    ):
        """Initialize RegimeContextLayer.
        
        Args:
            context_dim: Dimension of context vector
            fusion_mode: Fusion strategy ('concat', 'add', 'multiply')
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.context_dim = context_dim
        self.fusion_mode = fusion_mode
        self.context_projection = None
        self.gate_weight = None
    
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build layer weights.
        
        Args:
            input_shape: Shape of input tensor
        """
        input_dim = input_shape[-1]
        # Actor features dimension (excluding context)
        actor_dim = input_dim - self.context_dim
        
        if self.fusion_mode == 'add':
            # Project context to actor dimension
            self.context_projection = self.add_weight(
                shape=(self.context_dim, actor_dim),
                initializer='glorot_uniform',
                trainable=True,
                name='context_projection'
            )
        elif self.fusion_mode == 'multiply':
            # Gating mechanism
            self.gate_weight = self.add_weight(
                shape=(self.context_dim, actor_dim),
                initializer='glorot_uniform',
                trainable=True,
                name='gate_weight'
            )
    
    def call(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass with context fusion.
        
        Args:
            inputs: Input tensor (n_actors, features)
            training: Whether in training mode
            
        Returns:
            Output tensor with context fused
        """
        # For this implementation, we'll assume context is embedded in the first
        # context_dim features of the input, or passed separately
        # Here we use a simplified approach where context is extracted from input
        
        if self.fusion_mode == 'concat':
            # Context is already concatenated in input, just pass through
            return inputs
        
        # Extract context and actor states
        actor_states = inputs[:, self.context_dim:]
        context = inputs[:, :self.context_dim]
        
        if self.fusion_mode == 'add':
            # Project and add context
            if self.context_projection is not None:
                context_contribution = context @ self.context_projection
                return actor_states + context_contribution
            return actor_states
        
        elif self.fusion_mode == 'multiply':
            # Use context as gating mechanism
            if self.gate_weight is not None:
                gate = self._sigmoid(context @ self.gate_weight)
                return actor_states * gate
            return actor_states
        
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation.
        
        Args:
            x: Input array
            
        Returns:
            Sigmoid output
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.
        
        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            'context_dim': self.context_dim,
            'fusion_mode': self.fusion_mode,
        })
        return config
