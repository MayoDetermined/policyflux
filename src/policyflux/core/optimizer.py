"""Optimizer classes for training neural networks.

This module provides optimizer implementations following TensorFlow/Keras conventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class Optimizer(ABC):
    """Abstract base class for optimizers.
    
    Optimizers update model weights based on computed gradients.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """Initialize the optimizer.
        
        Args:
            learning_rate: Learning rate for weight updates
        """
        self.learning_rate = learning_rate
        self._iterations = 0
    
    @abstractmethod
    def apply_gradients(self, gradients_and_weights: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Apply gradients to weights.
        
        Args:
            gradients_and_weights: List of (gradient, weight) tuples
        """
        pass
    
    def get_config(self) -> Dict[str, float]:
        """Get optimizer configuration.
        
        Returns:
            Dictionary containing optimizer configuration
        """
        return {'learning_rate': self.learning_rate}


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.
    
    Args:
        learning_rate: Learning rate
        momentum: Momentum factor (0 to 1)
        
    Example:
        >>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
        >>> optimizer.apply_gradients([(grad1, weight1), (grad2, weight2)])
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate for weight updates
            momentum: Momentum factor (0 to 1)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self._velocities: Dict[int, np.ndarray] = {}
    
    def apply_gradients(self, gradients_and_weights: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Apply SGD updates to weights.
        
        Args:
            gradients_and_weights: List of (gradient, weight) tuples
        """
        for i, (grad, weight) in enumerate(gradients_and_weights):
            if self.momentum > 0:
                # Initialize velocity if needed
                if i not in self._velocities:
                    self._velocities[i] = np.zeros_like(grad)
                
                # Update velocity and apply momentum
                velocity = self._velocities[i]
                velocity[:] = self.momentum * velocity - self.learning_rate * grad
                weight += velocity
            else:
                # Standard gradient descent
                weight -= self.learning_rate * grad
        
        self._iterations += 1
    
    def get_config(self) -> Dict[str, float]:
        """Get optimizer configuration.
        
        Returns:
            Dictionary containing optimizer configuration
        """
        config = super().get_config()
        config['momentum'] = self.momentum
        return config


class Adam(Optimizer):
    """Adam optimizer.
    
    Adam (Adaptive Moment Estimation) combines momentum with adaptive learning rates.
    
    Args:
        learning_rate: Learning rate
        beta1: Exponential decay rate for first moment estimates
        beta2: Exponential decay rate for second moment estimates
        epsilon: Small constant for numerical stability
        
    Example:
        >>> optimizer = Adam(learning_rate=0.001)
        >>> optimizer.apply_gradients([(grad1, weight1), (grad2, weight2)])
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7
    ):
        """Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate for weight updates
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m: Dict[int, np.ndarray] = {}  # First moment estimate
        self._v: Dict[int, np.ndarray] = {}  # Second moment estimate
    
    def apply_gradients(self, gradients_and_weights: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Apply Adam updates to weights.
        
        Args:
            gradients_and_weights: List of (gradient, weight) tuples
        """
        self._iterations += 1
        
        for i, (grad, weight) in enumerate(gradients_and_weights):
            # Initialize moment estimates if needed
            if i not in self._m:
                self._m[i] = np.zeros_like(grad)
                self._v[i] = np.zeros_like(grad)
            
            # Update biased first moment estimate
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self._m[i] / (1 - self.beta1 ** self._iterations)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self._v[i] / (1 - self.beta2 ** self._iterations)
            
            # Update weights
            weight -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def get_config(self) -> Dict[str, float]:
        """Get optimizer configuration.
        
        Returns:
            Dictionary containing optimizer configuration
        """
        config = super().get_config()
        config.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
        })
        return config




