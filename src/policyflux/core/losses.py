"""Loss functions for model training.

This module provides loss function implementations following TensorFlow/Keras conventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


class Loss(ABC):
    """Abstract base class for loss functions.
    
    A Loss computes the difference between predictions and targets.
    """
    
    def __init__(self, name: str = 'loss'):
        """Initialize the loss function.
        
        Args:
            name: Name of the loss function
        """
        self.name = name
    
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the loss value.
        
        Args:
            y_true: Ground truth targets
            y_pred: Predicted values
            
        Returns:
            Loss value (scalar)
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration.
        
        Returns:
            Dictionary containing loss configuration
        """
        return {'name': self.name}


class MSE(Loss):
    """Mean Squared Error loss.
    
    Computes the mean squared error between predictions and targets.
    
    Example:
        >>> loss_fn = MSE()
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 2.1, 2.9])
        >>> loss = loss_fn(y_true, y_pred)
    """
    
    def __init__(self):
        """Initialize MSE loss."""
        super().__init__(name='mse')
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error.
        
        Args:
            y_true: Ground truth targets
            y_pred: Predicted values
            
        Returns:
            Mean squared error (scalar)
        """
        return float(np.mean((y_true - y_pred) ** 2))


class CrossEntropy(Loss):
    """Cross-entropy loss for classification.
    
    Computes the cross-entropy loss between true labels and predictions.
    Supports both binary and multi-class classification.
    
    Args:
        from_logits: Whether predictions are logits (True) or probabilities (False)
        
    Example:
        >>> loss_fn = CrossEntropy()
        >>> y_true = np.array([0, 1, 1])
        >>> y_pred = np.array([0.1, 0.9, 0.8])
        >>> loss = loss_fn(y_true, y_pred)
    """
    
    def __init__(self, from_logits: bool = False):
        """Initialize CrossEntropy loss.
        
        Args:
            from_logits: Whether predictions are logits (True) or probabilities (False)
        """
        super().__init__(name='cross_entropy')
        self.from_logits = from_logits
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute cross-entropy loss.
        
        Args:
            y_true: Ground truth labels (integer or one-hot encoded)
            y_pred: Predicted probabilities or logits
            
        Returns:
            Cross-entropy loss (scalar)
        """
        # Clip predictions to avoid log(0)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if self.from_logits:
            # Apply softmax if predictions are logits
            y_pred = self._softmax(y_pred)
        
        # Check if binary or multi-class
        if y_pred.ndim == 1 or y_pred.shape[-1] == 1:
            # Binary classification
            loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Multi-class classification
            if y_true.ndim == 1 or y_true.shape[-1] == 1:
                # Convert integer labels to one-hot
                y_true = self._to_one_hot(y_true, y_pred.shape[-1])
            
            loss = -np.sum(y_true * np.log(y_pred), axis=-1)
        
        return float(np.mean(loss))
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Apply softmax activation.
        
        Args:
            x: Input array
            
        Returns:
            Softmax probabilities
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def _to_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert integer labels to one-hot encoding.
        
        Args:
            y: Integer labels
            num_classes: Number of classes
            
        Returns:
            One-hot encoded labels
        """
        y = y.astype(int).flatten()
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot
    
    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration.
        
        Returns:
            Dictionary containing loss configuration
        """
        config = super().get_config()
        config['from_logits'] = self.from_logits
        return config




