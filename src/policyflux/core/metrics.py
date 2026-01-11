"""Metrics for model evaluation.

This module provides metric implementations following TensorFlow/Keras conventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class Metric(ABC):
    """Abstract base class for metrics.
    
    Metrics track model performance during training and evaluation.
    """
    
    def __init__(self, name: str = 'metric'):
        """Initialize the metric.
        
        Args:
            name: Name of the metric
        """
        self.name = name
        self._state = None
        self._count = 0
    
    @abstractmethod
    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Update metric state with new batch of data.
        
        Args:
            y_true: Ground truth targets
            y_pred: Predicted values
        """
        pass
    
    @abstractmethod
    def result(self) -> float:
        """Compute the current metric value.
        
        Returns:
            Metric value (scalar)
        """
        pass
    
    def reset_state(self) -> None:
        """Reset metric state for new epoch."""
        self._state = None
        self._count = 0
    
    def get_config(self) -> Dict[str, Any]:
        """Get metric configuration.
        
        Returns:
            Dictionary containing metric configuration
        """
        return {'name': self.name}


class Accuracy(Metric):
    """Accuracy metric for classification tasks.
    
    Computes the fraction of predictions that match the true labels.
    
    Example:
        >>> metric = Accuracy()
        >>> metric.update_state(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.8]))
        >>> accuracy = metric.result()
    """
    
    def __init__(self):
        """Initialize accuracy metric."""
        super().__init__(name='accuracy')
        self._correct = 0
        self._total = 0
    
    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Update accuracy with new predictions.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities or class indices
        """
        # Convert predictions to class indices if probabilities
        if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
            y_pred = np.argmax(y_pred, axis=-1)
        else:
            # Binary classification threshold
            y_pred = (y_pred > 0.5).astype(int).flatten()
        
        y_true = y_true.astype(int).flatten()
        y_pred = y_pred.astype(int).flatten()
        
        self._correct += np.sum(y_true == y_pred)
        self._total += len(y_true)
    
    def result(self) -> float:
        """Compute current accuracy.
        
        Returns:
            Accuracy value between 0 and 1
        """
        if self._total == 0:
            return 0.0
        return float(self._correct / self._total)
    
    def reset_state(self) -> None:
        """Reset accuracy state."""
        super().reset_state()
        self._correct = 0
        self._total = 0


class MeanSquaredError(Metric):
    """Mean Squared Error metric.
    
    Computes the average squared difference between predictions and targets.
    
    Example:
        >>> metric = MeanSquaredError()
        >>> metric.update_state(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
        >>> mse = metric.result()
    """
    
    def __init__(self):
        """Initialize MSE metric."""
        super().__init__(name='mse')
        self._sum_squared_error = 0.0
        self._count = 0
    
    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Update MSE with new predictions.
        
        Args:
            y_true: Ground truth targets
            y_pred: Predicted values
        """
        squared_error = np.sum((y_true - y_pred) ** 2)
        self._sum_squared_error += squared_error
        self._count += y_true.size
    
    def result(self) -> float:
        """Compute current MSE.
        
        Returns:
            Mean squared error value
        """
        if self._count == 0:
            return 0.0
        return float(self._sum_squared_error / self._count)
    
    def reset_state(self) -> None:
        """Reset MSE state."""
        super().reset_state()
        self._sum_squared_error = 0.0
        self._count = 0


class Precision(Metric):
    """Precision metric for binary classification.
    
    Computes the fraction of positive predictions that are correct.
    Precision = TP / (TP + FP)
    
    Example:
        >>> metric = Precision()
        >>> metric.update_state(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.3]))
        >>> precision = metric.result()
    """
    
    def __init__(self, threshold: float = 0.5):
        """Initialize precision metric.
        
        Args:
            threshold: Threshold for binary classification
        """
        super().__init__(name='precision')
        self.threshold = threshold
        self._true_positives = 0
        self._false_positives = 0
    
    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Update precision with new predictions.
        
        Args:
            y_true: Ground truth binary labels
            y_pred: Predicted probabilities
        """
        # Convert to binary predictions
        y_pred_binary = (y_pred > self.threshold).astype(int).flatten()
        y_true = y_true.astype(int).flatten()
        
        self._true_positives += np.sum((y_true == 1) & (y_pred_binary == 1))
        self._false_positives += np.sum((y_true == 0) & (y_pred_binary == 1))
    
    def result(self) -> float:
        """Compute current precision.
        
        Returns:
            Precision value between 0 and 1
        """
        denominator = self._true_positives + self._false_positives
        if denominator == 0:
            return 0.0
        return float(self._true_positives / denominator)
    
    def reset_state(self) -> None:
        """Reset precision state."""
        super().reset_state()
        self._true_positives = 0
        self._false_positives = 0
    
    def get_config(self) -> Dict[str, Any]:
        """Get metric configuration.
        
        Returns:
            Dictionary containing metric configuration
        """
        config = super().get_config()
        config['threshold'] = self.threshold
        return config


class Recall(Metric):
    """Recall metric for binary classification.
    
    Computes the fraction of actual positives that are correctly identified.
    Recall = TP / (TP + FN)
    
    Example:
        >>> metric = Recall()
        >>> metric.update_state(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.3]))
        >>> recall = metric.result()
    """
    
    def __init__(self, threshold: float = 0.5):
        """Initialize recall metric.
        
        Args:
            threshold: Threshold for binary classification
        """
        super().__init__(name='recall')
        self.threshold = threshold
        self._true_positives = 0
        self._false_negatives = 0
    
    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Update recall with new predictions.
        
        Args:
            y_true: Ground truth binary labels
            y_pred: Predicted probabilities
        """
        # Convert to binary predictions
        y_pred_binary = (y_pred > self.threshold).astype(int).flatten()
        y_true = y_true.astype(int).flatten()
        
        self._true_positives += np.sum((y_true == 1) & (y_pred_binary == 1))
        self._false_negatives += np.sum((y_true == 1) & (y_pred_binary == 0))
    
    def result(self) -> float:
        """Compute current recall.
        
        Returns:
            Recall value between 0 and 1
        """
        denominator = self._true_positives + self._false_negatives
        if denominator == 0:
            return 0.0
        return float(self._true_positives / denominator)
    
    def reset_state(self) -> None:
        """Reset recall state."""
        super().reset_state()
        self._true_positives = 0
        self._false_negatives = 0
    
    def get_config(self) -> Dict[str, Any]:
        """Get metric configuration.
        
        Returns:
            Dictionary containing metric configuration
        """
        config = super().get_config()
        config['threshold'] = self.threshold
        return config




