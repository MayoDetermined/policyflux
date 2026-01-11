"""Callbacks for training lifecycle management.

This module provides callback implementations following TensorFlow/Keras conventions.
"""

from __future__ import annotations

import copy
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class Callback(ABC):
    """Abstract base class for callbacks.
    
    Callbacks provide hooks into the training lifecycle to implement
    custom behavior like early stopping, logging, or checkpointing.
    """
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training.
        
        Args:
            logs: Dictionary of logs
        """
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training.
        
        Args:
            logs: Dictionary of logs
        """
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of an epoch.
        
        Args:
            epoch: Epoch number
            logs: Dictionary of logs
        """
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of an epoch.
        
        Args:
            epoch: Epoch number
            logs: Dictionary of logs (e.g., loss, metrics)
        """
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of a training batch.
        
        Args:
            batch: Batch number
            logs: Dictionary of logs
        """
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of a training batch.
        
        Args:
            batch: Batch number
            logs: Dictionary of logs
        """
        pass


class CallbackList:
    """Container for managing multiple callbacks.
    
    Args:
        callbacks: List of callback instances
        
    Example:
        >>> callbacks = CallbackList([EarlyStopping(), ModelCheckpoint()])
        >>> callbacks.on_epoch_end(epoch=1, logs={'loss': 0.5})
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """Initialize callback list.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        """Add a callback to the list.
        
        Args:
            callback: Callback instance to add
        """
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.
    
    Args:
        monitor: Metric name to monitor (e.g., 'loss', 'val_loss')
        patience: Number of epochs with no improvement to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: One of 'min' or 'max'. In 'min' mode, training stops when
            the monitored quantity stops decreasing
        restore_best_weights: Whether to restore model weights from the epoch
            with the best value of the monitored metric
        
    Example:
        >>> early_stop = EarlyStopping(monitor='val_loss', patience=5)
    """
    
    def __init__(
        self,
        monitor: str = 'loss',
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = False
    ):
        """Initialize EarlyStopping callback.
        
        Args:
            monitor: Metric name to monitor
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            restore_best_weights: Whether to restore best weights
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.model = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        else:
            self.monitor_op = np.greater
            self.best = -np.inf
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Initialize early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        if self.mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check if training should stop.
        
        Args:
            epoch: Current epoch number
            logs: Dictionary containing metrics
        """
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        # Check if current value is better than best
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights and self.model is not None:
                self.best_weights = self._get_model_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_weights is not None:
                    self._restore_model_weights()
                # Signal to stop training
                logs['stop_training'] = True
    
    def _get_model_weights(self) -> List[np.ndarray]:
        """Get a copy of model weights."""
        if self.model is None:
            return []
        weights = []
        for layer in self.model.layers:
            weights.extend([w.copy() for w in layer.trainable_weights])
        return weights
    
    def _restore_model_weights(self) -> None:
        """Restore model weights from best epoch."""
        if self.model is None or self.best_weights is None:
            return
        idx = 0
        for layer in self.model.layers:
            for weight in layer.trainable_weights:
                weight[:] = self.best_weights[idx]
                idx += 1


class ModelCheckpoint(Callback):
    """Save model weights during training.
    
    Args:
        filepath: Path to save model weights
        monitor: Metric name to monitor
        save_best_only: Only save when monitored metric improves
        mode: One of 'min' or 'max'
        
    Example:
        >>> checkpoint = ModelCheckpoint('model_weights.npy', monitor='val_loss')
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'loss',
        save_best_only: bool = False,
        mode: str = 'min'
    ):
        """Initialize ModelCheckpoint callback.
        
        Args:
            filepath: Path to save model weights
            monitor: Metric name to monitor
            save_best_only: Only save when metric improves
            mode: 'min' or 'max'
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.model = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        else:
            self.monitor_op = np.greater
            self.best = -np.inf
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Initialize checkpoint state."""
        if self.mode == 'min':
            self.best = np.inf
        else:
            self.best = -np.inf
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save model if conditions are met.
        
        Args:
            epoch: Current epoch number
            logs: Dictionary containing metrics
        """
        if logs is None or self.model is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.best = current
                self._save_weights()
        else:
            self._save_weights()
    
    def _save_weights(self) -> None:
        """Save model weights to file."""
        if self.model is None:
            return
        
        weights_dict = {}
        for i, layer in enumerate(self.model.layers):
            for j, weight in enumerate(layer.weights):
                weights_dict[f'layer_{i}_weight_{j}'] = weight
        
        # Create directory if needed
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        np.savez(self.filepath, **weights_dict)


class TensorBoard(Callback):
    """TensorBoard logging callback (stub for future implementation).
    
    Args:
        log_dir: Directory to save TensorBoard logs
        
    Note:
        This is a stub implementation for future integration with
        TensorBoard or other logging frameworks.
    """
    
    def __init__(self, log_dir: str = './logs'):
        """Initialize TensorBoard callback.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TensorBoard writer."""
        # Stub: would initialize TensorBoard writer here
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log metrics to TensorBoard."""
        # Stub: would log metrics here
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Close TensorBoard writer."""
        # Stub: would close writer here
        pass
