"""Model classes for building and training neural networks.

This module provides Model and Sequential implementations following TensorFlow/Keras conventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from policyflux.core.callbacks import Callback, CallbackList
from policyflux.core.layer import Layer
from policyflux.core.losses import Loss
from policyflux.core.metrics import Metric
from policyflux.core.optimizer import Optimizer


class Model(ABC):
    """Abstract base class for models.
    
    A Model combines layers, loss, optimizer, and metrics for training
    and evaluation following the compile/fit/predict pattern.
    """
    
    def __init__(self):
        """Initialize the model."""
        self.layers: List[Layer] = []
        self.optimizer: Optional[Optimizer] = None
        self.loss_fn: Optional[Loss] = None
        self.metrics_list: List[Metric] = []
        self.history: Dict[str, List[float]] = {}
        self._compiled = False
    
    @abstractmethod
    def call(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass computation.
        
        Must be implemented by subclasses.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        pass
    
    def __call__(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """Call the model on inputs.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        return self.call(inputs, training=training)
    
    def compile(
        self,
        optimizer: Optimizer,
        loss: Loss,
        metrics: Optional[List[Metric]] = None
    ) -> None:
        """Configure the model for training.
        
        Args:
            optimizer: Optimizer instance
            loss: Loss function instance
            metrics: List of metric instances
        """
        self.optimizer = optimizer
        self.loss_fn = loss
        self.metrics_list = metrics or []
        self._compiled = True
    
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        callbacks: Optional[List[Callback]] = None,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            x: Training input data
            y: Training target data
            epochs: Number of epochs to train
            batch_size: Batch size for training
            validation_data: Optional (x_val, y_val) tuple
            callbacks: List of callbacks
            verbose: Verbosity mode (0=silent, 1=progress)
            
        Returns:
            Dictionary of training history
        """
        if not self._compiled:
            raise RuntimeError("Model must be compiled before training")
        
        # Initialize history
        self.history = {
            'loss': [],
        }
        for metric in self.metrics_list:
            self.history[metric.name] = []
        
        if validation_data is not None:
            self.history['val_loss'] = []
            for metric in self.metrics_list:
                self.history[f'val_{metric.name}'] = []
        
        # Setup callbacks
        callback_list = CallbackList(callbacks)
        for callback in callback_list.callbacks:
            if hasattr(callback, 'model'):
                callback.model = self
        
        callback_list.on_train_begin()
        
        # Training loop
        n_samples = len(x)
        stop_training = False
        
        for epoch in range(epochs):
            if stop_training:
                break
            
            callback_list.on_epoch_begin(epoch)
            
            # Reset metrics
            for metric in self.metrics_list:
                metric.reset_state()
            
            # Train epoch
            epoch_logs = self._train_epoch(x, y, batch_size, callback_list, verbose)
            
            # Validation
            if validation_data is not None:
                x_val, y_val = validation_data
                val_logs = self._evaluate(x_val, y_val, batch_size)
                epoch_logs.update({f'val_{k}': v for k, v in val_logs.items()})
            
            # Update history
            for key, value in epoch_logs.items():
                if key in self.history:
                    self.history[key].append(value)
            
            # Print progress
            if verbose:
                log_str = f"Epoch {epoch + 1}/{epochs}"
                for key, value in epoch_logs.items():
                    log_str += f" - {key}: {value:.4f}"
                print(log_str)
            
            # Check for early stopping
            callback_list.on_epoch_end(epoch, epoch_logs)
            if epoch_logs.get('stop_training', False):
                stop_training = True
        
        callback_list.on_train_end()
        
        return self.history
    
    def _train_epoch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        callback_list: CallbackList,
        verbose: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            x: Training input data
            y: Training target data
            batch_size: Batch size
            callback_list: Callback list
            verbose: Verbosity mode
            
        Returns:
            Dictionary of epoch metrics
        """
        n_samples = len(x)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]
            
            callback_list.on_batch_begin(n_batches)
            
            # Forward pass
            y_pred = self(x_batch, training=True)
            
            # Compute loss
            batch_loss = self.loss_fn(y_batch, y_pred)
            epoch_loss += batch_loss
            n_batches += 1
            
            # Compute gradients and update weights (simplified - no true autodiff)
            # In a real implementation, this would use automatic differentiation
            self._update_weights(x_batch, y_batch, y_pred)
            
            # Update metrics
            for metric in self.metrics_list:
                metric.update_state(y_batch, y_pred)
            
            callback_list.on_batch_end(n_batches, {'loss': batch_loss})
        
        # Compute epoch metrics
        logs = {'loss': epoch_loss / n_batches}
        for metric in self.metrics_list:
            logs[metric.name] = metric.result()
        
        return logs
    
    def _update_weights(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """Update model weights (simplified gradient descent).
        
        Note: This is a simplified implementation using numerical differentiation.
        For production use, this should be replaced with automatic differentiation
        (e.g., using JAX or PyTorch). The current O(n²) complexity makes it slow
        for models with many parameters.
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
            y_pred: Predictions
        """
        # Simplified gradient computation
        # TODO: Replace with automatic differentiation for production use
        gradients_and_weights = []
        
        for layer in self.layers:
            for weight in layer.trainable_weights:
                # Use finite differences for gradient approximation
                # NOTE: This is O(n²) and very slow - use for prototyping only
                grad = np.zeros_like(weight)
                epsilon = 1e-7
                
                # Central difference approximation
                for idx in np.ndindex(weight.shape):
                    weight[idx] += epsilon
                    loss_plus = self.loss_fn(y_batch, self(x_batch, training=False))
                    weight[idx] -= 2 * epsilon
                    loss_minus = self.loss_fn(y_batch, self(x_batch, training=False))
                    weight[idx] += epsilon
                    
                    grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                
                gradients_and_weights.append((grad, weight))
        
        if self.optimizer and gradients_and_weights:
            self.optimizer.apply_gradients(gradients_and_weights)
    
    def _evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int
    ) -> Dict[str, float]:
        """Evaluate the model on validation data.
        
        Args:
            x: Validation input data
            y: Validation target data
            batch_size: Batch size
            
        Returns:
            Dictionary of validation metrics
        """
        # Reset metrics
        for metric in self.metrics_list:
            metric.reset_state()
        
        n_samples = len(x)
        total_loss = 0.0
        n_batches = 0
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            x_batch = x[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            y_pred = self(x_batch, training=False)
            batch_loss = self.loss_fn(y_batch, y_pred)
            total_loss += batch_loss
            n_batches += 1
            
            for metric in self.metrics_list:
                metric.update_state(y_batch, y_pred)
        
        logs = {'loss': total_loss / n_batches}
        for metric in self.metrics_list:
            logs[metric.name] = metric.result()
        
        return logs
    
    def predict(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Generate predictions for input data.
        
        Args:
            x: Input data
            batch_size: Batch size for prediction
            
        Returns:
            Predictions array
        """
        n_samples = len(x)
        predictions = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            x_batch = x[start_idx:end_idx]
            y_pred = self(x_batch, training=False)
            predictions.append(y_pred)
        
        return np.concatenate(predictions, axis=0)
    
    def summary(self) -> None:
        """Print model architecture summary."""
        print("Model Summary")
        print("=" * 60)
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_params = sum(w.size for w in layer.weights)
            total_params += layer_params
            print(f"Layer {i}: {layer.name}")
            print(f"  Trainable: {layer.trainable}")
            print(f"  Parameters: {layer_params}")
        
        print("=" * 60)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {sum(w.size for layer in self.layers for w in layer.trainable_weights)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'layers': [layer.get_config() for layer in self.layers],
        }


class Sequential(Model):
    """Sequential model for linear stack of layers.
    
    A Sequential model is appropriate for a plain stack of layers
    where each layer has exactly one input tensor and one output tensor.
    
    Example:
        >>> model = Sequential()
        >>> model.add(ActorLayer(units=64))
        >>> model.add(NetworkInfluenceLayer(influence_strength=0.5))
        >>> model.compile(optimizer=Adam(), loss=MSE())
        >>> model.fit(x_train, y_train, epochs=10)
    """
    
    def __init__(self, layers: Optional[List[Layer]] = None):
        """Initialize sequential model.
        
        Args:
            layers: Optional list of layers to add to the model
        """
        super().__init__()
        if layers:
            for layer in layers:
                self.add(layer)
    
    def add(self, layer: Layer) -> None:
        """Add a layer to the model.
        
        Args:
            layer: Layer instance to add
        """
        self.layers.append(layer)
    
    def call(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through all layers.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        x = inputs
        for layer in self.layers:
            x = layer(x, training=training)
        return x
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        config = super().get_config()
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Sequential':
        """Create a model from its configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Sequential model instance
        """
        # Note: This is a simplified implementation
        # A full implementation would reconstruct layers from their configs
        return cls()
