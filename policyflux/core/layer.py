"""Base Layer class for building neural network components.

This module provides the fundamental Layer abstraction following TensorFlow/Keras conventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Layer(ABC):
    """Abstract base class for all layers.
    
    A Layer encapsulates weights and computation. Layers are built lazily
    on first call to allow automatic shape inference.
    
    Example:
        >>> class DenseLayer(Layer):
        ...     def __init__(self, units: int):
        ...         super().__init__()
        ...         self.units = units
        ...     
        ...     def build(self, input_shape: Tuple[int, ...]):
        ...         input_dim = input_shape[-1]
        ...         self.kernel = self.add_weight(
        ...             shape=(input_dim, self.units),
        ...             initializer='glorot_uniform',
        ...             name='kernel'
        ...         )
        ...         self.bias = self.add_weight(
        ...             shape=(self.units,),
        ...             initializer='zeros',
        ...             name='bias'
        ...         )
        ...     
        ...     def call(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        ...         return inputs @ self.kernel + self.bias
    """
    
    def __init__(self, name: Optional[str] = None, trainable: bool = True):
        """Initialize the layer.
        
        Args:
            name: Optional name for the layer
            trainable: Whether the layer's weights should be trainable
        """
        self.name = name or self.__class__.__name__
        self._trainable = trainable
        self._built = False
        self._trainable_weights: List[np.ndarray] = []
        self._non_trainable_weights: List[np.ndarray] = []
        self._weight_names: List[str] = []
        
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Create layer weights based on input shape.
        
        This method should be overridden in subclasses to create weights.
        It is called automatically on first __call__.
        
        Args:
            input_shape: Shape of the input tensor (including batch dimension)
        """
        pass
    
    @abstractmethod
    def call(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass computation.
        
        Must be implemented by subclasses.
        
        Args:
            inputs: Input tensor
            training: Whether the layer is in training mode
            
        Returns:
            Output tensor
        """
        pass
    
    def __call__(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """Call the layer on inputs, building if necessary.
        
        Args:
            inputs: Input tensor
            training: Whether the layer is in training mode
            
        Returns:
            Output tensor
        """
        if not self._built:
            input_shape = inputs.shape
            self.build(input_shape)
            self._built = True
        
        return self.call(inputs, training=training)
    
    def add_weight(
        self,
        shape: Tuple[int, ...],
        initializer: str = 'glorot_uniform',
        trainable: bool = True,
        name: str = 'weight'
    ) -> np.ndarray:
        """Add a weight tensor to the layer.
        
        Args:
            shape: Shape of the weight tensor
            initializer: Initialization strategy ('zeros', 'ones', 'glorot_uniform', 'he_normal')
            trainable: Whether this weight should be trainable
            name: Name for the weight
            
        Returns:
            The initialized weight array
        """
        weight = self._initialize_weight(shape, initializer)
        
        if trainable and self._trainable:
            self._trainable_weights.append(weight)
        else:
            self._non_trainable_weights.append(weight)
        
        self._weight_names.append(name)
        return weight
    
    def _initialize_weight(self, shape: Tuple[int, ...], initializer: str) -> np.ndarray:
        """Initialize a weight tensor.
        
        Args:
            shape: Shape of the weight tensor
            initializer: Initialization strategy
            
        Returns:
            Initialized weight array
        """
        if initializer == 'zeros':
            return np.zeros(shape, dtype=np.float32)
        elif initializer == 'ones':
            return np.ones(shape, dtype=np.float32)
        elif initializer == 'glorot_uniform':
            # Xavier/Glorot uniform initialization
            # For 2D tensors: fan_in = shape[0], fan_out = shape[-1]
            # For other tensors: use first and last dimensions
            if len(shape) < 2:
                # For 1D tensors (e.g., bias), use uniform distribution
                limit = np.sqrt(3.0)
            else:
                fan_in = shape[0]
                fan_out = shape[-1]
                limit = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, shape).astype(np.float32)
        elif initializer == 'he_normal':
            # He normal initialization
            if len(shape) < 2:
                # For 1D tensors, use standard normal
                std = 1.0
            else:
                std = np.sqrt(2.0 / shape[0])
            return np.random.normal(0, std, shape).astype(np.float32)
        else:
            raise ValueError(f"Unknown initializer: {initializer}")
    
    @property
    def weights(self) -> List[np.ndarray]:
        """Get all weights (trainable and non-trainable)."""
        return self._trainable_weights + self._non_trainable_weights
    
    @property
    def trainable_weights(self) -> List[np.ndarray]:
        """Get trainable weights only."""
        return self._trainable_weights if self._trainable else []
    
    @property
    def non_trainable_weights(self) -> List[np.ndarray]:
        """Get non-trainable weights."""
        return self._non_trainable_weights
    
    @property
    def built(self) -> bool:
        """Check if the layer has been built."""
        return self._built
    
    @property
    def trainable(self) -> bool:
        """Check if the layer is trainable."""
        return self._trainable
    
    @trainable.setter
    def trainable(self, value: bool) -> None:
        """Set whether the layer is trainable."""
        self._trainable = value
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.
        
        Returns:
            Dictionary containing the layer configuration
        """
        return {
            'name': self.name,
            'trainable': self._trainable,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Layer':
        """Create a layer from its configuration.
        
        Args:
            config: Layer configuration dictionary
            
        Returns:
            Layer instance
        """
        return cls(**config)
    
    def __repr__(self) -> str:
        """String representation of the layer."""
        return f"{self.__class__.__name__}(name='{self.name}')"
