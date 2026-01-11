"""Tests for core Layer class functionality."""

import numpy as np
import pytest

from policyflux.core.layer import Layer


class SimpleDenseLayer(Layer):
    """Simple dense layer for testing."""
    
    def __init__(self, units, activation='linear', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.kernel = None
        self.bias = None
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
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
    
    def call(self, inputs, training=False):
        output = inputs @ self.kernel + self.bias
        if self.activation == 'relu':
            output = np.maximum(0, output)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
        })
        return config


def test_layer_initialization():
    """Test layer initialization."""
    layer = SimpleDenseLayer(units=10, name='test_layer')
    
    assert layer.name == 'test_layer'
    assert layer.trainable is True
    assert layer.built is False
    assert len(layer.weights) == 0


def test_layer_lazy_building():
    """Test that layers build lazily on first call."""
    layer = SimpleDenseLayer(units=5)
    
    # Layer should not be built initially
    assert not layer.built
    
    # Call the layer
    inputs = np.random.randn(2, 3).astype(np.float32)
    output = layer(inputs)
    
    # Layer should now be built
    assert layer.built
    assert output.shape == (2, 5)


def test_layer_weight_initialization():
    """Test different weight initialization strategies."""
    layer = SimpleDenseLayer(units=10)
    
    # Build layer
    inputs = np.random.randn(2, 5).astype(np.float32)
    _ = layer(inputs)
    
    # Check that weights exist
    assert len(layer.weights) == 2
    assert len(layer.trainable_weights) == 2
    assert len(layer.non_trainable_weights) == 0
    
    # Check weight shapes
    assert layer.kernel.shape == (5, 10)
    assert layer.bias.shape == (10,)
    
    # Check bias is initialized to zeros
    assert np.allclose(layer.bias, 0)


def test_layer_weight_tracking():
    """Test that layer correctly tracks trainable and non-trainable weights."""
    layer = SimpleDenseLayer(units=5)
    
    # Manually add non-trainable weight
    layer.build((None, 3))
    non_trainable = layer.add_weight(
        shape=(3,),
        initializer='ones',
        trainable=False,
        name='non_trainable'
    )
    
    # Check tracking
    assert len(layer.trainable_weights) == 2  # kernel and bias
    assert len(layer.non_trainable_weights) == 1  # non_trainable
    assert len(layer.weights) == 3


def test_layer_trainable_property():
    """Test layer trainable property."""
    layer = SimpleDenseLayer(units=5, trainable=True)
    
    # Build layer
    inputs = np.random.randn(2, 3).astype(np.float32)
    _ = layer(inputs)
    
    # Should have trainable weights
    assert len(layer.trainable_weights) == 2
    
    # Set trainable to False
    layer.trainable = False
    assert layer.trainable is False
    # Note: In this simplified implementation, already-added weights
    # don't change their trainable status retroactively


def test_layer_forward_pass():
    """Test forward pass computation."""
    np.random.seed(42)
    layer = SimpleDenseLayer(units=4, activation='relu')
    
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    output = layer(inputs)
    
    # Check output shape
    assert output.shape == (2, 4)
    
    # Check that ReLU was applied (no negative values)
    assert np.all(output >= 0)


def test_layer_multiple_calls():
    """Test that layer can be called multiple times without rebuilding."""
    layer = SimpleDenseLayer(units=5)
    
    # First call builds the layer
    inputs1 = np.random.randn(2, 3).astype(np.float32)
    output1 = layer(inputs1)
    
    # Store weights
    kernel_copy = layer.kernel.copy()
    
    # Second call should not rebuild
    inputs2 = np.random.randn(2, 3).astype(np.float32)
    output2 = layer(inputs2)
    
    # Weights should be the same
    assert np.allclose(layer.kernel, kernel_copy)
    
    # Outputs should be different
    assert not np.allclose(output1, output2)


def test_layer_get_config():
    """Test layer configuration serialization."""
    layer = SimpleDenseLayer(units=10, activation='relu', name='test')
    
    config = layer.get_config()
    
    assert config['name'] == 'test'
    assert config['trainable'] is True
    assert config['units'] == 10
    assert config['activation'] == 'relu'


def test_layer_weight_initializers():
    """Test different weight initializers."""
    layer = SimpleDenseLayer(units=5)
    
    # Test zeros initializer
    zeros_weight = layer._initialize_weight((3, 5), 'zeros')
    assert np.allclose(zeros_weight, 0)
    
    # Test ones initializer
    ones_weight = layer._initialize_weight((3, 5), 'ones')
    assert np.allclose(ones_weight, 1)
    
    # Test glorot uniform initializer
    glorot_weight = layer._initialize_weight((3, 5), 'glorot_uniform')
    assert glorot_weight.shape == (3, 5)
    limit = np.sqrt(6.0 / (3 + 5))
    assert np.all(np.abs(glorot_weight) <= limit)
    
    # Test he normal initializer
    he_weight = layer._initialize_weight((3, 5), 'he_normal')
    assert he_weight.shape == (3, 5)


def test_layer_invalid_initializer():
    """Test that invalid initializer raises error."""
    layer = SimpleDenseLayer(units=5)
    
    with pytest.raises(ValueError, match="Unknown initializer"):
        layer._initialize_weight((3, 5), 'invalid_initializer')


def test_layer_repr():
    """Test layer string representation."""
    layer = SimpleDenseLayer(units=10, name='my_layer')
    
    repr_str = repr(layer)
    assert 'SimpleDenseLayer' in repr_str
    assert 'my_layer' in repr_str
