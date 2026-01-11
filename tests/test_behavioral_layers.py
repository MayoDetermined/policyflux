"""Tests for behavioral layers."""

import numpy as np
import pytest

from policyflux.layers.behavioral import (
    ActorLayer,
    NetworkInfluenceLayer,
    RegimeContextLayer,
    VotingLayer,
)


def test_actor_layer_basic():
    """Test basic ActorLayer functionality."""
    layer = ActorLayer(units=10, activation='tanh')
    
    inputs = np.random.randn(5, 8).astype(np.float32)
    output = layer(inputs)
    
    assert output.shape == (5, 10)
    assert layer.built
    assert len(layer.trainable_weights) == 2  # kernel and bias


def test_actor_layer_activations():
    """Test different activation functions."""
    inputs = np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)
    
    # ReLU
    layer_relu = ActorLayer(units=3, activation='relu')
    output_relu = layer_relu(inputs)
    assert np.all(output_relu >= 0)
    
    # Tanh
    layer_tanh = ActorLayer(units=3, activation='tanh')
    output_tanh = layer_tanh(inputs)
    assert np.all(np.abs(output_tanh) <= 1)
    
    # Sigmoid
    layer_sigmoid = ActorLayer(units=3, activation='sigmoid')
    output_sigmoid = layer_sigmoid(inputs)
    assert np.all((output_sigmoid >= 0) & (output_sigmoid <= 1))
    
    # Linear
    layer_linear = ActorLayer(units=3, activation='linear')
    output_linear = layer_linear(inputs)
    # Linear should pass through without constraints


def test_actor_layer_with_ideology():
    """Test ActorLayer with ideology features."""
    layer = ActorLayer(units=8, activation='tanh', use_ideology=True, ideology_dim=2)
    
    # Input with ideology features in first 2 dimensions
    inputs = np.random.randn(4, 10).astype(np.float32)
    output = layer(inputs)
    
    assert output.shape == (4, 8)
    assert len(layer.trainable_weights) == 3  # kernel, bias, ideology_kernel


def test_actor_layer_config():
    """Test ActorLayer configuration."""
    layer = ActorLayer(
        units=16,
        activation='relu',
        use_ideology=True,
        ideology_dim=3,
        name='actor'
    )
    
    config = layer.get_config()
    assert config['units'] == 16
    assert config['activation'] == 'relu'
    assert config['use_ideology'] is True
    assert config['ideology_dim'] == 3
    assert config['name'] == 'actor'


def test_network_influence_layer_basic():
    """Test basic NetworkInfluenceLayer functionality."""
    n_actors = 5
    layer = NetworkInfluenceLayer(influence_strength=0.5, normalization='none')
    
    # Create simple adjacency matrix
    adjacency = np.ones((n_actors, n_actors), dtype=np.float32)
    np.fill_diagonal(adjacency, 0)
    layer.set_adjacency(adjacency)
    
    inputs = np.random.randn(n_actors, 8).astype(np.float32)
    output = layer(inputs)
    
    assert output.shape == inputs.shape


def test_network_influence_layer_influence_strength():
    """Test influence strength parameter."""
    n_actors = 3
    inputs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    
    # Create adjacency where actor 0 and 1 are neighbors
    adjacency = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    # Zero influence
    layer_zero = NetworkInfluenceLayer(influence_strength=0.0, normalization='row')
    layer_zero.set_adjacency(adjacency)
    output_zero = layer_zero(inputs)
    assert np.allclose(output_zero, inputs)
    
    # Full influence
    layer_full = NetworkInfluenceLayer(influence_strength=1.0, normalization='row')
    layer_full.set_adjacency(adjacency)
    output_full = layer_full(inputs)
    # Actor 0 should be influenced by actor 1
    assert not np.allclose(output_full[0], inputs[0])


def test_network_influence_layer_normalizations():
    """Test different normalization strategies."""
    n_actors = 4
    adjacency = np.random.rand(n_actors, n_actors).astype(np.float32)
    adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
    np.fill_diagonal(adjacency, 0)
    
    inputs = np.random.randn(n_actors, 6).astype(np.float32)
    
    # Test 'none' normalization
    layer_none = NetworkInfluenceLayer(influence_strength=0.5, normalization='none')
    layer_none.set_adjacency(adjacency)
    output_none = layer_none(inputs)
    assert output_none.shape == inputs.shape
    
    # Test 'row' normalization
    layer_row = NetworkInfluenceLayer(influence_strength=0.5, normalization='row')
    layer_row.set_adjacency(adjacency)
    output_row = layer_row(inputs)
    assert output_row.shape == inputs.shape
    
    # Test 'symmetric' normalization
    layer_sym = NetworkInfluenceLayer(influence_strength=0.5, normalization='symmetric')
    layer_sym.set_adjacency(adjacency)
    output_sym = layer_sym(inputs)
    assert output_sym.shape == inputs.shape


def test_network_influence_layer_no_adjacency():
    """Test NetworkInfluenceLayer without setting adjacency."""
    layer = NetworkInfluenceLayer(influence_strength=0.5)
    
    inputs = np.random.randn(5, 8).astype(np.float32)
    output = layer(inputs)
    
    # Without adjacency, should pass through unchanged
    assert np.allclose(output, inputs)


def test_network_influence_layer_config():
    """Test NetworkInfluenceLayer configuration."""
    layer = NetworkInfluenceLayer(
        influence_strength=0.7,
        normalization='symmetric',
        name='network'
    )
    
    config = layer.get_config()
    assert config['influence_strength'] == 0.7
    assert config['normalization'] == 'symmetric'
    assert config['name'] == 'network'


def test_voting_layer_deterministic():
    """Test VotingLayer in deterministic mode."""
    layer = VotingLayer(temperature=1.0, stochastic=False)
    
    inputs = np.array([[-1.0, 0.5, 2.0, -0.5]], dtype=np.float32)
    output = layer(inputs, training=False)
    
    # Deterministic: threshold at 0
    expected = np.array([[0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    assert np.allclose(output, expected)


def test_voting_layer_stochastic():
    """Test VotingLayer in stochastic mode."""
    np.random.seed(42)
    layer = VotingLayer(temperature=1.0, stochastic=True)
    
    # Positive input should have high probability of voting yes
    inputs = np.array([[10.0] * 100], dtype=np.float32)
    output = layer(inputs, training=True)
    
    # Most votes should be 1
    assert np.mean(output) > 0.8


def test_voting_layer_temperature():
    """Test VotingLayer temperature parameter."""
    layer_low = VotingLayer(temperature=0.1, stochastic=False)
    layer_high = VotingLayer(temperature=10.0, stochastic=False)
    
    inputs = np.array([[0.5]], dtype=np.float32)
    
    # Both should still use threshold for deterministic mode
    output_low = layer_low(inputs, training=False)
    output_high = layer_high(inputs, training=False)
    
    # Both should give same result for deterministic
    assert output_low.shape == output_high.shape


def test_voting_layer_config():
    """Test VotingLayer configuration."""
    layer = VotingLayer(temperature=0.8, stochastic=True, name='voting')
    
    config = layer.get_config()
    assert config['temperature'] == 0.8
    assert config['stochastic'] is True
    assert config['name'] == 'voting'


def test_regime_context_layer_concat():
    """Test RegimeContextLayer with concat fusion."""
    context_dim = 3
    layer = RegimeContextLayer(context_dim=context_dim, fusion_mode='concat')
    
    # Input includes context in first 3 dimensions
    inputs = np.random.randn(5, 10).astype(np.float32)
    output = layer(inputs)
    
    # Concat mode should pass through
    assert output.shape == inputs.shape


def test_regime_context_layer_add():
    """Test RegimeContextLayer with add fusion."""
    context_dim = 3
    actor_features = 5
    layer = RegimeContextLayer(context_dim=context_dim, fusion_mode='add')
    
    # Build layer
    inputs = np.random.randn(4, context_dim + actor_features).astype(np.float32)
    output = layer(inputs)
    
    assert output.shape == (4, actor_features)
    assert len(layer.trainable_weights) == 1  # context_projection


def test_regime_context_layer_multiply():
    """Test RegimeContextLayer with multiply fusion."""
    context_dim = 2
    actor_features = 4
    layer = RegimeContextLayer(context_dim=context_dim, fusion_mode='multiply')
    
    inputs = np.random.randn(3, context_dim + actor_features).astype(np.float32)
    output = layer(inputs)
    
    assert output.shape == (3, actor_features)
    assert len(layer.trainable_weights) == 1  # gate_weight


def test_regime_context_layer_config():
    """Test RegimeContextLayer configuration."""
    layer = RegimeContextLayer(
        context_dim=5,
        fusion_mode='add',
        name='regime'
    )
    
    config = layer.get_config()
    assert config['context_dim'] == 5
    assert config['fusion_mode'] == 'add'
    assert config['name'] == 'regime'


def test_regime_context_layer_invalid_fusion():
    """Test RegimeContextLayer with invalid fusion mode raises error."""
    layer = RegimeContextLayer(context_dim=3, fusion_mode='invalid')
    
    inputs = np.random.randn(2, 8).astype(np.float32)
    
    with pytest.raises(ValueError, match="Unknown fusion mode"):
        layer(inputs)




