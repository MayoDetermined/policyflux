# TensorFlow-like API Documentation

## Overview

PolicyFlux now includes a TensorFlow/Keras-style API for building, training, and evaluating behavioral models. This API makes it easy to experiment with different model architectures and enables rapid prototyping of congressional dynamics models.

## Design Philosophy

The TensorFlow-like API follows these principles:

1. **Composability**: Build complex models from simple, reusable components
2. **Familiar Interface**: Use the compile/fit/predict pattern familiar to ML practitioners
3. **Flexibility**: Support both simple sequential models and complex custom architectures
4. **Domain-Specific**: Provide specialized layers for political behavioral modeling
5. **Backward Compatibility**: Work seamlessly with existing PolicyFlux components

## Core Concepts

### Layers

Layers are the fundamental building blocks of models. Each layer performs a specific transformation on data.

```python
import policyflux as pf

# Create a layer
layer = pf.ActorLayer(units=64, activation='tanh')

# Layers build their weights automatically on first call
output = layer(inputs)
```

**Base Layer Properties:**
- `weights`: All weights (trainable + non-trainable)
- `trainable_weights`: Only trainable weights
- `built`: Whether the layer has been built
- `trainable`: Whether the layer should be trained

### Models

Models combine multiple layers and provide training/evaluation functionality.

```python
# Create a sequential model
model = pf.Sequential()
model.add(pf.ActorLayer(units=32))
model.add(pf.VotingLayer())

# Or create with layers in constructor
model = pf.Sequential([
    pf.ActorLayer(units=32),
    pf.VotingLayer()
])
```

### Compile, Fit, Predict Pattern

```python
# 1. Compile: Configure the model for training
model.compile(
    optimizer=pf.Adam(learning_rate=0.001),
    loss=pf.MSE(),
    metrics=[pf.Accuracy()]
)

# 2. Fit: Train the model
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[pf.EarlyStopping()]
)

# 3. Predict: Generate predictions
predictions = model.predict(x_test)
```

## Available Components

### Core Layers

#### ActorLayer

Dense transformation layer with ideology-aware features for modeling actor behavior.

```python
layer = pf.ActorLayer(
    units=64,                    # Number of output units
    activation='tanh',           # Activation function
    use_ideology=True,           # Use ideology features
    ideology_dim=1               # Dimension of ideology features
)
```

**Activations:** `'tanh'`, `'relu'`, `'sigmoid'`, `'linear'`

#### NetworkInfluenceLayer

Incorporates network/social influence on actor states using adjacency matrices.

```python
layer = pf.NetworkInfluenceLayer(
    influence_strength=0.5,      # Strength of influence (0-1)
    normalization='symmetric'    # Normalization type
)

# Set the adjacency matrix
layer.set_adjacency(adjacency_matrix)
```

**Normalization Options:**
- `'symmetric'`: D^(-1/2) A D^(-1/2)
- `'row'`: Row-wise normalization
- `'none'`: No normalization

**Formula:** `output = (1-α) * input + α * (normalized_adjacency @ input)`

#### VotingLayer

Converts continuous utility/preference scores to binary voting decisions.

```python
layer = pf.VotingLayer(
    temperature=1.0,             # Softmax temperature
    stochastic=False             # Use stochastic sampling
)
```

**Modes:**
- `stochastic=False`: Deterministic threshold at 0
- `stochastic=True`: Sample from sigmoid probabilities

#### RegimeContextLayer

Fuses external context (regime pressure, public opinion) with actor states.

```python
layer = pf.RegimeContextLayer(
    context_dim=10,              # Dimension of context vector
    fusion_mode='concat'         # Fusion strategy
)
```

**Fusion Modes:**
- `'concat'`: Concatenate context vector
- `'add'`: Project and add context to states
- `'multiply'`: Use context as gating mechanism

### Optimizers

#### Adam

Adaptive Moment Estimation optimizer with momentum.

```python
optimizer = pf.Adam(
    learning_rate=0.001,
    beta1=0.9,                   # First moment decay
    beta2=0.999,                 # Second moment decay
    epsilon=1e-7                 # Numerical stability
)
```

#### SGD

Stochastic Gradient Descent with optional momentum.

```python
optimizer = pf.SGD(
    learning_rate=0.01,
    momentum=0.9                 # Momentum factor (0-1)
)
```

### Loss Functions

#### MSE

Mean Squared Error for regression tasks.

```python
loss = pf.MSE()
```

#### CrossEntropy

Cross-entropy loss for classification tasks.

```python
loss = pf.CrossEntropy(
    from_logits=False            # Whether inputs are logits
)
```

### Metrics

#### Accuracy

Classification accuracy metric.

```python
metric = pf.Accuracy()
```

#### Precision & Recall

Binary classification metrics.

```python
precision = pf.Precision(threshold=0.5)
recall = pf.Recall(threshold=0.5)
```

### Callbacks

#### EarlyStopping

Stop training when a metric stops improving.

```python
callback = pf.EarlyStopping(
    monitor='val_loss',          # Metric to monitor
    patience=5,                  # Epochs to wait
    min_delta=0.001,            # Minimum improvement
    mode='min',                  # 'min' or 'max'
    restore_best_weights=True    # Restore best weights
)
```

#### ModelCheckpoint

Save model weights during training.

```python
callback = pf.ModelCheckpoint(
    filepath='model_weights.npz',
    monitor='val_loss',
    save_best_only=True
)
```

## Complete Example

```python
import numpy as np
import policyflux as pf

# Generate synthetic data
X_train = np.random.randn(100, 10).astype(np.float32)
y_train = (X_train[:, 0] > 0).astype(np.float32)

# Build model
model = pf.Sequential([
    pf.ActorLayer(units=32, activation='tanh', use_ideology=True),
    pf.ActorLayer(units=16, activation='relu'),
    pf.VotingLayer(temperature=1.0)
])

# Compile
model.compile(
    optimizer=pf.Adam(learning_rate=0.001),
    loss=pf.MSE(),
    metrics=[pf.Accuracy()]
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    callbacks=[pf.EarlyStopping(patience=3)]
)

# Predict
predictions = model.predict(X_train)
```

## Creating Custom Layers

You can create custom layers by subclassing `Layer`:

```python
from policyflux.core import Layer
import numpy as np

class MyCustomLayer(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
    
    def call(self, inputs, training=False):
        return inputs @ self.kernel
    
    def get_config(self):
        config = super().get_config()
        config['units'] = self.units
        return config
```

## Integration with CongressSimulator

The new API integrates seamlessly with the existing CongressSimulator:

```python
# Create a custom actor model
actor_model = pf.Sequential([
    pf.ActorLayer(units=64, activation='tanh'),
    pf.NetworkInfluenceLayer(influence_strength=0.4),
    pf.VotingLayer()
])

# Compile the model
actor_model.compile(
    optimizer=pf.Adam(learning_rate=0.001),
    loss=pf.MSE(),
    metrics=[pf.Accuracy()]
)

# Use with CongressSimulator
simulator = pf.CongressSimulator(
    scenario="crisis",
    actor_model=actor_model
)

# Traditional workflow still works
simulator.compile()
simulator.fit(use_cache=True)
report = simulator.simulate(n_simulations=50, steps=10)
```

## Migration Guide

### From Old API to New API

**Old API (Hardcoded behavior):**
```python
simulator = pf.CongressSimulator()
simulator.compile(scenario="crisis")
simulator.fit()
```

**New API (Custom behavior model):**
```python
# Define custom behavior
model = pf.Sequential([
    pf.ActorLayer(units=32),
    pf.VotingLayer()
])
model.compile(optimizer=pf.Adam(), loss=pf.MSE())

# Use with simulator
simulator = pf.CongressSimulator(actor_model=model)
simulator.compile(scenario="crisis")
simulator.fit()
```

## Best Practices

1. **Start Simple**: Begin with small sequential models before building complex architectures
2. **Use Callbacks**: Leverage EarlyStopping to prevent overfitting
3. **Monitor Metrics**: Track multiple metrics during training
4. **Normalize Data**: Scale input features for better training stability
5. **Batch Size**: Use appropriate batch sizes (16-32 typically work well)
6. **Learning Rate**: Start with Adam's default (0.001) and adjust as needed

## Limitations

1. **Gradient Computation**: Current implementation uses simplified gradient approximation. For production use, consider integrating with automatic differentiation frameworks.
2. **GPU Support**: Core API uses NumPy (CPU). For GPU acceleration, consider PyTorch/JAX backend integration.
3. **Advanced Features**: Some advanced Keras features (e.g., custom training loops, multiple inputs/outputs) are not yet implemented.

## Future Enhancements

- [ ] Automatic differentiation integration
- [ ] PyTorch/JAX backend support
- [ ] Model serialization (save/load full models)
- [ ] Distributed training support
- [ ] Advanced model architectures (Graph Neural Networks)
- [ ] TensorBoard integration
- [ ] Hyperparameter optimization tools

## API Reference

For complete API documentation, see the docstrings in:
- `policyflux/core/layer.py`
- `policyflux/core/model.py`
- `policyflux/core/optimizer.py`
- `policyflux/core/losses.py`
- `policyflux/core/metrics.py`
- `policyflux/core/callbacks.py`
- `policyflux/layers/behavioral.py`
