"""TensorFlow-style API Usage Examples for PolicyFlux.

This script demonstrates how to use the new TensorFlow/Keras-like API
for building, training, and evaluating behavioral models.
"""

import numpy as np

import policyflux as pf


def example_1_simple_sequential_model():
    """Example 1: Simple sequential model for voting prediction."""
    print("\n" + "=" * 70)
    print("Example 1: Simple Sequential Model")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Create synthetic actor states
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Create binary voting targets - reshape to match output
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32).reshape(-1, 1)
    
    # Build a sequential model - output single value
    model = pf.Sequential()
    model.add(pf.ActorLayer(units=32, activation='tanh'))
    model.add(pf.ActorLayer(units=16, activation='relu'))
    model.add(pf.ActorLayer(units=1, activation='sigmoid'))  # Output layer
    
    # Compile the model
    model.compile(
        optimizer=pf.Adam(learning_rate=0.01),
        loss=pf.MSE(),
        metrics=[pf.Accuracy()]
    )
    
    # Print model summary
    model.summary()
    
    # Train the model
    history = model.fit(
        X, y,
        epochs=5,
        batch_size=16,
        verbose=1
    )
    
    # Make predictions
    predictions = model.predict(X[:5])
    print(f"\nSample predictions: {predictions.flatten()}")
    print(f"Sample targets: {y[:5]}")
    
    return model, history


def example_2_network_influence_model():
    """Example 2: Model with network influence."""
    print("\n" + "=" * 70)
    print("Example 2: Network Influence Model")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(123)
    n_actors = 50
    n_features = 8
    
    # Create synthetic actor states
    X = np.random.randn(n_actors, n_features).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32).reshape(-1, 1)
    
    # Create a network adjacency matrix (random network)
    adjacency = np.random.rand(n_actors, n_actors).astype(np.float32)
    adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
    np.fill_diagonal(adjacency, 0)  # No self-loops
    
    # Build model with network influence
    model = pf.Sequential()
    model.add(pf.ActorLayer(units=16, activation='tanh'))
    
    # Add network influence layer
    network_layer = pf.NetworkInfluenceLayer(
        influence_strength=0.3,
        normalization='symmetric'
    )
    network_layer.set_adjacency(adjacency)
    model.add(network_layer)
    
    model.add(pf.ActorLayer(units=8, activation='relu'))
    model.add(pf.ActorLayer(units=1, activation='sigmoid'))  # Output layer
    
    # Compile
    model.compile(
        optimizer=pf.SGD(learning_rate=0.01, momentum=0.9),
        loss=pf.MSE(),
        metrics=[pf.Accuracy(), pf.Precision(), pf.Recall()]
    )
    
    # Train (use full batch size to work with network influence)
    history = model.fit(
        X, y,
        epochs=3,
        batch_size=n_actors,  # Must match adjacency matrix size
        verbose=1
    )
    
    print(f"\nFinal accuracy: {history['accuracy'][-1]:.4f}")
    
    return model, history


def example_3_with_callbacks():
    """Example 3: Training with callbacks."""
    print("\n" + "=" * 70)
    print("Example 3: Training with Callbacks")
    print("=" * 70)
    
    # Generate synthetic data with validation split
    np.random.seed(456)
    n_samples = 200
    n_features = 12
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] > 0).astype(np.float32).reshape(-1, 1)
    
    # Split into train and validation
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Build model
    model = pf.Sequential([
        pf.ActorLayer(units=24, activation='tanh', use_ideology=True, ideology_dim=2),
        pf.ActorLayer(units=12, activation='relu'),
        pf.ActorLayer(units=1, activation='sigmoid')  # Output layer
    ])
    
    # Compile
    model.compile(
        optimizer=pf.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999),
        loss=pf.MSE(),
        metrics=[pf.Accuracy()]
    )
    
    # Setup callbacks
    early_stopping = pf.EarlyStopping(
        monitor='val_loss',
        patience=3,
        min_delta=0.001,
        restore_best_weights=True
    )
    
    # Note: Using /tmp for temporary checkpoint file
    checkpoint = pf.ModelCheckpoint(
        filepath='/tmp/model_best.npz',
        monitor='val_accuracy',
        save_best_only=True
    )
    
    # Train with callbacks
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=20,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    print(f"\nTraining stopped at epoch: {len(history['loss'])}")
    print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    
    return model, history


def example_4_regime_context():
    """Example 4: Model with regime context."""
    print("\n" + "=" * 70)
    print("Example 4: Regime Context Model")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(789)
    n_actors = 80
    context_dim = 5
    actor_features = 10
    
    # Create data with context features prepended
    context = np.random.randn(n_actors, context_dim).astype(np.float32)
    actor_states = np.random.randn(n_actors, actor_features).astype(np.float32)
    X = np.concatenate([context, actor_states], axis=1)
    
    # Target depends on both context and actor states
    y = (context[:, 0] + actor_states[:, 0] > 0).astype(np.float32).reshape(-1, 1)
    
    # Build model with regime context
    model = pf.Sequential()
    model.add(pf.RegimeContextLayer(context_dim=context_dim, fusion_mode='add'))
    model.add(pf.ActorLayer(units=16, activation='tanh'))
    model.add(pf.VotingLayer(temperature=1.0))
    
    # Compile
    model.compile(
        optimizer=pf.Adam(learning_rate=0.01),
        loss=pf.MSE(),
        metrics=[pf.Accuracy()]
    )
    
    # Train
    history = model.fit(
        X, y,
        epochs=3,
        batch_size=16,
        verbose=1
    )
    
    return model, history


def example_5_integration_with_congress_simulator():
    """Example 5: Integration with existing CongressSimulator."""
    print("\n" + "=" * 70)
    print("Example 5: Integration with CongressSimulator")
    print("=" * 70)
    
    # Create a custom actor model
    actor_model = pf.Sequential([
        pf.ActorLayer(units=32, activation='tanh', use_ideology=True),
        pf.ActorLayer(units=16, activation='relu'),
        pf.VotingLayer(temperature=0.8)
    ])
    
    # Compile the actor model
    actor_model.compile(
        optimizer=pf.Adam(learning_rate=0.001),
        loss=pf.MSE(),
        metrics=[pf.Accuracy()]
    )
    
    # Create CongressSimulator with custom actor model
    # Note: This demonstrates the API, but won't run without data
    simulator = pf.CongressSimulator(
        scenario="stable",
        actor_model=actor_model
    )
    
    print("CongressSimulator created with custom actor model")
    print(f"Actor model has {len(actor_model.layers)} layers")
    
    # The simulator can still be used with the traditional API
    # simulator.compile(scenario="crisis")
    # simulator.fit(use_cache=True)
    # report = simulator.simulate(n_simulations=10, steps=5)
    
    return simulator


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("PolicyFlux TensorFlow-Style API Examples")
    print("=" * 70)
    
    # Run examples
    try:
        model1, history1 = example_1_simple_sequential_model()
        print("✓ Example 1 completed successfully")
    except Exception as e:
        print(f"✗ Example 1 failed: {e}")
    
    try:
        model2, history2 = example_2_network_influence_model()
        print("✓ Example 2 completed successfully")
    except Exception as e:
        print(f"✗ Example 2 failed: {e}")
    
    try:
        model3, history3 = example_3_with_callbacks()
        print("✓ Example 3 completed successfully")
    except Exception as e:
        print(f"✗ Example 3 failed: {e}")
    
    try:
        model4, history4 = example_4_regime_context()
        print("✓ Example 4 completed successfully")
    except Exception as e:
        print(f"✗ Example 4 failed: {e}")
    
    try:
        simulator = example_5_integration_with_congress_simulator()
        print("✓ Example 5 completed successfully")
    except Exception as e:
        print(f"✗ Example 5 failed: {e}")
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


