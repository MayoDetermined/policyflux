"""Core components for building neural networks.

This module provides the fundamental building blocks for creating,
training, and evaluating models following TensorFlow/Keras conventions.
"""

from policyflux.core.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from policyflux.core.layer import Layer
from policyflux.core.losses import CrossEntropy, Loss, MSE
from policyflux.core.metrics import (
    Accuracy,
    MeanSquaredError,
    Metric,
    Precision,
    Recall,
)
from policyflux.core.model import Model, Sequential
from policyflux.core.optimizer import Adam, Optimizer, SGD

__all__ = [
    # Base classes
    'Layer',
    'Model',
    'Sequential',
    'Optimizer',
    'Loss',
    'Metric',
    'Callback',
    # Optimizers
    'Adam',
    'SGD',
    # Losses
    'MSE',
    'CrossEntropy',
    # Metrics
    'Accuracy',
    'MeanSquaredError',
    'Precision',
    'Recall',
    # Callbacks
    'CallbackList',
    'EarlyStopping',
    'ModelCheckpoint',
    'TensorBoard',
]
