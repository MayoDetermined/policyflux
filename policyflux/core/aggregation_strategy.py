"""
Aggregation strategies for combining multiple layer outputs in voting decisions.

This module implements the Strategy Pattern to allow flexible ways of combining
layer outputs (e.g., sequential chaining, averaging, weighted sums).
"""

from abc import ABC, abstractmethod
from typing import List
from .layer_template import Layer
from .types import UtilitySpace


class AggregationStrategy(ABC):
    """Abstract base class for layer aggregation strategies."""
    
    @abstractmethod
    def aggregate(self, layers: List[Layer], bill_space: UtilitySpace, **context) -> float:
        """
        Aggregate outputs from multiple layers.
        
        Args:
            layers: List of Layer objects to aggregate
            bill_space: Bill's position in policy space
            **context: Additional context for layers
            
        Returns:
            Aggregated decision probability [0, 1]
        """
        pass


class SequentialAggregation(AggregationStrategy):
    """
    Sequential aggregation: each layer modifies the output of the previous one.
    
    This is the default behavior where layers form a chain, with each layer
    receiving the previous probability as 'base_prob' in context.
    """
    
    def aggregate(self, layers: List[Layer], bill_space: UtilitySpace, **context) -> float:
        if not layers:
            return 0.5  # Neutral default
        
        # Start with first layer
        decision_prob = layers[0].call(bill_space, **context)
        
        # Apply subsequent layers sequentially
        for layer in layers[1:]:
            context['base_prob'] = decision_prob
            decision_prob = layer.call(bill_space, **context)
        
        # Ensure output is in valid range [0, 1]
        return max(0.0, min(1.0, decision_prob))


class AverageAggregation(AggregationStrategy):
    """
    Average aggregation: compute simple average of all layer outputs.
    
    Each layer computes independently, and the final decision is the mean.
    """
    
    def aggregate(self, layers: List[Layer], bill_space: UtilitySpace, **context) -> float:
        if not layers:
            return 0.5
        
        total = sum(layer.call(bill_space, **context) for layer in layers)
        avg = total / len(layers)
        
        return max(0.0, min(1.0, avg))


class WeightedAggregation(AggregationStrategy):
    """
    Weighted aggregation: compute weighted sum of layer outputs.
    
    Each layer has a weight, and the final decision is the weighted average.
    Weights must sum to 1.0.
    """
    
    def __init__(self, weights: List[float]):
        """
        Initialize with layer weights.
        
        Args:
            weights: List of weights corresponding to layers (must sum to 1.0)
        """
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
        self.weights = weights
    
    def aggregate(self, layers: List[Layer], bill_space: UtilitySpace, **context) -> float:
        if not layers:
            return 0.5
        
        if len(layers) != len(self.weights):
            raise ValueError(f"Number of layers ({len(layers)}) must match number of weights ({len(self.weights)})")
        
        total = sum(
            weight * layer.call(bill_space, **context)
            for weight, layer in zip(self.weights, layers)
        )
        
        return max(0.0, min(1.0, total))


class MultiplicativeAggregation(AggregationStrategy):
    """
    Multiplicative aggregation: multiply all layer outputs.
    
    This creates a "veto" effect where low probability from any layer
    significantly reduces the final probability.
    """
    
    def aggregate(self, layers: List[Layer], bill_space: UtilitySpace, **context) -> float:
        if not layers:
            return 0.5
        
        result = 1.0
        for layer in layers:
            result *= layer.call(bill_space, **context)
        
        return max(0.0, min(1.0, result))
