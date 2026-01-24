from typing import List, Optional
from ..core.actors_template import CongressMan
from ..core.bill_template import Bill
from ..core.layer_template import Layer
from ..core.types import UtilitySpace
from ..core.id_generator import get_id_generator
from ..core.aggregation_strategy import AggregationStrategy, SequentialAggregation
import importlib
import policyflux.pfrandom as pfrandom
from policyflux.logging_config import logger

class SequentialVoter(CongressMan):
    """
    Voter with dependency-injected layers for voting decisions.
    
    Layers are computed in order and their results are aggregated
    to produce a final voting decision using an AggregationStrategy.
    """
    
    def __init__(
        self, 
        id: Optional[int] = None, 
        layers: Optional[List[Layer]] = None, 
        name: str = "",
        aggregation_strategy: Optional[AggregationStrategy] = None
    ) -> None:
        if id is None:
            id = get_id_generator().generate_actor_id()
        super().__init__(id)
        self.name = name or f"Voter_{id}"
        self.layers: List[Layer] = layers if layers is not None else []
        self.aggregation_strategy = aggregation_strategy or SequentialAggregation()
    
    def add_layer(self, layer: Layer) -> None:
        """Inject a new layer into the voter."""
        if not isinstance(layer, Layer):
            raise TypeError(f"Expected Layer instance, got {type(layer)}")
        self.layers.append(layer)
    
    def set_aggregation_strategy(self, strategy: AggregationStrategy) -> None:
        """Change the aggregation strategy for combining layer outputs."""
        if not isinstance(strategy, AggregationStrategy):
            raise TypeError(f"Expected AggregationStrategy instance, got {type(strategy)}")
        self.aggregation_strategy = strategy
    
    def remove_layer(self, layer_id: int) -> bool:
        """Remove a layer by its ID."""
        self.layers = [l for l in self.layers if l.id != layer_id]
        return True
    
    def compute_layers(self, bill_space: UtilitySpace, **context) -> float:
        """
        Aggregate layer outputs using the configured aggregation strategy.
        
        Args:
            bill_space: The bill's position in policy space
            **context: Additional context for layers (e.g., lobbying intensity, public opinion)
            
        Returns:
            Aggregated decision probability [0, 1]
        """
        if not self.layers:
            return self.yes_chance  # Fallback to default if no layers
        
        return self.aggregation_strategy.aggregate(self.layers, bill_space, **context)
    
    def vote(self, bill: Bill, bill_space: Optional[UtilitySpace] = None, **context) -> bool:
        """
        Cast a vote based on layers.
        
        Args:
            bill: The Bill object
            bill_space: Bill's position in policy space (if None, uses dummy)
            **context: Additional context for layers
            
        Returns:
            True if voting YES, False if voting NO
        """
        if bill_space is None:
            bill_space = [0.0]  # Dummy space if not provided
        
        decision_prob = self.compute_layers(bill_space, **context)
        # Use package RNG for deterministic behaviour when seeded
        return pfrandom.random() < decision_prob