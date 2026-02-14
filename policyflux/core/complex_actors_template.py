from abc import ABC
from typing import List

from policyflux.core.aggregation_strategy import AggregationStrategy
from policyflux.core.layer_template import Layer

class ComplexActor(ABC):
    """Abstract base class for complex"""

    def __init__(self, name: str, layers: List[Layer], aggregation_strategy: AggregationStrategy):
        self.name = name
        self.layers = layers
        self.aggregation_strategy = aggregation_strategy