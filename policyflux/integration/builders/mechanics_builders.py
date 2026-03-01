from policyflux.exceptions import ConfigurationError

from ...core.aggregation_strategy import (
    AggregationStrategy,
    AverageAggregation,
    MultiplicativeAggregation,
    SequentialAggregation,
    WeightedAggregation,
)
from ..config import IntegrationConfig


def build_aggregation_strategy(config: IntegrationConfig) -> AggregationStrategy:
    if config.aggregation_strategy == "average":
        return AverageAggregation()
    if config.aggregation_strategy == "multiplicative":
        return MultiplicativeAggregation()
    if config.aggregation_strategy == "weighted":
        if not config.aggregation_weights:
            raise ConfigurationError("aggregation_weights must be provided for weighted strategy")
        return WeightedAggregation(config.aggregation_weights)
    return SequentialAggregation()
