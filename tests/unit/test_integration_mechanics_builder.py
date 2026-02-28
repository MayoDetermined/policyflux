import pytest

from policyflux.core.aggregation_strategy import (
    AverageAggregation,
    MultiplicativeAggregation,
    SequentialAggregation,
    WeightedAggregation,
)
from policyflux.exceptions import ConfigurationError
from policyflux.integration.builders.mechanics_builders import build_aggregation_strategy
from policyflux.integration.config import IntegrationConfig


def test_build_aggregation_strategy_returns_average() -> None:
    config = IntegrationConfig(aggregation_strategy="average")
    strategy = build_aggregation_strategy(config)

    assert isinstance(strategy, AverageAggregation)


def test_build_aggregation_strategy_returns_multiplicative() -> None:
    config = IntegrationConfig(aggregation_strategy="multiplicative")
    strategy = build_aggregation_strategy(config)

    assert isinstance(strategy, MultiplicativeAggregation)


def test_build_aggregation_strategy_returns_weighted() -> None:
    config = IntegrationConfig(
        aggregation_strategy="weighted",
        aggregation_weights=[0.5, 0.5],
    )
    strategy = build_aggregation_strategy(config)

    assert isinstance(strategy, WeightedAggregation)


def test_build_aggregation_strategy_requires_weights_for_weighted() -> None:
    config = IntegrationConfig(aggregation_strategy="weighted", aggregation_weights=None)

    with pytest.raises(ConfigurationError):
        build_aggregation_strategy(config)


def test_build_aggregation_strategy_falls_back_to_sequential() -> None:
    config = IntegrationConfig(aggregation_strategy="unknown")
    strategy = build_aggregation_strategy(config)

    assert isinstance(strategy, SequentialAggregation)
