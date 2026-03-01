import pytest

from policyflux.core.aggregation_strategy import (
    AverageAggregation,
    MultiplicativeAggregation,
    SequentialAggregation,
    WeightedAggregation,
)
from policyflux.core.abstract_layer import Layer
from policyflux.core.pf_typing import PolicyPosition
from policyflux.exceptions import ValidationError


class _ConstLayer(Layer):
    def __init__(self, value: float) -> None:
        super().__init__(id=1, name="const", input_dim=2, output_dim=2)
        self.value = value

    def call(self, bill_space, **kwargs) -> float:
        return self.value

    def compile(self) -> None:
        return None


class _AddBaseLayer(Layer):
    def __init__(self, delta: float) -> None:
        super().__init__(id=2, name="add_base", input_dim=2, output_dim=2)
        self.delta = delta

    def call(self, bill_space, **kwargs) -> float:
        return kwargs.get("base_prob", 0.0) + self.delta

    def compile(self) -> None:
        return None


def test_sequential_aggregation_passes_base_prob() -> None:
    strategy = SequentialAggregation()
    layers = [_ConstLayer(0.4), _AddBaseLayer(0.1)]

    value = strategy.aggregate(layers, bill_position=PolicyPosition((0.0, 0.0)))

    assert value == pytest.approx(0.5)


def test_sequential_aggregation_clips_output_to_one() -> None:
    strategy = SequentialAggregation()
    layers = [_ConstLayer(0.9), _AddBaseLayer(0.8)]

    value = strategy.aggregate(layers, bill_position=PolicyPosition((0.0, 0.0)))

    assert value == 1.0


def test_average_aggregation_returns_mean() -> None:
    strategy = AverageAggregation()
    layers = [_ConstLayer(0.2), _ConstLayer(0.8)]

    value = strategy.aggregate(layers, bill_position=PolicyPosition((0.0, 0.0)))

    assert value == pytest.approx(0.5)


def test_weighted_aggregation_requires_weights_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        WeightedAggregation([0.3, 0.3])


def test_weighted_aggregation_computes_weighted_sum() -> None:
    strategy = WeightedAggregation([0.25, 0.75])
    layers = [_ConstLayer(0.2), _ConstLayer(0.6)]

    value = strategy.aggregate(layers, bill_position=PolicyPosition((0.0, 0.0)))

    assert value == pytest.approx(0.5)


def test_weighted_aggregation_validates_layer_count() -> None:
    strategy = WeightedAggregation([1.0])

    with pytest.raises(ValidationError):
        strategy.aggregate(
            [_ConstLayer(0.7), _ConstLayer(0.2)],
            bill_position=PolicyPosition((0.0, 0.0)),
        )


def test_multiplicative_aggregation_multiplies_values() -> None:
    strategy = MultiplicativeAggregation()
    layers = [_ConstLayer(0.8), _ConstLayer(0.5)]

    value = strategy.aggregate(layers, bill_position=PolicyPosition((0.0, 0.0)))

    assert value == pytest.approx(0.4)
