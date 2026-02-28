"""Tests for policyflux.toolbox.actors.SequentialVoter."""

import pytest

import policyflux.pfrandom as pfrandom
from policyflux.core.aggregation_strategy import (
    AggregationStrategy,
    AverageAggregation,
    SequentialAggregation,
)
from policyflux.core.layer import Layer
from policyflux.core.types import UtilitySpace
from policyflux.exceptions import ValidationError
from policyflux.toolbox.actors import SequentialVoter
from policyflux.toolbox.bill import SequentialBill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubLayer(Layer):
    """A minimal concrete Layer that returns a fixed probability."""

    def __init__(self, return_value: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self._return_value = return_value

    def call(self, bill_space: UtilitySpace, **kwargs) -> float:
        return self._return_value

    def compile(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSequentialVoterConstruction:
    def test_default_params_auto_id(self) -> None:
        voter = SequentialVoter()
        assert isinstance(voter.id, int)
        assert voter.id > 0
        assert voter.name.startswith("Voter_")
        assert isinstance(voter.layers, list)
        assert len(voter.layers) == 0
        assert isinstance(voter.aggregation, AggregationStrategy)

    def test_explicit_id(self) -> None:
        voter = SequentialVoter(id=999, name="CustomVoter")
        assert voter.id == 999
        assert voter.name == "CustomVoter"

    def test_default_yes_chance(self) -> None:
        voter = SequentialVoter()
        assert voter.yes_chance == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class TestSequentialVoterLayers:
    def test_add_layer(self) -> None:
        voter = SequentialVoter()
        layer = _StubLayer()
        voter.add_layer(layer)
        assert len(voter.layers) == 1
        assert voter.layers[0] is layer

    def test_add_layer_rejects_non_layer(self) -> None:
        voter = SequentialVoter()
        with pytest.raises(ValidationError):
            voter.add_layer("not_a_layer")  # type: ignore[arg-type]

    def test_remove_layer_by_id(self) -> None:
        voter = SequentialVoter()
        layer = _StubLayer()
        voter.add_layer(layer)
        assert len(voter.layers) == 1

        result = voter.remove_layer(layer.id)
        assert result is True
        assert len(voter.layers) == 0

    def test_remove_layer_nonexistent_id(self) -> None:
        voter = SequentialVoter()
        layer = _StubLayer()
        voter.add_layer(layer)

        voter.remove_layer(999999)
        # The original layer is still present
        assert len(voter.layers) == 1


# ---------------------------------------------------------------------------
# Aggregation strategy
# ---------------------------------------------------------------------------

class TestSequentialVoterAggregation:
    def test_set_aggregation_strategy(self) -> None:
        voter = SequentialVoter()
        new_strategy = AverageAggregation()
        voter.set_aggregation_strategy(new_strategy)
        assert voter.aggregation is new_strategy

    def test_set_aggregation_strategy_rejects_non_strategy(self) -> None:
        voter = SequentialVoter()
        with pytest.raises(ValidationError):
            voter.set_aggregation_strategy("not_a_strategy")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

class TestSequentialVoterVote:
    def test_vote_returns_bool(self) -> None:
        pfrandom.set_seed(42)
        voter = SequentialVoter()
        bill = SequentialBill(position=[0.5, 0.5])
        result = voter.vote(bill)
        assert isinstance(result, bool)

    def test_compute_layers_no_layers_returns_yes_chance(self) -> None:
        voter = SequentialVoter()
        result = voter.compute_layers([0.5, 0.5])
        assert result == pytest.approx(voter.yes_chance)

    def test_compute_layers_with_layer(self) -> None:
        voter = SequentialVoter()
        voter.add_layer(_StubLayer(return_value=0.9))
        result = voter.compute_layers([0.5, 0.5])
        assert result == pytest.approx(0.9)

    def test_vote_with_seed_is_deterministic(self) -> None:
        """Same seed + same layers should produce the same vote sequence."""
        results_a: list[bool] = []
        results_b: list[bool] = []

        bill = SequentialBill(position=[0.5, 0.5])

        for container in (results_a, results_b):
            pfrandom.set_seed(42)
            voter = SequentialVoter(id=1)
            voter.add_layer(_StubLayer(return_value=0.8))
            for _ in range(20):
                container.append(voter.vote(bill))

        assert results_a == results_b

    def test_vote_high_prob_layer_mostly_true(self) -> None:
        """A layer returning 0.99 should yield True almost all the time."""
        pfrandom.set_seed(42)
        voter = SequentialVoter()
        voter.add_layer(_StubLayer(return_value=0.99))
        bill = SequentialBill(position=[0.5])
        votes = [voter.vote(bill) for _ in range(100)]
        assert sum(votes) > 90

    def test_vote_low_prob_layer_mostly_false(self) -> None:
        """A layer returning 0.01 should yield False almost all the time."""
        pfrandom.set_seed(42)
        voter = SequentialVoter()
        voter.add_layer(_StubLayer(return_value=0.01))
        bill = SequentialBill(position=[0.5])
        votes = [voter.vote(bill) for _ in range(100)]
        assert sum(votes) < 10
