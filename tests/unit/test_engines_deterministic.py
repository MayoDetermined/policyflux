"""Tests for policyflux.engines.deterministic_engine.DeterministicEngine."""


import policyflux.pfrandom as pfrandom
from policyflux.core.abstract_layer import Layer
from policyflux.core.pf_typing import UtilitySpace
from policyflux.engines.deterministic_engine import DeterministicEngine
from policyflux.engines.session_management import Session
from policyflux.toolbox.actor_models import SequentialVoter
from policyflux.toolbox.bill_models import SequentialBill
from policyflux.toolbox.congress_model import SequentialCongressModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubLayer(Layer):
    """Minimal layer that returns a fixed probability."""

    def __init__(self, return_value: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self._return_value = return_value

    def call(self, bill_space: UtilitySpace, **kwargs) -> float:
        return self._return_value

    def compile(self) -> None:
        pass


def _build_deterministic_engine(
    num_actors: int = 5,
    layer_prob: float = 0.7,
    seed: int = 42,
) -> DeterministicEngine:
    pfrandom.set_seed(seed)
    model = SequentialCongressModel()
    for _ in range(num_actors):
        voter = SequentialVoter()
        voter.add_layer(_StubLayer(return_value=layer_prob))
        model.add_congressman(voter)
    model.compile()

    bill = SequentialBill()
    bill.make_random_position(dim=2)

    session = Session(
        n=1,
        seed=seed,
        bill=bill,
        description="deterministic test",
        congress_model=model,
    )
    return DeterministicEngine(session_params=session)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeterministicEngine:
    def test_run_returns_int(self) -> None:
        engine = _build_deterministic_engine()
        result = engine.run()
        assert isinstance(result, int)

    def test_result_within_range(self) -> None:
        n = 10
        engine = _build_deterministic_engine(num_actors=n)
        result = engine.run()
        assert 0 <= result <= n

    def test_deterministic_same_input_same_output(self) -> None:
        """Running twice with the same seed produces the same output."""
        result_a = _build_deterministic_engine(seed=42).run()
        result_b = _build_deterministic_engine(seed=42).run()
        assert result_a == result_b

    def test_different_seeds_may_differ(self) -> None:
        """Different seeds are very likely to produce different results (with enough actors)."""
        result_a = _build_deterministic_engine(num_actors=50, seed=1).run()
        result_b = _build_deterministic_engine(num_actors=50, seed=999).run()
        # With 50 actors, extremely unlikely to be the same
        assert result_a != result_b

    def test_results_stored_on_engine(self) -> None:
        engine = _build_deterministic_engine()
        returned = engine.run()
        assert engine.results == returned

    def test_high_prob_layer_many_votes_for(self) -> None:
        """With probability 0.99, almost all actors should vote for."""
        engine = _build_deterministic_engine(num_actors=20, layer_prob=0.99, seed=42)
        result = engine.run()
        assert result >= 18

    def test_low_prob_layer_few_votes_for(self) -> None:
        """With probability 0.01, almost no actors should vote for."""
        engine = _build_deterministic_engine(num_actors=20, layer_prob=0.01, seed=42)
        result = engine.run()
        assert result <= 2
