"""Tests for policyflux.toolbox.congress_model.SequentialCongressModel."""

import pytest

import policyflux.pfrandom as pfrandom
from policyflux.core.abstract_layer import Layer
from policyflux.core.pf_typing import PolicySpace, UtilitySpace
from policyflux.exceptions import DimensionMismatchError
from policyflux.layers.ideal_point import IdealPointLayer
from policyflux.toolbox.actor_models import SequentialVoter
from policyflux.toolbox.special_actors.speaker import SequentialSpeaker
from policyflux.toolbox.special_actors.white_house import SequentialPresident
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


def _make_voter_with_stub(prob: float = 0.7) -> SequentialVoter:
    voter = SequentialVoter()
    voter.add_layer(_StubLayer(return_value=prob))
    return voter


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSequentialCongressModelConstruction:
    def test_auto_id(self) -> None:
        model = SequentialCongressModel()
        assert isinstance(model.id, int)
        assert model.id > 0

    def test_explicit_id(self) -> None:
        model = SequentialCongressModel(id=55)
        assert model.id == 55

    def test_initial_state(self) -> None:
        model = SequentialCongressModel()
        assert model.congressmen == []
        assert model.lobbyists == []
        assert model.whips == []
        assert model.speaker is None
        assert model.president is None


# ---------------------------------------------------------------------------
# Adding congressmen
# ---------------------------------------------------------------------------

class TestSequentialCongressModelCongresmenManagement:
    def test_add_n_congressmen(self) -> None:
        model = SequentialCongressModel()
        model.add_n_congressmen(10)
        assert len(model.congressmen) == 10

    def test_add_n_congressmen_unique_ids(self) -> None:
        model = SequentialCongressModel()
        model.add_n_congressmen(5)
        ids = [c.id for c in model.congressmen]
        assert len(set(ids)) == 5

    def test_add_n_congressmen_with_layers(self) -> None:
        model = SequentialCongressModel()
        layer = _StubLayer()
        model.add_n_congressmen(3, layers=[layer])
        for voter in model.congressmen:
            assert len(voter.layers) == 1

    def test_add_and_pop_congressman(self) -> None:
        model = SequentialCongressModel()
        voter = SequentialVoter()
        model.add_congressman(voter)
        assert len(model.congressmen) == 1

        popped = model.pop_congressman()
        assert popped is voter
        assert len(model.congressmen) == 0


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

class TestSequentialCongressModelVoting:
    def test_cast_votes_returns_int(self) -> None:
        pfrandom.set_seed(42)
        model = SequentialCongressModel()
        for _ in range(5):
            model.add_congressman(_make_voter_with_stub(0.8))

        bill = SequentialBill(position=[0.5, 0.5])
        votes = model.cast_votes(bill)
        assert isinstance(votes, int)

    def test_cast_votes_within_range(self) -> None:
        pfrandom.set_seed(42)
        model = SequentialCongressModel()
        n = 10
        for _ in range(n):
            model.add_congressman(_make_voter_with_stub(0.5))

        bill = SequentialBill(position=[0.5])
        votes = model.cast_votes(bill)
        assert 0 <= votes <= n

    def test_cast_votes_all_yes_with_high_prob(self) -> None:
        pfrandom.set_seed(42)
        model = SequentialCongressModel()
        n = 20
        for _ in range(n):
            model.add_congressman(_make_voter_with_stub(0.999))

        bill = SequentialBill(position=[0.5])
        votes = model.cast_votes(bill)
        assert votes >= n - 2  # Allow tiny margin for randomness


# ---------------------------------------------------------------------------
# Special actors
# ---------------------------------------------------------------------------

class TestSequentialCongressModelSpecialActors:
    def test_set_speaker(self) -> None:
        model = SequentialCongressModel()
        speaker = SequentialSpeaker()
        model.set_speaker(speaker)
        assert model.speaker is speaker

    def test_set_president(self) -> None:
        model = SequentialCongressModel()
        president = SequentialPresident()
        model.set_president(president)
        assert model.president is president


# ---------------------------------------------------------------------------
# Compile / Report
# ---------------------------------------------------------------------------

class TestSequentialCongressModelCompileAndReport:
    def test_compile_does_not_raise(self) -> None:
        model = SequentialCongressModel()
        model.add_n_congressmen(3)
        model.compile()  # Should not raise even with no layers

    def test_compile_with_layers(self) -> None:
        model = SequentialCongressModel()
        model.add_n_congressmen(3, layers=[_StubLayer()])
        model.compile()  # Should not raise

    def test_make_report_returns_string(self) -> None:
        model = SequentialCongressModel()
        model.add_n_congressmen(5)
        report = model.make_report()
        assert isinstance(report, str)
        assert "5" in report


# ---------------------------------------------------------------------------
# Dimension mismatch detection
# ---------------------------------------------------------------------------

class TestSequentialCongressModelDimensionMismatch:
    def test_dimension_mismatch_raises(self) -> None:
        """Bill dim differs from voter ideal-point dim -> DimensionMismatchError."""
        pfrandom.set_seed(42)
        model = SequentialCongressModel()

        # Create voter with 2-dimensional ideal point layer
        space_2d = PolicySpace(2)
        space_2d.set_position([0.3, 0.7])
        sq_2d = PolicySpace(2)
        sq_2d.set_position([0.5, 0.5])
        ip_layer = IdealPointLayer(space=space_2d, status_quo=sq_2d)

        voter = SequentialVoter()
        voter.add_layer(ip_layer)
        model.add_congressman(voter)

        # 3-dimensional bill position -- dimension mismatch with 2D voter
        bill = SequentialBill(position=[0.5, 0.5, 0.5])
        with pytest.raises(DimensionMismatchError):
            model.cast_votes(bill)

    def test_matching_dimensions_do_not_raise(self) -> None:
        """Bill dim matches voter ideal-point dim -> no error."""
        pfrandom.set_seed(42)
        model = SequentialCongressModel()

        space_2d = PolicySpace(2)
        space_2d.set_position([0.3, 0.7])
        sq_2d = PolicySpace(2)
        sq_2d.set_position([0.5, 0.5])
        ip_layer = IdealPointLayer(space=space_2d, status_quo=sq_2d)

        voter = SequentialVoter()
        voter.add_layer(ip_layer)
        model.add_congressman(voter)

        bill = SequentialBill(position=[0.5, 0.5])
        # Should not raise
        votes = model.cast_votes(bill)
        assert isinstance(votes, int)
