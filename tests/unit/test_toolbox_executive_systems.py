"""Tests for policyflux.toolbox.executive_systems."""

import pytest

from policyflux.core.abstract_executive import ExecutiveType
from policyflux.core.pf_typing import PolicySpace
from policyflux.toolbox.bill_models import SequentialBill
from policyflux.toolbox.executive_systems import (
    ParliamentaryExecutive,
    President,
    PresidentialExecutive,
    PrimeMinister,
    SemiPresidentialExecutive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_president(
    approval: float = 0.5,
    ideology_pos: list[float] | None = None,
    dim: int = 2,
) -> President:
    ideology = PolicySpace(dim)
    if ideology_pos is not None:
        ideology.set_position(ideology_pos)
    else:
        ideology.set_position([0.5] * dim)
    return President(approval_rating=approval, ideology=ideology)


def _make_pm(party_strength: float = 0.55) -> PrimeMinister:
    return PrimeMinister(party_strength=party_strength)


# ===========================================================================
# PresidentialExecutive
# ===========================================================================

class TestPresidentialExecutive:
    def test_type_is_presidential(self) -> None:
        pres = _make_president()
        executive = PresidentialExecutive(pres)
        assert executive.executive_type == ExecutiveType.PRESIDENTIAL

    def test_get_primary_actor(self) -> None:
        pres = _make_president()
        executive = PresidentialExecutive(pres)
        assert executive.get_primary_actor() is pres

    def test_inject_context_adds_keys(self) -> None:
        pres = _make_president(approval=0.7)
        executive = PresidentialExecutive(pres)
        ctx = executive.inject_context({})
        assert "president" in ctx
        assert ctx["president"] is pres
        assert "president_approval" in ctx
        assert ctx["president_approval"] == pytest.approx(0.7)
        assert "executive_influence" in ctx

    def test_veto_not_triggered_when_bill_fails(self) -> None:
        """Bill that fails the vote (votes_for <= total/2) is returned as-is."""
        pres = _make_president(approval=0.2, ideology_pos=[0.0, 0.0])
        executive = PresidentialExecutive(pres)
        bill = SequentialBill(position=[1.0, 1.0])

        # 3 out of 10 -- bill already failed, no veto needed
        result = executive.process_bill_result(bill, votes_for=3, total_votes=10)
        assert result == 3

    def test_veto_sustained_high_distance_low_approval(self) -> None:
        """President with low approval and bill far from ideology -> veto sustained."""
        # Low approval means low tolerance (threshold = 0.3 + 0.5*0.1 = 0.35)
        # Bill at [1.0,1.0] vs president at [0.0,0.0] -> distance = sqrt(2)/sqrt(2) = 1.0 > 0.35
        pres = _make_president(approval=0.1, ideology_pos=[0.0, 0.0])
        executive = PresidentialExecutive(pres, veto_override_threshold=2 / 3)
        bill = SequentialBill(position=[1.0, 1.0])

        # 6 out of 10 -- passes majority but below 2/3 override
        result = executive.process_bill_result(bill, votes_for=6, total_votes=10)
        assert result == 0  # Veto sustained

    def test_veto_overridden_with_supermajority(self) -> None:
        """Supermajority overrides the veto even when president opposes bill."""
        pres = _make_president(approval=0.1, ideology_pos=[0.0, 0.0])
        executive = PresidentialExecutive(pres, veto_override_threshold=2 / 3)
        bill = SequentialBill(position=[1.0, 1.0])

        # 7 out of 10 -- above 2/3 override threshold
        result = executive.process_bill_result(bill, votes_for=7, total_votes=10)
        assert result == 7

    def test_no_veto_when_bill_close_to_ideology(self) -> None:
        """President does not veto bill close to their own ideology."""
        pres = _make_president(approval=0.5, ideology_pos=[0.5, 0.5])
        executive = PresidentialExecutive(pres)
        bill = SequentialBill(position=[0.5, 0.5])

        result = executive.process_bill_result(bill, votes_for=6, total_votes=10)
        assert result == 6

    def test_no_veto_when_bill_has_no_position(self) -> None:
        """No veto when bill has empty position list."""
        pres = _make_president(approval=0.1, ideology_pos=[0.0, 0.0])
        executive = PresidentialExecutive(pres)
        bill = SequentialBill(position=[])

        result = executive.process_bill_result(bill, votes_for=6, total_votes=10)
        assert result == 6


# ===========================================================================
# ParliamentaryExecutive
# ===========================================================================

class TestParliamentaryExecutive:
    def test_type_is_parliamentary(self) -> None:
        pm = _make_pm()
        executive = ParliamentaryExecutive(pm)
        assert executive.executive_type == ExecutiveType.PARLIAMENTARY

    def test_get_primary_actor(self) -> None:
        pm = _make_pm()
        executive = ParliamentaryExecutive(pm)
        assert executive.get_primary_actor() is pm

    def test_inject_context_adds_keys(self) -> None:
        pm = _make_pm(party_strength=0.6)
        executive = ParliamentaryExecutive(pm)
        ctx = executive.inject_context({})
        assert "prime_minister" in ctx
        assert ctx["prime_minister"] is pm
        assert "pm_party_strength" in ctx
        assert ctx["pm_party_strength"] == pytest.approx(0.6)

    def test_inject_context_government_bill_adds_discipline(self) -> None:
        pm = _make_pm()
        executive = ParliamentaryExecutive(pm)
        ctx = executive.inject_context({"is_government_bill": True})
        assert ctx.get("party_discipline_strength") == pytest.approx(0.9)

    def test_confidence_vote_fail_removes_pm(self) -> None:
        """Losing a confidence vote sets PM in_office to False."""
        pm = _make_pm()
        executive = ParliamentaryExecutive(pm, confidence_threshold=0.5)
        assert pm.in_office is True

        bill = SequentialBill()
        bill.is_confidence_vote = True
        # 4 out of 10 <= 10 * 0.5 -> confidence lost
        executive.process_bill_result(bill, votes_for=4, total_votes=10)
        assert pm.in_office is False

    def test_confidence_vote_pass_keeps_pm(self) -> None:
        """Winning a confidence vote keeps PM in office."""
        pm = _make_pm()
        executive = ParliamentaryExecutive(pm, confidence_threshold=0.5)
        assert pm.in_office is True

        bill = SequentialBill()
        bill.is_confidence_vote = True
        # 6 out of 10 > 10 * 0.5 -> confidence retained
        executive.process_bill_result(bill, votes_for=6, total_votes=10)
        assert pm.in_office is True

    def test_process_normal_bill_returns_votes_unchanged(self) -> None:
        pm = _make_pm()
        executive = ParliamentaryExecutive(pm)
        bill = SequentialBill()
        result = executive.process_bill_result(bill, votes_for=7, total_votes=10)
        assert result == 7

    def test_pm_cannot_veto(self) -> None:
        pm = _make_pm()
        bill = SequentialBill()
        assert pm.can_veto_bill(bill) is False


# ===========================================================================
# SemiPresidentialExecutive
# ===========================================================================

class TestSemiPresidentialExecutive:
    def test_type_is_semi_presidential(self) -> None:
        pres = _make_president()
        pm = _make_pm()
        executive = SemiPresidentialExecutive(pres, pm)
        assert executive.executive_type == ExecutiveType.SEMI_PRESIDENTIAL

    def test_cohabitation_detection_true(self) -> None:
        """Low presidential approval + high PM strength = cohabitation."""
        pres = _make_president(approval=0.3)
        pm = _make_pm(party_strength=0.7)
        executive = SemiPresidentialExecutive(pres, pm)
        assert executive.cohabitation is True

    def test_cohabitation_detection_false(self) -> None:
        """High presidential approval + low PM strength = no cohabitation."""
        pres = _make_president(approval=0.7)
        pm = _make_pm(party_strength=0.4)
        executive = SemiPresidentialExecutive(pres, pm)
        assert executive.cohabitation is False

    def test_primary_actor_during_cohabitation_is_pm(self) -> None:
        pres = _make_president(approval=0.3)
        pm = _make_pm(party_strength=0.7)
        executive = SemiPresidentialExecutive(pres, pm)
        assert executive.get_primary_actor() is pm

    def test_primary_actor_without_cohabitation_is_president(self) -> None:
        pres = _make_president(approval=0.7)
        pm = _make_pm(party_strength=0.4)
        executive = SemiPresidentialExecutive(pres, pm)
        assert executive.get_primary_actor() is pres

    def test_inject_context_adds_all_keys(self) -> None:
        pres = _make_president(approval=0.6)
        pm = _make_pm(party_strength=0.5)
        executive = SemiPresidentialExecutive(pres, pm)
        ctx = executive.inject_context({})
        assert "president" in ctx
        assert "prime_minister" in ctx
        assert "president_approval" in ctx
        assert "cohabitation" in ctx
        assert "executive_influence" in ctx

    def test_confidence_vote_fail_removes_pm(self) -> None:
        pres = _make_president(approval=0.6)
        pm = _make_pm(party_strength=0.6)
        executive = SemiPresidentialExecutive(pres, pm)

        bill = SequentialBill()
        bill.is_confidence_vote = True
        # 4 out of 10 <= 10*0.5 => PM loses
        executive.process_bill_result(bill, votes_for=4, total_votes=10)
        assert pm.in_office is False

    def test_no_veto_during_cohabitation(self) -> None:
        """During cohabitation, president cannot exercise veto."""
        pres = _make_president(approval=0.3, ideology_pos=[0.0, 0.0])
        pm = _make_pm(party_strength=0.7)
        executive = SemiPresidentialExecutive(pres, pm)
        assert executive.cohabitation is True

        bill = SequentialBill(position=[1.0, 1.0])
        result = executive.process_bill_result(bill, votes_for=6, total_votes=10)
        assert result == 6  # No veto because of cohabitation

    def test_bill_already_failed_returns_unchanged(self) -> None:
        pres = _make_president(approval=0.7)
        pm = _make_pm(party_strength=0.4)
        executive = SemiPresidentialExecutive(pres, pm)

        bill = SequentialBill(position=[1.0, 1.0])
        result = executive.process_bill_result(bill, votes_for=3, total_votes=10)
        assert result == 3
