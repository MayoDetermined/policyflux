import pytest

from policyflux.core.pf_typing import PolicyPosition
from policyflux.layers.government_agenda import GovernmentAgendaLayer


_BILL = PolicyPosition((0.2, 0.8))


def test_government_bill_blends_base_probability_with_pm_strength() -> None:
    layer = GovernmentAgendaLayer(pm_party_strength=0.8)

    value = layer.call(_BILL, base_prob=0.3, is_government_bill=True)

    expected = 0.3 * 0.1 + 0.8 * 0.9
    assert value == pytest.approx(expected)


def test_private_bill_returns_base_probability() -> None:
    layer = GovernmentAgendaLayer(pm_party_strength=0.9)

    value = layer.call(_BILL, base_prob=0.37)

    assert value == pytest.approx(0.37)


def test_confidence_flag_without_government_bill_keeps_base_prob() -> None:
    layer = GovernmentAgendaLayer(pm_party_strength=0.7)

    value = layer.call(_BILL, base_prob=0.41, is_confidence_vote=True)

    assert value == pytest.approx(0.41)


def test_pm_party_strength_is_clamped_to_unit_interval() -> None:
    low = GovernmentAgendaLayer(pm_party_strength=-10)
    high = GovernmentAgendaLayer(pm_party_strength=10)

    assert low.pm_party_strength == 0.0
    assert high.pm_party_strength == 1.0
