from policyflux.integration.config import AdvancedActorsConfig


def test_new_semi_presidential_fields_are_available() -> None:
    config = AdvancedActorsConfig(
        semi_presidential_approval_rating=0.61,
        semi_presidential_pm_party_strength=0.57,
    )

    assert config.semi_presidential_approval_rating == 0.61
    assert config.semi_presidential_pm_party_strength == 0.57


def test_legacy_alias_fields_are_mapped_to_new_fields() -> None:
    config = AdvancedActorsConfig(
        semi_president_approval=0.66,
        semi_pm_party_strength=0.52,
    )

    assert config.semi_presidential_approval_rating == 0.66
    assert config.semi_presidential_pm_party_strength == 0.52


def test_legacy_aliases_remain_readable_after_initialization() -> None:
    config = AdvancedActorsConfig(
        semi_presidential_approval_rating=0.72,
        semi_presidential_pm_party_strength=0.49,
    )

    assert config.semi_president_approval == 0.72
    assert config.semi_pm_party_strength == 0.49
