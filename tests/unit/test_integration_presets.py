"""Tests for policyflux.integration.presets (all 3 political system presets)."""

import pytest

from policyflux.core.executive import ExecutiveType
from policyflux.integration.config import IntegrationConfig, LayerConfig
from policyflux.integration.presets import (
    create_parliamentary_config,
    create_presidential_config,
    create_semi_presidential_config,
)


# ===========================================================================
# Presidential preset
# ===========================================================================

class TestPresidentialPreset:
    def test_returns_integration_config(self) -> None:
        config = create_presidential_config()
        assert isinstance(config, IntegrationConfig)

    def test_default_values(self) -> None:
        config = create_presidential_config()
        assert config.num_actors == 100
        assert config.policy_dim == 4
        assert config.iterations == 300
        assert config.seed == 42
        assert config.actors_config.executive_type == ExecutiveType.PRESIDENTIAL
        assert config.actors_config.president_approval_rating == pytest.approx(0.5)
        assert config.actors_config.veto_override_threshold == pytest.approx(2 / 3)

    def test_custom_overrides(self) -> None:
        config = create_presidential_config(
            num_actors=50,
            policy_dim=3,
            iterations=100,
            seed=99,
            president_approval=0.8,
            veto_override_threshold=0.75,
        )
        assert config.num_actors == 50
        assert config.policy_dim == 3
        assert config.iterations == 100
        assert config.seed == 99
        assert config.actors_config.president_approval_rating == pytest.approx(0.8)
        assert config.actors_config.veto_override_threshold == pytest.approx(0.75)

    def test_layer_config_defaults(self) -> None:
        config = create_presidential_config()
        assert config.layer_config.include_ideal_point is True
        assert config.layer_config.include_public_opinion is True


# ===========================================================================
# Parliamentary preset
# ===========================================================================

class TestParliamentaryPreset:
    def test_returns_integration_config(self) -> None:
        config = create_parliamentary_config()
        assert isinstance(config, IntegrationConfig)

    def test_default_values(self) -> None:
        config = create_parliamentary_config()
        assert config.num_actors == 100
        assert config.policy_dim == 4
        assert config.actors_config.executive_type == ExecutiveType.PARLIAMENTARY
        assert config.actors_config.pm_party_strength == pytest.approx(0.55)
        assert config.actors_config.confidence_threshold == pytest.approx(0.5)

    def test_government_agenda_enabled(self) -> None:
        config = create_parliamentary_config()
        assert config.layer_config.include_government_agenda is True

    def test_government_agenda_pm_strength_matches(self) -> None:
        config = create_parliamentary_config(pm_party_strength=0.7)
        assert config.layer_config.government_agenda_pm_strength == pytest.approx(0.7)

    def test_custom_overrides(self) -> None:
        config = create_parliamentary_config(
            num_actors=200,
            policy_dim=5,
            pm_party_strength=0.65,
            confidence_threshold=0.6,
            government_bill_rate=0.8,
        )
        assert config.num_actors == 200
        assert config.policy_dim == 5
        assert config.actors_config.pm_party_strength == pytest.approx(0.65)
        assert config.actors_config.confidence_threshold == pytest.approx(0.6)
        assert config.actors_config.government_bill_rate == pytest.approx(0.8)


# ===========================================================================
# Semi-presidential preset
# ===========================================================================

class TestSemiPresidentialPreset:
    def test_returns_integration_config(self) -> None:
        config = create_semi_presidential_config()
        assert isinstance(config, IntegrationConfig)

    def test_default_values(self) -> None:
        config = create_semi_presidential_config()
        assert config.num_actors == 100
        assert config.policy_dim == 4
        assert config.actors_config.executive_type == ExecutiveType.SEMI_PRESIDENTIAL
        assert config.actors_config.semi_presidential_approval_rating == pytest.approx(0.5)
        assert config.actors_config.semi_presidential_pm_party_strength == pytest.approx(0.55)

    def test_custom_overrides(self) -> None:
        config = create_semi_presidential_config(
            num_actors=80,
            policy_dim=2,
            president_approval=0.4,
            pm_party_strength=0.7,
        )
        assert config.num_actors == 80
        assert config.policy_dim == 2
        assert config.actors_config.semi_presidential_approval_rating == pytest.approx(0.4)
        assert config.actors_config.semi_presidential_pm_party_strength == pytest.approx(0.7)

    def test_seed_override(self) -> None:
        config = create_semi_presidential_config(seed=123)
        assert config.seed == 123
