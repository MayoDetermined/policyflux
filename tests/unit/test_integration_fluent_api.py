import pytest

from policyflux.core.abstract_executive import ExecutiveType
from policyflux.integration.config import AdvancedActorsConfig, IntegrationConfig, LayerConfig


def test_integration_config_supports_method_chaining() -> None:
    config = (
        IntegrationConfig()
        .with_simulation(num_actors=120, policy_dim=2, iterations=80, seed=7)
        .with_aggregation("weighted", weights=[0.7, 0.3])
        .with_layers(include_media_pressure=False, public_support=0.61)
        .with_actors(n_lobbyists=3, lobbyist_strength=0.8)
    )

    assert config.num_actors == 120
    assert config.policy_dim == 2
    assert config.iterations == 80
    assert config.seed == 7
    assert config.aggregation_strategy == "weighted"
    assert config.aggregation_weights == [0.7, 0.3]
    assert config.layer_config.include_media_pressure is False
    assert config.layer_config.public_support == 0.61
    assert config.actors_config.n_lobbyists == 3
    assert config.actors_config.lobbyist_strength == 0.8


def test_layer_config_supports_method_chaining() -> None:
    layer_config = (
        LayerConfig()
        .with_layer("public_opinion", enabled=False)
        .with_public_support(0.4)
        .with_lobbying_intensity(0.25)
        .with_party_discipline(line_support=0.58, discipline_strength=0.66)
        .with_layer_override("media", pressure=0.42)
    )

    assert layer_config.include_public_opinion is False
    assert layer_config.public_support == 0.4
    assert layer_config.lobbying_intensity == 0.25
    assert layer_config.party_line_support == 0.58
    assert layer_config.party_discipline_strength == 0.66
    assert layer_config.layer_overrides["media"] == {"pressure": 0.42}


def test_actors_config_supports_method_chaining() -> None:
    actors_config = (
        AdvancedActorsConfig()
        .with_executive_type(ExecutiveType.SEMI_PRESIDENTIAL)
        .with_lobbyists(count=2, strength=0.77, stance=0.65)
        .with_whips(count=1, discipline_strength=0.7)
        .with_presidential(approval_rating=0.55)
        .with_semi_presidential(approval_rating=0.63, pm_party_strength=0.52)
    )

    assert actors_config.executive_type == ExecutiveType.SEMI_PRESIDENTIAL
    assert actors_config.n_lobbyists == 2
    assert actors_config.lobbyist_strength == 0.77
    assert actors_config.lobbyist_stance == 0.65
    assert actors_config.n_whips == 1
    assert actors_config.whip_discipline_strength == 0.7
    assert actors_config.president_approval_rating == 0.55
    assert actors_config.semi_presidential_approval_rating == 0.63
    assert actors_config.semi_president_approval == 0.63
    assert actors_config.semi_presidential_pm_party_strength == 0.52
    assert actors_config.semi_pm_party_strength == 0.52


def test_fluent_api_rejects_unknown_fields() -> None:
    config = IntegrationConfig()

    with pytest.raises(ValueError):
        config.with_layers(not_a_real_field=True)

    with pytest.raises(ValueError):
        config.with_actors(not_a_real_field=True)

    with pytest.raises(ValueError):
        LayerConfig().with_layer("not_a_real_layer", enabled=True)


def test_integration_config_supports_flat_configuration_with_defaults() -> None:
    config = IntegrationConfig.from_flat(
        num_actors=140,
        include_public_opinion=False,
        public_support=0.63,
        n_lobbyists=2,
        lobbyist_strength=0.71,
        aggregation_strategy="average",
    )

    assert config.num_actors == 140
    assert config.policy_dim == 4
    assert config.iterations == 300
    assert config.seed == 42
    assert config.aggregation_strategy == "average"

    assert config.layer_config.include_public_opinion is False
    assert config.layer_config.public_support == 0.63
    assert config.layer_config.include_media_pressure is True

    assert config.actors_config.n_lobbyists == 2
    assert config.actors_config.lobbyist_strength == 0.71
    assert config.actors_config.executive_type == ExecutiveType.PRESIDENTIAL


def test_integration_config_with_flat_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        IntegrationConfig.from_flat(not_a_real_field=True)

    with pytest.raises(ValueError):
        IntegrationConfig().with_flat(not_a_real_field=True)
