import pytest

from policyflux.core.abstract_executive import ExecutiveType
from policyflux.integration.config import AdvancedActorsConfig, IntegrationConfig, LayerConfig
from policyflux.integration.fluent import PolicyFlux


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


def test_policyflux_layer_subbuilder_configures_fields_and_returns_parent() -> None:
    neural_factory = object()

    policy_flux = (
        PolicyFlux.builder()
        .actors(120)
        .policy_dim(3)
        .iterations(55)
        .seed(9)
        .description("fluent test")
    )

    parent = (
        policy_flux.layers()
        .ideal_point(enabled=False)
        .public_opinion(support=0.62)
        .lobbying(intensity=0.33)
        .media_pressure(pressure=0.21)
        .party_discipline(line_support=0.58, strength=0.7)
        .government_agenda(pm_strength=0.64)
        .neural(factory=neural_factory)
        .override("public_opinion", alpha=0.4)
        .override("public_opinion", beta=0.2)
        .names(["public_opinion", "media_pressure"])
        .done()
    )

    assert parent is policy_flux

    config = policy_flux.to_config()
    assert config.num_actors == 120
    assert config.policy_dim == 3
    assert config.iterations == 55
    assert config.seed == 9
    assert config.description == "fluent test"
    assert config.layer_config.include_ideal_point is False
    assert config.layer_config.public_support == 0.62
    assert config.layer_config.lobbying_intensity == 0.33
    assert config.layer_config.media_pressure == 0.21
    assert config.layer_config.party_line_support == 0.58
    assert config.layer_config.party_discipline_strength == 0.7
    assert config.layer_config.government_agenda_pm_strength == 0.64
    assert config.layer_config.include_neural is True
    assert config.layer_config.neural_layer_factory is neural_factory
    assert config.layer_config.layer_overrides["public_opinion"] == {"alpha": 0.4, "beta": 0.2}
    assert config.layer_config.layer_names == ["public_opinion", "media_pressure"]


def test_policyflux_executive_subbuilder_sets_expected_values() -> None:
    policy_flux = PolicyFlux.builder()

    assert (
        policy_flux.executive().presidential(approval_rating=0.72, veto_override=0.61).done()
        is policy_flux
    )

    assert (
        policy_flux.executive()
        .parliamentary(
            pm_party_strength=0.57,
            confidence_threshold=0.45,
            government_bill_rate=0.39,
        )
        .done()
        is policy_flux
    )

    assert (
        policy_flux.executive().semi_presidential(approval_rating=0.68, pm_party_strength=0.52).done()
        is policy_flux
    )

    config = policy_flux.build_config()
    assert config.actors_config.executive_type == ExecutiveType.SEMI_PRESIDENTIAL
    assert config.actors_config.president_approval_rating == 0.72
    assert config.actors_config.veto_override_threshold == 0.61
    assert config.actors_config.pm_party_strength == 0.57
    assert config.actors_config.confidence_threshold == 0.45
    assert config.actors_config.government_bill_rate == 0.39
    assert config.layer_config.include_government_agenda is True
    assert config.layer_config.government_agenda_pm_strength == 0.57
    assert config.actors_config.semi_presidential_approval_rating == 0.68
    assert config.actors_config.semi_president_approval == 0.68
    assert config.actors_config.semi_presidential_pm_party_strength == 0.52
    assert config.actors_config.semi_pm_party_strength == 0.52


def test_policyflux_special_actors_and_flat_aliases() -> None:
    policy_flux = PolicyFlux.builder()

    assert (
        policy_flux.special_actors()
        .lobbyists(3, strength=0.86, stance=0.44)
        .whips(2, discipline_strength=0.7, party_line_support=0.59)
        .speaker(agenda_support=0.66)
        .done()
        is policy_flux
    )

    config = (
        policy_flux.without_ideal_point()
        .with_public_opinion(support=0.41)
        .without_public_opinion()
        .with_lobbying(intensity=0.2)
        .without_lobbying()
        .with_media_pressure(pressure=0.18)
        .without_media_pressure()
        .with_party_discipline(line_support=0.54, strength=0.63)
        .without_party_discipline()
        .with_government_agenda(pm_strength=0.49)
        .without_government_agenda()
        .with_neural(factory=lambda: None)
        .with_layer_override("party_discipline", gamma=0.11)
        .layer_override("party_discipline", delta=0.22)
        .layer_names(["ideal_point"])
        .aggregation("weighted", weights=[0.6, 0.4])
        .lobbyists(4, strength=0.91, stance=0.5)
        .whips(1, discipline_strength=0.75, party_line_support=0.6)
        .speaker(agenda_support=0.7)
        .build_config()
    )

    assert config.actors_config.n_lobbyists == 4
    assert config.actors_config.lobbyist_strength == 0.91
    assert config.actors_config.lobbyist_stance == 0.5
    assert config.actors_config.n_whips == 1
    assert config.actors_config.whip_discipline_strength == 0.75
    assert config.actors_config.whip_party_line_support == 0.6
    assert config.actors_config.speaker_agenda_support == 0.7
    assert config.layer_config.include_ideal_point is False
    assert config.layer_config.include_public_opinion is False
    assert config.layer_config.include_lobbying is False
    assert config.layer_config.include_media_pressure is False
    assert config.layer_config.include_party_discipline is False
    assert config.layer_config.include_government_agenda is False
    assert config.layer_config.include_neural is True
    assert config.layer_config.layer_overrides["party_discipline"] == {
        "gamma": 0.11,
        "delta": 0.22,
    }
    assert config.layer_config.layer_names == ["ideal_point"]
    assert config.aggregation_strategy == "weighted"
    assert config.aggregation_weights == [0.6, 0.4]


def test_policyflux_build_uses_engine_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, IntegrationConfig] = {}
    sentinel_engine = object()

    def _fake_build_engine(config: IntegrationConfig) -> object:
        captured["config"] = config
        return sentinel_engine

    monkeypatch.setattr("policyflux.integration.builders.engine_builder.build_engine", _fake_build_engine)

    result = PolicyFlux.builder().actors(150).policy_dim(5).iterations(7).seed(99).build()

    assert result is sentinel_engine
    assert captured["config"].num_actors == 150
    assert captured["config"].policy_dim == 5
    assert captured["config"].iterations == 7
    assert captured["config"].seed == 99
