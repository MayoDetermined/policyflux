from policyflux.integration import (
    AdvancedActorsConfig,
    IntegrationConfig,
    LayerConfig,
    build_engine,
)


def _make_small_config(seed: int = 123) -> IntegrationConfig:
    return IntegrationConfig(
        num_actors=8,
        policy_dim=3,
        iterations=5,
        seed=seed,
        layer_config=LayerConfig(
            include_ideal_point=True,
            include_public_opinion=True,
            include_lobbying=False,
            include_media_pressure=True,
            include_party_discipline=False,
            include_government_agenda=False,
            include_neural=False,
        ),
        actors_config=AdvancedActorsConfig(
            n_lobbyists=0,
            n_whips=0,
        ),
    )


def test_smoke_build_engine_and_run() -> None:
    config = _make_small_config(seed=21)
    engine = build_engine(config)

    results = engine.run()

    assert isinstance(results, list)
    assert len(results) == config.iterations
    assert all(0 <= votes <= config.num_actors for votes in results)


def test_smoke_engine_run_is_reproducible_with_same_seed() -> None:
    config_a = _make_small_config(seed=77)
    config_b = _make_small_config(seed=77)

    results_a = build_engine(config_a).run()
    results_b = build_engine(config_b).run()

    assert results_a == results_b
