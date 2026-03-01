"""Tests for policyflux.engines.sequential_monte_carlo.SequentialMonteCarlo."""

import pytest

import policyflux.pfrandom as pfrandom
from policyflux.integration import (
    AdvancedActorsConfig,
    IntegrationConfig,
    LayerConfig,
    build_engine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_config(
    num_actors: int = 5,
    policy_dim: int = 2,
    iterations: int = 10,
    seed: int = 42,
) -> IntegrationConfig:
    return IntegrationConfig(
        num_actors=num_actors,
        policy_dim=policy_dim,
        iterations=iterations,
        seed=seed,
        layer_config=LayerConfig(
            include_ideal_point=True,
            include_public_opinion=True,
            include_lobbying=False,
            include_media_pressure=False,
            include_party_discipline=False,
            include_government_agenda=False,
            include_neural=False,
        ),
        actors_config=AdvancedActorsConfig(
            n_lobbyists=0,
            n_whips=0,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSequentialMonteCarlo:
    def test_run_returns_list_of_correct_length(self) -> None:
        config = _make_small_config(iterations=15, seed=42)
        engine = build_engine(config)
        results = engine.run()
        assert isinstance(results, list)
        assert len(results) == 15

    def test_results_contain_ints_in_range(self) -> None:
        config = _make_small_config(num_actors=8, iterations=20, seed=42)
        engine = build_engine(config)
        results = engine.run()
        for v in results:
            assert isinstance(v, int)
            assert 0 <= v <= 8

    def test_same_seed_produces_same_results(self) -> None:
        config_a = _make_small_config(seed=77)
        config_b = _make_small_config(seed=77)
        results_a = build_engine(config_a).run()
        results_b = build_engine(config_b).run()
        assert results_a == results_b

    def test_different_seeds_produce_different_results(self) -> None:
        config_a = _make_small_config(seed=1)
        config_b = _make_small_config(seed=999)
        results_a = build_engine(config_a).run()
        results_b = build_engine(config_b).run()
        # Extremely unlikely to be equal with different seeds
        assert results_a != results_b

    def test_str_before_run(self) -> None:
        config = _make_small_config()
        engine = build_engine(config)
        s = str(engine)
        assert "No simulations run yet" in s

    def test_str_after_run(self) -> None:
        config = _make_small_config(iterations=5, seed=42)
        engine = build_engine(config)
        engine.run()
        s = str(engine)
        assert "Simulations" in s
        assert "Average votes for" in s
        assert "Average votes against" in s

    def test_results_stored_on_engine(self) -> None:
        config = _make_small_config(iterations=5, seed=42)
        engine = build_engine(config)
        returned = engine.run()
        assert engine.results == returned

    def test_single_iteration(self) -> None:
        config = _make_small_config(iterations=1, seed=42)
        engine = build_engine(config)
        results = engine.run()
        assert len(results) == 1
