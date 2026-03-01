"""End-to-end smoke tests for all 3 political systems.

Each test builds a small config, constructs an engine via ``build_engine``,
runs the simulation, and verifies the basic shape of results.
"""


from policyflux.integration import build_engine
from policyflux.integration.config import (
    IntegrationConfig,
)
from policyflux.integration.presets import (
    create_parliamentary_config,
    create_presidential_config,
    create_semi_presidential_config,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SMALL_NUM_ACTORS = 5
_SMALL_POLICY_DIM = 2
_SMALL_ITERATIONS = 3


def _run_and_verify(config: IntegrationConfig) -> list[int]:
    """Build engine from config, run, and assert basic invariants."""
    engine = build_engine(config)
    results = engine.run()

    assert isinstance(results, list)
    assert len(results) == config.iterations
    for v in results:
        assert isinstance(v, int)
        assert 0 <= v <= config.num_actors

    return results


# ===========================================================================
# Presidential system
# ===========================================================================

class TestSmokePresidentialSystem:
    def test_build_and_run(self) -> None:
        config = create_presidential_config(
            num_actors=_SMALL_NUM_ACTORS,
            policy_dim=_SMALL_POLICY_DIM,
            iterations=_SMALL_ITERATIONS,
            seed=42,
        )
        _run_and_verify(config)

    def test_reproducibility(self) -> None:
        cfg_a = create_presidential_config(
            num_actors=_SMALL_NUM_ACTORS,
            policy_dim=_SMALL_POLICY_DIM,
            iterations=_SMALL_ITERATIONS,
            seed=123,
        )
        cfg_b = create_presidential_config(
            num_actors=_SMALL_NUM_ACTORS,
            policy_dim=_SMALL_POLICY_DIM,
            iterations=_SMALL_ITERATIONS,
            seed=123,
        )
        results_a = build_engine(cfg_a).run()
        results_b = build_engine(cfg_b).run()
        assert results_a == results_b

    def test_custom_approval_and_veto(self) -> None:
        config = create_presidential_config(
            num_actors=_SMALL_NUM_ACTORS,
            policy_dim=_SMALL_POLICY_DIM,
            iterations=_SMALL_ITERATIONS,
            seed=42,
            president_approval=0.9,
            veto_override_threshold=0.8,
        )
        _run_and_verify(config)


# ===========================================================================
# Parliamentary system
# ===========================================================================

class TestSmokeParliamentarySystem:
    def test_build_and_run(self) -> None:
        config = create_parliamentary_config(
            num_actors=_SMALL_NUM_ACTORS,
            policy_dim=_SMALL_POLICY_DIM,
            iterations=_SMALL_ITERATIONS,
            seed=42,
        )
        _run_and_verify(config)

    def test_reproducibility(self) -> None:
        cfg_a = create_parliamentary_config(
            num_actors=_SMALL_NUM_ACTORS,
            policy_dim=_SMALL_POLICY_DIM,
            iterations=_SMALL_ITERATIONS,
            seed=55,
        )
        cfg_b = create_parliamentary_config(
            num_actors=_SMALL_NUM_ACTORS,
            policy_dim=_SMALL_POLICY_DIM,
            iterations=_SMALL_ITERATIONS,
            seed=55,
        )
        results_a = build_engine(cfg_a).run()
        results_b = build_engine(cfg_b).run()
        assert results_a == results_b

    def test_custom_pm_params(self) -> None:
        config = create_parliamentary_config(
            num_actors=_SMALL_NUM_ACTORS,
            policy_dim=_SMALL_POLICY_DIM,
            iterations=_SMALL_ITERATIONS,
            seed=42,
            pm_party_strength=0.7,
            confidence_threshold=0.6,
        )
        _run_and_verify(config)


# ===========================================================================
# Semi-presidential system
# ===========================================================================

class TestSmokeSemiPresidentialSystem:
    def test_build_and_run(self) -> None:
        config = create_semi_presidential_config(
            num_actors=_SMALL_NUM_ACTORS,
            policy_dim=_SMALL_POLICY_DIM,
            iterations=_SMALL_ITERATIONS,
            seed=42,
        )
        _run_and_verify(config)

    def test_reproducibility(self) -> None:
        cfg_a = create_semi_presidential_config(
            num_actors=_SMALL_NUM_ACTORS,
            policy_dim=_SMALL_POLICY_DIM,
            iterations=_SMALL_ITERATIONS,
            seed=88,
        )
        cfg_b = create_semi_presidential_config(
            num_actors=_SMALL_NUM_ACTORS,
            policy_dim=_SMALL_POLICY_DIM,
            iterations=_SMALL_ITERATIONS,
            seed=88,
        )
        results_a = build_engine(cfg_a).run()
        results_b = build_engine(cfg_b).run()
        assert results_a == results_b

    def test_cohabitation_params(self) -> None:
        """Run with params that trigger cohabitation (low pres, high PM)."""
        config = create_semi_presidential_config(
            num_actors=_SMALL_NUM_ACTORS,
            policy_dim=_SMALL_POLICY_DIM,
            iterations=_SMALL_ITERATIONS,
            seed=42,
            president_approval=0.3,
            pm_party_strength=0.7,
        )
        _run_and_verify(config)

    def test_no_cohabitation_params(self) -> None:
        """Run with params that avoid cohabitation (high pres, low PM)."""
        config = create_semi_presidential_config(
            num_actors=_SMALL_NUM_ACTORS,
            policy_dim=_SMALL_POLICY_DIM,
            iterations=_SMALL_ITERATIONS,
            seed=42,
            president_approval=0.8,
            pm_party_strength=0.3,
        )
        _run_and_verify(config)
