"""One-Liner presets - ready-to-use simulation models.

Each ``run_*`` function creates, builds, and executes a complete simulation
in a single call, returning the raw vote results.

Each ``*_engine`` function creates and returns a fully configured
:class:`~policyflux.engines.Engine` instance ready to call ``.run()`` on.

Pre-built default :class:`~policyflux.integration.IntegrationConfig` constants
(``PRESIDENTIAL_DEFAULT``, ``PARLIAMENTARY_DEFAULT``, ``SEMI_PRESIDENTIAL_DEFAULT``)
can be passed directly to :func:`~policyflux.integration.build_engine`.

Examples
--------
Minimal presidential simulation::

    from policyflux import run_presidential
    results = run_presidential()

Get a reusable engine::

    from policyflux import parliamentary_engine
    engine = parliamentary_engine(num_actors=150, seed=7)
    results = engine.run()

Use a named default config::

    from policyflux import build_engine, SEMI_PRESIDENTIAL_DEFAULT
    engine = build_engine(SEMI_PRESIDENTIAL_DEFAULT)
    results = engine.run()
"""

from __future__ import annotations

from typing import Any

from ..config import IntegrationConfig
from .basic_parliament_preset import create_parliamentary_config
from .president_preset import create_presidential_config
from .semi_presidential_preset import create_semi_presidential_config

# ---------------------------------------------------------------------------
# Default configs (module-level constants - zero-argument ready-to-use)
# ---------------------------------------------------------------------------

#: Default configuration for a presidential system (US-style).
PRESIDENTIAL_DEFAULT: IntegrationConfig = create_presidential_config()

#: Default configuration for a parliamentary system (Westminster-style).
PARLIAMENTARY_DEFAULT: IntegrationConfig = create_parliamentary_config()

#: Default configuration for a semi-presidential system (France/Poland-style).
SEMI_PRESIDENTIAL_DEFAULT: IntegrationConfig = create_semi_presidential_config()


# ---------------------------------------------------------------------------
# Engine builders (return a ready Engine, not yet run)
# ---------------------------------------------------------------------------


def presidential_engine(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    president_approval: float = 0.5,
    veto_override_threshold: float = 2 / 3,
    **kwargs: Any,
) -> Any:
    """Build a ready-to-run presidential-system engine.

    Parameters
    ----------
    num_actors:
        Number of congress members.
    policy_dim:
        Policy space dimensionality.
    iterations:
        Monte Carlo iterations.
    seed:
        Random seed.
    president_approval:
        Presidential approval rating ``[0, 1]``.
    veto_override_threshold:
        Vote share required to override a presidential veto (default ``2/3``).
    **kwargs:
        Additional overrides forwarded to :func:`create_presidential_config`.

    Returns
    -------
    Engine
        Configured engine; call ``.run()`` to execute the simulation.
    """
    from ..builders.engine_builder import build_engine

    config = create_presidential_config(
        num_actors=num_actors,
        policy_dim=policy_dim,
        iterations=iterations,
        seed=seed,
        president_approval=president_approval,
        veto_override_threshold=veto_override_threshold,
        **kwargs,
    )
    return build_engine(config)


def parliamentary_engine(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    pm_party_strength: float = 0.55,
    confidence_threshold: float = 0.5,
    government_bill_rate: float = 0.7,
    **kwargs: Any,
) -> Any:
    """Build a ready-to-run parliamentary-system engine.

    Parameters
    ----------
    num_actors:
        Number of MPs.
    policy_dim:
        Policy space dimensionality.
    iterations:
        Monte Carlo iterations.
    seed:
        Random seed.
    pm_party_strength:
        Prime Minister's party strength ``[0, 1]``.
    confidence_threshold:
        Threshold for confidence votes.
    government_bill_rate:
        Proportion of bills that are government bills.
    **kwargs:
        Additional overrides forwarded to :func:`create_parliamentary_config`.

    Returns
    -------
    Engine
        Configured engine; call ``.run()`` to execute the simulation.
    """
    from ..builders.engine_builder import build_engine

    config = create_parliamentary_config(
        num_actors=num_actors,
        policy_dim=policy_dim,
        iterations=iterations,
        seed=seed,
        pm_party_strength=pm_party_strength,
        confidence_threshold=confidence_threshold,
        government_bill_rate=government_bill_rate,
        **kwargs,
    )
    return build_engine(config)


def semi_presidential_engine(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    president_approval: float = 0.5,
    pm_party_strength: float = 0.55,
    **kwargs: Any,
) -> Any:
    """Build a ready-to-run semi-presidential-system engine.

    Parameters
    ----------
    num_actors:
        Number of representatives.
    policy_dim:
        Policy space dimensionality.
    iterations:
        Monte Carlo iterations.
    seed:
        Random seed.
    president_approval:
        Presidential approval rating ``[0, 1]``.
    pm_party_strength:
        Prime Minister's party strength ``[0, 1]``.
    **kwargs:
        Additional overrides forwarded to :func:`create_semi_presidential_config`.

    Returns
    -------
    Engine
        Configured engine; call ``.run()`` to execute the simulation.
    """
    from ..builders.engine_builder import build_engine

    config = create_semi_presidential_config(
        num_actors=num_actors,
        policy_dim=policy_dim,
        iterations=iterations,
        seed=seed,
        president_approval=president_approval,
        pm_party_strength=pm_party_strength,
        **kwargs,
    )
    return build_engine(config)


# ---------------------------------------------------------------------------
# Full run helpers (create + build + run in one call)
# ---------------------------------------------------------------------------


def run_presidential(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    president_approval: float = 0.5,
    veto_override_threshold: float = 2 / 3,
    **kwargs: Any,
) -> list[int]:
    """Run a complete presidential-system simulation and return vote results.

    Parameters
    ----------
    num_actors:
        Number of congress members.
    policy_dim:
        Policy space dimensionality.
    iterations:
        Monte Carlo iterations.
    seed:
        Random seed.
    president_approval:
        Presidential approval rating ``[0, 1]``.
    veto_override_threshold:
        Vote share required to override a presidential veto (default ``2/3``).
    **kwargs:
        Additional overrides forwarded to :func:`create_presidential_config`.

    Returns
    -------
    list
        Raw vote results from the Monte Carlo engine.
    """
    result: list[int] = presidential_engine(
        num_actors=num_actors,
        policy_dim=policy_dim,
        iterations=iterations,
        seed=seed,
        president_approval=president_approval,
        veto_override_threshold=veto_override_threshold,
        **kwargs,
    ).run()
    return result


def run_parliamentary(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    pm_party_strength: float = 0.55,
    confidence_threshold: float = 0.5,
    government_bill_rate: float = 0.7,
    **kwargs: Any,
) -> list[int]:
    """Run a complete parliamentary-system simulation and return vote results.

    Parameters
    ----------
    num_actors:
        Number of MPs.
    policy_dim:
        Policy space dimensionality.
    iterations:
        Monte Carlo iterations.
    seed:
        Random seed.
    pm_party_strength:
        Prime Minister's party strength ``[0, 1]``.
    confidence_threshold:
        Threshold for confidence votes.
    government_bill_rate:
        Proportion of bills that are government bills.
    **kwargs:
        Additional overrides forwarded to :func:`create_parliamentary_config`.

    Returns
    -------
    list
        Raw vote results from the Monte Carlo engine.
    """
    result: list[int] = parliamentary_engine(
        num_actors=num_actors,
        policy_dim=policy_dim,
        iterations=iterations,
        seed=seed,
        pm_party_strength=pm_party_strength,
        confidence_threshold=confidence_threshold,
        government_bill_rate=government_bill_rate,
        **kwargs,
    ).run()
    return result


def run_semi_presidential(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    president_approval: float = 0.5,
    pm_party_strength: float = 0.55,
    **kwargs: Any,
) -> list[int]:
    """Run a complete semi-presidential-system simulation and return vote results.

    Parameters
    ----------
    num_actors:
        Number of representatives.
    policy_dim:
        Policy space dimensionality.
    iterations:
        Monte Carlo iterations.
    seed:
        Random seed.
    president_approval:
        Presidential approval rating ``[0, 1]``.
    pm_party_strength:
        Prime Minister's party strength ``[0, 1]``.
    **kwargs:
        Additional overrides forwarded to :func:`create_semi_presidential_config`.

    Returns
    -------
    list
        Raw vote results from the Monte Carlo engine.
    """
    result: list[int] = semi_presidential_engine(
        num_actors=num_actors,
        policy_dim=policy_dim,
        iterations=iterations,
        seed=seed,
        president_approval=president_approval,
        pm_party_strength=pm_party_strength,
        **kwargs,
    ).run()
    return result
