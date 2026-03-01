"""PolicyFlux - ready-made research scenarios.

Each submodule poses a concrete political-science research question and
provides a ``run(**kwargs)`` function that executes the simulation,
prints a structured summary, and returns typed results.

Available scenarios
-------------------
- :mod:`~policyflux.scenarios.comparative_systems`
    How do presidential, parliamentary, and semi-presidential systems
    differ in average bill-passage rates?

- :mod:`~policyflux.scenarios.lobbying_sweep`
    How does increasing lobbying intensity shift legislative outcomes?

- :mod:`~policyflux.scenarios.party_discipline_sweep`
    How does party-whip discipline affect passage rates and vote variance?

- :mod:`~policyflux.scenarios.country_comparison`
    Passage-rate comparison across real-world multi-chamber parliaments
    (UK, US, Germany, France, Italy, Poland, Sweden, Spain, Australia,
    Canada).

- :mod:`~policyflux.scenarios.veto_player_sweep`
    How does presidential approval / semi-presidential cohabitation
    tension affect bill passage as the executive grows stronger?

Quick start
-----------
::

    from policyflux.scenarios import comparative_systems
    results = comparative_systems.run()

    from policyflux.scenarios import country_comparison
    results = country_comparison.run(n_bills=50)

    # Run every scenario with defaults
    from policyflux.scenarios import run_all
    run_all()
"""

from __future__ import annotations

from . import (
    comparative_systems,
    country_comparison,
    lobbying_sweep,
    party_discipline_sweep,
    veto_player_sweep,
)

__all__ = [
    "comparative_systems",
    "country_comparison",
    "lobbying_sweep",
    "party_discipline_sweep",
    "run_all",
    "veto_player_sweep",
]


def run_all(**kwargs: object) -> None:
    """Run every built-in research scenario with default parameters.

    Parameters
    ----------
    **kwargs:
        Common overrides forwarded to every scenario (e.g. ``seed=0``).
        Scenario-specific parameters are ignored silently if not accepted.
    """
    _divider = "=" * 60

    for module in (
        comparative_systems,
        lobbying_sweep,
        party_discipline_sweep,
        country_comparison,
        veto_player_sweep,
    ):
        print(_divider)
        # Pass only kwargs that the scenario's run() accepts
        import inspect

        sig = inspect.signature(module.run)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        module.run(**accepted)
        print()
