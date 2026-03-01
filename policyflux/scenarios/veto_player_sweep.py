"""Scenario: Veto Player Sweep.

Research question
-----------------
How does an **executive's approval rating** affect legislative outcomes as
it rises from low (politically weak executive) to high (dominant executive)?

This scenario sweeps the executive approval rating from 0.1 to 0.9 across
two executive systems:

- **Presidential** - the president's approval determines how much weight
  legislators give to presidential cues and how sustainable a presidential
  veto is.
- **Semi-presidential** - both the president's approval *and* the PM's
  party strength matter; the sweep varies the president's approval while
  holding the PM's strength constant (cohabitation configuration when
  approval is low vs. unified-executive when it is high).

For each level the scenario records average votes cast in favour, bill
passage rate, and vote-count standard deviation.

Usage
-----
::

    from policyflux.scenarios import veto_player_sweep

    results = veto_player_sweep.run()

    # Finer grid, stronger PM baseline
    results = veto_player_sweep.run(
        n_steps=18,
        pm_party_strength=0.65,
        num_actors=150,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class VetoPoint:
    """Results at one approval-rating level for one executive system."""

    system: str
    """'Presidential' or 'Semi-Presidential'."""

    approval: float
    """Executive approval rating in ``[0, 1]``."""

    pm_party_strength: float
    """PM's party strength (semi-presidential only; 0.0 for presidential)."""

    avg_votes_for: float
    passage_rate: float
    vote_std: float
    num_actors: int
    iterations: int

    @property
    def avg_vote_share(self) -> float:
        return self.avg_votes_for / self.num_actors if self.num_actors else 0.0


def _sweep_system(
    system: str,
    approval_levels: list[float],
    pm_party_strength: float,
    num_actors: int,
    policy_dim: int,
    iterations: int,
    seed: int,
    veto_override_threshold: float,
) -> list[VetoPoint]:
    from ..integration.builders.engine_builder import build_engine
    from ..integration.presets import (
        create_presidential_config,
        create_semi_presidential_config,
    )

    threshold = num_actors / 2
    points: list[VetoPoint] = []

    for approval in approval_levels:
        if system == "Presidential":
            config = create_presidential_config(
                num_actors=num_actors,
                policy_dim=policy_dim,
                iterations=iterations,
                seed=seed,
                president_approval=approval,
                veto_override_threshold=veto_override_threshold,
            )
        else:
            config = create_semi_presidential_config(
                num_actors=num_actors,
                policy_dim=policy_dim,
                iterations=iterations,
                seed=seed,
                president_approval=approval,
                pm_party_strength=pm_party_strength,
            )

        votes: list[int] = build_engine(config).run()
        n = len(votes)
        avg = sum(votes) / n if n else 0.0
        passage_rate = sum(1 for v in votes if v > threshold) / n if n else 0.0
        variance = sum((v - avg) ** 2 for v in votes) / n if n > 1 else 0.0

        points.append(
            VetoPoint(
                system=system,
                approval=approval,
                pm_party_strength=pm_party_strength if system != "Presidential" else 0.0,
                avg_votes_for=avg,
                passage_rate=passage_rate,
                vote_std=math.sqrt(variance),
                num_actors=num_actors,
                iterations=iterations,
            )
        )

    return points


def run(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    n_steps: int = 9,
    pm_party_strength: float = 0.55,
    veto_override_threshold: float = 2 / 3,
    min_approval: float = 0.1,
    max_approval: float = 0.9,
) -> dict[str, list[VetoPoint]]:
    """Run the veto-player approval-rating sweep scenario.

    Parameters
    ----------
    num_actors:
        Number of legislators.
    policy_dim:
        Dimensionality of the policy space.
    iterations:
        Monte Carlo iterations per approval level.
    seed:
        Random seed (constant across all levels).
    n_steps:
        Number of evenly-spaced approval levels from *min_approval* to
        *max_approval*.
    pm_party_strength:
        Prime minister's party strength used in the semi-presidential
        series (a constant baseline).
    veto_override_threshold:
        Share of legislature required to override a presidential veto.
    min_approval:
        Lower bound of the approval sweep.
    max_approval:
        Upper bound of the approval sweep.

    Returns
    -------
    dict[str, list[VetoPoint]]
        Keys ``"presidential"`` and ``"semi_presidential"``; each maps to a
        list of :class:`VetoPoint` objects ordered by ascending approval.
    """
    steps = max(n_steps, 2)
    approval_levels = [
        min_approval + (max_approval - min_approval) * i / (steps - 1) for i in range(steps)
    ]

    presidential = _sweep_system(
        "Presidential",
        approval_levels,
        pm_party_strength=0.0,  # not used
        num_actors=num_actors,
        policy_dim=policy_dim,
        iterations=iterations,
        seed=seed,
        veto_override_threshold=veto_override_threshold,
    )
    semi_presidential = _sweep_system(
        "Semi-Presidential",
        approval_levels,
        pm_party_strength=pm_party_strength,
        num_actors=num_actors,
        policy_dim=policy_dim,
        iterations=iterations,
        seed=seed,
        veto_override_threshold=veto_override_threshold,
    )

    def _print_series(label: str, series: list[VetoPoint]) -> None:
        print(f"\n  {label}")
        print(f"  {'Approval':>8} {'Avg votes':>9} {'Vote share':>10} {'Passage':>9} {'Std':>7}")
        print("  " + "-" * 52)
        for p in series:
            bar = "#" * int(p.passage_rate * 20)
            print(
                f"  {p.approval:>8.2f} {p.avg_votes_for:>9.1f} "
                f"{p.avg_vote_share:>9.1%} "
                f"{p.passage_rate:>8.1%} "
                f"{p.vote_std:>7.1f}  {bar}"
            )

    print("Veto Player - Approval Rating Sweep")
    print("=" * 58)
    print(f"Actors: {num_actors}  |  Policy dim: {policy_dim}  |  Iterations: {iterations}")
    print(
        f"Veto override threshold: {veto_override_threshold:.0%}  |  "
        f"Semi-presidential PM strength: {pm_party_strength}"
    )
    _print_series("Presidential", presidential)
    _print_series(f"Semi-Presidential (PM strength={pm_party_strength})", semi_presidential)

    return {"presidential": presidential, "semi_presidential": semi_presidential}


if __name__ == "__main__":
    run()
