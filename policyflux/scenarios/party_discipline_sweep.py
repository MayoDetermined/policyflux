"""Scenario: Party Discipline Sweep.

Research question
-----------------
How does **party-whip discipline strength** affect bill-passage rates and
vote predictability, and does the direction of the party line (pro-bill
vs. anti-bill) interact with discipline level?

This scenario sweeps ``party_discipline_strength`` from 0 (no enforced
discipline) to 1 (full whip control), running two parallel series:

- **Pro-bill party line** (``party_line_support = 0.7``) - whip pushes
  members toward supporting the bill.
- **Anti-bill party line** (``party_line_support = 0.3``) - whip pushes
  members toward opposing the bill.

At each discipline level the scenario records average votes cast in favour,
the passage rate, and the standard deviation of per-iteration vote counts.
A falling standard deviation indicates that discipline is reducing random
individual variation (i.e. making outcomes more predictable).

Usage
-----
::

    from policyflux.scenarios import party_discipline_sweep

    results = party_discipline_sweep.run()

    # Custom party positions
    results = party_discipline_sweep.run(
        pro_support=0.8,
        anti_support=0.2,
        n_steps=15,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DisciplinePoint:
    """Results at one discipline-strength level and one party-line stance."""

    discipline_strength: float
    """Party discipline strength in ``[0, 1]``."""

    party_line_support: float
    """Party-line position: ``> 0.5`` = pro-bill, ``< 0.5`` = anti-bill."""

    avg_votes_for: float
    passage_rate: float
    vote_std: float
    num_actors: int
    iterations: int

    @property
    def stance_label(self) -> str:
        return "pro-bill" if self.party_line_support > 0.5 else "anti-bill"

    @property
    def avg_vote_share(self) -> float:
        return self.avg_votes_for / self.num_actors if self.num_actors else 0.0


def _sweep(
    party_line_support: float,
    discipline_levels: list[float],
    num_actors: int,
    policy_dim: int,
    iterations: int,
    seed: int,
) -> list[DisciplinePoint]:
    from ..core.abstract_executive import ExecutiveType
    from ..integration.builders.engine_builder import build_engine
    from ..integration.config import AdvancedActorsConfig, IntegrationConfig, LayerConfig

    threshold = num_actors / 2
    points: list[DisciplinePoint] = []

    for strength in discipline_levels:
        config = IntegrationConfig(
            num_actors=num_actors,
            policy_dim=policy_dim,
            iterations=iterations,
            seed=seed,
            layer_config=LayerConfig(
                include_ideal_point=True,
                include_public_opinion=False,
                include_lobbying=False,
                include_media_pressure=False,
                include_party_discipline=True,
                party_line_support=party_line_support,
                party_discipline_strength=strength,
            ),
            actors_config=AdvancedActorsConfig(
                executive_type=ExecutiveType.PARLIAMENTARY,
                pm_party_strength=0.55,
            ),
        )

        votes: list[int] = build_engine(config).run()
        n = len(votes)
        avg = sum(votes) / n if n else 0.0
        passage_rate = sum(1 for v in votes if v > threshold) / n if n else 0.0
        variance = sum((v - avg) ** 2 for v in votes) / n if n > 1 else 0.0

        points.append(
            DisciplinePoint(
                discipline_strength=strength,
                party_line_support=party_line_support,
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
    n_steps: int = 10,
    pro_support: float = 0.7,
    anti_support: float = 0.3,
) -> dict[str, list[DisciplinePoint]]:
    """Run the party-discipline sweep scenario.

    Parameters
    ----------
    num_actors:
        Number of legislators.
    policy_dim:
        Dimensionality of the policy space.
    iterations:
        Monte Carlo iterations per discipline level.
    seed:
        Random seed (constant across all levels).
    n_steps:
        Number of evenly-spaced discipline levels from 0.0 to 1.0.
    pro_support:
        ``party_line_support`` value used for the "pro-bill" series.
    anti_support:
        ``party_line_support`` value used for the "anti-bill" series.

    Returns
    -------
    dict[str, list[DisciplinePoint]]
        Keys ``"pro"`` and ``"anti"``; each maps to a list of
        :class:`DisciplinePoint` objects ordered by discipline strength.
    """
    levels = [i / max(n_steps - 1, 1) for i in range(n_steps)]

    pro_series = _sweep(pro_support, levels, num_actors, policy_dim, iterations, seed)
    anti_series = _sweep(anti_support, levels, num_actors, policy_dim, iterations, seed)

    def _col(label: str, series: list[DisciplinePoint]) -> None:
        print(f"\n  Party line: {label}")
        print(f"  {'Strength':>8} {'Avg votes':>9} {'Vote share':>10} {'Passage':>9} {'Std':>7}")
        print("  " + "-" * 50)
        for p in series:
            bar = "#" * int(p.passage_rate * 20)
            print(
                f"  {p.discipline_strength:>8.2f} {p.avg_votes_for:>9.1f} "
                f"{p.avg_vote_share:>9.1%} "
                f"{p.passage_rate:>8.1%} "
                f"{p.vote_std:>7.1f}  {bar}"
            )

    print("Party Discipline Sweep")
    print("=" * 56)
    print(f"Actors: {num_actors}  |  Policy dim: {policy_dim}  |  Iterations: {iterations}")
    _col(f"pro-bill  (support={pro_support})", pro_series)
    _col(f"anti-bill (support={anti_support})", anti_series)

    return {"pro": pro_series, "anti": anti_series}


if __name__ == "__main__":
    run()
