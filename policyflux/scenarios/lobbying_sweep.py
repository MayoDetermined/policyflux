"""Scenario: Lobbying Intensity Sweep.

Research question
-----------------
How does **lobbying intensity** shift legislative outcomes as it increases
from zero (no lobbying) to full dominance?

This scenario holds all other parameters constant (presidential system,
fixed actor count, same seed) and sweeps ``lobbying_intensity`` across
ten equally-spaced levels from 0.0 to 1.0.  At each level it records:

- Average votes cast in favour
- Bill passage rate
- Vote-count standard deviation (a proxy for outcome predictability)

A second optional sweep varies the **number of lobbyist actors** at fixed
intensity to isolate the effect of having more organised lobbying groups.

Usage
-----
::

    from policyflux.scenarios import lobbying_sweep

    results = lobbying_sweep.run()

    # More steps, more actors
    results = lobbying_sweep.run(
        n_steps=20,
        num_actors=200,
        n_lobbyists=5,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LobbyingPoint:
    """Results at one lobbying-intensity level."""

    lobbying_intensity: float
    """Lobbying intensity in ``[0, 1]``."""

    n_lobbyists: int
    """Number of explicit lobbyist actors."""

    avg_votes_for: float
    passage_rate: float
    vote_std: float
    num_actors: int
    iterations: int

    @property
    def avg_vote_share(self) -> float:
        return self.avg_votes_for / self.num_actors if self.num_actors else 0.0


def run(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    n_steps: int = 10,
    n_lobbyists: int = 0,
    lobbyist_strength: float = 0.6,
) -> list[LobbyingPoint]:
    """Run the lobbying-intensity sweep scenario.

    Parameters
    ----------
    num_actors:
        Number of legislators.
    policy_dim:
        Dimensionality of the policy space.
    iterations:
        Monte Carlo iterations per intensity level.
    seed:
        Random seed (held constant across all levels).
    n_steps:
        Number of evenly-spaced intensity levels from 0.0 to 1.0.
    n_lobbyists:
        Number of explicit lobbyist special actors added on top of the
        layer-level lobbying effect (0 = layer only).
    lobbyist_strength:
        Strength of each explicit lobbyist actor (0-1).

    Returns
    -------
    list[LobbyingPoint]
        One entry per intensity level, ordered from 0.0 to 1.0.
    """
    from ..core.abstract_executive import ExecutiveType
    from ..integration.builders.engine_builder import build_engine
    from ..integration.config import AdvancedActorsConfig, IntegrationConfig, LayerConfig

    threshold = num_actors / 2
    intensities = [i / max(n_steps - 1, 1) for i in range(n_steps)]
    results: list[LobbyingPoint] = []

    for intensity in intensities:
        config = IntegrationConfig(
            num_actors=num_actors,
            policy_dim=policy_dim,
            iterations=iterations,
            seed=seed,
            layer_config=LayerConfig(
                include_ideal_point=True,
                include_public_opinion=False,
                include_lobbying=True,
                include_media_pressure=False,
                include_party_discipline=False,
                lobbying_intensity=intensity,
            ),
            actors_config=AdvancedActorsConfig(
                executive_type=ExecutiveType.PRESIDENTIAL,
                n_lobbyists=n_lobbyists,
                lobbyist_strength=lobbyist_strength,
            ),
        )

        votes: list[int] = build_engine(config).run()
        n = len(votes)
        avg = sum(votes) / n if n else 0.0
        passage_rate = sum(1 for v in votes if v > threshold) / n if n else 0.0
        variance = sum((v - avg) ** 2 for v in votes) / n if n > 1 else 0.0

        results.append(
            LobbyingPoint(
                lobbying_intensity=intensity,
                n_lobbyists=n_lobbyists,
                avg_votes_for=avg,
                passage_rate=passage_rate,
                vote_std=math.sqrt(variance),
                num_actors=num_actors,
                iterations=iterations,
            )
        )

    # --- Print summary ---
    print("Lobbying Intensity Sweep")
    print("=" * 58)
    print(
        f"{'Intensity':>9} {'Avg votes':>9} {'Vote share':>10} "
        f"{'Passage':>9} {'Std':>7}"
    )
    print("-" * 58)
    for p in results:
        bar = "#" * int(p.passage_rate * 20)
        print(
            f"{p.lobbying_intensity:>9.2f} {p.avg_votes_for:>9.1f} "
            f"{p.avg_vote_share:>9.1%} "
            f"{p.passage_rate:>8.1%} "
            f"{p.vote_std:>7.1f}  {bar}"
        )
    print("-" * 58)
    print(
        f"Actors: {num_actors}  |  Policy dim: {policy_dim}  |  "
        f"Iterations: {iterations}  |  Lobbyists: {n_lobbyists}"
    )

    return results


if __name__ == "__main__":
    run()
