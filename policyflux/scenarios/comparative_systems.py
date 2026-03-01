"""Scenario: Comparative Systems Analysis.

Research question
-----------------
How do **presidential**, **parliamentary**, and **semi-presidential**
executive systems differ in average bill-passage rates when operating on
the same underlying legislative body?

This scenario runs Monte Carlo simulations for all three system types
with identical actor counts, policy dimensionality, and random seed.
It then compares:

- Average number of votes cast in favour
- Bill passage rate (fraction of iterations in which a majority was reached)
- Vote-share distribution (mean ± std)

Usage
-----
::

    from policyflux.scenarios import comparative_systems

    # Default run (prints summary, returns results)
    results = comparative_systems.run()

    # Custom parameters
    results = comparative_systems.run(
        num_actors=150,
        policy_dim=3,
        iterations=500,
        president_approval=0.65,
        pm_party_strength=0.6,
    )
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SystemResult:
    """Simulation result for a single executive-system type."""

    system: str
    """Human-readable system label."""

    avg_votes_for: float
    """Mean number of votes cast in favour across all MC iterations."""

    passage_rate: float
    """Fraction of iterations in which a majority voted in favour (0-1)."""

    vote_std: float
    """Standard deviation of per-iteration vote counts."""

    num_actors: int
    """Total number of legislators in the simulation."""

    iterations: int
    """Number of Monte Carlo iterations run."""

    @property
    def avg_vote_share(self) -> float:
        """Mean vote share (0-1)."""
        return self.avg_votes_for / self.num_actors if self.num_actors else 0.0


def run(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    president_approval: float = 0.5,
    pm_party_strength: float = 0.55,
    confidence_threshold: float = 0.5,
    government_bill_rate: float = 0.7,
    veto_override_threshold: float = 2 / 3,
) -> list[SystemResult]:
    """Run comparative executive-systems scenario.

    Parameters
    ----------
    num_actors:
        Number of legislators in each simulation.
    policy_dim:
        Dimensionality of the policy space.
    iterations:
        Monte Carlo iterations per system.
    seed:
        Random seed (same for all three systems).
    president_approval:
        Presidential approval rating used for both presidential and
        semi-presidential simulations.
    pm_party_strength:
        PM's party strength used for parliamentary and
        semi-presidential simulations.
    confidence_threshold:
        Confidence-vote threshold (parliamentary system).
    government_bill_rate:
        Fraction of bills treated as government bills (parliamentary).
    veto_override_threshold:
        Supermajority needed to override a presidential veto.

    Returns
    -------
    list[SystemResult]
        One entry per executive system, sorted by descending passage rate.
    """
    import math

    from ..integration.builders.engine_builder import build_engine
    from ..integration.presets import (
        create_parliamentary_config,
        create_presidential_config,
        create_semi_presidential_config,
    )

    configs = {
        "Presidential": create_presidential_config(
            num_actors=num_actors,
            policy_dim=policy_dim,
            iterations=iterations,
            seed=seed,
            president_approval=president_approval,
            veto_override_threshold=veto_override_threshold,
        ),
        "Parliamentary": create_parliamentary_config(
            num_actors=num_actors,
            policy_dim=policy_dim,
            iterations=iterations,
            seed=seed,
            pm_party_strength=pm_party_strength,
            confidence_threshold=confidence_threshold,
            government_bill_rate=government_bill_rate,
        ),
        "Semi-Presidential": create_semi_presidential_config(
            num_actors=num_actors,
            policy_dim=policy_dim,
            iterations=iterations,
            seed=seed,
            president_approval=president_approval,
            pm_party_strength=pm_party_strength,
        ),
    }

    threshold = num_actors / 2
    results: list[SystemResult] = []

    for system_name, config in configs.items():
        votes: list[int] = build_engine(config).run()
        n = len(votes)
        avg = sum(votes) / n if n else 0.0
        passage_rate = sum(1 for v in votes if v > threshold) / n if n else 0.0
        variance = sum((v - avg) ** 2 for v in votes) / n if n > 1 else 0.0
        std = math.sqrt(variance)

        results.append(
            SystemResult(
                system=system_name,
                avg_votes_for=avg,
                passage_rate=passage_rate,
                vote_std=std,
                num_actors=num_actors,
                iterations=iterations,
            )
        )

    results.sort(key=lambda r: r.passage_rate, reverse=True)

    # --- Print summary ---
    print("Comparative Executive-Systems Analysis")
    print("=" * 56)
    print(f"{'System':<20} {'Avg votes':>9} {'Vote share':>10} {'Passage':>9} {'Std':>7}")
    print("-" * 56)
    for r in results:
        print(
            f"{r.system:<20} {r.avg_votes_for:>9.1f} "
            f"{r.avg_vote_share:>9.1%} "
            f"{r.passage_rate:>8.1%} "
            f"{r.vote_std:>7.1f}"
        )
    print("-" * 56)
    print(f"Actors: {num_actors}  |  Policy dim: {policy_dim}  |  Iterations: {iterations}")

    return results


if __name__ == "__main__":
    run()
