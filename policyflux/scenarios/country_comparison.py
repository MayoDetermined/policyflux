"""Scenario: Country Parliament Comparison.

Research question
-----------------
How do the **structural rules** of real-world multi-chamber parliaments
(chamber composition, veto powers, override thresholds) affect bill
passage rates, independent of legislators' political preferences?

Each parliament preset uses the same random-seed configuration: all
legislators receive ideal points drawn from a uniform distribution, and
``n_bills`` bills are submitted at random positions in policy space.  The
only variable is the constitutional structure of each legislature.

Supported systems (via :mod:`policyflux.integration.presets.parliament_presets`):

===========  ===========================================  =================
Key          Full name                                    Upper-house power
===========  ===========================================  =================
uk           UK Parliament                                Suspensive veto
us           US Congress                                  Full veto
germany      Bundestag / Bundesrat (consent law)          Full veto
france       Parlement français (navette législative)     Suspensive veto
italy        Parlamento italiano (bicameralismo perfetto) Full veto
poland       Parlament RP (Sejm + Senat)                  Override by lower
sweden       Riksdag                                      Unicameral
spain        Cortes Generales                             Suspensive veto
australia    Parliament of Australia                      Full veto
canada       Parliament of Canada                         Suspensive veto
===========  ===========================================  =================

Usage
-----
::

    from policyflux.scenarios import country_comparison

    results = country_comparison.run()

    # Subset of countries, more bills
    results = country_comparison.run(
        presets=["uk", "us", "germany", "sweden"],
        n_bills=100,
        policy_dim=3,
    )

    # Use smaller chambers for speed (overrides realistic membership sizes)
    results = country_comparison.run(chamber_size=50)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CountryResult:
    """Passage-rate result for one country's parliament."""

    country_key: str
    parliament_name: str
    n_bills: int
    n_passed: int
    passage_rate: float
    n_chambers: int
    policy_dim: int
    avg_rounds: float = 1.0
    """Average number of ping-pong rounds needed (bicameral systems only)."""

    chamber_sizes: list[int] = field(default_factory=list)
    """Membership size of each chamber in order (lower first)."""


def run(
    policy_dim: int = 2,
    n_bills: int = 30,
    seed: int = 42,
    presets: list[str] | None = None,
    chamber_size: int | None = None,
) -> list[CountryResult]:
    """Run the country parliament comparison scenario.

    Parameters
    ----------
    policy_dim:
        Dimensionality of the policy space for member ideal points and
        bill positions.
    n_bills:
        Number of randomly positioned bills to submit to each parliament.
    seed:
        Seed passed to :func:`~policyflux.set_seed` before creating each
        parliament and before drawing each bill position so that all
        parliaments face *identical* bills.
    presets:
        List of country keys to include (see table in module docstring).
        ``None`` runs all available presets.
    chamber_size:
        If set, override both chamber sizes with this value (useful for
        fast exploratory runs). ``None`` uses realistic membership sizes.

    Returns
    -------
    list[CountryResult]
        One entry per country, sorted by descending passage rate.
    """
    from ..core.pf_typing import PolicySpace
    from ..integration.presets.parliament_presets import (
        PARLIAMENT_PRESETS,
        ParliamentPresetConfig,
    )
    from ..pfrandom import random as pf_random
    from ..pfrandom import set_seed
    from ..toolbox.bill_models import SequentialBill

    all_keys = list(PARLIAMENT_PRESETS.keys())
    selected = presets if presets is not None else all_keys

    # Validate
    unknown = [k for k in selected if k not in PARLIAMENT_PRESETS]
    if unknown:
        raise ValueError(f"Unknown preset(s): {unknown!r}. Available: {sorted(all_keys)}")

    # Build bill positions once so every parliament faces the same bills
    set_seed(seed)
    bill_positions: list[list[float]] = [
        [pf_random() for _ in range(policy_dim)] for _ in range(n_bills)
    ]

    results: list[CountryResult] = []

    for key in selected:
        factory = PARLIAMENT_PRESETS[key]

        cfg = ParliamentPresetConfig(policy_dim=policy_dim)
        if chamber_size is not None:
            cfg = ParliamentPresetConfig(
                policy_dim=policy_dim,
                lower_house_size=chamber_size,
                upper_house_size=max(chamber_size // 2, 10),
            )

        # Recreate parliament with same seed for reproducibility
        set_seed(seed)
        parliament = factory(cfg)

        n_chambers = len(parliament._chambers)
        sizes = [cfg.size for cfg in parliament._configs]

        n_passed = 0
        total_rounds = 0

        for i, pos in enumerate(bill_positions):
            bill = SequentialBill(id=i + 1)
            bill_position = PolicySpace(policy_dim)
            bill_position.set_position(pos)

            vote_result = parliament.cast_votes(bill, bill_position=bill_position)
            if vote_result.passed:
                n_passed += 1
            total_rounds += vote_result.rounds

        passage_rate = n_passed / n_bills if n_bills else 0.0
        avg_rounds = total_rounds / n_bills if n_bills else 1.0

        results.append(
            CountryResult(
                country_key=key,
                parliament_name=parliament.name,
                n_bills=n_bills,
                n_passed=n_passed,
                passage_rate=passage_rate,
                n_chambers=n_chambers,
                policy_dim=policy_dim,
                avg_rounds=avg_rounds,
                chamber_sizes=sizes,
            )
        )

    results.sort(key=lambda r: r.passage_rate, reverse=True)

    # --- Print summary ---

    # Chamber-size label helper
    def _sizes_label(r: CountryResult) -> str:
        if not r.chamber_sizes:
            return ""
        return "+".join(str(s) for s in r.chamber_sizes)

    max_name = max(len(r.parliament_name) for r in results)
    col_w = max(max_name, 20)

    print("Country Parliament Comparison")
    print("=" * (col_w + 50))
    print(
        f"{'Parliament':<{col_w}} {'Chambers':>8} {'Sizes':>12} "
        f"{'Passed':>7} {'Rate':>8} {'Avg rnd':>8}"
    )
    print("-" * (col_w + 50))
    for r in results:
        bar = "#" * int(r.passage_rate * 20)
        print(
            f"{r.parliament_name:<{col_w}} {r.n_chambers:>8} "
            f"{_sizes_label(r):>12} {r.n_passed:>7} "
            f"{r.passage_rate:>7.1%} {r.avg_rounds:>8.2f}  {bar}"
        )
    print("-" * (col_w + 50))
    print(f"Bills per parliament: {n_bills}  |  Policy dim: {policy_dim}  |  Seed: {seed}")

    return results


if __name__ == "__main__":
    run()
