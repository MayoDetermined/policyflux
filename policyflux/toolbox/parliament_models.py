"""Multi-chamber parliament models.

This module provides infrastructure for simulating bicameral and multi-chamber
legislative systems. Key abstractions:

- ``ChamberRole``         - role of a chamber in the legislature
- ``UpperChamberPowers``  - how much power the upper house has over bills
- ``PassageThreshold``    - quorum/threshold required to pass in a chamber
- ``ChamberConfig``       - full configuration for one chamber
- ``ChamberVoteResult``   - result of a single chamber's vote
- ``ParliamentVoteResult``- aggregated result across all chambers
- ``MultiChamberParliamentModel`` - orchestrator that routes bills through chambers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from policyflux.logging_config import logger

from ..core.abstract_bill import Bill
from ..core.pf_typing import PolicyPosition
from .congress_model import SequentialCongressModel

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ChamberRole(str, Enum):
    """Functional role of a legislative chamber."""

    LOWER = "lower"
    """Primary/lower chamber (House of Commons, House of Representatives, Bundestag …)."""

    UPPER = "upper"
    """Revising/upper chamber (House of Lords, Senate, Bundesrat, Senat …)."""

    UNICAMERAL = "unicameral"
    """Single-chamber parliament (Riksdag, Knesset, Storting …)."""


class UpperChamberPowers(str, Enum):
    """How much blocking power the upper chamber possesses."""

    FULL_VETO = "full_veto"
    """Bill must pass *both* chambers to become law (US Congress, Italian Senate)."""

    SUSPENSIVE_VETO = "suspensive_veto"
    """Upper chamber can delay but lower chamber may override after ping-pong rounds
    (UK House of Lords, Czech Senate for most bills)."""

    OVERRIDE_BY_LOWER = "override_by_lower"
    """Lower chamber can override the upper chamber's rejection with a configurable
    supermajority (Polish Sejm overriding the Senate)."""

    ADVISORY = "advisory"
    """Upper house opinion is formally non-binding; lower chamber decides."""


class PassageThreshold(str, Enum):
    """Threshold required for a bill to pass in a chamber."""

    SIMPLE_MAJORITY = "simple_majority"
    """Strictly more than half of votes *cast*."""

    ABSOLUTE_MAJORITY = "absolute_majority"
    """Strictly more than half of *total seats* in the chamber."""

    SUPERMAJORITY_3_5 = "supermajority_3_5"
    """At least 60 % of votes cast (3/5 majority)."""

    SUPERMAJORITY_2_3 = "supermajority_2_3"
    """At least 66.7 % of votes cast (2/3 majority)."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ChamberConfig:
    """Configuration for a single legislative chamber.

    Parameters
    ----------
    name:
        Human-readable name (e.g. ``"House of Commons"``).
    role:
        Functional role within the parliament (lower, upper, unicameral).
    size:
        Official number of seats.
    passage_threshold:
        Threshold to pass a bill. Defaults to :attr:`PassageThreshold.SIMPLE_MAJORITY`.
    powers:
        Veto/override powers of the chamber when it acts as the upper house.
        Only meaningful for ``role == ChamberRole.UPPER``.
    max_ping_pong_rounds:
        Maximum number of inter-chamber exchange rounds (navette) before the
        lower chamber's last vote is treated as final. Default 2.
    override_threshold:
        Fraction of lower-chamber votes required to override an upper-house
        veto when ``powers == UpperChamberPowers.OVERRIDE_BY_LOWER``.
        Default 0.5 (absolute majority of *total* members).
    budget_bill_exempt:
        If ``True`` the upper chamber has *no* blocking power over money/budget bills.
    """

    name: str
    role: ChamberRole
    size: int
    passage_threshold: PassageThreshold = PassageThreshold.SIMPLE_MAJORITY
    powers: UpperChamberPowers = UpperChamberPowers.FULL_VETO
    max_ping_pong_rounds: int = 2
    override_threshold: float = 0.5
    budget_bill_exempt: bool = False


@dataclass
class ChamberVoteResult:
    """Outcome of a single chamber's vote on a bill.

    Attributes
    ----------
    chamber_name:
        Name of the chamber.
    chamber_role:
        Role of the chamber.
    votes_for:
        Number of votes in favour.
    votes_total:
        Total number of members who voted.
    passed:
        Whether the bill passed the threshold in this chamber.
    round_number:
        Ping-pong round in which this vote occurred (1-indexed).
    """

    chamber_name: str
    chamber_role: ChamberRole
    votes_for: int
    votes_total: int
    passed: bool
    round_number: int = 1

    @property
    def vote_share(self) -> float:
        """Fraction of votes cast in favour."""
        if self.votes_total == 0:
            return 0.0
        return self.votes_for / self.votes_total


@dataclass
class ParliamentVoteResult:
    """Aggregated result of a bill's journey through the whole parliament.

    Attributes
    ----------
    bill_id:
        Identifier of the bill.
    passed:
        Whether the bill ultimately became law.
    rounds:
        Total number of inter-chamber exchange rounds (1 = both voted once).
    chamber_results:
        Ordered list of individual chamber vote results.
    final_votes_for:
        Votes in favour in the decisive/final chamber vote.
    final_votes_total:
        Total votes cast in the decisive/final chamber vote.
    notes:
        Free-text explanation of the outcome (veto, override, etc.).
    """

    bill_id: int
    passed: bool
    rounds: int
    chamber_results: list[ChamberVoteResult] = field(default_factory=list)
    final_votes_for: int = 0
    final_votes_total: int = 0
    notes: str = ""

    @property
    def final_vote_share(self) -> float:
        """Fraction of votes in favour in the decisive chamber."""
        if self.final_votes_total == 0:
            return 0.0
        return self.final_votes_for / self.final_votes_total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passes_threshold(votes_for: int, votes_total: int, threshold: PassageThreshold) -> bool:
    """Return True if *votes_for* satisfies *threshold* in a chamber of *votes_total* seats."""
    if votes_total == 0:
        return False
    if threshold == PassageThreshold.SIMPLE_MAJORITY:
        return votes_for > votes_total / 2
    if threshold == PassageThreshold.ABSOLUTE_MAJORITY:
        return votes_for > votes_total / 2
    if threshold == PassageThreshold.SUPERMAJORITY_3_5:
        return votes_for >= votes_total * 0.6
    if threshold == PassageThreshold.SUPERMAJORITY_2_3:
        return votes_for >= votes_total * (2 / 3)
    return votes_for > votes_total / 2  # fallback


def _is_money_bill(bill: Bill) -> bool:
    """Heuristic: return True if the bill is marked as a money/budget bill."""
    return bool(getattr(bill, "is_money_bill", False))


# ---------------------------------------------------------------------------
# Multi-chamber parliament model
# ---------------------------------------------------------------------------


class MultiChamberParliamentModel:
    """Orchestrates bill passage through one or more legislative chambers.

    Each chamber is a :class:`~policyflux.toolbox.congress_model.SequentialCongressModel`
    paired with a :class:`ChamberConfig` that describes its role, size, and powers.

    Examples
    --------
    Build a generic bicameral parliament::

        lower_chamber = SequentialCongressModel()
        # … add members, layers …
        upper_chamber = SequentialCongressModel()

        parliament = MultiChamberParliamentModel("My Parliament")
        parliament.add_chamber(
            lower_chamber,
            ChamberConfig("Lower House", ChamberRole.LOWER, size=400)
        )
        parliament.add_chamber(
            upper_chamber,
            ChamberConfig(
                "Upper House", ChamberRole.UPPER, size=100,
                powers=UpperChamberPowers.SUSPENSIVE_VETO,
                max_ping_pong_rounds=1,
            ),
        )

        bill = SequentialBill(id=1)
        result = parliament.cast_votes(bill)
        print(result.passed, result.notes)
    """

    def __init__(self, name: str = "Parliament") -> None:
        self.name = name
        self._chambers: list[SequentialCongressModel] = []
        self._configs: list[ChamberConfig] = []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def add_chamber(self, chamber: SequentialCongressModel, config: ChamberConfig) -> None:
        """Register a chamber with its configuration."""
        self._chambers.append(chamber)
        self._configs.append(config)

    @property
    def chambers(self) -> list[tuple[SequentialCongressModel, ChamberConfig]]:
        """Ordered list of (chamber, config) pairs."""
        return list(zip(self._chambers, self._configs, strict=True))

    def lower_chamber(self) -> tuple[SequentialCongressModel, ChamberConfig] | None:
        """Return the first chamber with role LOWER, or None."""
        for ch, cfg in self.chambers:
            if cfg.role == ChamberRole.LOWER:
                return ch, cfg
        return None

    def upper_chamber(self) -> tuple[SequentialCongressModel, ChamberConfig] | None:
        """Return the first chamber with role UPPER, or None."""
        for ch, cfg in self.chambers:
            if cfg.role == ChamberRole.UPPER:
                return ch, cfg
        return None

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def cast_votes(
        self,
        bill: Bill,
        bill_position: PolicyPosition | None = None,
        **context: Any,
    ) -> ParliamentVoteResult:
        """Route *bill* through all chambers according to configuration.

        Parameters
        ----------
        bill:
            Bill to vote on.
        bill_position:
            Bill's position in policy space. Falls back to ``bill.position``.
        **context:
            Additional voting context forwarded to each chamber's
            :meth:`~policyflux.toolbox.congress_model.SequentialCongressModel.cast_votes`.

        Returns
        -------
        ParliamentVoteResult
        """
        if not self._chambers:
            raise ValueError("Parliament has no chambers configured.")

        if len(self._chambers) == 1:
            return self._unicameral_vote(bill, bill_position, **context)

        upper_cfg = next(
            (cfg for cfg in self._configs if cfg.role == ChamberRole.UPPER), None
        )

        if upper_cfg is not None and _is_money_bill(bill) and upper_cfg.budget_bill_exempt:
            # Money bill: upper chamber has no say - only lower chamber votes.
            return self._lower_only_vote(bill, bill_position, **context)

        if len(self._chambers) == 2:
            return self._bicameral_vote(bill, bill_position, **context)

        return self._sequential_multicameral_vote(bill, bill_position, **context)

    # ------------------------------------------------------------------
    # Unicameral path
    # ------------------------------------------------------------------

    def _unicameral_vote(
        self,
        bill: Bill,
        bill_position: PolicyPosition | None,
        **context: Any,
    ) -> ParliamentVoteResult:
        ch, cfg = self._chambers[0], self._configs[0]
        votes_for = ch.cast_votes(bill, bill_position, **context)
        votes_total = len(ch.congressmen)
        passed = _passes_threshold(votes_for, votes_total, cfg.passage_threshold)
        result = ChamberVoteResult(
            chamber_name=cfg.name,
            chamber_role=cfg.role,
            votes_for=votes_for,
            votes_total=votes_total,
            passed=passed,
            round_number=1,
        )
        logger.info(
            "[%s] %s: %d/%d (%s)",
            self.name,
            cfg.name,
            votes_for,
            votes_total,
            "PASS" if passed else "FAIL",
        )
        return ParliamentVoteResult(
            bill_id=bill.id,
            passed=passed,
            rounds=1,
            chamber_results=[result],
            final_votes_for=votes_for,
            final_votes_total=votes_total,
            notes="Unicameral vote",
        )

    # ------------------------------------------------------------------
    # Money-bill path (upper chamber exempt)
    # ------------------------------------------------------------------

    def _lower_only_vote(
        self,
        bill: Bill,
        bill_position: PolicyPosition | None,
        **context: Any,
    ) -> ParliamentVoteResult:
        lower_pair = self.lower_chamber()
        if lower_pair is None:
            raise ValueError("No lower chamber configured for money bill path.")
        ch, cfg = lower_pair
        votes_for = ch.cast_votes(bill, bill_position, **context)
        votes_total = len(ch.congressmen)
        passed = _passes_threshold(votes_for, votes_total, cfg.passage_threshold)
        result = ChamberVoteResult(
            chamber_name=cfg.name,
            chamber_role=cfg.role,
            votes_for=votes_for,
            votes_total=votes_total,
            passed=passed,
            round_number=1,
        )
        logger.info(
            "[%s] Money bill - only %s votes: %d/%d (%s)",
            self.name,
            cfg.name,
            votes_for,
            votes_total,
            "PASS" if passed else "FAIL",
        )
        return ParliamentVoteResult(
            bill_id=bill.id,
            passed=passed,
            rounds=1,
            chamber_results=[result],
            final_votes_for=votes_for,
            final_votes_total=votes_total,
            notes="Money bill - upper chamber exempt",
        )

    # ------------------------------------------------------------------
    # Bicameral path
    # ------------------------------------------------------------------

    def _bicameral_vote(
        self,
        bill: Bill,
        bill_position: PolicyPosition | None,
        **context: Any,
    ) -> ParliamentVoteResult:
        lower_pair = self.lower_chamber()
        upper_pair = self.upper_chamber()

        # If both chambers have the same role (e.g. two LOWER chambers in an unusual setup)
        # fall back to sequential mandatory passage.
        if lower_pair is None or upper_pair is None:
            return self._sequential_mandatory_both(bill, bill_position, **context)

        lower_ch, lower_cfg = lower_pair
        upper_ch, upper_cfg = upper_pair

        all_results: list[ChamberVoteResult] = []

        # ---- Step 1: lower chamber ----------------------------------------
        lower_votes_for = lower_ch.cast_votes(bill, bill_position, **context)
        lower_votes_total = len(lower_ch.congressmen)
        lower_passed = _passes_threshold(
            lower_votes_for, lower_votes_total, lower_cfg.passage_threshold
        )
        all_results.append(
            ChamberVoteResult(
                chamber_name=lower_cfg.name,
                chamber_role=lower_cfg.role,
                votes_for=lower_votes_for,
                votes_total=lower_votes_total,
                passed=lower_passed,
                round_number=1,
            )
        )
        logger.info(
            "[%s] %s round 1: %d/%d (%s)",
            self.name,
            lower_cfg.name,
            lower_votes_for,
            lower_votes_total,
            "PASS" if lower_passed else "FAIL",
        )

        if not lower_passed:
            return ParliamentVoteResult(
                bill_id=bill.id,
                passed=False,
                rounds=1,
                chamber_results=all_results,
                final_votes_for=lower_votes_for,
                final_votes_total=lower_votes_total,
                notes=f"Failed in {lower_cfg.name}",
            )

        # ---- Step 2: upper chamber ----------------------------------------
        if upper_cfg.powers == UpperChamberPowers.ADVISORY:
            # Advisory: upper votes but outcome is non-binding.
            upper_votes_for = upper_ch.cast_votes(bill, bill_position, **context)
            upper_votes_total = len(upper_ch.congressmen)
            upper_passed = _passes_threshold(
                upper_votes_for, upper_votes_total, upper_cfg.passage_threshold
            )
            all_results.append(
                ChamberVoteResult(
                    chamber_name=upper_cfg.name,
                    chamber_role=upper_cfg.role,
                    votes_for=upper_votes_for,
                    votes_total=upper_votes_total,
                    passed=upper_passed,
                    round_number=1,
                )
            )
            logger.info(
                "[%s] %s (advisory): %d/%d (%s)",
                self.name,
                upper_cfg.name,
                upper_votes_for,
                upper_votes_total,
                "PASS" if upper_passed else "FAIL",
            )
            return ParliamentVoteResult(
                bill_id=bill.id,
                passed=True,  # advisory vote never blocks
                rounds=1,
                chamber_results=all_results,
                final_votes_for=lower_votes_for,
                final_votes_total=lower_votes_total,
                notes=f"Advisory upper vote: {upper_cfg.name} {'approved' if upper_passed else 'rejected'} (non-binding)",
            )

        if upper_cfg.powers == UpperChamberPowers.FULL_VETO:
            return self._full_veto_path(
                bill, bill_position, upper_ch, upper_cfg,
                lower_votes_for, lower_votes_total, all_results, **context
            )

        if upper_cfg.powers == UpperChamberPowers.SUSPENSIVE_VETO:
            return self._suspensive_veto_path(
                bill, bill_position, lower_ch, lower_cfg, upper_ch, upper_cfg,
                all_results, **context
            )

        if upper_cfg.powers == UpperChamberPowers.OVERRIDE_BY_LOWER:
            return self._override_by_lower_path(
                bill, bill_position, lower_ch, lower_cfg, upper_ch, upper_cfg,
                lower_votes_for, lower_votes_total, all_results, **context
            )

        # Safety fallback
        return self._full_veto_path(
            bill, bill_position, upper_ch, upper_cfg,
            lower_votes_for, lower_votes_total, all_results, **context
        )

    def _full_veto_path(
        self,
        bill: Bill,
        bill_position: PolicyPosition | None,
        upper_ch: SequentialCongressModel,
        upper_cfg: ChamberConfig,
        lower_votes_for: int,
        lower_votes_total: int,
        results: list[ChamberVoteResult],
        **context: Any,
    ) -> ParliamentVoteResult:
        """Both chambers must pass (US, Italy)."""
        upper_votes_for = upper_ch.cast_votes(bill, bill_position, **context)
        upper_votes_total = len(upper_ch.congressmen)
        upper_passed = _passes_threshold(
            upper_votes_for, upper_votes_total, upper_cfg.passage_threshold
        )
        results.append(
            ChamberVoteResult(
                chamber_name=upper_cfg.name,
                chamber_role=upper_cfg.role,
                votes_for=upper_votes_for,
                votes_total=upper_votes_total,
                passed=upper_passed,
                round_number=1,
            )
        )
        logger.info(
            "[%s] %s: %d/%d (%s)",
            self.name,
            upper_cfg.name,
            upper_votes_for,
            upper_votes_total,
            "PASS" if upper_passed else "FAIL",
        )
        notes = "Both chambers passed" if upper_passed else f"Failed in {upper_cfg.name} (full veto)"
        return ParliamentVoteResult(
            bill_id=bill.id,
            passed=upper_passed,
            rounds=1,
            chamber_results=results,
            final_votes_for=upper_votes_for,
            final_votes_total=upper_votes_total,
            notes=notes,
        )

    def _suspensive_veto_path(
        self,
        bill: Bill,
        bill_position: PolicyPosition | None,
        lower_ch: SequentialCongressModel,
        lower_cfg: ChamberConfig,
        upper_ch: SequentialCongressModel,
        upper_cfg: ChamberConfig,
        results: list[ChamberVoteResult],
        **context: Any,
    ) -> ParliamentVoteResult:
        """Upper chamber has a suspensive veto; lower chamber can override after N rounds (UK Lords)."""
        max_rounds = max(1, upper_cfg.max_ping_pong_rounds)

        for round_num in range(1, max_rounds + 2):  # +1 so lower always gets the last word
            # Upper chamber votes
            upper_votes_for = upper_ch.cast_votes(bill, bill_position, **context)
            upper_votes_total = len(upper_ch.congressmen)
            upper_passed = _passes_threshold(
                upper_votes_for, upper_votes_total, upper_cfg.passage_threshold
            )
            results.append(
                ChamberVoteResult(
                    chamber_name=upper_cfg.name,
                    chamber_role=upper_cfg.role,
                    votes_for=upper_votes_for,
                    votes_total=upper_votes_total,
                    passed=upper_passed,
                    round_number=round_num,
                )
            )
            logger.info(
                "[%s] %s round %d: %d/%d (%s)",
                self.name, upper_cfg.name, round_num,
                upper_votes_for, upper_votes_total,
                "PASS" if upper_passed else "FAIL",
            )

            if upper_passed:
                return ParliamentVoteResult(
                    bill_id=bill.id,
                    passed=True,
                    rounds=round_num,
                    chamber_results=results,
                    final_votes_for=upper_votes_for,
                    final_votes_total=upper_votes_total,
                    notes=f"Upper chamber approved in round {round_num}",
                )

            # Upper rejected → lower can override if within allowed rounds
            if round_num > max_rounds:
                # Lower chamber has exhausted ping-pong patience; re-passes with simple majority
                lower_votes_for = lower_ch.cast_votes(bill, bill_position, **context)
                lower_votes_total = len(lower_ch.congressmen)
                lower_passed = _passes_threshold(
                    lower_votes_for, lower_votes_total, lower_cfg.passage_threshold
                )
                results.append(
                    ChamberVoteResult(
                        chamber_name=lower_cfg.name,
                        chamber_role=lower_cfg.role,
                        votes_for=lower_votes_for,
                        votes_total=lower_votes_total,
                        passed=lower_passed,
                        round_number=round_num + 1,
                    )
                )
                logger.info(
                    "[%s] %s final override round %d: %d/%d (%s)",
                    self.name, lower_cfg.name, round_num + 1,
                    lower_votes_for, lower_votes_total,
                    "PASS" if lower_passed else "FAIL",
                )
                note = (
                    f"Lower chamber overrode suspensive veto after {max_rounds} round(s)"
                    if lower_passed
                    else f"Override attempt failed in {lower_cfg.name}"
                )
                return ParliamentVoteResult(
                    bill_id=bill.id,
                    passed=lower_passed,
                    rounds=round_num + 1,
                    chamber_results=results,
                    final_votes_for=lower_votes_for,
                    final_votes_total=lower_votes_total,
                    notes=note,
                )

            # More rounds allowed: lower chamber re-votes (ping-pong)
            lower_votes_for = lower_ch.cast_votes(bill, bill_position, **context)
            lower_votes_total = len(lower_ch.congressmen)
            lower_passed = _passes_threshold(
                lower_votes_for, lower_votes_total, lower_cfg.passage_threshold
            )
            results.append(
                ChamberVoteResult(
                    chamber_name=lower_cfg.name,
                    chamber_role=lower_cfg.role,
                    votes_for=lower_votes_for,
                    votes_total=lower_votes_total,
                    passed=lower_passed,
                    round_number=round_num + 1,
                )
            )
            logger.info(
                "[%s] %s ping-pong round %d: %d/%d (%s)",
                self.name, lower_cfg.name, round_num + 1,
                lower_votes_for, lower_votes_total, "PASS" if lower_passed else "FAIL",
            )
            if not lower_passed:
                return ParliamentVoteResult(
                    bill_id=bill.id,
                    passed=False,
                    rounds=round_num + 1,
                    chamber_results=results,
                    final_votes_for=lower_votes_for,
                    final_votes_total=lower_votes_total,
                    notes=f"Failed in {lower_cfg.name} during ping-pong round {round_num + 1}",
                )

        # Should not reach here
        return ParliamentVoteResult(
            bill_id=bill.id,
            passed=False,
            rounds=max_rounds + 1,
            chamber_results=results,
            final_votes_for=0,
            final_votes_total=0,
            notes="Suspensive veto exhausted all rounds",
        )

    def _override_by_lower_path(
        self,
        bill: Bill,
        bill_position: PolicyPosition | None,
        lower_ch: SequentialCongressModel,
        lower_cfg: ChamberConfig,
        upper_ch: SequentialCongressModel,
        upper_cfg: ChamberConfig,
        lower_votes_for: int,
        lower_votes_total: int,
        results: list[ChamberVoteResult],
        **context: Any,
    ) -> ParliamentVoteResult:
        """Upper chamber can be overridden by lower chamber supermajority (Poland Sejm/Senat)."""
        upper_votes_for = upper_ch.cast_votes(bill, bill_position, **context)
        upper_votes_total = len(upper_ch.congressmen)
        upper_passed = _passes_threshold(
            upper_votes_for, upper_votes_total, upper_cfg.passage_threshold
        )
        results.append(
            ChamberVoteResult(
                chamber_name=upper_cfg.name,
                chamber_role=upper_cfg.role,
                votes_for=upper_votes_for,
                votes_total=upper_votes_total,
                passed=upper_passed,
                round_number=1,
            )
        )
        logger.info(
            "[%s] %s: %d/%d (%s)",
            self.name, upper_cfg.name,
            upper_votes_for, upper_votes_total,
            "PASS" if upper_passed else "FAIL",
        )

        if upper_passed:
            return ParliamentVoteResult(
                bill_id=bill.id,
                passed=True,
                rounds=1,
                chamber_results=results,
                final_votes_for=upper_votes_for,
                final_votes_total=upper_votes_total,
                notes="Both chambers approved",
            )

        # Upper rejected → check if lower can override
        override_votes_needed = upper_cfg.override_threshold * lower_votes_total
        can_override = lower_votes_for >= override_votes_needed
        logger.info(
            "[%s] %s rejected; override check: %d/%d (need %.0f): %s",
            self.name, upper_cfg.name,
            lower_votes_for, lower_votes_total, override_votes_needed,
            "OVERRIDE" if can_override else "FAILED",
        )
        notes = (
            f"Lower chamber overrode upper chamber rejection ({lower_votes_for}/{lower_votes_total})"
            if can_override
            else f"Override failed in {lower_cfg.name} ({lower_votes_for}/{lower_votes_total}, needed {override_votes_needed:.0f})"
        )
        return ParliamentVoteResult(
            bill_id=bill.id,
            passed=can_override,
            rounds=2,
            chamber_results=results,
            final_votes_for=lower_votes_for,
            final_votes_total=lower_votes_total,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Sequential multi-chamber (all must pass)
    # ------------------------------------------------------------------

    def _sequential_multicameral_vote(
        self,
        bill: Bill,
        bill_position: PolicyPosition | None,
        **context: Any,
    ) -> ParliamentVoteResult:
        """Each chamber votes in order; all must pass for the bill to become law."""
        all_results: list[ChamberVoteResult] = []
        for idx, (ch, cfg) in enumerate(self.chambers, start=1):
            votes_for = ch.cast_votes(bill, bill_position, **context)
            votes_total = len(ch.congressmen)
            passed = _passes_threshold(votes_for, votes_total, cfg.passage_threshold)
            all_results.append(
                ChamberVoteResult(
                    chamber_name=cfg.name,
                    chamber_role=cfg.role,
                    votes_for=votes_for,
                    votes_total=votes_total,
                    passed=passed,
                    round_number=idx,
                )
            )
            logger.info(
                "[%s] %s: %d/%d (%s)",
                self.name, cfg.name, votes_for, votes_total, "PASS" if passed else "FAIL",
            )
            if not passed:
                return ParliamentVoteResult(
                    bill_id=bill.id,
                    passed=False,
                    rounds=idx,
                    chamber_results=all_results,
                    final_votes_for=votes_for,
                    final_votes_total=votes_total,
                    notes=f"Failed in chamber {idx}: {cfg.name}",
                )
        last = all_results[-1]
        return ParliamentVoteResult(
            bill_id=bill.id,
            passed=True,
            rounds=len(all_results),
            chamber_results=all_results,
            final_votes_for=last.votes_for,
            final_votes_total=last.votes_total,
            notes="All chambers approved",
        )

    def _sequential_mandatory_both(
        self,
        bill: Bill,
        bill_position: PolicyPosition | None,
        **context: Any,
    ) -> ParliamentVoteResult:
        """Fallback: treat all chambers as equal and require sequential passage."""
        return self._sequential_multicameral_vote(bill, bill_position, **context)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def make_report(self) -> str:
        lines = [f"Parliament: {self.name}", f"Chambers: {len(self._chambers)}"]
        for ch, cfg in self.chambers:
            lines.append(
                f"  [{cfg.role.value.upper()}] {cfg.name} "
                f"| size={cfg.size} | members={len(ch.congressmen)} "
                f"| threshold={cfg.passage_threshold.value} "
                f"| powers={cfg.powers.value}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MultiChamberParliamentModel(name={self.name!r}, chambers={len(self._chambers)})"
