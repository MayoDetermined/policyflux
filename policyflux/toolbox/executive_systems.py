"""Concrete implementations of executive systems."""

from math import sqrt
from typing import Optional, Dict, Any, List

from ..core.executive import Executive, ExecutiveActor, ExecutiveType
from ..core.bill_template import Bill
from ..core.types import PolicySpace
from ..core.id_generator import get_id_generator
from policyflux.logging_config import logger


def _euclidean_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two policy positions."""
    return sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ============ PRESIDENTIAL SYSTEM ============

class President(ExecutiveActor):
    """President in a presidential system (US-style)."""

    def __init__(
        self,
        id: Optional[int] = None,
        name: str = "",
        approval_rating: float = 0.5,
        ideology: Optional[PolicySpace] = None,
    ):
        if id is None:
            id = get_id_generator().generate_actor_id()
        super().__init__(id, name or f"President_{id}")
        self.approval_rating = max(0.0, min(1.0, approval_rating))
        self.ideology = ideology or PolicySpace(2)

    def get_influence_on_bill(self, bill: Bill, **context) -> float:
        return self.approval_rating * 0.3

    def can_veto_bill(self, bill: Bill) -> bool:
        return True

    def set_approval_rating(self, rating: float) -> None:
        self.approval_rating = max(0.0, min(1.0, rating))


class PresidentialExecutive(Executive):
    """Presidential system executive branch."""

    def __init__(self, president: President, veto_override_threshold: float = 2/3):
        super().__init__(ExecutiveType.PRESIDENTIAL)
        self.president = president
        self.veto_override_threshold = veto_override_threshold

    def get_primary_actor(self) -> President:
        return self.president

    def inject_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["president"] = self.president
        context["president_approval"] = self.president.approval_rating
        context["executive_influence"] = self.president.get_influence_on_bill(None)
        return context

    def process_bill_result(self, bill: Bill, votes_for: int, total_votes: int) -> int:
        if votes_for <= total_votes / 2:
            return votes_for  # Bill failed, no veto needed

        if self._should_veto(bill):
            if votes_for < total_votes * self.veto_override_threshold:
                logger.info(
                    "President vetoes bill %s (votes %d/%d, override needs %.0f%%)",
                    getattr(bill, 'id', '?'), votes_for, total_votes,
                    self.veto_override_threshold * 100,
                )
                return 0  # Veto sustained
            else:
                logger.info("Presidential veto overridden on bill %s", getattr(bill, 'id', '?'))

        return votes_for

    def _should_veto(self, bill: Bill) -> bool:
        """Veto if the bill's position is far from the president's ideology.

        Uses normalised euclidean distance: if distance > (1 - approval_rating)
        the president opposes the bill. High-approval presidents tolerate more
        distance; low-approval presidents veto more aggressively.
        """
        bill_pos = getattr(bill, 'position', None)
        if bill_pos is None or not bill_pos:
            return False

        pres_pos = self.president.ideology.position
        if pres_pos is None or not pres_pos:
            return False

        dim = min(len(bill_pos), len(pres_pos))
        distance = _euclidean_distance(bill_pos[:dim], pres_pos[:dim])
        max_distance = sqrt(dim)  # max possible distance in [0,1]^dim
        normalised = distance / max_distance if max_distance > 0 else 0.0

        # Threshold: high approval → tolerant (high threshold), low → aggressive
        threshold = 0.3 + 0.5 * self.president.approval_rating
        return normalised > threshold


# ============ PARLIAMENTARY SYSTEM ============

class PrimeMinister(ExecutiveActor):
    """Prime Minister in a parliamentary system."""

    def __init__(
        self,
        id: Optional[int] = None,
        name: str = "",
        party_strength: float = 0.55,
        ideology: Optional[PolicySpace] = None,
    ):
        if id is None:
            id = get_id_generator().generate_actor_id()
        super().__init__(id, name or f"PM_{id}")
        self.party_strength = max(0.0, min(1.0, party_strength))
        self.ideology = ideology or PolicySpace(2)
        self.in_office = True

    def get_influence_on_bill(self, bill: Bill, **context) -> float:
        if context.get("is_government_bill", False):
            return self.party_strength * 0.85
        return self.party_strength * 0.3

    def can_veto_bill(self, bill: Bill) -> bool:
        return False


class ParliamentaryExecutive(Executive):
    """Parliamentary system executive branch."""

    def __init__(
        self,
        prime_minister: PrimeMinister,
        confidence_threshold: float = 0.5,
    ):
        super().__init__(ExecutiveType.PARLIAMENTARY)
        self.prime_minister = prime_minister
        self.confidence_threshold = confidence_threshold

    def get_primary_actor(self) -> PrimeMinister:
        return self.prime_minister

    def inject_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["prime_minister"] = self.prime_minister
        context["pm_party_strength"] = self.prime_minister.party_strength
        context["executive_influence"] = self.prime_minister.get_influence_on_bill(None, **context)

        if context.get("is_government_bill", False):
            context["party_discipline_strength"] = 0.9

        return context

    def process_bill_result(self, bill: Bill, votes_for: int, total_votes: int) -> int:
        if getattr(bill, "is_confidence_vote", False):
            if votes_for <= total_votes * self.confidence_threshold:
                self.prime_minister.in_office = False
                logger.info("Government falls! %s loses confidence vote.", self.prime_minister.name)

        return votes_for


# ============ SEMI-PRESIDENTIAL SYSTEM ============

class SemiPresidentialExecutive(Executive):
    """Semi-presidential system (France/Poland style)."""

    def __init__(
        self,
        president: President,
        prime_minister: PrimeMinister,
        veto_override_threshold: float = 3/5,
    ):
        super().__init__(ExecutiveType.SEMI_PRESIDENTIAL)
        self.president = president
        self.prime_minister = prime_minister
        self.veto_override_threshold = veto_override_threshold
        self.cohabitation = self._check_cohabitation()

    def _check_cohabitation(self) -> bool:
        """Cohabitation: president and PM from opposing camps."""
        return (self.president.approval_rating < 0.5
                and self.prime_minister.party_strength > 0.5)

    def get_primary_actor(self) -> ExecutiveActor:
        if self.cohabitation:
            return self.prime_minister
        return self.president

    def inject_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["president"] = self.president
        context["prime_minister"] = self.prime_minister
        context["president_approval"] = self.president.approval_rating
        context["cohabitation"] = self.cohabitation

        if self.cohabitation:
            context["executive_influence"] = max(
                self.president.get_influence_on_bill(None),
                self.prime_minister.get_influence_on_bill(None, **context)
            ) * 0.4
        else:
            context["executive_influence"] = (
                self.president.get_influence_on_bill(None)
                + self.prime_minister.get_influence_on_bill(None, **context)
            ) / 2 * 0.7

        return context

    def process_bill_result(self, bill: Bill, votes_for: int, total_votes: int) -> int:
        # Confidence vote handling (parliamentary side)
        if getattr(bill, "is_confidence_vote", False):
            if votes_for <= total_votes * 0.5:
                self.prime_minister.in_office = False
                logger.info("Government falls! %s loses confidence vote.", self.prime_minister.name)

        if votes_for <= total_votes / 2:
            return votes_for  # Bill already failed

        # Presidential veto (weaker than pure presidential system)
        if self.president.can_veto_bill(bill) and not self.cohabitation:
            bill_pos = getattr(bill, 'position', None)
            pres_pos = self.president.ideology.position if self.president.ideology else None
            if bill_pos and pres_pos:
                dim = min(len(bill_pos), len(pres_pos))
                distance = _euclidean_distance(bill_pos[:dim], pres_pos[:dim])
                max_dist = sqrt(dim) if dim > 0 else 1.0
                normalised = distance / max_dist if max_dist > 0 else 0.0

                threshold = 0.4 + 0.4 * self.president.approval_rating
                if normalised > threshold:
                    if votes_for < total_votes * self.veto_override_threshold:
                        logger.info("Semi-presidential veto on bill %s", getattr(bill, 'id', '?'))
                        return 0

        return votes_for
