"""Concrete implementations of executive systems."""

from typing import Optional, Dict, Any

from ..core.executive import Executive, ExecutiveActor, ExecutiveType
from ..core.bill_template import Bill
from ..core.types import PolicySpace
from ..core.id_generator import get_id_generator


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
        # Presidents have INDIRECT influence through approval rating
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
        # Bill passed?
        if votes_for <= total_votes / 2:
            return votes_for  # Bill failed, no veto needed
        
        # Check if president vetoes
        if self._should_veto(bill):
            # Check if veto can be overridden
            if votes_for < total_votes * self.veto_override_threshold:
                return 0  # Veto sustained, bill fails
        
        return votes_for  # Bill passes
    
    def _should_veto(self, bill: Bill) -> bool:
        """Simplified veto logic based on ideological distance."""
        # TODO: Implement proper distance calculation
        return False  # Placeholder


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
        self.in_office = True  # Can lose confidence vote
    
    def get_influence_on_bill(self, bill: Bill, **context) -> float:
        # PMs have STRONG influence on government bills
        if context.get("is_government_bill", False):
            return self.party_strength * 0.85  # Very high influence
        return self.party_strength * 0.3  # Lower on private members' bills
    
    def can_veto_bill(self, bill: Bill) -> bool:
        return False  # No veto in parliamentary systems


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
        
        # Parliamentary systems have HIGH party discipline on government bills
        if context.get("is_government_bill", False):
            context["party_discipline_strength"] = 0.9
        
        return context
    
    def process_bill_result(self, bill: Bill, votes_for: int, total_votes: int) -> int:
        # No veto power, result stands
        
        # Check if this was a confidence vote
        if getattr(bill, "is_confidence_vote", False):
            if votes_for <= total_votes * self.confidence_threshold:
                self.prime_minister.in_office = False
                print(f"Government falls! {self.prime_minister.name} loses confidence vote.")
        
        return votes_for


# ============ SEMI-PRESIDENTIAL SYSTEM ============

class SemiPresidentialExecutive(Executive):
    """Semi-presidential system (France/Poland style)."""
    
    def __init__(
        self,
        president: President,
        prime_minister: PrimeMinister,
    ):
        super().__init__(ExecutiveType.SEMI_PRESIDENTIAL)
        self.president = president
        self.prime_minister = prime_minister
        
        # Cohabitation: president and PM from different political camps
        self.cohabitation = self._check_cohabitation()
    
    def _check_cohabitation(self) -> bool:
        """Check if president and PM are from opposing camps."""
        # Simplified: weak president + strong PM = cohabitation
        return (self.president.approval_rating < 0.5 and 
                self.prime_minister.party_strength > 0.5)
    
    def get_primary_actor(self) -> ExecutiveActor:
        # In cohabitation, PM is stronger
        if self.cohabitation:
            return self.prime_minister
        return self.president
    
    def inject_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["president"] = self.president
        context["prime_minister"] = self.prime_minister
        context["cohabitation"] = self.cohabitation
        
        # Combine influences (competing or cooperating)
        if self.cohabitation:
            context["executive_influence"] = max(
                self.president.get_influence_on_bill(None),
                self.prime_minister.get_influence_on_bill(None, **context)
            ) * 0.4  # Weakened by conflict
        else:
            context["executive_influence"] = (
                self.president.get_influence_on_bill(None) +
                self.prime_minister.get_influence_on_bill(None, **context)
            ) / 2 * 0.7  # Strengthened by unity
        
        return context
    
    def process_bill_result(self, bill: Bill, votes_for: int, total_votes: int) -> int:
        # President can veto (limited power)
        if votes_for > total_votes / 2 and self.president.can_veto_bill(bill):
            # In cohabitation, veto is weaker
            if not self.cohabitation:
                # Simplified veto check
                pass
        
        return votes_for