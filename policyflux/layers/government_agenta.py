"""Government agenda layer for parliamentary systems."""

from typing import Optional

from ..core.layer_template import Layer
from ..core.id_generator import get_id_generator
from ..core.types import UtilitySpace

class GovernmentAgendaLayer(Layer):
    """Models government control over legislative agenda in parliamentary systems."""
    
    def __init__(
        self,
        id: Optional[int] = None,
        pm_party_strength: float = 0.6,
        name: str = "GovernmentAgenda",
        input_dim: int = 2,
        output_dim: int = 2,
    ):
        if id is None:
            id = get_id_generator().generate_layer_id()
        super().__init__(id, name, input_dim, output_dim)
        self.pm_party_strength = max(0.0, min(1.0, pm_party_strength))
    
    def compile(self) -> None:
        return None
    
    def call(self, bill_space: UtilitySpace, **kwargs) -> float:
        base_prob = kwargs.get("base_prob", 0.5)
        
        # Check if this is a government bill
        is_government_bill = kwargs.get("is_government_bill", False)
        is_confidence_vote = kwargs.get("is_confidence_vote", False)
        
        if is_confidence_vote:
            # Confidence votes have EXTREME discipline
            return 0.98 if self.pm_party_strength > 0.5 else 0.02
        
        if is_government_bill:
            # Government bills have strong but not total discipline
            return base_prob * 0.1 + self.pm_party_strength * 0.9
        
        # Private member bills: normal voting
        return base_prob