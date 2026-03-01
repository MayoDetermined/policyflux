from typing import TYPE_CHECKING, Any

from policyflux.core.abstract_executive import ExecutiveActor
from policyflux.core.id_generator import get_id_generator
from policyflux.core.pf_typing import PolicySpace

if TYPE_CHECKING:
    from policyflux.core.abstract_bill import Bill


class SequentialPresident(ExecutiveActor):
    """Represents the president with an approval rating."""

    def __init__(
        self,
        id: int | None = None,
        name: str = "",
        approval_rating: float = 0.5,
        ideology: PolicySpace | None = None,
    ) -> None:
        if id is None:
            id = get_id_generator().generate_actor_id()
        super().__init__(id, name or f"President_{id}")
        self.ideology: PolicySpace = ideology if ideology is not None else PolicySpace(2)
        self.approval_rating: float = max(0.0, min(1.0, approval_rating))

    def set_approval_rating(self, rating: float) -> None:
        self.approval_rating = max(0.0, min(1.0, rating))

    def get_influence_on_bill(self, bill: "Bill", **context: Any) -> float:
        return self.approval_rating

    def can_veto_bill(self, bill: "Bill") -> bool:
        return True
