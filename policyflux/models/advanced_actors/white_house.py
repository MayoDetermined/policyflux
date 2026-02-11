from typing import Optional

from policyflux.core.id_generator import get_id_generator
from policyflux.core.types import PolicySpace


class SequentialPresident:
    """Represents the president with an approval rating."""

    def __init__(
        self,
        id: Optional[int] = None,
        name: str = "",
        approval_rating: float = 0.5,
        ideology: Optional[PolicySpace] = None
    ) -> None:
        if id is None:
            id = get_id_generator().generate_actor_id()
        self.id: int = id
        self.name: str = name or f"President_{id}"
        self.ideology: PolicySpace = ideology if ideology is not None else PolicySpace(2)
        self.approval_rating: float = max(0.0, min(1.0, approval_rating))

    def set_approval_rating(self, rating: float) -> None:
        self.approval_rating = max(0.0, min(1.0, rating))