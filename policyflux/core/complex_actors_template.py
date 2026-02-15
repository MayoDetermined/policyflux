from abc import ABC, abstractmethod
from typing import Optional

from policyflux.core.types import PolicySpace


class ComplexActor(ABC):
    """Abstract base class for advanced political actors that influence the legislative process.

    ComplexActors are contextual modifiers (speakers, lobbyists, whips) whose
    attributes are consumed by Layer implementations during vote aggregation.
    """

    def __init__(self, id: int, name: str, ideology: Optional[PolicySpace] = None) -> None:
        self.id: int = id
        self.name: str = name
        self.ideology: PolicySpace = ideology if ideology is not None else PolicySpace(2)

    @abstractmethod
    def get_influence(self) -> float:
        """Return this actor's primary influence metric [0, 1]."""
        pass
