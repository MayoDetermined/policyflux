from abc import ABC, abstractmethod
from typing import Any

from policyflux.core.abstract_bill import Bill
from policyflux.core.pf_typing import PolicySpace


class ComplexActor(ABC):
    """Abstract base class for advanced political actors that influence the legislative process.

    ComplexActors are contextual modifiers (speakers, lobbyists, whips) whose
    attributes are consumed by Layer implementations during vote aggregation.
    """

    def __init__(self, id: int, name: str, ideology: PolicySpace | None = None) -> None:
        self.id: int = id
        self.name: str = name
        self.ideology: PolicySpace = ideology if ideology is not None else PolicySpace(2)

    @abstractmethod
    def get_influence(self) -> float:
        """Return this actor's primary influence metric [0, 1]."""
        pass


class CongressMember(ABC):
    def __init__(self, id: int, yes_chance: float | None = None) -> None:
        self.id: int = id
        self.yes_chance: float = yes_chance if yes_chance is not None else 0.5

    @abstractmethod
    def vote(self, bill: Bill, **kwargs: Any) -> bool:
        """
        Docstring for vote

        :param self: Description
        :param bill: Description
        :type bill: Bill
        :param kwargs: Description
        :return: Description
        :rtype: bool
        """
        pass
