from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from policyflux.core.abstract_executive import Executive, ExecutiveType
from policyflux.core.actors_abstract import ComplexActor

from .abstract_bill import Bill
from .actors_abstract import CongressMember

if TYPE_CHECKING:
    pass


class CongressModel(ABC):
    def __init__(self, id: int) -> None:
        self.id: int = id
        self.congressmen: list[CongressMember] = []

        self.executive: Executive | None = None
        self.executive_type: ExecutiveType | None = None
        self.whips: ComplexActor | None = None

    def add_congressman(self, congressman: CongressMember) -> None:
        """Add a congressman to the Congress."""
        self.congressmen.append(congressman)

    def pop_congressman(self) -> CongressMember | None:
        """Remove and return the last congressman added."""
        if self.congressmen:
            return self.congressmen.pop()
        return None

    def delete_congressman(self, congressman: CongressMember) -> bool:
        """Delete a specific congressman from the Congress."""
        if congressman in self.congressmen:
            self.congressmen.remove(congressman)
            return True
        return False

    def cast_votes(self, bill: Bill) -> int:
        votes_for: int = 0
        for congressman in self.congressmen:
            if congressman.vote(bill):
                votes_for += 1
        return votes_for

    @abstractmethod
    def make_report(self) -> str:
        pass
