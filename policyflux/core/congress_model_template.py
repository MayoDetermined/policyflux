from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

from policyflux.core.complex_actors_template import ComplexActor
from policyflux.core.executive import Executive, ExecutiveType
from .simple_actors_template import CongressMan
from .bill_template import Bill

if TYPE_CHECKING:
    from policyflux.toolbox.advanced_actors.whips import SequentialWhip

class CongressModel(ABC):
    def __init__(self, id: int) -> None:
        self.id: int = id
        self.congressmen: List[CongressMan] = []

        self.executive: Optional[Executive] = None
        self.executive_type: Optional[ExecutiveType] = None
        self.whips: Optional[ComplexActor] = None

    def add_congressman(self, congressman: CongressMan) -> None:
        """Add a congressman to the Congress."""
        self.congressmen.append(congressman)

    def pop_congressman(self) -> Optional[CongressMan]:
        """Remove and return the last congressman added."""
        if self.congressmen:
            return self.congressmen.pop()
        return None
    
    def delete_congressman(self, congressman: CongressMan) -> bool:
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