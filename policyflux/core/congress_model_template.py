from abc import ABC, abstractmethod
from ..core.actors_template import CongressMan
from ..core.bill_template import Bill

class CongressModel(ABC):
    def __init__(self, id: int) -> None:
        self.id: int = id
        self.congressmen: list[CongressMan] = []

    def add_congressman(self, congressman: CongressMan) -> None:
        """Add a congressman to the Congress."""
        self.congressmen.append(congressman)

    def pop_congressman(self) -> CongressMan | None:
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
        votes_for = 0 
        for congressman in self.congressmen:
            if congressman.vote(bill):
                votes_for += 1
        return votes_for
    
    @abstractmethod
    def make_report(self):
        pass