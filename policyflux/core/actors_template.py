from abc import ABC, abstractmethod
from ..core.bill_template import Bill
from ..core.layer_template import Layer

class CongressMan(ABC):
    def __init__(self, id: int, yes_chance: float | None = None) -> None:
        self.id: int = id
        self.yes_chance: float = yes_chance if yes_chance is not None else 0.5

    @abstractmethod
    def vote(self, bill: Bill, **kwargs) -> bool:
        pass