from abc import ABC, abstractmethod
from typing import Any

from .bill import Bill


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
