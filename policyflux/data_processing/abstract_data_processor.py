from abc import ABC, abstractmethod
from typing import Any


class DataProcessor(ABC):
    """Abstract base class for data processors."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def fit(self, data: list[Any]) -> None:
        """Fit the processor to the data. Default implementation does nothing."""
        pass

    @abstractmethod
    def process(self, data: list[Any]) -> list[Any]:
        """Process the input data and return the processed data."""
        pass
