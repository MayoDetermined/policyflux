from abc import ABC, abstractmethod
from typing import Any


class LayerDataProcessor(ABC):
    """Abstract base class for layer data processors."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def forward(self, data: list[Any]) -> None:
        """Forward the data through the processor. Default implementation does nothing."""
        pass

    @abstractmethod
    def encode(self, data: list[Any]) -> list[Any]:
        """Encode the input data and return the encoded data."""
        pass
