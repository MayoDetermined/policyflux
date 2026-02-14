from abc import ABC, abstractmethod
from typing import List, Optional

class LayerDataProcessor(ABC):
    """Abstract base class for layer data processors."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def forward(self, data: List) -> None:
        """Forward the data through the processor. Default implementation does nothing."""
        pass

    @abstractmethod
    def encode(self, data: List) -> List:
        """Encode the input data and return the encoded data."""
        pass