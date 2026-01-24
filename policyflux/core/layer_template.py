from abc import ABC, abstractmethod
from typing import Optional
from ..core.id_generator import get_id_generator
from .types import UtilitySpace

class Layer(ABC):
    def __init__(self, id: Optional[int] = None,
                name: str = "",
                input_dim: int = 2,
                output_dim: int = 2,) -> None:
        if id is None:
            id = get_id_generator().generate_layer_id()
        self.id = id
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name or self.__class__.__name__

    @abstractmethod
    def call(self, bill_space: UtilitySpace, **kwargs) -> float:
        """
        Compute layer's influence on voting decision.
        
        Args:
            bill_space: Bill's position in policy space
            **kwargs: Additional context (e.g., actor's ideal point, lobbying intensity)
            
        Returns:
            Float between 0 and 1 representing likelihood of support
        """
        pass

    @abstractmethod
    def compile(self) -> None:
        """Prepare layer for use (e.g., precompute values)."""
        pass