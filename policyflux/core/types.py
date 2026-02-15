"""
Common type definitions used across the Congress simulation framework.
"""

from typing import List, TypeAlias
from dataclasses import dataclass
import math

# Policy space representation (type alias)
PolicyVector: TypeAlias = List[float]
UtilitySpace: TypeAlias = List[float]

class PolicySpace:
    """Manages a multi-dimensional policy space with dimension validation."""
    
    def __init__(self, dimensions: int):
        if dimensions <= 0:
            raise ValueError("Dimensions must be positive")
        self.dimensions = dimensions
        self._position: PolicyVector = [0.0] * dimensions

    def set_position(self, position: PolicyVector) -> None:
        """Set actor's position in policy space."""
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions, got {len(position)}")
        self._position = list(position)  # Create copy to avoid external mutations

    def get_position(self) -> PolicyVector:
        """Get actor's current position in policy space."""
        return self._position.copy()  # Return copy for immutability

    @property
    def position(self) -> PolicyVector:
        """Read-only access to position."""
        return self._position.copy()
    
    def __str__(self):
        return f"PolicySpace(dimensions={self.dimensions}, position={self._position})"

@dataclass(frozen=True)
class PolicyPosition:
    """Immutable position in policy space."""
    coordinates: tuple[float, ...]
    
    def __post_init__(self):
        if not all(0.0 <= x <= 1.0 for x in self.coordinates):
            raise ValueError("Coordinates must be in [0, 1]")
    
    @property
    def dimensions(self) -> int:
        return len(self.coordinates)
    
    def distance_to(self, other: "PolicyPosition") -> float:
        """Euclidean distance to another position."""
        if self.dimensions != other.dimensions:
            raise ValueError("Dimension mismatch")
        return math.sqrt(sum((a - b) ** 2 
                            for a, b in zip(self.coordinates, other.coordinates)))
    
    def utility(self, bill_position: "PolicyPosition") -> float:
        """Utility function (inverse distance)."""
        dist = self.distance_to(bill_position)
        return 1.0 / (1.0 + dist)
    
    @classmethod
    def random(cls, dimensions: int) -> "PolicyPosition":
        from policyflux.pfrandom import random
        return cls(tuple(random() for _ in range(dimensions)))