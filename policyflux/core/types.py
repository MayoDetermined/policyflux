"""
Common type definitions used across the Congress simulation framework.
"""

from typing import List, TypeAlias

# Policy space representation (type alias)
PolicyPosition: TypeAlias = List[float]
UtilitySpace: TypeAlias = List[float]

class PolicySpace:
    """Manages a multi-dimensional policy space with dimension validation."""
    
    def __init__(self, dimensions: int):
        if dimensions <= 0:
            raise ValueError("Dimensions must be positive")
        self.dimensions = dimensions
        self._position: PolicyPosition = [0.0] * dimensions

    def set_position(self, position: PolicyPosition) -> None:
        """Set actor's position in policy space."""
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} dimensions, got {len(position)}")
        self._position = list(position)  # Create copy to avoid external mutations

    def get_position(self) -> PolicyPosition:
        """Get actor's current position in policy space."""
        return self._position.copy()  # Return copy for immutability
