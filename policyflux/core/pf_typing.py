"""
Common type definitions used across the Congress simulation framework.

Position types are unified around ``PolicyPosition`` (immutable, validated)
and ``PolicySpace`` (mutable wrapper for actors whose position can change).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from policyflux.exceptions import DimensionMismatchError, ValidationError


@dataclass(frozen=True)
class PolicyPosition:
    """Immutable position in policy space.

    This is the canonical representation for all positions (bill positions,
    ideal points, status-quo points, etc.).  Coordinates are constrained to
    the [0, 1] range.

    ``PolicyPosition`` implements the sequence protocol so it can be used
    as a drop-in replacement for ``list[float]`` in iteration, indexing,
    and length queries.
    """

    coordinates: tuple[float, ...]

    def __post_init__(self) -> None:
        if not all(0.0 <= x <= 1.0 for x in self.coordinates):
            raise ValidationError("Coordinates must be in [0, 1]")

    # -- Sequence protocol --------------------------------------------------

    def __iter__(self) -> Iterator[float]:
        return iter(self.coordinates)

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, index: int) -> float:
        return self.coordinates[index]

    # -- Properties ---------------------------------------------------------

    @property
    def dimensions(self) -> int:
        return len(self.coordinates)

    # -- Convenience constructors / converters ------------------------------

    @classmethod
    def from_list(cls, values: list[float]) -> PolicyPosition:
        """Create a ``PolicyPosition`` from a plain list of floats."""
        return cls(tuple(values))

    @classmethod
    def random(cls, dimensions: int) -> PolicyPosition:
        from policyflux.pfrandom import random

        return cls(tuple(random() for _ in range(dimensions)))

    def to_list(self) -> list[float]:
        """Return a plain ``list[float]`` copy of the coordinates."""
        return list(self.coordinates)

    # -- Spatial helpers ----------------------------------------------------

    def distance_to(self, other: PolicyPosition) -> float:
        """Euclidean distance to another position."""
        if self.dimensions != other.dimensions:
            raise DimensionMismatchError("Dimension mismatch")
        return math.sqrt(
            sum((a - b) ** 2 for a, b in zip(self.coordinates, other.coordinates, strict=False))
        )

    def utility(self, bill_position: PolicyPosition) -> float:
        """Utility function (inverse distance)."""
        dist = self.distance_to(bill_position)
        return 1.0 / (1.0 + dist)


class PolicySpace:
    """Mutable wrapper around ``PolicyPosition`` with dimension validation.

    Use this when an actor's position needs to change over time (e.g.
    ideal-point evolution).  Internally stores an immutable
    ``PolicyPosition`` that is swapped on each ``set_position`` call.
    """

    def __init__(self, dimensions: int):
        if dimensions <= 0:
            raise ValidationError("Dimensions must be positive")
        self.dimensions = dimensions
        self._position: PolicyPosition = PolicyPosition(tuple(0.0 for _ in range(dimensions)))

    def set_position(self, position: PolicyPosition | list[float] | tuple[float, ...]) -> None:
        """Set actor's position in policy space."""
        if isinstance(position, PolicyPosition):
            if position.dimensions != self.dimensions:
                raise DimensionMismatchError(
                    f"Position must have {self.dimensions} dimensions, got {position.dimensions}"
                )
            self._position = position
        else:
            if len(position) != self.dimensions:
                raise DimensionMismatchError(
                    f"Position must have {self.dimensions} dimensions, got {len(position)}"
                )
            self._position = PolicyPosition(tuple(position))

    def get_position(self) -> PolicyPosition:
        """Get actor's current position in policy space."""
        return self._position

    @property
    def position(self) -> PolicyPosition:
        """Read-only access to position."""
        return self._position

    def __str__(self) -> str:
        return f"PolicySpace(dimensions={self.dimensions}, position={self._position.coordinates})"
