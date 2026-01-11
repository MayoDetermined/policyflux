from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class BehavioralParameters:
    """Mutable behavioral knobs that can be tuned at runtime."""

    vulnerability: float = 0.1
    loyalty: float = 0.5
    volatility: float = 0.1
    extras: dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs: Any) -> None:
        """Update known attributes and stash unknown ones in extras."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extras[key] = value


class AbstractAgent(Protocol):
    """Minimal protocol for pluggable agents."""

    id: int
    params: BehavioralParameters

    def compute_utility(self, law: Any, context: Any, network_view: Any) -> float:
        ...

    def make_decision(self, utility: float) -> int:
        ...

    def update_state(self, outcome: Any, context: Any) -> None:
        ...
