"""Environment interface used by simulation engine."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class Environment(ABC):
    """Minimal environment contract to support step-based simulations.

    `reset` returns initial observations indexed by actor id. `step` accepts a
    mapping of actions and returns (observations, rewards, done, info).
    """

    @abstractmethod
    def reset(self) -> Dict[int, Any]:
        raise NotImplementedError()

    @abstractmethod
    def step(self, actions: Dict[int, Any]) -> Tuple[Dict[int, Any], Dict[int, float], bool, Dict]:
        raise NotImplementedError()
