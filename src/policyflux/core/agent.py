"""Agent interface for behavior models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract agent contract for simulation and inference.

    Concrete agents should implement `act` and optionally `reset`. Keep the
    interface minimal so both rule-based and learned agents can conform.
    """

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Return an action for a given observation."""
        raise NotImplementedError()

    def step(self, observation: Any) -> Any:
        """Alias for `act` to match simulator step loops."""
        return self.act(observation)

    def reset(self) -> None:
        """Optional hook to reset internal state between episodes."""
        return None
