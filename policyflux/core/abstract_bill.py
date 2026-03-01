from __future__ import annotations

from abc import ABC, abstractmethod

from policyflux.core.pf_typing import PolicyPosition


class Bill(ABC):
    def __init__(self, id: int) -> None:
        self.id: int = id
        self.position: PolicyPosition | None = None

    def make_random_position(self, dim: int) -> None:
        self.position = PolicyPosition.random(dim)

    @abstractmethod
    def make_report(self) -> str:
        pass
