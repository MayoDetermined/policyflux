from abc import ABC, abstractmethod

# Import pfrandom to ensure deterministic random number generation
import policyflux.pfrandom as pfrandom


class Bill(ABC):
    def __init__(self, id: int) -> None:
        self.id: int = id
        self.position: list[float] = []  # Placeholder for bill's position in policy space

    def make_random_position(self, dim: int) -> None:
        self.position = [pfrandom.random() for _ in range(dim)]

    @abstractmethod
    def make_report(self) -> str:
        pass
