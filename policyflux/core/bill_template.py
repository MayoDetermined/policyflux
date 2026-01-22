from abc import ABC, abstractmethod
from random import random

class Bill(ABC):
    def __init__(self, id: int) -> None:
        self.id: int = id
        self.position: list = []  # Placeholder for bill's position in policy space

    def make_random_position(self, dim: int) -> None:
        self.position = [random() for _ in range(dim)]

    @abstractmethod
    def make_report(self):
        pass