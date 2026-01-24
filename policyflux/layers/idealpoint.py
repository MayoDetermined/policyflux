from math import exp
from typing import Optional
from ..core.layer_template import Layer
from ..core.id_generator import get_id_generator
from ..core.types import UtilitySpace

## TO DO: Train functionality to be implemented

class IdealPointEncoder(Layer):
    def __init__(self, id: Optional[int] = None, space: Optional[UtilitySpace] = None, status_quo: Optional[UtilitySpace] = None, name: str = "IdealPoint"):
        if id is None:
            id = get_id_generator().generate_layer_id()
        super().__init__(id, name)
        self.space = space if space is not None else []
        self.status_quo = status_quo if status_quo is not None else []

    def train(self) -> None:
        pass

    def _sq_distance(self, a: UtilitySpace, b: UtilitySpace) -> float:
        if len(a) != len(b):
            raise ValueError(f"Dimension mismatch: {len(a)} != {len(b)}")
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def _delta_utility(self, bill_space: UtilitySpace) -> float:
        return (
            self._sq_distance(self.space, self.status_quo)
            - self._sq_distance(self.space, bill_space)
        )

    def _sigmoid(self, t: float) -> float:
        return 1 / (1 + exp(-t))

    def call(self, bill_space: UtilitySpace, **kwargs) -> float:
        delta_u = self._delta_utility(bill_space)
        return self._sigmoid(delta_u)