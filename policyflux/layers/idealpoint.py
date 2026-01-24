from math import exp
from typing import Optional, List
from ..core.layer_template import Layer
from ..core.id_generator import get_id_generator
from ..core.types import UtilitySpace

## TO DO: Train functionality to be implemented

class IdealPointEncoder(Layer):
    def __init__(self, 
                id: Optional[int] = None,
                input_dim: int = 2,
                output_dim: int = 2,
                space: Optional[UtilitySpace] = None,
                status_quo: Optional[UtilitySpace] = None,
                name: str = "IdealPoint") -> None:
        if id is None:
            id = get_id_generator().generate_layer_id()
        super().__init__(id, name, input_dim, output_dim)
        self.space = space if space is not None else []
        self.status_quo = status_quo if status_quo is not None else []

    def train(self, data: Optional[List[UtilitySpace]] = None) -> None:
        """Train the encoder by creating the `space` from provided data.

        The `space` is set to the centroid (element-wise mean) of the
        list of utility-space vectors in `data`. If `status_quo` is empty,
        it will be initialized to the same centroid.

        Args:
            data: A list of utility-space vectors (each a sequence of numbers).

        Raises:
            ValueError: If no data is provided or input dimensions mismatch.
        """
        if not data:
            raise ValueError("No data provided to train IdealPointEncoder")

        # Validate consistent dimensions
        first_len = len(data[0])
        if any(len(d) != first_len for d in data):
            raise ValueError("All samples in data must have the same dimensionality")

        # Compute centroid (element-wise mean)
        centroid = [sum(values) / len(data) for values in zip(*data)]

        # Assign the learned space and (optionally) initialize status quo
        self.space = centroid
        if not self.status_quo:
            self.status_quo = centroid.copy()

    def compile(self) -> None:
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