from typing import List, Optional

import torch
import torch.nn as nn
from typing_extensions import Self

from ..core.id_generator import get_id_generator
from ..core.layer_template import Layer

# TO DO: Complete implementation

class SequentialNeuralLayer(Layer, nn.Sequential):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 2,
        id: Optional[int] = None,
        name: str = "SequentialNeuralLayer",
        architecture: Optional[List[nn.Module]] = None,
    ) -> None:
        if id is None:
            id = get_id_generator().generate_layer_id()

        Layer.__init__(self, id, name)
        nn.Sequential.__init__(self, *(architecture or []))

    def call(self, bill_space: List[float], **kwargs) -> float:
        tensor_input = torch.tensor(bill_space, dtype=torch.float32)
        output = self.forward(tensor_input)
        return float(output.squeeze().item())

    def append_neural_layer(self, layer: nn.Module) -> Self:
        super().append(layer)
        return self

    def insert_neural_layer(self, index: int, layer: nn.Module) -> Self:
        super().insert(index, layer)
        return self
    
    def pop_neural_layer(self, index: int = -1) -> nn.Module:
        return super().pop(index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
    
    def _run_train_step(self) -> None:
        pass

    def _run_train_loop(self) -> None:
        pass

    def run_validation(self) -> None:
        pass

    def compile(self) -> None:
        # Placeholder for future optimizer / loss setup
        return None