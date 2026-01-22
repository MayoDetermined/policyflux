from typing_extensions import Self
import torch.nn as nn
from typing import List, Optional
from transformers import Optional
from ..core.id_generator import get_id_generator
from ..core.layer_template import Layer

# TO DO: Complete implementation

class NeuralLayer(Layer, nn.Sequential):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, 
                 id: Optional[int] = None, name: str = "NeuralLayer"):
        # Initialize Layer
        if id is None:
            id = get_id_generator().generate_layer_id()
        Layer.__init__(self, id, name)
        
        # Initialize nn.Module
        nn.Module.__init__(self)
        
        # Define network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Output between 0-1 like other layers
        )

    def call(self, bill_space: List[float]):
        pass

    def forward(self):
        pass

    def compile(self):
        pass