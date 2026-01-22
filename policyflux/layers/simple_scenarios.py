from typing import List
from ..core.layer_template import Layer

utilitySpace = List[float]

class MediaPressureLayer(Layer):
    pass

class RandomEventLayer(Layer):
    pass

class PublicOpinionLayer(Layer):
    """Models public opinion influence on voting decision."""
    
    def __init__(self, id: int, support_level: float = 0.5, name: str = "PublicOpinion"):
        super().__init__(id, name)
        self.support_level = max(0.0, min(1.0, support_level))  # [0, 1] public support
    
    def set_support(self, support_level: float) -> None:
        """Update public support level for a bill."""
        self.support_level = max(0.0, min(1.0, support_level))
    
    def call(self, bill_space: utilitySpace, **kwargs) -> float:
        """
        Apply public opinion influence on the vote.
        
        Public opinion shifts the vote probability toward the support level.
        """
        base_prob = kwargs.get('base_prob', 0.5)
        # Blend base probability with public support (50/50 weight)
        return 0.5 * base_prob + 0.5 * self.support_level
