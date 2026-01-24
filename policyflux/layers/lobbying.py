from typing import List, TypeAlias

from policyflux.models.advanced_actors.lobby import SequentialLobbyer
from ..core.layer_template import Layer

utilitySpace: TypeAlias = List[float]

## TO DO: Complete implementation

class LobbyingLayer(Layer):
    """Models external lobbying influence on voting decision."""
    
    def __init__(self, id: int, intensity: float = 0.0, name: str = "Lobbying") -> None:
        super().__init__(id, name)
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"Intensity must be in [0, 1], got {intensity}")
        self.intensity = intensity  # [0, 1] intensity of lobbying pressure
        
        self.lobbysts: list[SequentialLobbyer] = []
    
    def set_intensity(self, intensity: float) -> None:
        """Update lobbying intensity for a bill."""
        self.intensity = max(0.0, min(1.0, intensity))

    def add_lobbyst(self, lobbyst: SequentialLobbyer) -> None:
        """
        Docstring for add_lobbyst
        
        :param self: Description
        :param lobbyst: Description
        :type lobbyst: SequentialLobbyer
        """
        self.lobbysts.append(lobbyst)

    def delete_lobbyst(self) -> None:
        pass

    def pop_lobbyst(self) -> SequentialLobbyer:
        """
        Docstring for pop_lobbyst
        
        :param self: Description
        :return: Description
        :rtype: SequentialLobbyer
        """
        if self.lobbysts:
            return self.lobbysts.pop()
        return None

    def compile(self) -> None:
        pass
    
    def call(self, bill_space: utilitySpace, **kwargs) -> float:
        """
        Apply lobbying modifier to voting decision.
        
        Lobbying pushes the vote probability toward 1.0 (yes) with given intensity.
        This acts as a multiplier, not a replacement value.
        """
        base_prob = kwargs.get('base_prob', 0.5)
        # Lobbying pushes probability upward
        return base_prob + (1.0 - base_prob) * self.intensity
