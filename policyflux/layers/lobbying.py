from typing import List, Optional, TypeAlias

from policyflux.toolbox.advanced_actors.lobby import SequentialLobbyer
from ..core.layer_template import Layer
from ..core.types import UtilitySpace

## TO DO: Complete implementation

class LobbyingLayer(Layer):
    """Models external lobbying influence on voting decision."""
    
    def __init__(self, 
                id: Optional[int] = None, 
                input_dim: int = 2,
                output_dim: int = 2,
                intensity: float = 0.0, 
                name: str = "Lobbying") -> None:
        super().__init__(id, name, input_dim, output_dim)
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"Intensity must be in [0, 1], got {intensity}")
        self.intensity: float = intensity  # [0, 1] intensity of lobbying pressure
        
        self.lobbysts: List[SequentialLobbyer] = []
    
    def set_intensity(self, intensity: float) -> None:
        """Update lobbying intensity for a bill."""
        self.intensity = max(0.0, min(1.0, intensity))

    def add_lobbyst(self, lobbyst: SequentialLobbyer) -> None:
        """Add a lobbyist to influence the layer."""
        self.lobbysts.append(lobbyst)

    def delete_lobbyst(self, lobbyst_id: Optional[int] = None) -> bool:
        """Delete a lobbyist by ID.

        Returns True if a lobbyist was removed.
        """
        if lobbyst_id is None:
            return False
        for i, lobbyst in enumerate(self.lobbysts):
            if getattr(lobbyst, "id", None) == lobbyst_id:
                del self.lobbysts[i]
                return True
        return False

    def pop_lobbyst(self) -> Optional[SequentialLobbyer]:
        """Remove and return the last lobbyist."""
        if self.lobbysts:
            return self.lobbysts.pop()
        return None

    def compile(self) -> None:
        pass
    
    def _aggregate_lobbyist_pressure(self) -> float:
        if not self.lobbysts:
            return 0.0

        total: float = 0.0
        for lobbyst in self.lobbysts:
            strength = max(0.0, min(1.0, getattr(lobbyst, "influence_strength", 0.0)))
            stance = max(-1.0, min(1.0, getattr(lobbyst, "stance", 1.0)))
            total += strength * stance

        avg = total / len(self.lobbysts)
        return max(-1.0, min(1.0, avg))

    def _apply_pressure(self, base_prob: float, pressure: float) -> float:
        if pressure >= 0:
            return base_prob + (1.0 - base_prob) * pressure
        return base_prob * (1.0 + pressure)

    def call(self, bill_space: UtilitySpace, **kwargs) -> float:
        """
        Apply lobbying modifier to voting decision.
        
        Lobbying pushes the vote probability toward 1.0 (yes) with given intensity.
        This acts as a multiplier, not a replacement value.
        """
        base_prob = kwargs.get('base_prob', 0.5)
        lobbyst_pressure = self._aggregate_lobbyist_pressure()
        combined_pressure = max(-1.0, min(1.0, self.intensity + lobbyst_pressure))
        return self._apply_pressure(base_prob, combined_pressure)
