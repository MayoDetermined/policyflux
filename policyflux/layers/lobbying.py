from typing import Any

from policyflux.exceptions import ValidationError
from policyflux.toolbox.special_actors.lobby import SequentialLobbyist

from ..core.abstract_layer import Layer
from ..core.pf_typing import PolicyPosition


class LobbyingLayer(Layer):
    """Models external lobbying influence on voting decision."""

    def __init__(
        self,
        id: int | None = None,
        input_dim: int = 2,
        output_dim: int = 2,
        intensity: float = 0.0,
        name: str = "Lobbying",
    ) -> None:
        super().__init__(id, name, input_dim, output_dim)
        if not 0.0 <= intensity <= 1.0:
            raise ValidationError(f"Intensity must be in [0, 1], got {intensity}")
        self.intensity: float = intensity  # [0, 1] intensity of lobbying pressure

        self.lobbyists: list[SequentialLobbyist] = []

    @property
    def lobbysts(self) -> list[SequentialLobbyist]:
        """Backward-compatible alias for lobbyists."""
        return self.lobbyists

    @lobbysts.setter
    def lobbysts(self, value: list[SequentialLobbyist]) -> None:
        self.lobbyists = value

    def set_intensity(self, intensity: float) -> None:
        """Update lobbying intensity for a bill."""
        self.intensity = max(0.0, min(1.0, intensity))

    def add_lobbyist(self, lobbyist: SequentialLobbyist) -> None:
        """Add a lobbyist to influence the layer."""
        self.lobbyists.append(lobbyist)

    def delete_lobbyist(self, lobbyist_id: int | None = None) -> bool:
        """Delete a lobbyist by ID.

        Returns True if a lobbyist was removed.
        """
        if lobbyist_id is None:
            return False
        for index, lobbyist in enumerate(self.lobbyists):
            if getattr(lobbyist, "id", None) == lobbyist_id:
                del self.lobbyists[index]
                return True
        return False

    def pop_lobbyist(self) -> SequentialLobbyist | None:
        """Remove and return the last lobbyist."""
        if self.lobbyists:
            return self.lobbyists.pop()
        return None

    def add_lobbyst(self, lobbyst: SequentialLobbyist) -> None:
        """Backward-compatible alias for add_lobbyist."""
        self.add_lobbyist(lobbyst)

    def delete_lobbyst(self, lobbyst_id: int | None = None) -> bool:
        """Backward-compatible alias for delete_lobbyist."""
        return self.delete_lobbyist(lobbyst_id)

    def pop_lobbyst(self) -> SequentialLobbyist | None:
        """Backward-compatible alias for pop_lobbyist."""
        return self.pop_lobbyist()

    def compile(self) -> None:
        pass

    def _aggregate_lobbyist_pressure(self) -> float:
        if not self.lobbyists:
            return 0.0

        total: float = 0.0
        for lobbyist in self.lobbyists:
            strength = max(0.0, min(1.0, getattr(lobbyist, "influence_strength", 0.0)))
            stance = max(-1.0, min(1.0, getattr(lobbyist, "stance", 1.0)))
            total += strength * stance

        avg = total / len(self.lobbyists)
        return max(-1.0, min(1.0, avg))

    def _apply_pressure(self, base_prob: float, pressure: float) -> float:
        if pressure >= 0:
            return base_prob + (1.0 - base_prob) * pressure
        return base_prob * (1.0 + pressure)

    def call(self, bill_position: PolicyPosition, **kwargs: Any) -> float:
        """
        Apply lobbying modifier to voting decision.

        Lobbying pushes the vote probability toward 1.0 (yes) with given intensity.
        This acts as a multiplier, not a replacement value.
        """
        base_prob: float = float(kwargs.get("base_prob", 0.5))
        lobbyist_pressure = self._aggregate_lobbyist_pressure()
        combined_pressure = max(-1.0, min(1.0, self.intensity + lobbyist_pressure))
        return self._apply_pressure(base_prob, combined_pressure)
<<<<<<< HEAD

=======
>>>>>>> 28724a8eb17f6081daef9177c037673d899cf2a9
