"""
Lobbying ERGM Layer - applies lobbying influence based on network structure.

Uses a LobbyingERGMPModel to determine which lobbyists can influence
each legislator, then aggregates their influence on voting decisions.
"""

from typing import Any

from policyflux.exceptions import ValidationError
from policyflux.math_models.lobbying_ergmp import LobbyingERGMPModel
from policyflux.toolbox.special_actors.lobby import SequentialLobbyist

from ..core.abstract_layer import Layer
from ..core.pf_typing import PolicyPosition


class LobbyingERGMPLayer(Layer):
    """
    Models lobbying influence based on ERGM network structure.

    This layer uses a LobbyingERGMPModel to determine which lobbyists
    can access each legislator, then applies their influence to voting
    probabilities based on network connectivity.
    """

    def __init__(
        self,
        ergmp_model: LobbyingERGMPModel,
        id: int | None = None,
        input_dim: int = 2,
        output_dim: int = 2,
        intensity: float = 0.5,
        name: str = "LobbyingERGMP",
    ) -> None:
        """
        Initialize LobbyingERGMPLayer.

        Args:
            ergmp_model: LobbyingERGMPModel instance defining network structure
            id: Layer ID (auto-generated if None)
            input_dim: Input dimension
            output_dim: Output dimension
            intensity: Base lobbying pressure intensity [0, 1]
            name: Layer name
        """
        super().__init__(id, name, input_dim, output_dim)

        self.ergmp_model: LobbyingERGMPModel = ergmp_model

        if not 0.0 <= intensity <= 1.0:
            raise ValidationError(f"Intensity must be in [0, 1], got {intensity}")
        self.intensity: float = intensity

        # Lobbyists registered with their influence data
        self.lobbyists: dict[int, SequentialLobbyist] = {}

    def add_lobbyist(self, lobbyist: SequentialLobbyist, lobbyist_id: int | None = None) -> None:
        """
        Add a lobbyist to influence the network.

        Args:
            lobbyist: SequentialLobbyist instance
            lobbyist_id: ID in the ERGM network (auto-set to index if None)
        """
        if lobbyist_id is None:
            lobbyist_id = len(self.lobbyists)
        if lobbyist_id >= self.ergmp_model.n_lobbyists:
            raise ValidationError(
                f"Lobbyist ID {lobbyist_id} exceeds model capacity {self.ergmp_model.n_lobbyists}"
            )
        self.lobbyists[lobbyist_id] = lobbyist

    def delete_lobbyist(self, lobbyist_id: int) -> bool:
        """
        Remove a lobbyist.

        Returns True if lobbyist was deleted.
        """
        if lobbyist_id in self.lobbyists:
            del self.lobbyists[lobbyist_id]
            return True
        return False

    def set_intensity(self, intensity: float) -> None:
        """Update base lobbying intensity."""
        self.intensity = max(0.0, min(1.0, intensity))

    def compile(self) -> None:
        """Prepare layer for use."""
        pass

    def _aggregate_lobbyist_pressure(self, connected_lobbyist_ids: list[int]) -> float:
        """
        Aggregate pressure from lobbyists connected to a legislator.

        Args:
            connected_lobbyist_ids: List of lobbyist IDs from ERGM network

        Returns:
            Aggregated pressure [-1.0, 1.0]
        """
        if not connected_lobbyist_ids:
            return 0.0

        total: float = 0.0
        count: int = 0

        for lobbyist_id in connected_lobbyist_ids:
            if lobbyist_id in self.lobbyists:
                lobbyist = self.lobbyists[lobbyist_id]
                strength = max(0.0, min(1.0, getattr(lobbyist, "influence_strength", 0.5)))
                stance = max(-1.0, min(1.0, getattr(lobbyist, "stance", 1.0)))
                total += strength * stance
                count += 1

        if count == 0:
            return 0.0

        avg = total / count
        return max(-1.0, min(1.0, avg))

    def _apply_pressure(self, base_prob: float, pressure: float) -> float:
        """
        Apply lobbying pressure to base probability.

        Args:
            base_prob: Base voting probability [0, 1]
            pressure: Lobbying pressure [-1, 1]

        Returns:
            Modified probability [0, 1]
        """
        if pressure >= 0:
            return base_prob + (1.0 - base_prob) * pressure
        return base_prob * (1.0 + pressure)

    def call(self, bill_position: PolicyPosition, **kwargs: Any) -> float:
        """
        Apply ERGM-based lobbying influence to voting decision.

        Args:
            bill_position: Bill's position in policy space
            **kwargs: Additional context including:
                - base_prob: Base voting probability [0, 1] (default 0.5)
                - actor_legislator_id: ID of legislator in ERGM model (default None)

        Returns:
            Modified voting probability [0, 1]
        """
        base_prob: float = float(kwargs.get("base_prob", 0.5))
        legislator_id: int | None = kwargs.get("actor_legislator_id")

        # If no legislator ID provided, just apply base intensity
        if legislator_id is None:
            pressure = self.intensity
            return self._apply_pressure(base_prob, pressure)

        # Get lobbyists connected to this legislator via ERGM network
        try:
            connected_lobbyists = self.ergmp_model.get_legislator_exposure(legislator_id)
        except ValidationError:
            # Legislator ID out of range, use base intensity
            pressure = self.intensity
            return self._apply_pressure(base_prob, pressure)

        # Aggregate their influence
        lobbyist_pressure = self._aggregate_lobbyist_pressure(connected_lobbyists)

        # Combine with base intensity
        combined_pressure = max(-1.0, min(1.0, self.intensity + lobbyist_pressure))

        return self._apply_pressure(base_prob, combined_pressure)
