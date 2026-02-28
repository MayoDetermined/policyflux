from policyflux.core.complex_actor import ComplexActor
from policyflux.core.id_generator import get_id_generator


class SequentialLobbyist(ComplexActor):
    """Represents a lobbying actor with influence strength and stance."""

    def __init__(
        self,
        id: int | None = None,
        name: str = "",
        influence_strength: float = 0.5,
        stance: float = 1.0,
    ) -> None:
        if id is None:
            id = get_id_generator().generate_actor_id()
        super().__init__(id, name or f"Lobbyist_{id}")
        self.influence_strength: float = max(0.0, min(1.0, influence_strength))
        self.stance: float = max(-1.0, min(1.0, stance))

    def get_influence(self) -> float:
        return self.influence_strength

    def set_influence_strength(self, strength: float) -> None:
        self.influence_strength = max(0.0, min(1.0, strength))

    def set_stance(self, stance: float) -> None:
        self.stance = max(-1.0, min(1.0, stance))
