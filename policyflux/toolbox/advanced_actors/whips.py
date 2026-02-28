from policyflux.core.complex_actor import ComplexActor
from policyflux.core.id_generator import get_id_generator
from policyflux.core.types import PolicySpace


class SequentialWhip(ComplexActor):
    """Represents a party whip enforcing party discipline."""

    def __init__(
        self,
        id: int | None = None,
        name: str = "",
        discipline_strength: float = 0.5,
        party_line_support: float = 0.5,
        ideology: PolicySpace | None = None,
    ) -> None:
        if id is None:
            id = get_id_generator().generate_actor_id()
        super().__init__(id, name or f"Whip_{id}", ideology)
        self.discipline_strength: float = max(0.0, min(1.0, discipline_strength))
        self.party_line_support: float = max(0.0, min(1.0, party_line_support))

    def get_influence(self) -> float:
        return self.discipline_strength

    def set_discipline_strength(self, strength: float) -> None:
        self.discipline_strength = max(0.0, min(1.0, strength))

    def set_party_line_support(self, support: float) -> None:
        self.party_line_support = max(0.0, min(1.0, support))
