from typing import Optional

from policyflux.core.id_generator import get_id_generator


class SequentialWhip:
    """Represents a party whip enforcing party discipline."""

    def __init__(
        self,
        id: Optional[int] = None,
        name: str = "",
        discipline_strength: float = 0.5,
        party_line_support: float = 0.5,
    ) -> None:
        if id is None:
            id = get_id_generator().generate_actor_id()
        self.id = id
        self.name = name or f"Whip_{id}"
        self.discipline_strength = max(0.0, min(1.0, discipline_strength))
        self.party_line_support = max(0.0, min(1.0, party_line_support))

    def set_discipline_strength(self, strength: float) -> None:
        self.discipline_strength = max(0.0, min(1.0, strength))

    def set_party_line_support(self, support: float) -> None:
        self.party_line_support = max(0.0, min(1.0, support))