from typing import List, Optional

from policyflux.core.layer_template import Layer
from policyflux.core.id_generator import get_id_generator
from policyflux.core.types import UtilitySpace
from policyflux.models.advanced_actors.whips import SequentialWhip


class PartyDisciplineLayer(Layer):
    """Models party discipline influence on voting decision."""

    def __init__(
        self,
        id: Optional[int] = None,
        party_whips: Optional[List[SequentialWhip]] = None,
        discipline_base_strength: float = 0.5,
        party_line_support: float = 0.5,
        name: str = "PartyDiscipline",
        input_dim: int = 2,
        output_dim: int = 2,
    ) -> None:
        if id is None:
            id = get_id_generator().generate_layer_id()
        super().__init__(id, name, input_dim, output_dim)
        self.whips: List[SequentialWhip] = party_whips if party_whips is not None else []
        self.discipline_base_strength = max(0.0, min(1.0, discipline_base_strength))
        self.party_line_support = max(0.0, min(1.0, party_line_support))

    def add_whip(self, whip: SequentialWhip) -> None:
        self.whips.append(whip)

    def delete_whip(self, whip_id: int) -> bool:
        for i, whip in enumerate(self.whips):
            if whip.id == whip_id:
                del self.whips[i]
                return True
        return False

    def set_party_line_support(self, support: float) -> None:
        self.party_line_support = max(0.0, min(1.0, support))

    def set_discipline_strength(self, strength: float) -> None:
        self.discipline_base_strength = max(0.0, min(1.0, strength))

    def compile(self) -> None:
        return None

    def _aggregate_whip_strength(self) -> float:
        if not self.whips:
            return self.discipline_base_strength
        total = 0.0
        for whip in self.whips:
            total += max(0.0, min(1.0, getattr(whip, "discipline_strength", 0.0)))
        avg = total / len(self.whips)
        return max(0.0, min(1.0, avg))

    def _aggregate_party_line(self) -> float:
        if not self.whips:
            return self.party_line_support
        total = 0.0
        for whip in self.whips:
            total += max(0.0, min(1.0, getattr(whip, "party_line_support", 0.5)))
        avg = total / len(self.whips)
        return max(0.0, min(1.0, avg))

    def call(self, bill_space: UtilitySpace, **kwargs) -> float:
        base_prob = kwargs.get("base_prob", 0.5)
        discipline_strength = self._aggregate_whip_strength()
        party_line = self._aggregate_party_line()
        speaker_agenda = kwargs.get("speaker_agenda_support")
        if speaker_agenda is not None:
            speaker_agenda = max(0.0, min(1.0, speaker_agenda))
            party_line = 0.7 * party_line + 0.3 * speaker_agenda
        blended = (1.0 - discipline_strength) * base_prob + discipline_strength * party_line
        return max(0.0, min(1.0, blended))