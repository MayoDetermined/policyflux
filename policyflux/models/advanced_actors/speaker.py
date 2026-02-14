from typing import Optional

from policyflux.core.id_generator import get_id_generator
from policyflux.layers.idealpoint import IdealPointLayer, IdealPointEncoderDF, IdealPointTextEncoder
from policyflux.core.types import PolicySpace

from policyflux.core.complex_actors_template import ComplexActor

class SequentialSpeaker(ComplexActor):
    """Represents a speaker with agenda-setting strength."""

    def __init__(
        self,
        id: Optional[int] = None,
        name: str = "",
        agenda_support: float = 0.5,
        ideology: Optional[PolicySpace] = None
    ) -> None:
        if id is None:
            id = get_id_generator().generate_actor_id()
        self.id: int = id
        self.name: str = name or f"Speaker_{id}"
        self.ideology: PolicySpace = ideology if ideology is not None else PolicySpace(2)
        self.agenda_support: float = max(0.0, min(1.0, agenda_support))

    def set_agenda_support(self, support: float) -> None:
        self.agenda_support = max(0.0, min(1.0, support))