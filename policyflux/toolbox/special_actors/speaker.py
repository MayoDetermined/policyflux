from policyflux.core.actors_abstract import ComplexActor
from policyflux.core.id_generator import get_id_generator
from policyflux.core.pf_typing import PolicySpace


class SequentialSpeaker(ComplexActor):
    """Represents a speaker with agenda-setting strength."""

    def __init__(
        self,
        id: int | None = None,
        name: str = "",
        agenda_support: float = 0.5,
        ideology: PolicySpace | None = None,
    ) -> None:
        if id is None:
            id = get_id_generator().generate_actor_id()
        super().__init__(id, name or f"Speaker_{id}", ideology)
        self.agenda_support: float = max(0.0, min(1.0, agenda_support))

    def get_influence(self) -> float:
        return self.agenda_support

    def set_agenda_support(self, support: float) -> None:
        self.agenda_support = max(0.0, min(1.0, support))
