from typing import Optional

from policyflux.core.id_generator import get_id_generator


class SequentialSpeaker:
    """Represents a speaker with agenda-setting strength."""

    def __init__(
        self,
        id: Optional[int] = None,
        name: str = "",
        agenda_support: float = 0.5,
    ) -> None:
        if id is None:
            id = get_id_generator().generate_actor_id()
        self.id = id
        self.name = name or f"Speaker_{id}"
        self.agenda_support = max(0.0, min(1.0, agenda_support))

    def set_agenda_support(self, support: float) -> None:
        self.agenda_support = max(0.0, min(1.0, support))