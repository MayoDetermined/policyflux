from typing import Optional

from policyflux.core.layer_template import Layer
from policyflux.core.id_generator import get_id_generator
from policyflux.core.types import UtilitySpace


class MediaPressureLayer(Layer):
    """Models media influence on voting decision.

    Pressure is a signed value in [-1, 1], where positive values push
    toward supporting a bill and negative values push toward opposition.
    """

    def __init__(
        self,
        id: Optional[int] = None,
        pressure: float = 0.0,
        name: str = "MediaPressure",
        input_dim: int = 2,
        output_dim: int = 2,
    ) -> None:
        if id is None:
            id = get_id_generator().generate_layer_id()
        super().__init__(id, name, input_dim, output_dim)
        self.pressure = max(-1.0, min(1.0, pressure))

    def set_pressure(self, pressure: float) -> None:
        self.pressure = max(-1.0, min(1.0, pressure))

    def compile(self) -> None:
        return None

    def _apply_pressure(self, base_prob: float, pressure: float) -> float:
        if pressure >= 0:
            return base_prob + (1.0 - base_prob) * pressure
        return base_prob * (1.0 + pressure)

    def call(self, bill_space: UtilitySpace, **kwargs) -> float:
        base_prob = kwargs.get("base_prob", 0.5)
        speaker_agenda = kwargs.get("speaker_agenda_support")
        president_approval = kwargs.get("president_approval")

        adjustment = 0.0
        if speaker_agenda is not None:
            adjustment += 0.2 * (max(0.0, min(1.0, speaker_agenda)) - 0.5)
        if president_approval is not None:
            adjustment += 0.2 * (max(0.0, min(1.0, president_approval)) - 0.5)

        pressure = max(-1.0, min(1.0, self.pressure + adjustment))
        return self._apply_pressure(base_prob, pressure)