from typing import Any

from ..core.abstract_layer import Layer
from ..core.pf_typing import UtilitySpace

## TO DO: Complete implementation


class PublicOpinionLayer(Layer):
    """Models public opinion influence on voting decision."""

    def __init__(
        self,
        id: int | None = None,
        support_level: float = 0.5,
        name: str = "PublicOpinion",
        input_dim: int = 2,
        output_dim: int = 2,
    ) -> None:
        super().__init__(id, name, input_dim, output_dim)
        self.support_level: float = max(0.0, min(1.0, support_level))  # [0, 1] public support

    def set_support(self, support_level: float) -> None:
        """Update public support level for a bill."""
        self.support_level = max(0.0, min(1.0, support_level))

    def compile(self) -> None:
        pass

    def call(self, bill_space: UtilitySpace, **kwargs: Any) -> float:
        """
        Apply public opinion influence on the vote.

        Public opinion shifts the vote probability toward the support level.
        """
        base_prob: float = float(kwargs.get("base_prob", 0.5))
        president_approval = kwargs.get("president_approval")
        support = self.support_level
        if president_approval is not None:
            support = 0.7 * support + 0.3 * max(0.0, min(1.0, president_approval))
        # Blend base probability with public support (50/50 weight)
        return 0.5 * base_prob + 0.5 * support
