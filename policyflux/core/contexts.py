# policyflux/core/contexts.py
from dataclasses import dataclass

from .pf_typing import PolicyPosition


@dataclass(frozen=True)
class VotingContext:
    """Immutable context for voting decisions."""

    bill_position: PolicyPosition
    actor_ideal_point: PolicyPosition
    base_prob: float = 0.5

    # Optional influence factors
    public_support: float | None = None
    lobbying_intensity: float | None = None
    media_pressure: float | None = None
    party_line_support: float | None = None

    def with_base_prob(self, prob: float) -> "VotingContext":
        """Return new context with updated base_prob."""
        return VotingContext(
            bill_position=self.bill_position,
            actor_ideal_point=self.actor_ideal_point,
            base_prob=prob,
            public_support=self.public_support,
            lobbying_intensity=self.lobbying_intensity,
            media_pressure=self.media_pressure,
            party_line_support=self.party_line_support,
        )


@dataclass(frozen=True)
class SimulationContext:
    """Context for entire simulation run."""

    policy_dimensions: int
    num_actors: int
    num_bills: int
    seed: int
