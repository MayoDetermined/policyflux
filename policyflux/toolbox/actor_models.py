from __future__ import annotations

from typing import Any

import policyflux.pfrandom as pfrandom
from policyflux.core import voting_strategy as vs_module
from policyflux.exceptions import ValidationError

from ..core.aggregation_strategy import AggregationStrategy, SequentialAggregation
from ..core.abstract_bill import Bill
from ..core.actors_abstract import CongressMember
from ..core.contexts import VotingContext
from ..core.id_generator import get_id_generator
from ..core.abstract_layer import Layer
from ..core.pf_typing import PolicyPosition


class SequentialVoter(CongressMember):
    """
    Voter with dependency-injected layers for voting decisions.

    Layers are computed in order and their results are aggregated
    to produce a final voting decision using an AggregationStrategy.
    """

    def __init__(
        self,
        id: int | None = None,
        name: str | None = None,
        layers: list[Layer] | None = None,
        aggregation_strategy: AggregationStrategy | None = None,
        voting_strategy: vs_module.VotingStrategy | None = None,
    ) -> None:
        if id is None:
            id = get_id_generator().generate_actor_id()
        super().__init__(id)
        self.name: str = name or f"Voter_{id}"
        self.layers: list[Layer] = layers if layers is not None else []
        self.aggregation: AggregationStrategy = aggregation_strategy or SequentialAggregation()
        self.voting_strategy: vs_module.VotingStrategy | None = voting_strategy

    def add_layer(self, layer: Layer) -> None:
        """Inject a new layer into the voter."""
        if not isinstance(layer, Layer):
            raise ValidationError(f"Expected Layer instance, got {type(layer)}")
        self.layers.append(layer)

    def set_aggregation_strategy(self, strategy: AggregationStrategy) -> None:
        """Change the aggregation strategy for combining layer outputs."""
        if not isinstance(strategy, AggregationStrategy):
            raise ValidationError(f"Expected AggregationStrategy instance, got {type(strategy)}")
        self.aggregation = strategy

    def remove_layer(self, layer_id: int) -> bool:
        """Remove a layer by its ID."""
        self.layers = [layer for layer in self.layers if layer.id != layer_id]
        return True

    def compute_layers(self, bill_position: PolicyPosition, **context: Any) -> float:
        """
        Aggregate layer outputs using the configured aggregation strategy.

        Returns:
            Aggregated decision probability [0, 1]
        """
        if not self.layers:
            return self.yes_chance
        return self.aggregation.aggregate(self.layers, bill_position, **context)

    def _get_ideal_point(self) -> PolicyPosition | None:
        """Extract ideal point from the first IdealPointLayer, if present."""
        for layer in self.layers:
            if hasattr(layer, "space") and layer.space is not None:
                space = layer.space
                if hasattr(space, "position"):
                    return space.position
        return None

    def _build_voting_context(
        self, bill_position: PolicyPosition, decision_prob: float, **context: Any
    ) -> VotingContext:
        """Build a VotingContext from available data."""
        ideal_point = self._get_ideal_point()
        return VotingContext(
            bill_position=bill_position if bill_position else PolicyPosition((0.5,)),
            actor_ideal_point=ideal_point if ideal_point else PolicyPosition((0.5,)),
            base_prob=decision_prob,
            public_support=context.get("public_support"),
            lobbying_intensity=context.get("lobbying_intensity"),
            media_pressure=context.get("media_pressure"),
            party_line_support=context.get("party_line_support"),
        )

    def vote(
        self, bill: Bill, bill_position: PolicyPosition | None = None, **context: Any
    ) -> bool:
        """Cast a vote on a bill using layers and voting strategy."""
        if bill_position is None:
            bill_position = getattr(bill, "position", None)
        decision_prob = self.compute_layers(bill_position, **context)

        if self.voting_strategy is not None:
            voting_ctx = self._build_voting_context(bill_position, decision_prob, **context)
            result = self.voting_strategy.decide(decision_prob, voting_ctx)
            return bool(result)

        return pfrandom.random() < decision_prob
