from typing import List, Optional

from policyflux.core import voting_strategy as vs_module
from ..core.simple_actors_template import CongressMan
from ..core.bill_template import Bill
from ..core.layer_template import Layer
from ..core.types import UtilitySpace, PolicyPosition
from ..core.id_generator import get_id_generator
from ..core.aggregation_strategy import AggregationStrategy, SequentialAggregation
from ..core.contexts import VotingContext
import policyflux.pfrandom as pfrandom


class SequentialVoter(CongressMan):
    """
    Voter with dependency-injected layers for voting decisions.

    Layers are computed in order and their results are aggregated
    to produce a final voting decision using an AggregationStrategy.
    """

    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        layers: Optional[List[Layer]] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        voting_strategy: Optional[vs_module.VotingStrategy] = None
    ) -> None:
        if id is None:
            id = get_id_generator().generate_actor_id()
        super().__init__(id)
        self.name: str = name or f"Voter_{id}"
        self.layers: List[Layer] = layers if layers is not None else []
        self.aggregation: AggregationStrategy = aggregation_strategy or SequentialAggregation()
        self.voting_strategy: Optional[vs_module.VotingStrategy] = voting_strategy

    def add_layer(self, layer: Layer) -> None:
        """Inject a new layer into the voter."""
        if not isinstance(layer, Layer):
            raise TypeError(f"Expected Layer instance, got {type(layer)}")
        self.layers.append(layer)

    def set_aggregation_strategy(self, strategy: AggregationStrategy) -> None:
        """Change the aggregation strategy for combining layer outputs."""
        if not isinstance(strategy, AggregationStrategy):
            raise TypeError(f"Expected AggregationStrategy instance, got {type(strategy)}")
        self.aggregation = strategy

    def remove_layer(self, layer_id: int) -> bool:
        """Remove a layer by its ID."""
        self.layers = [l for l in self.layers if l.id != layer_id]
        return True

    def compute_layers(self, bill_space: UtilitySpace, **context) -> float:
        """
        Aggregate layer outputs using the configured aggregation strategy.

        Returns:
            Aggregated decision probability [0, 1]
        """
        if not self.layers:
            return self.yes_chance
        return self.aggregation.aggregate(self.layers, bill_space, **context)

    def _get_ideal_point(self) -> Optional[List[float]]:
        """Extract ideal point from the first IdealPointLayer, if present."""
        for layer in self.layers:
            if hasattr(layer, 'space') and layer.space is not None:
                space = layer.space
                if hasattr(space, 'position'):
                    return space.position
        return None

    def _build_voting_context(
        self, bill_space: UtilitySpace, decision_prob: float, **context
    ) -> VotingContext:
        """Build a VotingContext from available data."""
        ideal_point = self._get_ideal_point()
        bill_pos = tuple(bill_space) if bill_space else (0.5,)
        actor_pos = tuple(ideal_point) if ideal_point else (0.5,)
        return VotingContext(
            bill_position=PolicyPosition(bill_pos),
            actor_ideal_point=PolicyPosition(actor_pos),
            base_prob=decision_prob,
            public_support=context.get("public_support"),
            lobbying_intensity=context.get("lobbying_intensity"),
            media_pressure=context.get("media_pressure"),
            party_line_support=context.get("party_line_support"),
        )

    def vote(self, bill: Bill, bill_space=None, **context) -> bool:
        """Cast a vote on a bill using layers and voting strategy."""
        if bill_space is None:
            bill_space = getattr(bill, 'position', [])
        decision_prob = self.compute_layers(bill_space, **context)

        if self.voting_strategy is not None:
            voting_ctx = self._build_voting_context(bill_space, decision_prob, **context)
            result = self.voting_strategy.decide(decision_prob, voting_ctx)
            return bool(result)

        return pfrandom.random() < decision_prob
