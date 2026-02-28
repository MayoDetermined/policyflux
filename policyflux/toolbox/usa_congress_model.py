from typing import Any

from policyflux.core.abstract_executive import Executive
from policyflux.core.pf_typing import PolicySpace
from policyflux.exceptions import DimensionMismatchError
from policyflux.logging_config import logger

from ..core.abstract_bill import Bill
from ..core.congress_model import CongressModel
from ..core.id_generator import get_id_generator
from ..core.abstract_layer import Layer
from .actor_models import SequentialVoter
from .special_actors.lobby import SequentialLobbyist
from .special_actors.speaker import SequentialSpeaker
from .special_actors.whips import SequentialWhip
from .special_actors.white_house import SequentialPresident


class SequentialCongressModel(CongressModel):
    """
    Congress model using sequential voters with dependency-injected layers.
    """

    def __init__(self, id: int | None = None) -> None:
        if id is None:
            id = get_id_generator().generate_model_id()
        super().__init__(id)
        self.lobbyists: list[SequentialLobbyist] = []
        self.congressmen: list[SequentialVoter] = []  # type: ignore[assignment]
        self.whips: list[SequentialWhip] = []  # type: ignore[assignment]
        self.speaker: SequentialSpeaker | None = None
        self.president: SequentialPresident | None = None

    def set_executive(self, executive: Executive) -> None:
        """Set the executive branch (Presidential/Parliamentary/Semi-Presidential)."""
        self.executive = executive

    def cast_votes(self, bill: Bill, bill_space: list[float] | None = None, **context: Any) -> int:
        """
        Cast votes from all congressmen on a bill.

        Args:
            bill: Bill object to vote on
            bill_space: Bill's position in policy space
            **context: Additional voting context (lobbying, public opinion, etc.)

        Returns:
            Number of votes in favor

        Raises:
            ValueError: If bill_space dimensions are inconsistent with voter ideal points
        """
        # Use bill.position if bill_space not explicitly provided
        if bill_space is None:
            bill_space = bill.position

        # Validate dimensions across all voters with ideal points
        if bill_space and self.congressmen:
            bill_dim = len(bill_space)
            for congressman in self.congressmen:
                # Check if congressman has IdealPointEncoder layer
                for layer in congressman.layers:
                    if hasattr(layer, "space") and layer.space:
                        # Handle both PolicySpace objects and lists
                        if isinstance(layer.space, PolicySpace):
                            voter_dim = layer.space.dimensions
                        else:
                            voter_dim = len(layer.space)
                        if voter_dim != bill_dim:
                            raise DimensionMismatchError(
                                f"Dimension mismatch: bill has {bill_dim} dimensions, "
                                f"but {congressman.name} has ideal point with {voter_dim} dimensions"
                            )

        context = dict(context)
        if self.speaker is not None:
            context.setdefault("speaker", self.speaker)
            context.setdefault(
                "speaker_agenda_support", getattr(self.speaker, "agenda_support", 0.5)
            )
        if self.president is not None:
            context.setdefault("president", self.president)
            context.setdefault(
                "president_approval", getattr(self.president, "approval_rating", 0.5)
            )

        # Inject executive context before voting
        if hasattr(self, "executive") and self.executive is not None:
            context = self.executive.inject_context(context)

        votes_for: int = 0
        for congressman in self.congressmen:
            if congressman.vote(bill, bill_space, **context):
                votes_for += 1

        # Process through executive (veto, confidence votes, etc.)
        if hasattr(self, "executive") and self.executive is not None:
            votes_for = self.executive.process_bill_result(bill, votes_for, len(self.congressmen))

        return votes_for

    def add_layer_to_congressmen(self, layer: Layer) -> bool:
        """Add a layer to all congressmen."""
        if not self.congressmen:
            return False
        for congressman in self.congressmen:
            congressman.add_layer(layer)
        return True

    def delete_layer_from_congressmen(self, layer_id: int) -> bool:
        """Delete a layer by ID from all congressmen."""
        if not self.congressmen:
            return False
        for congressman in self.congressmen:
            congressman.remove_layer(layer_id)
        return True

    def add_n_congressmen(self, n: int, layers: list[Layer] | None = None) -> None:
        """Add n congressmen with unique IDs."""
        for _ in range(n):
            new_id = get_id_generator().generate_actor_id()
            congressman = SequentialVoter(id=new_id)
            if layers:
                for layer in layers:
                    congressman.add_layer(layer)
            self.add_congressman(congressman)

    def set_speaker(self, speaker: SequentialSpeaker) -> None:
        """Add a Speaker to the Congress."""
        self.speaker = speaker

    def set_president(self, president: SequentialPresident) -> None:
        """Add a President to the Congress context."""
        self.president = president

    def add_whip(self, whip: SequentialWhip) -> None:
        """Add a Whip to the Congress."""
        self.whips.append(whip)

    def delete_whip(self, whip_id: int) -> bool:
        """Delete a Whip by ID from the Congress."""
        for i, whip in enumerate(self.whips):
            if whip.id == whip_id:
                del self.whips[i]
                return True
        return False

    def pop_whip(self) -> SequentialWhip | None:
        """Remove and return the last added Whip."""
        if self.whips:
            return self.whips.pop()
        return None

    def compile(self) -> None:
        """Compile/validate the model structure."""
        # Validate all congressmen have at least one layer
        for congressman in self.congressmen:
            if not congressman.layers:
                logger.warning("%s has no decision layers", congressman.name)
            else:
                for layer in congressman.layers:
                    layer.compile()

    def make_report(self) -> str:
        """Generate a report about the Congress model."""
        report = f"Congress Model {self.id}\n"
        report += f"Total Congressmen: {len(self.congressmen)}\n"
        report += f"Congressmen: {[c.name for c in self.congressmen]}\n"
        return report

    def print_layers_summary(self) -> None:
        """Print a summary of layers for each congressman."""
        for congressman in self.congressmen:
            layer_names = [layer.name for layer in congressman.layers]
            logger.info("%s Layers: %s", congressman.name, layer_names)
