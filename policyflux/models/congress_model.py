from typing import List, Optional
from random import randint

from policyflux.models.advanced_actors.lobby import SequentialLobbyer
from policyflux.models.advanced_actors.speaker import SequentialSpeaker
from policyflux.models.advanced_actors.whips import SequentialWhip

from ..core.layer_template import Layer
from ..core.congress_model_template import CongressModel
from .actors import SequentialVoter
from ..core.bill_template import Bill
from ..core.id_generator import get_id_generator

class SequentialCongressModel(CongressModel):
    """
    Congress model using sequential voters with dependency-injected layers.
    """
    
    def __init__(self, id: Optional[int] = None) -> None:
        if id is None:
            id = get_id_generator().generate_model_id()
                
        super().__init__(id)
        self.lobbysts: List[SequentialLobbyer] = []
        self.congressmen: List[SequentialVoter] = []
        self.whips: List[SequentialWhip] = []
        self.speaker: Optional[SequentialSpeaker] = None
    
    def cast_votes(self, bill: Bill, bill_space=None, **context) -> int:
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
                    if hasattr(layer, 'space') and layer.space:
                        voter_dim = len(layer.space)
                        if voter_dim != bill_dim:
                            raise ValueError(
                                f"Dimension mismatch: bill has {bill_dim} dimensions, "
                                f"but {congressman.name} has ideal point with {voter_dim} dimensions"
                            )
        
        votes_for = 0
        for congressman in self.congressmen:
            if congressman.vote(bill, bill_space, **context):
                votes_for += 1
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
    
    def add_n_congressmen(self, n: int, layers: list[Layer] = None) -> None:
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

    def pop_whip(self) -> Optional[SequentialWhip]:
        """Remove and return the last added Whip."""
        if self.whips:
            return self.whips.pop()
        return None 

    def compile(self) -> None:
        """Compile/validate the model structure."""
        # Validate all congressmen have at least one layer
        for congressman in self.congressmen:
            if not congressman.layers:
                print(f"Warning: {congressman.name} has no decision layers")
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
            print(f"{congressman.name} Layers: {layer_names}")