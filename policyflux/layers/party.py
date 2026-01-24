from policyflux.models.advanced_actors.whips import SequentialWhip

## TO DO: Complete implementation

class PartyDisciplineLayer:
    def __init__(self,
                 party_whips: SequentialWhip = None,
                 disicpiline_base_strength: float = 0.5) -> None:
        self.whips: SequentialWhip = party_whips
        self.discipline_base_strength: float = disicpiline_base_strength