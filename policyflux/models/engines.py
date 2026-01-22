from dataclasses import dataclass
from typing import List
import random
from ..core.bill_template import Bill
from ..core.congress_model_template import CongressModel

@dataclass(frozen=True)
class Session:
    n: int
    seed: int
    bill: Bill
    description: str
    congress_model: CongressModel

class SequentialMonteCarlo:
    def __init__(self, session_params: Session) -> None:
        self.n_simulations: int = session_params.n
        self.congress_model: CongressModel = session_params.congress_model
        self.bill: Bill = session_params.bill
        self.results: List[int] = []
        self.seed: int = session_params.seed

    def run_simulation(self) -> List[int]:
        # Ensure deterministic randomness for voting across runs
        random.seed(self.seed)
        for _ in range(self.n_simulations):
            result = self.congress_model.cast_votes(self.bill)
            self.results.append(result)
        return self.results

    def __str__(self):
        if not self.results:
            return "No simulations run yet"
        
        total_congressmen = len(self.congress_model.congressmen)
        avg_votes_for = sum(self.results) / len(self.results)
        avg_votes_against = total_congressmen - avg_votes_for
        
        return f"Simulations: {self.n_simulations}\nAverage votes for: {avg_votes_for:.2f}\nAverage votes against: {avg_votes_against:.2f}"