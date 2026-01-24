from dataclasses import dataclass
from typing import List

from policyflux.utils.reports import craft_a_bar
from ..core.bill_template import Bill
from ..core.congress_model_template import CongressModel
#import importlib
#pfrandom = importlib.import_module("policyflux.random")
from policyflux import pfrandom
from policyflux.logging_config import logger

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
        # Use package RNG manager so all modules draw from the same source.
        pfrandom.set_seed(self.seed)
        for _ in range(self.n_simulations):
            result = self.congress_model.cast_votes(self.bill)
            self.results.append(result)
        return self.results
    
    def get_pretty_votes(self) -> None:
        avg_votes_for = sum(self.results) / len(self.results)
        craft_a_bar(
            data = [avg_votes_for, len(self.congress_model.congressmen) - avg_votes_for],
            labels = ['Votes For', 'Votes Against'],
            title = 'Average Voting Results',
            xlabel = 'Vote Type',
            ylabel = 'Number of Votes'
        )

    def __str__(self):
        if not self.results:
            return "No simulations run yet"
        
        total_congressmen = len(self.congress_model.congressmen)
        avg_votes_for = sum(self.results) / len(self.results)
        avg_votes_against = total_congressmen - avg_votes_for
        
        return f"Simulations: {self.n_simulations}\nAverage votes for: {avg_votes_for:.2f}\nAverage votes against: {avg_votes_against:.2f}"