# import importlib
# pfrandom = importlib.import_module("policyflux.random")
from policyflux import pfrandom

from ..core.abstract_bill import Bill
from ..core.congress_model import CongressModel
from .abstract_engine import Engine
from .session_management import Session


class SequentialMonteCarlo(Engine):
    """Sequential Monte Carlo engine that runs multiple simulations of the congress model.
    This engine is useful for estimating the distribution of outcomes based on the initial conditions of the model.
    Useful for stochastic models and when you want to get a sense of the variability in outcomes."""

    def __init__(self, session_params: Session) -> None:
        self.n_simulations: int = session_params.n
        self.congress_model: CongressModel = session_params.congress_model
        self.bill: Bill = session_params.bill
        self.results: list[int] = []
        self.seed: int = session_params.seed

    def run(self) -> list[int]:
        # Ensure deterministic randomness for voting across runs
        # Use package RNG manager so all modules draw from the same source.
        pfrandom.set_seed(self.seed)
        for _ in range(self.n_simulations):
            result = self.congress_model.cast_votes(self.bill)
            self.results.append(result)
        return self.results

    def __str__(self) -> str:
        if not self.results:
            return "No simulations run yet"

        total_congressmen = len(self.congress_model.congressmen)
        avg_votes_for = sum(self.results) / len(self.results)
        avg_votes_against = total_congressmen - avg_votes_for

        return f"Simulations: {self.n_simulations}\nAverage votes for: {avg_votes_for:.2f}\nAverage votes against: {avg_votes_against:.2f}"
