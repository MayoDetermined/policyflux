from typing import List

from policyflux import pfrandom
from policyflux.logging_config import logger

from .sessions_mamagment import Session
from .engine_template import Engine
from ..core.bill_template import Bill
from ..core.congress_model_template import CongressModel

class DeterministicEngine(Engine):
    
    """Deterministic engine that runs multiple simulations of the congress model.
    This engine is useful for estimating the distribution of outcomes based on the initial conditions of the model.
    Useful for deterministic models."""
    
    def __init__(self, session_params: Session) -> None:
        self.congress_model: CongressModel = session_params.congress_model
        self.bill: Bill = session_params.bill
        self.results: int = 0
        self.seed: int = session_params.seed

    def run(self) -> int:
        # Ensure deterministic randomness for voting across runs
        # Use package RNG manager so all modules draw from the same source.
        pfrandom.set_seed(self.seed)
        result = self.congress_model.cast_votes(self.bill)
        self.results = result
        return self.results