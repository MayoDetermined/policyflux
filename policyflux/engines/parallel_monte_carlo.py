from .abstract_engine import MPEngine
from .sequential_monte_carlo import SequentialMonteCarlo
from .session_management import Session


class ParallelMonteCarlo(SequentialMonteCarlo, MPEngine):  # type: ignore[misc]
    """Parallel Monte Carlo engine that runs multiple simulations of the congress model in parallel.
    This engine is useful for estimating the distribution of outcomes based on the initial conditions of the model.
    Useful for stochastic models and when you want to get a sense of the variability in outcomes."""

    def __init__(self, session_params: Session, processes: int = 1) -> None:
        SequentialMonteCarlo.__init__(self, session_params)
        MPEngine.__init__(self, session_params, processes)

    def _run_simulation(self) -> None:
        result = self.congress_model.cast_votes(self.bill)
        self.results.append(result)
