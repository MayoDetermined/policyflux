from policyflux.engines.engine_template import MPEngine
from sequential_monte_carlo import SequentialMonteCarlo

from sessions_mamagment import Session

class ParallelMonteCarlo(SequentialMonteCarlo, MPEngine):
    """Parallel Monte Carlo engine that runs multiple simulations of the congress model in parallel.
    This engine is useful for estimating the distribution of outcomes based on the initial conditions of the model.
    Useful for stochastic models and when you want to get a sense of the variability in outcomes."""

    def __init__(self, session_params: Session, processes: int = 1) -> None:
        SequentialMonteCarlo.__init__(self, session_params)
        MPEngine.__init__(self, session_params, processes)

    def _run_simulation(self) -> None:
        # Run a single simulation and store the result
        result = self._simulate_congress()
        self.results.append(result)