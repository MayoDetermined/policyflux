from abc import ABC, abstractmethod
from multiprocessing.dummy import Process

from policyflux.utils.reports.bar_charts import craft_a_bar
from sessions_mamagment import Session

class Engine(ABC):
    """Abstract base class for simulation engines."""
    def __init__(self) -> None:
        self.results: list[int] | int = None

    @abstractmethod
    def run(self) -> None:
        """Run the simulation engine."""
        pass

    def get_pretty_votes(self) -> None:
        if isinstance(self.results, int):
            avg_votes_for = self.results
        craft_a_bar(
            data = [avg_votes_for, len(self.congress_model.congressmen) - avg_votes_for],
            labels = ['Votes For', 'Votes Against'],
            title = 'Average Voting Results',
            xlabel = 'Vote Type',
            ylabel = 'Number of Votes'
        )

class MPEngine(ABC):
    """Abstract base class for multi-processing simulation engines."""

    def __init__(self, session_params: Session, processes: int = 1) -> None:
        self.session_params = session_params
        self.processes = processes

    @abstractmethod
    def _run_simulation(self) -> None:
        """Run a single simulation. This method should be implemented by subclasses."""
        pass

    def run(self) -> None:
        process_list = []
        for _ in range(self.processes):
            p = Process(target=self._run_simulation)
            p.start()
            process_list.append(p)
        
        for p in process_list:
            p.join()