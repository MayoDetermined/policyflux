from abc import ABC, abstractmethod
from multiprocessing.dummy import Process
from typing import Union, List

from policyflux.utils.reports.bar_charts import craft_a_bar
from .sessions_mamagment import Session


class Engine(ABC):
    """Abstract base class for simulation engines.

    Subclasses must set ``self.congress_model`` and ``self.results`` in their
    ``__init__`` so that :meth:`get_pretty_votes` can visualise the outcome.
    """

    @abstractmethod
    def run(self) -> Union[List[int], int]:
        """Run the simulation engine and return results."""
        pass

    def get_pretty_votes(self) -> None:
        """Render a bar chart of the latest simulation results."""
        results = self.results  # type: ignore[attr-defined]
        congress = self.congress_model  # type: ignore[attr-defined]
        total = len(congress.congressmen)

        if isinstance(results, int):
            avg_votes_for = results
        else:
            avg_votes_for = sum(results) / len(results) if results else 0

        craft_a_bar(
            data=[avg_votes_for, total - avg_votes_for],
            labels=['Votes For', 'Votes Against'],
            title='Average Voting Results',
            xlabel='Vote Type',
            ylabel='Number of Votes',
        )


class MPEngine(ABC):
    """Abstract base class for multi-processing simulation engines."""

    def __init__(self, session_params: Session, processes: int = 1) -> None:
        self.session_params = session_params
        self.processes = processes

    @abstractmethod
    def _run_simulation(self) -> None:
        """Run a single simulation. Implemented by subclasses."""
        pass

    def run(self) -> None:
        process_list = []
        for _ in range(self.processes):
            p = Process(target=self._run_simulation)
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()
