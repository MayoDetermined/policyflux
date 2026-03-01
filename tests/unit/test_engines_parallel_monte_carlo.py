from policyflux.engines.abstract_engine import MPEngine
from policyflux.engines.parallel_monte_carlo import ParallelMonteCarlo
from policyflux.engines.session_management import Session


class _DummyCongress:
    def __init__(self, votes_to_return: int = 3, members: int = 5) -> None:
        self.votes_to_return = votes_to_return
        self.calls = 0
        self.congressmen = [object() for _ in range(members)]

    def cast_votes(self, bill: object) -> int:
        self.calls += 1
        return self.votes_to_return


def _session(votes_to_return: int = 3) -> Session:
    congress = _DummyCongress(votes_to_return=votes_to_return)
    return Session(
        n=4,
        seed=42,
        bill=object(),
        description="parallel test",
        congress_model=congress,
    )


def test_parallel_monte_carlo_init_sets_parent_fields() -> None:
    engine = ParallelMonteCarlo(session_params=_session(), processes=2)

    assert engine.n_simulations == 4
    assert engine.seed == 42
    assert engine.processes == 2
    assert engine.results == []


def test_parallel_monte_carlo_run_simulation_appends_result() -> None:
    engine = ParallelMonteCarlo(session_params=_session(votes_to_return=7), processes=1)

    engine._run_simulation()

    assert engine.results == [7]


def test_parallel_monte_carlo_run_uses_sequential_parent_by_default() -> None:
    engine = ParallelMonteCarlo(session_params=_session(votes_to_return=5), processes=3)
    results = engine.run()

    assert results == [5, 5, 5, 5]


def test_parallel_monte_carlo_mpengine_run_starts_all_processes(monkeypatch) -> None:
    events: list[str] = []

    class _FakeProcess:
        def __init__(self, target):
            self.target = target

        def start(self) -> None:
            events.append("start")
            self.target()

        def join(self) -> None:
            events.append("join")

    monkeypatch.setattr("policyflux.engines.abstract_engine.Process", _FakeProcess)

    engine = ParallelMonteCarlo(session_params=_session(votes_to_return=5), processes=3)
    MPEngine.run(engine)

    assert engine.results == [5, 5, 5]
    assert events.count("start") == 3
    assert events.count("join") == 3
