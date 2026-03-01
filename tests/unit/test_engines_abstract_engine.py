from policyflux.engines.abstract_engine import Engine, MPEngine
from policyflux.engines.session_management import Session


class _DummyEngine(Engine):
    def __init__(self, results: list[int] | int, members: int) -> None:
        self.results = results
        self.congress_model = type("Congress", (), {"congressmen": [object() for _ in range(members)]})()

    def run(self) -> list[int] | int:
        return self.results


class _DummyMPEngine(MPEngine):
    def __init__(self, session_params: Session, processes: int = 1) -> None:
        super().__init__(session_params=session_params, processes=processes)
        self.calls = 0

    def _run_simulation(self) -> None:
        self.calls += 1


def test_get_pretty_votes_handles_int_result(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_bar(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("policyflux.engines.abstract_engine.craft_a_bar", _fake_bar)

    engine = _DummyEngine(results=7, members=10)
    engine.get_pretty_votes()

    assert captured["data"] == [7, 3]
    assert captured["labels"] == ["Votes For", "Votes Against"]


def test_get_pretty_votes_handles_list_result_and_empty_list(monkeypatch) -> None:
    captured: list[list[float]] = []

    def _fake_bar(**kwargs):
        captured.append(kwargs["data"])

    monkeypatch.setattr("policyflux.engines.abstract_engine.craft_a_bar", _fake_bar)

    _DummyEngine(results=[2, 4, 6], members=10).get_pretty_votes()
    _DummyEngine(results=[], members=10).get_pretty_votes()

    assert captured[0] == [4.0, 6.0]
    assert captured[1] == [0, 10]


def test_mpengine_run_starts_and_joins_all_processes(monkeypatch) -> None:
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

    engine = _DummyMPEngine(
        session_params=Session(n=1, seed=1, bill=None, description="x", congress_model=None),
        processes=3,
    )
    engine.run()

    assert engine.calls == 3
    assert events.count("start") == 3
    assert events.count("join") == 3
