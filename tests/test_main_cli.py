import tensorflow as tf

def _dummy_compiled(monkeypatch):
    from policyflux.behavioral_sim.engine.compiler import CompiledSystem
    from policyflux.behavioral_sim.context import PublicRegime

    class DummyEngine:
        def __init__(self):
            self.template = None

        def run_monte_carlo(self, **kwargs):
            return {"probability_of_passing": 0.5}

    compiled = CompiledSystem(
        actors=[],
        regime=PublicRegime("stable"),
        congress=None,
        dynamic_network=None,
        engine=DummyEngine(),
        device="/CPU:0",
    )

    def fake_compile(*args, **kwargs):
        return compiled

    monkeypatch.setattr("policyflux.behavioral_sim.engine.compiler.CongressCompiler.compile", fake_compile)
    return compiled


def test_run_full_simulation_forwards_tqdm_flags(monkeypatch):
    from main import run_full_simulation
    from policyflux.behavioral_sim.engine.runner import CongressRunner

    compiled = _dummy_compiled(monkeypatch)
    called = {}

    def fake_run(self, n_simulations=None, steps=None, progress_callback=None, show_tqdm=False, show_small_steps=False, show_step_tqdm=False, show_component_tqdm=False, **kwargs):
        called["args"] = dict(
            show_tqdm=show_tqdm,
            show_small_steps=show_small_steps,
            show_step_tqdm=show_step_tqdm,
            show_component_tqdm=show_component_tqdm,
        )
        return {"probability_of_passing": 0.5}

    monkeypatch.setattr(CongressRunner, "run_monte_carlo", fake_run, raising=True)

    report, _, _ = run_full_simulation(n_simulations=1, steps=1, show_tqdm=True, show_small_steps=True, show_step_tqdm=True, show_component_tqdm=True)

    assert isinstance(report, dict)
    assert "args" in called
    assert called["args"]["show_step_tqdm"] is True
    assert called["args"]["show_component_tqdm"] is True


def test_run_full_simulation_defaults_are_verbose(monkeypatch):
    from main import run_full_simulation
    from policyflux.behavioral_sim.engine.runner import CongressRunner

    compiled = _dummy_compiled(monkeypatch)
    called = {}

    def fake_run(self, n_simulations=None, steps=None, progress_callback=None, show_tqdm=False, show_small_steps=False, show_step_tqdm=False, show_component_tqdm=False, **kwargs):
        called["args"] = dict(
            show_tqdm=show_tqdm,
            show_small_steps=show_small_steps,
            show_step_tqdm=show_step_tqdm,
            show_component_tqdm=show_component_tqdm,
        )
        return {"probability_of_passing": 0.6}

    monkeypatch.setattr(CongressRunner, "run_monte_carlo", fake_run, raising=True)

    report, _, _ = run_full_simulation(n_simulations=1, steps=1)
    assert isinstance(report, dict)
    assert called["args"]["show_tqdm"] is True
    assert called["args"]["show_small_steps"] is True
    assert called["args"]["show_step_tqdm"] is True
    assert called["args"]["show_component_tqdm"] is True




