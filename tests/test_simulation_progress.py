import numpy as np

from policyflux.engine.symulations import CongressEngine
from policyflux.congress.actors import CongressMan
from policyflux.congress.congress import TheCongress
from policyflux.public_opinion.regime import PublicRegime


def make_minimal_engine():
    # Build a tiny, deterministic congress template
    actors = [
        CongressMan(1, {"ideology_multidim": [0.1], "party": "Democratic", "political_capital": 3.0}),
        CongressMan(2, {"ideology_multidim": [-0.2], "party": "Republican", "political_capital": 3.0}),
        CongressMan(3, {"ideology_multidim": [0.0], "party": "Democratic", "political_capital": 3.0}),
    ]
    adj = np.zeros((3, 3), dtype=float)
    regime = PublicRegime(scenario="stable")
    cong = TheCongress(actors, adj, regime)
    engine = CongressEngine(cong, use_gpu=False)
    return engine


def test_progress_callback_called_and_counts_match():
    engine = make_minimal_engine()
    events = []

    def cb(event):
        events.append(event)

    report = engine.run_monte_carlo(n_simulations=2, steps=3, progress_callback=cb, show_tqdm=False)

    # Expect a start, 2*(3 step_complete) events, 2 simulation_complete, and one final complete
    phases = [e.get("phase") for e in events if isinstance(e, dict) and "phase" in e]
    assert "start" in phases
    assert phases.count("simulation_complete") == 2
    assert phases.count("step_complete") == 2 * 3
    assert "complete" in phases

    # Report should be a dict with expected keys
    assert "probability_of_passing" in report
    assert "expected_votes" in report


def test_show_tqdm_flag_does_not_raise():
    engine = make_minimal_engine()
    # Should not raise whether tqdm installed or not
    rpt = engine.run_monte_carlo(n_simulations=1, steps=1, show_tqdm=True)
    assert isinstance(rpt, dict)
    assert "probability_of_passing" in rpt


def test_slow_step_reports():
    from policyflux import config as cfg

    old = getattr(cfg, "SLOW_STEP_THRESHOLD_SECONDS", None)
    # Force threshold to zero so any non-zero step time is reported
    cfg.SLOW_STEP_THRESHOLD_SECONDS = 0.0
    try:
        engine = make_minimal_engine()
        events = []

        def cb(ev):
            events.append(ev)

        engine.run_monte_carlo(n_simulations=1, steps=1, progress_callback=cb, show_tqdm=False)

        assert any(isinstance(e, dict) and e.get("phase") == "slow_step" for e in events)
    finally:
        if old is None:
            delattr(cfg, "SLOW_STEP_THRESHOLD_SECONDS")
        else:
            cfg.SLOW_STEP_THRESHOLD_SECONDS = old


def test_show_small_steps_reports():
    # Avoid expensive community detection on larger graphs by patching a fast stub
    from policyflux.congress import policyflux.congress as cong_mod
    cong_mod.TheCongress.detect_communities = lambda self, method="louvain": np.zeros(len(self.actors), dtype=int)

    engine = make_minimal_engine()
    events = []

    def cb(ev):
        events.append(ev)

    engine.run_monte_carlo(n_simulations=1, steps=1, progress_callback=cb, show_small_steps=True)

    assert any(isinstance(e, dict) and e.get("phase") == "step_timing" for e in events)


def test_show_step_and_component_tqdm_does_not_raise():
    from policyflux.congress import policyflux.congress as cong_mod
    cong_mod.TheCongress.detect_communities = lambda self, method="louvain": np.zeros(len(self.actors), dtype=int)

    engine = make_minimal_engine()

    # Should not raise regardless of tqdm availability
    rpt = engine.run_monte_carlo(n_simulations=1, steps=1, show_tqdm=True, show_step_tqdm=True, show_component_tqdm=True)
    assert isinstance(rpt, dict)
    assert "probability_of_passing" in rpt




