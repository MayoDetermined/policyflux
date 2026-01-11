import numpy as np

from engine.simulations import CongressEngine
from congress.actors import CongressMan
from congress.congress import TheCongress
from public_opinion.regime import PublicRegime


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


def test_votes_recorded_and_loyalty_changed():
    engine = make_minimal_engine()

    # Snapshot initial loyalties on template
    initial_loyalties = [float(a.loyalty) for a in engine.template.actors]

    engine.run_monte_carlo(n_simulations=1, steps=1, show_tqdm=False)

    # Retrieve last sim instance created during run
    sim = getattr(engine, '_last_sim_instance', None)
    assert sim is not None, "Engine should keep reference to last sim instance"

    # At least one actor should have recorded a vote
    assert any(len(a.recent_votes) >= 1 for a in sim.actors), "At least one actor should have a recorded vote"

    # At least one actor's loyalty should have changed from the initial template value
    post_loyalties = [float(a.loyalty) for a in sim.actors]
    assert any(pl != il for pl, il in zip(post_loyalties, initial_loyalties)), "Loyalty should change for at least one actor after voting"


def test_sim_gpu_context_attached():
    engine = make_minimal_engine()
    engine.run_monte_carlo(n_simulations=1, steps=1, show_tqdm=False)
    sim = getattr(engine, '_last_sim_instance', None)
    assert sim is not None
    # Ensure engine propagated GPU context and xp attribute
    assert hasattr(sim, 'xp') and hasattr(sim, 'use_gpu'), "Sim instance should have xp and use_gpu attributes set"
    assert sim.use_gpu == engine.use_gpu


def test_regime_pressure_propagates_to_rnn():
    import numpy as _np
    engine = make_minimal_engine()
    # Initialize RNNs for all actors so predict_ideology_rnn is used
    engine.init_rnn_for_all_actors()

    # Force 'crisis' scenario and deterministic noise (no shock)
    engine.template.regime_engine.scenario = 'crisis'
    engine.template.regime_engine._configure_scenario()

    # Monkeypatch numpy normal to return 0.0 to remove stochastic shocks
    old_normal = _np.random.normal
    _np.random.normal = lambda *args, **kwargs: 0.0

    # Patch CongressMan.predict_ideology_rnn to verify the pressure argument is passed through
    from congress.actors import CongressMan
    original_predict_ideology = CongressMan.predict_ideology_rnn
    captured = {"seen": False}

    def patched_predict_ideology(self, current_pressure):
        assert abs(float(current_pressure) - 0.8) < 1e-3, f"Expected crisis pressure 0.8, got {current_pressure}"
        captured["seen"] = True
        return original_predict_ideology(self, current_pressure)

    CongressMan.predict_ideology_rnn = patched_predict_ideology

    try:
        engine.run_monte_carlo(n_simulations=1, steps=1, show_tqdm=False)
        assert captured["seen"], "RNN predict should have been called with crisis pressure argument"
    finally:
        # Restore monkeypatches
        _np.random.normal = old_normal
        CongressMan.predict_ideology_rnn = original_predict_ideology
