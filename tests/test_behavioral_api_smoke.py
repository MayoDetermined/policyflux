import pytest
import numpy as np
import tensorflow as tf

from policyflux.behavioral_sim.agents.congressman import CongressAgent
from policyflux.behavioral_sim.engine.compiler import CompiledSystem
from policyflux.behavioral_sim.engine.runner import CongressRunner
from policyflux.behavioral_sim.network import DynamicNetwork, HomophilyInfluence, LeaderBoostInfluence, CommitteeInfluence
from policyflux.behavioral_sim.context import PublicRegime


class _StubEngine:
    def __init__(self, template=None):
        self.template = template

    def run_monte_carlo(self, **kwargs):
        return {"probability_of_passing": 0.42, "expected_votes": 0.0}


class _StubCongress:
    def __init__(self):
        self.base_adj_matrix = np.zeros((2, 2), dtype=np.float32)
        self.current_adj_matrix = self.base_adj_matrix.copy()
        self.adj_matrix = self.base_adj_matrix.copy()


def _make_compiled(device: str) -> CompiledSystem:
    actors = [
        CongressAgent(1, {"ideology_multidim": [0.1], "party": "A"}),
        CongressAgent(2, {"ideology_multidim": [-0.2], "party": "B"}),
    ]
    with tf.device(device):
        base_adj = tf.zeros((2, 2), dtype=tf.float32)
    dyn = DynamicNetwork(
        base_adj=base_adj,
        influence_functions=[HomophilyInfluence(beta=1.0)],
        device=device,
    )
    congress = _StubCongress()
    engine = _StubEngine(template=congress)
    return CompiledSystem(
        actors=actors,
        regime=PublicRegime("stable"),
        congress=congress,
        dynamic_network=dyn,
        engine=engine,
        device=device,
    )


def test_runner_smoke_cpu():
    compiled = _make_compiled("/CPU:0")
    runner = CongressRunner(compiled)
    result = runner.run_monte_carlo(n_simulations=1, steps=1, show_tqdm=False)
    assert isinstance(result, dict)
    assert "probability_of_passing" in result


def test_runner_smoke_gpu_or_skip():
    has_gpu = bool(tf.config.list_physical_devices("GPU"))
    compiled = _make_compiled("/GPU:0" if has_gpu else "/CPU:0")
    runner = CongressRunner(compiled)
    result = runner.run_monte_carlo(n_simulations=1, steps=1, show_tqdm=False)
    assert isinstance(result, dict)
    assert "probability_of_passing" in result


def test_influence_functions_chain():
    device = "/CPU:0"
    with tf.device(device):
        base = tf.ones((3, 3), dtype=tf.float32)
        X = tf.constant([[0.0], [0.5], [-0.5]], dtype=tf.float32)
        committee = tf.zeros_like(base)
        committee = tf.tensor_scatter_nd_update(committee, indices=[[0, 1]], updates=[1.0])
    dyn = DynamicNetwork(
        base_adj=base,
        influence_functions=[
            HomophilyInfluence(beta=0.5),
            LeaderBoostInfluence(boost=1.5),
            CommitteeInfluence(committee_matrix=committee, weight=0.25),
        ],
        device=device,
    )
    G = dyn.compute(X)
    assert G.shape == base.shape
    diag_zero = tf.reduce_all(tf.equal(tf.linalg.diag_part(G), 0.0))
    assert bool(diag_zero.numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(G)).numpy())




