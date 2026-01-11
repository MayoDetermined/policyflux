import pytest
import torch
import numpy as np

from behavioral_sim.agents.congressman import CongressAgent
from behavioral_sim.engine.compiler import CompiledSystem
from behavioral_sim.engine.runner import CongressRunner
from behavioral_sim.network import DynamicNetwork, HomophilyInfluence, LeaderBoostInfluence, CommitteeInfluence
from behavioral_sim.context import PublicRegime


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


def _make_compiled(device: torch.device) -> CompiledSystem:
    actors = [
        CongressAgent(1, {"ideology_multidim": [0.1], "party": "A"}),
        CongressAgent(2, {"ideology_multidim": [-0.2], "party": "B"}),
    ]
    base_adj = torch.zeros((2, 2), dtype=torch.float32, device=device)
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
    compiled = _make_compiled(torch.device("cpu"))
    runner = CongressRunner(compiled)
    result = runner.run_monte_carlo(n_simulations=1, steps=1, show_tqdm=False)
    assert isinstance(result, dict)
    assert "probability_of_passing" in result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_runner_smoke_gpu():
    compiled = _make_compiled(torch.device("cuda"))
    runner = CongressRunner(compiled)
    result = runner.run_monte_carlo(n_simulations=1, steps=1, show_tqdm=False)
    assert isinstance(result, dict)
    assert "probability_of_passing" in result


def test_influence_functions_chain():
    device = torch.device("cpu")
    base = torch.ones((3, 3), dtype=torch.float32, device=device)
    X = torch.tensor([[0.0], [0.5], [-0.5]], device=device)
    committee = torch.zeros_like(base)
    committee[0, 1] = 1.0
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
    assert torch.allclose(torch.diag(G), torch.zeros(3, device=device))
    assert torch.isfinite(G).all()
