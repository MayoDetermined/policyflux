"""Quick smoke test for regime dynamics and contextual adjacency."""
import os
import sys
# Ensure project root is on path when running standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from congress.actors import CongressMan
from congress.congress import TheCongress
from public_opinion.regime import PublicRegime
from policyflux._backend.symulations import CongressEngine

# Create small ensemble of synthetic actors
actors = []
for i in range(6):
    data = {
        "ideology_multidim": [np.random.uniform(-1, 1) for _ in range(3)],
        "loyalty": np.random.uniform(0.2, 0.9),
        "vulnerability": np.random.uniform(0.05, 0.3),
        "volatility": np.random.uniform(0.01, 0.2),
        "party": 'Republican' if i % 2 == 0 else 'Democratic',
        "presidential_support_score": np.random.uniform(0.0, 1.0)
    }
    actors.append(CongressMan(i, data))

# base adjacency: small random
adj = np.random.rand(len(actors), len(actors)) * 0.2
np.fill_diagonal(adj, 0.0)

regime = PublicRegime(scenario='polarized')
cong = TheCongress(actors, adj, regime)
engine = CongressEngine(cong, use_gpu=False)

def _print_progress(ev):
    try:
        print("PROGRESS:", ev)
    except Exception:
        pass

report = engine.run_monte_carlo(n_simulations=10, steps=5, progress_callback=_print_progress, show_tqdm=True)
print('REPORT:', report)
print('REGIME STATE:', getattr(cong.regime_engine, 'scenario', None))
