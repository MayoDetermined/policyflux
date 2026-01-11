# PolicyFlux

PolicyFlux packages the congressional dynamics simulator into an installable Python package with a TensorFlow/Keras-like compile → fit → simulate workflow.

## Installation

```bash
pip install .
# Optional extras
pip install .[gpu]   # CuPy + cuML (GPU acceleration, Linux-friendly)
pip install .[dev]   # pytest and development tools
```

## Quickstart

Python API:

```python
import policyflux as pf

# High-level convenience run
report, congress, actors = pf.run_full_simulation(use_cache=True, n_simulations=50, steps=10)

# Lower-level, Keras-like control
sim = pf.CongressSimulator().compile(scenario="crisis", use_hmm_state=True)
sim.fit(use_cache=True)
report = sim.simulate(n_simulations=20, steps=8)
```

CLI (mirrors legacy `main.py`):

```bash
policyflux --simulations 10 --steps 6
```

## Notes
- `main.py` remains as an example/legacy CLI and is wrapped by the `policyflux` console script.
- GPU acceleration via CuPy/cuML is optional; install the `[gpu]` extra only on platforms where binaries are available.
- Key dependencies: PyTorch, NumPy, pandas, scikit-learn, networkx, python-louvain, pyvoteview, tqdm, requests, matplotlib, hmmlearn.
