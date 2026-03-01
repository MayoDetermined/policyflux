# PolicyFlux

<div align="center">

**A Python library for modeling legislative behavior, voting dynamics, and institutional political systems**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/policyflux.svg)](https://pypi.org/project/policyflux/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-early%20development-orange.svg)](https://github.com/piotrpawelec/policyflux)

</div>

## Project status

PolicyFlux is in active early development. It is suitable for research workflows and prototyping, while parts of the API may still evolve between versions.

Recommended usage:
- ✅ Research and academic experiments
- ✅ Scenario and policy prototyping
- ⚠️ Workloads where occasional API changes are acceptable
- ❌ Stable long-term production systems

## What PolicyFlux models

PolicyFlux simulates legislative outcomes by combining:
- actors with preferences in an n-dimensional policy space,
- bills represented in the same policy space,
- composable influence layers (public opinion, lobbying, media, party discipline, agenda control),
- institutional presets (presidential, parliamentary, semi-presidential),
- deterministic and Monte Carlo execution engines.

Typical users:
- political scientists studying institutional dynamics,
- data scientists running Monte Carlo experiments,
- policy analysts exploring what-if scenarios,
- educators teaching decision-making in political systems.

## Installation

Requirements:
- Python 3.10+
- `pip`

Install from PyPI:

```bash
pip install policyflux
```

Install from source:

```bash
git clone https://github.com/piotrpawelec/policyflux.git
cd policyflux
pip install -e .
```

Optional extras:

```bash
# Neural layers (PyTorch)
pip install -e ".[torch]"

# Text encoders
pip install -e ".[text-encoders]"

# Notebook/examples support
pip install -e ".[examples]"

# Developer tooling
pip install -e ".[dev]"

# All common extras
pip install -e ".[torch,text-encoders,examples,dev]"
```

Verify installation:

```bash
python -c "import policyflux; print(policyflux.__version__)"
```

## Quick start

```python
from policyflux import IntegrationConfig, LayerConfig, build_engine

config = IntegrationConfig(
    num_actors=50,
    policy_dim=2,
    iterations=100,
    seed=12345,
    layer_config=LayerConfig(
        include_ideal_point=True,
        include_public_opinion=True,
        include_party_discipline=True,
        public_support=0.60,
        party_discipline_strength=0.5,
    ),
)

engine = build_engine(config)
engine.run()

print(f"Pass rate: {engine.pass_rate:.1%}")
print(f"Accepted: {engine.accepted_bills}, Rejected: {engine.rejected_bills}")
```

## Main API entry points

- `IntegrationConfig` – top-level simulation parameters.
- `LayerConfig` – layer toggles and layer strengths/intensities.
- `AdvancedActorsConfig` – lobbyists, whips, and executive-specific behavior.
- `build_engine(config)` – build the configured simulation engine.
- `create_presidential_config(...)`, `create_parliamentary_config(...)`, `create_semi_presidential_config(...)` – ready-to-run presets.

## Architecture at a glance

```text
policyflux/
├── core/            # abstractions, contexts, strategies
├── layers/          # composable decision layers
├── engines/         # deterministic + Monte Carlo execution
├── integration/     # config, builders, presets, registry
├── toolbox/         # concrete actor/bill/congress implementations
├── data_processing/ # text and embedding helpers
└── utils/           # reports and utility helpers
```

## Documentation

- [Documentation index](docs/index.md)
- [Getting started](docs/getting-started.md)
- [API overview](docs/api-overview.md)
- [Architecture](docs/architecture.md)
- [Release guide](docs/release.md)

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff check policyflux/
mypy policyflux/
```

For contribution workflow and standards, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
