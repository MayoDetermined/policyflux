# PolicyFlux

<div align="center">

**An advanced Python library for modeling legislative processes, parliamentary behavior, and political dynamics**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/policyflux.svg)](https://pypi.org/project/policyflux/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-early%20development-orange.svg)](https://github.com/piotrpawelec/policyflux)

</div>

---

## ⚠️ Project Status

**PolicyFlux is in early development.**
The library is already useful for research experiments, but its API is still evolving and may change between versions.

### Recommended use
- ✅ Research and academic experiments
- ✅ Rapid prototyping of political science models
- ⚠️ Expect breaking API changes
- ❌ Not recommended for production systems

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Quick Start API](#-quick-start-api)
- [Architecture](#-architecture)
- [Documentation](#-documentation)
- [Development](#-development)
- [License](#-license)

---

## 🎯 Overview

PolicyFlux helps you build and run simulations of legislative voting under complex political conditions.

You can model interactions between:
- Legislators (actors) with ideological preferences
- Bills in a multi-dimensional policy space
- Influence layers (public pressure, lobbying, media, party discipline, agenda control)
- Political systems (presidential, parliamentary, semi-presidential)

It is designed for:
- **Political scientists**: legislative behavior and institutional comparisons
- **Data scientists**: Monte Carlo workflows and optional ML integration
- **Policy analysts**: scenario-based what-if simulations
- **Educators**: teaching institutional dynamics with reproducible experiments

---

## 🚀 Key Features

### Legislative behavior modeling
- Multi-dimensional policy space
- Utility-based voting behavior
- Deterministic and Monte Carlo simulation engines

### Decision layers
- Ideal point preferences
- Public pressure
- Lobbying influence
- Media pressure
- Party discipline
- Government agenda control
- Optional neural layer (PyTorch)

### Aggregation strategies
- Sequential (default)
- Average
- Weighted
- Multiplicative

### Political system presets
- Presidential
- Parliamentary
- Semi-presidential

---

## 💻 Installation

### Requirements
- Python 3.10+
- `pip` (or another Python package manager)

### From PyPI

```bash
pip install policyflux
```

### From source

```bash
git clone https://github.com/piotrpawelec/policyflux.git
cd policyflux
pip install -e .
```

### Optional extras

```bash
# PyTorch support for neural layers
pip install -e ".[torch]"

# Text encoders
pip install -e ".[text-encoders]"

# Developer tools
pip install -e ".[dev]"

# All extras
pip install -e ".[torch,text-encoders,dev]"
```

### Verify installation

```bash
python -c "import policyflux; print(policyflux.__file__)"
```

---

## ⚡ Quick Start

```python
from policyflux import build_engine, IntegrationConfig, LayerConfig

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
engine.run_simulation()
print(engine)
```

More examples are available in [docs/getting-started.md](docs/getting-started.md).

---

## 🔌 Quick Start API

### Configuration

| Class | Purpose |
|---|---|
| `IntegrationConfig` | Top-level simulation config: actors, dimensions, iterations, seed |
| `LayerConfig` | Toggle and tune decision layers (ideal point, lobbying, media, etc.) |
| `AdvancedActorsConfig` | Configure lobbyists, whips, speaker, executive actors |

### Preset factories

```python
from policyflux import (
    create_presidential_config,    # US-style: presidential veto, override threshold
    create_parliamentary_config,   # UK-style: PM strength, confidence votes
    create_semi_presidential_config,  # France-style: president + PM cohabitation
)

config = create_presidential_config(
    num_actors=100, policy_dim=2, iterations=200, seed=42,
    president_approval=0.5, veto_override_threshold=2/3,
)
```

### Build and run

```python
from policyflux import build_engine

engine = build_engine(config)
engine.run_simulation()
print(engine)           # formatted summary with pass rate, mean votes, std
```

### Layers

Built-in layers that affect each legislator's vote probability:

| Layer | Config flag | Key parameter |
|---|---|---|
| `IdealPointLayer` | `include_ideal_point` | actor's position in policy space |
| `PublicOpinionLayer` | `include_public_opinion` | `public_support` (0-1) |
| `LobbyingLayer` | `include_lobbying` | `lobbying_intensity` (0-1) |
| `MediaPressureLayer` | `include_media_pressure` | `media_pressure` (0-1) |
| `PartyDisciplineLayer` | `include_party_discipline` | `party_discipline_strength` (0-1) |
| `GovernmentAgendaLayer` | `include_government_agenda` | `government_agenda_pm_strength` (0-1) |

### Aggregation strategies

Control how layer outputs are combined per actor:

```python
config = IntegrationConfig(
    aggregation_strategy="sequential",  # default: each layer modifies previous output
    # also: "average", "weighted", "multiplicative"
)
```

### Voting strategies

| Strategy | Behavior |
|---|---|
| `ProbabilisticVoting` | Monte Carlo: `random() < prob` (default) |
| `DeterministicVoting` | Threshold: `prob >= 0.5` |
| `SoftVoting` | Returns raw probability (for ensemble use) |

### Parliament presets

```python
from policyflux.toolbox import create_polish_parliament, create_uk_parliament, list_presets

print(list_presets())  # all available country presets

parliament = create_polish_parliament()
```

Available: `create_uk_parliament`, `create_us_congress`, `create_german_parliament`, `create_french_parliament`, `create_italian_parliament`, `create_polish_parliament`, `create_swedish_parliament`, `create_spanish_parliament`, `create_australian_parliament`, `create_canadian_parliament`.

### Low-level builders

For fine-grained control, individual builders are available:

```python
from policyflux import build_session, build_bill, build_congress, build_layers
```

Full API details: [docs/api-overview.md](docs/api-overview.md).

---

## 🏗️ Architecture

```text
policyflux/
├── core/              # Base abstractions, contexts, voting and aggregation strategies
├── layers/            # Decision layers affecting vote probabilities
├── engines/           # Deterministic and Monte Carlo engines
├── integration/       # High-level config, builders, and presets
├── toolbox/           # Concrete actor, bill, congress, executive implementations
├── data_processing/   # Text processing and encoders
└── utils/             # Reporting helpers and utilities
```

See [docs/architecture.md](docs/architecture.md) for a deeper breakdown.

---

## 📖 Documentation

- [Documentation index](docs/index.md)
- [Getting started guide](docs/getting-started.md)
- [Architecture guide](docs/architecture.md)
- [API overview](docs/api-overview.md)

---

## 🛠️ Development

### Install with development extras

```bash
pip install -e ".[dev,torch,text-encoders]"
```

### Run tests

```bash
pytest
pytest -v
```

### Lint and type-check

```bash
ruff check policyflux/
mypy policyflux/
```

---

## 📄 License

PolicyFlux is released under the [MIT License](LICENSE).

---

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/piotrpawelec/policyflux/issues)
- Discussions: [GitHub Discussions](https://github.com/piotrpawelec/policyflux/discussions)
- Email: pawelecpiotr404@gmail.com