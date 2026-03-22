# PolicyFlux

<div align="center">

**A Python library for modeling legislative behavior, voting dynamics, and institutional political systems**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/policyflux.svg)](https://pypi.org/project/policyflux/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-early%20development-orange.svg)](https://github.com/MayoDetermined/policyflux)

</div>

## Project status

PolicyFlux is in active early development. It is suitable for research workflows and prototyping, while parts of the API may still evolve between versions.

## What PolicyFlux models

PolicyFlux simulates legislative outcomes by combining:

- **Actors** with preferences in an n-dimensional policy space (legislators, lobbyists, whips, speakers, executives).
- **Bills** represented as positions in the same policy space.
- **Decision layers** that influence vote probabilities -- ideal point proximity, public opinion, lobbying pressure, media pressure, party discipline, government agenda, and neural layers.
- **Aggregation strategies** that combine layer outputs (sequential, average, weighted, multiplicative).
- **Voting strategies** -- probabilistic (Monte Carlo), deterministic (threshold), or soft (probability passthrough).
- **Institutional presets** for presidential, parliamentary, and semi-presidential systems, including 10 country-specific parliament presets (UK, US, Germany, France, Italy, Poland, Sweden, Spain, Australia, Canada).
- **Executive systems** with veto power, confidence votes, and cohabitation dynamics.
- **Multi-chamber parliaments** with full veto, suspensive veto, override, and advisory models.
- **Mathematical models** -- Exponential Random Graph Models (ERGM), ERGM-based lobbying networks, and Tullock contest rent-seeking.
- **Deterministic, sequential Monte Carlo, and parallel Monte Carlo** simulation engines.

## Installation

```bash
pip install policyflux
```

From source:

```bash
git clone https://github.com/MayoDetermined/policyflux.git
cd policyflux
pip install -e .
```

Optional extras:

```bash
pip install -e ".[torch]"           # Neural layers (PyTorch)
pip install -e ".[text-encoders]"   # Sentence-transformers text encoding
pip install -e ".[examples]"        # Jupyter notebook support
pip install -e ".[dev]"             # Development tools (pytest, ruff, mypy)
```

## Quick start

### Dataclass-based configuration

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
votes = engine.run()

num_actors = config.num_actors
passed = sum(1 for v in votes if v > num_actors / 2)
print(f"Pass rate: {passed / len(votes):.1%}")
print(f"Accepted: {passed}, Rejected: {len(votes) - passed}")
```

### One-liner runners

Run a full simulation in a single call:

```python
from policyflux import run_presidential

votes = run_presidential(num_actors=100, policy_dim=2, iterations=200, seed=42)
passed = sum(1 for v in votes if v > 50)
print(f"Pass rate: {passed / len(votes):.1%}")
```

Available runners: `run_presidential`, `run_parliamentary`, `run_semi_presidential`.

### Fluent builder API

```python
from policyflux import PolicyFlux

engine = (
    PolicyFlux()
    .actors(100)
    .policy_dim(2)
    .iterations(200)
    .seed(42)
    .with_ideal_point()
    .with_public_opinion(support=0.6)
    .with_lobbying(intensity=0.3)
    .presidential(approval_rating=0.5)
    .build()
)
votes = engine.run()
passed = sum(1 for v in votes if v > 50)
print(f"Pass rate: {passed / len(votes):.1%}")
```

### TensorFlow-style model API

```python
from policyflux.model import Sequential
from policyflux.model import layers as L

model = Sequential(num_actors=100, policy_dim=2)
model.add(L.IdealPoint())
model.add(L.PublicOpinion(support=0.6))
model.add(L.PartyDiscipline(strength=0.5, line_support=0.7))

model.compile(executive="presidential", aggregation="sequential")
results = model.run(iterations=200, seed=42)
model.summary()
```

Functional API variant:

```python
from policyflux.model import Model, Input
from policyflux.model import layers as L

bill = Input(policy_dim=2, num_actors=100)
x = L.IdealPoint()(bill)
x = L.PublicOpinion(support=0.6)(x)
x = L.Lobbying(intensity=0.4)(x)

model = Model(inputs=bill, outputs=x)
model.compile(executive="parliamentary", aggregation="average")
results = model.run(iterations=200, seed=42)
```

### Flat configuration

```python
from policyflux import IntegrationConfig, build_engine

config = IntegrationConfig.from_flat(
    num_actors=120,
    policy_dim=3,
    iterations=250,
    include_public_opinion=True,
    public_support=0.62,
    include_lobbying=True,
    n_lobbyists=3,
    lobbyist_strength=0.7,
    aggregation_strategy="average",
)

engine = build_engine(config)
engine.run()
```

### Compare institutional systems

```python
from policyflux import build_engine
from policyflux import create_presidential_config, create_parliamentary_config

eng1 = build_engine(create_presidential_config(num_actors=100, iterations=200, seed=42))
eng2 = build_engine(create_parliamentary_config(num_actors=100, iterations=200, seed=42))

votes1 = eng1.run()
votes2 = eng2.run()

print(f"Presidential pass rate: {sum(1 for v in votes1 if v > 50) / len(votes1):.1%}")
print(f"Parliamentary pass rate: {sum(1 for v in votes2 if v > 50) / len(votes2):.1%}")
```

### Country-specific parliament presets

These return `MultiChamberParliamentModel` instances for structural bicameral simulation:

```python
from policyflux.integration.presets.parliament_presets import (
    create_uk_parliament, create_us_congress, list_presets,
)
from policyflux.toolbox import SequentialBill

uk = create_uk_parliament()   # Commons (650) + Lords (800), suspensive veto
us = create_us_congress()     # House (435) + Senate (100), full veto

# Simulate a bill vote
bill = SequentialBill()
bill.make_random_position(dim=2)
result = uk.cast_votes(bill)
print(f"Passed: {result.passed}, Rounds: {result.rounds}")
```

Available: UK, US, Germany, France, Italy, Poland, Sweden, Spain, Australia, Canada. Use `list_presets()` to see all names.

### Built-in scenario runners

```python
from policyflux.scenarios import comparative_systems, lobbying_sweep

# Compare presidential vs parliamentary vs semi-presidential
results = comparative_systems.run(num_actors=100, iterations=300, seed=42)
for r in results:
    print(f"{r.system}: passage rate {r.passage_rate:.1%}")

# Sweep lobbying intensity from 0 to 1
points = lobbying_sweep.run(num_actors=100, n_steps=10, seed=42)
for p in points:
    print(f"Intensity {p.lobbying_intensity:.1f}: passage rate {p.passage_rate:.1%}")
```

Available scenarios: `comparative_systems`, `lobbying_sweep`, `party_discipline_sweep`, `veto_player_sweep`, `country_comparison`.

### Mathematical models

```python
from policyflux import import_models
math = import_models()

# Tullock contest (rent-seeking)
contest = math.TullockContest(n_contestants=5, prize_value=1.0, r=0.5)
contest.set_expenditure(0, 0.3)
contest.set_expenditure(1, 0.5)
probs = contest.compute_win_probabilities()

# ERGM network generation
ergm = math.ExponentialRandomGraphModel(n_nodes=20)
adjacency = ergm.generate(seed=42)
print(f"Network density: {ergm.get_density():.3f}")

# Lobbying ERGM (bipartite lobbyist-legislator network)
lobby_net = math.LobbyingERGMPModel(n_lobbyists=5, n_legislators=50)
lobby_net.generate(seed=42)
print(f"Avg lobbyist reach: {lobby_net.get_average_lobbyist_reach():.1f}")
```

## Key features

### Decision layers

| Layer | Parameter | Effect |
|---|---|---|
| `IdealPointLayer` | -- | Vote based on distance between actor and bill in policy space |
| `PublicOpinionLayer` | `support_level` | Blend public support into vote probability |
| `LobbyingLayer` | `intensity` | Aggregate lobbyist pressure on legislators |
| `MediaPressureLayer` | `pressure` | Directional media framing influence |
| `PartyDisciplineLayer` | `discipline_strength`, `party_line_support` | Whip-enforced party-line voting |
| `GovernmentAgendaLayer` | `pm_party_strength` | PM/cabinet agenda influence on government bills |
| `SequentialNeuralLayer` | custom architecture | Trainable PyTorch neural layer (optional) |
| `LobbyingERGMPLayer` | ERGM model | Network-aware lobbying via ERGM graphs |

### Aggregation strategies

| Strategy | Behavior |
|---|---|
| `SequentialAggregation` | Layers modify output of the previous one (chained) |
| `AverageAggregation` | Simple mean of all layer outputs |
| `WeightedAggregation` | Weighted sum (weights must sum to 1.0) |
| `MultiplicativeAggregation` | Product of all layer outputs (veto effect) |

### Voting strategies

| Strategy | Behavior |
|---|---|
| `ProbabilisticVoting` | Stochastic: `random() < prob` (Monte Carlo) |
| `DeterministicVoting` | Threshold: `prob >= 0.5` |
| `SoftVoting` | Returns probability directly (for ensemble use) |

### Executive systems

| System | Key mechanics |
|---|---|
| Presidential | Veto power (distance-based), override threshold (default 2/3) |
| Parliamentary | PM influence on government bills, confidence votes, no veto |
| Semi-presidential | Cohabitation detection, weaker veto (3/5 override), dual executive |

### Special actors

| Actor | Role |
|---|---|
| `SequentialLobbyist` | Exerts directional pressure with configurable strength and stance |
| `SequentialWhip` | Enforces party discipline with configurable strength |
| `SequentialSpeaker` | Controls agenda support level |
| `SequentialPresident` | Presidential executive with approval rating and veto |

### Simulation engines

| Engine | Use case |
|---|---|
| `DeterministicEngine` | Single-run deterministic simulation |
| `SequentialMonteCarlo` | Repeated Monte Carlo iterations (main engine) |
| `ParallelMonteCarlo` | Multi-process Monte Carlo for large runs |

### Multi-chamber parliaments

Bicameral models with configurable chamber powers:

- **Full veto** -- both chambers must pass (e.g., US, Italy)
- **Suspensive veto** -- ping-pong rounds; lower chamber has final say (e.g., UK, France)
- **Override by lower** -- upper rejection can be overridden (e.g., Poland, Germany)
- **Advisory** -- upper chamber vote is recorded but non-binding

Money bill exemption supported (e.g., UK House of Lords).

## Architecture

```text
policyflux/
├── core/            # abstractions, typing, contexts, strategies, DI container
├── layers/          # composable decision layers (7 built-in + neural + ERGM)
├── engines/         # deterministic + sequential/parallel Monte Carlo
├── integration/     # config, builders, presets, registry, fluent API
│   ├── builders/    # engine, congress, layer, actor, mechanics builders
│   └── presets/     # presidential, parliamentary, semi-presidential, 10 countries
├── toolbox/         # concrete actor/bill/congress/executive implementations
│   └── special_actors/  # lobbyist, whip, speaker, president
├── math_models/     # ERGM, lobbying ERGM, Tullock contest
├── model/           # TF-style Sequential + Functional model API
├── scenarios/       # comparative systems, lobbying/discipline/veto sweeps
├── data_processing/ # text vectorization and encoding
└── utils/           # bar chart and pie chart reporting
```

## API entry points

| Entry point | Purpose |
|---|---|
| `IntegrationConfig` / `LayerConfig` / `AdvancedActorsConfig` | Simulation configuration |
| `IntegrationConfig.from_flat(...)` / `.with_flat(...)` | Flat-style configuration |
| `build_engine(config)` | Build simulation engine from config |
| `create_presidential_config(...)` | Presidential system preset |
| `create_parliamentary_config(...)` | Parliamentary system preset |
| `create_semi_presidential_config(...)` | Semi-presidential system preset |
| `run_presidential(...)` / `run_parliamentary(...)` / `run_semi_presidential(...)` | One-liner: build + run, returns `list[int]` of vote counts |
| `presidential_engine(...)` / `parliamentary_engine(...)` / `semi_presidential_engine(...)` | Build engine without running it |
| `PolicyFlux()` | Fluent builder API |
| `Sequential` / `Model` / `Input` | TF-style model API |
| `import_models()` | Lazy import of `math_models` (ERGM, Tullock) |
| `LAYER_REGISTRY` / `register_layer()` | Custom layer registration |
| `set_seed()` / `get_rng()` | Global seed and RNG management |
| `bake_a_pie()` / `craft_a_bar()` | Visualization (pie chart / bar chart) |

## Documentation

- [Getting started](docs/getting-started.md) -- installation, first simulation, all API styles
- [API overview](docs/api-overview.md) -- configuration objects, layers, engines, model API
- [Architecture](docs/architecture.md) -- module layout, runtime flow, design principles
- [Release guide](docs/release.md) -- version bumping, PyPI publishing

## Citation

If you use PolicyFlux in your research, please cite:

```bibtex
@software{policyflux,
  author    = {Pawelec, Piotr},
  title     = {PolicyFlux},
  year      = {2026},
  url       = {https://github.com/MayoDetermined/policyflux},
  version   = {0.1.0}
}
```

See also [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff check policyflux/
mypy policyflux/
```

For contribution workflow, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
