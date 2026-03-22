# Getting Started

This guide walks through installation and progressively more advanced simulation examples.

## 1) Installation

### From PyPI

```bash
pip install policyflux
```

### From source

```bash
git clone https://github.com/MayoDetermined/policyflux.git
cd policyflux
pip install -e .
```

### Optional extras

```bash
pip install -e ".[torch]"           # Neural layers (PyTorch)
pip install -e ".[text-encoders]"   # Sentence-transformers text encoding
pip install -e ".[examples]"        # Jupyter notebook support
pip install -e ".[dev]"             # Development tools (pytest, ruff, mypy)
```

## 2) Verify installation

```bash
python -c "import policyflux; print(policyflux.__version__)"
```

## 3) Minimal simulation

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
engine.run()

print(f"Pass rate: {engine.pass_rate:.1%}")
print(f"Accepted: {engine.accepted_bills}, Rejected: {engine.rejected_bills}")
```

## 4) Compare systems quickly

```python
from policyflux import build_engine
from policyflux import create_presidential_config, create_parliamentary_config

eng1 = build_engine(create_presidential_config(num_actors=100, iterations=200, seed=42))
eng2 = build_engine(create_parliamentary_config(num_actors=100, iterations=200, seed=42))

eng1.run()
eng2.run()

print(f"Presidential pass rate: {eng1.pass_rate:.1%}")
print(f"Parliamentary pass rate: {eng2.pass_rate:.1%}")
```

## 5) One-liner runners

Skip config and engine construction entirely:

```python
from policyflux import run_presidential, run_parliamentary, run_semi_presidential

result = run_presidential(num_actors=100, policy_dim=2, iterations=200, seed=42)
print(f"Pass rate: {result.pass_rate:.1%}")
```

## 6) Flat config

Map all parameters across `IntegrationConfig`, `LayerConfig`, and `AdvancedActorsConfig` in a single flat call:

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
print(engine.pass_rate)
```

## 7) Fluent builder API

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
    .lobbyists(2, strength=0.5)
    .aggregation("average")
    .build()
)
engine.run()
print(f"Pass rate: {engine.pass_rate:.1%}")
```

With sub-builders for grouped configuration:

```python
engine = (
    PolicyFlux()
    .actors(100)
    .policy_dim(2)
    .layers()
        .ideal_point()
        .public_opinion(support=0.6)
        .lobbying(intensity=0.3)
        .done()
    .executive()
        .presidential(approval_rating=0.5)
        .done()
    .special_actors()
        .lobbyists(3, strength=0.6)
        .whips(1, discipline_strength=0.7)
        .done()
    .build()
)
```

## 8) TensorFlow-style model API

### Sequential model

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

Pipe operator shorthand:

```python
model = Sequential(num_actors=100, policy_dim=2)
model = model | L.IdealPoint() | L.PublicOpinion(support=0.6)
model.compile(executive="presidential")
results = model.run(iterations=200)
```

### Functional model

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

## 9) Country-specific parliament presets

```python
from policyflux.integration.presets import create_uk_parliament, create_us_congress

uk = create_uk_parliament()   # Commons (650) + Lords (800), suspensive veto
us = create_us_congress()     # House (435) + Senate (100), full veto
```

Available presets: UK, US, Germany, France, Italy, Poland, Sweden, Spain, Australia, Canada.

## 10) Built-in scenario runners

```python
from policyflux.scenarios import comparative_systems, lobbying_sweep

# Compare passage rates across three institutional systems
results = comparative_systems.run(num_actors=100, iterations=300, seed=42)
for r in results:
    print(f"{r.system}: passage rate {r.passage_rate:.1%}")

# Sweep lobbying intensity from 0 to 1
points = lobbying_sweep.run(num_actors=100, n_steps=10, seed=42)
for p in points:
    print(f"Intensity {p.lobbying_intensity:.1f}: passage rate {p.passage_rate:.1%}")
```

Available scenarios: `comparative_systems`, `lobbying_sweep`, `party_discipline_sweep`, `veto_player_sweep`, `country_comparison`.

## 11) Mathematical models

```python
from policyflux import import_models
math = import_models()

# Tullock contest (rent-seeking competition)
contest = math.TullockContest(n_contestants=5, prize_value=1.0, r=0.5)
contest.set_expenditure(0, 0.3)
contest.set_expenditure(1, 0.5)
probs = contest.compute_win_probabilities()

# ERGM network generation
ergm = math.ExponentialRandomGraphModel(n_nodes=20)
adjacency = ergm.generate(seed=42)
print(f"Network density: {ergm.get_density():.3f}")
```

## 12) Custom layer registration

```python
from policyflux import register_layer, IntegrationConfig, LayerConfig, build_engine

# Register a custom layer factory
def my_factory(context):
    return MyCustomLayer(dim=context.policy_dim)

register_layer("my_layer", my_factory)

# Use it via layer_names
config = IntegrationConfig(
    num_actors=50,
    iterations=100,
    layer_config=LayerConfig(layer_names=["ideal_point", "my_layer"]),
)
engine = build_engine(config)
```

## 13) Development commands

```bash
pytest
pytest tests/unit -m unit
pytest tests/smoke -m smoke
ruff check policyflux/
mypy policyflux/
```

## Troubleshooting

- Ensure Python 3.10+ is active.
- Use a clean virtual environment when optional dependencies conflict.
- In Windows PowerShell, activate venv with `.venv\Scripts\Activate.ps1`.
- If imports fail, verify installation with:

```bash
python -c "import policyflux; print(policyflux.__file__)"
```

## Next step

Continue with [API Overview](api-overview.md) to explore the full public interface.
