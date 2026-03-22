# API Overview

This page covers all public entry points in the `policyflux` package.

## Configuration

### `IntegrationConfig`

Top-level simulation configuration (dataclass).

| Field | Default | Description |
|---|---|---|
| `num_actors` | `100` | Number of legislators |
| `policy_dim` | `4` | Policy space dimensionality |
| `iterations` | `300` | Monte Carlo iterations |
| `seed` | `42` | Random seed |
| `description` | `"PolicyFlux modular simulation"` | Human-readable label |
| `layer_config` | `LayerConfig()` | Layer toggles and parameters |
| `actors_config` | `AdvancedActorsConfig()` | Special actors and executive config |
| `aggregation_strategy` | `"sequential"` | One of: `sequential`, `average`, `weighted`, `multiplicative` |
| `aggregation_weights` | `None` | Weights for `weighted` strategy (must sum to 1.0) |

Construction methods:

```python
# Direct
config = IntegrationConfig(num_actors=50, policy_dim=2, iterations=100, seed=42)

# Flat (maps keys across IntegrationConfig + LayerConfig + AdvancedActorsConfig)
config = IntegrationConfig.from_flat(num_actors=50, include_lobbying=True, n_lobbyists=3)

# Fluent update
config = IntegrationConfig().with_flat(public_support=0.58, n_whips=1)

# Fluent sub-config
config = IntegrationConfig().with_layers(include_lobbying=True).with_actors(n_lobbyists=2)
```

### `LayerConfig`

| Field | Default | Description |
|---|---|---|
| `include_ideal_point` | `True` | Enable ideal-point layer |
| `include_public_opinion` | `True` | Enable public opinion layer |
| `include_lobbying` | `True` | Enable lobbying layer |
| `include_media_pressure` | `True` | Enable media pressure layer |
| `include_party_discipline` | `True` | Enable party discipline layer |
| `include_government_agenda` | `False` | Enable government agenda layer |
| `include_neural` | `False` | Enable neural layer (requires PyTorch) |
| `layer_names` | `None` | Build layers by registry name instead of flags |
| `layer_overrides` | `{}` | Per-layer parameter overrides |
| `public_support` | `0.5` | Public support level [0, 1] |
| `lobbying_intensity` | `0.0` | Lobbying intensity [0, 1] |
| `media_pressure` | `0.0` | Media pressure [-1, 1] |
| `party_line_support` | `0.5` | Party line support [0, 1] |
| `party_discipline_strength` | `0.5` | Whip enforcement strength [0, 1] |
| `government_agenda_pm_strength` | `0.6` | PM agenda influence [0, 1] |
| `neural_layer_factory` | `None` | Callable that creates a neural layer |

### `AdvancedActorsConfig`

| Field | Default | Description |
|---|---|---|
| `executive_type` | `PRESIDENTIAL` | `PRESIDENTIAL`, `PARLIAMENTARY`, or `SEMI_PRESIDENTIAL` |
| `n_lobbyists` | `0` | Number of lobbyist actors |
| `lobbyist_strength` | `0.5` | Each lobbyist's influence [0, 1] |
| `lobbyist_stance` | `1.0` | Lobbyist stance [-1, 1] |
| `n_whips` | `0` | Number of party whips |
| `whip_discipline_strength` | `0.5` | Whip enforcement strength [0, 1] |
| `whip_party_line_support` | `0.5` | Party line direction [0, 1] |
| `speaker_agenda_support` | `0.5` | Speaker's agenda support [0, 1] |
| `president_approval_rating` | `0.5` | Presidential approval [0, 1] |
| `veto_override_threshold` | `2/3` | Votes needed to override veto |
| `pm_party_strength` | `0.55` | PM's party strength [0, 1] |
| `confidence_threshold` | `0.5` | Votes needed to survive confidence vote |
| `government_bill_rate` | `0.7` | Fraction of bills that are government bills |

## Presets

### System presets

```python
from policyflux import (
    create_presidential_config,
    create_parliamentary_config,
    create_semi_presidential_config,
)

config = create_presidential_config(
    num_actors=100, policy_dim=2, iterations=200, seed=42,
    president_approval=0.5, veto_override_threshold=2/3,
)
```

Each preset accepts `num_actors`, `policy_dim`, `iterations`, `seed`, plus system-specific parameters.

### Country-specific parliament presets

These are **not** re-exported from `policyflux.integration.presets` -- import from the submodule directly:

```python
from policyflux.integration.presets.parliament_presets import (
    create_uk_parliament,
    create_us_congress,
    create_german_parliament,
    create_french_parliament,
    create_italian_parliament,
    create_polish_parliament,
    create_swedish_parliament,
    create_spanish_parliament,
    create_australian_parliament,
    create_canadian_parliament,
    list_presets,
    create_parliament,
    ParliamentPresetConfig,
)

# Returns MultiChamberParliamentModel, NOT an engine
uk = create_uk_parliament()

# Simulate a bill vote
from policyflux.toolbox import SequentialBill
bill = SequentialBill()
bill.make_random_position(dim=2)
result = uk.cast_votes(bill)  # -> ParliamentVoteResult
print(f"Passed: {result.passed}, Rounds: {result.rounds}")

# List all available presets
print(list_presets())  # ['australia', 'canada', 'france', ...]

# Create by name with extra kwargs
parliament = create_parliament("germany", consent_law=True)

# Customize chamber sizes via ParliamentPresetConfig
config = ParliamentPresetConfig(policy_dim=3, lower_house_size=100, upper_house_size=50)
small_uk = create_uk_parliament(config)
```

| Country | Lower | Upper | Upper powers |
|---|---|---|---|
| UK | Commons (650) | Lords (800) | Suspensive veto, budget bill exempt |
| US | House (435) | Senate (100) | Full veto |
| Germany | Bundestag (736) | Bundesrat (69) | Full veto or override (consent law toggle) |
| France | AN (577) | Senat (348) | Suspensive veto (navette) |
| Italy | Camera (400) | Senato (206) | Full veto (perfect bicameralism) |
| Poland | Sejm (460) | Senat (100) | Override by lower (231/460) |
| Sweden | Riksdag (349) | -- | Unicameral |
| Spain | Congreso (350) | Senado (265) | Suspensive veto |
| Australia | House (151) | Senate (76) | Full veto |
| Canada | Commons (338) | Senate (105) | Suspensive veto |

## Builders

### `build_engine(config) -> SequentialMonteCarlo`

Main entry point. Creates all components and returns a configured engine.

```python
from policyflux import build_engine, IntegrationConfig

engine = build_engine(IntegrationConfig(num_actors=50, iterations=100))
votes = engine.run()  # list[int] -- vote-for counts per iteration
passed = sum(1 for v in votes if v > 25)
print(f"Passage rate: {passed / len(votes):.1%}")
```

### Low-level builders

For finer control:

- `build_session(config) -> Session` -- seeds RNG, builds congress and bill
- `build_bill(config) -> SequentialBill` -- creates bill with random position
- `build_congress(config) -> SequentialCongressModel` -- builds actors, layers, aggregation, executive
- `build_layers(config, lobbyists, whips) -> list[Layer]` -- builds layers from config flags or registry
- `build_advanced_actors(config) -> tuple` -- creates lobbyists, whips, speaker, president
- `build_executive(config) -> Executive | None` -- creates the executive system
- `build_aggregation_strategy(config) -> AggregationStrategy` -- creates the aggregation strategy

## One-liner runners

Build and run a simulation in a single call. Returns `list[int]` (vote-for counts per iteration):

```python
from policyflux import run_presidential, run_parliamentary, run_semi_presidential

votes = run_presidential(num_actors=100, policy_dim=2, iterations=200, seed=42)
passed = sum(1 for v in votes if v > 50)
print(f"Passage rate: {passed / len(votes):.1%}")
```

Engine-only variants (build without running):

```python
from policyflux import presidential_engine, parliamentary_engine, semi_presidential_engine

engine = presidential_engine(num_actors=100, iterations=200, seed=42)
votes = engine.run()  # run manually
```

Default config constants: `PRESIDENTIAL_DEFAULT`, `PARLIAMENTARY_DEFAULT`, `SEMI_PRESIDENTIAL_DEFAULT`.

## Fluent builder API

`PolicyFlux` provides method-chaining for simulation construction.

### Basic usage

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
    .lobbyists(3, strength=0.6)
    .build()
)
votes = engine.run()
```

### Simulation parameters

| Method | Description |
|---|---|
| `.actors(n)` | Number of legislators |
| `.policy_dim(dim)` | Policy space dimensions |
| `.iterations(n)` | Monte Carlo iterations |
| `.seed(value)` | Random seed |
| `.description(text)` | Simulation label |

### Layer toggles

| Method | Description |
|---|---|
| `.with_ideal_point()` / `.without_ideal_point()` | Toggle ideal-point layer |
| `.with_public_opinion(support=)` / `.without_public_opinion()` | Toggle public opinion |
| `.with_lobbying(intensity=)` / `.without_lobbying()` | Toggle lobbying |
| `.with_media_pressure(pressure=)` / `.without_media_pressure()` | Toggle media pressure |
| `.with_party_discipline(line_support=, strength=)` / `.without_party_discipline()` | Toggle party discipline |
| `.with_government_agenda(pm_strength=)` / `.without_government_agenda()` | Toggle government agenda |
| `.with_neural(factory=)` | Enable neural layer |
| `.with_layer_override(name, **overrides)` | Override layer parameters |
| `.layer_names(names)` | Build layers by registry name |

### Executive system

| Method | Description |
|---|---|
| `.presidential(approval_rating=, veto_override=)` | Presidential system |
| `.parliamentary(pm_party_strength=, confidence_threshold=, government_bill_rate=)` | Parliamentary system |
| `.semi_presidential(approval_rating=, pm_party_strength=)` | Semi-presidential system |

### Special actors

| Method | Description |
|---|---|
| `.lobbyists(count, strength=, stance=)` | Add lobbyist actors |
| `.whips(count, discipline_strength=, party_line_support=)` | Add party whips |
| `.speaker(agenda_support=)` | Set speaker agenda support |

### Aggregation

| Method | Description |
|---|---|
| `.aggregation(strategy, weights=)` | Set aggregation strategy |

### Section sub-builders

For grouped configuration:

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

### Build methods

| Method | Description |
|---|---|
| `.build()` | Build and return a simulation engine |
| `.build_config()` / `.to_config()` | Return `IntegrationConfig` without building engine |

## Model API (TensorFlow-style)

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

### Layer specs

| Spec | Parameters | Description |
|---|---|---|
| `L.IdealPoint()` | -- | Distance-based voting |
| `L.PublicOpinion(support=0.5)` | `support` | Public support influence |
| `L.Lobbying(intensity=0.5)` | `intensity` | Lobbying pressure |
| `L.MediaPressure(pressure=0.5)` | `pressure` | Media framing |
| `L.PartyDiscipline(strength=0.5, line_support=0.5)` | `strength`, `line_support` | Party whip enforcement |
| `L.GovernmentAgenda(pm_strength=0.6)` | `pm_strength` | PM agenda influence |

### `compile()` parameters

| Parameter | Aliases | Description |
|---|---|---|
| `executive` | `"presidential"`, `"president"`, `"us"`, `"congress"` | Presidential system |
| | `"parliamentary"`, `"parliament"`, `"uk"`, `"westminster"` | Parliamentary system |
| | `"semi_presidential"`, `"france"`, `"cohabitation"` | Semi-presidential system |
| `aggregation` | `"sequential"`, `"average"`, `"weighted"`, `"multiplicative"` | Aggregation strategy |
| `n_lobbyists` | -- | Number of lobbyist actors |
| `n_whips` | -- | Number of party whips |

### Model methods

| Method | Description |
|---|---|
| `model.run(iterations=, seed=)` | Run simulation, return list of vote counts |
| `model(iterations=, seed=)` | Alias for `run()` |
| `model.summary()` | Print architecture and last-run statistics |
| `model.get_config()` | Serialize model to dict |
| `Model.from_config(config)` | Reconstruct model from dict |

## Decision layers

| Layer class | Key params | Description |
|---|---|---|
| `IdealPointLayer` | -- | Sigmoid of distance-based utility differential |
| `PublicOpinionLayer` | `support_level` | Blends base probability with public support (50/50) |
| `LobbyingLayer` | `intensity` | Aggregates lobbyist actors' strength and stance |
| `MediaPressureLayer` | `pressure` | Directional media influence [-1, 1] |
| `PartyDisciplineLayer` | `discipline_base_strength`, `party_line_support` | `(1-strength)*base_prob + strength*party_line` |
| `GovernmentAgendaLayer` | `pm_party_strength` | `base_prob*0.1 + pm_strength*0.9` for government bills |
| `SequentialNeuralLayer` | PyTorch architecture | Trainable neural network layer (optional, requires torch) |
| `LobbyingERGMPLayer` | ERGM model | Network-aware lobbying via bipartite ERGM graph |

### Text encoding layers

For deriving policy positions from text (requires `[text-encoders]` extra):

- `IdealPointEncoderDF` -- DataFrame-based encoder with `nn.Linear`
- `IdealPointTextEncoder` -- Hybrid TF-IDF + sentence-transformers encoder with trainable network

## Aggregation strategies

| Class | Behavior |
|---|---|
| `SequentialAggregation` | Each layer receives previous output as `base_prob` context (chained) |
| `AverageAggregation` | Simple mean of all layer outputs |
| `WeightedAggregation(weights)` | Weighted sum; weights must sum to 1.0 |
| `MultiplicativeAggregation` | Product of all outputs (veto-like effect) |

## Voting strategies

| Class | Behavior |
|---|---|
| `ProbabilisticVoting` | `random() < prob` -- stochastic (Monte Carlo) |
| `DeterministicVoting` | `prob >= 0.5` -- threshold-based |
| `SoftVoting` | Returns `prob` directly -- for ensemble aggregation |

## Executive systems

| Class | Key mechanics |
|---|---|
| `PresidentialExecutive` | Veto based on Euclidean distance (>0.5), override threshold (default 2/3) |
| `ParliamentaryExecutive` | PM influence on govt bills, confidence votes (PM leaves office if failed) |
| `SemiPresidentialExecutive` | Cohabitation detection (ideology distance >0.5), weaker veto (3/5 override) |

## Simulation engines

| Class | Description |
|---|---|
| `DeterministicEngine` | Single-run deterministic simulation |
| `SequentialMonteCarlo` | N iterations, returns list of vote counts |
| `ParallelMonteCarlo` | Multi-process Monte Carlo |

Engine attributes after `run()`:

- `results` -- `list[int]` (Monte Carlo engines) or `int` (deterministic engine) -- raw vote-for counts per iteration
- `n_simulations` -- number of Monte Carlo iterations (on `SequentialMonteCarlo` / `ParallelMonteCarlo`)
- `congress_model` -- the congress model used for voting
- `get_pretty_votes()` -- render bar chart of results

To compute passage rate from results:

```python
votes = engine.run()
num_actors = config.num_actors
passage_rate = sum(1 for v in votes if v > num_actors / 2) / len(votes)
```

## Scenario runners

```python
from policyflux.scenarios import comparative_systems, lobbying_sweep, party_discipline_sweep, veto_player_sweep, country_comparison
```

| Scenario | Returns | Sweeps |
|---|---|---|
| `comparative_systems.run(...)` | `list[SystemResult]` | Presidential vs parliamentary vs semi-presidential |
| `lobbying_sweep.run(...)` | `list[LobbyingPoint]` | Lobbying intensity from 0 to 1 |
| `party_discipline_sweep.run(...)` | `dict[str, list[DisciplinePoint]]` | Discipline strength, pro vs anti stances |
| `veto_player_sweep.run(...)` | `dict[str, list[VetoPoint]]` | Presidential approval from min to max |

## Mathematical models

```python
from policyflux import import_models
math = import_models()
```

| Class | Description |
|---|---|
| `TullockContest` | Rent-seeking competition. Nash equilibrium via best-response dynamics. HHI, efficiency, dissipation metrics. |
| `ExponentialRandomGraphModel` | Undirected network generation with density, transitivity, homophily parameters. Clustering and connected components. |
| `LobbyingERGMPModel` | Bipartite lobbyist-legislator network. Lobbyist reach, legislator exposure metrics. |

## Layer registry

Register custom layers to use with `layer_names` configuration:

```python
from policyflux import LAYER_REGISTRY, register_layer

def my_factory(context):
    return MyCustomLayer(dim=context.policy_dim)

register_layer("my_layer", my_factory)
```

Built-in registered names: `"ideal_point"`, `"public_opinion"`, `"lobbying"`, `"media_pressure"`, `"party_discipline"`, `"government_agenda"`.

## Multi-chamber parliaments

```python
from policyflux.toolbox import MultiChamberParliamentModel
```

Bicameral model with configurable chamber powers:

| Power | Behavior |
|---|---|
| `FULL_VETO` | Both chambers must pass |
| `SUSPENSIVE_VETO` | Ping-pong rounds; lower chamber gets final say |
| `OVERRIDE_BY_LOWER` | Lower chamber can override upper rejection |
| `ADVISORY` | Upper vote recorded but non-binding |

Additional features: passage thresholds (simple majority, absolute majority, 3/5 supermajority, 2/3 supermajority), money bill exemption.

## Exception hierarchy

```
PolicyFluxError
├── ConfigurationError       -- invalid configuration values
├── DimensionMismatchError   -- policy space dimension conflicts
├── BuildError               -- engine/component construction failures
│   └── RegistryError        -- layer registry lookup failures
├── SimulationError          -- runtime simulation errors
│   └── EngineNotConfiguredError -- engine used before configuration
├── OptionalDependencyError  -- missing optional packages (also ImportError)
└── ValidationError          -- general validation errors (also ValueError)
```

## Utilities

| Function | Description |
|---|---|
| `set_seed(seed)` | Set global RNG seed (`None` for non-deterministic) |
| `get_rng()` | Get the global `random.Random` instance |
| `get_settings()` | Get `Settings` (pydantic-settings, env prefix `POLICYFLUX_`) |
| `get_id_generator()` | Get singleton ID generator for actors, layers, bills, models |
| `bake_a_pie(data, labels, title)` | Render pie chart (matplotlib) |
| `craft_a_bar(data, labels, title, xlabel, ylabel)` | Render bar chart (matplotlib) |

## Notes

- Public API is still evolving in early-stage development.
- Prefer importing from `policyflux` package root for forward compatibility.
- For advanced extension points, inspect `policyflux/integration/registry.py` and builder modules.
