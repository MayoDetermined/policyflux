# Architecture

PolicyFlux is organized around composable abstractions for actors, bills, decision layers, and simulation engines.

## High-level module layout

```text
policyflux/
<<<<<<< HEAD
├── core/               # abstractions, typing, contexts, strategies, DI container
├── layers/             # composable decision layers (7 built-in + neural + ERGM)
├── engines/            # deterministic + sequential/parallel Monte Carlo
├── integration/        # config, builders, presets, registry, fluent API
│   ├── builders/       # engine, congress, layer, actor, mechanics builders
│   └── presets/        # presidential, parliamentary, semi-presidential, 10 countries
├── toolbox/            # concrete actor/bill/congress/executive implementations
│   └── special_actors/ # lobbyist, whip, speaker, president
├── math_models/        # ERGM, lobbying ERGM, Tullock contest
├── model/              # TF-style Sequential + Functional model API
├── scenarios/          # comparative systems, sweeps, country comparison
├── data_processing/    # text vectorization and encoding
└── utils/              # bar chart and pie chart reporting
=======
├── core/
├── layers/
├── engines/
├── integration/
├── toolbox/
├── data_processing/
└── utils/
>>>>>>> 28724a8eb17f6081daef9177c037673d899cf2a9
```

## `core/`

Foundational abstractions and mechanics:

<<<<<<< HEAD
- **Abstract types**: `Bill`, `CongressMember`, `ComplexActor`, `CongressModel`, `Executive`, `ExecutiveActor`, `Layer`
- **Executive enum**: `ExecutiveType` (presidential, parliamentary, semi-presidential)
- **Policy typing**: `PolicyPosition` (frozen, coordinates in [0, 1]), `PolicySpace` (mutable wrapper), `PolicyVector`, `UtilitySpace`
- **Aggregation strategies**: `SequentialAggregation`, `AverageAggregation`, `WeightedAggregation`, `MultiplicativeAggregation`
- **Voting strategies**: `ProbabilisticVoting`, `DeterministicVoting`, `SoftVoting`
- **Immutable contexts**: `VotingContext` (per-vote state), `SimulationContext` (per-run state)
- **Service container**: lightweight dependency injection (`register_factory`, `register_singleton`, `resolve`)
- **ID generator**: thread-safe singleton with counters for actors, layers, bills, models

## `layers/`

Decision layers that transform vote probabilities. Each layer inherits from `Layer` and implements `call(bill_position, **kwargs) -> float`.

| Layer class | Key parameters | Behavior |
|---|---|---|
| `IdealPointLayer` | `space`, `status_quo` | Sigmoid of utility delta (status quo vs bill) |
| `PublicOpinionLayer` | `support_level` | 50/50 blend of base probability and public support |
| `LobbyingLayer` | `intensity`, lobbyists | Asymmetric pressure from aggregated lobbyist strength x stance |
| `MediaPressureLayer` | `pressure` [-1, 1] | Signed media pressure with speaker/president adjustments |
| `PartyDisciplineLayer` | `discipline_base_strength`, `party_line_support` | Blend of base prob and whip-aggregated party line |
| `GovernmentAgendaLayer` | `pm_party_strength` | Strong discipline on government bills, passthrough on private bills |
| `LobbyingERGMPLayer` | `ergmp_model`, `intensity` | Network-aware lobbying using ERGM bipartite graph |
| `SequentialNeuralLayer` | `input_size`, `architecture` | Trainable PyTorch sequential neural network (optional) |

Additional:
- `IdealPointEncoderDF` -- neural encoder mapping DataFrame features to ideal point space (requires torch)
- `IdealPointTextEncoder` -- hybrid TF-IDF + sentence embedding encoder (requires torch + sentence-transformers)
=======
- actor, bill, executive, congress interfaces
- policy typing utilities
- voting and aggregation strategies
- immutable contexts
- lightweight dependency container

## `layers/`

Decision layers that transform vote probabilities:

- ideal point
- public pressure
- lobbying
- media pressure
- party discipline
- government agenda
- neural layers (optional dependency)
>>>>>>> 28724a8eb17f6081daef9177c037673d899cf2a9

## `engines/`

Simulation execution backends:

<<<<<<< HEAD
| Engine | Description |
|---|---|
| `DeterministicEngine` | Single-run, seed-controlled. Calls `cast_votes()` once, returns `int` |
| `SequentialMonteCarlo` | Runs `n` iterations sequentially, returns `list[int]` |
| `ParallelMonteCarlo` | Multi-process Monte Carlo using `multiprocessing.dummy.Process` |

All engines expose after `run()`: `pass_rate`, `accepted_bills`, `rejected_bills`, `get_pretty_votes()`.

**Session management**: `Session` is a frozen dataclass holding `n` (iterations), `seed`, `bill`, `description`, and `congress_model`.

## `integration/`

High-level API and composition entry points. This is the primary public interface for most users.

### Configuration (`config.py`)

- `IntegrationConfig` -- top-level dataclass (num_actors, policy_dim, iterations, seed, layer_config, actors_config, aggregation_strategy). Supports `from_flat()`, `with_flat()`, and fluent `with_*` methods.
- `LayerConfig` -- layer toggle flags (`include_ideal_point`, `include_public_opinion`, etc.) plus corresponding parameter fields.
- `AdvancedActorsConfig` -- lobbyist count/strength/stance, whip count/strength, speaker agenda, presidential approval/veto, PM strength/confidence, government bill rate.
- `Settings` -- Pydantic settings with `POLICYFLUX_` env prefix for seed and log level.

### Builders (`builders/`)

- `build_engine(config)` -- main entry: creates ServiceContainer, sets seed, builds session, returns `SequentialMonteCarlo`.
- `build_session(config)` -- creates Bill + Congress, returns `Session`.
- `build_congress(config)` -- builds advanced actors, aggregation, executive, creates voters with layers.
- `build_layers(config, lobbyists, whips)` -- resolves layers from flags or registry names.
- `build_executive(config)` -- creates `PresidentialExecutive`, `ParliamentaryExecutive`, or `SemiPresidentialExecutive`.
- `build_advanced_actors(config)` -- creates lobbyists, whips, speaker, president.
- `build_aggregation_strategy(config)` -- maps strategy string to concrete class.

### Registry (`registry.py`)

Layer registry with default entries for all built-in layers. Custom layers can be registered via `register_layer(name, factory)` and looked up with `build_layer_by_name(name, context)`.

### Fluent API (`fluent.py`)

`PolicyFlux` -- method-chaining builder with sub-builders:
- `LayerBuilder` -- toggle/configure individual layers
- `ExecutiveBuilder` -- set presidential/parliamentary/semi-presidential parameters
- `ActorBuilder` -- configure lobbyists, whips, speaker

### Presets (`presets/`)

- **System presets**: `create_presidential_config()`, `create_parliamentary_config()`, `create_semi_presidential_config()`
- **One-liner runners**: `run_presidential()`, `run_parliamentary()`, `run_semi_presidential()` (build + run + return engine)
- **Engine builders**: `presidential_engine()`, `parliamentary_engine()`, `semi_presidential_engine()` (build without running)
- **Default constants**: `PRESIDENTIAL_DEFAULT`, `PARLIAMENTARY_DEFAULT`, `SEMI_PRESIDENTIAL_DEFAULT`
- **Country presets**: 10 real-world parliament configurations (UK, US, Germany, France, Italy, Poland, Sweden, Spain, Australia, Canada)
=======
- deterministic engine
- sequential Monte Carlo
- parallel Monte Carlo
- session management

Engines orchestrate repeated sessions and expose summary metrics after `run()`.

## `integration/`

High-level API and composition entry points:

- configuration objects
- builders (engine, layers, actors, congress)
- preset factories for political systems
- registry for layer wiring

This module is the primary public integration point for most users.
>>>>>>> 28724a8eb17f6081daef9177c037673d899cf2a9

## `toolbox/`

Concrete implementations of abstract types:

<<<<<<< HEAD
- `SequentialVoter` -- congress member with injected layers and aggregation strategy
- `SequentialBill` -- bill with position, pass/fail tracking, government/confidence flags
- `SequentialCongressModel` -- congress with voters, executive, layers, special actors, compilation

### Executive systems (`executive_systems.py`)

| System | Class | Key mechanics |
|---|---|---|
| Presidential | `PresidentialExecutive` + `President` | Distance-based veto; override threshold (default 2/3) |
| Parliamentary | `ParliamentaryExecutive` + `PrimeMinister` | Government bill rate; confidence votes; PM falls on loss |
| Semi-presidential | `SemiPresidentialExecutive` | Cohabitation detection; weaker veto (3/5 override) |

### Multi-chamber parliament (`parliament_models.py`)

`MultiChamberParliamentModel` supports unicameral, bicameral, and multi-chamber configurations:

- **Passage thresholds**: simple majority, absolute majority, 3/5 supermajority, 2/3 supermajority
- **Upper chamber powers**: full veto, suspensive veto, override by lower, advisory
- **Ping-pong rounds**: configurable navette for suspensive veto
- **Budget bill exemption**: upper chamber bypassed for money bills

### Special actors (`special_actors/`)

| Actor | Class | Key fields |
|---|---|---|
| Lobbyist | `SequentialLobbyist` | `influence_strength` [0, 1], `stance` [-1, 1] |
| Speaker | `SequentialSpeaker` | `agenda_support` [0, 1] |
| Whip | `SequentialWhip` | `discipline_strength`, `party_line_support` |
| President | `SequentialPresident` | `approval_rating` [0, 1], veto capability |

## `math_models/`

Formal mathematical models for network and contest analysis:

| Model | Description |
|---|---|
| `ExponentialRandomGraphModel` | Network generation with density, transitivity, homophily; adjacency matrix, clustering, components |
| `LobbyingERGMPModel` | Bipartite ERGM for lobbyist-legislator networks; lobbyist reach, legislator exposure |
| `TullockContest` | Rent-seeking contest: win probabilities, payoffs, waste, efficiency, equilibrium simulation, sensitivity analysis |

## `model/`

TensorFlow/Keras-style model construction API:

- `Sequential` -- linear stack of layer specs. Supports `add()`, `pop()`, and pipe operator (`model | L.IdealPoint()`).
- `Model` -- functional API with computation graph. Constructor: `Model(inputs=Input(...), outputs=node)`.
- `Input` -- symbolic input node specifying `policy_dim` and `num_actors`.
- Layer specs: `L.IdealPoint()`, `L.PublicOpinion(support)`, `L.Lobbying(intensity)`, `L.MediaPressure(pressure)`, `L.PartyDiscipline(strength, line_support)`, `L.GovernmentAgenda(pm_strength)`.
- Both model types share: `compile(executive, aggregation, ...)`, `run(iterations, seed)`, `summary()`, `get_config()`, `from_config()`.

## `scenarios/`

Pre-built comparative experiment runners:

| Scenario | Returns | Description |
|---|---|---|
| `comparative_systems.run()` | `list[SystemResult]` | Compare presidential vs parliamentary vs semi-presidential |
| `country_comparison.run()` | `list[CountryResult]` | Compare bill passage across 10 real-world parliaments |
| `lobbying_sweep.run()` | `list[LobbyingPoint]` | Sweep lobbying intensity from 0.0 to 1.0 |
| `party_discipline_sweep.run()` | `dict[str, list[DisciplinePoint]]` | Sweep discipline for pro-bill and anti-bill lines |
| `veto_player_sweep.run()` | `dict[str, list[VetoPoint]]` | Sweep executive approval for presidential and semi-presidential |

`run_all(**kwargs)` executes all five scenarios.

## `data_processing/`

Text processing and encoder tools:

- `DataProcessor` -- abstract base with `fit()` and `process()` methods.
- `SimpleTextVectorizer` -- tokenization, vocabulary building, and tensor conversion (requires torch).

## `utils/`

Reporting and visualization helpers:

- `craft_a_bar(data, labels, title, xlabel, ylabel)` -- matplotlib bar chart
- `bake_a_pie(data, labels, title)` -- matplotlib pie chart

## Runtime flow

A standard simulation run follows this sequence:

1. Build `IntegrationConfig` (direct, from_flat, preset, fluent, or Model API).
2. Resolve concrete components through integration builders (congress, layers, executive, session).
3. For each iteration: generate random bill position, evaluate active decision layers per actor.
4. Aggregate layer outputs with the selected aggregation strategy.
5. Apply voting strategy (probabilistic, deterministic, or soft).
6. Process executive actions (veto, confidence vote, cohabitation).
7. Collect and expose aggregated outcome metrics (`pass_rate`, `accepted_bills`, `rejected_bills`).

## Design principles

- **Composability**: layers, strategies, and executive systems are independently swappable.
- **Reproducibility**: all simulation paths are seed-aware. `set_seed()` controls the global RNG.
- **Extensibility**: custom layers can be registered via `register_layer()`. Custom actors, bills, and executives can extend the abstract base classes.
- **Separation of concerns**: abstractions (`core/`) are separated from implementations (`toolbox/`), configuration (`integration/`) from execution (`engines/`).
- **Multiple API levels**: from one-liner functions to full dataclass control, users choose the right level of detail for their use case.
=======
- actor models
- bill models
- congress models
- executive systems
- special actors

## `data_processing/`

Text processing and encoder tools for deriving policy representations from textual corpora.

## `utils/`

Auxiliary utilities, including reporting helpers and runtime utility modules.

## Runtime flow

At a high level, a standard simulation run follows this sequence:

1. Build `IntegrationConfig` (`direct`, `from_flat`, or `create_*_config`).
2. Resolve concrete components through integration builders.
3. For each actor/bill/session, evaluate active decision layers.
4. Aggregate layer outputs with the selected aggregation strategy.
5. Apply voting strategy (deterministic or probabilistic).
6. Collect and expose aggregated outcome metrics.

## Design principles

- **Composability**: layers and strategies are swappable.
- **Reproducibility**: seed-aware simulation workflows.
- **Extensibility**: custom layers and models can be registered and integrated.
- **Separation of concerns**: abstractions (`core`) separated from concrete implementations (`toolbox`).
>>>>>>> 28724a8eb17f6081daef9177c037673d899cf2a9
