# Architecture

PolicyFlux is organized around composable abstractions for actors, bills, decision layers, and simulation engines.

## High-level module layout

```text
policyflux/
├── core/
├── layers/
├── engines/
├── integration/
├── toolbox/
├── data_processing/
└── utils/
```

## `core/`

Foundational abstractions and mechanics:

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

## `engines/`

Simulation execution backends:

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

## `toolbox/`

Concrete implementations of abstract types:

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