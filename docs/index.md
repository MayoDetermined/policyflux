# PolicyFlux Documentation

PolicyFlux is a Python package for simulating legislative decision-making under different institutional and political conditions.

## Start here

- New to the project: [Getting Started](getting-started.md)
- Need public API summary: [API Overview](api-overview.md)
- Want internals and module layout: [Architecture](architecture.md)
- Publishing a release: [Release Guide](release.md)

## Core concepts

- **Policy space**: an n-dimensional space used to represent policy positions.
- **Actors and bills**: legislators and bills located in the same policy space.
- **Decision layers**: composable influences that modify vote probabilities.
- **Aggregation strategy**: method used to combine multiple layer outputs.
- **Simulation engine**: deterministic or Monte Carlo runtime for voting sessions.

## Typical workflow

1. Create an `IntegrationConfig` (directly or from a preset).
2. Build an engine with `build_engine(config)`.
3. Execute simulation with `engine.run()`.
4. Read summary metrics such as pass rate and vote outcomes.