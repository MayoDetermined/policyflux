# PolicyFlux Documentation

PolicyFlux is a Python package for simulating legislative decision-making under different institutional and political conditions.

## Start here

- New to the project: [Getting Started](getting-started.md)
- Need public API summary: [API Overview](api-overview.md)
- Want internals and module layout: [Architecture](architecture.md)
- Publishing a release: [Release Guide](release.md)

## Core concepts

- **Policy space** -- an n-dimensional space where each dimension represents a policy axis. Both actors and bills are positioned in this space.
- **Actors** -- legislators (`CongressMember`), lobbyists, whips, speakers, and executives, each with an ideology in the policy space.
- **Bills** -- legislative proposals positioned in the policy space. Each simulation iteration generates a random bill.
- **Decision layers** -- composable influences that modify vote probabilities:
  - *Ideal point* -- vote based on distance between actor preference and bill position
  - *Public opinion* -- blend public support level into the decision
  - *Lobbying* -- aggregate pressure from lobbyist actors
  - *Media pressure* -- directional media framing influence
  - *Party discipline* -- whip-enforced party-line voting
  - *Government agenda* -- PM/cabinet influence on government bills
  - *Neural* -- trainable PyTorch layer (optional)
  - *ERGM lobbying* -- network-aware lobbying via graph models
- **Aggregation strategies** -- method for combining multiple layer outputs: sequential (chained), average, weighted, or multiplicative (veto effect).
- **Voting strategies** -- how the aggregated probability becomes a vote: probabilistic (Monte Carlo), deterministic (threshold), or soft (raw probability).
- **Simulation engines** -- deterministic (single run), sequential Monte Carlo, or parallel Monte Carlo.
- **Executive systems** -- presidential (veto power), parliamentary (confidence votes, PM agenda), semi-presidential (cohabitation, dual executive).
- **Multi-chamber parliaments** -- bicameral models with full veto, suspensive veto, override, or advisory upper chambers, plus money bill exemptions.
- **Institutional presets** -- ready-made configurations for presidential, parliamentary, and semi-presidential systems, plus 10 country-specific parliaments (UK, US, Germany, France, Italy, Poland, Sweden, Spain, Australia, Canada).
- **Scenarios** -- built-in comparative experiments: system comparison, lobbying sweeps, party discipline sweeps, veto player analysis.
- **Mathematical models** -- ERGM for network generation, bipartite lobbying ERGM for lobbyist-legislator networks, Tullock contest for rent-seeking competition.

## Six ways to define a simulation

1. **Dataclass config** -- create `IntegrationConfig` with nested `LayerConfig` and `AdvancedActorsConfig`, then `build_engine(config)`.
2. **Flat config** -- `IntegrationConfig.from_flat(...)` maps all parameters in a single flat call.
3. **Preset factories** -- `create_presidential_config(...)`, `create_parliamentary_config(...)`, `create_semi_presidential_config(...)`.
4. **One-liner runners** -- `run_presidential(...)` builds, runs, and returns the engine in one call.
5. **Fluent builder** -- `PolicyFlux().actors(100).policy_dim(2).with_ideal_point().build()`.
6. **Model API** -- TensorFlow-style `Sequential` or functional `Model` with layer specs like `L.IdealPoint()`, `L.PublicOpinion(support=0.6)`.

## Typical workflow

1. Create an `IntegrationConfig` (any of the methods above).
2. Build an engine with `build_engine(config)` or `.build()`.
3. Execute simulation with `engine.run()`.
4. Read summary metrics: `engine.pass_rate`, `engine.accepted_bills`, `engine.rejected_bills`.
