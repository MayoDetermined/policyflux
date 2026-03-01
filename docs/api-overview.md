# API Overview

This page summarizes the most commonly used public API entry points.

## Primary imports

```python
from policyflux import (
    build_engine,
    IntegrationConfig,
    LayerConfig,
    AdvancedActorsConfig,
    create_presidential_config,
    create_parliamentary_config,
    create_semi_presidential_config,
)
```

## Quick Start API

```python
from policyflux import build_engine, create_presidential_config

config = create_presidential_config(
    num_actors=100,
    policy_dim=2,
    iterations=200,
    seed=42,
)

engine = build_engine(config)
engine.run()

print(f"Pass rate: {engine.pass_rate:.1%}")
print(f"Accepted bills: {engine.accepted_bills}")
print(f"Rejected bills: {engine.rejected_bills}")
```

For a custom setup, build an `IntegrationConfig` directly and pass it to `build_engine(config)`.

## Flat Configuration with defaults

`IntegrationConfig` supports a flat style configuration API that maps fields across:

- top-level `IntegrationConfig`
- nested `LayerConfig`
- nested `AdvancedActorsConfig`

Any omitted fields keep their dataclass defaults.

```python
from policyflux import IntegrationConfig, build_engine

config = IntegrationConfig.from_flat(
    num_actors=140,
    include_public_opinion=False,
    public_support=0.63,
    n_lobbyists=2,
    lobbyist_strength=0.71,
    aggregation_strategy="average",
)

# Defaults are preserved for non-provided fields
assert config.policy_dim == 4
assert config.iterations == 300
assert config.seed == 42

engine = build_engine(config)
engine.run()
```

You can also apply flat updates to an existing config:

```python
config = IntegrationConfig().with_flat(public_support=0.58, n_whips=1)
```

Unknown flat keys are rejected with `ValueError`.

## Configuration objects

### `IntegrationConfig`
Top-level simulation configuration.

Typical fields include:

- `num_actors`
- `policy_dim`
- `iterations`
- `seed`
- `layer_config`
- `actors_config`

### `LayerConfig`
Controls layer inclusion and parameters, such as:

- `include_ideal_point`
- `include_public_opinion`
- `include_lobbying`
- `include_media_pressure`
- `include_party_discipline`
- `include_government_agenda`

And corresponding strengths/intensities.

### `AdvancedActorsConfig`
Enables additional actor mechanics, such as lobbyists and whips.

## Builders and presets

### `build_engine(config)`
Constructs a fully configured simulation engine.

### Preset factories

- `create_presidential_config(...)`
- `create_parliamentary_config(...)`
- `create_semi_presidential_config(...)`

These helpers produce ready-to-use `IntegrationConfig` setups for common institutional systems.

## Typical workflow

```python
config = create_presidential_config(num_actors=100, policy_dim=2, iterations=200, seed=42)
engine = build_engine(config)
engine.run()
print(engine.pass_rate)
```

## Notes

- Public API is still evolving in early-stage development.
- Prefer importing from `policyflux` package root for forward compatibility.
- For advanced extension points, inspect `policyflux/integration/registry.py` and builder modules.