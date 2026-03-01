# Getting Started

This guide walks through installation and a first end-to-end simulation run.

## 1) Installation

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
# Neural layers (PyTorch)
pip install -e ".[torch]"

# Text encoders
pip install -e ".[text-encoders]"

# Notebook/examples support
pip install -e ".[examples]"

# Development tools
pip install -e ".[dev]"
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

presidential = create_presidential_config(
    num_actors=100,
    policy_dim=2,
    iterations=200,
    seed=42,
)

parliamentary = create_parliamentary_config(
    num_actors=100,
    policy_dim=2,
    iterations=200,
    seed=42,
)

eng1 = build_engine(presidential)
eng2 = build_engine(parliamentary)

eng1.run()
eng2.run()

print(f"Presidential pass rate: {eng1.pass_rate:.1%}")
print(f"Parliamentary pass rate: {eng2.pass_rate:.1%}")
```

## 5) Try flat config updates

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

## 6) Development commands

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

Continue with [API Overview](api-overview.md) to explore the main public interfaces.