# Getting Started

This guide shows how to install PolicyFlux and run your first simulation.

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

# Development tools
pip install -e ".[dev]"
```

## 2) Minimal simulation

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
print(engine)
```

## 3) Compare systems quickly

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

## 4) Development commands

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
- If imports fail, verify installation with:

```bash
python -c "import policyflux; print(policyflux.__file__)"
```