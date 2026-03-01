# Contributing to PolicyFlux

Thank you for your interest in contributing to PolicyFlux. This guide describes the preferred contribution workflow.

## Code of Conduct

This project follows the [PolicyFlux Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

- Use [GitHub Issues](https://github.com/piotrpawelec/policyflux/issues) to report bugs.
- Include a minimal reproducible example when possible.
- Describe the expected versus actual behavior.
- Include your Python version and operating system.

### Suggesting Features

- Open a [GitHub Issue](https://github.com/piotrpawelec/policyflux/issues) with the "enhancement" label.
- Describe the use case and why the feature would be valuable.

### Submitting Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Make your changes, following the code style guidelines below.
4. Add or update tests for your changes.
5. Run tests, linting, and type checks (see below).
6. Submit a pull request against the `main` branch.

## Development Setup

```bash
git clone https://github.com/piotrpawelec/policyflux.git
cd policyflux
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows
pip install -e ".[dev]"
```

For PowerShell, you may need:

```powershell
.venv\Scripts\Activate.ps1
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=policyflux --cov-report=term-missing

# Run only unit tests
pytest tests/unit/ -m unit

# Run only smoke tests
pytest tests/smoke/ -m smoke
```

Use targeted test runs when touching a specific module, then run the full suite before opening a PR.

## Linting and Type Checking

```bash
# Lint
ruff check policyflux/

# Format check
ruff format --check policyflux/

# Type check
mypy policyflux/
```

All checks must pass before a pull request can be merged.

## Release Readiness (maintainers)

Before publishing a new version, run:

```bash
pytest tests/
ruff check policyflux/
mypy policyflux/
python -m build
twine check dist/*
```

Then follow [docs/release.md](docs/release.md) for the full GitHub/PyPI release flow.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use type annotations for all public functions and methods.
- Write docstrings in [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- Keep line length under 100 characters (enforced by ruff).

## Commit and PR quality

- Keep PRs focused and small when possible.
- Update docs when behavior or public API changes.
- Add changelog entries for user-visible changes.
- Include migration notes when making breaking API changes.

## Project Structure

```
policyflux/
  core/          # Base abstractions and strategies
  layers/        # Decision layers (ideal point, lobbying, etc.)
  engines/       # Simulation engines (Monte Carlo, deterministic)
  integration/   # High-level config, builders, presets, fluent API
  toolbox/       # Concrete implementations
  data_processing/  # Text processing and encoders
  utils/         # Reporting and utility helpers
tests/
  smoke/         # Integration smoke tests
  unit/          # Unit tests
```

## Adding a Custom Layer

To add a new decision layer:

1. Create a new class in `policyflux/layers/` that inherits from `Layer`.
2. Implement the `call()` and `compile()` methods.
3. Register it in `policyflux/integration/registry.py` with a factory function.
4. Add tests in `tests/unit/`.
5. Export it from `policyflux/layers/__init__.py`.

## Questions?

Open a [GitHub Discussion](https://github.com/piotrpawelec/policyflux/discussions) for general questions about contributing or using the library.
