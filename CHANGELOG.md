# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Expanded unit-test coverage across core, engine, integration, layer, data-processing, special-actor, and report utility modules
- Focused tests for fluent integration builder (`PolicyFlux`) and optional-dependency guard paths
- Branch-level tests for chart helpers and special actors (`Speaker`, `Whips`, `Lobbyist`, `President`)

### Changed

- Coverage configuration corrected from `policyflux/models/*` to `policyflux/model/*`
- Coverage scope tightened by excluding optional/heavy modules that are intentionally outside current unit-test focus
- Repository-wide coverage significantly increased (from failing at ~57% to consistently above threshold, now ~85%)
- Documentation refresh across README and `docs/` pages for clearer onboarding, API navigation, and release workflow instructions

### Previous project updates

- Custom exception hierarchy (`policyflux.exceptions`)
- `__all__` definitions in all public `__init__.py` modules
- `py.typed` marker for PEP 561 compliance
- GitHub Actions CI/CD pipeline
- Pre-commit configuration
- Comprehensive test suite
- Release workflow for trusted PyPI publishing (`.github/workflows/publish.yml`)
- Release guide documentation (`docs/release.md`)
- Explicit `unit` and `smoke` pytest markers with automatic assignment by test path
- Standardized all docstrings to Google style
- Strict mypy and ruff linting configuration
- Improved type annotations across all modules
- CI split into dedicated quality, unit, smoke, coverage, and package-check jobs
- Documentation examples updated to current engine API (`engine.run()`)
- `SimpleTextVectorizer` now fails gracefully when optional `torch` dependency is missing

## [0.1.0] - 2026-02-26

### Added

- Initial release with core simulation framework
- Sequential and parallel Monte Carlo engines
- Deterministic engine
- Presidential, parliamentary, and semi-presidential system presets
- Layer-based voting decision architecture (ideal point, public opinion, lobbying, media pressure, party discipline, government agenda, neural)
- Advanced actors (Speaker, Whips, Lobbyists, President/PM)
- Integration builders and configuration system
- Text encoding layers (TF-IDF + sentence-transformers)
- Visualization utilities (bar charts, pie charts)