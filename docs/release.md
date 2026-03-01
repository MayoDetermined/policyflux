# Release Guide

This guide describes the release process for PolicyFlux from local validation to PyPI publication.

## 1) Pre-release checklist

Run all required checks locally:

```bash
pip install -e ".[dev]"
pytest tests/
ruff check policyflux/
mypy policyflux/
python -m build
twine check dist/*
```

If any command fails, fix it before continuing.

## 2) Versioning and changelog

1. Update `policyflux/__init__.py` version (`__version__`).
2. Update `pyproject.toml` project version.
3. Move relevant entries from `CHANGELOG.md` `Unreleased` to a new version section with date.
4. Commit and push changes to `main`.

Keep version values identical in both files.

## 3) GitHub release

1. Create and push a git tag, e.g. `v0.1.1`.
2. Create a GitHub Release from that tag.
3. Mark the release as `Published`.

The `publish.yml` workflow should:
- build the wheel and source distribution,
- validate metadata with `twine check`,
- publish artifacts to PyPI via trusted publishing.

## 4) PyPI trusted publishing setup

In PyPI project settings, ensure a trusted publisher exists for this repository:
- Owner/repo: `piotrpawelec/policyflux`
- Workflow: `.github/workflows/publish.yml`
- Environment (if used): `pypi`

If publishing fails, first verify the tag points to the expected commit and the workflow has required permissions.

## 5) Post-release checks

- Confirm the package is visible on PyPI.
- Verify installation in a clean environment:

```bash
pip install policyflux
python -c "import policyflux; print(policyflux.__version__)"
```

- Confirm `pip install policyflux` resolves the newly released version.
- Announce release notes (GitHub release notes, docs, social channels as needed).
