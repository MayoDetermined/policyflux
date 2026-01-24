"""Deterministic RNG management for reproducible experiments.

Provides a minimal wrapper around `random.Random` so the whole package
can share a single seeded RNG or re-seed per-session.
"""
import random as _random
from typing import Optional
from .config import get_settings


# module-level RNG instance
_RNG: _random.Random = _random.Random(get_settings().seed)


def set_seed(seed: Optional[int]) -> None:
    """Re-seed the package RNG (use None to reinitialise non-deterministically)."""
    global _RNG
    if seed is None:
        _RNG = _random.Random()
    else:
        _RNG = _random.Random(seed)


def get_rng() -> _random.Random:
    return _RNG


def random() -> float:
    return _RNG.random()


def randint(a: int, b: int) -> int:
    return _RNG.randint(a, b)
