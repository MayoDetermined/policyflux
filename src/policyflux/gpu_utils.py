"""Shared helpers for GPU-aware array modules and device selection (TensorFlow-first)."""
from __future__ import annotations

from typing import Any

import numpy as np
from policyflux import config

# TensorFlow is the primary backend; torch has been removed.
import tensorflow as tf

try:
    import cupy as cp  # type: ignore[import]
    _HAS_CUPY = True
except ImportError:  # pragma: no cover - optional dependency
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False


def has_cupy() -> bool:
    """Return True when CuPy is installed and can accelerate numerical work."""
    return _HAS_CUPY


def get_array_module(use_gpu: bool = True) -> Any:
    """Return the array module (NumPy or CuPy) depending on availability."""
    if use_gpu and _HAS_CUPY and cp is not None:
        return cp
    return np


def to_cpu_array(array: Any) -> np.ndarray:
    """Convert an array into a NumPy array, copying from GPU if necessary."""
    if _HAS_CUPY and cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.array(array)


def get_tf_device(prefer_gpu: bool | None = None) -> str:
    """Return a TensorFlow device string respecting availability and config."""

    pref = config.TF_DEVICE.strip().lower()
    if prefer_gpu is None:
        prefer_gpu = pref in {"cuda", "gpu", "auto"}

    gpus = tf.config.list_physical_devices("GPU")
    if prefer_gpu and gpus:
        return "/GPU:0"
    return "/CPU:0"




