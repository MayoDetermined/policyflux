"""Shared helpers for GPU-aware array modules and device selection."""
from __future__ import annotations

from typing import Any

import numpy as np
import config

try:
    import cupy as cp  # type: ignore[import]
    _HAS_CUPY = True
except ImportError:  # pragma: no cover - optional dependency
    cp = None  # type: ignore[assignment]
    _HAS_CUPY = False

try:
    import torch  # type: ignore[import]
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


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


def get_torch_device(force_gpu: bool | None = None) -> "torch.device":
    """Resolve the preferred PyTorch device respecting configuration and availability."""
    if torch is None:
        raise RuntimeError("PyTorch is required for GPU-aware torch helpers")

    pref = config.PYTORCH_DEVICE.strip().lower()
    if force_gpu is None:
        prefer_gpu = pref in {"cuda", "gpu", "auto"}
    else:
        prefer_gpu = force_gpu

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
