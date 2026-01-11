from __future__ import annotations

from typing import Any

import torch

from gpu_utils import get_torch_device


def get_device(force_gpu: bool | None = None) -> torch.device:
    return get_torch_device(force_gpu=force_gpu)


def to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    try:
        return torch.as_tensor(obj, device=device)
    except Exception:
        return obj


def ensure_tensor(obj: Any, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.as_tensor(obj, device=device, dtype=dtype)
