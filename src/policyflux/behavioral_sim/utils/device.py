from __future__ import annotations

from typing import Any

import tensorflow as tf

from policyflux.gpu_utils import get_tf_device


def get_device(force_gpu: bool | None = None) -> str:
    return get_tf_device(prefer_gpu=force_gpu)


def to_device(obj: Any, device: str) -> Any:
    with tf.device(device):
        try:
            return tf.convert_to_tensor(obj)
        except Exception:
            return obj


def ensure_tensor(obj: Any, device: str, dtype: tf.dtypes.DType | None = None) -> tf.Tensor:
    with tf.device(device):
        return tf.convert_to_tensor(obj, dtype=dtype)




