from __future__ import annotations
import numpy as np


def polarization_index(embeddings: np.ndarray) -> float:
    """Simple polarization metric: variance of 1D ideal points or mean pairwise distance.

    This is a placeholder; real analyses should provide robust implementations.
    """
    if embeddings.size == 0:
        return 0.0
    try:
        # embeddings: (N, D)
        mean = embeddings.mean(axis=0)
        d = np.linalg.norm(embeddings - mean, axis=1)
        return float(d.mean())
    except Exception:
        return float(np.var(embeddings))
