"""Lightweight observation container used by agents and collectors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class Observation:
    """Per-actor observation payload.

    Keeps features minimal and serializable so collectors can emit a standard
    shape while simulators remain free to adapt them.
    """

    actor_id: Any
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[Any] = None

    def serialize(self) -> Dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "features": dict(self.features),
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }

    def to_tensor(self) -> np.ndarray:
        """Best-effort numeric view for downstream models."""
        numeric = [v for v in self.features.values() if isinstance(v, (int, float))]
        if not numeric:
            return np.zeros(1, dtype=float)
        return np.asarray(numeric, dtype=float)
