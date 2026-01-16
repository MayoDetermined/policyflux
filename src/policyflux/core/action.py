from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class Action:
    """Simple action object produced by an actor.

    Payload can be any numerical encoding (e.g., vote vector, proposal features).
    """
    actor_id: Any
    payload: Optional[np.ndarray] = None
    meta: Dict[str, Any] = None

    def serialize(self) -> Dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "payload": None if self.payload is None else self.payload.tolist(),
            "meta": self.meta,
        }
