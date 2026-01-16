from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class Actor:
    """Basic actor ontology for PolicyFlux.

    Fields are intentionally minimal; concrete models should extend this class.
    """
    id: Any
    latent_ideology: Optional[np.ndarray] = None
    utility_params: Dict[str, Any] = field(default_factory=dict)
    memory: List[Any] = field(default_factory=list)
    susceptibility: float = 1.0
    constraints: Dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "latent_ideology": None if self.latent_ideology is None else self.latent_ideology.tolist(),
            "utility_params": self.utility_params,
            "memory": list(self.memory),
            "susceptibility": float(self.susceptibility),
            "constraints": self.constraints,
        }

    def to_tensor(self):
        """Return a numerical representation for differentiable models.

        By default returns the latent ideology if present, otherwise a zero vector.
        Concrete implementations may return torch tensors.
        """
        if self.latent_ideology is None:
            return np.zeros(1, dtype=float)
        return np.asarray(self.latent_ideology, dtype=float)

    def update_memory(self, observation: Any) -> None:
        self.memory.append(observation)
