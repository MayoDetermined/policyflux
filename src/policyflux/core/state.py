from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List
import numpy as np

from policyflux.core.actor import Actor


@dataclass
class State:
    """Political system state container.

    - actors: mapping actor_id -> `Actor`
    - relations: list of relation matrices (networks)
    - macro: dict of macro variables (polarization, agenda, etc.)
    """
    actors: Dict[Any, Actor] = field(default_factory=dict)
    relations: List[Any] = field(default_factory=list)
    macro: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Actor]:
        return iter(self.actors.values())

    def __len__(self) -> int:
        return len(self.actors)

    def serialize(self) -> Dict[str, Any]:
        return {
            "actors": {k: v.serialize() for k, v in self.actors.items()},
            "macro": dict(self.macro),
        }

    def to_tensor(self):
        """Return stacked actor embeddings as 2D array."""
        rows = []
        for a in self.actors.values():
            vec = a.to_tensor()
            rows.append(np.asarray(vec).reshape(-1))
        if not rows:
            return np.zeros((0, 1), dtype=float)
        return np.vstack(rows)
