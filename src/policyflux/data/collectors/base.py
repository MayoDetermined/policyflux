"""Base collector interface for external data adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableMapping, Optional

from policyflux.core.observation import Observation


@dataclass
class CollectorResult(Mapping[str, Any]):
    """Typed wrapper returned by collectors.

    Exposes a dict-like interface for backward compatibility while also
    providing helpers to iterate over `Observation` objects.
    """

    actors: Dict[str, Observation] = field(default_factory=dict)
    matrices: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "actors": {k: dict(v.features) for k, v in self.actors.items()},
        }
        payload.update(self.matrices)
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload

    def observations(self) -> Iterator[Observation]:
        return iter(self.actors.values())

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.to_dict().get(key, default)


class BaseCollector(ABC):
    """Abstract base for data collectors that produce actor-level signals.

    Implementations should be lightweight adapters that transform external
    sources (files, APIs) into the project's internal signal format. Keeping
    collectors small simplifies testing and replacement.
    """

    @abstractmethod
    def collect(self, actor_ids: List[int]) -> CollectorResult:
        """Collect signals for the provided `actor_ids`.

        Implementations should return a :class:`CollectorResult` that exposes
        observations and any auxiliary matrices (e.g., networks).
        """
        raise NotImplementedError()

    def collect_observations(self, actor_ids: List[int]) -> Iterator[Observation]:
        """Convenience iterator over observations for a set of actors."""
        return self.collect(actor_ids).observations()
