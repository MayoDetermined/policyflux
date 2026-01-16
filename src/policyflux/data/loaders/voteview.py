from __future__ import annotations
from typing import Any


class VoteViewLoader:
    """Placeholder adapter for VoteView-like roll-call data."""

    def load(self, path: str) -> Any:
        raise NotImplementedError()
