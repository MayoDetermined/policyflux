"""Data loading and collection for congressional analysis.

Provides:
- CongressBuilder: builds Congress systems from VoteView data
- ExternalSignalCollector: collects finance/district signals
- VoteViewLoader: loads voting records
"""
from __future__ import annotations

# Lightweight imports
from policyflux.data.collectors.base import BaseCollector, CollectorResult
from policyflux.data.collectors.external_signals import (
    ExternalSignalCollector,
    MacroSignalProvider,
)
from policyflux.data.loaders.voteview import VoteViewLoader


def __getattr__(name: str):
    """Lazy import for TensorFlow-dependent CongressBuilder."""
    if name in ("CongressBuilder", "CongressMenBuilder"):
        from policyflux.data.collectors.actors_architectural_bureau import CongressMenBuilder
        if name == "CongressBuilder":
            return CongressMenBuilder
        return CongressMenBuilder
    raise AttributeError(f"module 'policyflux.data' has no attribute '{name}'")


__all__ = [
    "CongressBuilder",
    "CongressMenBuilder",
    "BaseCollector",
    "CollectorResult", 
    "ExternalSignalCollector",
    "MacroSignalProvider",
    "VoteViewLoader",
]
