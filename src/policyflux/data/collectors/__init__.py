"""Data collectors for congressional behavioral analysis.

This package provides adapters for external data sources (APIs, files)
that produce standardized `Observation` objects for actors.
"""
from policyflux.data.collectors.base import BaseCollector, CollectorResult
from policyflux.data.collectors.external_signals import (
    ExternalSignalCollector,
    MacroSignalProvider,
)

__all__ = [
    "BaseCollector",
    "CollectorResult",
    "ExternalSignalCollector",
    "MacroSignalProvider",
]
