"""PolicyFlux - Congressional Behavioral Analysis Toolkit.

Simple API for simulating and analyzing US Congress voting behavior.

Example:
    >>> from policyflux import CongressSimulator
    >>> sim = CongressSimulator()
    >>> sim.fit()
    >>> results = sim.simulate(n_simulations=10, steps=5)
"""
from __future__ import annotations

__version__ = "0.2.0"

# Core types (always available, lightweight)
from policyflux.core import (
    Actor,
    Action, 
    Observation,
    State,
    BaseAgent,
    Environment,
)

# Configuration defaults
from policyflux.defaults import SIMULATION, MODELS, PATHS

# Lazy imports for heavy dependencies
def __getattr__(name: str):
    """Lazy import for TensorFlow-dependent modules."""
    if name == "CongressSimulator":
        from policyflux.congress_simulator import CongressSimulator
        return CongressSimulator
    if name == "CongressBuilder":
        from policyflux.data import CongressBuilder
        return CongressBuilder
    if name in ("VoteAutoencoder", "IdealPointModel", "DBCongressModel"):
        from policyflux import models
        return getattr(models, name)
    if name == "polarization_index":
        from policyflux.analysis import polarization_index
        return polarization_index
    if name in ("MLE", "Variational", "ParticleFilter"):
        from policyflux import inference
        return getattr(inference, name)
    raise AttributeError(f"module 'policyflux' has no attribute '{name}'")


__all__ = [
    # Version
    "__version__",
    # Core types
    "Actor",
    "Action",
    "Observation", 
    "State",
    "BaseAgent",
    "Environment",
    # Configuration
    "SIMULATION",
    "MODELS",
    "PATHS",
    # Main API (lazy)
    "CongressSimulator",
    "CongressBuilder",
    # Models (lazy)
    "VoteAutoencoder",
    "IdealPointModel",
    "DBCongressModel",
    # Analysis (lazy)
    "polarization_index",
    # Inference (lazy)
    "MLE",
    "Variational",
    "ParticleFilter",
]
