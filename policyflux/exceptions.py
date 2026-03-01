"""Custom exception hierarchy for PolicyFlux.

All public exceptions inherit from :class:`PolicyFluxError` so that
callers can catch the base class for broad error handling.
"""


class PolicyFluxError(Exception):
    """Base exception for all PolicyFlux errors."""


class ConfigurationError(PolicyFluxError):
    """Raised when configuration is invalid or inconsistent."""


class DimensionMismatchError(PolicyFluxError):
    """Raised when policy-space dimensions do not match across components."""


class BuildError(PolicyFluxError):
    """Raised when a builder cannot construct an object from the given config."""


class RegistryError(BuildError):
    """Raised when a registry lookup fails (e.g. layer name not found)."""


class SimulationError(PolicyFluxError):
    """Raised for errors that occur during simulation execution."""


class EngineNotConfiguredError(SimulationError):
    """Raised when an engine is used before being properly configured."""


class OptionalDependencyError(PolicyFluxError, ImportError):
    """Raised when an optional dependency (torch, sentence-transformers) is missing."""


class ValidationError(PolicyFluxError, ValueError):
    """Raised when a parameter fails validation (out-of-range, wrong type, etc.)."""
