import pytest

from policyflux.exceptions import (
    BuildError,
    ConfigurationError,
    DimensionMismatchError,
    EngineNotConfiguredError,
    OptionalDependencyError,
    PolicyFluxError,
    RegistryError,
    SimulationError,
    ValidationError,
)


def test_policy_flux_error_is_exception() -> None:
    assert issubclass(PolicyFluxError, Exception)


def test_configuration_error_is_policy_flux_error() -> None:
    assert issubclass(ConfigurationError, PolicyFluxError)


def test_dimension_mismatch_error_is_policy_flux_error() -> None:
    assert issubclass(DimensionMismatchError, PolicyFluxError)


def test_validation_error_is_policy_flux_and_value_error() -> None:
    assert issubclass(ValidationError, PolicyFluxError)
    assert issubclass(ValidationError, ValueError)


def test_optional_dependency_error_is_policy_flux_and_import_error() -> None:
    assert issubclass(OptionalDependencyError, PolicyFluxError)
    assert issubclass(OptionalDependencyError, ImportError)


def test_registry_error_is_build_error() -> None:
    assert issubclass(RegistryError, BuildError)
    assert issubclass(BuildError, PolicyFluxError)


def test_engine_not_configured_is_simulation_error() -> None:
    assert issubclass(EngineNotConfiguredError, SimulationError)
    assert issubclass(SimulationError, PolicyFluxError)


def test_can_catch_by_base_class() -> None:
    with pytest.raises(PolicyFluxError):
        raise ValidationError("test")
