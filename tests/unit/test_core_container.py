import pytest

from policyflux.core.container import ServiceContainer
from policyflux.exceptions import RegistryError


class _Service:
    def __init__(self, value: int) -> None:
        self.value = value


def test_register_singleton_and_resolve_returns_same_instance() -> None:
    container = ServiceContainer()
    service = _Service(10)

    container.register_singleton(_Service, service)

    resolved = container.resolve(_Service)
    assert resolved is service


def test_register_factory_creates_instances() -> None:
    container = ServiceContainer()
    call_count = {"n": 0}

    def factory(_: ServiceContainer) -> _Service:
        call_count["n"] += 1
        return _Service(call_count["n"])

    container.register_factory(_Service, factory)

    first = container.resolve(_Service)
    second = container.resolve(_Service)

    assert first.value == 1
    assert second.value == 2
    assert first is not second


def test_resolve_without_factory_raises() -> None:
    container = ServiceContainer()

    with pytest.raises(RegistryError):
        container.resolve(_Service)
