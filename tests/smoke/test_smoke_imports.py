import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "policyflux",
        "policyflux.layers.ideal_point",
        "policyflux.data_processing",
        "policyflux.engines.session_management",
        "policyflux.integration.builders.mechanics_builders",
    ],
)
def test_smoke_module_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None
