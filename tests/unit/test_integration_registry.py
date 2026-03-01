"""Tests for policyflux.integration.registry."""

import pytest

from policyflux.exceptions import RegistryError
from policyflux.integration.config import LayerConfig
from policyflux.integration.builders.layer_builder import LayerBuilderContext
from policyflux.integration.registry import (
    LAYER_REGISTRY,
    build_layer_by_name,
    register_layer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(policy_dim: int = 2) -> LayerBuilderContext:
    return LayerBuilderContext(
        policy_dim=policy_dim,
        layer_config=LayerConfig(),
        lobbyists=[],
        whips=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegisterLayer:
    def test_register_and_retrieve(self) -> None:
        """Registering a custom layer factory makes it buildable by name."""
        called_with = {}

        def my_factory(ctx, overrides):
            called_with["ctx"] = ctx
            called_with["overrides"] = overrides
            return "my_layer_instance"

        register_layer("test_custom_layer", my_factory)
        assert "test_custom_layer" in LAYER_REGISTRY

        ctx = _make_context()
        result = build_layer_by_name("test_custom_layer", ctx)
        assert result == "my_layer_instance"
        assert called_with["ctx"] is ctx

        # Clean up to avoid polluting other tests
        del LAYER_REGISTRY["test_custom_layer"]

    def test_register_overwrites_existing(self) -> None:
        """Re-registering under the same name replaces the old factory."""
        register_layer("test_overwrite", lambda ctx, ov: "first")
        register_layer("test_overwrite", lambda ctx, ov: "second")

        ctx = _make_context()
        result = build_layer_by_name("test_overwrite", ctx)
        assert result == "second"

        del LAYER_REGISTRY["test_overwrite"]


class TestBuildLayerByName:
    def test_unknown_layer_raises_registry_error(self) -> None:
        with pytest.raises(RegistryError, match="not registered"):
            build_layer_by_name("nonexistent_layer_xyz", _make_context())

    def test_build_ideal_point(self) -> None:
        """The default 'ideal_point' layer should be registered and buildable."""
        ctx = _make_context(policy_dim=3)
        layer = build_layer_by_name("ideal_point", ctx)
        assert layer is not None
        # IdealPointLayer should have a space attribute
        assert hasattr(layer, "space")

    def test_build_public_opinion(self) -> None:
        ctx = _make_context()
        layer = build_layer_by_name("public_opinion", ctx)
        assert layer is not None

    def test_build_media_pressure(self) -> None:
        ctx = _make_context()
        layer = build_layer_by_name("media_pressure", ctx)
        assert layer is not None

    def test_build_lobbying(self) -> None:
        ctx = _make_context()
        layer = build_layer_by_name("lobbying", ctx)
        assert layer is not None

    def test_build_party_discipline(self) -> None:
        ctx = _make_context()
        layer = build_layer_by_name("party_discipline", ctx)
        assert layer is not None

    def test_build_government_agenda(self) -> None:
        ctx = _make_context()
        layer = build_layer_by_name("government_agenda", ctx)
        assert layer is not None


class TestDefaultLayersRegistered:
    """Verify that all expected default layers are present in the registry."""

    @pytest.mark.parametrize(
        "name",
        [
            "ideal_point",
            "public_opinion",
            "lobbying",
            "media_pressure",
            "party_discipline",
            "government_agenda",
        ],
    )
    def test_default_layer_is_registered(self, name: str) -> None:
        assert name in LAYER_REGISTRY, f"Expected '{name}' in LAYER_REGISTRY"
