# Integration module: configuration, builders, registry, and presets.
#
# Config is imported eagerly (no heavy dependencies).
# Builders, registry, and presets are loaded lazily to avoid circular imports
# with pfrandom/logging_config which also depend on integration.config.

from .config import (
    Settings,
    get_settings,
    IntegrationConfig,
    AdvancedActorsConfig,
    LayerConfig,
)


def __getattr__(name):
    """Lazy-load heavy submodules to break circular imports."""
    _registry_names = {
        "LAYER_REGISTRY", "register_layer", "build_layer_by_name",
    }
    _builder_names = {
        "build_executive", "build_advanced_actors", "build_layers",
        "build_congress", "build_bill", "build_session", "build_engine",
        "build_aggregation_strategy", "LayerBuilderContext",
    }
    _preset_names = {
        "create_presidential_config", "create_parliamentary_config",
        "create_semi_presidential_config",
    }

    if name in _registry_names:
        from . import registry
        return getattr(registry, name)

    if name in _builder_names:
        from . import builders
        return getattr(builders, name)

    if name in _preset_names:
        from . import presets
        return getattr(presets, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
