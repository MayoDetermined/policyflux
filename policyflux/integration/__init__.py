# Integration module: configuration, builders, registry, and presets.
#
# Config is imported eagerly (no heavy dependencies).
# Builders, registry, and presets are loaded lazily to avoid circular imports
# with pfrandom/logging_config which also depend on integration.config.

from typing import Any

__all__ = [
    "LAYER_REGISTRY",
    "PARLIAMENTARY_DEFAULT",
    "PRESIDENTIAL_DEFAULT",
    "SEMI_PRESIDENTIAL_DEFAULT",
    "ActorBuilder",
    "AdvancedActorsConfig",
    "ExecutiveBuilder",
    "IntegrationConfig",
    "LayerBuilder",
    "LayerBuilderContext",
    "LayerConfig",
    "PolicyFlux",
    "Settings",
    "build_advanced_actors",
    "build_aggregation_strategy",
    "build_bill",
    "build_congress",
    "build_engine",
    "build_executive",
    "build_layer_by_name",
    "build_layers",
    "build_session",
    "create_parliamentary_config",
    "create_presidential_config",
    "create_semi_presidential_config",
    "get_settings",
    "parliamentary_engine",
    "presidential_engine",
    "register_layer",
    "run_parliamentary",
    "run_presidential",
    "run_semi_presidential",
    "semi_presidential_engine",
]

from .config import (
    AdvancedActorsConfig,
    IntegrationConfig,
    LayerConfig,
    Settings,
    get_settings,
)


def __getattr__(name: str) -> Any:
    """Lazy-load heavy submodules to break circular imports."""
    _registry_names = {
        "LAYER_REGISTRY",
        "register_layer",
        "build_layer_by_name",
    }
    _builder_names = {
        "build_executive",
        "build_advanced_actors",
        "build_layers",
        "build_congress",
        "build_bill",
        "build_session",
        "build_engine",
        "build_aggregation_strategy",
        "LayerBuilderContext",
    }
    _preset_names = {
        "create_presidential_config",
        "create_parliamentary_config",
        "create_semi_presidential_config",
        "PRESIDENTIAL_DEFAULT",
        "PARLIAMENTARY_DEFAULT",
        "SEMI_PRESIDENTIAL_DEFAULT",
        "presidential_engine",
        "parliamentary_engine",
        "semi_presidential_engine",
        "run_presidential",
        "run_parliamentary",
        "run_semi_presidential",
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

    _fluent_names = {
        "PolicyFlux",
        "LayerBuilder",
        "ExecutiveBuilder",
        "ActorBuilder",
    }

    if name in _fluent_names:
        from . import fluent

        return getattr(fluent, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
