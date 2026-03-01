"""PolicyFlux - Political simulation framework.

Recommended imports:

- Core abstractions: `from policyflux.core import Bill, CongressModel, Layer`
- Toolbox (implementations): `from policyflux.toolbox import SequentialVoter, SequentialBill`
- Layers: `from policyflux.layers import IdealPointLayer, PublicOpinionLayer`
- Engines: `from policyflux.engines import SequentialMonteCarlo, Session`
- Integration (builders & presets): `from policyflux.integration import build_engine, create_presidential_config`

This module exposes core abstractions, utilities, and provides access to all submodules.
"""

# ruff: noqa: RUF022

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LAYER_REGISTRY",
    "model",
    "PARLIAMENTARY_DEFAULT",
    "PRESIDENTIAL_DEFAULT",
    "SEMI_PRESIDENTIAL_DEFAULT",
    "AdvancedActorsConfig",
    "AggregationStrategy",
    "AverageAggregation",
    # Core abstractions
    "Bill",
    "BuildError",
    "ComplexActor",
    "ConfigurationError",
    "CongressMember",
    "CongressModel",
    "DeterministicEngine",
    "DeterministicVoting",
    "DimensionMismatchError",
    # Engines
    "Engine",
    "EngineNotConfiguredError",
    "Executive",
    "ExecutiveActor",
    "ExecutiveType",
    "GovernmentAgendaLayer",
    "IdGenerator",
    "IdealPointEncoderDF",
    # Layers
    "IdealPointLayer",
    "IdealPointTextEncoder",
    # Integration
    "IntegrationConfig",
    "Layer",
    "LayerBuilderContext",
    "LayerConfig",
    "LobbyingLayer",
    "MPEngine",
    "MediaPressureLayer",
    "MultiplicativeAggregation",
    "OptionalDependencyError",
    "ParallelMonteCarlo",
    "PartyDisciplineLayer",
    # Fluent API / Fluent Builder API
    "ActorBuilder",
    "ExecutiveBuilder",
    "LayerBuilder",
    "PolicyFlux",
    # Exceptions
    "PolicyFluxError",
    "PolicyPosition",
    "PolicySpace",
    "PolicyVector",
    "ProbabilisticVoting",
    "PublicOpinionLayer",
    "RegistryError",
    "SequentialAggregation",
    "SequentialBill",
    "SequentialCongressModel",
    "SequentialLobbyist",
    "SequentialMonteCarlo",
    "SequentialPresident",
    "SequentialSpeaker",
    # Toolbox
    "SequentialVoter",
    "SequentialWhip",
    "ServiceContainer",
    "Session",
    "Settings",
    "SimulationContext",
    "SimulationError",
    "SoftVoting",
    "UtilitySpace",
    "ValidationError",
    "VotingContext",
    "VotingStrategy",
    "WeightedAggregation",
    "bake_a_pie",
    "build_advanced_actors",
    "build_aggregation_strategy",
    "build_bill",
    "build_congress",
    "build_engine",
    "build_executive",
    "build_layer_by_name",
    "build_layers",
    "build_session",
    # Reports
    "craft_a_bar",
    "create_parliamentary_config",
    "create_presidential_config",
    "create_semi_presidential_config",
    "get_id_generator",
    "get_rng",
    # Configuration & utilities
    "get_settings",
    # Lazy import
    "import_models",
    "logger",
    "parliamentary_engine",
    "presidential_engine",
    "random",
    "register_layer",
    "run_parliamentary",
    "run_presidential",
    "run_semi_presidential",
    "semi_presidential_engine",
    "set_seed",
]

import importlib
from typing import Any

# --- Core abstractions ---
from .core import (
    AggregationStrategy,
    AverageAggregation,
    Bill,
    ComplexActor,
    CongressMember,
    CongressModel,
    DeterministicVoting,
    Executive,
    ExecutiveActor,
    ExecutiveType,
    IdGenerator,
    Layer,
    MultiplicativeAggregation,
    PolicyPosition,
    PolicySpace,
    PolicyVector,
    ProbabilisticVoting,
    SequentialAggregation,
    ServiceContainer,
    SimulationContext,
    SoftVoting,
    UtilitySpace,
    VotingContext,
    VotingStrategy,
    WeightedAggregation,
    get_id_generator,
)

# --- Engines ---
from .engines import (
    DeterministicEngine,
    Engine,
    MPEngine,
    ParallelMonteCarlo,
    SequentialMonteCarlo,
    Session,
)

# --- Exceptions ---
from .exceptions import (
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

# --- Integration (builders, registry, presets) ---
from .integration import (
    LAYER_REGISTRY,
    PARLIAMENTARY_DEFAULT,
    PRESIDENTIAL_DEFAULT,
    SEMI_PRESIDENTIAL_DEFAULT,
    ActorBuilder,
    AdvancedActorsConfig,
    ExecutiveBuilder,
    IntegrationConfig,
    LayerBuilder,
    LayerBuilderContext,
    LayerConfig,
    PolicyFlux,
    build_advanced_actors,
    build_aggregation_strategy,
    build_bill,
    build_congress,
    build_engine,
    build_executive,
    build_layer_by_name,
    build_layers,
    build_session,
    create_parliamentary_config,
    create_presidential_config,
    create_semi_presidential_config,
    parliamentary_engine,
    presidential_engine,
    register_layer,
    run_parliamentary,
    run_presidential,
    run_semi_presidential,
    semi_presidential_engine,
)

# --- Configuration & utilities ---
from .integration.config import Settings, get_settings

# --- Layers ---
from .layers import (
    GovernmentAgendaLayer,
    IdealPointEncoderDF,
    IdealPointLayer,
    IdealPointTextEncoder,
    LobbyingLayer,
    MediaPressureLayer,
    PartyDisciplineLayer,
    PublicOpinionLayer,
)
from .logging_config import logger
from .pfrandom import get_rng, random, set_seed

# --- Toolbox (concrete implementations) ---
from .toolbox import (
    SequentialBill,
    SequentialCongressModel,
    SequentialLobbyist,
    SequentialPresident,
    SequentialSpeaker,
    SequentialVoter,
    SequentialWhip,
)

# --- Reports ---
from .utils.reports import bake_a_pie, craft_a_bar


def import_models() -> Any:
    """Import `policyflux.math_models` on demand and return the module.

    Use this to avoid importing model implementations at package import time.
    """
    return importlib.import_module("policyflux.math_models")


# --- Model API (TF-style) ---
from . import model  # noqa: E402  (must be after all other imports)
