"""PolicyFlux - Political simulation framework.

Recommended imports:

- Core abstractions: `from policyflux.core import Bill, CongressModel, Layer`
- Toolbox (implementations): `from policyflux.toolbox import SequentialVoter, SequentialBill`
- Layers: `from policyflux.layers import IdealPointLayer, PublicOpinionLayer`
- Engines: `from policyflux.engines import SequentialMonteCarlo, Session`
- Integration (builders & presets): `from policyflux.integration import build_engine, create_presidential_config`

This module exposes core abstractions, utilities, and provides access to all submodules.
"""

from typing import Any
import importlib

# --- Core abstractions ---
from .core import (  # noqa: F401
    Bill, CongressModel, CongressMan, Layer, ComplexActor,
    ExecutiveType, ExecutiveActor, Executive,
    PolicyVector, UtilitySpace, PolicySpace, PolicyPosition,
    VotingContext, SimulationContext,
    VotingStrategy, ProbabilisticVoting, DeterministicVoting, SoftVoting,
    AggregationStrategy, SequentialAggregation, AverageAggregation,
    WeightedAggregation, MultiplicativeAggregation,
    ServiceContainer,
    IdGenerator, get_id_generator,
)

# --- Configuration & utilities ---
from .integration.config import get_settings, Settings  # noqa: F401
from .logging_config import logger  # noqa: F401
from .pfrandom import set_seed, get_rng, random  # noqa: F401

# --- Toolbox (concrete implementations) ---
from .toolbox import (  # noqa: F401
    SequentialVoter,
    SequentialBill,
    SequentialCongressModel,
    SequentialLobbyer,
    SequentialSpeaker,
    SequentialWhip,
    SequentialPresident,
)

# --- Layers ---
from .layers import (  # noqa: F401
    IdealPointLayer,
    IdealPointEncoderDF,
    IdealPointTextEncoder,
    PublicOpinionLayer,
    LobbyingLayer,
    MediaPressureLayer,
    PartyDisciplineLayer,
    GovernmentAgendaLayer,
)

# --- Engines ---
from .engines import (  # noqa: F401
    Engine,
    MPEngine,
    Session,
    SequentialMonteCarlo,
    ParallelMonteCarlo,
    DeterministicEngine,
)

# --- Integration (builders, registry, presets) ---
from .integration import (  # noqa: F401
    IntegrationConfig,
    AdvancedActorsConfig,
    LayerConfig,
    LAYER_REGISTRY,
    register_layer,
    build_layer_by_name,
    build_executive,
    build_advanced_actors,
    build_layers,
    build_congress,
    build_bill,
    build_session,
    build_engine,
    build_aggregation_strategy,
    LayerBuilderContext,
    create_presidential_config,
    create_parliamentary_config,
    create_semi_presidential_config,
)

# --- Reports ---
from .utils.reports import craft_a_bar, bake_a_pie  # noqa: F401


def import_models() -> Any:
    """Import `policyflux.models` on demand and return the module.

    Use this to avoid importing model implementations at package import time.
    """
    return importlib.import_module("policyflux.models")
