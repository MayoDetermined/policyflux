"""High-level modular integration helpers for PolicyFlux.

This module provides builder utilities to wire together models, layers,
advanced actors, and simulation engines in a configurable way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from .core.aggregation_strategy import (
    AggregationStrategy,
    AverageAggregation,
    MultiplicativeAggregation,
    SequentialAggregation,
    WeightedAggregation,
)
from .models import (
    SequentialBill,
    SequentialCongressModel,
    SequentialMonteCarlo,
    SequentialVoter,
    Session,
)
from .models.advanced_actors import (
    SequentialLobbyer,
    SequentialPresident,
    SequentialSpeaker,
    SequentialWhip,
)
from .layers import (
    IdealPointEncoder,
    LobbyingLayer,
    MediaPressureLayer,
    PartyDisciplineLayer,
    PublicOpinionLayer,
)
from .pfrandom import random as pf_random
from .pfrandom import set_seed


@dataclass
class LayerConfig:
    include_ideal_point: bool = True
    include_public_opinion: bool = True
    include_lobbying: bool = True
    include_media_pressure: bool = True
    include_party_discipline: bool = True
    include_neural: bool = False

    layer_names: Optional[List[str]] = None
    layer_overrides: Dict[str, Dict] = field(default_factory=dict)

    public_support: float = 0.5
    lobbying_intensity: float = 0.0
    media_pressure: float = 0.0
    party_line_support: float = 0.5
    party_discipline_strength: float = 0.5

    neural_layer_factory: Optional[Callable[[], object]] = None


@dataclass
class LayerBuilderContext:
    policy_dim: int
    layer_config: LayerConfig
    lobbyists: Iterable[SequentialLobbyer]
    whips: Iterable[SequentialWhip]


LayerFactory = Callable[[LayerBuilderContext, Dict], object]


LAYER_REGISTRY: Dict[str, LayerFactory] = {}


def register_layer(name: str, factory: LayerFactory) -> None:
    LAYER_REGISTRY[name] = factory


def build_layer_by_name(name: str, context: LayerBuilderContext) -> object:
    factory = LAYER_REGISTRY.get(name)
    if factory is None:
        raise KeyError(f"Layer '{name}' is not registered")
    overrides = context.layer_config.layer_overrides.get(name, {})
    return factory(context, overrides)


def _register_default_layers() -> None:
    def ideal_point_factory(ctx: LayerBuilderContext, overrides: Dict) -> object:
        space = overrides.get(
            "space", [pf_random() for _ in range(ctx.policy_dim)]
        )
        status_quo = overrides.get("status_quo", [0.5] * ctx.policy_dim)
        return IdealPointEncoder(space=space, status_quo=status_quo)

    def public_opinion_factory(ctx: LayerBuilderContext, overrides: Dict) -> object:
        support = overrides.get("support_level", ctx.layer_config.public_support)
        return PublicOpinionLayer(support_level=support)

    def lobbying_factory(ctx: LayerBuilderContext, overrides: Dict) -> object:
        intensity = overrides.get("intensity", ctx.layer_config.lobbying_intensity)
        lobbying = LobbyingLayer(intensity=intensity)
        for lobbyer in ctx.lobbyists:
            lobbying.add_lobbyst(lobbyer)
        return lobbying

    def media_factory(ctx: LayerBuilderContext, overrides: Dict) -> object:
        pressure = overrides.get("pressure", ctx.layer_config.media_pressure)
        return MediaPressureLayer(pressure=pressure)

    def party_factory(ctx: LayerBuilderContext, overrides: Dict) -> object:
        party_line = overrides.get("party_line_support", ctx.layer_config.party_line_support)
        discipline = overrides.get(
            "discipline_base_strength", ctx.layer_config.party_discipline_strength
        )
        return PartyDisciplineLayer(
            party_whips=list(ctx.whips),
            discipline_base_strength=discipline,
            party_line_support=party_line,
        )

    register_layer("ideal_point", ideal_point_factory)
    register_layer("public_opinion", public_opinion_factory)
    register_layer("lobbying", lobbying_factory)
    register_layer("media_pressure", media_factory)
    register_layer("party_discipline", party_factory)


_register_default_layers()


@dataclass
class AdvancedActorsConfig:
    n_lobbyists: int = 0
    lobbyist_strength: float = 0.5
    lobbyist_stance: float = 1.0

    n_whips: int = 0
    whip_discipline_strength: float = 0.5
    whip_party_line_support: float = 0.5

    speaker_agenda_support: float = 0.5
    president_approval_rating: float = 0.5


@dataclass
class IntegrationConfig:
    num_actors: int = 100
    policy_dim: int = 4
    iterations: int = 300
    seed: int = 42
    description: str = "PolicyFlux modular simulation"

    layer_config: LayerConfig = field(default_factory=LayerConfig)
    actors_config: AdvancedActorsConfig = field(default_factory=AdvancedActorsConfig)

    aggregation_strategy: str = "sequential"  # sequential|average|weighted|multiplicative
    aggregation_weights: Optional[List[float]] = None


def build_aggregation_strategy(config: IntegrationConfig) -> AggregationStrategy:
    if config.aggregation_strategy == "average":
        return AverageAggregation()
    if config.aggregation_strategy == "multiplicative":
        return MultiplicativeAggregation()
    if config.aggregation_strategy == "weighted":
        if not config.aggregation_weights:
            raise ValueError("aggregation_weights must be provided for weighted strategy")
        return WeightedAggregation(config.aggregation_weights)
    return SequentialAggregation()


def build_advanced_actors(config: IntegrationConfig) -> tuple[List[SequentialLobbyer], List[SequentialWhip], SequentialSpeaker, SequentialPresident]:
    lobbyists = [
        SequentialLobbyer(
            influence_strength=config.actors_config.lobbyist_strength,
            stance=config.actors_config.lobbyist_stance,
            name=f"Lobbyer_{i + 1}",
        )
        for i in range(config.actors_config.n_lobbyists)
    ]

    whips = [
        SequentialWhip(
            discipline_strength=config.actors_config.whip_discipline_strength,
            party_line_support=config.actors_config.whip_party_line_support,
            name=f"Whip_{i + 1}",
        )
        for i in range(config.actors_config.n_whips)
    ]

    speaker = SequentialSpeaker(agenda_support=config.actors_config.speaker_agenda_support)
    president = SequentialPresident(approval_rating=config.actors_config.president_approval_rating)

    return lobbyists, whips, speaker, president


def build_layers(
    config: IntegrationConfig,
    lobbyists: Iterable[SequentialLobbyer],
    whips: Iterable[SequentialWhip],
) -> List:
    layers = []
    layer_cfg = config.layer_config

    if layer_cfg.layer_names:
        context = LayerBuilderContext(
            policy_dim=config.policy_dim,
            layer_config=layer_cfg,
            lobbyists=lobbyists,
            whips=whips,
        )
        for name in layer_cfg.layer_names:
            layers.append(build_layer_by_name(name, context))
        if layer_cfg.include_neural:
            if layer_cfg.neural_layer_factory is None:
                raise ValueError("neural_layer_factory must be provided when include_neural=True")
            layers.append(layer_cfg.neural_layer_factory())
        return layers

    if layer_cfg.include_ideal_point:
        layers.append(
            IdealPointEncoder(
                space=[pf_random() for _ in range(config.policy_dim)],
                status_quo=[0.5] * config.policy_dim,
            )
        )

    if layer_cfg.include_public_opinion:
        layers.append(PublicOpinionLayer(id=None, support_level=layer_cfg.public_support))

    if layer_cfg.include_lobbying:
        lobbying = LobbyingLayer(id=None, intensity=layer_cfg.lobbying_intensity)
        for lobbyer in lobbyists:
            lobbying.add_lobbyst(lobbyer)
        layers.append(lobbying)

    if layer_cfg.include_media_pressure:
        layers.append(MediaPressureLayer(id=None, pressure=layer_cfg.media_pressure))

    if layer_cfg.include_party_discipline:
        party_layer = PartyDisciplineLayer(
            id=None,
            party_whips=list(whips),
            discipline_base_strength=layer_cfg.party_discipline_strength,
            party_line_support=layer_cfg.party_line_support,
        )
        layers.append(party_layer)

    if layer_cfg.include_neural:
        if layer_cfg.neural_layer_factory is None:
            raise ValueError("neural_layer_factory must be provided when include_neural=True")
        layers.append(layer_cfg.neural_layer_factory())

    return layers


def build_congress(config: IntegrationConfig) -> SequentialCongressModel:
    lobbyists, whips, speaker, president = build_advanced_actors(config)
    aggregation_strategy = build_aggregation_strategy(config)

    congress = SequentialCongressModel(id=None)
    for i in range(1, config.num_actors + 1):
        voter_layers = build_layers(config, lobbyists, whips)
        voter = SequentialVoter(
            id=None,
            name=f"Rep-{i}",
            layers=voter_layers,
            aggregation_strategy=aggregation_strategy,
        )
        congress.add_congressman(voter)

    congress.lobbysts = lobbyists
    for whip in whips:
        congress.add_whip(whip)
    congress.set_speaker(speaker)
    congress.set_president(president)

    congress.compile()
    return congress


def build_bill(config: IntegrationConfig) -> SequentialBill:
    bill = SequentialBill(id=None)
    bill.make_random_position(dim=config.policy_dim)
    return bill


def build_session(config: IntegrationConfig) -> Session:
    set_seed(config.seed)
    congress = build_congress(config)
    bill = build_bill(config)
    return Session(
        n=config.iterations,
        seed=config.seed,
        bill=bill,
        description=config.description,
        congress_model=congress,
    )


def build_engine(config: IntegrationConfig) -> SequentialMonteCarlo:
    set_seed(config.seed)
    session = build_session(config)
    return SequentialMonteCarlo(session_params=session)
