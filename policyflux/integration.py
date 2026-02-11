"""High-level modular integration helpers for PolicyFlux.

This module provides builder utilities to wire together models, layers,
advanced actors, and simulation engines in a configurable way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeAlias

from .core.aggregation_strategy import (
    AggregationStrategy,
    AverageAggregation,
    MultiplicativeAggregation,
    SequentialAggregation,
    WeightedAggregation,
)
from .core.types import PolicySpace
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
    IdealPointLayer,
    LobbyingLayer,
    MediaPressureLayer,
    PartyDisciplineLayer,
    PublicOpinionLayer,
    GovernmentAgendaLayer,
)

from .core.executive import ExecutiveType
from .models.executive_systems import (
    President,
    PrimeMinister,
    PresidentialExecutive,
    ParliamentaryExecutive,
    SemiPresidentialExecutive,
)

from .pfrandom import random as pf_random
from .pfrandom import set_seed
from policyflux.core import executive


@dataclass
class LayerConfig:
    include_ideal_point: bool = True
    include_public_opinion: bool = True
    include_lobbying: bool = True
    include_media_pressure: bool = True
    include_party_discipline: bool = True
    include_government_agenda: bool = False
    include_neural: bool = False

    layer_names: Optional[List[str]] = None
    layer_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    public_support: float = 0.5
    lobbying_intensity: float = 0.0
    media_pressure: float = 0.0
    party_line_support: float = 0.5
    party_discipline_strength: float = 0.5
    government_agenda_pm_strength: float = 0.6

    neural_layer_factory: Optional[Callable[[], object]] = None


@dataclass
class LayerBuilderContext:
    policy_dim: int
    layer_config: LayerConfig
    lobbyists: Iterable[SequentialLobbyer]
    whips: Iterable[SequentialWhip]


LayerFactory: TypeAlias = Callable[[LayerBuilderContext, Dict[str, Any]], object]


LAYER_REGISTRY: Dict[str, LayerFactory] = {}


def register_layer(name: str, factory: LayerFactory) -> None:
    LAYER_REGISTRY[name] = factory


def build_layer_by_name(name: str, context: LayerBuilderContext) -> object:
    factory = LAYER_REGISTRY.get(name)
    if factory is None:
        raise KeyError(f"Layer '{name}' is not registered")
    overrides: Dict[str, Any] = context.layer_config.layer_overrides.get(name, {})
    return factory(context, overrides)


def _register_default_layers() -> None:
    def ideal_point_factory(ctx: LayerBuilderContext, overrides: Dict[str, Any]) -> object:
        space_list: List[float] = overrides.get(
            "space", [pf_random() for _ in range(ctx.policy_dim)]
        )
        status_quo_list: List[float] = overrides.get("status_quo", [0.5] * ctx.policy_dim)

        # Convert lists to PolicySpace objects
        space = PolicySpace(len(space_list))
        space.set_position(space_list)

        status_quo = PolicySpace(len(status_quo_list))
        status_quo.set_position(status_quo_list)

        return IdealPointLayer(space=space, status_quo=status_quo)

    def public_opinion_factory(ctx: LayerBuilderContext, overrides: Dict[str, Any]) -> object:
        support: float = overrides.get("support_level", ctx.layer_config.public_support)
        return PublicOpinionLayer(support_level=support)

    def lobbying_factory(ctx: LayerBuilderContext, overrides: Dict[str, Any]) -> object:
        intensity: float = overrides.get("intensity", ctx.layer_config.lobbying_intensity)
        lobbying = LobbyingLayer(intensity=intensity)
        for lobbyer in ctx.lobbyists:
            lobbying.add_lobbyst(lobbyer)
        return lobbying

    def media_factory(ctx: LayerBuilderContext, overrides: Dict[str, Any]) -> object:
        pressure: float = overrides.get("pressure", ctx.layer_config.media_pressure)
        return MediaPressureLayer(pressure=pressure)

    def party_factory(ctx: LayerBuilderContext, overrides: Dict[str, Any]) -> object:
        party_line: float = overrides.get("party_line_support", ctx.layer_config.party_line_support)
        discipline: float = overrides.get(
            "discipline_base_strength", ctx.layer_config.party_discipline_strength
        )
        return PartyDisciplineLayer(
            party_whips=list(ctx.whips),
            discipline_base_strength=discipline,
            party_line_support=party_line,
        )

    def government_agenda_factory(ctx: LayerBuilderContext, overrides: Dict[str, Any]) -> object:
        pm_strength: float = overrides.get("pm_party_strength", ctx.layer_config.government_agenda_pm_strength)
        return GovernmentAgendaLayer(pm_party_strength=pm_strength)

    register_layer("ideal_point", ideal_point_factory)
    register_layer("public_opinion", public_opinion_factory)
    register_layer("lobbying", lobbying_factory)
    register_layer("media_pressure", media_factory)
    register_layer("party_discipline", party_factory)
    register_layer("government_agenda", government_agenda_factory)


_register_default_layers()


@dataclass
class AdvancedActorsConfig:
    # System type
    executive_type: ExecutiveType = ExecutiveType.PRESIDENTIAL
    
    # Lobbyists
    n_lobbyists: int = 0
    lobbyist_strength: float = 0.5
    lobbyist_stance: float = 1.0

    # Whips 
    n_whips: int = 0
    whip_discipline_strength: float = 0.5
    whip_party_line_support: float = 0.5

    # Speaker 
    speaker_agenda_support: float = 0.5
    
    # PRESIDENTIAL SYSTEM
    president_approval_rating: float = 0.5
    veto_override_threshold: float = 2/3
    
    # PARLIAMENTARY SYSTEM
    pm_party_strength: float = 0.55
    confidence_threshold: float = 0.5
    government_bill_rate: float = 0.7  # % of bills that are government bills
    
    # SEMI-PRESIDENTIAL SYSTEM
    semi_president_approval: float = 0.5
    semi_pm_party_strength: float = 0.55


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


def build_executive(config: IntegrationConfig) -> Optional[Executive]:
    """Build executive branch based on configuration.

    Args:
        config: Integration configuration with actors_config

    Returns:
        Executive instance (Presidential, Parliamentary, or SemiPresidential) or None
    """
    exec_type = config.actors_config.executive_type

    if exec_type == ExecutiveType.PRESIDENTIAL:
        president = President(
            approval_rating=config.actors_config.president_approval_rating,
            name="President"
        )
        return PresidentialExecutive(
            president=president,
            veto_override_threshold=config.actors_config.veto_override_threshold
        )

    elif exec_type == ExecutiveType.PARLIAMENTARY:
        prime_minister = PrimeMinister(
            party_strength=config.actors_config.pm_party_strength,
            name="PrimeMinister"
        )
        # Auto-enable government agenda layer for parliamentary systems
        if not config.layer_config.include_government_agenda:
            config.layer_config.include_government_agenda = True
            config.layer_config.government_agenda_pm_strength = config.actors_config.pm_party_strength

        return ParliamentaryExecutive(
            prime_minister=prime_minister,
            confidence_threshold=config.actors_config.confidence_threshold
        )

    elif exec_type == ExecutiveType.SEMI_PRESIDENTIAL:
        president = President(
            approval_rating=config.actors_config.semi_president_approval,
            name="President"
        )
        prime_minister = PrimeMinister(
            party_strength=config.actors_config.semi_pm_party_strength,
            name="PrimeMinister"
        )
        return SemiPresidentialExecutive(
            president=president,
            prime_minister=prime_minister
        )

    return None


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
        # Convert lists to PolicySpace objects
        space = PolicySpace(config.policy_dim)
        space.set_position([pf_random() for _ in range(config.policy_dim)])

        status_quo = PolicySpace(config.policy_dim)
        status_quo.set_position([0.5] * config.policy_dim)

        layers.append(
            IdealPointLayer(
                space=space,
                status_quo=status_quo,
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

    if layer_cfg.include_government_agenda:
        government_layer = GovernmentAgendaLayer(
            id=None,
            pm_party_strength=layer_cfg.government_agenda_pm_strength,
        )
        layers.append(government_layer)

    if layer_cfg.include_neural:
        if layer_cfg.neural_layer_factory is None:
            raise ValueError("neural_layer_factory must be provided when include_neural=True")
        layers.append(layer_cfg.neural_layer_factory())

    return layers


def build_congress(config: IntegrationConfig) -> SequentialCongressModel:
    lobbyists, whips, speaker, president = build_advanced_actors(config)
    aggregation_strategy = build_aggregation_strategy(config)
    executive = build_executive(config)

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

    # Set executive system if built
    if executive is not None:
        congress.set_executive(executive)

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


# ============ HELPER FUNCTIONS FOR EASY CONFIGURATION ============

def create_presidential_config(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    president_approval: float = 0.5,
    veto_override_threshold: float = 2/3,
    **kwargs
) -> IntegrationConfig:
    """Create configuration for a presidential system (US-style).

    Args:
        num_actors: Number of congressmen
        policy_dim: Policy space dimensionality
        iterations: Monte Carlo iterations
        seed: Random seed
        president_approval: Presidential approval rating [0-1]
        veto_override_threshold: Threshold to override veto (default 2/3)
        **kwargs: Additional configuration overrides

    Returns:
        IntegrationConfig configured for presidential system
    """
    actors_config = AdvancedActorsConfig(
        executive_type=ExecutiveType.PRESIDENTIAL,
        president_approval_rating=president_approval,
        veto_override_threshold=veto_override_threshold,
        **{k: v for k, v in kwargs.items() if k in AdvancedActorsConfig.__dataclass_fields__}
    )

    layer_config = LayerConfig(
        **{k: v for k, v in kwargs.items() if k in LayerConfig.__dataclass_fields__}
    )

    return IntegrationConfig(
        num_actors=num_actors,
        policy_dim=policy_dim,
        iterations=iterations,
        seed=seed,
        layer_config=layer_config,
        actors_config=actors_config,
        **{k: v for k, v in kwargs.items() if k in IntegrationConfig.__dataclass_fields__
           and k not in ['layer_config', 'actors_config']}
    )


def create_parliamentary_config(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    pm_party_strength: float = 0.55,
    confidence_threshold: float = 0.5,
    government_bill_rate: float = 0.7,
    **kwargs
) -> IntegrationConfig:
    """Create configuration for a parliamentary system (UK/Canada-style).

    Args:
        num_actors: Number of MPs
        policy_dim: Policy space dimensionality
        iterations: Monte Carlo iterations
        seed: Random seed
        pm_party_strength: PM's party strength [0-1]
        confidence_threshold: Threshold for confidence votes
        government_bill_rate: Proportion of bills that are government bills
        **kwargs: Additional configuration overrides

    Returns:
        IntegrationConfig configured for parliamentary system
    """
    actors_config = AdvancedActorsConfig(
        executive_type=ExecutiveType.PARLIAMENTARY,
        pm_party_strength=pm_party_strength,
        confidence_threshold=confidence_threshold,
        government_bill_rate=government_bill_rate,
        **{k: v for k, v in kwargs.items() if k in AdvancedActorsConfig.__dataclass_fields__}
    )

    layer_config = LayerConfig(
        include_government_agenda=True,
        government_agenda_pm_strength=pm_party_strength,
        **{k: v for k, v in kwargs.items() if k in LayerConfig.__dataclass_fields__}
    )

    return IntegrationConfig(
        num_actors=num_actors,
        policy_dim=policy_dim,
        iterations=iterations,
        seed=seed,
        layer_config=layer_config,
        actors_config=actors_config,
        **{k: v for k, v in kwargs.items() if k in IntegrationConfig.__dataclass_fields__
           and k not in ['layer_config', 'actors_config']}
    )


def create_semi_presidential_config(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    president_approval: float = 0.5,
    pm_party_strength: float = 0.55,
    **kwargs
) -> IntegrationConfig:
    """Create configuration for a semi-presidential system (France/Poland-style).

    Args:
        num_actors: Number of representatives
        policy_dim: Policy space dimensionality
        iterations: Monte Carlo iterations
        seed: Random seed
        president_approval: Presidential approval rating [0-1]
        pm_party_strength: PM's party strength [0-1]
        **kwargs: Additional configuration overrides

    Returns:
        IntegrationConfig configured for semi-presidential system
    """
    actors_config = AdvancedActorsConfig(
        executive_type=ExecutiveType.SEMI_PRESIDENTIAL,
        semi_president_approval=president_approval,
        semi_pm_party_strength=pm_party_strength,
        **{k: v for k, v in kwargs.items() if k in AdvancedActorsConfig.__dataclass_fields__}
    )

    layer_config = LayerConfig(
        **{k: v for k, v in kwargs.items() if k in LayerConfig.__dataclass_fields__}
    )

    return IntegrationConfig(
        num_actors=num_actors,
        policy_dim=policy_dim,
        iterations=iterations,
        seed=seed,
        layer_config=layer_config,
        actors_config=actors_config,
        **{k: v for k, v in kwargs.items() if k in IntegrationConfig.__dataclass_fields__
           and k not in ['layer_config', 'actors_config']}
    )
