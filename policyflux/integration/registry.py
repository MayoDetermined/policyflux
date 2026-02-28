from collections.abc import Callable
from typing import Any, TypeAlias

from policyflux.core.pf_typing import PolicySpace
from policyflux.exceptions import RegistryError
from policyflux.integration.builders.layer_builder import LayerBuilderContext
from policyflux.layers import GovernmentAgendaLayer
from policyflux.layers.ideal_point import IdealPointLayer
from policyflux.layers.lobbying import LobbyingLayer
from policyflux.layers.media_pressure import MediaPressureLayer
from policyflux.layers.party import PartyDisciplineLayer
from policyflux.layers.public_pressure import PublicOpinionLayer
from policyflux.pfrandom import random as pf_random

LayerFactory: TypeAlias = Callable[[LayerBuilderContext, dict[str, Any]], object]


LAYER_REGISTRY: dict[str, LayerFactory] = {}


def register_layer(name: str, factory: LayerFactory) -> None:
    LAYER_REGISTRY[name] = factory


def build_layer_by_name(name: str, context: LayerBuilderContext) -> object:
    factory = LAYER_REGISTRY.get(name)
    if factory is None:
        raise RegistryError(f"Layer '{name}' is not registered")
    overrides: dict[str, Any] = context.layer_config.layer_overrides.get(name, {})
    return factory(context, overrides)


def _register_default_layers() -> None:
    def ideal_point_factory(ctx: LayerBuilderContext, overrides: dict[str, Any]) -> object:
        space_list: list[float] = overrides.get(
            "space", [pf_random() for _ in range(ctx.policy_dim)]
        )
        status_quo_list: list[float] = overrides.get("status_quo", [0.5] * ctx.policy_dim)

        # Convert lists to PolicySpace objects
        space = PolicySpace(len(space_list))
        space.set_position(space_list)

        status_quo = PolicySpace(len(status_quo_list))
        status_quo.set_position(status_quo_list)

        return IdealPointLayer(space=space, status_quo=status_quo)

    def public_opinion_factory(ctx: LayerBuilderContext, overrides: dict[str, Any]) -> object:
        support: float = overrides.get("support_level", ctx.layer_config.public_support)
        return PublicOpinionLayer(support_level=support)

    def lobbying_factory(ctx: LayerBuilderContext, overrides: dict[str, Any]) -> object:
        intensity: float = overrides.get("intensity", ctx.layer_config.lobbying_intensity)
        lobbying = LobbyingLayer(intensity=intensity)
        for lobbyist in ctx.lobbyists:
            lobbying.add_lobbyist(lobbyist)
        return lobbying

    def media_factory(ctx: LayerBuilderContext, overrides: dict[str, Any]) -> object:
        pressure: float = overrides.get("pressure", ctx.layer_config.media_pressure)
        return MediaPressureLayer(pressure=pressure)

    def party_factory(ctx: LayerBuilderContext, overrides: dict[str, Any]) -> object:
        party_line: float = overrides.get("party_line_support", ctx.layer_config.party_line_support)
        discipline: float = overrides.get(
            "discipline_base_strength", ctx.layer_config.party_discipline_strength
        )
        return PartyDisciplineLayer(
            party_whips=list(ctx.whips),
            discipline_base_strength=discipline,
            party_line_support=party_line,
        )

    def government_agenda_factory(ctx: LayerBuilderContext, overrides: dict[str, Any]) -> object:
        pm_strength: float = overrides.get(
            "pm_party_strength", ctx.layer_config.government_agenda_pm_strength
        )
        return GovernmentAgendaLayer(pm_party_strength=pm_strength)

    register_layer("ideal_point", ideal_point_factory)
    register_layer("public_opinion", public_opinion_factory)
    register_layer("lobbying", lobbying_factory)
    register_layer("media_pressure", media_factory)
    register_layer("party_discipline", party_factory)
    register_layer("government_agenda", government_agenda_factory)


_register_default_layers()
