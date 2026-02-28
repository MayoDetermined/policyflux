from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from policyflux.exceptions import ConfigurationError

from ...core.pf_typing import PolicySpace
from ...layers.government_agenda import GovernmentAgendaLayer
from ...layers.ideal_point import IdealPointLayer
from ...layers.lobbying import LobbyingLayer
from ...layers.media_pressure import MediaPressureLayer
from ...layers.party import PartyDisciplineLayer
from ...layers.public_pressure import PublicOpinionLayer
from ...pfrandom import random as pf_random
from ...toolbox.special_actors.lobby import SequentialLobbyist
from ...toolbox.special_actors.whips import SequentialWhip
from ..config import IntegrationConfig, LayerConfig


@dataclass
class LayerBuilderContext:
    policy_dim: int
    layer_config: LayerConfig
    lobbyists: Iterable[SequentialLobbyist]
    whips: Iterable[SequentialWhip]


def build_layers(
    config: IntegrationConfig,
    lobbyists: Iterable[SequentialLobbyist],
    whips: Iterable[SequentialWhip],
) -> list[Any]:
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
            from ..registry import build_layer_by_name

            layers.append(build_layer_by_name(name, context))
        if layer_cfg.include_neural:
            if layer_cfg.neural_layer_factory is None:
                raise ConfigurationError(
                    "neural_layer_factory must be provided when include_neural=True"
                )
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
        for lobbyist in lobbyists:
            lobbying.add_lobbyist(lobbyist)
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
            raise ConfigurationError(
                "neural_layer_factory must be provided when include_neural=True"
            )
        layers.append(layer_cfg.neural_layer_factory())

    return layers
