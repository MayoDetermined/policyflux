import pytest

from policyflux.exceptions import ValidationError
from policyflux.layers.lobbying import LobbyingLayer
from policyflux.toolbox.special_actors.lobby import SequentialLobbyist


def test_lobbying_construction_valid() -> None:
    layer = LobbyingLayer(intensity=0.3)
    assert layer.intensity == pytest.approx(0.3)


def test_lobbying_invalid_intensity_raises() -> None:
    with pytest.raises(ValidationError):
        LobbyingLayer(intensity=1.5)


def test_lobbying_negative_intensity_raises() -> None:
    with pytest.raises(ValidationError):
        LobbyingLayer(intensity=-0.1)


def test_lobbying_add_and_delete_lobbyist() -> None:
    layer = LobbyingLayer(intensity=0.5)
    lobby = SequentialLobbyist(id=10, influence_strength=0.5, stance=1.0)
    layer.add_lobbyist(lobby)
    assert len(layer.lobbyists) == 1
    layer.delete_lobbyist(10)
    assert len(layer.lobbyists) == 0


def test_lobbying_pop_lobbyist() -> None:
    layer = LobbyingLayer(intensity=0.5)
    lobby = SequentialLobbyist(id=10, influence_strength=0.5, stance=1.0)
    layer.add_lobbyist(lobby)
    popped = layer.pop_lobbyist()
    assert popped is lobby
    assert len(layer.lobbyists) == 0


def test_lobbying_call_returns_float_in_range() -> None:
    layer = LobbyingLayer(intensity=0.5)
    result = layer.call([0.5, 0.5], base_prob=0.5)
    assert 0.0 <= result <= 1.0


def test_lobbying_compile_no_raise() -> None:
    layer = LobbyingLayer(intensity=0.0)
    layer.compile()
