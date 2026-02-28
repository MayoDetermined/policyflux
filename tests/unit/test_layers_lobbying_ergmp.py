"""Unit tests for LobbyingERGMPLayer."""

import pytest

from policyflux.exceptions import ValidationError
from policyflux.layers.lobbying_ergmp import LobbyingERGMPLayer
from policyflux.models.lobbying_ergmp import LobbyingERGMPModel
from policyflux.toolbox.special_actors.lobby import SequentialLobbyist


def test_lobbying_ergmp_layer_construction_valid() -> None:
    """Test valid layer construction."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    layer = LobbyingERGMPLayer(ergmp_model=model, intensity=0.3)
    assert layer.intensity == pytest.approx(0.3)


def test_lobbying_ergmp_layer_invalid_intensity_raises() -> None:
    """Test that invalid intensity raises."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    with pytest.raises(ValidationError):
        LobbyingERGMPLayer(ergmp_model=model, intensity=1.5)


def test_lobbying_ergmp_layer_negative_intensity_raises() -> None:
    """Test that negative intensity raises."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    with pytest.raises(ValidationError):
        LobbyingERGMPLayer(ergmp_model=model, intensity=-0.1)


def test_lobbying_ergmp_layer_set_intensity() -> None:
    """Test updating intensity."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    layer = LobbyingERGMPLayer(ergmp_model=model, intensity=0.5)
    layer.set_intensity(0.7)
    assert layer.intensity == pytest.approx(0.7)


def test_lobbying_ergmp_layer_clamps_intensity() -> None:
    """Test that set_intensity clamps to [0, 1]."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    layer = LobbyingERGMPLayer(ergmp_model=model, intensity=0.5)

    layer.set_intensity(1.5)
    assert layer.intensity == pytest.approx(1.0)

    layer.set_intensity(-0.5)
    assert layer.intensity == pytest.approx(0.0)


def test_lobbying_ergmp_layer_add_lobbyist() -> None:
    """Test adding a lobbyist."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    layer = LobbyingERGMPLayer(ergmp_model=model)

    lobbyist = SequentialLobbyist(id=1, influence_strength=0.5, stance=1.0)
    layer.add_lobbyist(lobbyist, lobbyist_id=0)

    assert 0 in layer.lobbyists
    assert layer.lobbyists[0] is lobbyist


def test_lobbying_ergmp_layer_add_lobbyist_exceeds_capacity_raises() -> None:
    """Test that adding lobbyist beyond model capacity raises."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    layer = LobbyingERGMPLayer(ergmp_model=model)

    lobbyist = SequentialLobbyist(id=1, influence_strength=0.5, stance=1.0)
    with pytest.raises(ValidationError):
        layer.add_lobbyist(lobbyist, lobbyist_id=10)


def test_lobbying_ergmp_layer_delete_lobbyist() -> None:
    """Test deleting a lobbyist."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    layer = LobbyingERGMPLayer(ergmp_model=model)

    lobbyist = SequentialLobbyist(id=1, influence_strength=0.5, stance=1.0)
    layer.add_lobbyist(lobbyist, lobbyist_id=0)
    assert len(layer.lobbyists) == 1

    deleted = layer.delete_lobbyist(0)
    assert deleted is True
    assert len(layer.lobbyists) == 0


def test_lobbying_ergmp_layer_delete_nonexistent_lobbyist() -> None:
    """Test that deleting nonexistent lobbyist returns False."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    layer = LobbyingERGMPLayer(ergmp_model=model)

    deleted = layer.delete_lobbyist(99)
    assert deleted is False


def test_lobbying_ergmp_layer_call_no_legislator_id() -> None:
    """Test call() without legislator ID applies base intensity."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    layer = LobbyingERGMPLayer(ergmp_model=model, intensity=0.5)

    result = layer.call([0.5, 0.5], base_prob=0.5)
    assert 0.0 <= result <= 1.0
    # With intensity=0.5 and base_prob=0.5: result = 0.5 + (1-0.5)*0.5 = 0.75
    assert result == pytest.approx(0.75)


def test_lobbying_ergmp_layer_call_with_legislator_id() -> None:
    """Test call() with legislator ID uses network."""
    model = LobbyingERGMPModel(n_lobbyists=2, n_legislators=2)
    model.adjacency = [[1, 0], [0, 1]]  # Lobbyist 0 -> Legislator 0, Lobbyist 1 -> Legislator 1

    layer = LobbyingERGMPLayer(ergmp_model=model, intensity=0.2)

    lobbyist0 = SequentialLobbyist(id=10, influence_strength=0.8, stance=1.0)
    layer.add_lobbyist(lobbyist0, lobbyist_id=0)

    # Legislator 0 is connected to lobbyist 0
    result = layer.call([0.5, 0.5], base_prob=0.5, actor_legislator_id=0)
    assert 0.0 <= result <= 1.0
    # intensity=0.2, lobbyist_pressure=0.8*1.0=0.8, combined=1.0 (clamped)
    # result = 0.5 + (1-0.5)*1.0 = 1.0
    assert result == pytest.approx(1.0)


def test_lobbying_ergmp_layer_call_legislator_no_connections() -> None:
    """Test call() for legislator with no connections."""
    model = LobbyingERGMPModel(n_lobbyists=2, n_legislators=2)
    model.adjacency = [[1, 0], [0, 0]]  # Only lobbyist 0 -> legislator 0

    layer = LobbyingERGMPLayer(ergmp_model=model, intensity=0.3)

    lobbyist0 = SequentialLobbyist(id=10, influence_strength=0.8, stance=1.0)
    layer.add_lobbyist(lobbyist0, lobbyist_id=0)

    # Legislator 1 has no connections
    result = layer.call([0.5, 0.5], base_prob=0.5, actor_legislator_id=1)
    # intensity=0.3, no connected lobbyists, combined=0.3
    # result = 0.5 + (1-0.5)*0.3 = 0.65
    assert result == pytest.approx(0.65)


def test_lobbying_ergmp_layer_call_invalid_legislator_id() -> None:
    """Test call() with invalid legislator ID."""
    model = LobbyingERGMPModel(n_lobbyists=2, n_legislators=2)
    layer = LobbyingERGMPLayer(ergmp_model=model, intensity=0.5)

    # Invalid legislator ID should fall back to intensity only
    result = layer.call([0.5, 0.5], base_prob=0.5, actor_legislator_id=99)
    assert 0.0 <= result <= 1.0
    assert result == pytest.approx(0.75)  # 0.5 + (1-0.5)*0.5


def test_lobbying_ergmp_layer_negative_pressure() -> None:
    """Test call() with negative pressure (opposing lobby)."""
    model = LobbyingERGMPModel(n_lobbyists=1, n_legislators=1)
    model.adjacency = [[1]]

    layer = LobbyingERGMPLayer(ergmp_model=model, intensity=0.0)

    lobbyist = SequentialLobbyist(id=10, influence_strength=0.8, stance=-1.0)  # Against
    layer.add_lobbyist(lobbyist, lobbyist_id=0)

    result = layer.call([0.5, 0.5], base_prob=0.5, actor_legislator_id=0)
    # intensity=0.0, lobbyist_pressure=-0.8, combined=-0.8
    # result = 0.5 * (1 + (-0.8)) = 0.5 * 0.2 = 0.1
    assert result == pytest.approx(0.1)


def test_lobbying_ergmp_layer_compile_no_raise() -> None:
    """Test that compile() doesn't raise."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    layer = LobbyingERGMPLayer(ergmp_model=model)
    layer.compile()  # Should not raise


def test_lobbying_ergmp_layer_multiple_connected_lobbyists() -> None:
    """Test aggregation of multiple connected lobbyists."""
    model = LobbyingERGMPModel(n_lobbyists=3, n_legislators=1)
    model.adjacency = [[1], [1], [1]]  # All 3 lobbyists connected to legislator 0

    layer = LobbyingERGMPLayer(ergmp_model=model, intensity=0.0)

    # Two lobbyists in favor, one against
    layer.add_lobbyist(
        SequentialLobbyist(id=1, influence_strength=0.5, stance=1.0), lobbyist_id=0
    )
    layer.add_lobbyist(
        SequentialLobbyist(id=2, influence_strength=0.5, stance=1.0), lobbyist_id=1
    )
    layer.add_lobbyist(
        SequentialLobbyist(id=3, influence_strength=0.5, stance=-1.0), lobbyist_id=2
    )

    result = layer.call([0.5, 0.5], base_prob=0.5, actor_legislator_id=0)
    # Average pressure = (0.5*1.0 + 0.5*1.0 + 0.5*(-1.0)) / 3 = 0.5 / 3 ≈ 0.167
    # result = 0.5 + (1-0.5)*0.167 ≈ 0.583
    assert result == pytest.approx(0.5833, rel=0.01)
