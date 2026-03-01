"""Unit tests for LobbyingERGMPModel."""

import pytest

from policyflux.exceptions import ValidationError
from policyflux.math_models.lobbying_ergmp import LobbyingERGMPModel


def test_lobbying_ergmp_construction_valid() -> None:
    """Test valid model construction."""
    model = LobbyingERGMPModel(
        n_lobbyists=10,
        n_legislators=20,
        theta_density=-2.0,
        theta_transitivity=0.5,
        theta_homophily=0.5,
    )
    assert model.n_lobbyists == 10
    assert model.n_legislators == 20
    assert model.theta_density == -2.0
    assert model.theta_transitivity == 0.5
    assert model.theta_homophily == 0.5


def test_lobbying_ergmp_invalid_n_lobbyists_raises() -> None:
    """Test that invalid lobbyist count raises."""
    with pytest.raises(ValidationError):
        LobbyingERGMPModel(n_lobbyists=0, n_legislators=20)


def test_lobbying_ergmp_invalid_n_legislators_raises() -> None:
    """Test that invalid legislator count raises."""
    with pytest.raises(ValidationError):
        LobbyingERGMPModel(n_lobbyists=10, n_legislators=0)


def test_lobbying_ergmp_set_lobbyist_attribute() -> None:
    """Test setting lobbyist attributes."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    model.set_lobbyist_attribute(0, "ideology", 0.5)
    assert model.get_lobbyist_attribute(0, "ideology") == 0.5


def test_lobbying_ergmp_set_legislator_attribute() -> None:
    """Test setting legislator attributes."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    model.set_legislator_attribute(0, "party", "Democrat")
    assert model.get_legislator_attribute(0, "party") == "Democrat"


def test_lobbying_ergmp_invalid_lobbyist_id_raises() -> None:
    """Test that invalid lobbyist ID raises."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    with pytest.raises(ValidationError):
        model.set_lobbyist_attribute(10, "ideology", 0.5)


def test_lobbying_ergmp_invalid_legislator_id_raises() -> None:
    """Test that invalid legislator ID raises."""
    model = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    with pytest.raises(ValidationError):
        model.set_legislator_attribute(10, "party", "Democrat")


def test_lobbying_ergmp_generate_creates_network() -> None:
    """Test that generate creates a bipartite network."""
    model = LobbyingERGMPModel(
        n_lobbyists=5,
        n_legislators=5,
        theta_density=-3.0,  # Very sparse
    )
    adjacency = model.generate(seed=42)

    assert len(adjacency) == 5  # n_lobbyists
    assert all(len(row) == 5 for row in adjacency)  # n_legislators per row
    assert all(cell in [0, 1] for row in adjacency for cell in row)


def test_lobbying_ergmp_generate_reproducible() -> None:
    """Test that generate is reproducible with seed."""
    model1 = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    adj1 = model1.generate(seed=123)

    model2 = LobbyingERGMPModel(n_lobbyists=5, n_legislators=5)
    adj2 = model2.generate(seed=123)

    assert adj1 == adj2


def test_lobbying_ergmp_get_lobbyist_reach() -> None:
    """Test getting which legislators a lobbyist reaches."""
    model = LobbyingERGMPModel(n_lobbyists=3, n_legislators=3)
    model.adjacency = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
    reach = model.get_lobbyist_reach(0)
    assert reach == [0, 2]


def test_lobbying_ergmp_get_legislator_exposure() -> None:
    """Test getting which lobbyists influence a legislator."""
    model = LobbyingERGMPModel(n_lobbyists=3, n_legislators=3)
    model.adjacency = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
    exposure = model.get_legislator_exposure(0)
    assert exposure == [0, 2]


def test_lobbying_ergmp_get_degree_lobbyist() -> None:
    """Test getting degree of a lobbyist."""
    model = LobbyingERGMPModel(n_lobbyists=3, n_legislators=3)
    model.adjacency = [[1, 1, 0], [0, 1, 1], [1, 0, 0]]
    degree = model.get_degree(is_lobbyist=True, node_id=0)
    assert degree == 2


def test_lobbying_ergmp_get_degree_legislator() -> None:
    """Test getting degree of a legislator."""
    model = LobbyingERGMPModel(n_lobbyists=3, n_legislators=3)
    model.adjacency = [[1, 1, 0], [0, 1, 1], [1, 0, 0]]
    degree = model.get_degree(is_lobbyist=False, node_id=0)
    assert degree == 2  # Columns 0 in rows 0 and 2


def test_lobbying_ergmp_get_density() -> None:
    """Test density calculation."""
    model = LobbyingERGMPModel(n_lobbyists=2, n_legislators=2)
    model.adjacency = [[1, 1], [1, 1]]
    density = model.get_density()
    assert density == pytest.approx(1.0)

    model.adjacency = [[1, 0], [0, 1]]
    density = model.get_density()
    assert density == pytest.approx(0.5)


def test_lobbying_ergmp_get_adjacency_deep_copy() -> None:
    """Test that get_adjacency returns a copy."""
    model = LobbyingERGMPModel(n_lobbyists=2, n_legislators=2)
    model.adjacency = [[1, 0], [0, 1]]

    adj = model.get_adjacency()
    adj[0][0] = 0  # Modify copy

    assert model.adjacency[0][0] == 1  # Original unchanged


def test_lobbying_ergmp_connected_counts() -> None:
    """Test counting connected nodes."""
    model = LobbyingERGMPModel(n_lobbyists=3, n_legislators=3)
    model.adjacency = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]

    assert model.get_connected_lobbyists() == 2
    assert model.get_connected_legislators() == 2


def test_lobbying_ergmp_average_reach_and_exposure() -> None:
    """Test average reach and exposure calculations."""
    model = LobbyingERGMPModel(n_lobbyists=2, n_legislators=2)
    model.adjacency = [[1, 1], [1, 0]]

    avg_reach = model.get_average_lobbyist_reach()
    assert avg_reach == pytest.approx(1.5)

    avg_exposure = model.get_average_legislator_exposure()
    assert avg_exposure == pytest.approx(1.5)
