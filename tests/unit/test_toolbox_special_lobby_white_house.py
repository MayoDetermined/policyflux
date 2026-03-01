from policyflux.toolbox.special_actors.lobby import SequentialLobbyist
from policyflux.toolbox.special_actors.white_house import SequentialPresident


def test_lobbyist_defaults_generate_id_name_and_values() -> None:
    lobbyist = SequentialLobbyist()

    assert isinstance(lobbyist.id, int)
    assert lobbyist.name.startswith("Lobbyist_")
    assert lobbyist.influence_strength == 0.5
    assert lobbyist.stance == 1.0


def test_lobbyist_constructor_clamps_values() -> None:
    lobbyist = SequentialLobbyist(id=11, influence_strength=9.0, stance=-3.0)

    assert lobbyist.id == 11
    assert lobbyist.name == "Lobbyist_11"
    assert lobbyist.influence_strength == 1.0
    assert lobbyist.stance == -1.0


def test_lobbyist_custom_name_influence_and_setters() -> None:
    lobbyist = SequentialLobbyist(id=5, name="LobbyA", influence_strength=0.72, stance=0.4)

    assert lobbyist.name == "LobbyA"
    assert lobbyist.get_influence() == 0.72

    lobbyist.set_influence_strength(-1.0)
    assert lobbyist.influence_strength == 0.0

    lobbyist.set_influence_strength(3.0)
    assert lobbyist.influence_strength == 1.0

    lobbyist.set_stance(2.0)
    assert lobbyist.stance == 1.0

    lobbyist.set_stance(-2.0)
    assert lobbyist.stance == -1.0


def test_president_defaults_generate_id_name_and_ideology() -> None:
    president = SequentialPresident()

    assert isinstance(president.id, int)
    assert president.name.startswith("President_")
    assert president.ideology.dimensions == 2
    assert president.approval_rating == 0.5


def test_president_constructor_clamps_approval_and_custom_name() -> None:
    president = SequentialPresident(id=8, name="POTUS", approval_rating=4.0)

    assert president.id == 8
    assert president.name == "POTUS"
    assert president.approval_rating == 1.0


def test_president_methods_return_expected_values() -> None:
    president = SequentialPresident(id=9, approval_rating=0.63)

    assert president.get_influence_on_bill(bill=object()) == 0.63
    assert president.can_veto_bill(bill=object()) is True

    president.set_approval_rating(-1.0)
    assert president.approval_rating == 0.0

    president.set_approval_rating(2.5)
    assert president.approval_rating == 1.0
