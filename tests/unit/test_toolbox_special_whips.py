from policyflux.toolbox.special_actors.whips import SequentialWhip


def test_whip_defaults_generate_id_name_and_mid_values() -> None:
    whip = SequentialWhip()

    assert isinstance(whip.id, int)
    assert whip.name.startswith("Whip_")
    assert whip.discipline_strength == 0.5
    assert whip.party_line_support == 0.5


def test_whip_constructor_clamps_strength_and_party_line() -> None:
    whip = SequentialWhip(id=10, discipline_strength=2.0, party_line_support=-3.0)

    assert whip.id == 10
    assert whip.name == "Whip_10"
    assert whip.discipline_strength == 1.0
    assert whip.party_line_support == 0.0


def test_whip_custom_name_and_get_influence() -> None:
    whip = SequentialWhip(id=7, name="ChiefWhip", discipline_strength=0.73)

    assert whip.name == "ChiefWhip"
    assert whip.get_influence() == 0.73


def test_whip_setters_clamp_values() -> None:
    whip = SequentialWhip(id=1)

    whip.set_discipline_strength(-1.0)
    assert whip.discipline_strength == 0.0

    whip.set_discipline_strength(9.0)
    assert whip.discipline_strength == 1.0

    whip.set_party_line_support(-0.2)
    assert whip.party_line_support == 0.0

    whip.set_party_line_support(1.8)
    assert whip.party_line_support == 1.0
