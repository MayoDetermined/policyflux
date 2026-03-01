from policyflux.layers.party_layers import PartyDisciplineLayer
from policyflux.toolbox.special_actors.whips import SequentialWhip


def test_party_discipline_construction() -> None:
    layer = PartyDisciplineLayer()
    assert layer.name == "PartyDiscipline"


def test_party_discipline_call_in_range() -> None:
    layer = PartyDisciplineLayer()
    result = layer.call([0.5, 0.5], base_prob=0.5)
    assert 0.0 <= result <= 1.0


def test_party_discipline_compile_no_raise() -> None:
    layer = PartyDisciplineLayer()
    layer.compile()


def test_party_discipline_add_and_delete_whip_paths() -> None:
    whip = SequentialWhip(id=101, discipline_strength=0.7, party_line_support=0.6)
    layer = PartyDisciplineLayer(party_whips=[whip])

    extra = SequentialWhip(id=202, discipline_strength=0.5, party_line_support=0.4)
    layer.add_whip(extra)
    assert len(layer.whips) == 2

    assert layer.delete_whip(202) is True
    assert layer.delete_whip(999) is False


def test_party_discipline_setters_clamp_values() -> None:
    layer = PartyDisciplineLayer(discipline_base_strength=2.0, party_line_support=-2.0)
    assert layer.discipline_base_strength == 1.0
    assert layer.party_line_support == 0.0

    layer.set_party_line_support(5.0)
    layer.set_discipline_strength(-1.0)
    assert layer.party_line_support == 1.0
    assert layer.discipline_base_strength == 0.0


def test_party_discipline_aggregates_whips_and_speaker_agenda() -> None:
    whips = [
        SequentialWhip(id=1, discipline_strength=0.8, party_line_support=0.6),
        SequentialWhip(id=2, discipline_strength=0.2, party_line_support=0.4),
    ]
    layer = PartyDisciplineLayer(party_whips=whips)

    no_speaker = layer.call([0.2, 0.8], base_prob=0.5)
    with_speaker = layer.call([0.2, 0.8], base_prob=0.5, speaker_agenda_support=1.0)

    assert no_speaker == 0.5
    assert with_speaker > no_speaker
    assert 0.0 <= with_speaker <= 1.0
