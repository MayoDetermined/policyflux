from policyflux.layers.party_layers import PartyDisciplineLayer


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
