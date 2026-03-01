
from policyflux.layers.public_pressure import PublicOpinionLayer


def test_public_opinion_construction() -> None:
    layer = PublicOpinionLayer(support_level=0.6)
    assert layer.name == "PublicOpinion"


def test_public_opinion_call_in_range() -> None:
    layer = PublicOpinionLayer(support_level=0.7)
    result = layer.call([0.5, 0.5], base_prob=0.5)
    assert 0.0 <= result <= 1.0


def test_public_opinion_higher_support_pushes_up() -> None:
    low = PublicOpinionLayer(support_level=0.2)
    high = PublicOpinionLayer(support_level=0.8)
    bill = [0.5, 0.5]
    assert high.call(bill, base_prob=0.5) > low.call(bill, base_prob=0.5)


def test_public_opinion_compile_no_raise() -> None:
    layer = PublicOpinionLayer(support_level=0.5)
    layer.compile()
