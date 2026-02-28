import pytest

from policyflux.core.types import PolicySpace
from policyflux.exceptions import DimensionMismatchError
from policyflux.layers.ideal_point import IdealPointLayer


def _make_layer(voter_pos: list[float], sq_pos: list[float]) -> IdealPointLayer:
    dim = len(voter_pos)
    space = PolicySpace(dim)
    space.set_position(voter_pos)
    status_quo = PolicySpace(dim)
    status_quo.set_position(sq_pos)
    return IdealPointLayer(input_dim=dim, output_dim=dim, space=space, status_quo=status_quo)


def test_ideal_point_construction() -> None:
    layer = IdealPointLayer(input_dim=2, output_dim=2)
    assert layer.name == "IdealPoint"


def test_ideal_point_call_returns_float_in_range() -> None:
    layer = _make_layer([0.3, 0.7], [0.5, 0.5])
    result = layer.call([0.4, 0.6])
    assert 0.0 <= result <= 1.0


def test_ideal_point_dimension_mismatch_raises() -> None:
    layer = _make_layer([0.3, 0.7], [0.5, 0.5])
    with pytest.raises(DimensionMismatchError):
        layer.call([0.5, 0.5, 0.5])


def test_ideal_point_closer_to_bill_higher_prob() -> None:
    sq = [0.5, 0.5]
    close_layer = _make_layer([0.4, 0.6], sq)
    far_layer = _make_layer([0.0, 0.0], sq)
    bill = [0.4, 0.6]
    assert close_layer.call(bill) > far_layer.call(bill)


def test_ideal_point_compile_no_raise() -> None:
    layer = IdealPointLayer(input_dim=2, output_dim=2)
    layer.compile()
