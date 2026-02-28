import dataclasses

import pytest

import policyflux.pfrandom as pfrandom
from policyflux.core.types import PolicyPosition, PolicySpace
from policyflux.exceptions import DimensionMismatchError, ValidationError


def test_policy_space_valid_construction() -> None:
    space = PolicySpace(3)
    assert space.dimensions == 3
    assert space.position == [0.0, 0.0, 0.0]


def test_policy_space_zero_dimensions_raises() -> None:
    with pytest.raises(ValidationError):
        PolicySpace(0)


def test_policy_space_negative_dimensions_raises() -> None:
    with pytest.raises(ValidationError):
        PolicySpace(-1)


def test_policy_space_set_get_position() -> None:
    space = PolicySpace(2)
    space.set_position([0.3, 0.7])
    assert space.get_position() == pytest.approx([0.3, 0.7])


def test_policy_space_set_position_copies_input() -> None:
    space = PolicySpace(2)
    original = [0.3, 0.7]
    space.set_position(original)
    original[0] = 999.0
    assert space.get_position() == pytest.approx([0.3, 0.7])


def test_policy_space_get_position_returns_copy() -> None:
    space = PolicySpace(2)
    space.set_position([0.3, 0.7])
    returned = space.get_position()
    returned[0] = 999.0
    assert space.get_position() == pytest.approx([0.3, 0.7])


def test_policy_space_wrong_dimension_raises() -> None:
    space = PolicySpace(2)
    with pytest.raises(DimensionMismatchError):
        space.set_position([0.1, 0.2, 0.3])


def test_policy_space_str() -> None:
    space = PolicySpace(2)
    result = str(space)
    assert "dimensions=2" in result


def test_policy_position_valid_construction() -> None:
    pos = PolicyPosition((0.5, 0.5))
    assert pos.dimensions == 2


def test_policy_position_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        PolicyPosition((1.5, 0.0))


def test_policy_position_negative_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        PolicyPosition((-0.1, 0.5))


def test_policy_position_frozen() -> None:
    pos = PolicyPosition((0.5, 0.5))
    with pytest.raises(dataclasses.FrozenInstanceError):
        pos.coordinates = (0.1, 0.1)  # type: ignore[misc]


def test_policy_position_distance_to_self_is_zero() -> None:
    pos = PolicyPosition((0.5, 0.5))
    assert pos.distance_to(pos) == pytest.approx(0.0)


def test_policy_position_distance_dimension_mismatch() -> None:
    a = PolicyPosition((0.5,))
    b = PolicyPosition((0.5, 0.5))
    with pytest.raises(DimensionMismatchError):
        a.distance_to(b)


def test_policy_position_utility_decreases_with_distance() -> None:
    voter = PolicyPosition((0.0, 0.0))
    close_bill = PolicyPosition((0.1, 0.1))
    far_bill = PolicyPosition((0.9, 0.9))
    assert voter.utility(close_bill) > voter.utility(far_bill)


def test_policy_position_random_creates_valid() -> None:
    pfrandom.set_seed(42)
    pos = PolicyPosition.random(3)
    assert pos.dimensions == 3
    assert all(0.0 <= c <= 1.0 for c in pos.coordinates)
