from policyflux import pfrandom


def test_set_seed_produces_repeatable_sequence() -> None:
    pfrandom.set_seed(42)
    seq_a = [pfrandom.random() for _ in range(5)]

    pfrandom.set_seed(42)
    seq_b = [pfrandom.random() for _ in range(5)]

    assert seq_a == seq_b


def test_randint_respects_bounds() -> None:
    pfrandom.set_seed(7)
    values = [pfrandom.randint(1, 3) for _ in range(50)]

    assert all(1 <= value <= 3 for value in values)
