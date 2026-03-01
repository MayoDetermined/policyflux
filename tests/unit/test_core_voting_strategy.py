from policyflux.core.contexts import VotingContext
from policyflux.core.pf_typing import PolicyPosition
from policyflux.core.voting_strategy import DeterministicVoting, ProbabilisticVoting, SoftVoting


def _context() -> VotingContext:
    return VotingContext(
        bill_position=PolicyPosition((0.5, 0.5)),
        actor_ideal_point=PolicyPosition((0.5, 0.5)),
        base_prob=0.5,
    )


def test_deterministic_voting_threshold() -> None:
    strategy = DeterministicVoting()
    context = _context()

    assert strategy.decide(0.49, context) is False
    assert strategy.decide(0.5, context) is True
    assert strategy.decide(0.99, context) is True


def test_probabilistic_voting_uses_random_draw(monkeypatch) -> None:
    strategy = ProbabilisticVoting()
    context = _context()

    monkeypatch.setattr("policyflux.pfrandom.random", lambda: 0.2)
    assert strategy.decide(0.3, context) is True

    monkeypatch.setattr("policyflux.pfrandom.random", lambda: 0.8)
    assert strategy.decide(0.3, context) is False


def test_soft_voting_returns_probability() -> None:
    strategy = SoftVoting()
    context = _context()
    assert strategy.decide(0.37, context) == 0.37
