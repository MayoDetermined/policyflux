# policyflux/core/voting_strategy.py
from abc import ABC, abstractmethod
from typing import Union
from .contexts import VotingContext


class VotingStrategy(ABC):
    """Strategy for converting decision probability to a vote outcome."""

    @abstractmethod
    def decide(self, decision_prob: float, context: VotingContext) -> Union[bool, float]:
        """Convert probability to a vote outcome.

        Returns bool for hard voting strategies, float for soft voting.
        """
        pass


class ProbabilisticVoting(VotingStrategy):
    """Monte Carlo voting: random() < prob."""

    def decide(self, decision_prob: float, context: VotingContext) -> bool:
        from policyflux.pfrandom import random
        return random() < decision_prob


class DeterministicVoting(VotingStrategy):
    """Threshold voting: prob >= 0.5."""

    def decide(self, decision_prob: float, context: VotingContext) -> bool:
        return decision_prob >= 0.5


class SoftVoting(VotingStrategy):
    """Return probability itself (for ensemble aggregation)."""

    def decide(self, decision_prob: float, context: VotingContext) -> float:
        return decision_prob
