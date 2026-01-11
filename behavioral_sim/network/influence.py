from __future__ import annotations

from typing import Any, Protocol, Optional

import torch


class InfluenceFunction(Protocol):
    """Protocol for influence mixers f(A_base, X, Z) -> G."""

    def __call__(self, A_base: torch.Tensor, X: torch.Tensor, Z: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        ...


class HomophilyInfluence:
    def __init__(self, beta: float = 2.0) -> None:
        self.beta = float(beta)

    def __call__(self, A_base: torch.Tensor, X: torch.Tensor, Z: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        # X: [n, d]
        diff = torch.cdist(X, X, p=1)
        weight = torch.exp(-self.beta * diff)
        return A_base * weight


class LeaderBoostInfluence:
    def __init__(self, boost: float = 2.0, leader_scores: Optional[torch.Tensor] = None) -> None:
        self.boost = float(boost)
        self.leader_scores = leader_scores

    def __call__(self, A_base: torch.Tensor, X: torch.Tensor, Z: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        n = A_base.shape[0]
        if self.leader_scores is not None:
            scores = self.leader_scores
        else:
            scores = torch.sum(torch.abs(A_base), dim=1)
            if torch.max(scores) > 0:
                scores = scores / torch.max(scores)
        multipliers = 1.0 + self.boost * scores
        return (A_base.T * multipliers).T


class CommitteeInfluence:
    def __init__(self, committee_matrix: Optional[torch.Tensor] = None, weight: float = 0.35) -> None:
        self.committee_matrix = committee_matrix
        self.weight = float(weight)

    def __call__(self, A_base: torch.Tensor, X: torch.Tensor, Z: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        if self.committee_matrix is None:
            return A_base
        if self.committee_matrix.shape != A_base.shape:
            return A_base
        return A_base + self.weight * self.committee_matrix
