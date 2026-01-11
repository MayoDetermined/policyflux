from __future__ import annotations

from typing import Any, Protocol, Optional

import tensorflow as tf


class InfluenceFunction(Protocol):
    """Protocol for influence mixers f(A_base, X, Z) -> G."""

    def __call__(self, A_base: tf.Tensor, X: tf.Tensor, Z: Optional[tf.Tensor] = None, **kwargs: Any) -> tf.Tensor:
        ...


class HomophilyInfluence:
    def __init__(self, beta: float = 2.0) -> None:
        self.beta = float(beta)

    def __call__(self, A_base: tf.Tensor, X: tf.Tensor, Z: Optional[tf.Tensor] = None, **kwargs: Any) -> tf.Tensor:
        # X: [n, d]
        diff = tf.norm(tf.expand_dims(X, 1) - tf.expand_dims(X, 0), ord=1, axis=-1)
        weight = tf.exp(-self.beta * diff)
        return A_base * weight


class LeaderBoostInfluence:
    def __init__(self, boost: float = 2.0, leader_scores: Optional[tf.Tensor] = None) -> None:
        self.boost = float(boost)
        self.leader_scores = leader_scores

    def __call__(self, A_base: tf.Tensor, X: tf.Tensor, Z: Optional[tf.Tensor] = None, **kwargs: Any) -> tf.Tensor:
        n = A_base.shape[0]
        if self.leader_scores is not None:
            scores = self.leader_scores
        else:
            scores = tf.reduce_sum(tf.abs(A_base), axis=1)
            max_val = tf.reduce_max(scores)
            if tf.greater(max_val, 0):
                scores = scores / max_val
        multipliers = 1.0 + self.boost * scores
        return tf.transpose(tf.transpose(A_base) * multipliers)


class CommitteeInfluence:
    def __init__(self, committee_matrix: Optional[tf.Tensor] = None, weight: float = 0.35) -> None:
        self.committee_matrix = committee_matrix
        self.weight = float(weight)

    def __call__(self, A_base: tf.Tensor, X: tf.Tensor, Z: Optional[tf.Tensor] = None, **kwargs: Any) -> tf.Tensor:
        if self.committee_matrix is None:
            return A_base
        if self.committee_matrix.shape != A_base.shape:
            return A_base
        return A_base + self.weight * self.committee_matrix




