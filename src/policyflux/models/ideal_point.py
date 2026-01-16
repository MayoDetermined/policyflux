"""TensorFlow multi-dimensional logistic ideal point model."""

from __future__ import annotations

import tensorflow as tf


class IdealPointModel(tf.keras.Model):
    """Simplified ideal point model implemented with TensorFlow variables."""

    def __init__(self, n_legislators: int, n_votes: int, dim: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.x = tf.Variable(tf.random.normal((n_legislators, dim), stddev=0.1), name="x")
        self.a = tf.Variable(tf.ones((n_votes, dim)), name="a")
        self.b = tf.Variable(tf.zeros((n_votes,)), name="b")

    def call(self, leg_ids: tf.Tensor, vote_ids: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        xi = tf.gather(self.x, leg_ids)
        a = tf.gather(self.a, vote_ids)
        b = tf.gather(self.b, vote_ids)

        if self.dim > 1:
            b_expanded = tf.expand_dims(b, axis=-1)
            distance = tf.norm(xi - b_expanded, axis=1)
        else:
            distance = tf.abs(tf.squeeze(xi, axis=-1) - b)

        salience = tf.norm(a, axis=1) if self.dim > 1 else tf.squeeze(a, axis=-1)
        return tf.math.sigmoid(salience * distance)




