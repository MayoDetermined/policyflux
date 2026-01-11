"""TensorFlow autoencoders for voting pattern compression."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import tensorflow as tf


class VoteAutoencoder(tf.keras.Model):
    """Autoencoder with BCE/MSE helpers for reconstruction metrics."""

    def __init__(self, input_dim: int, latent_dim: int = 8) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(latent_dim, activation=None),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(input_dim, activation="tanh"),
            ]
        )

        self.reconstruction_errors: List[float] = []
        self.bce_losses: List[float] = []
        self.mse_losses: List[float] = []

    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:  # type: ignore[override]
        z = self.encoder(inputs, training=training)
        out = self.decoder(z, training=training)
        return out, z

    def compute_bce_loss(self, original: tf.Tensor, reconstructed: tf.Tensor) -> tf.Tensor:
        reconstructed_prob = (reconstructed + 1.0) / 2.0
        return tf.keras.losses.binary_crossentropy(original, reconstructed_prob)

    def compute_mse_loss(self, original: tf.Tensor, reconstructed: tf.Tensor) -> tf.Tensor:
        return tf.keras.losses.mean_squared_error(original, reconstructed)




