"""TensorFlow LSTM model for congressional actor ideology memory."""

from __future__ import annotations

from typing import Optional, Tuple
import tensorflow as tf

from policyflux.gpu_utils import get_tf_device


class ActorLSTM(tf.keras.Model):
    """LSTM-based ideology memory model for a single actor."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = tf.keras.layers.LSTM(
            hidden_dim,
            dropout=dropout if num_layers > 1 else 0.0,
            return_sequences=True,
            return_state=False,
        )
        self.proj = tf.keras.layers.Dense(output_dim, activation="tanh")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:  # type: ignore[override]
        lstm_out = self.lstm(inputs, training=training)
        last_out = lstm_out[:, -1, :]
        return self.proj(last_out, training=training)

    def init_hidden(self, batch_size: int, device: Optional[str] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        device = device or get_tf_device()
        with tf.device(device):
            h = tf.zeros((self.num_layers, batch_size, self.hidden_dim), dtype=tf.float32)
            c = tf.zeros((self.num_layers, batch_size, self.hidden_dim), dtype=tf.float32)
            return h, c


class ActorLSTMTrainer:
    """Trainer wrapper for ActorLSTM using eager TensorFlow."""

    def __init__(self, model: ActorLSTM, learning_rate: float = 1e-4, device: Optional[str] = None) -> None:
        self.model = model
        self.device = device or get_tf_device()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.criterion = tf.keras.losses.MeanSquaredError()

    def train_step(self, history_data: tf.Tensor, target_ideology: tf.Tensor) -> float:
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                predicted = self.model(history_data, training=True)
                loss = self.criterion(target_ideology, predicted)
            grads = tape.gradient(loss, self.model.trainable_variables)
            if grads:
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return float(loss.numpy())

    def predict(self, history_data: tf.Tensor) -> tf.Tensor:
        with tf.device(self.device):
            return self.model(history_data, training=False)




