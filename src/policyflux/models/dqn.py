"""TensorFlow DQN for congressional voting decisions.

Reimplements the previous torch-based agent using tf.keras primitives in eager mode.
The API surface (VoteDQN, ReplayBuffer, DQNAgent) remains similar for callers.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple
import random

import numpy as np
import tensorflow as tf

from policyflux.defaults import SIMULATION, MODELS, PATHS
from policyflux.gpu_utils import get_tf_device


class VoteDQN(tf.keras.Model):
    """Feedforward network mapping state vectors to Q-values."""

    def __init__(self, state_dim: int, hidden_dims: Optional[List[int]] = None, output_dim: int = 2) -> None:
        super().__init__()
        hidden_dims = hidden_dims or defaults.DQN_HIDDEN_DIMS
        layers: List[tf.keras.layers.Layer] = []
        for dim in hidden_dims:
            layers.append(tf.keras.layers.Dense(dim, activation="relu"))
        layers.append(tf.keras.layers.Dense(output_dim, activation=None))
        self.net = tf.keras.Sequential(layers)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:  # type: ignore[override]
        return self.net(inputs, training=training)

    def select_action(self, state: tf.Tensor, epsilon: float = 0.0, device: Optional[str] = None) -> int:
        device = device or get_tf_device()
        with tf.device(device):
            if random.random() < epsilon:
                return random.randint(0, int(self.net.layers[-1].units) - 1)
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            if len(state_tensor.shape) == 1:
                state_tensor = tf.expand_dims(state_tensor, axis=0)
            q_values = self(state_tensor, training=False)
            return int(tf.argmax(q_values, axis=1).numpy()[0])


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((np.array(state, dtype=np.float32), int(action), float(reward), np.array(next_state, dtype=np.float32), bool(done)))

    def sample(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        states_t = tf.convert_to_tensor(np.stack(states), dtype=tf.float32)
        actions_t = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_t = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_t = tf.convert_to_tensor(np.stack(next_states), dtype=tf.float32)
        dones_t = tf.convert_to_tensor(dones, dtype=tf.float32)
        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """TensorFlow DQN agent with target network and replay buffer."""

    def __init__(
        self,
        state_dim: int = defaults.DQN_STATE_DIM,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = defaults.DQN_LEARNING_RATE,
        gamma: float = defaults.DQN_GAMMA,
        epsilon: float = defaults.DQN_EPSILON_START,
        epsilon_decay: float = defaults.DQN_EPSILON_DECAY,
        epsilon_min: float = defaults.DQN_EPSILON_MIN,
        target_update_freq: int = defaults.DQN_TARGET_UPDATE_FREQ,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or get_tf_device()
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.step_count = 0

        self.policy_net = VoteDQN(state_dim, hidden_dims)
        self.target_net = VoteDQN(state_dim, hidden_dims)
        self._sync_target()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.Huber()
        self.replay_buffer = ReplayBuffer(capacity=10000)

    def _sync_target(self) -> None:
        dummy = tf.zeros((1, self.state_dim), dtype=tf.float32)
        self.policy_net(dummy)
        self.target_net(dummy)
        for target_w, policy_w in zip(self.target_net.weights, self.policy_net.weights):
            target_w.assign(policy_w)

    def select_action(self, state: tf.Tensor, use_exploration: bool = True) -> int:
        eps = self.epsilon if use_exploration else 0.0
        return self.policy_net.select_action(state, epsilon=eps, device=self.device)

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        with tf.device(self.device):
            with tf.GradientTape() as tape:
                q_values = self.policy_net(states, training=True)
                action_indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
                q_action = tf.gather_nd(q_values, action_indices)

                next_q_values = self.target_net(next_states, training=False)
                max_next_q = tf.reduce_max(next_q_values, axis=1)
                target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

                loss = self.loss_fn(target_q, q_action)

            grads = tape.gradient(loss, self.policy_net.trainable_variables)
            if grads:
                grads, _ = tf.clip_by_global_norm(grads, defaults.DQN_GRAD_CLIP)
                self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self._sync_target()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.numpy())




