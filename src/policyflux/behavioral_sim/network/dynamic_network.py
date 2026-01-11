from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import tensorflow as tf

from policyflux.behavioral_sim.network.influence import InfluenceFunction


@dataclass
class DynamicNetwork:
    """Applies a chain of influence functions to produce G(t)."""

    base_adj: tf.Tensor
    influence_functions: List[InfluenceFunction] = field(default_factory=list)
    device: str = "/CPU:0"

    def compute(self, X: tf.Tensor, Z: Optional[tf.Tensor] = None) -> tf.Tensor:
        with tf.device(self.device):
            G = tf.identity(self.base_adj)
            features = tf.identity(X)
            context = tf.identity(Z) if Z is not None else None
            for fn in self.influence_functions:
                G = fn(G, features, context)
            G = tf.linalg.set_diag(G, tf.zeros_like(tf.linalg.diag_part(G)))
        return G




