from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from policyflux.behavioral_sim.engine.compiler import CompiledSystem


class CongressRunner:
    """Executes Monte Carlo simulations using a compiled system."""

    def __init__(self, compiled: CompiledSystem) -> None:
        self.compiled = compiled
        self.device = compiled.device

    def _compute_features(self) -> tf.Tensor:
        feats = [tf.convert_to_tensor(a.ideology, dtype=tf.float32) for a in self.compiled.actors]
        with tf.device(self.device):
            return tf.stack(feats, axis=0)

    def _compute_context(self) -> tf.Tensor:
        vec = self.compiled.regime.get_context_vector()
        with tf.device(self.device):
            return tf.convert_to_tensor(vec, dtype=tf.float32)

    def _refresh_network(self) -> None:
        if self.compiled.dynamic_network is None or self.compiled.congress is None:
            return
        X = self._compute_features()
        Z = self._compute_context()
        G = self.compiled.dynamic_network.compute(X, Z)
        adj_np = tf.convert_to_tensor(G).numpy().astype(np.float32)
        self.compiled.congress.current_adj_matrix = adj_np
        self.compiled.congress.adj_matrix = adj_np
        self.compiled.congress.base_adj_matrix = adj_np
        if hasattr(self.compiled.engine, "template"):
            self.compiled.engine.template.current_adj_matrix = adj_np
            self.compiled.engine.template.base_adj_matrix = adj_np

    def run_monte_carlo(
        self,
        n_simulations: int = 100,
        steps: int = 10,
        use_gpu: bool = True,
        progress_callback: Optional[Any] = None,
        show_tqdm: bool = True,
        show_small_steps: bool = True,
        show_step_tqdm: bool = True,
        show_component_tqdm: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self.compiled.engine is None:
            raise RuntimeError("Compiled system has no engine; call compiler.compile() first.")

        self._refresh_network()
        return self.compiled.engine.run_monte_carlo(
            n_simulations=n_simulations,
            steps=steps,
            progress_callback=progress_callback,
            show_tqdm=show_tqdm,
            show_small_steps=show_small_steps,
            show_step_tqdm=show_step_tqdm,
            show_component_tqdm=show_component_tqdm,
        )




