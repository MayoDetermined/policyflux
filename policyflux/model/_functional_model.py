"""Functional-style Model API (TensorFlow-style).

The functional API lets you compose a simulation model by:

1. Creating an :class:`~policyflux.model._node.Input` node that declares
   ``policy_dim`` and ``num_actors``.
2. Calling layer spec instances on nodes to build a directed chain.
3. Wrapping the chain with ``Model(inputs=…, outputs=…)``.

Under the hood, :class:`Model` traces the linked list of
:class:`~policyflux.model._node._SymbolicNode` objects from *outputs*
back to *inputs*, extracting the ordered sequence of layer specs.  The
resulting layer list is then compiled to an
:class:`~policyflux.integration.LayerConfig` in exactly the same way as
:class:`~policyflux.model.Sequential`.

Example
-------
::

    from policyflux.model import Model, Input, layers as L

    # 1. Declare input space
    bill = Input(policy_dim=4, num_actors=100)

    # 2. Chain layers by calling spec instances on nodes
    x = L.IdealPoint()(bill)
    x = L.PublicOpinion(support=0.6)(x)
    x = L.PartyDiscipline(strength=0.5, line_support=0.55)(x)

    # 3. Build model
    model = Model(inputs=bill, outputs=x)
    model.compile(executive='parliamentary', pm_strength=0.65)

    results = model.run(iterations=300, seed=42)
    model.summary()
"""

from __future__ import annotations

from typing import Any

from ._base import _ModelBase
from ._node import Input, _SymbolicNode
from ._sequential import _layer_config_from_specs
from .layers import _LayerSpec


class Model(_ModelBase):
    """Functional-API model: graph of layer specs from ``Input`` to output.

    Parameters
    ----------
    inputs:
        An :class:`~policyflux.model._node.Input` node that declares the
        policy dimensionality and actor count for the simulation.
    outputs:
        The terminal :class:`~policyflux.model._node._SymbolicNode`
        produced by calling one or more layer specs on *inputs*.

    Raises
    ------
    TypeError
        If *inputs* is not an :class:`~policyflux.model._node.Input`
        instance.
    ValueError
        If the node chain from *outputs* never reaches *inputs* (i.e. the
        graph is disconnected).

    Notes
    -----
    ``Model(inputs=x, outputs=x)`` with the same node for both creates a
    model with no layers (equivalent to an empty :class:`Sequential`).
    """

    def __init__(self, inputs: Input, outputs: _SymbolicNode) -> None:
        if not isinstance(inputs, Input):
            raise TypeError(
                f"'inputs' must be an Input node, got {type(inputs).__name__!r}."
            )
        layer_specs = self._trace(inputs, outputs)
        super().__init__(
            num_actors=inputs.num_actors,
            policy_dim=inputs.policy_dim,
        )
        self._layer_specs = layer_specs
        # Keep references for repr / inspection
        self._inputs_node = inputs
        self._outputs_node = outputs

    # ------------------------------------------------------------------
    # Graph tracing
    # ------------------------------------------------------------------

    @staticmethod
    def _trace(inputs: Input, outputs: _SymbolicNode) -> list[_LayerSpec]:
        """Walk the node chain from *outputs* back to *inputs*.

        Returns the layer specs in *forward* order (i.e. the order they
        were applied by the user).

        Raises
        ------
        ValueError
            If the chain reaches a ``None`` inbound pointer without
            encountering *inputs*.
        """
        if outputs is inputs:
            # No layers between input and output
            return []

        specs: list[_LayerSpec] = []
        node: _SymbolicNode | None = outputs

        while node is not inputs:
            if node is None:
                raise ValueError(
                    "The 'outputs' node is not connected to the 'inputs' node. "
                    "Make sure every layer in your graph is called on the "
                    "result of the previous layer (or on 'inputs' itself)."
                )
            if node._layer_spec is not None:
                specs.append(node._layer_spec)
            node = node._inbound

        # specs are in reverse order (from output → input); flip them
        return list(reversed(specs))

    # ------------------------------------------------------------------
    # Internal: build LayerConfig
    # ------------------------------------------------------------------

    def _build_layer_config(self) -> Any:
        return _layer_config_from_specs(self._layer_specs)

    # ------------------------------------------------------------------
    # summary() override
    # ------------------------------------------------------------------

    def _print_summary_body(self) -> None:
        """Print body including traced layer chain."""
        super()._print_summary_body()
        print(f"  Graph layers ({len(self._layer_specs)}):")
        if not self._layer_specs:
            print("    (none - using defaults)")
        for i, spec in enumerate(self._layer_specs):
            print(f"    [{i}] {spec!r}")

    # ------------------------------------------------------------------
    # get_config / from_config
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        """Return serialisable configuration including the traced layer list."""
        config = super().get_config()
        config["layers"] = [
            {
                "class": spec.__class__.__name__,
                **{k: v for k, v in vars(spec).items() if not k.startswith("_")},
            }
            for spec in self._layer_specs
        ]
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Any:
        """Reconstruct from a config dict as a :class:`Sequential` model.

        Because the functional graph topology is captured by the layer
        sequence, the reconstructed model is a :class:`Sequential` (which
        is semantically equivalent for linear graphs).

        Parameters
        ----------
        config:
            Dictionary as returned by :meth:`get_config`.

        Returns
        -------
        Sequential
        """
        from ._sequential import Sequential

        return Sequential.from_config(config)

    def __repr__(self) -> str:
        return (
            f"Model("
            f"inputs={self._inputs_node!r}, "
            f"layers={len(self._layer_specs)}, "
            f"policy_dim={self._policy_dim}, "
            f"num_actors={self._num_actors})"
        )
