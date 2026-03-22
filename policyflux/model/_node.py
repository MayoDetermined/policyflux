"""Symbolic computation-graph nodes for the PolicyFlux functional API.

The functional API mirrors TensorFlow's ``keras.Input`` + ``keras.Model``
pattern.  Users build a *computation graph* by:

1. Creating a root :class:`Input` node that fixes ``policy_dim`` and
   ``num_actors``.
2. Calling layer spec instances on that node to create successor nodes.
3. Passing the resulting chain to :class:`~policyflux.model.Model`.

Each node records the layer spec that created it and a reference to its
parent node.  :class:`~policyflux.model._functional_model.Model` traces
this linked list from *outputs* back to *inputs* to reconstruct the
ordered list of layer specs.

Example
-------
::

    from policyflux.model import Input, Model, layers as L

    bill = Input(policy_dim=4, num_actors=100)
    x    = L.IdealPoint()(bill)
    x    = L.PublicOpinion(support=0.6)(x)
    x    = L.PartyDiscipline(strength=0.5)(x)

    model = Model(inputs=bill, outputs=x)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .layers import _LayerSpec


class _SymbolicNode:
    """A node in the PolicyFlux functional computation graph.

    Attributes
    ----------
    _inbound : _SymbolicNode | None
        The predecessor node in the graph, or ``None`` for an
        :class:`Input` root node.
    _layer_spec : _LayerSpec | None
        The layer spec that produced this node, or ``None`` for
        an :class:`Input` root node.
    """

    __slots__ = ("_inbound", "_layer_spec")

    def __init__(
        self,
        inbound: _SymbolicNode | None = None,
        layer_spec: _LayerSpec | None = None,
    ) -> None:
        self._inbound = inbound
        self._layer_spec = layer_spec

    def __repr__(self) -> str:
        spec = self._layer_spec
        spec_name = spec.__class__.__name__ if spec is not None else "Input"
        return f"<{self.__class__.__name__} via {spec_name}>"


class Input(_SymbolicNode):
    """Root node of a functional computation graph.

    Declares the dimensionality of the policy space and the number of
    legislators.  Pass an :class:`Input` instance (and the final output
    node) to :class:`~policyflux.model.Model` to create a compiled model.

    Parameters
    ----------
    policy_dim:
        Number of policy dimensions (≥ 1).
    num_actors:
        Number of legislators in the simulation (≥ 1).

    Example
    -------
    ::

        bill = Input(policy_dim=4, num_actors=100)
        x    = layers.IdealPoint()(bill)
        x    = layers.PublicOpinion(support=0.6)(x)
        model = Model(inputs=bill, outputs=x)
    """

    __slots__ = ("num_actors", "policy_dim")

    def __init__(self, policy_dim: int = 4, num_actors: int = 100) -> None:
        super().__init__(inbound=None, layer_spec=None)
        if policy_dim < 1:
            raise ValueError(f"policy_dim must be ≥ 1, got {policy_dim}")
        if num_actors < 1:
            raise ValueError(f"num_actors must be ≥ 1, got {num_actors}")
        self.policy_dim = policy_dim
        self.num_actors = num_actors

    def __repr__(self) -> str:
        return f"Input(policy_dim={self.policy_dim}, num_actors={self.num_actors})"
