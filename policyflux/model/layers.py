"""Layer specification descriptors for the PolicyFlux Model API.

Each class is a lightweight *descriptor* - it carries the parameters for
one type of voting layer and knows how to translate itself into the
``LayerConfig`` kwargs used by the integration builders.

Instances are callable: when called on a :class:`~policyflux.model._node._SymbolicNode`
they return a new node, threading the computation graph needed by the
:class:`~policyflux.model.Model` functional API.

Example
-------
::

    from policyflux.model import layers as L

    spec = L.PublicOpinion(support=0.7)

    # Sequential use: pass to model.add()
    model.add(spec)

    # Functional use: call on a node
    bill = Input(policy_dim=4, num_actors=100)
    x = L.IdealPoint()(bill)
    x = L.PublicOpinion(support=0.7)(x)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._node import _SymbolicNode


class _LayerSpec:
    """Abstract base for all layer specifications.

    Subclasses must implement :meth:`_to_layer_config`, which returns a
    partial dict of :class:`~policyflux.integration.LayerConfig` keyword
    arguments.
    """

    def _to_layer_config(self) -> dict[str, Any]:
        """Return partial ``LayerConfig`` kwargs that enable this layer."""
        raise NotImplementedError

    def __call__(self, node: _SymbolicNode) -> _SymbolicNode:
        """Functional API: apply this layer spec to a symbolic node.

        Returns a new :class:`~policyflux.model._node._SymbolicNode` that
        records *self* as the layer and *node* as its predecessor.
        """
        from ._node import _SymbolicNode

        return _SymbolicNode(inbound=node, layer_spec=self)

    def __repr__(self) -> str:
        params = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"


class IdealPoint(_LayerSpec):
    """Ideal-point voting layer.

    The core layer: each legislator votes based on the Euclidean distance
    between their ideal policy position and the bill's position.
    No parameters - the ideal points are drawn at random per legislator.
    """

    def _to_layer_config(self) -> dict[str, Any]:
        return {"include_ideal_point": True}


class PublicOpinion(_LayerSpec):
    """Public-opinion pressure layer.

    Parameters
    ----------
    support:
        Public support level for the bill, ``[0, 1]``.
        Values above 0.5 push legislators toward yes; below 0.5 toward no.
    """

    def __init__(self, support: float = 0.5) -> None:
        self.support = support

    def _to_layer_config(self) -> dict[str, Any]:
        return {"include_public_opinion": True, "public_support": self.support}


class Lobbying(_LayerSpec):
    """Lobbying-influence layer.

    Parameters
    ----------
    intensity:
        Lobbying intensity, ``[0, 1]``.  Higher values give external
        interests more weight in each legislator's voting decision.
    """

    def __init__(self, intensity: float = 0.5) -> None:
        self.intensity = intensity

    def _to_layer_config(self) -> dict[str, Any]:
        return {"include_lobbying": True, "lobbying_intensity": self.intensity}


class MediaPressure(_LayerSpec):
    """Media-pressure layer.

    Parameters
    ----------
    pressure:
        Intensity of media pressure, ``[0, 1]``.  Higher values amplify
        the media's framing of the bill in legislators' decisions.
    """

    def __init__(self, pressure: float = 0.5) -> None:
        self.pressure = pressure

    def _to_layer_config(self) -> dict[str, Any]:
        return {"include_media_pressure": True, "media_pressure": self.pressure}


class PartyDiscipline(_LayerSpec):
    """Party-whip discipline layer.

    Parameters
    ----------
    strength:
        Whip's enforcement strength, ``[0, 1]``.
    line_support:
        The party's official position on the bill, ``[0, 1]``.
        Values above 0.5 mean the party supports the bill;
        below 0.5 means opposition.
    """

    def __init__(self, strength: float = 0.5, line_support: float = 0.5) -> None:
        self.strength = strength
        self.line_support = line_support

    def _to_layer_config(self) -> dict[str, Any]:
        return {
            "include_party_discipline": True,
            "party_discipline_strength": self.strength,
            "party_line_support": self.line_support,
        }


class GovernmentAgenda(_LayerSpec):
    """Government-agenda layer (parliamentary PM / cabinet influence).

    Parameters
    ----------
    pm_strength:
        Prime minister's effective influence on the legislative agenda,
        ``[0, 1]``.
    """

    def __init__(self, pm_strength: float = 0.6) -> None:
        self.pm_strength = pm_strength

    def _to_layer_config(self) -> dict[str, Any]:
        return {
            "include_government_agenda": True,
            "government_agenda_pm_strength": self.pm_strength,
        }


#: All built-in layer spec classes, in registration order.
ALL: tuple[type[_LayerSpec], ...] = (
    IdealPoint,
    PublicOpinion,
    Lobbying,
    MediaPressure,
    PartyDiscipline,
    GovernmentAgenda,
)

#: Map from class name string → class (for ``from_config`` reconstruction).
_LAYER_CLASS_MAP: dict[str, type[_LayerSpec]] = {cls.__name__: cls for cls in ALL}
