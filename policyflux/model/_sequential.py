"""Sequential model - layer-by-layer composition API.

:class:`Sequential` is the simplest way to build a PolicyFlux simulation
model.  Add layer specs one at a time with :meth:`add` or the ``|``
pipe operator, call :meth:`compile` to choose an executive system, then
:meth:`run` to execute the Monte Carlo simulation.

Example
-------
::

    from policyflux.model import Sequential, layers as L

    model = (
        Sequential(num_actors=100, policy_dim=4)
        | L.IdealPoint()
        | L.PublicOpinion(support=0.6)
        | L.PartyDiscipline(strength=0.5, line_support=0.55)
    )
    model.compile(executive='parliamentary', pm_strength=0.65)
    results = model.run(iterations=300, seed=42)
    model.summary()

    # get_config / from_config round-trip
    cfg = model.get_config()
    model2 = Sequential.from_config(cfg)
"""

from __future__ import annotations

from typing import Any

from ._base import _ModelBase
from .layers import _LAYER_CLASS_MAP, _LayerSpec

# Shared helper to build a LayerConfig with only specified layers enabled.
_ALL_OFF: dict[str, bool] = {
    "include_ideal_point": False,
    "include_public_opinion": False,
    "include_lobbying": False,
    "include_media_pressure": False,
    "include_party_discipline": False,
    "include_government_agenda": False,
    "include_neural": False,
}


def _layer_config_from_specs(specs: list[_LayerSpec]) -> Any:
    """Build a :class:`~policyflux.integration.LayerConfig` from *specs*.

    Starts with all layers disabled and enables only the ones declared in
    *specs*.  If *specs* is empty, returns the default ``LayerConfig()``
    (all main layers enabled - matches existing behaviour).
    """
    from ..integration.config import LayerConfig

    if not specs:
        return LayerConfig()

    params: dict[str, Any] = dict(_ALL_OFF)
    for spec in specs:
        params.update(spec._to_layer_config())
    return LayerConfig(**params)


class Sequential(_ModelBase):
    """Linear stack of layer specs.

    Parameters
    ----------
    num_actors:
        Number of legislators in the simulation.
    policy_dim:
        Dimensionality of the policy space.

    Notes
    -----
    When no layers are added before :meth:`run` is called,
    the default :class:`~policyflux.integration.LayerConfig` is used
    (ideal point, public opinion, lobbying, media, and party discipline
    all enabled with their default parameters).
    """

    def __init__(self, num_actors: int = 100, policy_dim: int = 4) -> None:
        super().__init__(num_actors=num_actors, policy_dim=policy_dim)
        self._layer_specs: list[_LayerSpec] = []

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def add(self, layer_spec: _LayerSpec) -> Sequential:
        """Append a layer spec to the stack.

        Parameters
        ----------
        layer_spec:
            Any instance of a class from :mod:`policyflux.model.layers`.

        Returns
        -------
        Sequential
            ``self`` for method chaining.
        """
        if not isinstance(layer_spec, _LayerSpec):
            raise TypeError(
                f"Expected a _LayerSpec instance, got {type(layer_spec).__name__!r}. "
                "Use one of the classes from policyflux.model.layers."
            )
        self._layer_specs.append(layer_spec)
        return self

    def pop(self) -> _LayerSpec | None:
        """Remove and return the last layer spec, or ``None`` if empty."""
        return self._layer_specs.pop() if self._layer_specs else None

    def __or__(self, layer_spec: _LayerSpec) -> Sequential:
        """Pipe operator: ``model | L.IdealPoint()`` appends the layer.

        Returns
        -------
        Sequential
            ``self`` for further chaining.
        """
        return self.add(layer_spec)

    # ------------------------------------------------------------------
    # Internal: build LayerConfig
    # ------------------------------------------------------------------

    def _build_layer_config(self) -> Any:
        return _layer_config_from_specs(self._layer_specs)

    # ------------------------------------------------------------------
    # summary() override
    # ------------------------------------------------------------------

    def _print_summary_body(self) -> None:
        """Override to include layer list in the printed summary."""
        super()._print_summary_body()
        print(f"  Layers ({len(self._layer_specs)}):")
        if not self._layer_specs:
            print("    (none - using defaults)")
        for i, spec in enumerate(self._layer_specs):
            print(f"    [{i}] {spec!r}")

    # ------------------------------------------------------------------
    # get_config / from_config
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        """Return serialisable configuration including the layer list.

        Each layer is stored as a dict with a ``'class'`` key (the class
        name) and any constructor parameters the spec has.
        """
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
    def from_config(cls, config: dict[str, Any]) -> Sequential:
        """Reconstruct a :class:`Sequential` model from a config dict.

        Parameters
        ----------
        config:
            Dictionary as returned by :meth:`get_config`.

        Returns
        -------
        Sequential
            Fully compiled model with the same layers and settings.

        Raises
        ------
        ValueError
            If a layer class name in *config* is not recognised.
        """
        obj = cls(
            num_actors=config.get("num_actors", 100),
            policy_dim=config.get("policy_dim", 4),
        )
        for layer_dict in config.get("layers", []):
            layer_dict = dict(layer_dict)  # copy so we can pop
            class_name = layer_dict.pop("class", None)
            if class_name is None:
                raise ValueError("Layer dict missing 'class' key.")
            if class_name not in _LAYER_CLASS_MAP:
                known = ", ".join(sorted(_LAYER_CLASS_MAP))
<<<<<<< HEAD
                raise ValueError(
                    f"Unknown layer class {class_name!r}. Known: {known}"
                )
=======
                raise ValueError(f"Unknown layer class {class_name!r}. Known: {known}")
>>>>>>> 28724a8eb17f6081daef9177c037673d899cf2a9
            layer_cls = _LAYER_CLASS_MAP[class_name]
            obj.add(layer_cls(**layer_dict))

        obj.compile(
            executive=config.get("executive", "presidential"),
            aggregation=config.get("aggregation", "sequential"),
            approval=config.get("exec_approval"),
            veto_override=config.get("exec_veto_override"),
            pm_strength=config.get("exec_pm_strength"),
            confidence_threshold=config.get("exec_confidence_threshold"),
            government_bill_rate=config.get("exec_government_bill_rate"),
            n_lobbyists=config.get("sa_n_lobbyists", 0),
            lobbyist_strength=config.get("sa_lobbyist_strength", 0.5),
            lobbyist_stance=config.get("sa_lobbyist_stance", 1.0),
            n_whips=config.get("sa_n_whips", 0),
            whip_strength=config.get("sa_whip_strength", 0.5),
            whip_line_support=config.get("sa_whip_line_support", 0.5),
        )
        return obj

    def __repr__(self) -> str:
        return (
            f"Sequential("
            f"num_actors={self._num_actors}, "
            f"policy_dim={self._policy_dim}, "
            f"layers={len(self._layer_specs)})"
        )
