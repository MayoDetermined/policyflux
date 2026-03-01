"""Shared base class for all PolicyFlux model types.

:class:`_ModelBase` provides:

- :meth:`compile` - configure executive system, aggregation, and special actors.
- :meth:`run` - build an engine from the current config and execute it.
- :meth:`summary` - print architecture + last-run statistics.
- :meth:`get_config` / :meth:`from_config` - serialisation round-trip.
- ``__call__`` - alias for :meth:`run`.

Subclasses (:class:`~policyflux.model.Sequential`,
:class:`~policyflux.model.Model`) must implement
:meth:`_build_layer_config` to translate their internal layer list into a
:class:`~policyflux.integration.LayerConfig`.
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Executive-type string aliases
# ---------------------------------------------------------------------------

_EXECUTIVE_ALIASES: dict[str, str] = {
    # Presidential
    "presidential": "presidential",
    "president": "presidential",
    "congress": "presidential",
    "us": "presidential",
    # Parliamentary
    "parliamentary": "parliamentary",
    "parliament": "parliamentary",
    "westminster": "parliamentary",
    "uk": "parliamentary",
    "canada": "parliamentary",
    # Semi-presidential
    "semi_presidential": "semi_presidential",
    "semi-presidential": "semi_presidential",
    "semipresidential": "semi_presidential",
    "france": "semi_presidential",
    "cohabitation": "semi_presidential",
}

_AGGREGATION_ALIASES: dict[str, str] = {
    "sequential": "sequential",
    "average": "average",
    "avg": "average",
    "weighted": "weighted",
    "multiplicative": "multiplicative",
    "mult": "multiplicative",
}


class _ModelBase:
    """Shared base for :class:`~policyflux.model.Sequential` and
    :class:`~policyflux.model.Model`.

    Parameters
    ----------
    num_actors:
        Number of legislators.
    policy_dim:
        Dimensionality of the policy space.
    """

    def __init__(self, num_actors: int = 100, policy_dim: int = 4) -> None:
        self._num_actors = num_actors
        self._policy_dim = policy_dim

        # Set by compile()
        self._executive: str = "presidential"
        self._aggregation: str = "sequential"
        self._executive_kwargs: dict[str, Any] = {}
        self._special_actor_kwargs: dict[str, Any] = {
            "n_lobbyists": 0,
            "lobbyist_strength": 0.5,
            "lobbyist_stance": 1.0,
            "n_whips": 0,
            "whip_strength": 0.5,
            "whip_line_support": 0.5,
        }
        self._is_compiled: bool = False

        # Set by run()
        self._last_results: list[int] | None = None
        self._last_iterations: int | None = None
        self._last_seed: int | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_actors(self) -> int:
        """Number of legislators."""
        return self._num_actors

    @property
    def policy_dim(self) -> int:
        """Dimensionality of the policy space."""
        return self._policy_dim

    @property
    def is_compiled(self) -> bool:
        """``True`` if :meth:`compile` has been called at least once."""
        return self._is_compiled

    # ------------------------------------------------------------------
    # compile()
    # ------------------------------------------------------------------

    def compile(
        self,
        executive: str = "presidential",
        aggregation: str = "sequential",
        *,
        # Presidential / semi-presidential
        approval: float | None = None,
        veto_override: float | None = None,
        # Parliamentary / semi-presidential
        pm_strength: float | None = None,
        confidence_threshold: float | None = None,
        government_bill_rate: float | None = None,
        # Special actors
        n_lobbyists: int = 0,
        lobbyist_strength: float = 0.5,
        lobbyist_stance: float = 1.0,
        n_whips: int = 0,
        whip_strength: float = 0.5,
        whip_line_support: float = 0.5,
    ) -> _ModelBase:
        """Configure the model before running.

        Parameters
        ----------
        executive:
            Executive system type.  Accepts ``'presidential'``,
            ``'parliamentary'``, or ``'semi_presidential'`` and a set of
            readable aliases (``'westminster'``, ``'france'``, ``'us'``, …).
        aggregation:
            How each voter aggregates its layer outputs.  One of
            ``'sequential'``, ``'average'``, ``'weighted'``,
            ``'multiplicative'`` (or short aliases ``'avg'``, ``'mult'``).
        approval:
            Presidential (or semi-presidential president's) approval
            rating ``[0, 1]``.
        veto_override:
            Supermajority share required to override a presidential veto
            (default ``2/3``).
        pm_strength:
            Prime minister's party strength ``[0, 1]`` - used for
            parliamentary and semi-presidential systems.
        confidence_threshold:
            Confidence-vote threshold (parliamentary systems).
        government_bill_rate:
            Fraction of bills treated as government bills (parliamentary).
        n_lobbyists:
            Number of explicit lobbyist special actors.
        lobbyist_strength:
            Base lobbying strength ``[0, 1]``.
        lobbyist_stance:
            Lobbyist's position on the bill (1.0 = pro-bill).
        n_whips:
            Number of party-whip special actors.
        whip_strength:
            Whip discipline-enforcement strength ``[0, 1]``.
        whip_line_support:
            Whip's party-line position on the bill ``[0, 1]``.

        Returns
        -------
        _ModelBase
            ``self`` - enables method chaining, e.g.
            ``model.compile('parliamentary').run()``.
        """
        # Validate executive
        key = executive.lower().replace("-", "_")
        if key not in _EXECUTIVE_ALIASES:
            available = ", ".join(sorted(set(_EXECUTIVE_ALIASES.values())))
            raise ValueError(f"Unknown executive type {executive!r}. Available: {available}")
        self._executive = _EXECUTIVE_ALIASES[key]

        # Validate aggregation
        agg_key = aggregation.lower()
        if agg_key not in _AGGREGATION_ALIASES:
            available = ", ".join(sorted(set(_AGGREGATION_ALIASES.values())))
            raise ValueError(f"Unknown aggregation {aggregation!r}. Available: {available}")
        self._aggregation = _AGGREGATION_ALIASES[agg_key]

        # Store executive kwargs (only non-None values)
        self._executive_kwargs = {
            k: v
            for k, v in {
                "approval": approval,
                "veto_override": veto_override,
                "pm_strength": pm_strength,
                "confidence_threshold": confidence_threshold,
                "government_bill_rate": government_bill_rate,
            }.items()
            if v is not None
        }

        self._special_actor_kwargs = {
            "n_lobbyists": n_lobbyists,
            "lobbyist_strength": lobbyist_strength,
            "lobbyist_stance": lobbyist_stance,
            "n_whips": n_whips,
            "whip_strength": whip_strength,
            "whip_line_support": whip_line_support,
        }

        self._is_compiled = True
        return self

    # ------------------------------------------------------------------
    # run() / __call__
    # ------------------------------------------------------------------

    def run(self, iterations: int = 300, seed: int = 42) -> list[int]:
        """Execute the simulation.

        If :meth:`compile` has not been called, default settings are used
        automatically (presidential system, sequential aggregation).

        Parameters
        ----------
        iterations:
            Number of Monte Carlo iterations.
        seed:
            Random seed.

        Returns
        -------
        list[int]
            Number of votes cast in favour in each iteration.
        """
        if not self._is_compiled:
            self.compile()

        from ..integration.builders.engine_builder import build_engine

        config = self._to_integration_config(iterations=iterations, seed=seed)
        engine = build_engine(config)
        self._last_results = engine.run()
        self._last_iterations = iterations
        self._last_seed = seed
        return self._last_results

    def __call__(self, iterations: int = 300, seed: int = 42) -> list[int]:
        """Alias for :meth:`run`."""
        return self.run(iterations=iterations, seed=seed)

    # ------------------------------------------------------------------
    # Internal: build IntegrationConfig
    # ------------------------------------------------------------------

    def _to_integration_config(self, iterations: int, seed: int) -> Any:
        """Assemble an :class:`~policyflux.integration.IntegrationConfig`."""
        from ..core.abstract_executive import ExecutiveType
        from ..integration.config import AdvancedActorsConfig, IntegrationConfig

        exec_type_map = {
            "presidential": ExecutiveType.PRESIDENTIAL,
            "parliamentary": ExecutiveType.PARLIAMENTARY,
            "semi_presidential": ExecutiveType.SEMI_PRESIDENTIAL,
        }
        exec_type = exec_type_map[self._executive]

        kw = self._executive_kwargs
        sa = self._special_actor_kwargs

        actors_config = AdvancedActorsConfig(
            executive_type=exec_type,
            # Presidential
            president_approval_rating=kw.get("approval", 0.5),
            veto_override_threshold=kw.get("veto_override", 2 / 3),
            # Parliamentary
            pm_party_strength=kw.get("pm_strength", 0.55),
            confidence_threshold=kw.get("confidence_threshold", 0.5),
            government_bill_rate=kw.get("government_bill_rate", 0.7),
            # Semi-presidential
            semi_presidential_approval_rating=kw.get("approval", 0.5),
            semi_presidential_pm_party_strength=kw.get("pm_strength", 0.55),
            # Special actors
            n_lobbyists=sa.get("n_lobbyists", 0),
            lobbyist_strength=sa.get("lobbyist_strength", 0.5),
            lobbyist_stance=sa.get("lobbyist_stance", 1.0),
            n_whips=sa.get("n_whips", 0),
            whip_discipline_strength=sa.get("whip_strength", 0.5),
            whip_party_line_support=sa.get("whip_line_support", 0.5),
        )

        layer_config = self._build_layer_config()

        return IntegrationConfig(
            num_actors=self._num_actors,
            policy_dim=self._policy_dim,
            iterations=iterations,
            seed=seed,
            aggregation_strategy=self._aggregation,
            layer_config=layer_config,
            actors_config=actors_config,
        )

    def _build_layer_config(self) -> Any:
        """Return a configured :class:`~policyflux.integration.LayerConfig`.

        Subclasses override this to translate their layer list.
        The base implementation returns the default ``LayerConfig()``
        (all main layers enabled).
        """
        from ..integration.config import LayerConfig

        return LayerConfig()

    # ------------------------------------------------------------------
    # summary()
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a human-readable summary of the model."""
        width = 58
        print(f"Model: {self.__class__.__name__}")
        print("=" * width)
        self._print_summary_body()
        if self._last_results:
            n = len(self._last_results)
            avg = sum(self._last_results) / n
            threshold = self._num_actors / 2
            passage = sum(1 for v in self._last_results if v > threshold) / n
            variance = sum((v - avg) ** 2 for v in self._last_results) / n
            std = math.sqrt(variance)
            print("-" * width)
            print("Last run:")
            print(f"  Iterations:    {n}")
            print(f"  Avg votes for: {avg:.1f} / {self._num_actors}")
            print(f"  Vote share:    {avg / self._num_actors:.1%}")
            print(f"  Passage rate:  {passage:.1%}")
            print(f"  Std dev:       {std:.1f}")
        print("=" * width)

    def _print_summary_body(self) -> None:
        """Print model-specific summary lines (override in subclasses)."""
        print(f"  Actors:       {self._num_actors}")
        print(f"  Policy dim:   {self._policy_dim}")
        print(f"  Executive:    {self._executive}")
        for k, v in self._executive_kwargs.items():
            print(f"    {k}: {v}")
        print(f"  Aggregation:  {self._aggregation}")
        sa = self._special_actor_kwargs
        if sa.get("n_lobbyists", 0):
            print(
                f"  Lobbyists: {sa['n_lobbyists']}  "
                f"(strength={sa['lobbyist_strength']}, "
                f"stance={sa['lobbyist_stance']})"
            )
        if sa.get("n_whips", 0):
            print(
                f"  Whips:     {sa['n_whips']}  "
                f"(strength={sa['whip_strength']}, "
                f"line={sa['whip_line_support']})"
            )
        print(f"  Compiled:     {self._is_compiled}")

    # ------------------------------------------------------------------
    # get_config() / from_config()
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        """Return the model's configuration as a plain serialisable dict.

        The returned dict is suitable for JSON / YAML serialisation.  Pass
        it to :meth:`from_config` to reconstruct an equivalent model.

        Returns
        -------
        dict
            Flat configuration dictionary.
        """
        return {
            "class": self.__class__.__name__,
            "num_actors": self._num_actors,
            "policy_dim": self._policy_dim,
            "executive": self._executive,
            "aggregation": self._aggregation,
            **{f"exec_{k}": v for k, v in self._executive_kwargs.items()},
            **{f"sa_{k}": v for k, v in self._special_actor_kwargs.items()},
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _ModelBase:
        """Reconstruct a model from a config dictionary.

        Parameters
        ----------
        config:
            Dictionary as returned by :meth:`get_config`.

        Returns
        -------
        _ModelBase
            New compiled model instance of the appropriate subclass.
        """
        obj = cls(
            num_actors=config.get("num_actors", 100),
            policy_dim=config.get("policy_dim", 4),
        )
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
            f"{self.__class__.__name__}("
            f"num_actors={self._num_actors}, "
            f"policy_dim={self._policy_dim})"
        )
