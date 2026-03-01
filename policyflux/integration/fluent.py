"""Top-level fluent builder for PolicyFlux simulations.

Provides two styles of usage:

**Flat API** (method chaining)::

    from policyflux import PolicyFlux

    results = (
        PolicyFlux()
        .actors(50)
        .policy_dim(2)
        .iterations(100)
        .seed(42)
        .with_ideal_point()
        .with_public_opinion(support=0.6)
        .presidential(approval_rating=0.65)
        .build()
        .run()
    )

**Fluent Builder API** (section sub-builders)::

    from policyflux import PolicyFlux

    results = (
        PolicyFlux.builder()
        .actors(50)
        .policy_dim(2)
        .iterations(100)
        .seed(42)
        .layers()
            .ideal_point()
            .public_opinion(support=0.6)
            .party_discipline(line_support=0.58, strength=0.66)
            .done()
        .executive()
            .presidential(approval_rating=0.65)
            .done()
        .special_actors()
            .lobbyists(3, strength=0.8)
            .whips(2, discipline_strength=0.7)
            .done()
        .build()
        .run()
    )
"""

from __future__ import annotations

from typing import Any

from ..core.abstract_executive import ExecutiveType
from .config import AdvancedActorsConfig, IntegrationConfig, LayerConfig

# ======================================================================
# Section sub-builders
# ======================================================================


class LayerBuilder:
    """Scoped builder for layer configuration.

    Returned by :pymethod:`PolicyFlux.layers`.  Call :meth:`done` to
    return to the parent :class:`PolicyFlux` builder.
    """

    def __init__(self, parent: PolicyFlux) -> None:
        self._parent = parent
        self._cfg = parent._layer_config

    # --- toggles -------------------------------------------------------

    def ideal_point(self, *, enabled: bool = True) -> LayerBuilder:
        self._cfg.include_ideal_point = enabled
        return self

    def public_opinion(
        self, *, support: float | None = None, enabled: bool = True
    ) -> LayerBuilder:
        self._cfg.include_public_opinion = enabled
        if support is not None:
            self._cfg.public_support = support
        return self

    def lobbying(
        self, *, intensity: float | None = None, enabled: bool = True
    ) -> LayerBuilder:
        self._cfg.include_lobbying = enabled
        if intensity is not None:
            self._cfg.lobbying_intensity = intensity
        return self

    def media_pressure(
        self, *, pressure: float | None = None, enabled: bool = True
    ) -> LayerBuilder:
        self._cfg.include_media_pressure = enabled
        if pressure is not None:
            self._cfg.media_pressure = pressure
        return self

    def party_discipline(
        self,
        *,
        line_support: float | None = None,
        strength: float | None = None,
        enabled: bool = True,
    ) -> LayerBuilder:
        self._cfg.include_party_discipline = enabled
        if line_support is not None:
            self._cfg.party_line_support = line_support
        if strength is not None:
            self._cfg.party_discipline_strength = strength
        return self

    def government_agenda(
        self, *, pm_strength: float | None = None, enabled: bool = True
    ) -> LayerBuilder:
        self._cfg.include_government_agenda = enabled
        if pm_strength is not None:
            self._cfg.government_agenda_pm_strength = pm_strength
        return self

    def neural(self, *, factory: Any = None, enabled: bool = True) -> LayerBuilder:
        self._cfg.include_neural = enabled
        if factory is not None:
            self._cfg.neural_layer_factory = factory
        return self

    # --- extras --------------------------------------------------------

    def override(self, layer_name: str, **overrides: Any) -> LayerBuilder:
        existing = self._cfg.layer_overrides.get(layer_name, {})
        self._cfg.layer_overrides[layer_name] = {**existing, **overrides}
        return self

    def names(self, names: list[str]) -> LayerBuilder:
        self._cfg.layer_names = names
        return self

    # --- exit ----------------------------------------------------------

    def done(self) -> PolicyFlux:
        """Return to the parent :class:`PolicyFlux` builder."""
        return self._parent


class ExecutiveBuilder:
    """Scoped builder for executive system configuration.

    Returned by :meth:`PolicyFlux.executive`.  Call :meth:`done` to
    return to the parent :class:`PolicyFlux` builder.
    """

    def __init__(self, parent: PolicyFlux) -> None:
        self._parent = parent
        self._actors = parent._actors_config
        self._layers = parent._layer_config

    def presidential(
        self,
        *,
        approval_rating: float | None = None,
        veto_override: float | None = None,
    ) -> ExecutiveBuilder:
        self._actors.executive_type = ExecutiveType.PRESIDENTIAL
        if approval_rating is not None:
            self._actors.president_approval_rating = approval_rating
        if veto_override is not None:
            self._actors.veto_override_threshold = veto_override
        return self

    def parliamentary(
        self,
        *,
        pm_party_strength: float | None = None,
        confidence_threshold: float | None = None,
        government_bill_rate: float | None = None,
    ) -> ExecutiveBuilder:
        self._actors.executive_type = ExecutiveType.PARLIAMENTARY
        if pm_party_strength is not None:
            self._actors.pm_party_strength = pm_party_strength
            self._layers.government_agenda_pm_strength = pm_party_strength
        if confidence_threshold is not None:
            self._actors.confidence_threshold = confidence_threshold
        if government_bill_rate is not None:
            self._actors.government_bill_rate = government_bill_rate
        self._layers.include_government_agenda = True
        return self

    def semi_presidential(
        self,
        *,
        approval_rating: float | None = None,
        pm_party_strength: float | None = None,
    ) -> ExecutiveBuilder:
        self._actors.executive_type = ExecutiveType.SEMI_PRESIDENTIAL
        if approval_rating is not None:
            self._actors.semi_presidential_approval_rating = approval_rating
            self._actors.semi_president_approval = approval_rating
        if pm_party_strength is not None:
            self._actors.semi_presidential_pm_party_strength = pm_party_strength
            self._actors.semi_pm_party_strength = pm_party_strength
        return self

    def done(self) -> PolicyFlux:
        """Return to the parent :class:`PolicyFlux` builder."""
        return self._parent


class ActorBuilder:
    """Scoped builder for special-actor configuration.

    Returned by :meth:`PolicyFlux.special_actors`.  Call :meth:`done` to
    return to the parent :class:`PolicyFlux` builder.
    """

    def __init__(self, parent: PolicyFlux) -> None:
        self._parent = parent
        self._cfg = parent._actors_config

    def lobbyists(
        self,
        count: int,
        *,
        strength: float | None = None,
        stance: float | None = None,
    ) -> ActorBuilder:
        self._cfg.n_lobbyists = count
        if strength is not None:
            self._cfg.lobbyist_strength = strength
        if stance is not None:
            self._cfg.lobbyist_stance = stance
        return self

    def whips(
        self,
        count: int,
        *,
        discipline_strength: float | None = None,
        party_line_support: float | None = None,
    ) -> ActorBuilder:
        self._cfg.n_whips = count
        if discipline_strength is not None:
            self._cfg.whip_discipline_strength = discipline_strength
        if party_line_support is not None:
            self._cfg.whip_party_line_support = party_line_support
        return self

    def speaker(self, *, agenda_support: float | None = None) -> ActorBuilder:
        if agenda_support is not None:
            self._cfg.speaker_agenda_support = agenda_support
        return self

    def done(self) -> PolicyFlux:
        """Return to the parent :class:`PolicyFlux` builder."""
        return self._parent


# ======================================================================
# Main builder
# ======================================================================


class PolicyFlux:
    """Fluent builder for configuring and running PolicyFlux simulations.

    Supports two usage styles:

    * **Flat chaining** - all methods live on ``PolicyFlux`` directly.
    * **Section builders** - call :meth:`layers`, :meth:`executive`, or
      :meth:`special_actors` to enter a scoped sub-builder, then
      ``.done()`` to return here.

    Call :meth:`build` at the end to obtain a ready-to-run engine, or
    :meth:`build_config` / :meth:`to_config` if you only need the
    :class:`IntegrationConfig`.
    """

    def __init__(self) -> None:
        self._layer_config = LayerConfig()
        self._actors_config = AdvancedActorsConfig()
        self._num_actors: int = 100
        self._policy_dim: int = 4
        self._iterations: int = 300
        self._seed: int = 42
        self._description: str = "PolicyFlux modular simulation"
        self._aggregation_strategy: str = "sequential"
        self._aggregation_weights: list[float] | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def builder(cls) -> PolicyFlux:
        """Create a new :class:`PolicyFlux` builder instance."""
        return cls()

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------

    def actors(self, n: int) -> PolicyFlux:
        """Set the number of legislators."""
        self._num_actors = n
        return self

    def policy_dim(self, dim: int) -> PolicyFlux:
        """Set dimensionality of the policy space."""
        self._policy_dim = dim
        return self

    def iterations(self, n: int) -> PolicyFlux:
        """Set number of Monte Carlo iterations."""
        self._iterations = n
        return self

    def seed(self, value: int) -> PolicyFlux:
        """Set the random seed for reproducibility."""
        self._seed = value
        return self

    def description(self, text: str) -> PolicyFlux:
        """Set a human-readable description for the simulation."""
        self._description = text
        return self

    # ------------------------------------------------------------------
    # Section sub-builders
    # ------------------------------------------------------------------

    def layers(self) -> LayerBuilder:
        """Enter the layer configuration sub-builder.

        Call ``.done()`` on the returned :class:`LayerBuilder` to return here.
        """
        return LayerBuilder(self)

    def executive(self) -> ExecutiveBuilder:
        """Enter the executive-system configuration sub-builder.

        Call ``.done()`` on the returned :class:`ExecutiveBuilder` to return here.
        """
        return ExecutiveBuilder(self)

    def special_actors(self) -> ActorBuilder:
        """Enter the special-actor configuration sub-builder.

        Call ``.done()`` on the returned :class:`ActorBuilder` to return here.
        """
        return ActorBuilder(self)

    # ------------------------------------------------------------------
    # Flat layer toggles & parameters (kept for backwards-compatibility)
    # ------------------------------------------------------------------

    def with_ideal_point(self, *, enabled: bool = True) -> PolicyFlux:
        """Enable or disable the ideal-point layer."""
        self._layer_config.include_ideal_point = enabled
        return self

    def without_ideal_point(self) -> PolicyFlux:
        """Disable the ideal-point layer."""
        self._layer_config.include_ideal_point = False
        return self

    def with_public_opinion(
        self,
        *,
        support: float | None = None,
        enabled: bool = True,
    ) -> PolicyFlux:
        """Enable the public-opinion layer, optionally setting support level."""
        self._layer_config.include_public_opinion = enabled
        if support is not None:
            self._layer_config.public_support = support
        return self

    def without_public_opinion(self) -> PolicyFlux:
        """Disable the public-opinion layer."""
        self._layer_config.include_public_opinion = False
        return self

    def with_lobbying(
        self,
        *,
        intensity: float | None = None,
        enabled: bool = True,
    ) -> PolicyFlux:
        """Enable the lobbying layer, optionally setting intensity."""
        self._layer_config.include_lobbying = enabled
        if intensity is not None:
            self._layer_config.lobbying_intensity = intensity
        return self

    def without_lobbying(self) -> PolicyFlux:
        """Disable the lobbying layer."""
        self._layer_config.include_lobbying = False
        return self

    def with_media_pressure(
        self,
        *,
        pressure: float | None = None,
        enabled: bool = True,
    ) -> PolicyFlux:
        """Enable the media-pressure layer, optionally setting pressure."""
        self._layer_config.include_media_pressure = enabled
        if pressure is not None:
            self._layer_config.media_pressure = pressure
        return self

    def without_media_pressure(self) -> PolicyFlux:
        """Disable the media-pressure layer."""
        self._layer_config.include_media_pressure = False
        return self

    def with_party_discipline(
        self,
        *,
        line_support: float | None = None,
        strength: float | None = None,
        enabled: bool = True,
    ) -> PolicyFlux:
        """Enable party-discipline layer with optional parameters."""
        self._layer_config.include_party_discipline = enabled
        if line_support is not None:
            self._layer_config.party_line_support = line_support
        if strength is not None:
            self._layer_config.party_discipline_strength = strength
        return self

    def without_party_discipline(self) -> PolicyFlux:
        """Disable the party-discipline layer."""
        self._layer_config.include_party_discipline = False
        return self

    def with_government_agenda(
        self,
        *,
        pm_strength: float | None = None,
        enabled: bool = True,
    ) -> PolicyFlux:
        """Enable the government-agenda layer (parliamentary systems)."""
        self._layer_config.include_government_agenda = enabled
        if pm_strength is not None:
            self._layer_config.government_agenda_pm_strength = pm_strength
        return self

    def without_government_agenda(self) -> PolicyFlux:
        """Disable the government-agenda layer."""
        self._layer_config.include_government_agenda = False
        return self

    def with_neural(self, *, factory: Any = None, enabled: bool = True) -> PolicyFlux:
        """Enable the neural layer, optionally providing a factory callable."""
        self._layer_config.include_neural = enabled
        if factory is not None:
            self._layer_config.neural_layer_factory = factory
        return self

    def with_layer_override(self, layer_name: str, **overrides: Any) -> PolicyFlux:
        """Set parameter overrides for a specific layer by name."""
        existing = self._layer_config.layer_overrides.get(layer_name, {})
        self._layer_config.layer_overrides[layer_name] = {**existing, **overrides}
        return self

    # Short aliases without ``with_`` prefix
    layer_override = with_layer_override

    def layer_names(self, names: list[str]) -> PolicyFlux:
        """Explicitly list layer names to use (overrides boolean flags)."""
        self._layer_config.layer_names = names
        return self

    # ------------------------------------------------------------------
    # Flat aggregation strategy
    # ------------------------------------------------------------------

    def aggregation(
        self,
        strategy: str,
        *,
        weights: list[float] | None = None,
    ) -> PolicyFlux:
        """Set the aggregation strategy (sequential|average|weighted|multiplicative)."""
        self._aggregation_strategy = strategy
        if weights is not None:
            self._aggregation_weights = weights
        return self

    # ------------------------------------------------------------------
    # Flat executive system presets
    # ------------------------------------------------------------------

    def presidential(
        self,
        *,
        approval_rating: float | None = None,
        veto_override: float | None = None,
    ) -> PolicyFlux:
        """Configure a presidential executive system."""
        self._actors_config.executive_type = ExecutiveType.PRESIDENTIAL
        if approval_rating is not None:
            self._actors_config.president_approval_rating = approval_rating
        if veto_override is not None:
            self._actors_config.veto_override_threshold = veto_override
        return self

    def parliamentary(
        self,
        *,
        pm_party_strength: float | None = None,
        confidence_threshold: float | None = None,
        government_bill_rate: float | None = None,
    ) -> PolicyFlux:
        """Configure a parliamentary executive system.

        Automatically enables the government-agenda layer.
        """
        self._actors_config.executive_type = ExecutiveType.PARLIAMENTARY
        if pm_party_strength is not None:
            self._actors_config.pm_party_strength = pm_party_strength
            self._layer_config.government_agenda_pm_strength = pm_party_strength
        if confidence_threshold is not None:
            self._actors_config.confidence_threshold = confidence_threshold
        if government_bill_rate is not None:
            self._actors_config.government_bill_rate = government_bill_rate
        self._layer_config.include_government_agenda = True
        return self

    def semi_presidential(
        self,
        *,
        approval_rating: float | None = None,
        pm_party_strength: float | None = None,
    ) -> PolicyFlux:
        """Configure a semi-presidential executive system."""
        self._actors_config.executive_type = ExecutiveType.SEMI_PRESIDENTIAL
        if approval_rating is not None:
            self._actors_config.semi_presidential_approval_rating = approval_rating
            self._actors_config.semi_president_approval = approval_rating
        if pm_party_strength is not None:
            self._actors_config.semi_presidential_pm_party_strength = pm_party_strength
            self._actors_config.semi_pm_party_strength = pm_party_strength
        return self

    # ------------------------------------------------------------------
    # Flat special actors
    # ------------------------------------------------------------------

    def lobbyists(
        self,
        count: int,
        *,
        strength: float | None = None,
        stance: float | None = None,
    ) -> PolicyFlux:
        """Add lobbyists to the simulation."""
        self._actors_config.n_lobbyists = count
        if strength is not None:
            self._actors_config.lobbyist_strength = strength
        if stance is not None:
            self._actors_config.lobbyist_stance = stance
        return self

    def whips(
        self,
        count: int,
        *,
        discipline_strength: float | None = None,
        party_line_support: float | None = None,
    ) -> PolicyFlux:
        """Add party whips to the simulation."""
        self._actors_config.n_whips = count
        if discipline_strength is not None:
            self._actors_config.whip_discipline_strength = discipline_strength
        if party_line_support is not None:
            self._actors_config.whip_party_line_support = party_line_support
        return self

    def speaker(self, *, agenda_support: float | None = None) -> PolicyFlux:
        """Configure the speaker's agenda-setting influence."""
        if agenda_support is not None:
            self._actors_config.speaker_agenda_support = agenda_support
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_config(self) -> IntegrationConfig:
        """Materialise the accumulated settings into an :class:`IntegrationConfig`."""
        return IntegrationConfig(
            num_actors=self._num_actors,
            policy_dim=self._policy_dim,
            iterations=self._iterations,
            seed=self._seed,
            description=self._description,
            layer_config=self._layer_config,
            actors_config=self._actors_config,
            aggregation_strategy=self._aggregation_strategy,
            aggregation_weights=self._aggregation_weights,
        )

    # Alias so callers can use either name.
    to_config = build_config

    def build(self) -> Any:
        """Build and return a ready-to-run simulation engine.

        Returns:
            A :class:`SequentialMonteCarlo` engine. Call ``.run()`` on it.
        """
        from .builders.engine_builder import build_engine

        return build_engine(self.build_config())
