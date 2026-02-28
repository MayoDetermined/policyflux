from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

try:
    # pydantic v1 / v2 compatibility: BaseSettings moved to pydantic_settings in v2
    from pydantic import BaseSettings
except Exception:
    from pydantic_settings import BaseSettings

try:
    from pydantic import ConfigDict

    _HAS_CONFIG_DICT = True
except ImportError:
    _HAS_CONFIG_DICT = False

from functools import lru_cache

from ..core.abstract_executive import ExecutiveType

TAdvancedActorsConfig = TypeVar("TAdvancedActorsConfig", bound="AdvancedActorsConfig")
TLayerConfig = TypeVar("TLayerConfig", bound="LayerConfig")
TIntegrationConfig = TypeVar("TIntegrationConfig", bound="IntegrationConfig")


class Settings(BaseSettings):  # type: ignore[valid-type,misc]
    """Central configuration for policyflux.

    Values can be overridden via environment variables with prefix
    `POLICYFLUX_` (e.g. `POLICYFLUX_SEED`, `POLICYFLUX_LOG_LEVEL`).
    """

    seed: int = 42
    log_level: str = "INFO"

    if _HAS_CONFIG_DICT:
        model_config = ConfigDict(env_prefix="POLICYFLUX_")  # type: ignore[typeddict-unknown-key]
    else:

        class Config:
            env_prefix = "POLICYFLUX_"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@dataclass
class AdvancedActorsConfig:
    # System type
    executive_type: ExecutiveType = ExecutiveType.PRESIDENTIAL

    # Lobbyists
    n_lobbyists: int = 0
    lobbyist_strength: float = 0.5
    lobbyist_stance: float = 1.0

    # Whips
    n_whips: int = 0
    whip_discipline_strength: float = 0.5
    whip_party_line_support: float = 0.5

    # Speaker
    speaker_agenda_support: float = 0.5

    # PRESIDENTIAL SYSTEM
    president_approval_rating: float = 0.5
    veto_override_threshold: float = 2 / 3

    # PARLIAMENTARY SYSTEM
    pm_party_strength: float = 0.55
    confidence_threshold: float = 0.5
    government_bill_rate: float = 0.7  # % of bills that are government bills

    # SEMI-PRESIDENTIAL SYSTEM
    semi_presidential_approval_rating: float = 0.5
    semi_presidential_pm_party_strength: float = 0.55

    # Backward-compatible aliases (deprecated)
    semi_president_approval: float | None = None
    semi_pm_party_strength: float | None = None

    def __post_init__(self) -> None:
        if self.semi_president_approval is not None:
            self.semi_presidential_approval_rating = self.semi_president_approval
        if self.semi_pm_party_strength is not None:
            self.semi_presidential_pm_party_strength = self.semi_pm_party_strength

        self.semi_president_approval = self.semi_presidential_approval_rating
        self.semi_pm_party_strength = self.semi_presidential_pm_party_strength

    def with_executive_type(
        self: TAdvancedActorsConfig,
        executive_type: ExecutiveType,
    ) -> TAdvancedActorsConfig:
        self.executive_type = executive_type
        return self

    def with_lobbyists(
        self: TAdvancedActorsConfig,
        count: int,
        strength: float | None = None,
        stance: float | None = None,
    ) -> TAdvancedActorsConfig:
        self.n_lobbyists = count
        if strength is not None:
            self.lobbyist_strength = strength
        if stance is not None:
            self.lobbyist_stance = stance
        return self

    def with_whips(
        self: TAdvancedActorsConfig,
        count: int,
        discipline_strength: float | None = None,
        party_line_support: float | None = None,
    ) -> TAdvancedActorsConfig:
        self.n_whips = count
        if discipline_strength is not None:
            self.whip_discipline_strength = discipline_strength
        if party_line_support is not None:
            self.whip_party_line_support = party_line_support
        return self

    def with_speaker_agenda_support(
        self: TAdvancedActorsConfig,
        support: float,
    ) -> TAdvancedActorsConfig:
        self.speaker_agenda_support = support
        return self

    def with_presidential(
        self: TAdvancedActorsConfig,
        approval_rating: float | None = None,
        veto_override_threshold: float | None = None,
    ) -> TAdvancedActorsConfig:
        if approval_rating is not None:
            self.president_approval_rating = approval_rating
        if veto_override_threshold is not None:
            self.veto_override_threshold = veto_override_threshold
        return self

    def with_parliamentary(
        self: TAdvancedActorsConfig,
        pm_party_strength: float | None = None,
        confidence_threshold: float | None = None,
        government_bill_rate: float | None = None,
    ) -> TAdvancedActorsConfig:
        if pm_party_strength is not None:
            self.pm_party_strength = pm_party_strength
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if government_bill_rate is not None:
            self.government_bill_rate = government_bill_rate
        return self

    def with_semi_presidential(
        self: TAdvancedActorsConfig,
        approval_rating: float | None = None,
        pm_party_strength: float | None = None,
    ) -> TAdvancedActorsConfig:
        if approval_rating is not None:
            self.semi_presidential_approval_rating = approval_rating
            self.semi_president_approval = approval_rating
        if pm_party_strength is not None:
            self.semi_presidential_pm_party_strength = pm_party_strength
            self.semi_pm_party_strength = pm_party_strength
        return self


@dataclass
class LayerConfig:
    include_ideal_point: bool = True
    include_public_opinion: bool = True
    include_lobbying: bool = True
    include_media_pressure: bool = True
    include_party_discipline: bool = True
    include_government_agenda: bool = False
    include_neural: bool = False

    layer_names: list[str] | None = None
    layer_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    public_support: float = 0.5
    lobbying_intensity: float = 0.0
    media_pressure: float = 0.0
    party_line_support: float = 0.5
    party_discipline_strength: float = 0.5
    government_agenda_pm_strength: float = 0.6

    neural_layer_factory: Callable[[], object] | None = None

    def with_layer(self: TLayerConfig, name: str, enabled: bool = True) -> TLayerConfig:
        attr_name = f"include_{name}"
        if not hasattr(self, attr_name):
            raise ValueError(f"Unknown layer flag: {attr_name}")
        setattr(self, attr_name, enabled)
        return self

    def with_layer_names(self: TLayerConfig, names: list[str]) -> TLayerConfig:
        self.layer_names = names
        return self

    def with_layer_override(self: TLayerConfig, layer_name: str, **overrides: Any) -> TLayerConfig:
        existing = self.layer_overrides.get(layer_name, {})
        self.layer_overrides[layer_name] = {**existing, **overrides}
        return self

    def with_public_support(self: TLayerConfig, support: float) -> TLayerConfig:
        self.public_support = support
        return self

    def with_lobbying_intensity(self: TLayerConfig, intensity: float) -> TLayerConfig:
        self.lobbying_intensity = intensity
        return self

    def with_media_pressure(self: TLayerConfig, pressure: float) -> TLayerConfig:
        self.media_pressure = pressure
        return self

    def with_party_discipline(
        self: TLayerConfig,
        line_support: float | None = None,
        discipline_strength: float | None = None,
    ) -> TLayerConfig:
        if line_support is not None:
            self.party_line_support = line_support
        if discipline_strength is not None:
            self.party_discipline_strength = discipline_strength
        return self

    def with_government_agenda_strength(self: TLayerConfig, pm_strength: float) -> TLayerConfig:
        self.government_agenda_pm_strength = pm_strength
        return self

    def with_neural_layer_factory(
        self: TLayerConfig,
        factory: Callable[[], object] | None,
    ) -> TLayerConfig:
        self.neural_layer_factory = factory
        return self


@dataclass
class IntegrationConfig:
    num_actors: int = 100
    policy_dim: int = 4
    iterations: int = 300
    seed: int = 42
    description: str = "PolicyFlux modular simulation"

    layer_config: LayerConfig = field(default_factory=LayerConfig)
    actors_config: AdvancedActorsConfig = field(default_factory=AdvancedActorsConfig)

    aggregation_strategy: str = "sequential"  # sequential|average|weighted|multiplicative
    aggregation_weights: list[float] | None = None

    @classmethod
    def from_flat(cls, **kwargs: Any) -> "IntegrationConfig":
        """Create config from a flat dictionary of fields.

        Unknown fields raise ``ValueError``. Missing fields keep dataclass defaults.
        """
        return cls().with_flat(**kwargs)

    def with_simulation(
        self: TIntegrationConfig,
        num_actors: int | None = None,
        policy_dim: int | None = None,
        iterations: int | None = None,
        seed: int | None = None,
        description: str | None = None,
    ) -> TIntegrationConfig:
        if num_actors is not None:
            self.num_actors = num_actors
        if policy_dim is not None:
            self.policy_dim = policy_dim
        if iterations is not None:
            self.iterations = iterations
        if seed is not None:
            self.seed = seed
        if description is not None:
            self.description = description
        return self

    def with_layer_config(
        self: TIntegrationConfig,
        config: LayerConfig,
    ) -> TIntegrationConfig:
        self.layer_config = config
        return self

    def with_actors_config(
        self: TIntegrationConfig,
        config: AdvancedActorsConfig,
    ) -> TIntegrationConfig:
        self.actors_config = config
        return self

    def with_layers(self: TIntegrationConfig, **kwargs: Any) -> TIntegrationConfig:
        for key, value in kwargs.items():
            if not hasattr(self.layer_config, key):
                raise ValueError(f"Unknown LayerConfig field: {key}")
            setattr(self.layer_config, key, value)
        return self

    def with_actors(self: TIntegrationConfig, **kwargs: Any) -> TIntegrationConfig:
        for key, value in kwargs.items():
            if not hasattr(self.actors_config, key):
                raise ValueError(f"Unknown AdvancedActorsConfig field: {key}")
            setattr(self.actors_config, key, value)
        return self

    def with_aggregation(
        self: TIntegrationConfig,
        strategy: str,
        weights: list[float] | None = None,
    ) -> TIntegrationConfig:
        self.aggregation_strategy = strategy
        if strategy == "weighted":
            self.aggregation_weights = weights
        elif weights is not None:
            self.aggregation_weights = weights
        return self

    def with_flat(self: TIntegrationConfig, **kwargs: Any) -> TIntegrationConfig:
        """Apply flat configuration across top-level, layer, and actor fields.

        Field resolution order:
        1) ``IntegrationConfig`` direct fields (except nested config objects)
        2) ``LayerConfig`` fields
        3) ``AdvancedActorsConfig`` fields
        """
        unknown_fields: list[str] = []

        for key, value in kwargs.items():
            if key in {"layer_config", "actors_config"}:
                unknown_fields.append(key)
                continue

            if hasattr(self, key):
                setattr(self, key, value)
                continue

            if hasattr(self.layer_config, key):
                setattr(self.layer_config, key, value)
                continue

            if hasattr(self.actors_config, key):
                setattr(self.actors_config, key, value)
                continue

            unknown_fields.append(key)

        if unknown_fields:
            unknown_joined = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown flat config field(s): {unknown_joined}")

        return self
