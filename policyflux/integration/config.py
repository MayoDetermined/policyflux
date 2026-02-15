from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any, Dict
try:
    # pydantic v1 / v2 compatibility: BaseSettings moved to pydantic_settings in v2
    from pydantic import BaseSettings  # type: ignore
except Exception:
    from pydantic_settings import BaseSettings  # type: ignore
from functools import lru_cache
from ..core.executive import ExecutiveType

class Settings(BaseSettings):
    """Central configuration for policyflux.

    Values can be overridden via environment variables with prefix
    `POLICYFLUX_` (e.g. `POLICYFLUX_SEED`, `POLICYFLUX_LOG_LEVEL`).
    """

    seed: int = 42
    log_level: str = "INFO"

    class Config:
        env_prefix = "POLICYFLUX_"


@lru_cache()
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
    veto_override_threshold: float = 2/3
    
    # PARLIAMENTARY SYSTEM
    pm_party_strength: float = 0.55
    confidence_threshold: float = 0.5
    government_bill_rate: float = 0.7  # % of bills that are government bills
    
    # SEMI-PRESIDENTIAL SYSTEM
    semi_president_approval: float = 0.5
    semi_pm_party_strength: float = 0.55


@dataclass
class LayerConfig:
    include_ideal_point: bool = True
    include_public_opinion: bool = True
    include_lobbying: bool = True
    include_media_pressure: bool = True
    include_party_discipline: bool = True
    include_government_agenda: bool = False
    include_neural: bool = False

    layer_names: Optional[List[str]] = None
    layer_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    public_support: float = 0.5
    lobbying_intensity: float = 0.0
    media_pressure: float = 0.0
    party_line_support: float = 0.5
    party_discipline_strength: float = 0.5
    government_agenda_pm_strength: float = 0.6

    neural_layer_factory: Optional[Callable[[], object]] = None


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
    aggregation_weights: Optional[List[float]] = None