from typing import Any

from ...core.abstract_executive import ExecutiveType
from ..config import AdvancedActorsConfig, IntegrationConfig, LayerConfig


def create_parliamentary_config(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    pm_party_strength: float = 0.55,
    confidence_threshold: float = 0.5,
    government_bill_rate: float = 0.7,
    **kwargs: Any,
) -> IntegrationConfig:
    """Create configuration for a parliamentary system (UK/Canada-style).

    Args:
        num_actors: Number of MPs
        policy_dim: Policy space dimensionality
        iterations: Monte Carlo iterations
        seed: Random seed
        pm_party_strength: PM's party strength [0-1]
        confidence_threshold: Threshold for confidence votes
        government_bill_rate: Proportion of bills that are government bills
        **kwargs: Additional configuration overrides

    Returns:
        IntegrationConfig configured for parliamentary system
    """
    actors_config = AdvancedActorsConfig(
        executive_type=ExecutiveType.PARLIAMENTARY,
        pm_party_strength=pm_party_strength,
        confidence_threshold=confidence_threshold,
        government_bill_rate=government_bill_rate,
        **{k: v for k, v in kwargs.items() if k in AdvancedActorsConfig.__dataclass_fields__},
    )

    layer_config = LayerConfig(
        include_government_agenda=True,
        government_agenda_pm_strength=pm_party_strength,
        **{k: v for k, v in kwargs.items() if k in LayerConfig.__dataclass_fields__},
    )

    return IntegrationConfig(
        num_actors=num_actors,
        policy_dim=policy_dim,
        iterations=iterations,
        seed=seed,
        layer_config=layer_config,
        actors_config=actors_config,
        **{
            k: v
            for k, v in kwargs.items()
            if k in IntegrationConfig.__dataclass_fields__
            and k not in ["layer_config", "actors_config"]
        },
    )
