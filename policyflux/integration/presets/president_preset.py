from typing import Any

from ...core.abstract_executive import ExecutiveType
from ..config import AdvancedActorsConfig, IntegrationConfig, LayerConfig


def create_presidential_config(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    president_approval: float = 0.5,
    veto_override_threshold: float = 2 / 3,
    **kwargs: Any,
) -> IntegrationConfig:
    """Create configuration for a presidential system (US-style).

    Args:
        num_actors: Number of congressmen
        policy_dim: Policy space dimensionality
        iterations: Monte Carlo iterations
        seed: Random seed
        president_approval: Presidential approval rating [0-1]
        veto_override_threshold: Threshold to override veto (default 2/3)
        **kwargs: Additional configuration overrides

    Returns:
        IntegrationConfig configured for presidential system
    """
    actors_config = AdvancedActorsConfig(
        executive_type=ExecutiveType.PRESIDENTIAL,
        president_approval_rating=president_approval,
        veto_override_threshold=veto_override_threshold,
        **{k: v for k, v in kwargs.items() if k in AdvancedActorsConfig.__dataclass_fields__},
    )

    layer_config = LayerConfig(
        **{k: v for k, v in kwargs.items() if k in LayerConfig.__dataclass_fields__}
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
