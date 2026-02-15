from ..config import IntegrationConfig, AdvancedActorsConfig, LayerConfig
from ...core.executive import ExecutiveType

def create_semi_presidential_config(
    num_actors: int = 100,
    policy_dim: int = 4,
    iterations: int = 300,
    seed: int = 42,
    president_approval: float = 0.5,
    pm_party_strength: float = 0.55,
    **kwargs
) -> IntegrationConfig:
    """Create configuration for a semi-presidential system (France/Poland-style).

    Args:
        num_actors: Number of representatives
        policy_dim: Policy space dimensionality
        iterations: Monte Carlo iterations
        seed: Random seed
        president_approval: Presidential approval rating [0-1]
        pm_party_strength: PM's party strength [0-1]
        **kwargs: Additional configuration overrides

    Returns:
        IntegrationConfig configured for semi-presidential system
    """
    actors_config = AdvancedActorsConfig(
        executive_type=ExecutiveType.SEMI_PRESIDENTIAL,
        semi_president_approval=president_approval,
        semi_pm_party_strength=pm_party_strength,
        **{k: v for k, v in kwargs.items() if k in AdvancedActorsConfig.__dataclass_fields__}
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
        **{k: v for k, v in kwargs.items() if k in IntegrationConfig.__dataclass_fields__
           and k not in ['layer_config', 'actors_config']}
    )
