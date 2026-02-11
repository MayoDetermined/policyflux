"""
Example: Using different executive systems in PolicyFlux

This example demonstrates how to easily configure and run simulations
with different executive systems (Presidential, Parliamentary, Semi-Presidential)
without any hard-coding.
"""

from policyflux.integration import (
    create_presidential_config,
    create_parliamentary_config,
    create_semi_presidential_config,
    build_engine,
)


def run_presidential_system():
    """Example: Presidential system (US-style)"""
    print("=" * 60)
    print("PRESIDENTIAL SYSTEM (US-style)")
    print("=" * 60)

    # Easy configuration with helper function
    config = create_presidential_config(
        num_actors=100,
        policy_dim=2,
        iterations=100,
        president_approval=0.6,  # 60% approval
        veto_override_threshold=2/3,  # Need 2/3 to override veto
        # Layer configuration
        include_lobbying=True,
        lobbying_intensity=0.3,
        n_lobbyists=2,
        lobbyist_strength=0.7,
    )

    # Build and run simulation
    engine = build_engine(config)
    results = engine.run_simulation()

    print(f"Average votes for: {sum(results) / len(results):.2f}")
    print(f"Average votes against: {config.num_actors - sum(results) / len(results):.2f}")
    print()


def run_parliamentary_system():
    """Example: Parliamentary system (UK/Canada-style)"""
    print("=" * 60)
    print("PARLIAMENTARY SYSTEM (UK/Canada-style)")
    print("=" * 60)

    # Easy configuration with helper function
    config = create_parliamentary_config(
        num_actors=100,
        policy_dim=2,
        iterations=100,
        pm_party_strength=0.55,  # Small majority
        confidence_threshold=0.5,
        government_bill_rate=0.7,  # 70% of bills are government bills
        # Layer configuration
        include_party_discipline=True,
        party_discipline_strength=0.8,  # High discipline
        n_whips=2,
        whip_discipline_strength=0.7,
    )

    # Government agenda layer is automatically enabled!
    print(f"Government agenda layer enabled: {config.layer_config.include_government_agenda}")

    # Build and run simulation
    engine = build_engine(config)
    results = engine.run_simulation()

    print(f"Average votes for: {sum(results) / len(results):.2f}")
    print(f"Average votes against: {config.num_actors - sum(results) / len(results):.2f}")
    print()


def run_semi_presidential_system():
    """Example: Semi-Presidential system (France/Poland-style)"""
    print("=" * 60)
    print("SEMI-PRESIDENTIAL SYSTEM (France/Poland-style)")
    print("=" * 60)

    # Easy configuration with helper function
    config = create_semi_presidential_config(
        num_actors=100,
        policy_dim=2,
        iterations=100,
        president_approval=0.45,  # Weak president
        pm_party_strength=0.6,    # Strong PM
        # This creates COHABITATION scenario!
        include_media_pressure=True,
        media_pressure=0.5,
    )

    # Build and run simulation
    engine = build_engine(config)
    results = engine.run_simulation()

    print(f"Average votes for: {sum(results) / len(results):.2f}")
    print(f"Average votes against: {config.num_actors - sum(results) / len(results):.2f}")
    print()


def custom_executive_configuration():
    """Example: Custom configuration with full control"""
    print("=" * 60)
    print("CUSTOM CONFIGURATION")
    print("=" * 60)

    from policyflux.integration import IntegrationConfig, AdvancedActorsConfig, LayerConfig
    from policyflux.core.executive import ExecutiveType

    # Full manual configuration
    config = IntegrationConfig(
        num_actors=50,
        policy_dim=3,
        iterations=50,
        seed=12345,

        # Configure executive
        actors_config=AdvancedActorsConfig(
            executive_type=ExecutiveType.PARLIAMENTARY,
            pm_party_strength=0.52,  # Razor-thin majority!
            confidence_threshold=0.5,
            n_whips=3,
            whip_discipline_strength=0.85,
            n_lobbyists=5,
            lobbyist_strength=0.4,
        ),

        # Configure layers
        layer_config=LayerConfig(
            include_ideal_point=True,
            include_public_opinion=True,
            include_lobbying=True,
            include_party_discipline=True,
            include_government_agenda=True,  # Parliamentary system
            public_support=0.6,
            lobbying_intensity=0.4,
            party_discipline_strength=0.75,
            government_agenda_pm_strength=0.52,
        ),

        # Aggregation strategy
        aggregation_strategy="sequential",
    )

    # Build and run
    engine = build_engine(config)
    results = engine.run_simulation()

    print(f"Average votes for: {sum(results) / len(results):.2f}")
    print(f"Simulation variance: {sum((x - sum(results)/len(results))**2 for x in results) / len(results):.2f}")
    print()


def main():
    """Run all examples"""
    run_presidential_system()
    run_parliamentary_system()
    run_semi_presidential_system()
    custom_executive_configuration()

    print("=" * 60)
    print("KEY FEATURES:")
    print("=" * 60)
    print("[+] No hard-coding - all configuration through parameters")
    print("[+] Easy helper functions for common systems")
    print("[+] Full manual control when needed")
    print("[+] Automatic layer configuration (e.g., government agenda for parliamentary)")
    print("[+] All executive systems fully integrated")
    print("[+] Veto, confidence votes, cohabitation all working")


if __name__ == "__main__":
    main()
