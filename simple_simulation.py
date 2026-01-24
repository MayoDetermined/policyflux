#!/usr/bin/env python3
"""
Prosty przykład użycia policyflux
Uruchom: python examples/simple_example.py
"""

from policyflux.integration import (
    AdvancedActorsConfig,
    IntegrationConfig,
    LayerConfig,
    build_engine,
)

SEED = 12345
NUM_ACTORS = 100
POLICY_DIM = 4  # Econ, Social, Foreign, Emotions
ITERATIONS = 300


def run_example():
    config = IntegrationConfig(
        num_actors=NUM_ACTORS,
        policy_dim=POLICY_DIM,
        iterations=ITERATIONS,
        seed=SEED,
        description="Prosty przykład policyflux",
        layer_config=LayerConfig(
            include_ideal_point=True,
            include_public_opinion=True,
            include_lobbying=True,
            include_media_pressure=True,
            include_party_discipline=True,
            public_support=0.55,
            lobbying_intensity=0.15,
            media_pressure=0.1,
            party_line_support=0.6,
            party_discipline_strength=0.4,
        ),
        actors_config=AdvancedActorsConfig(
            n_lobbyists=3,
            lobbyist_strength=0.4,
            lobbyist_stance=1.0,
            n_whips=2,
            whip_discipline_strength=0.6,
            whip_party_line_support=0.65,
            speaker_agenda_support=0.55,
            president_approval_rating=0.52,
        ),
    )
    engine = build_engine(config)
    engine.run_simulation()

    print("\nWynik symulacji:")
    print(engine)

    engine.get_pretty_votes()


if __name__ == "__main__":
    run_example()
