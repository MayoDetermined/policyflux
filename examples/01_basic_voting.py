#!/usr/bin/env python3
"""
01 - PODSTAWY: Najprostsze głosowanie ideologiczne
===================================================

Ten przykład pokazuje najbardziej podstawową funkcjonalność PolicyFlux:
- Tworzenie kongresu z posłami
- Definiowanie przestrzeni politycznej (1D: osi Left-Right)
- Pojedyncza warstwa: Ideal Point (ideologiczne preferencje)
- Symulacja Monte Carlo głosowań

Uruchom: python examples/01_basic_voting.py
"""

from policyflux.integration import IntegrationConfig, LayerConfig, build_engine

# Konfiguracja
SEED = 12345
NUM_ACTORS = 50  # 50 posłów
POLICY_DIM = 1  # 1D: tylko oś Left-Right
ITERATIONS = 100  # 100 głosowań


def run_basic_voting():
    """Najprostsza symulacja z samym Ideal Point."""

    print("\n" + "="*60)
    print("01 - PODSTAWY: Głosowanie ideologiczne")
    print("="*60)

    # Konfiguracja z TYLKO warstwą Ideal Point
    config = IntegrationConfig(
        num_actors=NUM_ACTORS,
        policy_dim=POLICY_DIM,
        iterations=ITERATIONS,
        seed=SEED,
        description="Podstawowe głosowanie - tylko ideologia",
        layer_config=LayerConfig(
            # Włącz TYLKO Ideal Point - wszystkie inne wyłączone
            include_ideal_point=True,
            include_public_opinion=False,
            include_lobbying=False,
            include_media_pressure=False,
            include_party_discipline=False,
            include_government_agenda=False,
        ),
        actors_config=None,  # Brak zaawansowanych aktorów
    )

    # Zbuduj silnik i uruchom symulację
    engine = build_engine(config)
    print(f"\nKongres: {NUM_ACTORS} posłów")
    print(f"Przestrzeń polityczna: {POLICY_DIM}D (Left-Right)")
    print(f"Iteracje: {ITERATIONS}")
    print("\nUruchamianie symulacji...")

    engine.run_simulation()

    # Wyniki
    print("\n" + "-"*60)
    print("WYNIKI SYMULACJI:")
    print("-"*60)
    print(engine)

    # Pokaż kilka przykładowych głosowań
    print("\n" + "-"*60)
    print("PRZYKŁADOWE GŁOSOWANIA:")
    print("-"*60)
    engine.get_pretty_votes()

    # Analiza
    total = len(engine.congress_model.congressmen)
    avg_votes_for = sum(engine.results) / len(engine.results)
    pass_rate = sum(1 for votes in engine.results if votes > total / 2) / len(engine.results)

    print("\n" + "-"*60)
    print("ANALIZA:")
    print("-"*60)
    print(f"Średnie głosy ZA: {avg_votes_for:.1f} / {total}")
    print(f"Pass rate: {pass_rate:.1%}")
    print(f"Status: {'Kongres wspierający' if pass_rate > 0.6 else 'Kongres zrównoważony' if pass_rate > 0.4 else 'Kongres opozycyjny'}")

    print("\n" + "="*60)
    print("KLUCZOWE WNIOSKI:")
    print("="*60)
    print("✓ Posłowie głosują wyłącznie na podstawie odległości ideologicznej")
    print("✓ Im bliżej ustawa do punktu idealnego posła, tym większa szansa na głos ZA")
    print("✓ W czystym modelu ideologicznym pass rate oscyluje wokół 50%")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_basic_voting()
