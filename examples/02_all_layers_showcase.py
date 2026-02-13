#!/usr/bin/env python3
"""
02 - WARSTWY DECYZYJNE: Kompleksowy przegląd wszystkich warstw
==============================================================

Ten przykład systematycznie pokazuje wszystkie dostępne warstwy decyzyjne:
1. Ideal Point - bazowe preferencje ideologiczne
2. Public Opinion - wpływ opinii publicznej
3. Lobbying - naciski grup interesów
4. Media Pressure - wpływ mediów
5. Party Discipline - dyscyplina partyjna
6. Government Agenda - kontrola rządowa (systemy parlamentarne)

Każda warstwa jest testowana osobno, a potem wszystkie razem.

Uruchom: python examples/02_all_layers_showcase.py
"""

from policyflux.integration import (
    IntegrationConfig,
    LayerConfig,
    AdvancedActorsConfig,
    build_engine,
)

SEED = 42
NUM_ACTORS = 80
POLICY_DIM = 2  # 2D: Left-Right + Libertarian-Authoritarian
ITERATIONS = 150


def run_with_config(description: str, layer_config: LayerConfig, actors_config=None):
    """Helper do uruchomienia symulacji z daną konfiguracją."""
    config = IntegrationConfig(
        num_actors=NUM_ACTORS,
        policy_dim=POLICY_DIM,
        iterations=ITERATIONS,
        seed=SEED,
        description=description,
        layer_config=layer_config,
        actors_config=actors_config,
    )

    engine = build_engine(config)
    engine.run_simulation()

    total = len(engine.congress_model.congressmen)
    avg_votes = sum(engine.results) / len(engine.results)
    pass_rate = sum(1 for v in engine.results if v > total / 2) / len(engine.results)

    return avg_votes, pass_rate


def showcase_all_layers():
    """Systematyczne porównanie wszystkich warstw."""

    print("\n" + "="*70)
    print("02 - WARSTWY DECYZYJNE: Kompleksowy przegląd")
    print("="*70)

    results = []

    # 1. TYLKO IDEAL POINT (baseline)
    print("\n[1] IDEAL POINT - Bazowa warstwa ideologiczna")
    print("-"*70)
    print("Opis: Posłowie głosują na podstawie odległości euklidesowej od ustawy")
    print("      w wielowymiarowej przestrzeni politycznej.")

    avg, pass_r = run_with_config(
        "Tylko Ideal Point",
        LayerConfig(include_ideal_point=True),
    )
    results.append(("Ideal Point (baseline)", avg, pass_r))
    print(f"Wynik: Średnio {avg:.1f}/{NUM_ACTORS} głosów ZA, Pass rate: {pass_r:.1%}")

    # 2. IDEAL POINT + PUBLIC OPINION
    print("\n[2] PUBLIC OPINION - Wpływ opinii publicznej")
    print("-"*70)
    print("Opis: Posłowie uwzględniają poparcie społeczne dla ustawy.")
    print("      Wysokie poparcie zwiększa szanse na głos ZA.")

    avg, pass_r = run_with_config(
        "Ideal Point + Public Opinion",
        LayerConfig(
            include_ideal_point=True,
            include_public_opinion=True,
            public_support=0.70,  # Wysokie poparcie
        ),
    )
    results.append(("+ Public Opinion (70%)", avg, pass_r))
    print(f"Wynik: Średnio {avg:.1f}/{NUM_ACTORS} głosów ZA, Pass rate: {pass_r:.1%}")
    print(f"Efekt: {'+' if pass_r > results[0][2] else '-'} {abs(pass_r - results[0][2]):.1%} vs baseline")

    # 3. IDEAL POINT + LOBBYING
    print("\n[3] LOBBYING - Naciski grup interesów")
    print("-"*70)
    print("Opis: Lobbyści wywierają presję na posłów, by głosowali zgodnie")
    print("      z interesami swoich klientów.")

    avg, pass_r = run_with_config(
        "Ideal Point + Lobbying",
        LayerConfig(
            include_ideal_point=True,
            include_lobbying=True,
            lobbying_intensity=0.60,  # Silny lobbing
        ),
        actors_config=AdvancedActorsConfig(
            n_lobbyists=5,
            lobbyist_strength=0.65,
            lobbyist_stance=0.85,  # Pro-ustawa
        ),
    )
    results.append(("+ Lobbying (strong)", avg, pass_r))
    print(f"Wynik: Średnio {avg:.1f}/{NUM_ACTORS} głosów ZA, Pass rate: {pass_r:.1%}")
    print(f"Efekt: {'+' if pass_r > results[0][2] else '-'} {abs(pass_r - results[0][2]):.1%} vs baseline")

    # 4. IDEAL POINT + MEDIA PRESSURE
    print("\n[4] MEDIA PRESSURE - Wpływ mediów")
    print("-"*70)
    print("Opis: Media kształtują narrację wokół ustawy, wpływając na")
    print("      percepcję posłów i ich decyzje głosowania.")

    avg, pass_r = run_with_config(
        "Ideal Point + Media",
        LayerConfig(
            include_ideal_point=True,
            include_media_pressure=True,
            media_pressure=0.55,  # Pozytywny spining
        ),
        actors_config=AdvancedActorsConfig(
            speaker_agenda_support=0.60,  # Speaker wspiera
            president_approval_rating=0.65,  # Popularny prezydent
        ),
    )
    results.append(("+ Media Pressure", avg, pass_r))
    print(f"Wynik: Średnio {avg:.1f}/{NUM_ACTORS} głosów ZA, Pass rate: {pass_r:.1%}")
    print(f"Efekt: {'+' if pass_r > results[0][2] else '-'} {abs(pass_r - results[0][2]):.1%} vs baseline")

    # 5. IDEAL POINT + PARTY DISCIPLINE
    print("\n[5] PARTY DISCIPLINE - Dyscyplina partyjna")
    print("-"*70)
    print("Opis: Bicze partyjne wymuszają zgodność z linią partii.")
    print("      Silna dyscyplina może przebić ideologiczne preferencje.")

    avg, pass_r = run_with_config(
        "Ideal Point + Party Discipline",
        LayerConfig(
            include_ideal_point=True,
            include_party_discipline=True,
            party_line_support=0.75,  # Silne wsparcie linii
            party_discipline_strength=0.80,  # Wysoka dyscyplina
        ),
        actors_config=AdvancedActorsConfig(
            n_whips=3,
            whip_discipline_strength=0.85,
            whip_party_line_support=0.78,
        ),
    )
    results.append(("+ Party Discipline", avg, pass_r))
    print(f"Wynik: Średnio {avg:.1f}/{NUM_ACTORS} głosów ZA, Pass rate: {pass_r:.1%}")
    print(f"Efekt: {'+' if pass_r > results[0][2] else '-'} {abs(pass_r - results[0][2]):.1%} vs baseline")

    # 6. WSZYSTKIE WARSTWY RAZEM
    print("\n[6] WSZYSTKIE WARSTWY - Pełna kompleksowość")
    print("-"*70)
    print("Opis: Równoczesne działanie wszystkich czynników wpływających")
    print("      na decyzje posłów. Realistyczny model legislacyjny.")

    avg, pass_r = run_with_config(
        "Wszystkie warstwy",
        LayerConfig(
            include_ideal_point=True,
            include_public_opinion=True,
            include_lobbying=True,
            include_media_pressure=True,
            include_party_discipline=True,
            public_support=0.58,
            lobbying_intensity=0.35,
            media_pressure=0.25,
            party_line_support=0.62,
            party_discipline_strength=0.55,
        ),
        actors_config=AdvancedActorsConfig(
            n_lobbyists=4,
            lobbyist_strength=0.50,
            lobbyist_stance=0.70,
            n_whips=2,
            whip_discipline_strength=0.60,
            whip_party_line_support=0.65,
            speaker_agenda_support=0.58,
            president_approval_rating=0.55,
        ),
    )
    results.append(("ALL LAYERS", avg, pass_r))
    print(f"Wynik: Średnio {avg:.1f}/{NUM_ACTORS} głosów ZA, Pass rate: {pass_r:.1%}")
    print(f"Efekt: {'+' if pass_r > results[0][2] else '-'} {abs(pass_r - results[0][2]):.1%} vs baseline")

    # PODSUMOWANIE
    print("\n" + "="*70)
    print("PODSUMOWANIE WPŁYWU WARSTW:")
    print("="*70)
    for name, avg, pass_r in results:
        bar = "█" * int(pass_r * 50)
        print(f"{name:30s} | {pass_r:5.1%} {bar}")

    print("\n" + "="*70)
    print("KLUCZOWE WNIOSKI:")
    print("="*70)
    print("✓ Każda warstwa modyfikuje bazowe preferencje ideologiczne")
    print("✓ Party Discipline ma zazwyczaj najsilniejszy wpływ")
    print("✓ Lobbying i Public Opinion działają jako wzmacniacze")
    print("✓ Media Pressure i Speaker Agenda kształtują narrację")
    print("✓ Wszystkie warstwy razem tworzą realistyczny model legislacyjny")
    print("="*70 + "\n")


if __name__ == "__main__":
    showcase_all_layers()
