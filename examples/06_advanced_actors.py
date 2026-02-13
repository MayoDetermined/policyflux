#!/usr/bin/env python3
"""
06 - ZAAWANSOWANI AKTORZY: Speaker, Whips, Lobbyści, Prezydent
==============================================================

PolicyFlux modeluje kluczowych aktorów systemu politycznego:

1. SPEAKER (Przewodniczący):
   - Kontrola agendy (scheduling power)
   - Wpływ na przebieg debaty
   - Agenda support [0, 1]

2. PARTY WHIPS (Bicze partyjne):
   - Dyscyplina partyjna
   - Egzekwowanie linii partii
   - Każdy whip ma discipline_strength i party_line_support

3. LOBBYŚCI (Lobbyists):
   - Reprezentują grupy interesów
   - Wywierają presję na posłów
   - Influence strength [0, 1], stance [-1, 1]

4. PREZYDENT (President):
   - Wpływ przez approval rating
   - Może wetować (w systemach prezydenckich)
   - Kształtuje opinię publiczną

Ten przykład pokazuje wpływ różnych konfiguracji tych aktorów.

Uruchom: python examples/06_advanced_actors.py
"""

from policyflux.integration import (
    IntegrationConfig,
    LayerConfig,
    AdvancedActorsConfig,
    build_engine,
)
from policyflux.utils.reports import craft_a_bar

SEED = 8888
NUM_ACTORS = 90
POLICY_DIM = 2
ITERATIONS = 180


def run_scenario(name: str, config: IntegrationConfig):
    """Helper do uruchomienia scenariusza."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print('='*70)

    engine = build_engine(config)
    engine.run_simulation()

    total = len(engine.congress_model.congressmen)
    avg_votes = sum(engine.results) / len(engine.results)
    pass_rate = sum(1 for v in engine.results if v > total / 2) / len(engine.results)

    print(f"\nWynik: {avg_votes:.1f}/{total} głosów ZA")
    print(f"Pass rate: {pass_rate:.1%}")

    return {
        "name": name,
        "avg_votes": avg_votes,
        "pass_rate": pass_rate,
        "total": total,
    }


def showcase_advanced_actors():
    """Porównanie wpływu zaawansowanych aktorów."""

    print("\n" + "="*70)
    print("06 - ZAAWANSOWANI AKTORZY: Władza poza posłami")
    print("="*70)

    results = []

    # BASELINE: Brak zaawansowanych aktorów
    print("\n" + "▼"*70)
    print("BASELINE: Tylko posłowie (brak zaawansowanych aktorów)")
    print("▼"*70)
    print("\nKonfiguracja:")
    print("  • Tylko Ideal Point + Public Opinion")
    print("  • Brak Speaker, Whips, Lobbyistów, Prezydenta")
    print("  • Czysta demokracja bazowa")

    baseline_config = IntegrationConfig(
        num_actors=NUM_ACTORS,
        policy_dim=POLICY_DIM,
        iterations=ITERATIONS,
        seed=SEED,
        description="Baseline - brak zaawansowanych aktorów",
        layer_config=LayerConfig(
            include_ideal_point=True,
            include_public_opinion=True,
            public_support=0.55,
        ),
        actors_config=None,  # Brak!
    )

    results.append(run_scenario("BASELINE (czysta demokracja)", baseline_config))

    # SCENARIUSZ 1: Silny Speaker
    print("\n" + "▼"*70)
    print("SCENARIUSZ 1: Silny Speaker - kontrola agendy")
    print("▼"*70)
    print("\nKonfiguracja:")
    print("  • Speaker z agenda_support = 0.85 (bardzo silny)")
    print("  • Kontroluje które ustawy trafiają pod głosowanie")
    print("  • Kształtuje narrację wokół ustaw")

    strong_speaker_config = IntegrationConfig(
        num_actors=NUM_ACTORS,
        policy_dim=POLICY_DIM,
        iterations=ITERATIONS,
        seed=SEED,
        description="Silny Speaker",
        layer_config=LayerConfig(
            include_ideal_point=True,
            include_public_opinion=True,
            include_media_pressure=True,
            public_support=0.55,
            media_pressure=0.45,
        ),
        actors_config=AdvancedActorsConfig(
            speaker_agenda_support=0.85,  # Bardzo silny!
        ),
    )

    results.append(run_scenario("SILNY SPEAKER", strong_speaker_config))

    # SCENARIUSZ 2: Armia Whips
    print("\n" + "▼"*70)
    print("SCENARIUSZ 2: Armia Whips - żelazna dyscyplina")
    print("▼"*70)
    print("\nKonfiguracja:")
    print("  • 5 Whips z wysoką discipline_strength (0.90)")
    print("  • Party line support = 0.88")
    print("  • Wymuszanie zgodności z linią partii")

    strong_whips_config = IntegrationConfig(
        num_actors=NUM_ACTORS,
        policy_dim=POLICY_DIM,
        iterations=ITERATIONS,
        seed=SEED,
        description="Armia Whips",
        layer_config=LayerConfig(
            include_ideal_point=True,
            include_public_opinion=True,
            include_party_discipline=True,
            public_support=0.55,
            party_line_support=0.88,
            party_discipline_strength=0.90,
        ),
        actors_config=AdvancedActorsConfig(
            n_whips=5,  # Wielu!
            whip_discipline_strength=0.90,  # Bardzo silna
            whip_party_line_support=0.88,
        ),
    )

    results.append(run_scenario("ARMIA WHIPS", strong_whips_config))

    # SCENARIUSZ 3: Lobbying Blitz
    print("\n" + "▼"*70)
    print("SCENARIUSZ 3: Lobbying Blitz - atak grup interesów")
    print("▼"*70)
    print("\nKonfiguracja:")
    print("  • 10 Lobbyistów (!))")
    print("  • Influence strength = 0.80 (bardzo silni)")
    print("  • Stance = 1.0 (maksymalnie pro)")
    print("  • Koordynowana kampania lobbingowa")

    lobbying_blitz_config = IntegrationConfig(
        num_actors=NUM_ACTORS,
        policy_dim=POLICY_DIM,
        iterations=ITERATIONS,
        seed=SEED,
        description="Lobbying Blitz",
        layer_config=LayerConfig(
            include_ideal_point=True,
            include_public_opinion=True,
            include_lobbying=True,
            public_support=0.45,  # Niskie poparcie publiczne!
            lobbying_intensity=0.75,
        ),
        actors_config=AdvancedActorsConfig(
            n_lobbyists=10,  # Armia lobbyistów!
            lobbyist_strength=0.80,
            lobbyist_stance=1.0,  # Maksymalnie pro
        ),
    )

    results.append(run_scenario("LOBBYING BLITZ", lobbying_blitz_config))

    # SCENARIUSZ 4: Popularny Prezydent
    print("\n" + "▼"*70)
    print("SCENARIUSZ 4: Popularny Prezydent - bully pulpit")
    print("▼"*70)
    print("\nKonfiguracja:")
    print("  • Prezydent z approval rating = 0.85 (bardzo popularny)")
    print("  • Wpływ przez media i opinię publiczną")
    print("  • 'Bully pulpit' - kształtowanie narracji")

    popular_president_config = IntegrationConfig(
        num_actors=NUM_ACTORS,
        policy_dim=POLICY_DIM,
        iterations=ITERATIONS,
        seed=SEED,
        description="Popularny Prezydent",
        layer_config=LayerConfig(
            include_ideal_point=True,
            include_public_opinion=True,
            include_media_pressure=True,
            public_support=0.72,  # Prezydent podnosi poparcie
            media_pressure=0.55,
        ),
        actors_config=AdvancedActorsConfig(
            president_approval_rating=0.85,  # Bardzo popularny!
            speaker_agenda_support=0.60,
        ),
    )

    results.append(run_scenario("POPULARNY PREZYDENT", popular_president_config))

    # SCENARIUSZ 5: Koalicja sił
    print("\n" + "▼"*70)
    print("SCENARIUSZ 5: Koalicja Sił - wszyscy razem")
    print("▼"*70)
    print("\nKonfiguracja:")
    print("  • Silny Speaker + Whips + Lobbyści + Popularny Prezydent")
    print("  • Koordynowana ofensywa wszystkich aktorów")
    print("  • Maksymalna presja na posłów")

    coalition_config = IntegrationConfig(
        num_actors=NUM_ACTORS,
        policy_dim=POLICY_DIM,
        iterations=ITERATIONS,
        seed=SEED,
        description="Koalicja Sił",
        layer_config=LayerConfig(
            include_ideal_point=True,
            include_public_opinion=True,
            include_lobbying=True,
            include_media_pressure=True,
            include_party_discipline=True,
            public_support=0.70,
            lobbying_intensity=0.60,
            media_pressure=0.50,
            party_line_support=0.75,
            party_discipline_strength=0.70,
        ),
        actors_config=AdvancedActorsConfig(
            n_lobbyists=6,
            lobbyist_strength=0.70,
            lobbyist_stance=0.90,
            n_whips=3,
            whip_discipline_strength=0.75,
            whip_party_line_support=0.78,
            speaker_agenda_support=0.75,
            president_approval_rating=0.75,
        ),
    )

    results.append(run_scenario("KOALICJA SIŁ", coalition_config))

    # PORÓWNANIE
    print("\n" + "="*70)
    print("PORÓWNANIE WPŁYWU AKTORÓW:")
    print("="*70)

    print("\nPass Rate:")
    for r in results:
        bar = "█" * int(r["pass_rate"] * 50)
        print(f"  {r['name']:30s} | {r['pass_rate']:5.1%} {bar}")

    # Różnice względem baseline
    baseline_rate = results[0]["pass_rate"]
    print(f"\nWpływ względem BASELINE ({baseline_rate:.1%}):")
    for r in results[1:]:
        diff = r["pass_rate"] - baseline_rate
        sign = "+" if diff > 0 else ""
        arrow = "↑" if diff > 0 else "↓"
        print(f"  {r['name']:30s} | {sign}{diff:+.1%} {arrow}")

    # Wykres
    try:
        craft_a_bar(
            data=[r["pass_rate"] * 100 for r in results],
            labels=[r["name"][:15] for r in results],  # Skróć nazwy
            title="Wpływ zaawansowanych aktorów na pass rate",
            xlabel="Scenariusz",
            ylabel="Pass Rate [%]",
        )
    except Exception as e:
        print(f"\n(Wykres niedostępny: {e})")

    # ANALIZA
    print("\n" + "="*70)
    print("ANALIZA WPŁYWU:")
    print("="*70)

    # Ranking
    ranked = sorted(results[1:], key=lambda r: r["pass_rate"], reverse=True)

    print("\nRanking skuteczności (od najsilniejszego):")
    for i, r in enumerate(ranked, 1):
        diff = r["pass_rate"] - baseline_rate
        print(f"  {i}. {r['name']:30s} (+{diff:.1%})")

    print("\n" + "="*70)
    print("KLUCZOWE WNIOSKI:")
    print("="*70)

    print("\n1. Speaker (Agenda Control):")
    print("   → Kontrola nad tym, co trafia pod głosowanie")
    print("   → Wpływ przez media pressure")
    print("   → Kluczowy w systemach parlamentarnych")

    print("\n2. Whips (Party Discipline):")
    print("   → Najsilniejszy pojedynczy wpływ na wyniki")
    print("   → Wymuszanie zgodności z linią partii")
    print("   → Kluczowi w strong party systems (UK, Kanada)")

    print("\n3. Lobbyści (Interest Groups):")
    print("   → Mogą przebić niskie poparcie publiczne")
    print("   → Wiele grup = skumulowany wpływ")
    print("   → Szczególnie silni w USA")

    print("\n4. Prezydent (Bully Pulpit):")
    print("   → Wpływ przez approval rating")
    print("   → Kształtuje opinię publiczną i media")
    print("   → Indirect power > direct veto")

    print("\n5. Koalicja Sił:")
    print("   → Synergia między aktorami")
    print("   → Najwyższa efektywność")
    print("   → Realistyczny model 'przekonywania'")

    print("\n" + "="*70)
    print("ZASTOSOWANIA:")
    print("="*70)
    print("✓ Modelowanie real-world influence networks")
    print("✓ Analiza power dynamics w parlamentach")
    print("✓ Predykcja sukcesu ustaw na podstawie coalitions")
    print("✓ Symulacje lobbying campaigns")
    print("✓ Badanie roli party discipline vs. ideology")
    print("="*70 + "\n")


if __name__ == "__main__":
    showcase_advanced_actors()
