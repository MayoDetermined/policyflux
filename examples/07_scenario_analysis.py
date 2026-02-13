#!/usr/bin/env python3
"""
07 - ANALIZA SCENARIUSZY: What-If Analysis
==========================================

Ten przykład pokazuje praktyczne zastosowanie PolicyFlux do analizy
scenariuszy "co by było, gdyby...?" (what-if analysis).

SCENARIUSZ: Kontrowersyjna ustawa healthcare reform

Testujemy 6 scenariuszy:
1. STATUS QUO: Obecny stan polityczny
2. SKANDAL: Prezydent traci poparcie po skandalu
3. GRASSROOTS: Mobilizacja oddolna zwiększa poparcie publiczne
4. LOBBY ATTACK: Potężne lobby farmaceutyczne atakuje ustawę
5. PARTY REVOLT: Bunty w partii, słabsza dyscyplina
6. COMPROMISE: Negocjowany kompromis wszystkich stron

Odpowiadamy na pytania:
- Czy ustawa przejdzie w każdym scenariuszu?
- Jaki jest margines zwycięstwa/porażki?
- Które czynniki są kluczowe?
- Jak zmiany wpływają na wynik?

Uruchom: python examples/07_scenario_analysis.py
"""

from dataclasses import dataclass
from typing import List

from policyflux.integration import (
    IntegrationConfig,
    LayerConfig,
    AdvancedActorsConfig,
    build_engine,
)
from policyflux.utils.reports import craft_a_bar

SEED = 31415
NUM_ACTORS = 100
POLICY_DIM = 3  # Healthcare, Economics, Social
ITERATIONS = 250


@dataclass
class Scenario:
    """Pojedynczy scenariusz do analizy."""
    name: str
    short_name: str
    description: str
    config: IntegrationConfig


def run_scenario(scenario: Scenario) -> dict:
    """Uruchom scenariusz i zbierz wyniki."""

    print(f"\n{'▼'*70}")
    print(f"SCENARIUSZ: {scenario.name}")
    print('▼'*70)
    print(f"{scenario.description}")

    engine = build_engine(scenario.config)
    engine.run_simulation()

    total = len(engine.congress_model.congressmen)
    votes_for = [v for v in engine.results]
    avg_votes = sum(votes_for) / len(votes_for)
    pass_rate = sum(1 for v in votes_for if v > total / 2) / len(votes_for)
    avg_margin = avg_votes - total / 2

    # Statystyki
    min_votes = min(votes_for)
    max_votes = max(votes_for)

    print(f"\nWYNIKI:")
    print(f"  Pass rate: {pass_rate:.1%}")
    print(f"  Średnie głosy ZA: {avg_votes:.1f}/{total}")
    print(f"  Średni margines: {avg_margin:+.1f} głosów")
    print(f"  Zakres: {min_votes}-{max_votes} głosów ZA")
    print(f"  Status: {'✓ USTAWA PRZECHODZI' if pass_rate > 0.5 else '✗ USTAWA PADA'}")

    return {
        "name": scenario.name,
        "short_name": scenario.short_name,
        "pass_rate": pass_rate,
        "avg_votes": avg_votes,
        "avg_margin": avg_margin,
        "min_votes": min_votes,
        "max_votes": max_votes,
        "total": total,
    }


def build_scenarios() -> List[Scenario]:
    """Zbuduj wszystkie scenariusze do porównania."""

    scenarios = []

    # SCENARIUSZ 1: STATUS QUO
    scenarios.append(Scenario(
        name="Status Quo",
        short_name="STATUS QUO",
        description=(
            "Obecny stan polityczny:\n"
            "  • Umiarkowane poparcie publiczne (58%)\n"
            "  • Standardowy lobbing\n"
            "  • Typowa dyscyplina partyjna\n"
            "  • Prezydent z przeciętnym poparciem"
        ),
        config=IntegrationConfig(
            num_actors=NUM_ACTORS,
            policy_dim=POLICY_DIM,
            iterations=ITERATIONS,
            seed=SEED,
            description="Status Quo",
            layer_config=LayerConfig(
                include_ideal_point=True,
                include_public_opinion=True,
                include_lobbying=True,
                include_media_pressure=True,
                include_party_discipline=True,
                public_support=0.58,
                lobbying_intensity=0.35,
                media_pressure=0.30,
                party_line_support=0.62,
                party_discipline_strength=0.55,
            ),
            actors_config=AdvancedActorsConfig(
                n_lobbyists=4,
                lobbyist_strength=0.50,
                lobbyist_stance=0.60,
                n_whips=2,
                whip_discipline_strength=0.60,
                whip_party_line_support=0.65,
                speaker_agenda_support=0.58,
                president_approval_rating=0.55,
            ),
        ),
    ))

    # SCENARIUSZ 2: SKANDAL PREZYDENTA
    scenarios.append(Scenario(
        name="Skandal Prezydenta",
        short_name="SKANDAL",
        description=(
            "Prezydent traci poparcie po skandalu:\n"
            "  • Approval rating spada do 32% (!)\n"
            "  • Opinia publiczna odwraca się (45%)\n"
            "  • Media krytyczne\n"
            "  • Słabsza kontrola agendy"
        ),
        config=IntegrationConfig(
            num_actors=NUM_ACTORS,
            policy_dim=POLICY_DIM,
            iterations=ITERATIONS,
            seed=SEED + 1,
            description="Skandal",
            layer_config=LayerConfig(
                include_ideal_point=True,
                include_public_opinion=True,
                include_lobbying=True,
                include_media_pressure=True,
                include_party_discipline=True,
                public_support=0.45,  # ↓
                lobbying_intensity=0.35,
                media_pressure=-0.20,  # Negatywny!
                party_line_support=0.50,  # ↓
                party_discipline_strength=0.45,  # ↓
            ),
            actors_config=AdvancedActorsConfig(
                n_lobbyists=4,
                lobbyist_strength=0.50,
                lobbyist_stance=0.40,  # ↓
                n_whips=2,
                whip_discipline_strength=0.50,  # ↓
                whip_party_line_support=0.48,  # ↓
                speaker_agenda_support=0.45,  # ↓
                president_approval_rating=0.32,  # ↓↓↓
            ),
        ),
    ))

    # SCENARIUSZ 3: GRASSROOTS MOBILIZATION
    scenarios.append(Scenario(
        name="Mobilizacja Oddolna",
        short_name="GRASSROOTS",
        description=(
            "Masowa kampania oddolna za ustawą:\n"
            "  • Poparcie publiczne wzrasta do 78% (!)\n"
            "  • Media pozytywne\n"
            "  • Presja na posłów 'od dołu'\n"
            "  • Słabszy wpływ lobbingu korporacyjnego"
        ),
        config=IntegrationConfig(
            num_actors=NUM_ACTORS,
            policy_dim=POLICY_DIM,
            iterations=ITERATIONS,
            seed=SEED + 2,
            description="Grassroots",
            layer_config=LayerConfig(
                include_ideal_point=True,
                include_public_opinion=True,
                include_lobbying=True,
                include_media_pressure=True,
                include_party_discipline=True,
                public_support=0.78,  # ↑↑
                lobbying_intensity=0.25,  # ↓
                media_pressure=0.55,  # ↑
                party_line_support=0.70,  # ↑
                party_discipline_strength=0.55,
            ),
            actors_config=AdvancedActorsConfig(
                n_lobbyists=4,
                lobbyist_strength=0.35,  # ↓
                lobbyist_stance=0.50,
                n_whips=2,
                whip_discipline_strength=0.60,
                whip_party_line_support=0.72,  # ↑
                speaker_agenda_support=0.68,  # ↑
                president_approval_rating=0.65,  # ↑
            ),
        ),
    ))

    # SCENARIUSZ 4: LOBBY ATTACK
    scenarios.append(Scenario(
        name="Atak Lobby Farmaceutycznego",
        short_name="LOBBY ATTACK",
        description=(
            "Potężne lobby farmaceutyczne ofensywa:\n"
            "  • 10 lobbyistów (zwykle 4)\n"
            "  • Strength 0.85 (bardzo silni)\n"
            "  • Maksymalnie anty-ustawa (stance -1.0)\n"
            "  • Masywne wydatki na kampanię"
        ),
        config=IntegrationConfig(
            num_actors=NUM_ACTORS,
            policy_dim=POLICY_DIM,
            iterations=ITERATIONS,
            seed=SEED + 3,
            description="Lobby Attack",
            layer_config=LayerConfig(
                include_ideal_point=True,
                include_public_opinion=True,
                include_lobbying=True,
                include_media_pressure=True,
                include_party_discipline=True,
                public_support=0.58,
                lobbying_intensity=0.75,  # ↑↑
                media_pressure=0.20,  # ↓ (lobby kontroluje narrację)
                party_line_support=0.55,  # ↓
                party_discipline_strength=0.50,  # ↓
            ),
            actors_config=AdvancedActorsConfig(
                n_lobbyists=10,  # ↑↑↑
                lobbyist_strength=0.85,  # ↑↑
                lobbyist_stance=-1.0,  # Maksymalnie przeciw!
                n_whips=2,
                whip_discipline_strength=0.55,
                whip_party_line_support=0.58,
                speaker_agenda_support=0.50,  # ↓
                president_approval_rating=0.55,
            ),
        ),
    ))

    # SCENARIUSZ 5: PARTY REVOLT
    scenarios.append(Scenario(
        name="Bunt w Partii",
        short_name="REVOLT",
        description=(
            "Posłowie buntują się przeciw linii partii:\n"
            "  • Discipline strength spada do 25%\n"
            "  • Party line support tylko 35%\n"
            "  • Whips tracą kontrolę\n"
            "  • Chaotyczne głosowanie"
        ),
        config=IntegrationConfig(
            num_actors=NUM_ACTORS,
            policy_dim=POLICY_DIM,
            iterations=ITERATIONS,
            seed=SEED + 4,
            description="Party Revolt",
            layer_config=LayerConfig(
                include_ideal_point=True,
                include_public_opinion=True,
                include_lobbying=True,
                include_media_pressure=True,
                include_party_discipline=True,
                public_support=0.58,
                lobbying_intensity=0.35,
                media_pressure=0.30,
                party_line_support=0.35,  # ↓↓
                party_discipline_strength=0.25,  # ↓↓
            ),
            actors_config=AdvancedActorsConfig(
                n_lobbyists=4,
                lobbyist_strength=0.50,
                lobbyist_stance=0.60,
                n_whips=1,  # ↓ (jeden rezygnuje!)
                whip_discipline_strength=0.30,  # ↓↓
                whip_party_line_support=0.40,  # ↓↓
                speaker_agenda_support=0.45,  # ↓
                president_approval_rating=0.55,
            ),
        ),
    ))

    # SCENARIUSZ 6: COMPROMISE
    scenarios.append(Scenario(
        name="Kompromis Negocjowany",
        short_name="COMPROMISE",
        description=(
            "Wszystkie strony negocjują kompromis:\n"
            "  • Wysokie poparcie publiczne (70%)\n"
            "  • Lobbyści zadowoleni (stance 0.8)\n"
            "  • Silna dyscyplina (consensus)\n"
            "  • Media pozytywne\n"
            "  • Win-win scenario"
        ),
        config=IntegrationConfig(
            num_actors=NUM_ACTORS,
            policy_dim=POLICY_DIM,
            iterations=ITERATIONS,
            seed=SEED + 5,
            description="Compromise",
            layer_config=LayerConfig(
                include_ideal_point=True,
                include_public_opinion=True,
                include_lobbying=True,
                include_media_pressure=True,
                include_party_discipline=True,
                public_support=0.70,  # ↑
                lobbying_intensity=0.40,
                media_pressure=0.45,  # ↑
                party_line_support=0.75,  # ↑
                party_discipline_strength=0.70,  # ↑
            ),
            actors_config=AdvancedActorsConfig(
                n_lobbyists=4,
                lobbyist_strength=0.50,
                lobbyist_stance=0.80,  # ↑ (zadowoleni z kompromisu)
                n_whips=3,  # ↑ (więcej wsparcia)
                whip_discipline_strength=0.70,  # ↑
                whip_party_line_support=0.78,  # ↑
                speaker_agenda_support=0.70,  # ↑
                president_approval_rating=0.68,  # ↑
            ),
        ),
    ))

    return scenarios


def analyze_scenarios():
    """Główna funkcja analizy scenariuszy."""

    print("\n" + "="*70)
    print("07 - ANALIZA SCENARIUSZY: What-If Analysis")
    print("="*70)
    print("\nKONTEKST: Kontrowersyjna ustawa healthcare reform")
    print("PYTANIE: Jak różne zmiany polityczne wpłyną na jej przejście?")
    print("="*70)

    scenarios = build_scenarios()
    results = [run_scenario(s) for s in scenarios]

    # PORÓWNANIE
    print("\n" + "="*70)
    print("PORÓWNAWCZE ZESTAWIENIE:")
    print("="*70)

    print("\nPass Rate (prawdopodobieństwo przejścia):")
    for r in results:
        bar = "█" * int(r["pass_rate"] * 50)
        status = "✓" if r["pass_rate"] > 0.5 else "✗"
        print(f"  {status} {r['short_name']:15s} | {r['pass_rate']:5.1%} {bar}")

    print("\nŚredni margines (+/- głosów względem 50%):")
    for r in results:
        margin = r["avg_margin"]
        bar_len = int(abs(margin) / 2)
        bar = "█" * bar_len
        side = ">" if margin > 0 else "<"
        print(f"    {r['short_name']:15s} | {margin:+5.1f} {side} {bar}")

    # Wykres
    try:
        craft_a_bar(
            data=[r["pass_rate"] * 100 for r in results],
            labels=[r["short_name"] for r in results],
            title="Porównanie scenariuszy - Pass Rate",
            xlabel="Scenariusz",
            ylabel="Pass Rate [%]",
        )
    except Exception as e:
        print(f"\n(Wykres niedostępny: {e})")

    # ANALIZA
    print("\n" + "="*70)
    print("ANALIZA WYNIKÓW:")
    print("="*70)

    # Ranking
    ranked = sorted(results, key=lambda r: r["pass_rate"], reverse=True)

    print("\nRanking (od najbardziej do najmniej korzystnego):")
    for i, r in enumerate(ranked, 1):
        status = "PRZECHODZI" if r["pass_rate"] > 0.5 else "PADA"
        print(f"  {i}. {r['short_name']:15s} | {r['pass_rate']:5.1%} | {status}")

    # Kluczowe czynniki
    best = ranked[0]
    worst = ranked[-1]

    print(f"\n✓ NAJLEPSZY SCENARIUSZ: {best['short_name']}")
    print(f"  Pass rate: {best['pass_rate']:.1%}")
    print(f"  Średni margines: +{best['avg_margin']:.1f} głosów")

    print(f"\n✗ NAJGORSZY SCENARIUSZ: {worst['short_name']}")
    print(f"  Pass rate: {worst['pass_rate']:.1%}")
    print(f"  Średni margines: {worst['avg_margin']:.1f} głosów")

    delta = best['pass_rate'] - worst['pass_rate']
    print(f"\nRóżnica: {delta:.1%} punktów procentowych")

    # Insights
    print("\n" + "="*70)
    print("KLUCZOWE WNIOSKI:")
    print("="*70)

    print("\n1. Opinia publiczna ma ogromne znaczenie:")
    print("   → Grassroots mobilization (+20%) może przechylić szalę")
    print("   → Skandal prezydenta (-13%) osłabia wsparcie")

    print("\n2. Lobby ma destrukcyjną moc:")
    print("   → Skoordynowany atak lobby może zablokować ustawę")
    print("   → Nawet przy umiarkowanym poparciu publicznym")

    print("\n3. Dyscyplina partyjna stabilizuje:")
    print("   → Party revolt prowadzi do nieprzewidywalności")
    print("   → Silne whips zwiększają pass rate")

    print("\n4. Kompromis wygrywa:")
    print("   → Negocjowanie satysfakcjonujące wszystkich")
    print("   → Najwyższy pass rate i margines")

    print("\n5. Status quo to niepewność:")
    print("   → Bez zmian - wynik na krawędzi")
    print("   → Małe zmiany mogą przechylić wynik")

    print("\n" + "="*70)
    print("REKOMENDACJE STRATEGICZNE:")
    print("="*70)

    print("\nDla zwolenników ustawy:")
    print("  1. Mobilizuj grassroots - zbuduj poparcie publiczne")
    print("  2. Neutralizuj lobby - negocjuj kompromis")
    print("  3. Umocnij dyscyplinę - przekonaj whips i leadership")
    print("  4. Unikaj skandali - stabilność approval ratings")

    print("\nDla przeciwników ustawy:")
    print("  1. Intensywny lobbing - maksymalizuj pressure")
    print("  2. Podkop approval prezydenta - osłab wsparcie")
    print("  3. Zachęcaj do party revolt - rozbij consensus")
    print("  4. Kontroluj narrację medialną - kształtuj opinię")

    print("\n" + "="*70)
    print("ZASTOSOWANIA W PRAKTYCE:")
    print("="*70)
    print("✓ Planowanie strategii legislacyjnych")
    print("✓ Ocena szans powodzenia kontrowersyjnych ustaw")
    print("✓ Identyfikacja kluczowych pressure points")
    print("✓ Symulacja wpływu zmian politycznych")
    print("✓ Risk analysis dla stakeholders")
    print("="*70 + "\n")


if __name__ == "__main__":
    analyze_scenarios()
