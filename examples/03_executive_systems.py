#!/usr/bin/env python3
"""
03 - SYSTEMY EGZEKUTYWNE: Prezydencki, Parlamentarny, Półprezydencki
=====================================================================

Ten przykład porównuje trzy fundamentalnie różne systemy polityczne:

1. PREZYDENCKI (USA-style):
   - Silny prezydent z prawem veta
   - Separacja władz (executive vs. legislative)
   - Veto-override wymaga superwiększości (2/3)

2. PARLAMENTARNY (UK/Kanada-style):
   - Premier Minister z kontrolą agendy
   - Silna dyscyplina partyjna
   - Głosowania nad zaufaniem mogą obalić rząd
   - Ustawy rządowe vs. ustawy posłów prywatnych

3. PÓŁPREZYDENCKI (Francja/Polska-style):
   - Zarówno Prezydent, jak i Premier Minister
   - Dzielona władza wykonawcza
   - Możliwość kohabitacji (różne partie)
   - Prezydent może wetować, PM kontroluje agendę

Uruchom: python examples/03_executive_systems.py
"""

from policyflux.integration import (
    create_presidential_config,
    create_parliamentary_config,
    create_semi_presidential_config,
    build_engine,
)
from policyflux.utils.reports import craft_a_bar

SEED = 7890
NUM_ACTORS = 100
POLICY_DIM = 3  # Left-Right, Libertarian-Authoritarian, Foreign Policy
ITERATIONS = 200


def run_system(name: str, config):
    """Helper do uruchomienia jednego systemu."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print('='*70)

    engine = build_engine(config)
    engine.run_simulation()

    total = len(engine.congress_model.congressmen)
    avg_votes = sum(engine.results) / len(engine.results)
    pass_rate = sum(1 for v in engine.results if v > total / 2) / len(engine.results)

    print(engine)
    print(f"\nŚrednie głosy ZA: {avg_votes:.1f}/{total}")
    print(f"Pass rate: {pass_rate:.1%}")

    return {
        "name": name,
        "avg_votes": avg_votes,
        "pass_rate": pass_rate,
        "total": total,
    }


def compare_executive_systems():
    """Porównanie trzech systemów politycznych."""

    print("\n" + "="*70)
    print("03 - SYSTEMY EGZEKUTYWNE: Porównanie systemów politycznych")
    print("="*70)

    results = []

    # SYSTEM 1: PREZYDENCKI (USA)
    print("\n" + "▼"*70)
    print("SYSTEM 1: PREZYDENCKI (USA-style)")
    print("▼"*70)
    print("\nCechy:")
    print("  • Prezydent z prawem veta")
    print("  • Veto-override wymaga 2/3 głosów")
    print("  • Separacja władz - słaba kontrola agendy")
    print("  • Wpływ przez approval rating i media")

    presidential_config = create_presidential_config(
        num_actors=NUM_ACTORS,
        policy_dim=POLICY_DIM,
        iterations=ITERATIONS,
        seed=SEED,
        description="System prezydencki (USA)",
        president_approval_rating=0.58,  # Umiarkowanie popularny prezydent
        veto_override_threshold=0.67,  # 2/3 dla override
        n_lobbyists=6,
        lobbyist_strength=0.55,  # Silny lobbing (USA!)
        lobbyist_stance=0.70,
        n_whips=2,
        whip_discipline_strength=0.45,  # Słabsza dyscyplina (USA!)
        whip_party_line_support=0.50,
        speaker_agenda_support=0.52,
        public_support=0.55,
        lobbying_intensity=0.40,
        media_pressure=0.35,
        party_line_support=0.52,
        party_discipline_strength=0.48,
    )

    results.append(run_system("PREZYDENCKI (USA)", presidential_config))

    # SYSTEM 2: PARLAMENTARNY (UK)
    print("\n" + "▼"*70)
    print("SYSTEM 2: PARLAMENTARNY (UK/Kanada-style)")
    print("▼"*70)
    print("\nCechy:")
    print("  • Premier Minister z silną kontrolą agendy")
    print("  • Bardzo silna dyscyplina partyjna")
    print("  • Ustawy rządowe mają priorytet")
    print("  • Głosowania nad zaufaniem (confidence votes)")
    print("  • Utrata głosowania = utrata władzy")

    parliamentary_config = create_parliamentary_config(
        num_actors=NUM_ACTORS,
        policy_dim=POLICY_DIM,
        iterations=ITERATIONS,
        seed=SEED + 1,
        description="System parlamentarny (UK)",
        pm_party_strength=0.68,  # Silna partia rządząca
        confidence_threshold=0.50,  # Prosta większość do utrzymania rządu
        government_bill_rate=0.75,  # 75% ustaw to ustawy rządowe
        n_lobbyists=3,
        lobbyist_strength=0.35,  # Słabszy lobbing (UK!)
        lobbyist_stance=0.60,
        n_whips=4,  # Więcej biczów!
        whip_discipline_strength=0.85,  # BARDZO silna dyscyplina (UK!)
        whip_party_line_support=0.82,
        speaker_agenda_support=0.72,  # Silna kontrola agendy
        public_support=0.58,
        lobbying_intensity=0.25,
        media_pressure=0.30,
        party_line_support=0.78,
        party_discipline_strength=0.80,
    )

    results.append(run_system("PARLAMENTARNY (UK)", parliamentary_config))

    # SYSTEM 3: PÓŁPREZYDENCKI (Francja/Polska)
    print("\n" + "▼"*70)
    print("SYSTEM 3: PÓŁPREZYDENCKI (Francja/Polska-style)")
    print("▼"*70)
    print("\nCechy:")
    print("  • Zarówno Prezydent (elected), jak i Premier Minister")
    print("  • Prezydent: prawo veta (ograniczone)")
    print("  • PM: kontrola agendy legislacyjnej")
    print("  • Możliwość kohabitacji (różne partie)")
    print("  • Dzielona władza wykonawcza")

    semi_presidential_config = create_semi_presidential_config(
        num_actors=NUM_ACTORS,
        policy_dim=POLICY_DIM,
        iterations=ITERATIONS,
        seed=SEED + 2,
        description="System półprezydencki (Francja/Polska)",
        semi_president_approval=0.52,  # Prezydent z umiarkowanym poparciem
        semi_pm_party_strength=0.55,  # PM z mniejszością
        veto_override_threshold=0.60,  # Niższy próg niż USA
        n_lobbyists=4,
        lobbyist_strength=0.48,
        lobbyist_stance=0.65,
        n_whips=3,
        whip_discipline_strength=0.65,  # Średnia dyscyplina
        whip_party_line_support=0.62,
        speaker_agenda_support=0.60,
        public_support=0.56,
        lobbying_intensity=0.32,
        media_pressure=0.38,
        party_line_support=0.64,
        party_discipline_strength=0.62,
    )

    results.append(run_system("PÓŁPREZYDENCKI (FR/PL)", semi_presidential_config))

    # PORÓWNANIE GRAFICZNE
    print("\n" + "="*70)
    print("PORÓWNANIE SYSTEMÓW:")
    print("="*70)

    print("\nPass Rate (skuteczność przechodzenia ustaw):")
    for r in results:
        bar = "█" * int(r["pass_rate"] * 50)
        print(f"  {r['name']:25s} | {r['pass_rate']:5.1%} {bar}")

    print("\nŚrednie głosy ZA:")
    for r in results:
        bar = "█" * int((r["avg_votes"] / r["total"]) * 50)
        print(f"  {r['name']:25s} | {r['avg_votes']:5.1f}/{r['total']} {bar}")

    # Wykres
    try:
        craft_a_bar(
            data=[r["pass_rate"] * 100 for r in results],
            labels=[r["name"] for r in results],
            title="Porównanie skuteczności systemów politycznych",
            xlabel="System",
            ylabel="Pass Rate [%]",
        )
    except Exception as e:
        print(f"\n(Wykres niedostępny: {e})")

    # ANALIZA
    print("\n" + "="*70)
    print("ANALIZA RÓŻNIC:")
    print("="*70)

    # Znajdź najefektywniejszy
    most_effective = max(results, key=lambda r: r["pass_rate"])
    least_effective = min(results, key=lambda r: r["pass_rate"])

    print(f"\n✓ Najskuteczniejszy: {most_effective['name']}")
    print(f"  Pass rate: {most_effective['pass_rate']:.1%}")
    print(f"  Powód: {'Silna dyscyplina i kontrola agendy' if 'PARLAMENTARNY' in most_effective['name'] else 'Sprawna separacja władz' if 'PREZYDENCKI' in most_effective['name'] else 'Balans między władzami'}")

    print(f"\n✗ Najmniej skuteczny: {least_effective['name']}")
    print(f"  Pass rate: {least_effective['pass_rate']:.1%}")
    print(f"  Powód: {'Silne weto i separacja' if 'PREZYDENCKI' in least_effective['name'] else 'Słaba koalicja' if 'PARLAMENTARNY' in least_effective['name'] else 'Konflikt władz wykonawczych'}")

    print("\n" + "="*70)
    print("KLUCZOWE WNIOSKI:")
    print("="*70)
    print("✓ Systemy parlamentarne: najwyższa efektywność legislacyjna")
    print("  → Silna dyscyplina + kontrola agendy = skuteczne uchwalanie")
    print("\n✓ Systemy prezydenckie: niższa efektywność, więcej checks & balances")
    print("  → Veto prezydenckie i separacja władz spowalniają proces")
    print("\n✓ Systemy półprezydenckie: balans między skutecznością a kontrolą")
    print("  → Dzielona władza wykonawcza może prowadzić do gridlock")
    print("\n✓ Każdy system ma zalety: efektywność vs. zabezpieczenia")
    print("="*70 + "\n")


if __name__ == "__main__":
    compare_executive_systems()
