#!/usr/bin/env python3
"""
04 - STRATEGIE AGREGACJI: Jak warstwy łączą się w decyzję
==========================================================

PolicyFlux oferuje 4 strategie agregacji warstw decyzyjnych:

1. SEQUENTIAL (domyślna):
   - Każda warstwa modyfikuje wynik poprzedniej sekwencyjnie
   - base_prob przekazywane w kontekście
   - Łańcuch wpływów

2. AVERAGE:
   - Każda warstwa działa niezależnie
   - Wynik = średnia arytmetyczna wszystkich warstw
   - Równy wpływ wszystkich czynników

3. WEIGHTED:
   - Ważona suma wyników warstw
   - Możliwość priorytetyzacji ważniejszych czynników
   - Wagi muszą sumować się do 1.0

4. MULTIPLICATIVE:
   - Wyniki mnożone przez siebie
   - Efekt "veta" - niska wartość z dowolnej warstwy obniża wynik
   - Wszystkie czynniki muszą "zagrać" razem

Ten przykład porównuje te strategie w identycznych warunkach.

Uruchom: python examples/04_aggregation_strategies.py
"""

from policyflux.integration import (
    IntegrationConfig,
    LayerConfig,
    AdvancedActorsConfig,
    build_engine,
)
from policyflux.utils.reports import craft_a_bar

SEED = 2024
NUM_ACTORS = 100
POLICY_DIM = 2
ITERATIONS = 200

# Wspólna konfiguracja warstw dla wszystkich strategii
BASE_LAYER_CONFIG = LayerConfig(
    include_ideal_point=True,
    include_public_opinion=True,
    include_lobbying=True,
    include_media_pressure=True,
    include_party_discipline=True,
    public_support=0.60,
    lobbying_intensity=0.40,
    media_pressure=0.35,
    party_line_support=0.65,
    party_discipline_strength=0.55,
)

BASE_ACTORS_CONFIG = AdvancedActorsConfig(
    n_lobbyists=4,
    lobbyist_strength=0.55,
    lobbyist_stance=0.75,
    n_whips=2,
    whip_discipline_strength=0.60,
    whip_party_line_support=0.68,
    speaker_agenda_support=0.58,
    president_approval_rating=0.57,
)


def run_strategy(strategy_name: str, aggregation_strategy: str, weights=None):
    """Helper do uruchomienia jednej strategii."""

    print(f"\n{'='*70}")
    print(f"{strategy_name}")
    print('='*70)

    config = IntegrationConfig(
        num_actors=NUM_ACTORS,
        policy_dim=POLICY_DIM,
        iterations=ITERATIONS,
        seed=SEED,  # Ten sam seed dla porównywalności!
        description=strategy_name,
        aggregation_strategy=aggregation_strategy,
        aggregation_weights=weights,
        layer_config=BASE_LAYER_CONFIG,
        actors_config=BASE_ACTORS_CONFIG,
    )

    engine = build_engine(config)
    engine.run_simulation()

    total = len(engine.congress_model.congressmen)
    avg_votes = sum(engine.results) / len(engine.results)
    pass_rate = sum(1 for v in engine.results if v > total / 2) / len(engine.results)

    print(f"\nWynik: {avg_votes:.1f}/{total} głosów ZA")
    print(f"Pass rate: {pass_rate:.1%}")

    # Przykładowe głosowania
    print("\nPrzykładowe wyniki (pierwsze 5 głosowań):")
    for i, votes in enumerate(engine.results[:5], 1):
        status = "✓ PASS" if votes > total / 2 else "✗ FAIL"
        print(f"  Głosowanie #{i}: {votes}/{total} {status}")

    return {
        "name": strategy_name,
        "short_name": aggregation_strategy.upper(),
        "avg_votes": avg_votes,
        "pass_rate": pass_rate,
        "total": total,
    }


def compare_aggregation_strategies():
    """Porównanie czterech strategii agregacji."""

    print("\n" + "="*70)
    print("04 - STRATEGIE AGREGACJI: Porównanie strategii łączenia warstw")
    print("="*70)
    print("\nWARUNKI POCZĄTKOWE (identyczne dla wszystkich strategii):")
    print("  • Liczba posłów: 100")
    print("  • Wymiary: 2D")
    print("  • Iteracje: 200")
    print("  • Warstwy: Ideal Point, Public Opinion, Lobbying, Media, Party")
    print("  • Ten sam seed dla porównywalności!")

    results = []

    # STRATEGIA 1: SEQUENTIAL
    print("\n" + "▼"*70)
    print("STRATEGIA 1: SEQUENTIAL (domyślna)")
    print("▼"*70)
    print("\nDziałanie:")
    print("  1. Ideal Point → base_prob")
    print("  2. Public Opinion(base_prob) → new_prob")
    print("  3. Lobbying(new_prob) → newer_prob")
    print("  4. Media(newer_prob) → ...")
    print("  5. Party(...) → final_prob")
    print("\nZalety:")
    print("  + Kolejność ma znaczenie - modeluje rzeczywiste procesy")
    print("  + Naturalny przepływ wpływów")
    print("  + Ostatnie warstwy mają większy wpływ")

    results.append(run_strategy(
        "SEQUENTIAL - łańcuch wpływów",
        "sequential",
    ))

    # STRATEGIA 2: AVERAGE
    print("\n" + "▼"*70)
    print("STRATEGIA 2: AVERAGE")
    print("▼"*70)
    print("\nDziałanie:")
    print("  final_prob = (prob1 + prob2 + prob3 + prob4 + prob5) / 5")
    print("\nZalety:")
    print("  + Równy wpływ wszystkich warstw")
    print("  + Prosta interpretacja")
    print("  + Brak dominacji jednej warstwy")

    results.append(run_strategy(
        "AVERAGE - równy wpływ",
        "average",
    ))

    # STRATEGIA 3: WEIGHTED
    print("\n" + "▼"*70)
    print("STRATEGIA 3: WEIGHTED")
    print("▼"*70)
    print("\nDziałanie:")
    print("  Wagi: Ideal(0.25), Public(0.15), Lobby(0.15), Media(0.15), Party(0.30)")
    print("  final_prob = 0.25*prob1 + 0.15*prob2 + 0.15*prob3 + 0.15*prob4 + 0.30*prob5")
    print("\nZalety:")
    print("  + Kontrola ważności każdego czynnika")
    print("  + Można priorytetyzować kluczowe warstwy")
    print("  + Elastyczność w modelowaniu")

    results.append(run_strategy(
        "WEIGHTED - priorytetyzacja",
        "weighted",
        weights=[0.25, 0.15, 0.15, 0.15, 0.30],  # Ideal, Public, Lobby, Media, Party
    ))

    # STRATEGIA 4: MULTIPLICATIVE
    print("\n" + "▼"*70)
    print("STRATEGIA 4: MULTIPLICATIVE")
    print("▼"*70)
    print("\nDziałanie:")
    print("  final_prob = prob1 * prob2 * prob3 * prob4 * prob5")
    print("\nZalety:")
    print("  + Efekt 'veta' - wszystkie muszą być wysokie")
    print("  + Niska wartość z jednej warstwy obniża całość")
    print("  + Modeluje sytuacje 'wszystko albo nic'")
    print("\nUwaga:")
    print("  ! Może dawać bardzo niskie wartości (mnożenie < 1)")
    print("  ! Najsurowsza strategia")

    results.append(run_strategy(
        "MULTIPLICATIVE - efekt veta",
        "multiplicative",
    ))

    # PORÓWNANIE
    print("\n" + "="*70)
    print("PORÓWNANIE WYNIKÓW:")
    print("="*70)

    print("\nPass Rate:")
    for r in results:
        bar = "█" * int(r["pass_rate"] * 60)
        print(f"  {r['short_name']:15s} | {r['pass_rate']:5.1%} {bar}")

    print("\nŚrednie głosy ZA:")
    for r in results:
        bar = "█" * int((r["avg_votes"] / r["total"]) * 60)
        pct = r["avg_votes"] / r["total"]
        print(f"  {r['short_name']:15s} | {r['avg_votes']:5.1f}/{r['total']} ({pct:.1%}) {bar}")

    # Różnice względem SEQUENTIAL
    baseline = results[0]["pass_rate"]
    print(f"\nRóżnice względem SEQUENTIAL (baseline: {baseline:.1%}):")
    for r in results[1:]:
        diff = r["pass_rate"] - baseline
        sign = "+" if diff > 0 else ""
        print(f"  {r['short_name']:15s} | {sign}{diff:+.1%}")

    # Wykres
    try:
        craft_a_bar(
            data=[r["pass_rate"] * 100 for r in results],
            labels=[r["short_name"] for r in results],
            title="Porównanie strategii agregacji - Pass Rate",
            xlabel="Strategia",
            ylabel="Pass Rate [%]",
        )
    except Exception as e:
        print(f"\n(Wykres niedostępny: {e})")

    # ANALIZA
    print("\n" + "="*70)
    print("ANALIZA I REKOMENDACJE:")
    print("="*70)

    highest = max(results, key=lambda r: r["pass_rate"])
    lowest = min(results, key=lambda r: r["pass_rate"])

    print(f"\n✓ Najwyższy pass rate: {highest['short_name']} ({highest['pass_rate']:.1%})")
    print(f"✗ Najniższy pass rate: {lowest['short_name']} ({lowest['pass_rate']:.1%})")

    print("\nKiedy używać której strategii:")
    print("\n1. SEQUENTIAL:")
    print("   → Gdy kolejność wpływów ma znaczenie")
    print("   → Modelowanie rzeczywistych procesów decyzyjnych")
    print("   → DEFAULT - dobra dla większości przypadków")

    print("\n2. AVERAGE:")
    print("   → Gdy wszystkie czynniki równie ważne")
    print("   → Prosta interpretacja wyników")
    print("   → Brak priorytetów między warstwami")

    print("\n3. WEIGHTED:")
    print("   → Gdy znasz względną ważność czynników")
    print("   → Potrzebujesz kontroli nad wpływem warstw")
    print("   → Eksperymenty z różnymi konfiguracjami")

    print("\n4. MULTIPLICATIVE:")
    print("   → Efekt 'veta' - wszystkie warstwy muszą zagrać")
    print("   → Sytuacje high-stakes (np. constitutional amendments)")
    print("   → Kiedy WSZYSTKIE czynniki są krytyczne")

    print("\n" + "="*70)
    print("KLUCZOWE WNIOSKI:")
    print("="*70)
    print("✓ Strategia agregacji znacząco wpływa na wyniki")
    print("✓ SEQUENTIAL: dobra równowaga, naturalny przepływ")
    print("✓ AVERAGE: demokratyczna - wszystkie równe")
    print("✓ WEIGHTED: kontrolowana - możesz priorytetyzować")
    print("✓ MULTIPLICATIVE: rygorystyczna - efekt veta")
    print("✓ Wybór strategii zależy od modelowanego scenariusza")
    print("="*70 + "\n")


if __name__ == "__main__":
    compare_aggregation_strategies()
