#!/usr/bin/env python3
"""
Spektakularne użycie policyflux: porównanie scenariuszy politycznych
Uruchom: python spectacular_simulation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from policyflux.integration import (
    AdvancedActorsConfig,
    IntegrationConfig,
    LayerConfig,
    build_engine,
)
from policyflux.utils.reports import craft_a_bar, bake_a_pie

SEED = 20260124
NUM_ACTORS = 120
POLICY_DIM = 4  # Econ, Social, Foreign, Emotions
ITERATIONS = 400


@dataclass
class Scenario:
    name: str
    description: str
    config: IntegrationConfig


def run_scenario(scenario: Scenario) -> dict:
    engine = build_engine(scenario.config)
    engine.run_simulation()

    total = len(engine.congress_model.congressmen)
    avg_votes_for = sum(engine.results) / len(engine.results)
    pass_rate = sum(1 for votes in engine.results if votes > total / 2) / len(engine.results)
    avg_margin = avg_votes_for - total / 2

    print(f"\n=== {scenario.name} ===")
    print(scenario.description)
    print(engine)
    print(f"Pass rate: {pass_rate:.1%} | Średni margines: {avg_margin:.2f}")

    return {
        "name": scenario.name,
        "avg_votes_for": avg_votes_for,
        "avg_votes_against": total - avg_votes_for,
        "pass_rate": pass_rate,
        "avg_margin": avg_margin,
    }


def build_scenarios() -> List[Scenario]:
    baseline = Scenario(
        name="Baseline: Zbalansowany parlament",
        description="Spokojny ekosystem: umiarkowane naciski i konsensus w środku skali.",
        config=IntegrationConfig(
            num_actors=NUM_ACTORS,
            policy_dim=POLICY_DIM,
            iterations=ITERATIONS,
            seed=SEED,
            description="Baseline",
            aggregation_strategy="average",
            layer_config=LayerConfig(
                include_ideal_point=True,
                include_public_opinion=True,
                include_lobbying=True,
                include_media_pressure=True,
                include_party_discipline=True,
                public_support=0.52,
                lobbying_intensity=0.15,
                media_pressure=0.10,
                party_line_support=0.55,
                party_discipline_strength=0.35,
            ),
            actors_config=AdvancedActorsConfig(
                n_lobbyists=3,
                lobbyist_strength=0.35,
                lobbyist_stance=0.6,
                n_whips=2,
                whip_discipline_strength=0.45,
                whip_party_line_support=0.55,
                speaker_agenda_support=0.52,
                president_approval_rating=0.50,
            ),
        ),
    )

    media_storm = Scenario(
        name="Sztorm medialny: agenda pod reflektorami",
        description="Media dyktują rytm, a opinia publiczna wzmacnia narrację.",
        config=IntegrationConfig(
            num_actors=NUM_ACTORS,
            policy_dim=POLICY_DIM,
            iterations=ITERATIONS,
            seed=SEED + 1,
            description="Media storm",
            aggregation_strategy="sequential",
            layer_config=LayerConfig(
                layer_names=[
                    "ideal_point",
                    "media_pressure",
                    "public_opinion",
                    "lobbying",
                    "party_discipline",
                ],
                layer_overrides={
                    "media_pressure": {"pressure": 0.65},
                    "public_opinion": {"support_level": 0.70},
                    "lobbying": {"intensity": 0.12},
                    "party_discipline": {
                        "party_line_support": 0.60,
                        "discipline_base_strength": 0.40,
                    },
                },
            ),
            actors_config=AdvancedActorsConfig(
                n_lobbyists=2,
                lobbyist_strength=0.25,
                lobbyist_stance=0.3,
                n_whips=2,
                whip_discipline_strength=0.40,
                whip_party_line_support=0.60,
                speaker_agenda_support=0.58,
                president_approval_rating=0.62,
            ),
        ),
    )

    lobbying_blitz = Scenario(
        name="Lobbying Blitz: ofensywa interesów",
        description="Silne lobby forsuje projekt, szukając przewagi w kuluarach.",
        config=IntegrationConfig(
            num_actors=NUM_ACTORS,
            policy_dim=POLICY_DIM,
            iterations=ITERATIONS,
            seed=SEED + 2,
            description="Lobbying blitz",
            aggregation_strategy="multiplicative",
            layer_config=LayerConfig(
                include_ideal_point=True,
                include_public_opinion=True,
                include_lobbying=True,
                include_media_pressure=False,
                include_party_discipline=True,
                public_support=0.48,
                lobbying_intensity=0.55,
                party_line_support=0.52,
                party_discipline_strength=0.30,
            ),
            actors_config=AdvancedActorsConfig(
                n_lobbyists=6,
                lobbyist_strength=0.70,
                lobbyist_stance=1.0,
                n_whips=1,
                whip_discipline_strength=0.35,
                whip_party_line_support=0.50,
                speaker_agenda_support=0.49,
                president_approval_rating=0.47,
            ),
        ),
    )

    party_lockdown = Scenario(
        name="Party Lockdown: żelazna dyscyplina",
        description="Silne bicze partyjne i rygorystyczna linia głosowań.",
        config=IntegrationConfig(
            num_actors=NUM_ACTORS,
            policy_dim=POLICY_DIM,
            iterations=ITERATIONS,
            seed=SEED + 3,
            description="Party lockdown",
            aggregation_strategy="weighted",
            aggregation_weights=[0.15, 0.15, 0.10, 0.10, 0.50],
            layer_config=LayerConfig(
                include_ideal_point=True,
                include_public_opinion=True,
                include_lobbying=True,
                include_media_pressure=True,
                include_party_discipline=True,
                public_support=0.50,
                lobbying_intensity=0.10,
                media_pressure=0.08,
                party_line_support=0.72,
                party_discipline_strength=0.75,
            ),
            actors_config=AdvancedActorsConfig(
                n_lobbyists=2,
                lobbyist_strength=0.30,
                lobbyist_stance=0.4,
                n_whips=4,
                whip_discipline_strength=0.85,
                whip_party_line_support=0.78,
                speaker_agenda_support=0.60,
                president_approval_rating=0.55,
            ),
        ),
    )

    return [baseline, media_storm, lobbying_blitz, party_lockdown]


def run_showcase() -> None:
    print("\nPOLICYFLUX: Spektakularny pokaz scenariuszy\n" + "=" * 50)

    scenarios = build_scenarios()
    results = [run_scenario(s) for s in scenarios]

    craft_a_bar(
        data=[r["pass_rate"] * 100 for r in results],
        labels=[r["name"] for r in results],
        title="Skuteczność przechodzenia ustaw (Pass rate)",
        xlabel="Scenariusz",
        ylabel="Odsetek uchwaleń [%]",
    )

    best = max(results, key=lambda r: r["pass_rate"])
    bake_a_pie(
        data=[best["avg_votes_for"], best["avg_votes_against"]],
        labels=["Za", "Przeciw"],
        title=f"Najsilniejszy scenariusz: {best['name']}",
    )

    print("\nFinał pokazu:")
    print(
        f"Najmocniejszy scenariusz to '{best['name']}', "
        f"z pass rate {best['pass_rate']:.1%}."
    )


if __name__ == "__main__":
    run_showcase()
