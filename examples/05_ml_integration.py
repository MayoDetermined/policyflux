#!/usr/bin/env python3
"""
05 - INTEGRACJA ML: Encodery i Neural Layers
============================================

PolicyFlux oferuje zaawansowaną integrację z Machine Learning (PyTorch):

1. ENCODERY:
   - IdealPointEncoderDF: DataFrame → Przestrzeń polityczna
   - IdealPointTextEncoder: Tekst → Przestrzeń (TF-IDF + PyTorch)

2. NEURAL LAYERS:
   - SequentialNeuralLayer: Warstwy neuronowe jako layer decyzyjny
   - Pełne wsparcie PyTorch (GPU/CPU)
   - Trening i walidacja
   - Konfigurowalna architektura

Ten przykład pokazuje:
- Kodowanie danych tabularycznych do przestrzeni politycznej
- Użycie warstw neuronowych do predykcji głosowań
- Trening i ewaluacja modelu

UWAGA: Ten przykład wymaga PyTorch! Zainstaluj: pip install torch

Uruchom: python examples/05_ml_integration.py
"""

import warnings
warnings.filterwarnings('ignore')

from policyflux.integration import IntegrationConfig, LayerConfig, build_engine


def example_neural_layer_concept():
    """
    Przykład koncepcyjny użycia neural layers.

    UWAGA: To jest DEMONSTRACJA koncepcji. Pełna implementacja wymaga:
    - Przygotowanych danych treningowych
    - Zdefiniowanej architektury sieci
    - Kompilacji i treningu modelu
    """

    print("\n" + "="*70)
    print("05 - INTEGRACJA ML: Neural Layers (Koncepcja)")
    print("="*70)

    print("\n" + "▼"*70)
    print("CZĘŚĆ 1: Czym są Neural Layers?")
    print("▼"*70)

    print("\nNeural Layers to warstwy decyzyjne oparte na sieciach neuronowych.")
    print("Zamiast ręcznych formuł (jak w Ideal Point), sieć uczy się")
    print("przewidywać prawdopodobieństwo głosu ZA na podstawie:")
    print("  • Pozycji ustawy w przestrzeni politycznej")
    print("  • Pozycji posła w przestrzeni politycznej")
    print("  • Kontekstu głosowania (public opinion, media, etc.)")
    print("  • Historycznych danych głosowań")

    print("\n" + "▼"*70)
    print("CZĘŚĆ 2: Jak używać Neural Layers?")
    print("▼"*70)

    print("\nKROK 1: Przygotuj dane treningowe")
    print("  Potrzebujesz historycznych głosowań z features:")
    print("  - bill_position: [dim1, dim2, ...]")
    print("  - congressman_position: [dim1, dim2, ...]")
    print("  - context: {public_support, lobbying_intensity, ...}")
    print("  - vote: 0 (przeciw) lub 1 (za)")

    print("\nKROK 2: Zdefiniuj architekturę")
    print("  Przykład PyTorch Sequential:")
    print("    nn.Sequential(")
    print("      nn.Linear(input_dim, 64),")
    print("      nn.ReLU(),")
    print("      nn.Dropout(0.3),")
    print("      nn.Linear(64, 32),")
    print("      nn.ReLU(),")
    print("      nn.Linear(32, 1),")
    print("      nn.Sigmoid()")
    print("    )")

    print("\nKROK 3: Kompiluj i trenuj")
    print("  neural_layer.compile(")
    print("    optimizer_cls=torch.optim.Adam,")
    print("    lr=0.001,")
    print("    loss_fn=nn.BCELoss(),")
    print("    epochs=50")
    print("  )")
    print("  neural_layer.train(X_train, y_train)")

    print("\nKROK 4: Użyj w symulacji")
    print("  config = IntegrationConfig(")
    print("    ...,")
    print("    layer_config=LayerConfig(")
    print("      include_neural=True,")
    print("      layer_overrides={")
    print("        'neural': {'model': trained_neural_layer}")
    print("      }")
    print("    )")
    print("  )")

    print("\n" + "▼"*70)
    print("CZĘŚĆ 3: Encodery - transformacja danych")
    print("▼"*70)

    print("\nPolicyFlux oferuje encodery do automatycznej transformacji:")

    print("\n1. IdealPointEncoderDF (PyTorch):")
    print("   DataFrame → Przestrzeń polityczna")
    print("   Koduje dane tabelaryczne (voting records, demographics)")
    print("   do N-wymiarowej przestrzeni politycznej")

    print("\n2. IdealPointTextEncoder:")
    print("   Tekst → Przestrzeń polityczna")
    print("   Używa TF-IDF + PyTorch do kodowania tekstów")
    print("   (np. speeches, bills, manifestos)")

    print("\nPrzykład użycia:")
    print("  from policyflux.layers.idealpoint import IdealPointEncoderDF")
    print("  import pandas as pd")
    print()
    print("  # Masz DataFrame z voting records")
    print("  votes_df = pd.DataFrame({")
    print("    'bill_1': [1, 0, 1, 0],")
    print("    'bill_2': [1, 1, 0, 0],")
    print("    'bill_3': [0, 1, 1, 1],")
    print("  })")
    print()
    print("  # Koduj do 2D przestrzeni")
    print("  encoder = IdealPointEncoderDF(input_dim=3, output_dim=2)")
    print("  positions = encoder.encode(votes_df)")
    print("  # → tensor([[0.23, -0.45], [0.12, 0.67], ...])")

    print("\n" + "▼"*70)
    print("CZĘŚĆ 4: Przykład praktyczny (symulowany)")
    print("▼"*70)

    print("\nSymulujemy scenariusz gdzie neural layer nauczył się")
    print("przewidywać głosowania na podstawie historycznych danych...")

    # Symulacja bez neural layer (baseline)
    print("\n[BASELINE] Bez neural layer (tylko tradycyjne warstwy):")

    config_baseline = IntegrationConfig(
        num_actors=60,
        policy_dim=2,
        iterations=100,
        seed=2025,
        description="Baseline - bez ML",
        layer_config=LayerConfig(
            include_ideal_point=True,
            include_public_opinion=True,
            include_party_discipline=True,
            public_support=0.55,
            party_line_support=0.60,
            party_discipline_strength=0.50,
        ),
    )

    engine_baseline = build_engine(config_baseline)
    engine_baseline.run_simulation()

    total = len(engine_baseline.congress_model.congressmen)
    avg_baseline = sum(engine_baseline.results) / len(engine_baseline.results)
    pass_baseline = sum(1 for v in engine_baseline.results if v > total / 2) / len(engine_baseline.results)

    print(f"  Średnie głosy ZA: {avg_baseline:.1f}/{total}")
    print(f"  Pass rate: {pass_baseline:.1%}")

    print("\n" + "="*70)
    print("WNIOSKI I ZASTOSOWANIA:")
    print("="*70)

    print("\n✓ Neural Layers pozwalają na:")
    print("  • Uczenie się z historycznych danych")
    print("  • Wykrywanie nieoczywistych wzorców")
    print("  • Predykcję głosowań z wysoką dokładnością")
    print("  • Adaptację do zmieniających się warunków")

    print("\n✓ Encodery pozwalają na:")
    print("  • Automatyczną transformację danych")
    print("  • Redukcję wymiarowości")
    print("  • Odkrywanie ukrytych struktur (latent dimensions)")

    print("\n✓ Przykładowe zastosowania:")
    print("  • Predykcja wyników głosowań w prawdziwych parlamentach")
    print("  • Analiza voting records do identyfikacji ideologii")
    print("  • Symulacje 'what-if' z nauczonych wzorców")
    print("  • Clustering posłów według behavior patterns")

    print("\n✓ Wymagania:")
    print("  • PyTorch (GPU opcjonalne, ale zalecane)")
    print("  • Dane treningowe (voting records, context)")
    print("  • Znajomość podstaw deep learning")

    print("\n" + "="*70)
    print("PRZYKŁADOWY KOD - Neural Layer Training:")
    print("="*70)

    print("""
# Przykład: trenowanie neural layer

import torch
import torch.nn as nn
from policyflux.layers.neural import SequentialNeuralLayer

# 1. Przygotuj dane
X_train = torch.tensor([
    # [bill_pos_x, bill_pos_y, congress_pos_x, congress_pos_y, public_support, ...]
    [0.5, 0.3, 0.6, 0.4, 0.65, ...],
    [0.2, 0.8, 0.1, 0.7, 0.45, ...],
    # ...
], dtype=torch.float32)

y_train = torch.tensor([1, 0, 1, 1, 0, ...], dtype=torch.float32)  # votes

# 2. Zdefiniuj architekturę
architecture = nn.Sequential(
    nn.Linear(INPUT_DIM, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# 3. Stwórz neural layer
neural_layer = SequentialNeuralLayer(
    layer_id="my_neural_layer",
    layer_name="Learned Voting Patterns"
)
neural_layer.set_nn_architecture(architecture)

# 4. Kompiluj
neural_layer.compile(
    optimizer_cls=torch.optim.Adam,
    lr=0.001,
    loss_fn=nn.BCELoss(),
    epochs=100
)

# 5. Trenuj
neural_layer.train(X_train, y_train)

# 6. Waliduj
X_val = ...
y_val = ...
val_loss, val_acc = neural_layer.run_validation(X_val, y_val)
print(f"Validation Accuracy: {val_acc:.2%}")

# 7. Użyj w symulacji
config = IntegrationConfig(
    ...,
    layer_config=LayerConfig(
        include_neural=True,
        layer_overrides={
            'neural': {'model': neural_layer}
        }
    )
)
    """)

    print("\n" + "="*70)
    print("NASTĘPNE KROKI:")
    print("="*70)
    print("1. Zainstaluj PyTorch: pip install torch")
    print("2. Przygotuj dane treningowe (voting records)")
    print("3. Eksperymentuj z różnymi architekturami")
    print("4. Trenuj modele i oceniaj ich dokładność")
    print("5. Integruj z PolicyFlux do zaawansowanych symulacji")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        example_neural_layer_concept()
    except ImportError as e:
        print("\n" + "!"*70)
        print("BŁĄD: Brak wymaganych bibliotek!")
        print("!"*70)
        print("\nAby uruchomić ten przykład, zainstaluj:")
        print("  pip install torch")
        print("\nAlternatywnie, pomiń ten przykład i przejdź do innych.")
        print("!"*70 + "\n")
