# PolicyFlux

<div align="center">

**Zaawansowana biblioteka do modelowania i symulacji procesów legislacyjnych, zachowań parlamentarnych i dynamiki politycznej**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Development Status](https://img.shields.io/badge/status-early%20development-orange.svg)](https://github.com/yourusername/policyflux)

</div>

---

## ⚠️ Status Projektu

**PolicyFlux znajduje się we wczesnej fazie rozwoju (early stage).** Biblioteka jest funkcjonalna i nadaje się do eksperymentów badawczych, ale API jest niestabilne i może ulegać znaczącym zmianom. Kod powinien być traktowany jako prototyp badawczy, a nie produkcyjna biblioteka.

**Zalecenia:**
- ✅ Eksperymentowanie i badania akademickie
- ✅ Prototypowanie modeli politologicznych
- ⚠️ Oczekuj breaking changes między wersjami
- ❌ Nie stosuj w środowisku produkcyjnym

---

## 📋 Spis treści

- [Opis projektu](#-opis-projektu)
- [Kluczowe cechy](#-kluczowe-cechy)
- [Instalacja](#-instalacja)
- [Szybki start](#-szybki-start)
- [Architektura](#-architektura)
- [Komponenty](#-komponenty)
- [Przykłady użycia](#-przykłady-użycia)
- [Dokumentacja](#-dokumentacja)
- [Rozwój](#-rozwój)
- [Licencja](#-licencja)

---

## 🎯 Opis projektu

PolicyFlux to biblioteka Python do budowy zaawansowanych symulacji procesów legislacyjnych i zachowań parlamentarnych. Umożliwia modelowanie złożonych interakcji między:
- Posłami (congressmen/actors) z ideologicznymi preferencjami
- Projektami ustaw (bills) w wielowymiarowej przestrzeni politycznej
- Warstwami wpływów (lobbying, media, opinia publiczna, dyscyplina partyjna)
- Zaawansowanymi aktorami (Speaker, Whips, Lobbyści, egzekutywa)
- Różnymi systemami politycznymi (prezydencki, parlamentarny, półprezydencki)

Biblioteka została zaprojektowana z myślą o:
- **Badaczach politologii**: Analiza zachowań legislatywnych, symulacje Monte Carlo
- **Data Scientists**: Integracja z ML/PyTorch, neural layers, text encoders
- **Analitykach politycznych**: Scenariusze "what-if", porównywanie systemów
- **Edukacji**: Demonstracje procesów politycznych i systemów wyborczych

---

## 🚀 Kluczowe cechy

### 📊 Modelowanie zachowań legislacyjnych
- **Wielowymiarowa przestrzeń polityczna**: Modeluj dowolną liczbę wymiarów (ekonomia, sprawy społeczne, polityka zagraniczna, etc.)
- **Utility-based voting**: Posłowie głosują na podstawie funkcji użyteczności uwzględniającej dystans ideologiczny
- **Symulacje Monte Carlo**: Deterministyczne i probabilistyczne modele głosowania

### 🎭 Warstwy decyzyjne (Decision Layers)
- **Ideal Point Layer**: Bazowe preferencje ideologiczne
- **Public Opinion Layer**: Wpływ opinii publicznej
- **Lobbying Layer**: Naciski grup interesów
- **Media Pressure Layer**: Wpływ mediów
- **Party Discipline Layer**: Dyscyplina partyjna
- **Government Agenda Layer**: Kontrola agendy przez egzekutywę
- **Neural Layer**: Warstwy neuronowe (PyTorch) jako decision layers

### 🏛️ Zaawansowani aktorzy polityczni
- **Speaker**: Kontrola agendy i scheduling power
- **Party Whips**: Egzekwowanie dyscypliny partyjnej
- **Lobbyści**: Reprezentacja grup interesów
- **President/Prime Minister**: Wpływ egzekutywy, veto, agenda setting

### 🌍 Systemy polityczne
- **Prezydencki** (USA-style): Separacja władz, veto power, veto override
- **Parlamentarny** (UK/Kanada): Kontrola agendy przez PM, votes of confidence
- **Półprezydencki** (Francja/Polska): Kohabitacja, dzielona władza wykonawcza

### 🤖 Integracja z Machine Learning
- **PyTorch support**: Pełne wsparcie dla GPU/CPU
- **Text encoders**: TF-IDF + sentence embeddings → ideal points
- **Neural layers**: Uczenie się wzorców głosowania z danych
- **Custom architectures**: Konfigurowalne sieci neuronowe

### 📈 Strategie agregacji warstw
- **Sequential**: Warstwy modyfikują się sekwencyjnie (domyślne)
- **Average**: Średnia arytmetyczna wszystkich warstw
- **Weighted**: Ważona suma z priorytetyzacją
- **Multiplicative**: Mnożenie wyników (efekt "veta")

---

## 💻 Instalacja

### Wymagania
- Python 3.10 lub nowszy
- pip lub conda

### Podstawowa instalacja

```bash
# Klonowanie repozytorium
git clone https://github.com/yourusername/policyflux.git
cd policyflux

# Instalacja w trybie deweloperskim (editable)
pip install -e .
```

### Instalacja z opcjonalnymi zależnościami

```bash
# PyTorch (dla neural layers)
pip install -e ".[torch]"

# Text encoders (sentence transformers)
pip install -e ".[text-encoders]"

# Narzędzia deweloperskie (pytest, mypy, ruff)
pip install -e ".[dev]"

# Wszystko razem
pip install -e ".[torch,text-encoders,dev]"
```

### Weryfikacja instalacji

```bash
python -c "import policyflux; print(policyflux.__file__)"
```

---

## ⚡ Szybki start

### Najprostsza symulacja (30 sekund)

```python
from policyflux import build_engine, IntegrationConfig, LayerConfig

# Konfiguracja symulacji
config = IntegrationConfig(
    num_actors=50,            # 50 posłów
    policy_dim=2,             # 2D: Left-Right + Liberal-Conservative
    iterations=100,           # 100 głosowań
    seed=12345,               # Deterministyczny RNG
    description="Moja pierwsza symulacja",
    layer_config=LayerConfig(
        include_ideal_point=True,       # Preferencje ideologiczne
        include_public_opinion=True,    # Opinia publiczna
        include_party_discipline=True,  # Dyscyplina partyjna
        public_support=0.60,            # 60% poparcia publicznego
        party_discipline_strength=0.5,  # Średnia dyscyplina
    ),
)

# Buduj silnik i uruchom
engine = build_engine(config)
engine.run_simulation()

# Wyniki
print(engine)
```

### Przykład: Porównanie systemów politycznych

```python
from policyflux import build_engine
from policyflux import create_presidential_config, create_parliamentary_config

# System prezydencki (USA)
prez_config = create_presidential_config(
    num_actors=100,
    policy_dim=2,
    iterations=200,
    seed=42,
    president_approval=0.52,
)
prez_engine = build_engine(prez_config)
prez_engine.run_simulation()

# System parlamentarny (UK)
parl_config = create_parliamentary_config(
    num_actors=100,
    policy_dim=2,
    iterations=200,
    seed=42,
)
parl_engine = build_engine(parl_config)
parl_engine.run_simulation()

# Porównanie
print(f"Prezydencki: {prez_engine.pass_rate:.1%}")
print(f"Parlamentarny: {parl_engine.pass_rate:.1%}")
```

### Przykład: Zaawansowani aktorzy

```python
from policyflux import build_engine, IntegrationConfig, LayerConfig, AdvancedActorsConfig

config = IntegrationConfig(
    num_actors=80,
    policy_dim=2,
    iterations=150,
    seed=999,
    layer_config=LayerConfig(
        include_ideal_point=True,
        include_lobbying=True,
        include_party_discipline=True,
        lobbying_intensity=0.4,
        party_discipline_strength=0.6,
    ),
    actors_config=AdvancedActorsConfig(
        n_lobbyists=5,                    # 5 lobbyistów
        lobbyist_strength=0.5,            # Siła wpływu
        n_whips=3,                        # 3 party whips
        whip_discipline_strength=0.7,     # Dyscyplina
        speaker_agenda_support=0.6,       # Speaker wspiera ustawę
    ),
)

engine = build_engine(config)
engine.run_simulation()
print(engine)
```

---

## 🏗️ Architektura

```
PolicyFlux
│
├── Core Abstractions (policyflux/core/)
│   ├── Actor Templates          # Bazowa reprezentacja posłów
│   ├── Bill Templates           # Projekty ustaw w przestrzeni politycznej
│   ├── Congress Models          # Modele parlamentu
│   ├── Layer Templates          # Abstrakcje warstw decyzyjnych
│   ├── Executive Templates      # Systemy egzekutywy
│   ├── Aggregation Strategies   # Strategie łączenia warstw
│   └── Types & Utilities        # PolicySpace, UtilitySpace, etc.
│
├── Decision Layers (policyflux/layers/)
│   ├── IdealPointLayer          # Preferencje ideologiczne
│   ├── PublicOpinionLayer       # Opinia publiczna
│   ├── LobbyingLayer            # Lobbying
│   ├── MediaPressureLayer       # Media
│   ├── PartyDisciplineLayer     # Dyscyplina partyjna
│   ├── GovernmentAgendaLayer    # Agenda rządowa
│   └── NeuralLayer              # PyTorch neural networks
│
├── Models (policyflux/models/)
│   ├── Sequential Models        # Implementacje sekwencyjne
│   ├── Simulation Engines       # Monte Carlo, Deterministic
│   ├── Executive Systems        # Presidential, Parliamentary, Semi-Presidential
│   └── Advanced Actors          # Speaker, Whips, Lobbyists, President
│
├── Integration (policyflux/integration.py)
│   ├── Config Builders          # IntegrationConfig, LayerConfig
│   ├── Engine Builder           # build_engine()
│   ├── Layer Registry           # Rejestr warstw
│   └── Preset Configs           # Presidential, Parliamentary configs
│
├── Data Processing (policyflux/dprocessing/)
│   └── Text Encoders            # TF-IDF + embeddings → ideal points
│
└── Utilities (policyflux/utils/)
    └── Reports                  # Wykresy, bar charts, pie charts
```

---

## 🧩 Komponenty

### 1. Core (policyflux/core/)

**Podstawowe abstrakcje i szablony:**

- **`simple_actors_template.py`**: `CongressMan` - bazowa klasa posła
- **`complex_actors_template.py`**: `ComplexActor` - posłowie z zaawansowanym zachowaniem
- **`bill_template.py`**: `Bill` - abstrakcja dla projektów ustaw
- **`congress_model_template.py`**: `CongressModel` - abstrakcja dla parlamentu
- **`layer_template.py`**: `Layer` - abstrakcja warstw decyzyjnych
- **`executive.py`**: `ExecutiveActor`, `Executive` - abstrakcje dla egzekutywy
- **`aggregation_strategy.py`**: Strategie łączenia output warstw (Sequential, Average, Weighted, Multiplicative)
- **`types.py`**: Definicje typów (`PolicySpace`, `PolicyVector`, `UtilitySpace`, `PolicyPosition`)
- **`contexts.py`** (NEW): `VotingContext`, `SimulationContext` - immutable konteksty decyzyjne
- **`voting_strategy.py`** (NEW): `VotingStrategy`, `ProbabilisticVoting`, `DeterministicVoting` - strategie głosowania
- **`container.py`** (NEW): `ServiceContainer` - lekkie dependency injection

### 2. Layers (policyflux/layers/)

**Warstwy decyzyjne modyfikujące prawdopodobieństwo głosowania:**

| Warstwa | Plik | Opis |
|---------|------|------|
| **IdealPoint** | `idealpoint.py` | Preferencje ideologiczne, dystans w przestrzeni politycznej |
| **PublicOpinion** | `public_pressure.py` | Wpływ opinii publicznej na głosowanie |
| **Lobbying** | `lobbying.py` | Naciski lobbyistów i grup interesów |
| **MediaPressure** | `media_pressure.py` | Wpływ mediów |
| **PartyDiscipline** | `party.py` | Dyscyplina partyjna i linia partii |
| **GovernmentAgenda** | `government_agenda.py` | Kontrola agendy przez PM (systemy parlamentarne) |
| **Neural** | `neural.py` | PyTorch neural networks jako decision layer |
| **Text Encoders** | `idealpoint.py` | `IdealPointTextEncoder`, `IdealPointEncoderDF` dla text→ideal points |

### 3. Toolbox (policyflux/toolbox/)

**Konkretne implementacje abstrakcji:**

- **`actors.py`**: `SequentialVoter` - głosujący posły z wielowarstwowym podejmowaniem decyzji
- **`bill.py`**: `SequentialBill` - projekty ustaw z pozycją w policy space
- **`congress_model.py`**: `SequentialCongressModel` - model parlamentu z głosowaniem
- **`executive_systems.py`**:
  - `PresidentialExecutive` (model prezydencki z veto)
  - `ParliamentaryExecutive` (model parlamentarny z kontrolą agendy)
  - `SemiPresidentialExecutive` (model półprezydencki)

**Advanced Actors** (policyflux/toolbox/advanced_actors/):
- **`speaker.py`**: `SequentialSpeaker` - kontrola agendy i scheduling power
- **`whips.py`**: `SequentialWhip` - egzekwowanie dyscypliny partyjnej
- **`lobby.py`**: `SequentialLobbyer` - kampanie lobbyingowe
- **`white_house.py`**: `SequentialPresident` - wpływ egzekutywy

### 4. Engines (policyflux/engines/)

**Silniki symulacji:**

- **`engine_template.py`**: `Engine`, `MPEngine` - klasy bazowe
- **`parallel_monte_carlo.py`**: `ParallelMonteCarlo` - wielowątkowe/wieloprocesowe Monte Carlo
- **`deterministic_engine.py`**: `DeterministicEngine` - głosowanie deterministyczne
- **`engine_template.py`**: `Session` - pojedyncza sesja głosowania

### 5. Integration (policyflux/integration/)

**High-level API do budowy symulacji (NOW REFACTORED):**

- **`config.py`**: `IntegrationConfig`, `LayerConfig`, `AdvancedActorsConfig` - klasy konfiguracyjne
- **`builders/engine_builder.py`**: `build_engine()`, `build_session()`, `build_bill()` - fabryki
- **`builders/congress_builder.py`**: `build_congress()` - budowanie parlamentu
- **`builders/layer_builder.py`**: `build_layers()` - budowanie warstw decyzyjnych
- **`builders/actor_builder.py`**: `build_executive()`, `build_advanced_actors()` - budowanie aktorów
- **`presets/president_preset.py`**: `create_presidential_config()` - preset dla systemów prezydenckich
- **`presets/parliament_preset.py`**: `create_parliamentary_config()` - preset dla systemów parlamentarnych
- **`presets/semipresident_preset.py`**: `create_semi_presidential_config()` - preset dla systemów półprezydenckich
- **`registry.py`**: `LAYER_REGISTRY`, `register_layer()` - dynamiczna rejestracja warstw

### 6. Data Processing (policyflux/dprocessing/)

**Przetwarzanie tekstów do przestrzeni politycznej:**

- Text vectorizers dla kodowania tekstów politycznych
- TF-IDF + sentence embeddings dla ekstrakcji ideal points
- Integracja z `sentence-transformers`

### 7. Utils (policyflux/utils/)

**Narzędzia pomocnicze:**

- **`reports/bar_charts.py`**: `craft_a_bar()` - wykresy słupkowe
- **`reports/pie_charts.py`**: `bake_a_pie()` - wykresy kołowe
- RNG management (`pfrandom.py`)
- Logging configuration

---

## 📚 Przykłady użycia

### Przykład 1: Podstawowe głosowanie ideologiczne

```python
from policyflux import build_engine, IntegrationConfig, LayerConfig

config = IntegrationConfig(
    num_actors=50,
    policy_dim=1,  # 1D: Left-Right
    iterations=100,
    seed=12345,
    layer_config=LayerConfig(
        include_ideal_point=True,   # Tylko ideologia
        include_public_opinion=False,
        include_lobbying=False,
        include_media_pressure=False,
        include_party_discipline=False,
        include_government_agenda=False,
    ),
)

engine = build_engine(config)
engine.run_simulation()
print(engine)
```

### Przykład 2: Wpływ opinii publicznej

```python
from policyflux import build_engine, IntegrationConfig, LayerConfig

# Scenariusz 1: Bez poparcia publicznego
config_no_support = IntegrationConfig(
    num_actors=100,
    policy_dim=2,
    iterations=200,
    seed=42,
    layer_config=LayerConfig(
        include_ideal_point=True,
        include_public_opinion=True,
        public_support=0.2,  # Tylko 20% poparcia
    ),
)

# Scenariusz 2: Z silnym poparciem publicznym
config_high_support = IntegrationConfig(
    num_actors=100,
    policy_dim=2,
    iterations=200,
    seed=42,
    layer_config=LayerConfig(
        include_ideal_point=True,
        include_public_opinion=True,
        public_support=0.8,  # 80% poparcia
    ),
)

engine1 = build_engine(config_no_support)
engine1.run_simulation()

engine2 = build_engine(config_high_support)
engine2.run_simulation()

print(f"Bez poparcia: {engine1.pass_rate:.1%}")
print(f"Z poparciem: {engine2.pass_rate:.1%}")
```

### Przykład 3: Wielowarstwowy model decyzyjny

```python
from policyflux import build_engine, IntegrationConfig, LayerConfig

# Wszystkie warstwy decyzyjne
config = IntegrationConfig(
    num_actors=80,
    policy_dim=2,
    iterations=150,
    seed=999,
    layer_config=LayerConfig(
        include_ideal_point=True,
        include_public_opinion=True,
        include_lobbying=True,
        include_media_pressure=True,
        include_party_discipline=True,
        include_government_agenda=False,  # Nie parlamentarny
        public_support=0.55,
        lobbying_intensity=0.3,
        media_pressure=0.4,
        party_discipline_strength=0.6,
    ),
)

engine = build_engine(config)
engine.run_simulation()
print(engine)
```

### Przykład 4: Porównanie systemów politycznych

```python
from policyflux import build_engine
from policyflux import create_presidential_config, create_parliamentary_config

# System prezydencki (USA-style)
prez_config = create_presidential_config(
    num_actors=100,
    policy_dim=2,
    iterations=200,
    seed=42,
    president_approval=0.50,
    veto_override_threshold=2/3,
)

# System parlamentarny (UK-style)
parl_config = create_parliamentary_config(
    num_actors=100,
    policy_dim=2,
    iterations=200,
    seed=42,
    pm_strength=0.75,
    party_discipline=0.7,
)

prez_engine = build_engine(prez_config)
prez_engine.run_simulation()

parl_engine = build_engine(parl_config)
parl_engine.run_simulation()

print(f"System prezydencki: {prez_engine.pass_rate:.1%}")
print(f"System parlamentarny: {parl_engine.pass_rate:.1%}")
```

### Przykład 5: Zaawansowani aktorzy (Whips, Lobbyiści, Speaker)

```python
from policyflux import build_engine, IntegrationConfig, LayerConfig, AdvancedActorsConfig

# Model z lobbyistami i party whips
config = IntegrationConfig(
    num_actors=100,
    policy_dim=2,
    iterations=200,
    seed=2024,
    layer_config=LayerConfig(
        include_ideal_point=True,
        include_lobbying=True,
        include_party_discipline=True,
        lobbying_intensity=0.5,
        party_discipline_strength=0.7,
    ),
    actors_config=AdvancedActorsConfig(
        n_lobbyists=8,                     # 8 organizacji lobbyingowych
        lobbyist_strength=0.6,             # Silny lobbing
        lobbyist_stance=0.85,              # Większość wspiera ustawę
        n_whips=4,                         # 4 party whips
        whip_discipline_strength=0.8,      # Silna dyscyplina
        speaker_agenda_support=0.7,        # Speaker wspiera ustawę
    ),
)

engine = build_engine(config)
engine.run_simulation()
print(engine)
```

### Przykład 6: Integracja z Text Encoders (NLP)

```python
from policyflux.layers import IdealPointTextEncoder

# Korpus wypowiedzi politycznych
corpus = [
    "Musimy zwiększyć podatki dla bogatych, aby finansować programy społeczne",
    "Niskie podatki stymulują wzrost gospodarczy i tworzą miejsca pracy",
    "Opieka zdrowotna to prawo człowieka i powinna być uniwersalna",
    "Wolny rynek poprawia jakość opieki zdrowotnej",
    "Musimy walczyć ze zmianami klimatu za wszelką cenę",
    "Ekologizm zagraża konkurencyjności gospodarki",
]

# Encoder: tekst → 2D przestrzeń polityczna
encoder = IdealPointTextEncoder(
    output_dim=2,  # 2D: Economic + Social
    corpus=corpus,
    use_embeddings=True,  # TF-IDF + sentence embeddings
    embedding_model="all-MiniLM-L6-v2",
)

# Koduj nowy tekst polityczny
text = "Progresywne opodatkowanie zmniejsza nierówności"
ideal_point = encoder.encode(text)
print(f"Ideal point: {ideal_point.numpy()}")
```

---

## 📖 Dokumentacja

### Struktura repozytorium

```
policyflux/
├── policyflux/                    # Główny pakiet
│   ├── core/                      # Abstrakcje bazowe
│   │   ├── simple_actors_template.py     # CongressMan
│   │   ├── complex_actors_template.py    # ComplexActor
│   │   ├── bill_template.py
│   │   ├── congress_model_template.py
│   │   ├── layer_template.py
│   │   ├── executive.py
│   │   ├── aggregation_strategy.py
│   │   ├── types.py
│   │   ├── contexts.py            # NEW
│   │   ├── voting_strategy.py      # NEW
│   │   ├── container.py            # NEW
│   │   └── __init__.py
│   │
│   ├── layers/                    # Warstwy decyzyjne
│   │   ├── idealpoint.py
│   │   ├── public_pressure.py
│   │   ├── lobbying.py
│   │   ├── media_pressure.py
│   │   ├── party.py
│   │   ├── government_agenda.py
│   │   ├── neural.py
│   │   └── __init__.py
│   │
│   ├── toolbox/                   # Implementacje
│   │   ├── actors.py
│   │   ├── bill.py
│   │   ├── congress_model.py
│   │   ├── executive_systems.py
│   │   ├── advanced_actors/
│   │   │   ├── speaker.py
│   │   │   ├── whips.py
│   │   │   ├── lobby.py
│   │   │   ├── white_house.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   │
│   ├── engines/                   # Silniki symulacji
│   │   ├── engine_template.py
│   │   ├── parallel_monte_carlo.py
│   │   ├── deterministic_engine.py
│   │   └── __init__.py
│   │
│   ├── integration/                # HIGH-LEVEL API (REFACTORED)
│   │   ├── config.py              # Configuration classes
│   │   ├── builders/              # Factory functions
│   │   │   ├── engine_builder.py
│   │   │   ├── congress_builder.py
│   │   │   ├── layer_builder.py
│   │   │   ├── actor_builder.py
│   │   │   ├── mechanic_builders.py
│   │   │   └── __init__.py
│   │   ├── presets/               # Pre-configured systems
│   │   │   ├── president_preset.py
│   │   │   ├── parliament_preset.py
│   │   │   ├── semipresident_preset.py
│   │   │   └── __init__.py
│   │   ├── registry.py
│   │   └── __init__.py
│   │
│   ├── dprocessing/               # Text encoding & data processing
│   │   └── __init__.py
│   │
│   ├── utils/                     # Utilities
│   │   ├── reports/               # Visualizations
│   │   │   ├── bar_charts.py
│   │   │   └── pie_charts.py
│   │   ├── pfrandom.py
│   │   ├── logging_config.py
│   │   └── __init__.py
│   │
│   ├── pipeline/                  # NEW (placeholder)
│   │   └── __init__.py
│   │
│   ├── __init__.py                # Public API
│   ├── logging_config.py
│   ├── pfrandom.py
│   └── __pycache__/
│
├── pyproject.toml                 # Metadata i zależności
├── readme.md                      # Ten plik
└── .gitignore
```

### Kluczowe koncepcje

**Policy Space**: Wielowymiarowa przestrzeń polityczna reprezentująca różne wymiary ideologiczne (np. Left-Right, Liberal-Conservative, Isolationist-Interventionist). Każdy aktor i projekt ustawy ma pozycję w tej przestrzeni.

**Ideal Point**: Pozycja aktora lub ustawy w policy space, reprezentująca preferencje ideologiczne. Zwykle wyrażana jako wektor w n-wymiarowej przestrzeni.

**Utility Function**: Funkcja określająca użyteczność dla aktora głosującego na projekt ustawy, zwykle oparta na dystansie euklidesowym między ideal point a pozycją ustawy (actor głosuje "tak" jeśli użyteczność > threshold).

**Decision Layer**: Warstwa wirtualna modyfikująca prawdopodobieństwo głosowania na podstawie różnych czynników (opinia publiczna, lobbying, dyscyplina partyjna, media, etc.). Każda warstwa przyjmuje wejściowe prawdopodobieństwo i zwraca zmodyfikowane.

**Aggregation Strategy**: Algorytm łączenia outputów wielu warstw w finalną decyzję. Dostępne strategie: Sequential (warstwy modyfikują się sekwencyjnie), Average (średnia arytmetyczna), Weighted (ważona suma), Multiplicative (mnożenie dla efektu weta).

**Voting Strategy**: Abstrakcja określająca, jak przekonwertować prawdopodobieństwo na ostateczną decyzję (głos "tak"/"nie"). Można wybrać między ProbabilisticVoting (losowo z danym prawdopodobieństwem), DeterministicVoting (threshold) lub innymi strategiami.

---

## 🛠️ Rozwój

### Struktura developerska

```bash
# Fork i clone
git clone https://github.com/yourusername/policyflux.git
cd policyflux

# Instalacja z dev tools
pip install -e ".[dev,torch,text-encoders]"

# Pre-commit hooks (opcjonalnie)
pip install pre-commit
pre-commit install
```

### Uruchom testy

```bash
# Wszystkie testy
pytest

# Z verbose output
pytest -v

# Konkretny plik
pytest tests/test_core.py

# Z coverage
pytest --cov=policyflux
```

### Linting i formatowanie

```bash
# Ruff (linting)
ruff check policyflux/

# Ruff (auto-fix)
ruff check --fix policyflux/

# MyPy (type checking)
mypy policyflux/
```

### Contributing

Wkład w projekt jest mile widziany! Proces:

1. **Otwórz Issue**: Opisz bug/feature przed rozpoczęciem pracy
2. **Fork i Branch**: Stwórz branch dla swojej funkcjonalności
3. **Implementuj**: Dodaj kod + testy + dokumentację
4. **Testy**: Upewnij się, że wszystkie testy przechodzą
5. **Pull Request**: Opisz zmiany, linkuj do issue

**Co potrzebujemy:**
- 🐛 Bug fixes testy (coverage jest niskie)
- ✨ Nowe warstwy decyzyjne (np. Media Sentiment Layer)
- 📚 Więcej przykładów i case studies
- 🧪 Lepsze testowanie (target >80% coverage)
- 📖 Lepsza dokumentacja API (docstrings, Sphinx)
- 🌍 Wsparcie dla więcej systemów politycznych (koalicje, kommitee)

### Ostatnie zmiany (Recent refactoring)

Projekt przeszedł niedawno znaczną refaktoryzację:

- **Reorganizacja Integration Module**: Była monolityczna `policyflux/integration.py`, teraz strukturyzowana hierarhicznie jako `policyflux/integration/` z podmodułami `builders/` i `presets/`
- **Nowe abstrakcje**: Dodane `VotingContext`, `SimulationContext` (immutable konteksty decyzyjne) i `VotingStrategy` (abstrakcja strategii głosowania)
- **Dependency Injection**: Dodany `ServiceContainer` dla zarządzania zależnościami
- **Usunięte przykłady**: Wszystkie pliki przykładów (examples/*.py) były tymczasowe; użytkownicy powinni pisać swoje
- **Lazy Loading**: Integration submodules używają `__getattr__` aby uniknąć circular imports

Migracja z starego API:
```python
# Stare (niewalidne)
from policyflux.integration import build_engine  # Może nie działać - stary singiel file

# Nowe (poprawne)
from policyflux import build_engine  # Importuj z głównego pakietu
```

### Roadmap

- [ ] Lepsza dokumentacja API (Sphinx docs)
- [ ] Test coverage >80%
- [ ] Real-world case studies (parlamenty, legislatury)
- [ ] Web dashboard/UI dla wizualizacji symulacji
- [ ] Export/import symulacji (JSON, YAML, HDF5)
- [ ] Integracja z real-world datasets (voteview.org, parlgov.org)
- [ ] Performance optimization (Cython dla hot paths, numba JIT)
- [ ] Coalition formation models
- [ ] Committee assignment models
- [ ] Veto point analysis
- [ ] Comparative statics (parameter sensitivity analysis)

---

## 📄 Licencja

Projekt nie ma jeszcze określonej licencji. Przed użyciem w celach komercyjnych lub publikacją skontaktuj się z autorem.

---

## 🙏 Podziękowania

Projekt inspirowany badaniami z zakresu:
- Spatial voting theory (Downs, 1957)
- Ideal point estimation (Clinton, Jackman, Rivers, 2004)
- Legislative behavior models (Poole & Rosenthal)

---

## 📧 Kontakt

- **Issues**: [GitHub Issues](https://github.com/yourusername/policyflux/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/policyflux/discussions)
- **Email**: pawelecpiotr404@gmail.com

---

<div align="center">

**Zbudowane z ❤️ dla badaczy politologii i data scientists**

[![Star on GitHub](https://img.shields.io/github/stars/yourusername/policyflux.svg?style=social)](https://github.com/yourusername/policyflux)

</div>
