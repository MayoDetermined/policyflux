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
from policyflux.integration import IntegrationConfig, LayerConfig, build_engine

# Konfiguracja symulacji
config = IntegrationConfig(
    num_actors=50,           # 50 posłów
    policy_dim=2,            # 2D: Left-Right + Liberal-Conservative
    iterations=100,          # 100 głosowań
    seed=12345,              # Deterministyczny RNG
    description="Moja pierwsza symulacja",
    layer_config=LayerConfig(
        include_ideal_point=True,      # Preferencje ideologiczne
        include_public_opinion=True,   # Opinia publiczna
        include_party_discipline=True, # Dyscyplina partyjna
        public_support=0.60,           # 60% poparcia publicznego
        party_discipline_strength=0.5, # Średnia dyscyplina
    ),
)

# Buduj silnik i uruchom
engine = build_engine(config)
engine.run_simulation()

# Wyniki
print(engine)
engine.get_pretty_votes()
```

### Uruchom gotowe przykłady

```bash
# Najprostsze głosowanie
python examples/01_basic_voting.py

# Wszystkie warstwy decyzyjne
python examples/02_all_layers_showcase.py

# Porównanie systemów politycznych
python examples/03_executive_systems.py

# Strategie agregacji
python examples/04_aggregation_strategies.py

# Integracja ML/PyTorch
python examples/05_ml_integration.py

# Zaawansowani aktorzy
python examples/06_advanced_actors.py

# Analiza scenariuszy "what-if"
python examples/07_scenario_analysis.py

# Text encoders dla ideal points
python examples/08_text_encoder_idealpoints.py

# Prosty showcase
python simple_simulation.py
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

- **`actors_template.py`**: Bazowe klasy dla posłów (Actor, Voter)
- **`bill_template.py`**: Reprezentacja projektów ustaw (Bill)
- **`congress_model_template.py`**: Model parlamentu (CongressModel)
- **`layer_template.py`**: Abstrakcja warstw decyzyjnych (Layer, DecisionLayer)
- **`executive.py`**: Abstrakcje dla egzekutywy (ExecutiveActor, Executive)
- **`aggregation_strategy.py`**: Strategie łączenia output warstw
- **`types.py`**: PolicySpace, UtilitySpace
- **`id_generator.py`**: Generator unikalnych ID

### 2. Layers (policyflux/layers/)

**Warstwy decyzyjne modyfikujące prawdopodobieństwo głosowania:**

| Warstwa | Opis | Parametry kluczowe |
|---------|------|-------------------|
| **IdealPoint** | Bazowe preferencje ideologiczne, dystans w przestrzeni politycznej | `policy_dim` |
| **PublicOpinion** | Wpływ opinii publicznej na głosowanie | `public_support` [0, 1] |
| **Lobbying** | Naciski lobbyistów i grup interesów | `lobbying_intensity` [0, 1] |
| **MediaPressure** | Wpływ mediów i pressure publicznego | `media_pressure` [0, 1] |
| **PartyDiscipline** | Dyscyplina partyjna i linia partii | `discipline_strength`, `party_line_support` |
| **GovernmentAgenda** | Kontrola agendy przez PM (systemy parlamentarne) | `pm_strength` |
| **Neural** | PyTorch neural networks jako decision layer | `model`, `training_data` |

### 3. Models (policyflux/models/)

**Implementacje modeli i silników:**

- **`actors.py`**: SequentialVoter z multi-layer decision making
- **`bill.py`**: SequentialBill z pozycją w policy space
- **`congress_model.py`**: SequentialCongressModel z głosowaniem
- **`engines.py`**: SequentialMonteCarlo, Session
- **`executive_systems.py`**: 
  - PresidentialExecutive (veto power, approval rating)
  - ParliamentaryExecutive (agenda control, confidence votes)
  - SemiPresidentialExecutive (kohabitacja)

**Advanced Actors** (policyflux/models/advanced_actors/):
- **`speaker.py`**: SequentialSpeaker (agenda setting)
- **`whips.py`**: SequentialWhip (party discipline)
- **`lobby.py`**: SequentialLobbyer (influence campaigns)
- **`white_house.py`**: SequentialPresident (executive influence)

### 4. Integration (policyflux/integration.py)

**High-level API do budowy symulacji:**

```python
from policyflux.integration import (
    IntegrationConfig,
    LayerConfig,
    AdvancedActorsConfig,
    build_engine,
    create_presidential_config,
    create_parliamentary_config,
    create_semi_presidential_config,
)
```

**Główne klasy konfiguracyjne:**
- `IntegrationConfig`: Główna konfiguracja symulacji
- `LayerConfig`: Konfiguracja warstw decyzyjnych
- `AdvancedActorsConfig`: Konfiguracja zaawansowanych aktorów

### 5. Data Processing (policyflux/dprocessing/)

**Przetwarzanie tekstów do przestrzeni politycznej:**

- **`text_processor.py`**: SimpleTextVectorizer (tokenizacja, vocab)
- **IdealPointTextEncoder**: TF-IDF + sentence embeddings → ideal points
- **IdealPointEncoderDF**: DataFrame → policy space

### 6. Utils (policyflux/utils/)

**Narzędzia pomocnicze:**

- **`reports/bar_charts.py`**: Wykresy słupkowe wyników głosowań
- **`reports/pie_charts.py`**: Wykresy kołowe podziałów

---

## 📚 Przykłady użycia

### Przykład 1: Podstawowe głosowanie ideologiczne

```python
from policyflux.integration import IntegrationConfig, LayerConfig, build_engine

config = IntegrationConfig(
    num_actors=50,
    policy_dim=1,  # 1D: Left-Right tylko
    iterations=100,
    seed=12345,
    layer_config=LayerConfig(
        include_ideal_point=True,  # Tylko preferencje ideologiczne
        include_public_opinion=False,
        include_lobbying=False,
        include_media_pressure=False,
        include_party_discipline=False,
    ),
)

engine = build_engine(config)
engine.run_simulation()
print(engine)
```

### Przykład 2: Porównanie systemów politycznych

```python
from policyflux.integration import (
    create_presidential_config,
    create_parliamentary_config,
    build_engine,
)

# System prezydencki (USA)
prez_config = create_presidential_config(
    num_actors=100,
    policy_dim=3,
    iterations=200,
    seed=42,
    president_approval=0.52,
    veto_override_threshold=2/3,
)
prez_engine = build_engine(prez_config)
prez_engine.run_simulation()

# System parlamentarny (UK)
parl_config = create_parliamentary_config(
    num_actors=100,
    policy_dim=3,
    iterations=200,
    seed=42,
    pm_strength=0.7,
    party_discipline=0.8,
)
parl_engine = build_engine(parl_config)
parl_engine.run_simulation()

# Porównanie wyników
print(f"Presidential pass rate: {prez_engine.pass_rate:.2%}")
print(f"Parliamentary pass rate: {parl_engine.pass_rate:.2%}")
```

### Przykład 3: Zaawansowani aktorzy (lobbyści, whips)

```python
from policyflux.integration import (
    IntegrationConfig,
    LayerConfig,
    AdvancedActorsConfig,
    build_engine,
)

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
        n_lobbyists=5,              # 5 lobbyistów
        lobbyist_strength=0.5,      # Średnia siła wpływu
        lobbyist_stance=0.8,        # Pro-bill stance
        n_whips=3,                  # 3 whips
        whip_discipline_strength=0.7,  # Silna dyscyplina
        speaker_agenda_support=0.6,    # Speaker wspiera ustawę
    ),
)

engine = build_engine(config)
engine.run_simulation()
print(engine)
```

### Przykład 4: Analiza scenariuszy "what-if"

```python
from policyflux.integration import IntegrationConfig, LayerConfig, build_engine

# SCENARIUSZ 1: Status quo
base_config = IntegrationConfig(
    num_actors=100,
    policy_dim=2,
    iterations=200,
    seed=2024,
    layer_config=LayerConfig(
        include_ideal_point=True,
        include_public_opinion=True,
        public_support=0.55,  # Neutralne poparcie
    ),
)

# SCENARIUSZ 2: Mobilizacja oddolna (grassroots)
grassroots_config = IntegrationConfig(
    num_actors=100,
    policy_dim=2,
    iterations=200,
    seed=2024,
    layer_config=LayerConfig(
        include_ideal_point=True,
        include_public_opinion=True,
        public_support=0.85,  # Wysokie poparcie!
    ),
)

# Uruchom oba scenariusze
base_engine = build_engine(base_config)
base_engine.run_simulation()

grassroots_engine = build_engine(grassroots_config)
grassroots_engine.run_simulation()

# Porównaj
print(f"Status quo: {base_engine.pass_rate:.1%}")
print(f"Grassroots: {grassroots_engine.pass_rate:.1%}")
```

### Przykład 5: Machine Learning integration

```python
from policyflux.layers.idealpoint import IdealPointTextEncoder

# Korpus tekstów politycznych
corpus = [
    "We need to increase taxes on the wealthy to fund social programs",
    "Lower taxes will stimulate economic growth and create jobs",
    "Healthcare is a human right and should be universally provided",
    "Free market competition improves healthcare quality",
]

# Encoder: text → 2D ideal point space
encoder = IdealPointTextEncoder(
    output_dim=2,  # 2D: Economic + Social
    corpus=corpus,
    use_embeddings=True,  # TF-IDF + sentence embeddings
    embedding_model="all-MiniLM-L6-v2",
    hidden_dims=[128, 64],
)

# Enkoduj nowy tekst
text = "Progressive taxation reduces inequality"
ideal_point = encoder.encode(text)
print(f"Ideal point: {ideal_point.numpy()}")
```

---

## 📖 Dokumentacja

### Struktura repozytorium

```
policyflux/
├── policyflux/              # Główny pakiet
│   ├── core/                # Podstawowe abstrakcje
│   ├── layers/              # Warstwy decyzyjne
│   ├── models/              # Implementacje modeli
│   │   └── advanced_actors/ # Zaawansowani aktorzy
│   ├── dprocessing/         # Przetwarzanie danych
│   ├── utils/               # Narzędzia pomocnicze
│   │   └── reports/         # Wykresy i raporty
│   ├── __init__.py          # Public API
│   ├── config.py            # Konfiguracja Settings
│   ├── integration.py       # High-level builder API
│   ├── logging_config.py    # Logger
│   └── pfrandom.py          # RNG manager
│
├── examples/                # Przykłady użycia
│   ├── 01_basic_voting.py
│   ├── 02_all_layers_showcase.py
│   ├── 03_executive_systems.py
│   ├── 04_aggregation_strategies.py
│   ├── 05_ml_integration.py
│   ├── 06_advanced_actors.py
│   ├── 07_scenario_analysis.py
│   └── 08_text_encoder_idealpoints.py
│
├── simple_simulation.py     # Prosty przykład
├── pyproject.toml           # Metadata i zależności
└── readme.md                # Ten plik
```

### Kluczowe koncepcje

**Policy Space**: Wielowymiarowa przestrzeń polityczna (np. [Left-Right, Liberal-Conservative, Isolationist-Interventionist])

**Ideal Point**: Pozycja aktora/ustawy w policy space, reprezentująca preferencje ideologiczne

**Utility Function**: Funkcja użyteczności określająca korzyść aktora z danej pozycji ustawy (zwykle dystans euklidesowy)

**Decision Layers**: Warstwy modyfikujące prawdopodobieństwo głosowania na podstawie różnych czynników (opinia publiczna, lobbying, etc.)

**Aggregation Strategy**: Sposób łączenia outputów warstw w finalną decyzję (sequential, average, weighted, multiplicative)

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
pytest test_integration.py

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
- 🐛 Bug fixes
- ✨ Nowe warstwy decyzyjne
- 📚 Więcej przykładów i case studies
- 🧪 Więcej testów (coverage jest niskie)
- 📖 Lepsza dokumentacja
- 🌍 Wsparcie dla więcej systemów politycznych

### Roadmap (planowane)

- [ ] Lepsza dokumentacja API (docstrings, Sphinx)
- [ ] Większa coverage testów (>80%)
- [ ] Więcej przykładów real-world
- [ ] Web UI/dashboard dla symulacji
- [ ] Export/import symulacji (JSON/YAML)
- [ ] Integracja z real-world datasets (voteview, parlgov)
- [ ] Performance optimization (Cython, numba)
- [ ] Więcej strategii agregacji
- [ ] Coalition formation models
- [ ] Committee assignment models

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
