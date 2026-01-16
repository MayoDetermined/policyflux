# Congress Simulator (PolicyFlux) ⚡

> Status: **w budowie** – 🚧 dokumentacja i funkcje są rozwijane na bieżąco.

![Symulacja w ruchu](https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif)

## Spis treści 🧭
- [O projekcie](#o-projekcie-)
- [Wymagania](#wymagania-)
- [Instalacja](#instalacja-)
- [Szybki start](#szybki-start-️)
- [Struktura repozytorium](#struktura-repozytorium-)
- [Rozwój](#rozwój-)
- [Kontakt i wsparcie](#kontakt-i-wsparcie-)

## O projekcie ✨
Symulator do eksperymentów nad procesem legislacyjnym i zachowaniami aktorów politycznych. Skupia się na analizie polityk, głosowań oraz interakcji między uczestnikami.

## Wymagania 🛠️
- Python 3.10+
- Pip/Poetry (zalecane)
- System operacyjny: Linux/macOS/Windows

## Instalacja 🚀
```bash
# klonowanie
git clone https://github.com/<org>/Congress.git
cd Congress

# instalacja zależności
pip install -r requirements.txt
# lub: poetry install
```

## Szybki start ▶️
```bash
python main.py --help
# przykład uruchomienia
python main.py simulate --config configs/example.yaml
```

## Struktura repozytorium 🗂️
- `src/policyflux/` – logika symulatora
- `configs/` – przykładowe konfiguracje uruchomień
- `docs/` – dokumentacja (PL)
- `scripts/` – narzędzia pomocnicze i testy dymne

## Rozwój 🤝
1. Utwórz gałąź feature/...
2. Dodaj testy do nowych funkcji.
3. Uruchom testy lokalnie: `pytest`.
4. Zgłoś PR z krótkim opisem zmian.

## Kontakt i wsparcie 📬
Masz pytania lub pomysły? Otwórz issue lub napisz na kanał projektu. Każdy feedback pomaga nam rosnąć!
