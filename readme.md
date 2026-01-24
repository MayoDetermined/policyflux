
# Policyflux

Policyflux — biblioteka do modelowania i analizy zachowań parlamentarnych, lobbyingowych i dynamiki legislacyjnej.

## Status

Development: w toku. Kod jest użyteczny do eksperymentów, ale API może się zmieniać.

## Szybki start

- **Wymagania:** Python 3.10+ rekomendowane.
- **Instalacja deweloperska:**

```bash
pip install -e .
```

- **Uruchom przykład:**

```bash
python simple_simulation.py
```

## Opis projektu

Repozytorium zawiera moduły do budowy prostych i zaawansowanych modeli parlamentarno-legislacyjnych: reprezentacje aktorów, propozycje ustaw, mechanizmy głosowania oraz warstwy wpływów (lobbying, media, presja publiczna).

## Struktura repozytorium

- **`policyflux/`**: główny pakiet z konfiguracją i modułami.
	- `core/` — szablony i podstawowe komponenty modeli (aktory, warstwy, ID generator itp.).
	- `layers/` — konkretne warstwy modelu (ideal point, lobbying, media, party, neural itp.).
	- `models/` — implementacje modeli i silniki symulacji.
	- `utils/` — narzędzia pomocnicze i raporty.
- `simple_simulation.py` — mały przykład uruchomienia symulacji.
- `pyproject.toml` — metadane i zależności projektu.

## Przykład użycia

Po zainstalowaniu (lub w środowisku virtualenv) uruchom `simple_simulation.py`, by zobaczyć podstawową symulację i wygenerowane raporty.

## Wkład i rozwój

Chętnie przyjmuję zgłoszenia (PR) i issues. Najpierw otwórz issue z opisem proponowanej zmiany/buga; następnie przygotuj PR z testami i krótkim opisem implementacji.

## Licencja

Brak określonej licencji w repozytorium — dodaj plik `LICENSE` jeśli chcesz jawnie udostępnić projekt.