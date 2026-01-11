from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from policyflux.main import main as _main  # loads src/policyflux/main.py

if __name__ == "__main__":
    _main()
