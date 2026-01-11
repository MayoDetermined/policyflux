"""Console entrypoint that forwards to the legacy CLI."""
from __future__ import annotations

from typing import List, Optional

from main import main as _run_cli


def main(argv: Optional[List[str]] = None) -> None:
    """Invoke the legacy CLI; exported as `policyflux` console script."""
    _run_cli(argv)


if __name__ == "__main__":
    main()
