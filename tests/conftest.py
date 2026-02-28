import os

import pytest

from policyflux.core.id_generator import get_id_generator


os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def reset_global_id_generator() -> None:
    get_id_generator().reset()
    yield
    get_id_generator().reset()
