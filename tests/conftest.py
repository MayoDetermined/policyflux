import os

import pytest

from policyflux.core.id_generator import get_id_generator


os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def reset_global_id_generator() -> None:
    get_id_generator().reset()
    yield
    get_id_generator().reset()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        node_id = item.nodeid.replace("\\", "/")
        if "/smoke/" in node_id:
            item.add_marker(pytest.mark.smoke)
        elif "/unit/" in node_id:
            item.add_marker(pytest.mark.unit)
