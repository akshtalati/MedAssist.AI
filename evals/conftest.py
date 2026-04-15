import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: hits live API or heavy deps; gate with RUN_EVAL_API=1")


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_EVAL_API") == "1":
        return
    skip_int = pytest.mark.skip(reason="set RUN_EVAL_API=1 to run integration eval tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_int)
