import os
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

_CONFIGS_FILE = os.path.join(THIS_DIR, "algorithm_configs.yaml")


def _load_algorithm_configs() -> list[dict]:
    with open(_CONFIGS_FILE) as f:
        return yaml.safe_load(f)["algorithms"]


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> None:
    """Attach per-phase reports and mark step-test classes when a step fails."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
    if call.when != "call" or not rep.failed:
        return
    test_cls = getattr(item, "cls", None)
    if test_cls is not None and getattr(test_cls, "track_step_teardown", False):
        test_cls.all_steps_passed = False


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize algorithm_config_entry from algorithm_configs.yaml.

    When ALGORITHM_NAME is set (e.g. by a CI matrix job), only that algorithm
    is parametrized so each runner handles exactly one algorithm.
    """
    if "algorithm_config_entry" not in metafunc.fixturenames:
        return

    configs = _load_algorithm_configs()

    algo_name = os.environ.get("ALGORITHM_NAME")
    if algo_name:
        configs = [c for c in configs if c["name"] == algo_name]

    metafunc.parametrize(
        "algorithm_config_entry",
        configs,
        ids=[c["name"] for c in configs],
    )
