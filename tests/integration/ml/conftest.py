import os

import pytest
import yaml

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIGS_FILE = os.path.join(THIS_DIR, "algorithm_configs.yaml")


def _load_algorithm_configs() -> list[dict]:
    with open(_CONFIGS_FILE) as f:
        return yaml.safe_load(f)["algorithms"]


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
