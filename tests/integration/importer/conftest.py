import os
import sys
from pathlib import Path

import pytest
import yaml

# Resolve neuracore from the installed wheel before repo_root joins sys.path.
import neuracore  # noqa: F401

ROBOTS_REPO_URL = "https://github.com/NeuracoreAI/neuracore_robots.git"
ROBOTS_REPO_COMMIT = "c3d17b303296686ceb9a2fcddea06af95a88fe4f"

THIS_DIR = Path(__file__).resolve().parent
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
CONFIG_DIR = THIS_DIR / "config"
RLDS_CONFIGS_FILE = CONFIG_DIR / "rlds_importer_datasets.yaml"
LEROBOT_CONFIGS_FILE = CONFIG_DIR / "lerobot_importer_datasets.yaml"
MCAP_CONFIGS_FILE = CONFIG_DIR / "mcap_importer_datasets.yaml"


def _load_configs(config_path: Path) -> list[dict]:
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)["datasets"]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize importer dataset cases from per-type config files."""
    if "importer_rlds_dataset_case" in metafunc.fixturenames:
        configs = _load_configs(RLDS_CONFIGS_FILE)
        dataset_name = os.environ.get("IMPORTER_RLDS_DATASET_NAME")
        if dataset_name:
            configs = [config for config in configs if config["name"] == dataset_name]
        metafunc.parametrize(
            "importer_rlds_dataset_case",
            configs,
            ids=[config["name"] for config in configs],
        )

    if "importer_lerobot_dataset_case" in metafunc.fixturenames:
        configs = _load_configs(LEROBOT_CONFIGS_FILE)
        dataset_name = os.environ.get("IMPORTER_LEROBOT_DATASET_NAME")
        if dataset_name:
            configs = [config for config in configs if config["name"] == dataset_name]
        metafunc.parametrize(
            "importer_lerobot_dataset_case",
            configs,
            ids=[config["name"] for config in configs],
        )

    if "importer_mcap_dataset_case" in metafunc.fixturenames:
        configs = _load_configs(MCAP_CONFIGS_FILE)
        dataset_name = os.environ.get("IMPORTER_MCAP_DATASET_NAME")
        if dataset_name:
            configs = [config for config in configs if config["name"] == dataset_name]
        metafunc.parametrize(
            "importer_mcap_dataset_case",
            configs,
            ids=[config["name"] for config in configs],
        )
