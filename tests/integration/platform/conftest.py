import logging
import os
import sys
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from recording_playback_shared import cleanup_test_profiles, daemon_cleanup

import neuracore as nc

# cspell:ignore hookwrapper makereport terminalreporter pluginmanager
# cspell:ignore getplugin nodeid longreprtext

logger = logging.getLogger(__name__)

_SUITE_TERMINATION_CLEANUP_RAN = False
_PREVIOUS_TERMINATION_HANDLERS: dict[int, object] = {}

sys.path.append(str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def daemon_setup_teardown():
    if os.getenv("NCD_SKIP_DAEMON_CLEANUP_FOR_DEBUG") == "1":
        yield
        return
    daemon_cleanup()
    yield
    daemon_cleanup()


@pytest.fixture(autouse=True)
def cleanup_profiles():
    yield
    cleanup_test_profiles()


@pytest.fixture
def dataset_cleanup() -> Generator[Callable[[str], None], None, None]:
    dataset_names: list[str] = []

    def register(dataset_name: str) -> None:
        dataset_names.append(dataset_name)

    yield register

    for dataset_name in dataset_names:
        try:
            nc.login()
            nc.get_dataset(dataset_name).delete()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to delete test dataset: %s", dataset_name)
