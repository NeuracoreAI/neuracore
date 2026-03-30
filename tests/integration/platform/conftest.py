# cspell:ignore terminalreporter exitstatus
import logging
import shutil
import sys
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from recording_playback_shared import (
    MATRIX_SESSION_RUNS,
    cleanup_test_profiles,
    daemon_cleanup,
)

import neuracore as nc
from neuracore.core import auth as auth_module
from neuracore.core.config.config_manager import CONFIG_DIR, Config, get_config_manager

logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def daemon_setup_teardown():
    daemon_cleanup()
    yield
    daemon_cleanup()


@pytest.fixture(autouse=True)
def reset_neuracore_config_state() -> Generator[None, None, None]:
    shutil.rmtree(CONFIG_DIR, ignore_errors=True)
    get_config_manager().config = Config()
    auth_module.get_auth()._access_token = None

    yield

    shutil.rmtree(CONFIG_DIR, ignore_errors=True)


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


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not MATRIX_SESSION_RUNS:
        return

    separator = "=" * 64
    lines = [
        "",
        separator,
        f"Matrix session summary  ({len(MATRIX_SESSION_RUNS)} test(s) completed)",
        separator,
    ]

    all_labels = sorted(
        {label for run in MATRIX_SESSION_RUNS for label in run["timer_stats"]}
    )
    for run in MATRIX_SESSION_RUNS:
        total_wall_s = sum(ctx["wall_s"] for ctx in run["context_results"])
        lines.append(f"\n  {run['case_id']}  (wall={total_wall_s:.1f}s)")
        for label in all_labels:
            stats = run["timer_stats"].get(label)
            if stats is None:
                continue
            count = int(stats["count"])
            avg = stats["total"] / count if count > 0 else 0.0
            lines.append(
                f"    {label:<42}  {count:3}x"
                f"  avg={avg:.3f}s  max={stats['max']:.3f}s"
            )

    lines.append(separator)
    terminalreporter.write_line("\n".join(lines))
