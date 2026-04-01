# cspell:ignore terminalreporter exitstatus
import logging
import os
import shutil
import sys
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from recording_playback_matrix_config import (
    RecordingPlaybackMatrixCase,
    log_run_analysis,
)
from recording_playback_shared import (
    MATRIX_SESSION_RUNS,
    OFFLINE_DB_PATH,
    OFFLINE_RECORDINGS_ROOT,
    Timer,
    cleanup_test_profiles,
    daemon_cleanup,
)

import neuracore as nc
from neuracore.api.globals import GlobalSingleton
from neuracore.core import auth as auth_module
from neuracore.core.config.config_manager import CONFIG_DIR, Config, get_config_manager
from neuracore.core.streaming import (
    recording_state_manager as recording_state_manager_module,
)

logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent))


def _cleanup_test_robot(active_robot: object | None) -> None:
    if active_robot is None:
        return

    temp_dir = getattr(active_robot, "_temp_dir", None)
    if temp_dir is not None:
        try:
            temp_dir.cleanup()
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to cleanup active test robot temp dir", exc_info=True
            )
        finally:
            active_robot._temp_dir = None

    # Test teardown must not terminate the shared ZMQ context via Robot.close().
    if hasattr(active_robot, "_daemon_recording_context"):
        active_robot._daemon_recording_context = None


def _reset_runtime_state() -> None:
    active_robot = GlobalSingleton()._active_robot
    _cleanup_test_robot(active_robot)

    GlobalSingleton()._active_robot = None
    GlobalSingleton()._active_dataset_id = None
    GlobalSingleton()._has_validated_version = False

    recording_manager_future = recording_state_manager_module._recording_manager
    if recording_manager_future is not None:
        try:
            recording_manager = recording_manager_future.result(timeout=1)
            recording_manager.recording_robot_instances.clear()
            recording_manager.active_dataset_ids.clear()
            recording_manager._expired_recording_ids.clear()
            for recording_id in list(recording_manager._recording_timers):
                recording_manager._cancel_recording_timers(recording_id)
            recording_manager._recording_timers.clear()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to reset recording state manager", exc_info=True)
        finally:
            recording_state_manager_module._recording_manager = None


def _assert_runtime_state_is_clean() -> None:
    assert GlobalSingleton()._active_robot is None, "Expected no active robot residue"
    assert (
        GlobalSingleton()._active_dataset_id is None
    ), "Expected no active dataset residue"
    assert (
        GlobalSingleton()._has_validated_version is False
    ), "Expected version validation cache to be reset"
    assert (
        recording_state_manager_module._recording_manager is None
    ), "Expected no cached recording state manager residue"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add CLI options for daemon lifecycle control."""
    parser.addoption(
        "--reuse-daemon",
        action="store_true",
        default=False,
        help="Do not kill the data daemon between tests.",
    )


@pytest.fixture(autouse=True)
def daemon_setup_teardown(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    reuse = request.config.getoption("--reuse-daemon", default=False)
    _reset_runtime_state()
    _assert_runtime_state_is_clean()
    if not reuse:
        daemon_cleanup(state_db_action="empty")
    yield
    _reset_runtime_state()
    _assert_runtime_state_is_clean()
    if not reuse:
        daemon_cleanup(state_db_action="empty")


@pytest.fixture(autouse=True)
def reset_neuracore_config_state() -> Generator[None, None, None]:
    shutil.rmtree(CONFIG_DIR, ignore_errors=True)
    get_config_manager().config = Config()
    auth_module.get_auth()._access_token = None
    _reset_runtime_state()
    _assert_runtime_state_is_clean()

    yield

    _reset_runtime_state()
    _assert_runtime_state_is_clean()
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


@pytest.fixture()
def reset_matrix_timer_stats() -> None:
    """Clear per-case matrix timer stats before each test that opts in."""
    for label in [key for key in Timer._stats if key.startswith("matrix.")]:
        del Timer._stats[label]


@pytest.fixture()
def daemon_offline_env() -> Generator[None, None, None]:
    """Set daemon env vars to the test-local state dir and restore on teardown."""
    previous_recordings_root = os.environ.get("NEURACORE_DAEMON_RECORDINGS_ROOT")
    previous_db_path = os.environ.get("NEURACORE_DAEMON_DB_PATH")
    os.environ["NEURACORE_DAEMON_RECORDINGS_ROOT"] = str(OFFLINE_RECORDINGS_ROOT)
    os.environ["NEURACORE_DAEMON_DB_PATH"] = str(OFFLINE_DB_PATH)
    try:
        yield
    finally:
        if previous_recordings_root is None:
            os.environ.pop("NEURACORE_DAEMON_RECORDINGS_ROOT", None)
        else:
            os.environ["NEURACORE_DAEMON_RECORDINGS_ROOT"] = previous_recordings_root
        if previous_db_path is None:
            os.environ.pop("NEURACORE_DAEMON_DB_PATH", None)
        else:
            os.environ["NEURACORE_DAEMON_DB_PATH"] = previous_db_path


@pytest.fixture()
def log_run_analysis_on_teardown(
    request: pytest.FixtureRequest,
) -> Callable[[RecordingPlaybackMatrixCase, list[dict[str, object]]], None]:
    """Register case+results to be passed to log_run_analysis at teardown."""
    state: dict[str, object] = {}

    def register(
        case: RecordingPlaybackMatrixCase, results: list[dict[str, object]]
    ) -> None:
        state["case"] = case
        state["results"] = results

    def finalizer() -> None:
        if state.get("results"):
            request.node._matrix_run_analysis_report = log_run_analysis(
                case=state["case"],  # type: ignore[arg-type]
                results=state["results"],  # type: ignore[arg-type]
            )

    request.addfinalizer(finalizer)
    return register


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"_report_{report.when}", report)
    if report.when != "teardown":
        return

    terminalreporter = item.config.pluginmanager.getplugin("terminalreporter")
    if terminalreporter is None:
        return

    setup_report = getattr(item, "_report_setup", None)
    call_report = getattr(item, "_report_call", None)
    teardown_report = getattr(item, "_report_teardown", None)
    reports = [
        report for report in (setup_report, call_report, teardown_report) if report
    ]
    if not reports:
        return

    if any(report.failed for report in reports):
        final_outcome = "FAILED"
    elif any(report.skipped for report in reports):
        final_outcome = "SKIPPED"
    else:
        final_outcome = "PASSED"

    terminalreporter.write_line("")
    terminalreporter.write_line(f"[{final_outcome}] {item.nodeid}", bold=True)

    failure_report = next(
        (
            candidate
            for candidate in (setup_report, call_report, teardown_report)
            if candidate is not None and candidate.failed
        ),
        None,
    )
    if failure_report is not None and getattr(failure_report, "longreprtext", ""):
        terminalreporter.write_line(failure_report.longreprtext)

    analysis_report = getattr(item, "_matrix_run_analysis_report", None)
    if analysis_report:
        terminalreporter.write_line(analysis_report)


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
                f"  limit={stats['limit']:.3f}s"
            )

    lines.append(separator)
    terminalreporter.write_line("\n".join(lines))
