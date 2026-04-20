from __future__ import annotations

import os
from collections.abc import Callable

import pytest

from tests.integration.platform.data_daemon.shared.assertions import (
    clear_daemon_timer_stats as _clear_daemon_timer_stats,
)
from tests.integration.platform.data_daemon.shared.process_control import (
    stop_daemon,
    wait_for_daemon_shutdown,
)
from tests.integration.platform.data_daemon.shared.profiles import cleanup_test_profiles
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    SESSION_RUNS,
    DataDaemonTestCase,
    _format_timer_stats_line,
    log_run_analysis,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
    ContextResult,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STORAGE_STATE_DELETE,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    OFFLINE_DB_PATH,
    OFFLINE_RECORDINGS_ROOT,
    apply_storage_state_action,
)

# cspell:ignore terminalreporter exitstatus finalizer NODEIDS exitfirst unparameterized
# cspell:ignore nodeid getfixturevalue


_BATCH_START_CLEANED_NODEIDS: set[str] = set()


@pytest.fixture(autouse=True, scope="session")
def daemon_test_state_env():
    """Point all daemon tests at the shared .data_daemon_test_state directory.

    Applied session-wide so every test — offline, online, behavioural, and
    performance — records and uploads to a single known root rather than
    scattering artefacts across ~/.neuracore or CWD.
    """
    OFFLINE_RECORDINGS_ROOT.mkdir(parents=True, exist_ok=True)
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
def daemon_offline_env():
    """Compatibility shim — session env already points to test state dir."""
    yield


@pytest.fixture(autouse=True)
def cleanup_profiles():
    """Remove test-created daemon profiles after each test."""
    yield
    cleanup_test_profiles()


@pytest.fixture(autouse=True)
def await_daemon_shutdown():
    """Wait for any in-progress daemon shutdown to complete before the test body runs.

    Call this fixture before :func:`~assertions.assert_daemon_cleanup` to
    avoid assertion races when the previous test left the daemon mid-shutdown.
    The fixture polls until all daemon PIDs have exited and the PID file and
    Unix socket are gone, then yields so the test can assert the clean state.

    Raises:
        TimeoutError: Propagated from :func:`wait_for_daemon_shutdown` when the
            daemon does not finish shutting down within the default timeout.
    """
    stop_daemon(method="sigkill")
    wait_for_daemon_shutdown()
    yield


@pytest.fixture(autouse=True)
def kill_daemon_on_teardown():
    """SIGKILL the daemon after each test to prevent orphaned processes.

    Ensures that if a test is stopped early (e.g. Ctrl-C, --exitfirst, or an
    unhandled exception that bypasses normal teardown), the daemon does not
    remain running and pollute subsequent tests or the host environment.
    """
    yield
    stop_daemon(method="sigkill")


@pytest.fixture(autouse=True)
def apply_batch_start_storage_state(request: pytest.FixtureRequest) -> None:
    """Apply local storage cleanup once before the first case in each batch.

    A batch is defined as one parametrized test function over ``case``.  The
    fixture keys by the unparameterized node id, so cleanup runs once before
    the first case and is skipped for the remaining cases in that batch.
    """
    if "case" not in request.fixturenames:
        return

    nodeid_without_param = request.node.nodeid.split("[", 1)[0]
    if nodeid_without_param in _BATCH_START_CLEANED_NODEIDS:
        return

    request.getfixturevalue("case")
    apply_storage_state_action(STORAGE_STATE_DELETE)
    _BATCH_START_CLEANED_NODEIDS.add(nodeid_without_param)


@pytest.fixture()
def clear_daemon_timer_stats() -> None:
    """Clear daemon timer stats before each matrix-style test."""
    _clear_daemon_timer_stats()


@pytest.fixture()
def log_run_analysis_on_teardown(
    request: pytest.FixtureRequest,
) -> Callable[[DataDaemonTestCase, list[ContextResult]], None]:
    """Register case+results to be passed to log_run_analysis at teardown."""
    state: dict[str, object] = {}

    def register(case: DataDaemonTestCase, results: list[ContextResult]) -> None:
        state["case"] = case
        state["results"] = results

    def finalizer() -> None:
        if state.get("results"):
            request.node.run_analysis_report = log_run_analysis(
                case=state["case"],  # type: ignore[arg-type]
                results=state["results"],  # type: ignore[arg-type]
            )

    request.addfinalizer(finalizer)
    return register


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a session summary at the end of the test run."""
    del exitstatus, config
    if not SESSION_RUNS:
        return

    separator = "=" * 64
    lines = [
        "",
        separator,
        f"Session summary  ({len(SESSION_RUNS)} test(s) completed)",
        separator,
    ]

    all_labels = sorted({label for run in SESSION_RUNS for label in run["timer_stats"]})
    for run in SESSION_RUNS:
        total_wall_s = sum(ctx["wall_s"] for ctx in run["context_results"])
        dataset_suffix = (
            f"  dataset={run['dataset_name']!r}" if run.get("dataset_name") else ""
        )
        lines.append(
            f"\n  {run['case_id']}  (wall={total_wall_s:.1f}s){dataset_suffix}"
        )
        for label in all_labels:
            stats = run["timer_stats"].get(label)
            if stats is None:
                continue
            lines.append(_format_timer_stats_line(label, stats))

    lines.append(separator)
    terminalreporter.write_line("\n".join(lines))
