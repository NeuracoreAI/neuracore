from __future__ import annotations

import os
import time
from collections.abc import Callable

import pytest

from tests.integration.platform.data_daemon.shared.assertions import (
    clear_daemon_timer_stats as _clear_daemon_timer_stats,
)
from tests.integration.platform.data_daemon.shared.process_control import stop_daemon
from tests.integration.platform.data_daemon.shared.profiles import cleanup_test_profiles
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    SESSION_RUNS,
    DataDaemonTestCase,
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
    build_isolation_run_analysis,
    build_terminal_summary,
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
    stop_daemon(method="sigkill")
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


@pytest.fixture(autouse=True)
def cleanup_profiles():
    """Remove test-created daemon profiles after each test."""
    yield
    cleanup_test_profiles()


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
) -> Callable[..., None]:
    """Register case+results to be passed to log_run_analysis at teardown."""
    fixture_start = time.perf_counter()
    state: dict[str, object] = {}

    def register(
        case: DataDaemonTestCase,
        results: list[ContextResult],
        *,
        label_prefix: str | None = None,
    ) -> None:
        state["case"] = case
        state["results"] = results
        state["label_prefix"] = label_prefix

    def finalizer() -> None:
        if state.get("results"):
            try:
                test_wall_s = time.perf_counter() - fixture_start
                request.node.run_analysis_report = build_isolation_run_analysis(
                    case=state["case"],  # type: ignore[arg-type]
                    results=state["results"],  # type: ignore[arg-type]
                    case_id_prefix=state.get("label_prefix"),  # type: ignore[arg-type]
                    test_wall_s=test_wall_s,
                )
            except Exception:  # noqa: BLE001
                pass

    request.addfinalizer(finalizer)
    return register


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a session summary at the end of the test run."""
    del exitstatus, config
    lines = build_terminal_summary(SESSION_RUNS)
    if lines:
        terminalreporter.write_line("\n".join(lines))
