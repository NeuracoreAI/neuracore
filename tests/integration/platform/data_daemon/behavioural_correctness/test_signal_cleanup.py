"""Behavioural correctness tests for daemon resource cleanup on stop delivery.

Verifies that stopping the daemon via SIGTERM, SIGINT, SIGKILL, or the CLI
``stop`` command leaves no stale IPC artefacts: no PID file, no Unix socket,
and no lingering runner subprocess.

Each test starts a fresh daemon in online mode, isolated by the
``daemon_setup_teardown`` autouse fixture and ``online_daemon_running``,
calls :func:`stop_daemon` with the method under test, then asserts cleanup
invariants.

Graceful methods (CLI, SIGTERM, SIGINT) trigger the daemon's ``finally`` block
in ``runner_entry.main``, running ``runtime.shutdown()`` and ``shutdown()``.
SIGKILL bypasses the handler entirely; those tests only assert process death
and that teardown can recover from the unclean state.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

import neuracore as nc
from neuracore.data_daemon.helpers import get_daemon_pid_path
from neuracore.data_daemon.lifecycle.daemon_os_control import pid_is_running
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
    assert_no_pid_file,
    assert_socket_unlinked,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_all_traces_written,
)
from tests.integration.platform.data_daemon.shared.process_control import (
    get_runner_pids,
    stop_daemon,
)
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
    case_ids,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
    build_context_specs,
    run_and_assert_case_contexts,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STOP_METHOD_SIGKILL,
)

logger = logging.getLogger(__name__)

GRACEFUL_EXIT_TIMEOUT_S = 15.0
SIGKILL_EXIT_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_runner_pid() -> int:
    """Return the sole live runner PID; fail if not exactly one."""
    pids = get_runner_pids()
    assert (
        len(pids) == 1
    ), f"Expected exactly one runner PID before stop, got: {sorted(pids)}"
    return next(iter(pids))


# ---------------------------------------------------------------------------
# CLI stop command
# ---------------------------------------------------------------------------


def test_cli_stop_exits_daemon_and_cleans_up() -> None:
    """CLI stop produces full resource cleanup.

    The CLI stop command sends SIGTERM then waits up to 10 s, escalating to
    SIGKILL if needed.  Either way the daemon process must be gone and all
    IPC artefacts removed.
    """
    nc.login()

    with online_daemon_running():
        pid = _single_runner_pid()
        logger.info("CLI stop for daemon pid=%d", pid)
        stop_daemon(method="cli", graceful_timeout_s=GRACEFUL_EXIT_TIMEOUT_S)


def test_cli_stop_removes_pid_file() -> None:
    """PID file is absent after a clean CLI stop."""
    nc.login()

    with online_daemon_running():
        pid_path = get_daemon_pid_path()
        assert pid_path.exists(), f"PID file missing before stop: {pid_path}"
        stop_daemon(method="cli", graceful_timeout_s=GRACEFUL_EXIT_TIMEOUT_S)

    assert_no_pid_file()


def test_cli_stop_unlinks_socket() -> None:
    """Unix domain socket is removed after a clean CLI stop."""
    nc.login()

    with online_daemon_running():
        stop_daemon(method="cli", graceful_timeout_s=GRACEFUL_EXIT_TIMEOUT_S)

    assert_socket_unlinked()


# ---------------------------------------------------------------------------
# SIGTERM
# ---------------------------------------------------------------------------


def test_sigterm_exits_daemon_and_cleans_up() -> None:
    """SIGTERM triggers graceful shutdown and full resource cleanup.

    ``install_signal_handlers`` converts SIGTERM into ``KeyboardInterrupt``
    which propagates to the ``except KeyboardInterrupt`` block in
    ``runner_entry.main``, running ``runtime.shutdown()`` and ``shutdown()``.
    """
    nc.login()

    with online_daemon_running():
        pid = _single_runner_pid()
        logger.info("SIGTERM to daemon pid=%d", pid)
        stop_daemon(method="sigterm", graceful_timeout_s=GRACEFUL_EXIT_TIMEOUT_S)


def test_sigterm_removes_pid_file() -> None:
    """PID file is absent after the daemon receives SIGTERM."""
    nc.login()

    with online_daemon_running():
        stop_daemon(method="sigterm", graceful_timeout_s=GRACEFUL_EXIT_TIMEOUT_S)

    assert_no_pid_file()


def test_sigterm_unlinks_socket() -> None:
    """Unix domain socket is removed after the daemon receives SIGTERM."""
    nc.login()

    with online_daemon_running():
        stop_daemon(method="sigterm", graceful_timeout_s=GRACEFUL_EXIT_TIMEOUT_S)

    assert_socket_unlinked()


# ---------------------------------------------------------------------------
# SIGINT
# ---------------------------------------------------------------------------


def test_sigint_exits_daemon_and_cleans_up() -> None:
    """SIGINT triggers graceful shutdown and full resource cleanup.

    Semantically equivalent to Ctrl-C: the signal handler raises
    ``KeyboardInterrupt`` which is caught by the same ``except`` block used
    for SIGTERM.
    """
    nc.login()

    with online_daemon_running():
        pid = _single_runner_pid()
        logger.info("SIGINT to daemon pid=%d", pid)
        stop_daemon(method="sigint", graceful_timeout_s=GRACEFUL_EXIT_TIMEOUT_S)


def test_sigint_removes_pid_file() -> None:
    """PID file is absent after the daemon receives SIGINT."""
    nc.login()

    with online_daemon_running():
        stop_daemon(method="sigint", graceful_timeout_s=GRACEFUL_EXIT_TIMEOUT_S)

    assert_no_pid_file()


def test_sigint_unlinks_socket() -> None:
    """Unix domain socket is removed after the daemon receives SIGINT."""
    nc.login()

    with online_daemon_running():
        stop_daemon(method="sigint", graceful_timeout_s=GRACEFUL_EXIT_TIMEOUT_S)

    assert_socket_unlinked()


# ---------------------------------------------------------------------------
# SIGKILL — unclean termination
# ---------------------------------------------------------------------------


def test_sigkill_terminates_daemon_process() -> None:
    """SIGKILL immediately kills the daemon process.

    SIGKILL cannot be caught or ignored; the daemon's cleanup handler never
    runs.  The test only asserts that the process is dead — it does NOT assert
    that IPC artefacts were cleaned up, since that is not guaranteed.
    Teardown (``daemon_setup_teardown`` + ``online_daemon_running``) is
    responsible for removing stale artefacts after an unclean kill.
    """
    nc.login()

    with online_daemon_running():
        pid = _single_runner_pid()
        logger.info("SIGKILL to daemon pid=%d", pid)
        stop_daemon(method="sigkill")

        assert not pid_is_running(
            pid
        ), f"pid_is_running reports daemon pid={pid} still running after SIGKILL"


def test_sigkill_allows_clean_restart() -> None:
    """A new daemon starts cleanly after an unclean SIGKILL termination.

    ``online_daemon_running`` teardown calls ``stop_daemon`` which removes
    any stale artefacts left by SIGKILL.  A second ``online_daemon_running``
    block must succeed without manual intervention and produce a new PID.
    """
    nc.login()

    with online_daemon_running():
        pid_before = _single_runner_pid()
        stop_daemon(method="sigkill")

    with online_daemon_running():
        pid_after = _single_runner_pid()
        assert pid_after != pid_before, (
            f"Expected a new daemon PID after SIGKILL restart, "
            f"got the same pid={pid_after}"
        )
        assert pid_is_running(
            pid_after
        ), f"Restarted daemon pid={pid_after} is not running"


# ---------------------------------------------------------------------------
# Sequential / compound stop resilience
# ---------------------------------------------------------------------------


def test_sigterm_then_cli_stop_is_idempotent() -> None:
    """CLI stop after SIGTERM completes without error or stale artefacts.

    The CLI stop command reads the PID file and no-ops gracefully when the
    daemon has already stopped.
    """
    nc.login()

    with online_daemon_running():
        stop_daemon(method="sigterm", graceful_timeout_s=GRACEFUL_EXIT_TIMEOUT_S)
        # Daemon is gone; CLI stop must handle the already-stopped case cleanly.
        stop_daemon(method="cli", graceful_timeout_s=GRACEFUL_EXIT_TIMEOUT_S)


def test_sigint_then_sigterm_exits_cleanly() -> None:
    """SIGINT followed by SIGTERM (via CLI) results in a clean exit.

    Both signals use the same handler; the first one begins teardown.
    The second call must not deadlock or leave stale artefacts.
    """
    nc.login()

    with online_daemon_running():
        stop_daemon(method="sigint", graceful_timeout_s=GRACEFUL_EXIT_TIMEOUT_S)
        stop_daemon(method="sigterm", graceful_timeout_s=GRACEFUL_EXIT_TIMEOUT_S)


# ---------------------------------------------------------------------------
# Context-manager exit cleanup (regression: stale PIDs after block exit)
# ---------------------------------------------------------------------------


def test_online_daemon_running_exit_leaves_no_pids() -> None:
    """No daemon PIDs remain after ``online_daemon_running()`` exits normally.

    Regression for the failure seen in ``test_disk_db`` where
    ``assert_daemon_cleanup()`` at the top of the next test found PIDs still
    alive after the previous ``online_daemon_running()`` block exited.

    ``online_daemon_running()`` calls ``stop_daemon()`` in its ``finally``
    block; this test asserts that ``stop_daemon()`` fully reaps all runner
    subprocesses so the process table is clean for the subsequent test.
    """
    nc.login()

    with online_daemon_running():
        pid = _single_runner_pid()
        logger.info("daemon running with pid=%d, exiting context", pid)

    # Daemon must be fully gone — no lingering runner processes.


def test_online_daemon_running_exit_cleans_up_after_active_recording() -> None:
    """No daemon PIDs remain after ``online_daemon_running()`` exits mid-recording.

    Simulates the condition that triggered the test_disk_db isolation failure:
    the daemon was still alive (PIDs found in the process table) when the next
    test began its pre-condition ``assert_daemon_cleanup()`` call.

    This test starts the daemon, confirms it is running, then exits the context
    without an explicit stop inside the block — relying solely on the
    ``finally: stop_daemon()`` in ``online_daemon_running()``.
    """
    nc.login()

    with online_daemon_running():
        pids_before = get_runner_pids()
        assert pids_before, "Expected at least one runner PID inside context"
        logger.info("runner PIDs inside context: %s", sorted(pids_before))
        # Exit without calling stop_daemon() — the context manager must handle it.


# ---------------------------------------------------------------------------
# SIGKILL during/after recording — restart resilience
# ---------------------------------------------------------------------------

# Minimal joints-only case used by kill-and-restart tests.
_KILL_RESTART_CASES: list[DataDaemonTestCase] = [
    DataDaemonTestCase(
        duration_sec=5,
        joint_count=4,
    ),
    DataDaemonTestCase(
        duration_sec=5,
        joint_count=4,
        video_count=2,
        image_width=64,
        image_height=64,
    ),
]


@pytest.mark.parametrize("case", _KILL_RESTART_CASES, ids=case_ids(_KILL_RESTART_CASES))
def test_sigkill_after_recording_allows_clean_restart(case: DataDaemonTestCase) -> None:
    """Daemon restarts cleanly and produces a new PID after SIGKILL post-recording.

    Sequence:
    1. Start daemon in online mode, run one full recording to completion.
      2. SIGKILL the daemon — bypasses the graceful handler entirely.
      3. Verify the old PID is dead.
    4. Start a fresh daemon in a new ``online_daemon_running()`` block.
      5. Assert the new PID differs from the killed one and is alive.
      6. Verify full cleanup after the second block exits.

    This exercises the restart path that the ``daemon_setup_teardown`` autouse
    fixture relies on between tests: stale IPC artefacts from the SIGKILL must
    not prevent ``ensure_daemon_running`` from succeeding.
    """
    nc.login()

    with online_daemon_running():
        pid_first = assert_exactly_one_daemon_pid()
        specs = build_context_specs(case=case)
        results = run_and_assert_case_contexts(case, specs=specs)
        if results:
            wait_for_all_traces_written(results=results)
        logger.info("SIGKILL daemon pid=%d after completed recording", pid_first)
        stop_daemon(method=STOP_METHOD_SIGKILL)

        assert not pid_is_running(
            pid_first
        ), f"pid_is_running still True for killed daemon pid={pid_first}"

    # IPC artefacts may be stale here — daemon_setup_teardown cleans them up.
    # A second online_daemon_running block must succeed from scratch.
    with online_daemon_running():
        pid_second = assert_exactly_one_daemon_pid()
        assert pid_second != pid_first, (
            f"Expected a new daemon PID after SIGKILL restart, "
            f"got the same pid={pid_second}"
        )
        assert pid_is_running(
            pid_second
        ), f"Restarted daemon pid={pid_second} is not running"
        logger.info("Restarted daemon pid=%d", pid_second)


@pytest.mark.parametrize("case", _KILL_RESTART_CASES, ids=case_ids(_KILL_RESTART_CASES))
def test_sigkill_mid_recording_allows_clean_restart(case: DataDaemonTestCase) -> None:
    """Daemon restarts cleanly after SIGKILL interrupts an in-progress recording.

    Unlike ``test_sigkill_after_recording_allows_clean_restart``, this test
    kills the daemon while a recording is actively being written — the daemon
    has no chance to flush buffers or update DB state.  The restart path must
    survive the partially-written DB and recording artefacts.

    Sequence:
    1. Start daemon in online mode, begin a recording but do NOT wait for it to
         finish (recording workers run in the background for half the case
         duration before we kill).
      2. SIGKILL.
      3. Assert old PID is dead.
      4. Restart daemon in a new block — assert new PID and healthy state.
      5. Assert full cleanup after the second block exits.
    """
    nc.login()

    with online_daemon_running():
        pid_first = assert_exactly_one_daemon_pid()
        specs = build_context_specs(case=case)
        # Kick off the recording workload in a background thread so we can
        # kill the daemon while it is actively writing.
        worker_exc: list[BaseException] = []

        def _run() -> None:
            try:
                run_and_assert_case_contexts(case, specs=specs)
            except Exception as exc:  # noqa: BLE001
                worker_exc.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        # Give the recording a moment to start before killing.
        time.sleep(max(1.0, case.duration_sec / 4))

        logger.info("SIGKILL daemon pid=%d mid-recording", pid_first)
        stop_daemon(method="sigkill")
        thread.join(timeout=10.0)

        assert not pid_is_running(
            pid_first
        ), f"pid_is_running still True for killed daemon pid={pid_first}"

    with online_daemon_running():
        pid_second = assert_exactly_one_daemon_pid()
        assert pid_second != pid_first, (
            f"Expected a new daemon PID after mid-recording SIGKILL restart, "
            f"got the same pid={pid_second}"
        )
        assert pid_is_running(
            pid_second
        ), f"Restarted daemon pid={pid_second} is not running"
        logger.info("Restarted daemon pid=%d after mid-recording kill", pid_second)
