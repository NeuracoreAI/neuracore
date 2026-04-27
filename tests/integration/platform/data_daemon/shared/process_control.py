"""Daemon process control primitives for integration tests.

Provides process introspection helpers and all stop/kill/wait utilities.
No assertions, no profile management, no storage management — sits at the
bottom of the import graph so that all other shared modules can import from
here without cycles.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from neuracore.data_daemon.const import SOCKET_PATH
from neuracore.data_daemon.helpers import (
    get_daemon_pid_path,
    get_daemon_recordings_root_path,
)
from neuracore.data_daemon.lifecycle.daemon_os_control import (
    ensure_daemon_running,
    force_kill,
    pid_is_running,
    read_pid_from_file,
    terminate_pid,
    wait_for_exit,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STOP_METHOD_SIGINT,
    STOP_METHOD_SIGKILL,
    STOP_METHOD_SIGTERM,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

MAX_TIME_TO_START_S = 20
"""Maximum seconds allowed for a daemon-startup or API-handshake operation."""

MAX_TIME_TO_LOG_S = 0.5
"""Maximum seconds allowed for a single data-logging call."""

LEAST_TIME_TO_STOP_S = 10
"""Minimum seconds expected for a recording stop."""

HIGH_TIME_TO_DATASET_READY_S = 500
"""Upper bound on waiting for an online dataset to become ready, in seconds."""


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


class Timer:
    """Context manager that measures wall-clock elapsed time for a block.

    Accumulates per-label statistics (count, total, max) in the class-level
    ``_stats`` dictionary so that test suites can report aggregate timing at
    the end of a run.  Optionally asserts that the block completed within
    ``max_time`` seconds.

    Attributes:
        _stats: Class-level dict mapping label strings to aggregate timing
            statistics with keys ``"count"``, ``"total"``, and ``"max"``.
        max_time: Upper time limit in seconds.
        label: Human-readable name for this timer.  Pass ``None`` to skip
            stat accumulation.
        always_log: When ``True``, log the elapsed time even if below
            ``max_time``.
        log_threshold: Log at INFO level when elapsed time meets or exceeds
            this value.  ``None`` disables.
        assert_limit: When ``True`` (default), raise ``AssertionError`` if
            the block exceeds ``max_time``.  Set to ``False`` to log only.
    """

    _stats: dict[str, dict[str, float]] = {}

    def __init__(
        self,
        max_time: float = MAX_TIME_TO_LOG_S,
        label: str | None = None,
        always_log: bool = False,
        log_threshold: float | None = None,
        assert_limit: bool = True,
        deadline: float | None = None,
        timing_tolerance: float | None = None,
    ) -> None:
        self.max_time = max_time
        self.label = label
        self.always_log = always_log
        self.log_threshold = log_threshold
        self.assert_limit = assert_limit
        self.deadline = deadline
        self.timing_tolerance = timing_tolerance

    def __enter__(self) -> Timer:
        self.wall_start = time.time()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> bool | None:
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        had_exception = len(args) > 0 and args[0] is not None
        if self.label:
            stats = self._stats.setdefault(
                self.label, {"count": 0.0, "total": 0.0, "max": 0.0}
            )
            stats["count"] += 1
            stats["total"] += self.interval
            stats["max"] = max(stats["max"], self.interval)

            should_log = self.always_log
            if self.log_threshold is not None and self.interval >= self.log_threshold:
                should_log = True
            if self.interval >= self.max_time:
                should_log = True

            if should_log:
                level = (
                    logging.WARNING if self.interval >= self.max_time else logging.INFO
                )
                logger.log(
                    level,
                    "Timer %-32s %.3fs (limit=%.3fs)",
                    self.label,
                    self.interval,
                    self.max_time,
                )

        if had_exception:
            return False

        if self.assert_limit:
            if self.deadline is not None and self.timing_tolerance is not None:
                lateness = self.wall_start - self.deadline
                assert abs(lateness) <= self.timing_tolerance, (
                    f"{self.label or 'Function'} logged at wrong moment: "
                    f"lateness={lateness:+.3f}s, "
                    f"tolerance=±{self.timing_tolerance:.3f}s"
                )
            assert self.interval < self.max_time, (
                f"{self.label or 'Function'} took too long: "
                f"{self.interval:.3f}s >= {self.max_time:.3f}s"
            )
        return None

    @classmethod
    def merge_stats(cls, stats: dict[str, dict[str, float]]) -> None:
        """Merge external timer stats (e.g. from a worker process) into the accumulator."""  # noqa: E501
        for label, incoming in stats.items():
            existing = cls._stats.setdefault(
                label, {"count": 0.0, "total": 0.0, "max": 0.0}
            )
            existing["count"] += incoming["count"]
            existing["total"] += incoming["total"]
            existing["max"] = max(existing["max"], incoming["max"])


# ---------------------------------------------------------------------------
# Process introspection helpers
# ---------------------------------------------------------------------------


def get_runner_pids() -> set[int]:
    """Return the PIDs of all running neuracore data-daemon runner processes."""
    output = subprocess.check_output(
        ["ps", "-eo", "pid=,args="], text=True, env={**os.environ, "COLUMNS": "32768"}
    )
    runner_pids: set[int] = set()
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, args = parts
        if "neuracore.data_daemon.runner_entry" in args:
            runner_pids.add(int(pid_text))
    return runner_pids


def _get_living_runner_pids() -> set[int]:
    """Return daemon runner PIDs that are actually running (excludes zombies/dead)."""
    living_pids: set[int] = set()
    for pid in get_runner_pids():
        if pid_is_running(pid):
            living_pids.add(pid)
    return living_pids


def _live_daemon_pids() -> set[int]:
    """Return PIDs of all live daemon processes (runner and PID-file)."""
    pid_path = get_daemon_pid_path()
    pids: set[int] = _get_living_runner_pids()
    stored_pid = read_pid_from_file(pid_path)
    if stored_pid is not None and pid_is_running(stored_pid):
        pids.add(stored_pid)
    return pids


# ---------------------------------------------------------------------------
# Daemon control helpers
# ---------------------------------------------------------------------------


def _collect_candidate_pids() -> set[int]:
    """Return all daemon PIDs that need to be waited on or killed."""
    pids: set[int] = set(get_runner_pids())
    pid_file_value = read_pid_from_file(get_daemon_pid_path())
    if pid_file_value is not None:
        pids.add(pid_file_value)
    return pids


def _send_initial_stop(method: str, candidate_pids: set[int]) -> None:
    """Deliver the initial stop signal or CLI command for ``method``."""
    if method == "cli":
        subprocess.run(
            [sys.executable, "-m", "neuracore.data_daemon", "stop"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif method == STOP_METHOD_SIGTERM:
        for pid in sorted(candidate_pids):
            if pid_is_running(pid):
                terminate_pid(pid)
    elif method == STOP_METHOD_SIGINT:
        for pid in sorted(candidate_pids):
            if pid_is_running(pid):
                try:
                    os.kill(pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
    elif method == STOP_METHOD_SIGKILL:
        for pid in sorted(candidate_pids):
            if pid_is_running(pid):
                force_kill(pid)
    else:
        raise ValueError(f"Unknown stop method: {method!r}")


def _wait_and_escalate(
    candidate_pids: set[int],
    *,
    graceful_timeout_s: float,
    kill_timeout_s: float = 5.0,
) -> None:
    """Wait for each PID to exit, escalating to SIGKILL on timeout."""
    for pid in sorted(candidate_pids):
        if not pid_is_running(pid):
            continue
        if wait_for_exit(pid, timeout_s=graceful_timeout_s):
            continue
        with Timer(kill_timeout_s, label="stop_daemon_escalated", assert_limit=False):
            force_kill(pid)
            if not wait_for_exit(pid, timeout_s=kill_timeout_s):
                logger.warning(
                    "PID %d did not exit after SIGKILL - " "process may be in D-state",
                    pid,
                )


def _remove_ipc_artefacts() -> None:
    """Remove the PID file and Unix socket, ignoring missing-file errors."""
    pid_path = get_daemon_pid_path()
    socket_path = Path(SOCKET_PATH)
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        socket_path.unlink(missing_ok=True)
    except OSError:
        pass


def stop_daemon(
    *,
    method: str = "cli",
    graceful_timeout_s: float = 10.0,
) -> None:
    """Stop all daemon processes and clean up IPC artefacts.

    Args:
        method: One of ``"cli"``, ``"sigterm"``, ``"sigint"``, ``"sigkill"``.
        graceful_timeout_s: Seconds to wait for graceful exit before escalating
            to SIGKILL.  Ignored when ``method="sigkill"``.
    """
    with Timer(15.0, label=f"stop_daemon[{method}]", assert_limit=False):
        candidate_pids = _collect_candidate_pids()
        _send_initial_stop(method, candidate_pids)
        if method == STOP_METHOD_SIGKILL:
            for pid in sorted(candidate_pids):
                wait_for_exit(pid, timeout_s=5.0)
        else:
            _wait_and_escalate(candidate_pids, graceful_timeout_s=graceful_timeout_s)
        _remove_ipc_artefacts()


def collect_daemon_pids_from_parallel_startup(worker_count: int) -> list[int]:
    """Start ``worker_count`` daemon instances in parallel and collect their PIDs.

    Args:
        worker_count: Number of parallel worker processes to spawn.

    Returns:
        A list of the PID returned by each worker.
    """

    def worker(barrier: object, results: dict[int, int], index: int) -> None:
        barrier.wait()
        results[index] = ensure_daemon_running()

    barrier = multiprocessing.Barrier(worker_count)
    manager = multiprocessing.Manager()
    results = manager.dict()
    processes = []

    for index in range(worker_count):
        process = multiprocessing.Process(target=worker, args=(barrier, results, index))
        process.start()
        processes.append(process)

    for process in processes:
        process.join(timeout=25)
        assert (
            not process.is_alive()
        ), f"worker process {process.pid} did not finish before timeout"
        assert (
            process.exitcode == 0
        ), f"worker process {process.pid} exited with code {process.exitcode}"

    return list(results.values())


def delete_recordings_folder() -> None:
    """Delete the daemon's recordings folder and all its contents."""
    recordings_root = get_daemon_recordings_root_path()
    if recordings_root.exists():
        import shutil

        shutil.rmtree(recordings_root, ignore_errors=True)


def wait_for_daemon_shutdown(
    *, timeout_s: float = 30.0, poll_interval_s: float = 0.5
) -> None:
    """Block until all daemon processes have exited and IPC artefacts are gone.

    Args:
        timeout_s: Maximum seconds to wait before raising :class:`TimeoutError`.
        poll_interval_s: Seconds between consecutive polls.

    Raises:
        TimeoutError: When the daemon has not fully exited within ``timeout_s``.
    """
    pid_path = get_daemon_pid_path()
    socket_path = Path(SOCKET_PATH)
    deadline = time.monotonic() + timeout_s

    while True:
        live_pids = _live_daemon_pids()
        pid_file_gone = not pid_path.exists()
        socket_gone = not socket_path.exists()

        if not live_pids and pid_file_gone and socket_gone:
            return

        if time.monotonic() >= deadline:
            still_running = live_pids
            details: list[str] = []
            if still_running:
                details.append(f"live PIDs: {sorted(still_running)}")
            if not pid_file_gone:
                details.append(f"PID file still present: {pid_path}")
            if not socket_gone:
                details.append(f"socket still present: {socket_path}")
            raise TimeoutError(
                f"Daemon did not shut down within {timeout_s}s — " + ", ".join(details)
            )

        time.sleep(poll_interval_s)
