"""Daemon process control primitives for integration tests.

Provides process introspection helpers and all stop/kill/wait utilities.
No assertions, no profile management, no storage management — sits at the
bottom of the import graph so that all other shared modules can import from
here without cycles.
"""

from __future__ import annotations

import functools
import logging
import logging.handlers
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager

import requests

from neuracore.data_daemon.daemon_control import (
    ensure_daemon_running,
    pid_is_running,
    read_pid_from_file,
)
from neuracore.data_daemon.helpers import (
    get_daemon_pid_path,
    get_daemon_recordings_root_path,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STOP_METHOD_CLI,
    STOP_METHOD_SIGINT,
    STOP_METHOD_SIGKILL,
    STOP_METHOD_SIGTERM,
)

# cspell:ignore WNOHANG waitpid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

# cspell:ignore nstat retrans unacked

HANG_DIAGNOSTICS_ENV = "NCD_HANG_DIAGNOSTICS"
"""Set to ``0`` to suppress socket and stack dumps for calls that are stuck.

On by default: the stall this captures is rare and leaves no server-side trace,
so a run that happens to hit one is worth catching first time.
"""

HANG_DIAGNOSTIC_MIN_BUDGET_S = 10.0
"""Only watch calls whose own budget is at least this long.

:class:`Timer` wraps every ``nc.log_*`` call — tens of thousands per case — so
arming timers indiscriminately would cost far more than the diagnostics are
worth. The stalls under investigation are cloud calls with budgets measured in
tens of seconds, which this admits while excluding the per-frame logging path.
"""

HANG_DIAGNOSTIC_DELAYS_S = (15.0, 60.0, 240.0)
"""Seconds after a timed call starts at which to capture state, if still running.

A stalled HTTP call leaves no trace in any server log — the request never
arrives — so the only place the cause is visible is the client's own socket
while it is still stuck. Several samples show whether the kernel is
retransmitting and how the retry counters evolve.
"""


def _run_diagnostic(command: list[str]) -> str:
    """Return a diagnostic command's output, or a short note on why it failed."""
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
    except FileNotFoundError:
        return f"({command[0]} not available)"
    except subprocess.SubprocessError as error:
        return f"({command[0]} failed: {error})"
    return (completed.stdout or completed.stderr or "").strip() or "(no output)"


def _blocked_thread_stacks() -> str:
    """Return a compact stack for every live thread."""
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    chunks = []
    for thread_id, frame in sys._current_frames().items():
        stack = traceback.format_stack(frame)[-6:]
        chunks.append(
            f"  thread {names.get(thread_id, thread_id)}:\n" + "".join(stack).rstrip()
        )
    return "\n".join(chunks)


def _dump_hang_diagnostics(label: str | None, waited_s: float) -> None:
    """Log socket state, TCP counters and stacks for a call still in flight."""
    if sys.platform == "darwin":
        sockets = _run_diagnostic(["netstat", "-an", "-p", "tcp"])
        counters = _run_diagnostic(["netstat", "-s", "-p", "tcp"])
    else:
        # -t TCP, -i internal info (rto, retransmits, unacked), -n numeric.
        sockets = _run_diagnostic(["ss", "-tin"])
        counters = _run_diagnostic(["nstat", "-az"])
    keep = [
        line
        for line in sockets.splitlines()
        if ":443" in line or "retrans" in line or "rto:" in line
    ]
    # Never come back empty: a capture that filtered everything out is worth
    # less than raw output, and there is no second chance at a rare stall.
    if not keep:
        keep = sockets.splitlines()[:60]
    tcp_counters = [
        line
        for line in counters.splitlines()
        if any(k in line for k in ("Retrans", "Timeout", "retransmit", "timeout"))
    ]
    logger.warning(
        "HANG DIAGNOSTICS %s still running after %.0fs\n"
        "-- sockets to :443 --\n%s\n"
        "-- tcp counters --\n%s\n"
        "-- stacks --\n%s",
        label or "call",
        waited_s,
        "\n".join(keep[:40]) or "(none)",
        "\n".join(tcp_counters[:20]) or "(none)",
        _blocked_thread_stacks(),
    )


def arm_hang_watchdogs(
    label: str | None, budget_s: float
) -> tuple[threading.Timer, ...]:
    """Schedule state captures for a call that has not returned yet.

    Args:
        label: The timer's label, used to identify the stuck call.
        budget_s: The call's own deadline. Calls budgeted below
            :data:`HANG_DIAGNOSTIC_MIN_BUDGET_S` are not watched.

    Returns:
        The scheduled timers, so the caller cancels them once the call
        completes. Empty when diagnostics are off or the call is short.
    """
    if os.environ.get(HANG_DIAGNOSTICS_ENV, "1") != "1":
        return ()
    if budget_s < HANG_DIAGNOSTIC_MIN_BUDGET_S:
        return ()
    watchdogs = []
    for delay_s in HANG_DIAGNOSTIC_DELAYS_S:
        watchdog = threading.Timer(delay_s, _dump_hang_diagnostics, (label, delay_s))
        watchdog.daemon = True
        watchdog.start()
        watchdogs.append(watchdog)
    return tuple(watchdogs)


def _install_http_hang_watchdog() -> None:
    """Watch every outbound HTTP request, not just the timed ones.

    Hooking :class:`Timer` alone would miss any call nobody wrapped — the
    dataset synchronisation path, for one — so the wrapper goes on the session
    itself. Applied at import so the pool's spawned workers, which import this
    module too, are covered as well as the main process.
    """
    if os.environ.get(HANG_DIAGNOSTICS_ENV, "1") != "1":
        return
    if getattr(requests.Session, "_ncd_hang_wrapped", False):
        return
    original_request = requests.Session.request

    @functools.wraps(original_request)
    def request_with_watchdog(
        self: requests.Session, method: str, url: str, *args: object, **kwargs: object
    ) -> requests.Response:
        watchdogs = arm_hang_watchdogs(
            f"HTTP {method} {str(url)[:90]}", HANG_DIAGNOSTIC_MIN_BUDGET_S
        )
        try:
            return original_request(self, method, url, *args, **kwargs)
        finally:
            for watchdog in watchdogs:
                watchdog.cancel()

    requests.Session.request = request_with_watchdog  # type: ignore[method-assign]
    requests.Session._ncd_hang_wrapped = True  # type: ignore[attr-defined]


_install_http_hang_watchdog()


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
        assert_deadline: When ``True`` (default), raise ``AssertionError`` if
            the block exceeds ``max_time``.  Set to ``False`` to log only.
    """

    _stats: dict[str, dict[str, float]] = {}

    def __init__(
        self,
        max_time: float = MAX_TIME_TO_LOG_S,
        label: str | None = None,
        always_log: bool = False,
        log_threshold: float | None = None,
        assert_deadline: bool = True,
    ) -> None:
        self.max_time = max_time
        self.label = label
        self.always_log = always_log
        self.log_threshold = log_threshold
        self.assert_deadline = assert_deadline

    def __enter__(self) -> Timer:
        self.wall_start = time.time()
        self.start = time.perf_counter()
        self._hang_watchdogs = arm_hang_watchdogs(self.label, self.max_time)
        return self

    def __exit__(self, *args: object) -> bool | None:
        for watchdog in getattr(self, "_hang_watchdogs", ()):
            watchdog.cancel()
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

        if self.assert_deadline:
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


def surface_worker_errors(fn):
    """Wrap a subprocess worker entry point so failures survive pickling.

    Exceptions raised in a ``multiprocessing.Pool`` worker are pickled back to
    the parent, which drops the traceback and ``__cause__`` chain and fails
    outright for exceptions that are not picklable. Re-raise as a plain
    ``RuntimeError`` whose message embeds the worker's full formatted
    traceback (chained causes included) so the parent's failure report shows
    the real error. The message names the worker process to match the
    ``processName`` column of the relayed live-log lines, so a failure can
    be joined against the worker's log output.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            raise RuntimeError(
                f"Worker {fn.__name__} "
                f"({multiprocessing.current_process().name}, pid {os.getpid()}) "
                f"failed with the traceback below\n{traceback.format_exc()}"
            ) from None

    return wrapper


def init_worker_logging(log_queue: multiprocessing.Queue, level: int) -> None:
    """Route a pool worker's log records to the parent process.

    Pool initializer. Spawned workers (macOS) start with no logging
    configuration, so their INFO records — Timer lines included — are
    silently dropped; forked workers (Linux) inherit pytest's handlers and
    write into its captured streams from another process. Replacing the
    root handlers with a ``QueueHandler`` ships every record to the parent
    instead, where :func:`relayed_worker_logs` replays them through the
    parent's handlers so worker Timer lines appear in the live log exactly
    like single-context runs.
    """
    root = logging.getLogger()
    root.handlers[:] = [logging.handlers.QueueHandler(log_queue)]
    root.setLevel(level)


@contextmanager
def relayed_worker_logs() -> Generator[multiprocessing.Queue]:
    """Replay pool-worker log records through this process's handlers."""
    log_queue: multiprocessing.Queue = multiprocessing.Queue()

    def _relay() -> None:
        while True:
            record = log_queue.get()
            if record is None:
                return
            logging.getLogger(record.name).handle(record)

    thread = threading.Thread(target=_relay, name="worker-log-relay", daemon=True)
    thread.start()
    try:
        yield log_queue
    finally:
        log_queue.put(None)
        thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Process introspection helpers
# ---------------------------------------------------------------------------


def get_runner_pids() -> set[int]:
    """Return the PIDs of all running neuracore data-daemon processes.

    Matches the bundled daemon binary
    (``neuracore/data_daemon/bin/data-daemon``).
    """
    env = {**os.environ, "COLUMNS": "32768"}
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True, env=env)
    runner_pids: set[int] = set()
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, args = parts
        if "neuracore/data_daemon/bin/data-daemon" in args:
            runner_pids.add(int(pid_text))
    return runner_pids


def _live_daemon_pids() -> set[int]:
    """Return PIDs of all live daemon processes (runner and PID-file)."""
    pid_path = get_daemon_pid_path()
    pids: set[int] = set(get_runner_pids())
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
    if method == STOP_METHOD_CLI:
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


def _wait_and_escalate(candidate_pids: set[int], *, graceful_timeout_s: float) -> None:
    """Wait for each PID to exit, escalating to SIGKILL on timeout."""
    for pid in sorted(candidate_pids):
        if not pid_is_running(pid):
            continue
        if not wait_for_exit(pid, timeout_s=graceful_timeout_s):
            with Timer(5.0, label="stop_daemon_escalated", assert_deadline=False):
                force_kill(pid)
                wait_for_exit(pid, timeout_s=5.0)


def terminate_pid(pid_value: int) -> bool:
    """Send SIGTERM to the given PID."""
    try:
        os.kill(pid_value, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def force_kill(pid_value: int) -> bool:
    """Send SIGKILL to the given PID."""
    try:
        os.kill(pid_value, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def _try_reap_zombie_child(pid_value: int) -> bool:
    """Non-blocking waitpid to reap a zombie child; True when reaped."""
    try:
        reaped_pid, _ = os.waitpid(pid_value, os.WNOHANG)
        return reaped_pid != 0
    except (ChildProcessError, OSError):
        return False


def wait_for_exit(pid_value: int, *, timeout_s: float) -> bool:
    """Wait for a PID to stop running until a timeout elapses."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _try_reap_zombie_child(pid_value) or not pid_is_running(pid_value):
            return True
        time.sleep(0.1)
    return False


def _remove_ipc_artefacts() -> None:
    """Remove the PID file, ignoring missing-file errors."""
    try:
        get_daemon_pid_path().unlink(missing_ok=True)
    except OSError:
        pass


def stop_daemon(
    *,
    method: str = STOP_METHOD_CLI,
    graceful_timeout_s: float = 10.0,
) -> None:
    """Stop all daemon processes and clean up IPC artefacts.

    Args:
        method: One of ``"cli"``, ``"sigterm"``, ``"sigint"``, ``"sigkill"``.
        graceful_timeout_s: Seconds to wait for graceful exit before escalating
            to SIGKILL.  Ignored when ``method="sigkill"``.
    """
    with Timer(15.0, label=f"stop_daemon[{method}]", assert_deadline=False):
        candidate_pids = _collect_candidate_pids()
        _send_initial_stop(method, candidate_pids)
        if method == STOP_METHOD_SIGKILL:
            for pid in sorted(candidate_pids):
                wait_for_exit(pid, timeout_s=5.0)
        else:
            _wait_and_escalate(candidate_pids, graceful_timeout_s=graceful_timeout_s)
        _remove_ipc_artefacts()


def _parallel_startup_worker(
    barrier: object, results: dict[int, int], index: int
) -> None:
    """Wait on the barrier then record the daemon PID for one parallel caller.

    Module-level so it is picklable under the ``spawn`` start method
    (the default on macOS).
    """
    barrier.wait()
    results[index] = ensure_daemon_running()


def collect_daemon_pids_from_parallel_startup(worker_count: int) -> list[int]:
    """Start ``worker_count`` daemon instances in parallel and collect their PIDs.

    Args:
        worker_count: Number of parallel worker processes to spawn.

    Returns:
        A list of the PID returned by each worker.
    """

    barrier = multiprocessing.Barrier(worker_count)
    manager = multiprocessing.Manager()
    results = manager.dict()
    processes = []

    for index in range(worker_count):
        process = multiprocessing.Process(
            target=_parallel_startup_worker, args=(barrier, results, index)
        )
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
    deadline = time.monotonic() + timeout_s

    while True:
        live_pids = _collect_candidate_pids()
        pid_file_gone = not pid_path.exists()

        if not live_pids and pid_file_gone:
            return

        if time.monotonic() >= deadline:
            still_running = {p for p in live_pids if pid_is_running(p)}
            details: list[str] = []
            if still_running:
                details.append(f"live PIDs: {sorted(still_running)}")
            if not pid_file_gone:
                details.append(f"PID file still present: {pid_path}")
            raise TimeoutError(
                f"Daemon did not shut down within {timeout_s}s — " + ", ".join(details)
            )

        time.sleep(poll_interval_s)
