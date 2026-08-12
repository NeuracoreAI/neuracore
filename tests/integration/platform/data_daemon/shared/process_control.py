"""Daemon process control primitives for integration tests.

Provides process introspection helpers and all stop/kill/wait utilities.
No assertions, no profile management, no storage management — sits at the
bottom of the import graph so that all other shared modules can import from
here without cycles.
"""

from __future__ import annotations

import faulthandler
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

from neuracore.core.utils.http_session import http_time_snapshot
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

STACK_DUMP_MIN_LIMIT_S = 5.0
"""Smallest ``Timer.max_time`` that arms a stack dump on breach.

``faulthandler.dump_traceback_later`` is a single per-process timer, so arming it
costs a rearm on every enter and exit. Per-frame logging timers run tens of
thousands of times per test at sub-second limits, so they stay out; the
coarse-grained operations where a hang is worth a stack trace stay in.
"""

_dump_deadlines: list[float] = []
"""Monotonic breach deadlines of the currently open Timers, innermost last.

Timers nest, so the single process-wide dump timer is always armed for the
earliest deadline still outstanding.
"""


def _rearm_stack_dump() -> None:
    """Point the process-wide stack-dump timer at the earliest open deadline."""
    if not _dump_deadlines:
        faulthandler.cancel_dump_traceback_later()
        return
    remaining = min(_dump_deadlines) - time.monotonic()
    faulthandler.dump_traceback_later(max(remaining, 0.001), exit=False)


LATENCY_BUCKETS_S: tuple[float, ...] = (
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)
"""Upper bounds of the latency histogram, in seconds, log-spaced."""

BUCKET_KEYS: tuple[str, ...] = tuple(f"le_{bound}" for bound in LATENCY_BUCKETS_S) + (
    "gt_max",
)
"""Stat keys for the histogram, one per bucket plus an overflow bucket."""


def _empty_stats() -> dict[str, float]:
    """Return a zeroed stat accumulator for one label."""
    stats = {"count": 0.0, "total": 0.0, "max": 0.0}
    stats.update({key: 0.0 for key in BUCKET_KEYS})
    return stats


def _bucket_key(interval: float) -> str:
    """Return the histogram key *interval* falls in."""
    for bound, key in zip(LATENCY_BUCKETS_S, BUCKET_KEYS):
        if interval <= bound:
            return key
    return "gt_max"


def percentile_upper_bound(stats: dict[str, float], quantile: float) -> float | None:
    """Return the bucket upper bound containing *quantile* of the samples.

    ``None`` when the samples overflow the largest bucket, or the label has no
    histogram.
    """
    count = stats.get("count", 0.0)
    if count <= 0:
        return None
    target = quantile * count
    seen = 0.0
    for bound, key in zip(LATENCY_BUCKETS_S, BUCKET_KEYS):
        seen += stats.get(key, 0.0)
        if seen >= target:
            return bound
    return None


class Timer:
    """Context manager that measures wall-clock elapsed time for a block.

    Accumulates per-label statistics in the class-level ``_stats`` so a suite
    can report aggregate timing, and optionally asserts the block ran within
    ``max_time``.

    Every logged line and assertion message apportions the elapsed time between
    HTTP requests this thread made and everything else, so a breach states on its
    face whether the backend was involved.  Blocks whose limit is at least
    ``STACK_DUMP_MIN_LIMIT_S`` also dump every thread's stack to stderr at the
    moment they breach, while the block is still stuck.

    Attributes:
        _stats: Class-level dict mapping label strings to aggregate timing
            statistics with keys ``"count"``, ``"total"``, ``"max"``, and one
            per :data:`BUCKET_KEYS` entry.
        _stats_lock: Guards :attr:`_stats`, which several producer threads
            accumulate into at once.
        max_time: Upper time limit in seconds.
        label: Human-readable name for this timer.  Pass ``None`` to skip
            stat accumulation.
        always_log: When ``True``, log the elapsed time even if below
            ``max_time``.
        log_threshold: Log at INFO level when elapsed time meets or exceeds
            this value.  ``None`` disables.
        assert_deadline: When ``True`` (default), raise ``AssertionError`` if
            the block exceeds ``max_time``.  Set to ``False`` to log only.
        http_calls: Requests this thread issued inside the block.  Set on exit.
        http_seconds: Seconds those requests took.  Set on exit.
        dumps_stack: Whether this timer armed the stack dump.
    """

    _stats: dict[str, dict[str, float]] = {}
    _stats_lock = threading.Lock()

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
        self.http_calls_start, self.http_seconds_start = http_time_snapshot()
        self.dumps_stack = self.max_time >= STACK_DUMP_MIN_LIMIT_S
        if self.dumps_stack:
            self.dump_deadline = time.monotonic() + self.max_time
            _dump_deadlines.append(self.dump_deadline)
            _rearm_stack_dump()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> bool | None:
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        if self.dumps_stack:
            try:
                _dump_deadlines.remove(self.dump_deadline)
            except ValueError:
                pass
            _rearm_stack_dump()
        http_calls, http_seconds = http_time_snapshot()
        self.http_calls = http_calls - self.http_calls_start
        self.http_seconds = http_seconds - self.http_seconds_start
        had_exception = len(args) > 0 and args[0] is not None
        if self.label:
            with self._stats_lock:
                stats = self._stats.setdefault(self.label, _empty_stats())
                stats["count"] += 1
                stats["total"] += self.interval
                stats["max"] = max(stats["max"], self.interval)
                stats[_bucket_key(self.interval)] += 1

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
                    "Timer %-32s %.3fs (limit=%.3fs) %s",
                    self.label,
                    self.interval,
                    self.max_time,
                    self._attribution(),
                )

        if had_exception:
            return False

        if self.assert_deadline:
            assert self.interval < self.max_time, (
                f"{self.label or 'Function'} took too long: "
                f"{self.interval:.3f}s >= {self.max_time:.3f}s "
                f"{self._attribution()}"
            )
        return None

    def _attribution(self) -> str:
        """Split the measured time between waiting on the API and local work.

        Names whichever side dominates so a breach does not need cross-referencing
        against backend request logs to apportion. HTTP time covers only requests
        this thread issued, so a block whose work happens in worker processes or
        threads reports its own HTTP time as near zero.

        Returns:
            A bracketed summary of HTTP time, local time and the dominant side.
        """
        local = max(self.interval - self.http_seconds, 0.0)
        dominant = "http" if self.http_seconds > local else "local"
        return (
            f"[http {self.http_seconds:.3f}s over {self.http_calls} calls, "
            f"local {local:.3f}s -> {dominant}-bound]"
        )

    @classmethod
    def merge_stats(cls, stats: dict[str, dict[str, float]]) -> None:
        """Merge external timer stats (e.g. from a worker process) into the accumulator."""  # noqa: E501
        with cls._stats_lock:
            for label, incoming in stats.items():
                existing = cls._stats.setdefault(label, _empty_stats())
                existing["count"] += incoming["count"]
                existing["total"] += incoming["total"]
                existing["max"] = max(existing["max"], incoming["max"])
                for key in BUCKET_KEYS:
                    existing[key] += incoming.get(key, 0.0)


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
