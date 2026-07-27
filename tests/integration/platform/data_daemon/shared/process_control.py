"""Daemon process control primitives for integration tests.

Provides process introspection helpers and all stop/kill/wait utilities.
No assertions, no profile management, no storage management — sits at the
bottom of the import graph so that all other shared modules can import from
here without cycles.
"""

from __future__ import annotations

import functools
import gc
import logging
import logging.handlers
import math
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
    STOP_METHOD_CLI,
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


def _percentile(ordered: list[float], percentile: float) -> float:
    """Return the nearest-rank percentile of an already-sorted list."""
    rank = math.ceil(percentile / 100.0 * len(ordered))
    return ordered[max(0, min(len(ordered), rank) - 1)]


class ScheduleMonitor:
    """Collect producer wake lateness across a recording.

    Samples every deadline instead of raising on the first breach, so a single
    OS or garbage-collection stall does not abandon a long recording before it
    has produced any distribution to read. The percentile check bounds
    sustained lag while tolerating isolated outliers.

    Attributes:
        tolerance_s: Lateness in seconds each sample is compared against.
        percentile: Percentile of samples that must stay within tolerance.
    """

    def __init__(self, tolerance_s: float, percentile: float) -> None:
        self.tolerance_s = tolerance_s
        self.percentile = percentile
        self._samples: dict[str, list[float]] = {}
        self._worst: dict[str, tuple[float, str, float]] = {}

    def record(
        self,
        deadline: float,
        label: str,
        prior_label: str = "",
        prior_elapsed_s: float = 0.0,
    ) -> None:
        """Record how late this deadline fired."""
        lateness = time.time() - deadline
        self._samples.setdefault(label, []).append(lateness)
        if lateness > self._worst.get(label, (float("-inf"), "", 0.0))[0]:
            self._worst[label] = (lateness, prior_label, prior_elapsed_s)

    def summary(self) -> str:
        """Render one line per stream with the lateness distribution."""
        lines = []
        for label in sorted(self._samples):
            ordered = sorted(self._samples[label])
            over = sum(1 for x in ordered if abs(x) > self.tolerance_s)
            worst, prior_label, prior_s = self._worst[label]
            at_percentile = _percentile(ordered, self.percentile) * 1000
            lines.append(
                f"    {label:<22} n={len(ordered):5d}"
                f"  p50={_percentile(ordered, 50) * 1000:7.1f}ms"
                f"  p{self.percentile:g}={at_percentile:7.1f}ms"
                f"  max={worst * 1000:7.1f}ms"
                f"  over_tolerance={over}"
                f"  (worst followed {prior_label or 'nothing'}"
                f" taking {prior_s * 1000:.1f}ms)"
            )
        return "\n".join(lines)

    def assert_within_budget(self) -> None:
        """Raise when a stream's percentile lateness exceeds the tolerance."""
        for label in sorted(self._samples):
            ordered = sorted(self._samples[label])
            at_percentile = _percentile(ordered, self.percentile)
            assert abs(at_percentile) <= self.tolerance_s, (
                f"{label} ran behind schedule: "
                f"p{self.percentile:g}={at_percentile:+.3f}s exceeds "
                f"tolerance=±{self.tolerance_s:.3f}s over {len(ordered)} frames"
            )


class GcPauseRecorder:
    """Record cyclic-collector pause times for the calling process.

    A generation-2 pass holds the GIL for the whole collection, so it delays
    every other thread in the process, including a producer waiting to wake
    from time.sleep. Pause cost grows with the number of tracked objects.
    """

    def __init__(self) -> None:
        self.pauses: list[tuple[int, float]] = []
        self._open: dict[int, float] = {}

    def _callback(self, phase: str, info: dict[str, int]) -> None:
        generation = info["generation"]
        if phase == "start":
            self._open[generation] = time.perf_counter()
            return
        started = self._open.pop(generation, None)
        if started is not None:
            self.pauses.append((generation, time.perf_counter() - started))

    def __enter__(self) -> GcPauseRecorder:
        gc.callbacks.append(self._callback)
        return self

    def __exit__(self, *args: object) -> None:
        if self._callback in gc.callbacks:
            gc.callbacks.remove(self._callback)

    def summary(self) -> str:
        """Render per-generation pause counts and the worst pause seen."""
        if not self.pauses:
            return "    no collections"
        lines = []
        for generation in sorted({g for g, _ in self.pauses}):
            durations = sorted(d for g, d in self.pauses if g == generation)
            lines.append(
                f"    gen-{generation}  runs={len(durations):5d}"
                f"  p50={_percentile(durations, 50) * 1000:7.1f}ms"
                f"  max={durations[-1] * 1000:7.1f}ms"
                f"  total={sum(durations) * 1000:8.1f}ms"
            )
        return "\n".join(lines)


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
    """Return the PIDs of all running neuracore data-daemon runner processes.

    Matches either the Python runner entry point
    (``neuracore.data_daemon.runner_entry``) or the bundled Rust binary
    (``neuracore/data_daemon/bin/data-daemon``) — the latter is what runs
    when ``NCD_RUST_DAEMON=1`` and the wheel includes the compiled binary.
    """
    env = {**os.environ, "COLUMNS": "32768"}
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True, env=env)
    runner_pids: set[int] = set()
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, args = parts
        if (
            "neuracore.data_daemon.runner_entry" in args
            or "neuracore/data_daemon/bin/data-daemon" in args
        ):
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
    socket_path = Path(SOCKET_PATH)
    deadline = time.monotonic() + timeout_s

    while True:
        live_pids = _collect_candidate_pids()
        pid_file_gone = not pid_path.exists()
        socket_gone = not socket_path.exists()

        if not live_pids and pid_file_gone and socket_gone:
            return

        if time.monotonic() >= deadline:
            still_running = {p for p in live_pids if pid_is_running(p)}
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
