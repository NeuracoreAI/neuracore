"""Low-contention sleep diagnostics correlated with macOS scheduler traces."""

from __future__ import annotations

import atexit
import json
import logging
import os
import platform
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ANOMALIES_PATH_ENV = "NEURACORE_SLEEP_ANOMALIES_PATH"
NORMAL_SAMPLE_EVERY_ENV = "NEURACORE_SLEEP_NORMAL_SAMPLE_EVERY"
DEFAULT_NORMAL_SAMPLE_EVERY = 100
RECENT_SLEEP_COUNT = 256

try:
    from neuracore.data_daemon import _data_bridge
except ImportError:  # pragma: no cover - exercised by source-only environments
    _data_bridge = None


@dataclass(frozen=True, slots=True)
class SleepDiagnosticRecord:
    """One measured sleep, using monotonic nanoseconds for interval timing."""

    sleep_id: int
    label: str
    correlation_id: str
    pid: int
    python_thread_name: str
    python_thread_ident: int | None
    native_thread_id: int
    requested_ns: int
    monotonic_start_ns: int
    expected_deadline_ns: int
    native_wait_start_ns: int | None
    native_wait_return_ns: int | None
    interpreter_resume_ns: int | None
    observed_return_ns: int
    actual_ns: int
    overshoot_ns: int
    native_wait_ns: int | None
    gil_reacquire_ns: int | None
    python_post_native_ns: int | None
    wall_return_ns: int
    cpu_before: int | None
    cpu_after: int | None
    qos_class: int | None
    qos_relative_priority: int | None
    process_nice: int | None
    foreground: bool | None
    native_helper: bool
    native_error: int | None
    anomalous: bool


class _AsyncSleepRecorder:
    """Per-process asynchronous sink; never takes application-owned locks."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.recent: deque[SleepDiagnosticRecord] = deque(maxlen=RECENT_SLEEP_COUNT)
        self.records: queue.SimpleQueue[
            SleepDiagnosticRecord | threading.Event | None
        ] = queue.SimpleQueue()
        self.thread = threading.Thread(
            target=self._run,
            name="sleep-diagnostic-logger",
            daemon=True,
        )
        self.thread.start()

    def submit(self, record: SleepDiagnosticRecord, *, emit: bool) -> None:
        # CPython deque.append and SimpleQueue.put do not run user callbacks or
        # acquire locks used by the producer workload.
        self.recent.append(record)
        if emit:
            self.records.put(record)

    def close(self) -> None:
        self.records.put(None)
        self.thread.join(timeout=2.0)

    def flush(self, timeout_s: float) -> bool:
        complete = threading.Event()
        self.records.put(complete)
        return complete.wait(timeout_s)

    def _run(self) -> None:
        path_value = os.getenv(ANOMALIES_PATH_ENV)
        anomaly_file = None
        try:
            if path_value:
                path = Path(path_value)
                path.parent.mkdir(parents=True, exist_ok=True)
                anomaly_file = path.open("a", encoding="utf-8", buffering=1)
            while True:
                record = self.records.get()
                if record is None:
                    return
                if isinstance(record, threading.Event):
                    if anomaly_file is not None:
                        anomaly_file.flush()
                    record.set()
                    continue
                payload = asdict(record)
                payload["requested_ms"] = record.requested_ns / 1_000_000
                payload["actual_ms"] = record.actual_ns / 1_000_000
                payload["overshoot_ms"] = record.overshoot_ns / 1_000_000
                if record.anomalous:
                    if anomaly_file is not None:
                        anomaly_file.write(
                            json.dumps(payload, separators=(",", ":")) + "\n"
                        )
                    logger.warning(
                        "SLEEP_OVERSHOOT sleep_id=%d label=%s correlation=%s "
                        "pid=%d native_tid=%d requested=%.3fms actual=%.3fms "
                        "overshoot=%.3fms native_wait=%s gil_reacquire=%s "
                        "python_post_native=%s cpu=%s->%s qos=%s/%s nice=%s "
                        "foreground=%s native_error=%s",
                        record.sleep_id,
                        record.label,
                        record.correlation_id,
                        record.pid,
                        record.native_thread_id,
                        payload["requested_ms"],
                        payload["actual_ms"],
                        payload["overshoot_ms"],
                        _format_optional_ns(record.native_wait_ns),
                        _format_optional_ns(record.gil_reacquire_ns),
                        _format_optional_ns(record.python_post_native_ns),
                        record.cpu_before,
                        record.cpu_after,
                        record.qos_class,
                        record.qos_relative_priority,
                        record.process_nice,
                        record.foreground,
                        record.native_error,
                    )
                else:
                    logger.info(
                        "Sleep diagnostic sample sleep_id=%d label=%s "
                        "correlation=%s requested=%.3fms actual=%.3fms "
                        "overshoot=%.3fms native_tid=%d",
                        record.sleep_id,
                        record.label,
                        record.correlation_id,
                        payload["requested_ms"],
                        payload["actual_ms"],
                        payload["overshoot_ms"],
                        record.native_thread_id,
                    )
        except Exception:  # noqa: BLE001
            logger.exception("Sleep diagnostic logger failed")
        finally:
            if anomaly_file is not None:
                anomaly_file.close()


_sleep_ids = count(1)
_recorder: _AsyncSleepRecorder | None = None
_recorder_pid: int | None = None
_recorder_guard = threading.Lock()


def _format_optional_ns(value: int | None) -> str:
    return "unavailable" if value is None else f"{value / 1_000_000:.3f}ms"


def _get_recorder() -> _AsyncSleepRecorder:
    global _recorder, _recorder_pid
    pid = os.getpid()
    if _recorder is not None and _recorder_pid == pid:
        return _recorder
    with _recorder_guard:
        if _recorder is None or _recorder_pid != pid:
            # A fork only preserves the calling thread. Do not touch the
            # inherited queue/thread; replace it lazily in the child.
            _recorder = _AsyncSleepRecorder(pid)
            _recorder_pid = pid
    return _recorder


def _close_recorder() -> None:
    if _recorder is not None and _recorder_pid == os.getpid():
        _recorder.close()


atexit.register(_close_recorder)


def flush_sleep_diagnostics(timeout_s: float = 2.0) -> bool:
    """Drain this process's diagnostic queue before a worker exits."""
    if _recorder is None or _recorder_pid != os.getpid():
        return True
    return _recorder.flush(timeout_s)


def _foreground_status() -> bool | None:
    try:
        return os.getpgrp() == os.tcgetpgrp(0)
    except (AttributeError, OSError):
        return None


def _process_nice() -> int | None:
    try:
        return os.getpriority(os.PRIO_PROCESS, 0)
    except (AttributeError, OSError):
        return None


def _native_sleep(
    seconds: float,
    *,
    sleep_id: int,
    label: str,
    correlation_id: str,
    anomaly_threshold_ms: float,
) -> tuple[int, int, int, int, int | None, int | None, int | None, int | None] | None:
    helper = (
        None
        if _data_bridge is None
        else getattr(_data_bridge, "diagnostic_native_sleep", None)
    )
    if helper is None or platform.system() != "Darwin":
        return None
    result: tuple[Any, ...] = helper(
        seconds,
        sleep_id,
        label,
        correlation_id,
        anomaly_threshold_ms,
    )
    return (
        int(result[0]),
        int(result[1]),
        int(result[2]),
        int(result[3]),
        None if result[4] is None else int(result[4]),
        None if result[5] is None else int(result[5]),
        None if result[6] is None else int(result[6]),
        None if result[7] is None else int(result[7]),
    )


def diagnostic_sleep(
    seconds: float,
    label: str,
    *,
    anomaly_threshold_ms: float,
    correlation_id: str = "",
) -> SleepDiagnosticRecord:
    """Sleep once and asynchronously record enough data to correlate a trace."""
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    if anomaly_threshold_ms < 0:
        raise ValueError("anomaly_threshold_ms must be non-negative")

    pid = os.getpid()
    sequence = next(_sleep_ids)
    sleep_id = (pid << 32) | sequence
    thread = threading.current_thread()
    native_tid = threading.get_native_id()
    requested_ns = round(seconds * 1_000_000_000)
    monotonic_start_ns = time.monotonic_ns()
    expected_deadline_ns = monotonic_start_ns + requested_ns

    native = _native_sleep(
        seconds,
        sleep_id=sleep_id,
        label=label,
        correlation_id=correlation_id,
        anomaly_threshold_ms=anomaly_threshold_ms,
    )
    if native is None:
        time.sleep(seconds)
        native_start_ns = None
        native_return_ns = None
        interpreter_resume_ns = None
        native_error = None
        cpu_before = None
        cpu_after = None
        qos_class = None
        qos_relative_priority = None
    else:
        (
            native_start_ns,
            native_return_ns,
            interpreter_resume_ns,
            native_error,
            cpu_before,
            cpu_after,
            qos_class,
            qos_relative_priority,
        ) = native

    observed_return_ns = time.monotonic_ns()
    actual_ns = observed_return_ns - monotonic_start_ns
    overshoot_ns = actual_ns - requested_ns
    native_wait_ns = (
        None
        if native_start_ns is None or native_return_ns is None
        else native_return_ns - native_start_ns
    )
    gil_reacquire_ns = (
        None
        if native_return_ns is None or interpreter_resume_ns is None
        else interpreter_resume_ns - native_return_ns
    )
    python_post_native_ns = (
        None
        if interpreter_resume_ns is None
        else observed_return_ns - interpreter_resume_ns
    )
    anomalous = overshoot_ns > round(anomaly_threshold_ms * 1_000_000)
    record = SleepDiagnosticRecord(
        sleep_id=sleep_id,
        label=label,
        correlation_id=correlation_id,
        pid=pid,
        python_thread_name=thread.name,
        python_thread_ident=thread.ident,
        native_thread_id=native_tid,
        requested_ns=requested_ns,
        monotonic_start_ns=monotonic_start_ns,
        expected_deadline_ns=expected_deadline_ns,
        native_wait_start_ns=native_start_ns,
        native_wait_return_ns=native_return_ns,
        interpreter_resume_ns=interpreter_resume_ns,
        observed_return_ns=observed_return_ns,
        actual_ns=actual_ns,
        overshoot_ns=overshoot_ns,
        native_wait_ns=native_wait_ns,
        gil_reacquire_ns=gil_reacquire_ns,
        python_post_native_ns=python_post_native_ns,
        wall_return_ns=time.time_ns(),
        cpu_before=cpu_before,
        cpu_after=cpu_after,
        qos_class=qos_class,
        qos_relative_priority=qos_relative_priority,
        process_nice=_process_nice(),
        foreground=_foreground_status(),
        native_helper=native is not None,
        native_error=native_error,
        anomalous=anomalous,
    )
    if anomalous and native is not None:
        try:
            _data_bridge.diagnostic_signpost_anomaly(
                sleep_id,
                label,
                correlation_id,
                requested_ns,
                actual_ns,
                overshoot_ns,
                native_tid,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to emit sleep anomaly signpost", exc_info=True)
    try:
        sample_every = max(
            1,
            int(os.getenv(NORMAL_SAMPLE_EVERY_ENV, str(DEFAULT_NORMAL_SAMPLE_EVERY))),
        )
    except ValueError:
        sample_every = DEFAULT_NORMAL_SAMPLE_EVERY
    _get_recorder().submit(
        record,
        emit=anomalous or sequence == 1 or sequence % sample_every == 0,
    )
    return record
