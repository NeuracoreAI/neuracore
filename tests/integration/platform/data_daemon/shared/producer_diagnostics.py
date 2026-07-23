"""Low-overhead producer timing diagnostics for data-daemon performance tests."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

from tests.integration.platform.data_daemon.shared.diagnostic_sleep import (
    diagnostic_sleep,
)

logger = logging.getLogger(__name__)

DIAGNOSTICS_ENV = "NEURACORE_PRODUCER_DIAGNOSTICS"
HISTORY_SIZE = 32
UNATTRIBUTED_GAP_NS = 5_000_000
WAIT_DELAY_LOG_THRESHOLD_MS = 5.0
WAIT_DELAY_LOG_LIMIT = 20


@dataclass(frozen=True, slots=True)
class ProducerDiagnosticEvent:
    """One small, immutable producer timing event."""

    timestamp_ns: int
    thread_name: str
    role_name: str
    operation: str
    frame_index: int | None
    duration_ms: float | None
    deadline_lateness_ms: float | None
    details: dict[str, object] = field(default_factory=dict)


class ProducerHeartbeatRegistry:
    """Process-local last-activity map shared by producer histories."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, ProducerDiagnosticEvent] = {}

    def update(self, event: ProducerDiagnosticEvent) -> None:
        with self._lock:
            self._events[event.role_name] = event

    def snapshot(self) -> dict[str, ProducerDiagnosticEvent]:
        with self._lock:
            return dict(self._events)


class ProducerDiagnosticHistory:
    """Bounded producer timeline with opt-in aggregate statistics."""

    def __init__(
        self,
        *,
        context_index: int,
        recording_index: int,
        heartbeat_registry: ProducerHeartbeatRegistry | None = None,
        enabled: bool = True,
        max_events: int = HISTORY_SIZE,
        collect_statistics: bool | None = None,
    ) -> None:
        self.context_index = context_index
        self.recording_index = recording_index
        self.enabled = enabled
        self.collect_statistics = (
            os.getenv(DIAGNOSTICS_ENV) == "1"
            if collect_statistics is None
            else collect_statistics
        )
        self._events: deque[ProducerDiagnosticEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._heartbeats = heartbeat_registry or ProducerHeartbeatRegistry()
        self._statistics: dict[str, list[float]] = defaultdict(list)
        self._last_operation_end_ns: int | None = None
        self._wait_delay_logs = 0

    @property
    def events(self) -> tuple[ProducerDiagnosticEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def record(
        self,
        operation: str,
        *,
        role_name: str,
        frame_index: int | None = None,
        started_ns: int | None = None,
        ended_ns: int | None = None,
        deadline: float | None = None,
        details: dict[str, object] | None = None,
        statistic_value_ms: float | None = None,
    ) -> ProducerDiagnosticEvent | None:
        """Record an event without allowing diagnostics to affect the producer."""
        if not self.enabled:
            return None
        try:
            return self._record(
                operation,
                role_name=role_name,
                frame_index=frame_index,
                started_ns=started_ns,
                ended_ns=ended_ns,
                deadline=deadline,
                details=details,
                statistic_value_ms=statistic_value_ms,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to collect producer diagnostics")
            return None

    def _record(
        self,
        operation: str,
        *,
        role_name: str,
        frame_index: int | None = None,
        started_ns: int | None = None,
        ended_ns: int | None = None,
        deadline: float | None = None,
        details: dict[str, object] | None = None,
        statistic_value_ms: float | None = None,
    ) -> ProducerDiagnosticEvent | None:
        if not self.enabled:
            return None
        end = time.perf_counter_ns() if ended_ns is None else ended_ns
        duration_ms = (
            None if started_ns is None else max(0.0, (end - started_ns) / 1_000_000)
        )
        lateness_ms = None if deadline is None else (time.time() - deadline) * 1_000
        event_details: dict[str, object] = {
            "context_index": self.context_index,
            "recording_index": self.recording_index,
            **(details or {}),
        }
        if deadline is not None:
            event_details.setdefault("scheduled_deadline", deadline)
        event = ProducerDiagnosticEvent(
            timestamp_ns=end,
            thread_name=threading.current_thread().name,
            role_name=role_name,
            operation=operation,
            frame_index=frame_index,
            duration_ms=duration_ms,
            deadline_lateness_ms=lateness_ms,
            details=event_details,
        )
        with self._lock:
            self._events.append(event)
            self._last_operation_end_ns = end
            if self.collect_statistics:
                value = (
                    duration_ms if statistic_value_ms is None else statistic_value_ms
                )
                if value is not None:
                    self._statistics[operation].append(value)
        self._heartbeats.update(event)
        return event

    @contextmanager
    def measure(
        self,
        operation: str,
        *,
        role_name: str,
        frame_index: int,
        deadline: float,
        details: dict[str, object] | None = None,
    ) -> Generator[None]:
        if not self.enabled:
            yield
            return
        started_ns = time.perf_counter_ns()
        event_details = {
            "pre_call_lateness_ms": (time.time() - deadline) * 1_000,
            **(details or {}),
        }
        try:
            yield
        finally:
            self.record(
                operation,
                role_name=role_name,
                frame_index=frame_index,
                started_ns=started_ns,
                deadline=deadline,
                details=event_details,
            )

    def sleep(
        self,
        requested_s: float,
        *,
        role_name: str,
        frame_index: int,
        deadline: float,
    ) -> None:
        """Sleep once and record requested, actual, and CPU-time attribution."""
        if not self.enabled:
            time.sleep(requested_s)
            return

        operation = "time.sleep"
        started_ns = time.perf_counter_ns()
        wall_started_ns = time.time_ns()
        monotonic_started_ns = time.monotonic_ns()
        process_cpu_started_ns = time.process_time_ns()
        thread_cpu_started_ns = time.thread_time_ns()

        sleep_record = diagnostic_sleep(
            requested_s,
            f"{role_name} frame {frame_index}",
            anomaly_threshold_ms=WAIT_DELAY_LOG_THRESHOLD_MS,
            correlation_id=(
                f"ctx={self.context_index}/rec={self.recording_index}/"
                f"role={role_name}/frame={frame_index}"
            ),
        )

        thread_cpu_ended_ns = time.thread_time_ns()
        process_cpu_ended_ns = time.process_time_ns()
        monotonic_ended_ns = time.monotonic_ns()
        wall_ended_ns = time.time_ns()
        ended_ns = time.perf_counter_ns()

        actual_ms = (ended_ns - started_ns) / 1_000_000
        wall_ms = (wall_ended_ns - wall_started_ns) / 1_000_000
        monotonic_ms = (monotonic_ended_ns - monotonic_started_ns) / 1_000_000
        process_cpu_ms = (process_cpu_ended_ns - process_cpu_started_ns) / 1_000_000
        thread_cpu_ms = (thread_cpu_ended_ns - thread_cpu_started_ns) / 1_000_000
        requested_ms = requested_s * 1_000
        overshoot_ms = actual_ms - requested_ms
        thread_not_running_ms = max(0.0, monotonic_ms - thread_cpu_ms)
        excess_thread_not_running_ms = max(
            0.0,
            thread_not_running_ms - requested_ms,
        )
        other_threads_cpu_ms = max(0.0, process_cpu_ms - thread_cpu_ms)
        clock_step_ms = wall_ms - monotonic_ms
        details: dict[str, object] = {
            "wait_mode": operation,
            "requested_ms": requested_ms,
            "actual_ms": actual_ms,
            "wall_ms": wall_ms,
            "monotonic_ms": monotonic_ms,
            "overshoot_ms": overshoot_ms,
            "thread_cpu_ms": thread_cpu_ms,
            "process_cpu_ms": process_cpu_ms,
            "thread_not_running_ms": thread_not_running_ms,
            "excess_thread_not_running_ms": excess_thread_not_running_ms,
            "other_threads_cpu_ms": other_threads_cpu_ms,
            "clock_step_ms": clock_step_ms,
            "pid": os.getpid(),
            "native_thread_id": threading.get_native_id(),
            "scheduled_deadline": deadline,
        }
        details.update({
            "sleep_id": sleep_record.sleep_id,
            "monotonic_start_ns": sleep_record.monotonic_start_ns,
            "expected_deadline_ns": sleep_record.expected_deadline_ns,
            "native_wait_start_ns": sleep_record.native_wait_start_ns,
            "native_wait_return_ns": sleep_record.native_wait_return_ns,
            "interpreter_resume_ns": sleep_record.interpreter_resume_ns,
            "observed_return_ns": sleep_record.observed_return_ns,
            "native_wait_ms": (
                None
                if sleep_record.native_wait_ns is None
                else sleep_record.native_wait_ns / 1_000_000
            ),
            "gil_reacquire_ms": (
                None
                if sleep_record.gil_reacquire_ns is None
                else sleep_record.gil_reacquire_ns / 1_000_000
            ),
            "python_post_native_ms": (
                None
                if sleep_record.python_post_native_ns is None
                else sleep_record.python_post_native_ns / 1_000_000
            ),
            "cpu_before": sleep_record.cpu_before,
            "cpu_after": sleep_record.cpu_after,
            "qos_class": sleep_record.qos_class,
            "qos_relative_priority": sleep_record.qos_relative_priority,
            "process_nice": sleep_record.process_nice,
            "foreground": sleep_record.foreground,
            "native_helper": sleep_record.native_helper,
            "native_error": sleep_record.native_error,
        })
        self.record(
            operation,
            role_name=role_name,
            frame_index=frame_index,
            started_ns=started_ns,
            ended_ns=ended_ns,
            deadline=deadline,
            details=details,
            statistic_value_ms=overshoot_ms,
        )
        if (
            self.collect_statistics
            and overshoot_ms > WAIT_DELAY_LOG_THRESHOLD_MS
            and self._wait_delay_logs < WAIT_DELAY_LOG_LIMIT
        ):
            self._wait_delay_logs += 1
            logger.warning(
                "Producer wait delay ctx=%d rec_idx=%d role=%s frame=%d "
                "mode=%s requested=%.1fms wall=%.1fms monotonic=%.1fms "
                "overshoot=%.1fms thread_cpu=%.1fms process_cpu=%.1fms "
                "thread_not_running=%.1fms excess_not_running=%.1fms "
                "other_threads_cpu=%.1fms "
                "clock_step=%+.1fms pid=%d native_tid=%d",
                self.context_index,
                self.recording_index,
                role_name,
                frame_index,
                operation,
                requested_ms,
                wall_ms,
                monotonic_ms,
                overshoot_ms,
                thread_cpu_ms,
                process_cpu_ms,
                thread_not_running_ms,
                excess_thread_not_running_ms,
                other_threads_cpu_ms,
                clock_step_ms,
                os.getpid(),
                threading.get_native_id(),
            )
            if self._wait_delay_logs == WAIT_DELAY_LOG_LIMIT:
                logger.warning(
                    "Producer wait delay log limit reached ctx=%d rec_idx=%d; "
                    "further individual delays are suppressed and remain in "
                    "the diagnostic summary",
                    self.context_index,
                    self.recording_index,
                )

    def record_gap(
        self,
        *,
        role_name: str,
        frame_index: int,
        deadline: float,
        observed_ns: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        now_ns = time.perf_counter_ns() if observed_ns is None else observed_ns
        with self._lock:
            previous_end_ns = self._last_operation_end_ns
        if (
            previous_end_ns is not None
            and now_ns - previous_end_ns >= UNATTRIBUTED_GAP_NS
        ):
            self.record(
                "unattributed_gap",
                role_name=role_name,
                frame_index=frame_index,
                started_ns=previous_end_ns,
                ended_ns=now_ns,
                deadline=deadline,
                details={"scheduled_deadline": deadline},
            )

    def format_failure(
        self,
        *,
        role_name: str,
        frame_index: int,
        deadline: float,
        observed_at: float,
        failure_ns: int | None = None,
    ) -> str:
        now_ns = time.perf_counter_ns() if failure_ns is None else failure_ns
        lines = [
            "Recent producer diagnostics:",
            f"thread={threading.current_thread().name} role={role_name} "
            f"frame={frame_index} deadline={deadline:.6f} "
            f"current_time={observed_at:.6f} "
            f"lateness={(observed_at - deadline) * 1_000:+.1f}ms",
        ]
        for event in self.events[-20:]:
            relative_ms = (event.timestamp_ns - now_ns) / 1_000_000
            frame = "-" if event.frame_index is None else str(event.frame_index)
            duration = (
                ""
                if event.duration_ms is None
                else f" duration={event.duration_ms:.1f}ms"
            )
            lateness = (
                ""
                if event.deadline_lateness_ms is None
                else f" lateness={event.deadline_lateness_ms:+.1f}ms"
            )
            detail = " ".join(
                (
                    f"{key}={value:.1f}ms"
                    if isinstance(value, float) and key.endswith("_ms")
                    else f"{key}={value}"
                )
                for key, value in event.details.items()
                if key not in {"context_index", "recording_index", "scheduled_deadline"}
            )
            lines.append(
                f"{relative_ms:+8.1f}ms {event.role_name:<18} frame={frame:<5} "
                f"{event.operation:<24}{duration}{lateness}"
                f"{(' ' + detail) if detail else ''}"
            )
        lines.append("Producer heartbeats:")
        for heartbeat_role, event in sorted(self._heartbeats.snapshot().items()):
            age_ms = max(0.0, (now_ns - event.timestamp_ns) / 1_000_000)
            lines.append(
                f"{heartbeat_role:<18} last_active={age_ms:.1f}ms_ago "
                f"operation={event.operation} frame={event.frame_index}"
            )
        return "\n".join(lines)

    def log_summary(self) -> None:
        """Log aggregate operation percentiles only when explicitly enabled."""
        if not self.collect_statistics:
            return
        with self._lock:
            statistics = {key: list(values) for key, values in self._statistics.items()}
        for operation, values in sorted(statistics.items()):
            ordered = sorted(values)

            def percentile(percent: int) -> float:
                return ordered[round((len(ordered) - 1) * percent / 100)]

            logger.info(
                "Producer diagnostic summary ctx=%d rec_idx=%d operation=%s "
                "count=%d mean=%.2fms p50=%.2fms p95=%.2fms p99=%.2fms "
                "max=%.2fms over_5ms=%d over_10ms=%d over_25ms=%d over_50ms=%d",
                self.context_index,
                self.recording_index,
                operation,
                len(ordered),
                sum(ordered) / len(ordered),
                percentile(50),
                percentile(95),
                percentile(99),
                ordered[-1],
                sum(value > 5 for value in ordered),
                sum(value > 10 for value in ordered),
                sum(value > 25 for value in ordered),
                sum(value > 50 for value in ordered),
            )
