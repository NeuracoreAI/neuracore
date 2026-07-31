"""Optional Allure reporting helpers for data-daemon integration tests.

The integration suite must remain runnable without the reporting dependency
(for example against an older production wheel). This module therefore keeps
the Allure import optional while providing one place for named steps,
parameters, attachments, and derived metrics used by the local report runner.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

# cspell:ignore nodeid

_EVENT_WRITE_LOCK = threading.Lock()
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})


def performance_metrics_enabled() -> bool:
    """Return whether structured daemon/test phase capture is explicitly on."""
    return os.environ.get("NCD_PERF_METRICS", "").strip().lower() in _TRUTHY_VALUES


def _allure_module():
    try:
        import allure
    except ImportError:
        return None
    return allure


@contextmanager
def report_step(title: str) -> Generator[None]:
    """Create an Allure step when available, otherwise act as a no-op."""
    allure = _allure_module()
    if allure is None:
        yield
        return
    with allure.step(title):
        yield


def report_parameter(name: str, value: object) -> None:
    """Add a visible parameter to the active Allure test when available."""
    allure = _allure_module()
    if allure is not None:
        allure.dynamic.parameter(name, value)


def attach_text(name: str, value: str) -> None:
    """Attach plain text to the active Allure test when available."""
    allure = _allure_module()
    if allure is not None:
        allure.attach(value, name=name, attachment_type=allure.attachment_type.TEXT)


def attach_json(name: str, value: object) -> None:
    """Attach formatted JSON to the active Allure test when available."""
    allure = _allure_module()
    if allure is not None:
        allure.attach(
            json.dumps(value, indent=2, sort_keys=True, default=str),
            name=name,
            attachment_type=allure.attachment_type.JSON,
        )


def record_performance_event(
    phase: str,
    event: str,
    *,
    elapsed_s: float | None = None,
    details: dict[str, object] | None = None,
) -> None:
    """Append a test-side event to the daemon's opt-in JSONL event stream."""
    if not performance_metrics_enabled():
        return
    path = os.environ.get("NCD_PERF_EVENTS_PATH")
    if not path:
        return
    payload: dict[str, object] = {
        "schema_version": 1,
        "timestamp_unix_ns": time.time_ns(),
        "case_id": os.environ.get("NCD_PERF_CASE_ID", ""),
        "component": "test",
        "phase": phase,
        "event": event,
        "details": details or {},
    }
    if elapsed_s is not None:
        payload["elapsed_ms"] = elapsed_s * 1_000.0
    encoded = json.dumps(payload, sort_keys=True, default=str) + "\n"
    try:
        with _EVENT_WRITE_LOCK, Path(path).open("a", encoding="utf-8") as output:
            output.write(encoded)
    except OSError:
        # Diagnostics must never change daemon/test correctness.
        return


def load_performance_events(
    path: str | Path | None, *, case_id: str
) -> list[dict[str, Any]]:
    """Read valid structured events for one pytest case from a shared JSONL file."""
    if not path:
        return []
    event_path = Path(path)
    if not event_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in event_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("case_id") == case_id:
            events.append(event)
    return sorted(events, key=lambda item: int(item.get("timestamp_unix_ns", 0)))


def summarize_performance_events(events: list[dict[str, Any]]) -> dict[str, object]:
    """Aggregate terminal phase durations, retry waits, and upload throughput."""
    phase_values: dict[str, list[float]] = {}
    phase_failures: dict[str, int] = {}
    upload_bytes = 0
    upload_elapsed_ms = 0.0
    retry_count = 0
    retry_backoff_ms = 0.0
    for event in events:
        phase = str(event.get("phase", "unknown"))
        event_name = str(event.get("event", ""))
        elapsed_ms = event.get("elapsed_ms")
        if event_name in {"completed", "failed"} and isinstance(
            elapsed_ms, (int, float)
        ):
            phase_values.setdefault(phase, []).append(float(elapsed_ms))
        if event_name == "failed":
            phase_failures[phase] = phase_failures.get(phase, 0) + 1
        details = event.get("details")
        details = details if isinstance(details, dict) else {}
        if phase == "upload" and event_name == "completed":
            upload_bytes += int(details.get("bytes_uploaded", 0))
            upload_elapsed_ms += float(elapsed_ms or 0.0)
        if phase == "retry_backoff" and event_name == "scheduled":
            retry_count += 1
            retry_backoff_ms += float(details.get("backoff_ms", 0.0))

    phases: dict[str, object] = {}
    for phase in sorted(set(phase_values) | set(phase_failures)):
        values = phase_values.get(phase, [])
        total_ms = sum(values)
        phases[phase] = {
            "count": len(values),
            "total_ms": round(total_ms, 3),
            "average_ms": round(total_ms / len(values), 3) if values else 0.0,
            "maximum_ms": round(max(values), 3) if values else 0.0,
            "failures": phase_failures.get(phase, 0),
        }
    upload_seconds = upload_elapsed_ms / 1_000.0
    return {
        "event_count": len(events),
        "phases": phases,
        "upload": {
            "bytes": upload_bytes,
            "mib": round(upload_bytes / (1024 * 1024), 3),
            "active_transfer_s": round(upload_seconds, 6),
            "mib_per_s": (
                round(upload_bytes / (1024 * 1024) / upload_seconds, 3)
                if upload_seconds > 0
                else None
            ),
        },
        "retries": {
            "count": retry_count,
            "scheduled_backoff_ms": round(retry_backoff_ms, 3),
        },
    }


def format_event_timeline(events: list[dict[str, Any]]) -> str:
    """Render an ordered, relative-time phase timeline for an Allure attachment."""
    if not events:
        return "No structured phase events were captured. Rebuild the Rust daemon."
    origin_event = next(
        (
            item
            for item in events
            if item.get("phase") == "holdback" and item.get("event") == "started"
        ),
        events[0],
    )
    origin_ns = int(origin_event.get("timestamp_unix_ns", 0))
    lines = [
        "Times are relative to the first daemon holdback start (stop received).",
        "Negative entries happened before that stop boundary.",
        "",
        (
            "relative   component phase                    event       elapsed   "
            "correlation"
        ),
        (
            "---------  --------- ------------------------ ----------- --------- "
            "----------------"
        ),
    ]
    for item in events:
        relative_s = (int(item.get("timestamp_unix_ns", origin_ns)) - origin_ns) / 1e9
        elapsed = item.get("elapsed_ms")
        elapsed_text = (
            f"{float(elapsed):8.1f}ms" if elapsed is not None else "        -"
        )
        correlation = []
        if item.get("recording_index") is not None:
            correlation.append(f"rec={item['recording_index']}")
        if item.get("trace_id"):
            correlation.append(f"trace={str(item['trace_id'])[:12]}")
        details = item.get("details")
        if isinstance(details, dict):
            if details.get("operation"):
                correlation.append(f"op={details['operation']}")
            if details.get("attempt") is not None:
                correlation.append(f"attempt={details['attempt']}")
            if details.get("reason"):
                correlation.append(f"reason={details['reason']}")
            for key in (
                "poll",
                "recording_count",
                "trace_count",
                "pending_encode_count",
                "chunks",
                "bytes_uploaded",
                "backoff_ms",
            ):
                if details.get(key) is not None:
                    correlation.append(f"{key}={details[key]}")
        lines.append(
            f"+{relative_s:8.3f}s {str(item.get('component', '')):9.9} "
            f"{str(item.get('phase', '')):24.24} "
            f"{str(item.get('event', '')):11.11} {elapsed_text} "
            f"{' '.join(correlation)}"
        )
    return "\n".join(lines)


def format_phase_summary(summary: dict[str, object]) -> str:
    """Render the aggregate phase table shown alongside the exact timeline."""
    phases = summary.get("phases", {})
    assert isinstance(phases, dict)
    lines = [
        (
            "phase                    count    total ms      avg ms      max ms "
            "failures"
        ),
        (
            "------------------------ ----- ------------ ------------ ------------ "
            "--------"
        ),
    ]
    for phase, raw in phases.items():
        assert isinstance(raw, dict)
        lines.append(
            f"{phase:24.24} {int(raw['count']):5d} {float(raw['total_ms']):12.1f} "
            f"{float(raw['average_ms']):12.1f} {float(raw['maximum_ms']):12.1f} "
            f"{int(raw['failures']):8d}"
        )
    upload = summary["upload"]
    retries = summary["retries"]
    assert isinstance(upload, dict)
    assert isinstance(retries, dict)
    lines.extend([
        "",
        "Totals are sums of per-trace work and may overlap in wall time.",
        f"Uploaded: {upload['mib']} MiB over {upload['active_transfer_s']} active s "
        f"({upload['mib_per_s']} MiB/s)",
        f"Retries: {retries['count']}; scheduled backoff: "
        f"{retries['scheduled_backoff_ms']} ms",
    ])
    return "\n".join(lines)


def write_metrics_artifact(
    directory: str | Path,
    *,
    nodeid: str,
    metrics: dict[str, object],
) -> Path:
    """Write one stable, human-discoverable metrics JSON file per pytest case."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_nodeid = re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid).strip("_")
    path = target_dir / f"{safe_nodeid[-180:]}.json"
    path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def report_headline_metrics(metrics: dict[str, object]) -> None:
    """Put the most useful rates directly in Allure's visible parameter table."""
    wall = metrics["wall_times"]
    producer = metrics["producer_throughput"]
    end_to_end = metrics["end_to_end_throughput"]
    assert isinstance(wall, dict)
    assert isinstance(producer, dict)
    assert isinstance(end_to_end, dict)
    report_parameter("measured producer span (s)", wall["producer_span_s"])
    report_parameter("measured end-to-end wall (s)", wall["end_to_end_wall_s"])
    report_parameter(
        "producer logging API calls/s", producer["logging_api_calls_per_s"]
    )
    report_parameter("producer raw RGB MiB/s", producer["raw_video_mib_per_s"])
    report_parameter(
        "end-to-end logging API calls/s", end_to_end["logging_api_calls_per_s"]
    )


def _timer_metric(stats: dict[str, float]) -> dict[str, float | int]:
    count = int(stats.get("count", 0.0))
    total_s = float(stats.get("total", 0.0))
    return {
        "count": count,
        "total_s": round(total_s, 6),
        "average_s": round(total_s / count, 6) if count else 0.0,
        "maximum_s": round(float(stats.get("max", 0.0)), 6),
    }


def build_performance_metrics(
    *,
    case: Any,
    results: list[Any],
    timer_stats: dict[str, dict[str, float]],
    test_wall_s: float | None,
) -> dict[str, object]:
    """Build measured timings and explicitly-labelled derived throughput metrics."""
    context_starts = [
        float(result.wall_started_at)
        for result in results
        if result.wall_started_at is not None
    ]
    context_stops = [float(result.wall_stopped_at) for result in results]
    producer_span_s = (
        max(context_stops) - min(context_starts)
        if context_starts and context_stops
        else None
    )

    # Context specs can vary duration, so completed results are the source of
    # truth for produced frames. Fall back to the base case only for a partial
    # run that aborted before any context returned.
    joint_frame_sets = (
        sum(result.joint_frame_count * len(result.recording_ids) for result in results)
        if results
        else case.recording_count * case.expected_joint_frames
    )
    video_frames = (
        sum(
            result.video_frame_count
            * len(result.recording_ids)
            * len(result.camera_names)
            for result in results
        )
        if results
        else (
            case.recording_count * case.expected_video_frames * case.video_count
            if case.has_video
            else 0
        )
    )
    raw_video_bytes = (
        video_frames * case.image_width * case.image_height * 3
        if case.has_video
        and case.image_width is not None
        and case.image_height is not None
        else 0
    )
    log_stats = {
        label: stats
        for label, stats in timer_stats.items()
        if label.startswith("nc.log_")
    }
    logging_api_calls = int(
        sum(float(stats.get("count", 0.0)) for stats in log_stats.values())
    )

    def per_second(value: float, duration_s: float | None) -> float | None:
        if duration_s is None or duration_s <= 0:
            return None
        return round(value / duration_s, 3)

    phase_timings = {
        label: _timer_metric(stats)
        for label, stats in sorted(timer_stats.items())
        if label.startswith(("performance.", "daemon.", "storage.", "cloud."))
        or label == "nc.stop_recording"
        or label.startswith("stop_daemon")
    }

    return {
        "definitions": {
            "producer_span_s": (
                "Wall time from the earliest context start to the latest context stop."
            ),
            "end_to_end_wall_s": (
                "Test wall time including dataset readiness, daemon shutdown, local "
                "cleanup, and cloud dataset deletion."
            ),
            "raw_video_mib": (
                "Uncompressed RGB bytes submitted by the test; this is not encoded "
                "or uploaded size."
            ),
            "producer_throughput": (
                "Derived workload counts divided by producer_span_s."
            ),
            "end_to_end_throughput": (
                "Derived workload counts divided by end_to_end_wall_s."
            ),
        },
        "workload": {
            "recordings": case.recording_count,
            "parallel_contexts": case.parallel_contexts,
            "joint_frame_sets": joint_frame_sets,
            "video_frames": video_frames,
            "logging_api_calls": logging_api_calls,
            "raw_video_mib": round(raw_video_bytes / (1024 * 1024), 3),
        },
        "wall_times": {
            "producer_span_s": (
                round(producer_span_s, 6) if producer_span_s is not None else None
            ),
            "end_to_end_wall_s": (
                round(test_wall_s, 6) if test_wall_s is not None else None
            ),
        },
        "producer_throughput": {
            "recordings_per_s": per_second(case.recording_count, producer_span_s),
            "joint_frame_sets_per_s": per_second(joint_frame_sets, producer_span_s),
            "video_frames_per_s": per_second(video_frames, producer_span_s),
            "logging_api_calls_per_s": per_second(logging_api_calls, producer_span_s),
            "raw_video_mib_per_s": per_second(
                raw_video_bytes / (1024 * 1024), producer_span_s
            ),
        },
        "end_to_end_throughput": {
            "recordings_per_s": per_second(case.recording_count, test_wall_s),
            "joint_frame_sets_per_s": per_second(joint_frame_sets, test_wall_s),
            "video_frames_per_s": per_second(video_frames, test_wall_s),
            "logging_api_calls_per_s": per_second(logging_api_calls, test_wall_s),
            "raw_video_mib_per_s": per_second(
                raw_video_bytes / (1024 * 1024), test_wall_s
            ),
        },
        "phase_timings": phase_timings,
        "logging_api_timings": {
            label: _timer_metric(stats) for label, stats in sorted(log_stats.items())
        },
    }


def format_metric_summary(metrics: dict[str, object]) -> str:
    """Format the headline metrics as a compact human-readable attachment."""
    workload = metrics["workload"]
    wall = metrics["wall_times"]
    producer = metrics["producer_throughput"]
    end_to_end = metrics["end_to_end_throughput"]
    assert isinstance(workload, dict)
    assert isinstance(wall, dict)
    assert isinstance(producer, dict)
    assert isinstance(end_to_end, dict)
    return "\n".join([
        f"Producer span:             {wall['producer_span_s']} s",
        f"End-to-end wall:           {wall['end_to_end_wall_s']} s",
        f"Recordings:                {workload['recordings']}",
        f"Logging API calls:         {workload['logging_api_calls']}",
        f"Joint frame sets:          {workload['joint_frame_sets']}",
        f"Video frames:              {workload['video_frames']}",
        f"Raw RGB submitted:         {workload['raw_video_mib']} MiB",
        "",
        "Producer-span throughput:",
        f"  recordings/s:            {producer['recordings_per_s']}",
        f"  logging API calls/s:     {producer['logging_api_calls_per_s']}",
        f"  joint frame sets/s:      {producer['joint_frame_sets_per_s']}",
        f"  video frames/s:          {producer['video_frames_per_s']}",
        f"  raw RGB MiB/s:           {producer['raw_video_mib_per_s']}",
        "",
        "End-to-end throughput:",
        f"  recordings/s:            {end_to_end['recordings_per_s']}",
        f"  logging API calls/s:     {end_to_end['logging_api_calls_per_s']}",
        f"  raw RGB MiB/s:           {end_to_end['raw_video_mib_per_s']}",
    ])


@dataclass(frozen=True)
class PerformanceReport:
    """Fixture-facing facade backed by Allure when its plugin is installed."""

    step = staticmethod(report_step)
    parameter = staticmethod(report_parameter)
    attach_text = staticmethod(attach_text)
    attach_json = staticmethod(attach_json)


@dataclass
class PerformanceReportContext:
    """Manage one case's parameters, results, wall time, and report emission.

    The pytest fixture supplies the completion callback, keeping pytest-specific
    behavior in its configuration while tests use the same context-manager API.
    """

    case: Any
    finalize: Callable[[Any, list[Any], float], None]
    dataset_name: str | None = None
    results: list[Any] = field(default_factory=list, init=False)
    _started_at: float | None = field(default=None, init=False)

    step = staticmethod(report_step)
    parameter = staticmethod(report_parameter)
    attach_text = staticmethod(attach_text)
    attach_json = staticmethod(attach_json)

    def __enter__(self) -> PerformanceReportContext:
        self._started_at = time.perf_counter()
        self.parameter("daemon phase metrics", performance_metrics_enabled())
        if self.dataset_name is not None:
            self.parameter("dataset", self.dataset_name)
        for name, attribute in (
            ("recordings", "recording_count"),
            ("parallel contexts", "parallel_contexts"),
            ("recording duration (s)", "duration_sec"),
            ("joint FPS", "joint_fps"),
            ("video FPS", "video_fps"),
            ("video streams", "video_count"),
            ("stop waits for readiness", "wait"),
        ):
            if hasattr(self.case, attribute):
                self.parameter(name, getattr(self.case, attribute))
        return self

    def capture_results(self, results: list[Any]) -> list[Any]:
        """Retain completed or partial context results for final reporting."""
        self.results = results
        return results

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        started_at = self._started_at
        elapsed_s = time.perf_counter() - started_at if started_at is not None else 0.0
        self.finalize(self.case, self.results, elapsed_s)
