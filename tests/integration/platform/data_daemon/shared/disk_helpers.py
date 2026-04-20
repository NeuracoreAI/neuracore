"""Filesystem helpers for data-daemon integration tests."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from neuracore.data_daemon.helpers import get_daemon_recordings_root_path
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    TIMESTAMP_MODE_REAL,
)

if TYPE_CHECKING:
    from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
        ContextResult,
    )


@dataclass(frozen=True, slots=True)
class TraceFailure:
    """A single assertion failure for one trace within a recording.

    Attributes:
        trace_key: Semantic trace identifier (``data_type/data_type_name``).
        body: Human-readable description of what failed, without the trace key prefix.
    """

    trace_key: str
    body: str


@dataclass(frozen=True, slots=True)
class RecordingFailures:
    """All assertion failures for one recording.

    Attributes:
        recording_id: The recording ID.
        recording_error: A recording-level error message (e.g. missing directory),
            or ``None`` if the recording itself was found.
        trace_failures: Per-trace failures, empty when ``recording_error`` is set.
    """

    recording_id: str
    recording_error: str | None
    trace_failures: list[TraceFailure]

    @property
    def total(self) -> int:
        return 1 if self.recording_error else len(self.trace_failures)

    def render(self) -> list[str]:
        """Return human-readable lines for this recording's failures."""
        if self.recording_error:
            return [self.recording_error]
        return _collapse_trace_failures(self.trace_failures)


TRACE_JSON_NAME = "trace.json"
"""File name of the JSON trace artefact written for every trace."""

VIDEO_TRACE_DATA_TYPES = {"RGB_IMAGES", "DEPTH_IMAGES"}
"""Data types whose traces produce video files in addition to trace.json."""

VIDEO_TRACE_FILENAMES = {TRACE_JSON_NAME, "lossy.mp4", "lossless.mp4"}
"""Expected on-disk file names for a video trace directory."""


def list_recording_ids_on_disk() -> set[str]:
    """Return recording IDs that exist as subdirectories under the recordings root."""
    recordings_root = get_daemon_recordings_root_path()
    if not recordings_root.exists():
        return set()
    return {child.name for child in recordings_root.iterdir() if child.is_dir()}


def normalize_recording_ids(
    expected_recording_ids: Iterable[str] | None,
) -> set[str]:
    """Return a clean set of non-empty recording ID strings."""
    if expected_recording_ids is None:
        return set()
    return {
        str(recording_id) for recording_id in expected_recording_ids if recording_id
    }


def _collect_trace_timestamps_per_file(recording_dir: Path) -> dict[str, list[float]]:
    """Return mapping of trace file key (joint/camera name) to timestamps
    from every trace.json under a recording dir."""
    trace_timestamps: dict[str, list[float]] = {}
    for trace_json_path in recording_dir.rglob(TRACE_JSON_NAME):
        # Key is the parent directory name (joint or camera name)
        key = trace_json_path.parent.name
        try:
            frames = json.loads(trace_json_path.read_bytes())
        except Exception:
            continue
        if not isinstance(frames, list):
            continue
        ts_list = []
        for frame in frames:
            if isinstance(frame, dict):
                ts = frame.get("timestamp")
                if isinstance(ts, (int, float)):
                    ts_list.append(float(ts))
        trace_timestamps[key] = ts_list
    return trace_timestamps


def _assert_manual_timestamps(
    *,
    recording_id: str,
    trace_key: str,
    timestamps: list[float],
    expected_timestamps: list[float],
    failures: list[TraceFailure],
    durations: dict[str, float],
) -> None:
    """Assert all timestamps exactly match the expected manual list (no tolerance).

    Appends :class:`TraceFailure` instances to *failures* so the caller can
    aggregate traces that share the same failure body (e.g. all joints failing
    with the same mismatch pattern).
    """
    if len(timestamps) != len(expected_timestamps):
        failures.append(
            TraceFailure(
                trace_key=trace_key,
                body=(
                    f"timestamp count mismatch: expected"
                    f" {len(expected_timestamps)}, got {len(timestamps)}"
                ),
            )
        )
        return

    mismatches = [
        (i, actual, expected)
        for i, (actual, expected) in enumerate(zip(timestamps, expected_timestamps))
        if actual != expected
    ]
    if mismatches:
        examples = "; ".join(
            f"[{i}] actual={actual:.6f} expected={expected:.6f}"
            for i, actual, expected in mismatches[:3]
        )
        body = (
            f"{len(mismatches)}/{len(timestamps)} timestamp(s) mismatch — {examples}"
            + (f" (+ {len(mismatches) - 3} more)" if len(mismatches) > 3 else "")
        )
        failures.append(TraceFailure(trace_key=trace_key, body=body))
        return

    if timestamps:
        durations[f"{recording_id}:{trace_key}"] = timestamps[-1] - timestamps[0]


def _collapse_trace_failures(failures: list[TraceFailure]) -> list[str]:
    """Collapse failures that share the same body across multiple traces.

    When many traces (e.g. one per joint) fail with identical mismatch details,
    emit a single aggregated line rather than one line per trace.

    Returns a list of human-readable failure strings.
    """
    body_to_keys: dict[str, list[str]] = defaultdict(list)
    for f in failures:
        body_to_keys[f.body].append(f.trace_key)

    lines = []
    for body, keys in body_to_keys.items():
        if len(keys) == 1:
            lines.append(f"trace {keys[0]}: {body}")
        else:
            # Find a common data-type prefix (part before the first '/').
            prefixes = {k.split("/")[0] for k in keys}
            prefix = (
                next(iter(prefixes))
                if len(prefixes) == 1
                else ", ".join(sorted(prefixes))
            )
            lines.append(f"{len(keys)} traces ({prefix}/*): {body}")
    return lines


def _assert_real_timestamps(
    *,
    recording_id: str,
    trace_key: str,
    timestamps: list[float],
    wall_started_at: float | None,
    wall_stopped_at: float,
    duration_sec: int,
    clock_tolerance_s: float,
    failures: list[TraceFailure],
    durations: dict[str, float],
) -> None:
    """Assert all timestamps are plausible wall-clock epoch values.

    Validates that:

    1. Every timestamp looks like a Unix epoch (> year-2000 threshold).
    2. All timestamps fall within the wall-clock window ``[wall_started_at -
       tol, wall_stopped_at + tol]``.
    3. Timestamps are non-decreasing (monotonic).
    4. The span from first to last timestamp is within a reasonable range
       of the expected recording duration.
    """
    # 946684800 = 2000-01-01T00:00:00Z — any real timestamp must exceed this.
    epoch_floor = 946_684_800.0
    non_epoch = [ts for ts in timestamps if ts < epoch_floor]
    if non_epoch:
        failures.append(
            TraceFailure(
                trace_key=trace_key,
                body=(
                    f"{len(non_epoch)} timestamp(s) are not valid epoch"
                    f" values (< year 2000) — "
                    f"e.g. {non_epoch[:5]}"
                ),
            )
        )
        return

    if wall_started_at is not None:
        out_of_wall = [
            ts
            for ts in timestamps
            if not (
                wall_started_at - clock_tolerance_s
                <= ts
                <= wall_stopped_at + clock_tolerance_s
            )
        ]
        if out_of_wall:
            failures.append(
                TraceFailure(
                    trace_key=trace_key,
                    body=(
                        f"{len(out_of_wall)} timestamp(s) outside wall-clock window "
                        f"[{wall_started_at - clock_tolerance_s:.2f}, "
                        f"{wall_stopped_at + clock_tolerance_s:.2f}] — "
                        f"e.g. {out_of_wall[:5]}"
                    ),
                )
            )
            return

    sorted_ts = sorted(timestamps)
    non_monotonic = [
        (i, sorted_ts[i], sorted_ts[i + 1])
        for i in range(len(sorted_ts) - 1)
        if sorted_ts[i] > sorted_ts[i + 1]
    ]
    if non_monotonic:
        failures.append(
            TraceFailure(
                trace_key=trace_key,
                body=f"timestamps are not monotonic — e.g. {non_monotonic[:5]}",
            )
        )

    actual_start_s = sorted_ts[0]
    actual_end_s = sorted_ts[-1]
    actual_duration_s = actual_end_s - actual_start_s
    durations[f"{recording_id}:{trace_key}"] = actual_duration_s

    expected_duration_s = float(duration_sec)
    if not (
        expected_duration_s - clock_tolerance_s
        <= actual_duration_s
        <= expected_duration_s + clock_tolerance_s
    ):
        failures.append(
            TraceFailure(
                trace_key=trace_key,
                body=(
                    f"timestamp span {actual_duration_s:.3f}s outside"
                    f" expected recording duration "
                    f"[{expected_duration_s - clock_tolerance_s:.2f}s, "
                    f"{expected_duration_s + clock_tolerance_s:.2f}s]"
                ),
            )
        )


def assert_disk_recording_properties(
    results: list[ContextResult],
    clock_tolerance_s: float = 1.0,
) -> dict[str, float]:
    """Assert on-disk trace timestamps fall within the expected recording window.

    Behaviour depends on ``result.timestamp_mode``:

    - **manual** — every timestamp must lie within the synthetic window
      ``[timestamp_start_s, timestamp_end_s]`` (plus tolerance).  Any
      timestamp outside this range (e.g. a leaked wall-clock epoch) is an
      explicit failure.
    - **real** — every timestamp must be a valid Unix epoch, fall within
      the wall-clock window ``[wall_started_at, wall_stopped_at]``, be
      monotonically non-decreasing, and span approximately
      ``duration_sec``.

    Must be called **after** :func:`wait_for_all_traces_written` so that all
    trace files are fully flushed to disk.

    Args:
        results: Per-context results from the completed recording workload.
        clock_tolerance_s: Tolerance in seconds applied around the expected
            timestamp window when filtering and asserting.

    Returns:
        Mapping of ``recording_id -> duration_s`` (``max - min`` of valid
        timestamps) for each recording that passes validation.

    Raises:
        AssertionError: When any recording's on-disk timestamps are out of
            range or when no timestamps can be read for a recording.
    """
    recordings_root = get_daemon_recordings_root_path()
    all_failures: list[RecordingFailures] = []
    durations: dict[str, float] = {}

    from tests.integration.platform.data_daemon.shared.db_helpers import (
        fetch_all_traces,
    )

    for result in results:
        use_real = result.timestamp_mode == TIMESTAMP_MODE_REAL
        for recording_id in result.recording_ids:
            recording_dir = recordings_root / recording_id
            if not recording_dir.exists():
                all_failures.append(
                    RecordingFailures(
                        recording_id=recording_id,
                        recording_error=(
                            f"directory not found on disk ({recording_dir})"
                        ),
                        trace_failures=[],
                    )
                )
                continue

            trace_timestamps = _collect_trace_timestamps_per_file(recording_dir)
            if not trace_timestamps:
                all_failures.append(
                    RecordingFailures(
                        recording_id=recording_id,
                        recording_error=(
                            f"no timestamps found in any trace.json"
                            f" under {recording_dir}"
                        ),
                        trace_failures=[],
                    )
                )
                continue

            # Build a mapping from trace UUID (directory name) to a unique semantic key.
            # Each trace is uniquely identified by data_type + data_type_name — e.g.
            # "JOINT_POSITIONS/vx300s_left\waist", "RGB_IMAGES/camera_0",
            # "CUSTOM_1D/marker".
            trace_rows = fetch_all_traces(
                recording_id,
                columns=["trace_id", "data_type", "data_type_name"],
            )
            uuid_to_semantic: dict[str, str] = {}
            for row in trace_rows:
                uuid = row.get("trace_id")
                if not uuid:
                    continue
                data_type = row.get("data_type") or ""
                data_type_name = row.get("data_type_name") or ""
                key = f"{data_type}/{data_type_name}" if data_type_name else data_type
                uuid_to_semantic[uuid] = key

            # Map trace_timestamps keys (UUIDs) to semantic keys for assertion
            mapped_trace_timestamps: dict[str, list[float]] = {}
            for uuid, timestamps in trace_timestamps.items():
                semantic = uuid_to_semantic.get(uuid, uuid)
                mapped_trace_timestamps[semantic] = timestamps

            trace_failures: list[TraceFailure] = []

            if use_real:
                for trace_key, timestamps in mapped_trace_timestamps.items():
                    _assert_real_timestamps(
                        recording_id=recording_id,
                        trace_key=trace_key,
                        timestamps=timestamps,
                        wall_started_at=result.wall_started_at,
                        wall_stopped_at=result.wall_stopped_at,
                        duration_sec=result.duration_sec,
                        clock_tolerance_s=clock_tolerance_s,
                        failures=trace_failures,
                        durations=durations,
                    )
            else:
                per_recording = (
                    result.expected_timestamps.by_recording.get(recording_id)
                    if result.expected_timestamps is not None
                    else None
                )
                if per_recording is None:
                    known = (
                        sorted(result.expected_timestamps.by_recording)
                        if result.expected_timestamps
                        else []
                    )
                    all_failures.append(
                        RecordingFailures(
                            recording_id=recording_id,
                            recording_error=(
                                f"no expected timestamps —"
                                f" known recording IDs: {known}"
                            ),
                            trace_failures=[],
                        )
                    )
                    continue
                expected = per_recording.by_trace
                for trace_key, timestamps in mapped_trace_timestamps.items():
                    if trace_key not in expected:
                        trace_failures.append(
                            TraceFailure(
                                trace_key=trace_key,
                                body=(
                                    f"found on disk but has no expected"
                                    f" timestamps — known traces:"
                                    f" {sorted(expected)}"
                                ),
                            )
                        )
                        continue
                    _assert_manual_timestamps(
                        recording_id=recording_id,
                        trace_key=trace_key,
                        timestamps=timestamps,
                        expected_timestamps=expected[trace_key],
                        failures=trace_failures,
                        durations=durations,
                    )

            if trace_failures:
                all_failures.append(
                    RecordingFailures(
                        recording_id=recording_id,
                        recording_error=None,
                        trace_failures=trace_failures,
                    )
                )

    if all_failures:
        total = sum(rf.total for rf in all_failures)
        sections = []
        for rf in all_failures:
            lines = rf.render()
            sections.append(f"  recording {rf.recording_id} ({len(lines)} failure(s)):")
            sections.extend(f"    - {line}" for line in lines)
        raise AssertionError(
            f"Disk trace assertion(s) failed "
            f"({total} failure(s) across {len(all_failures)} recording(s)):\n"
            + "\n".join(sections)
        )

    return durations
