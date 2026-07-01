"""Filesystem helpers for data-daemon integration tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from neuracore.data_daemon.helpers import get_daemon_recordings_root_path
from neuracore.data_daemon.rust_selection import rust_daemon_enabled
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    TIMESTAMP_MODE_REAL,
    TIMESTAMP_MODE_STOCHASTIC,
    stochastic_jitter_window,
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
    """Return cloud recording IDs that exist as subdirectories on disk.

    Legacy Python daemon: the on-disk layout is
    ``{recordings_root}/{recording_id}/...`` — the top directory segment is the
    cloud ``recording_id``. Under the Rust daemon use
    :func:`list_recording_indexes_on_disk` instead.
    """
    recordings_root = get_daemon_recordings_root_path()
    if not recordings_root.exists():
        return set()
    return {child.name for child in recordings_root.iterdir() if child.is_dir()}


def normalize_recording_ids(
    expected_recording_ids: Iterable[str] | None,
) -> set[str]:
    """Return a clean set of non-empty cloud recording ID strings."""
    if expected_recording_ids is None:
        return set()
    return {
        str(recording_id) for recording_id in expected_recording_ids if recording_id
    }


def list_recording_indexes_on_disk() -> set[int]:
    """Return ``recording_index`` values that exist under the recordings root.

    Thin-shipper rewrite: the on-disk layout is now
    ``{recordings_root}/{recording_index}/{data_type}/{trace_id}/`` — the top
    directory segment is the daemon-assigned INTEGER ``recording_index``, not a
    cloud recording id. Only integer-named directories are recording roots.
    """
    recordings_root = get_daemon_recordings_root_path()
    if not recordings_root.exists():
        return set()
    indexes: set[int] = set()
    for child in recordings_root.iterdir():
        if not child.is_dir():
            continue
        try:
            indexes.add(int(child.name))
        except ValueError:
            continue
    return indexes


def normalize_recording_indexes(
    expected_recording_indexes: Iterable[int | str] | None,
) -> set[int]:
    """Return a clean set of integer ``recording_index`` values."""
    if expected_recording_indexes is None:
        return set()
    normalized: set[int] = set()
    for recording_index in expected_recording_indexes:
        if recording_index is None or recording_index == "":
            continue
        normalized.add(int(recording_index))
    return normalized


def _result_recording_keys(result: ContextResult) -> list[tuple[str, int | str]]:
    """Yield ``(disk_dir_name, db_correlation_key)`` per recording in *result*.

    Rust daemon: the on-disk directory and traces join key are both the integer
    ``recording_index``. Legacy Python daemon: both are the cloud ``recording_id``
    string. Keeping these two values together lets the assertion body stay
    identical across modes.
    """
    if rust_daemon_enabled():
        return [
            (str(recording_index), recording_index)
            for recording_index in result.recording_indexes
        ]
    return [(recording_id, recording_id) for recording_id in result.recording_ids]


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
        recording_path = get_daemon_recordings_root_path() / recording_id
        failures.append(
            TraceFailure(
                trace_key=trace_key,
                body=(
                    f"timestamp span {actual_duration_s:.3f}s outside expected "
                    f"{expected_duration_s:.2f}s ± {clock_tolerance_s:.2f}s "
                    f"[{expected_duration_s - clock_tolerance_s:.2f}s, "
                    f"{expected_duration_s + clock_tolerance_s:.2f}s]; "
                    f"recording data at {recording_path}"
                ),
            )
        )


def _assert_stochastic_timestamps(
    *,
    recording_id: str,
    trace_key: str,
    timestamps: list[float],
    expected_timestamps: list[float],
    failures: list[TraceFailure],
    durations: dict[str, float],
    fps: int,
) -> None:
    """Assert every timestamp is within the fps-derived jitter window.

    The intended timestamps are the pre-jitter values (``start + i / fps``).
    Each actual timestamp must satisfy ``|actual - intended| <= window``, where
    ``window`` is :func:`stochastic_jitter_window` of the trace's ``fps``.
    """
    window = stochastic_jitter_window(fps)
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

    out_of_window = [
        (i, actual, intended)
        for i, (actual, intended) in enumerate(zip(timestamps, expected_timestamps))
        if abs(actual - intended) > window
    ]
    if out_of_window:
        examples = "; ".join(
            f"[{i}] actual={actual:.6f} intended={intended:.6f}"
            f" delta={actual - intended:+.6f}"
            for i, actual, intended in out_of_window[:3]
        )
        body = (
            f"{len(out_of_window)}/{len(timestamps)} timestamp(s) outside"
            f" ±{window}s jitter window — {examples}"
            + (f" (+ {len(out_of_window) - 3} more)" if len(out_of_window) > 3 else "")
        )
        failures.append(TraceFailure(trace_key=trace_key, body=body))
        return

    if timestamps:
        durations[f"{recording_id}:{trace_key}"] = timestamps[-1] - timestamps[0]


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
        use_stochastic = result.timestamp_mode == TIMESTAMP_MODE_STOCHASTIC
        for recording_key, fetch_key in _result_recording_keys(result):
            recording_dir = recordings_root / recording_key
            if not recording_dir.exists():
                all_failures.append(
                    RecordingFailures(
                        recording_id=recording_key,
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
                        recording_id=recording_key,
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
                fetch_key,
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

            # Resolve mode-specific state once before iterating traces.
            assert_ts = None
            expected: dict[str, list[float]] = {}
            if not use_real:
                assert_ts = (
                    _assert_stochastic_timestamps
                    if use_stochastic
                    else _assert_manual_timestamps
                )
                per_recording = (
                    result.expected_timestamps.by_recording.get(recording_key)
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
                            recording_id=recording_key,
                            recording_error=(
                                f"no expected timestamps —"
                                f" known recording_index keys: {known}"
                            ),
                            trace_failures=[],
                        )
                    )
                    continue
                expected = per_recording.by_trace

            for trace_key, timestamps in mapped_trace_timestamps.items():
                if use_real:
                    _assert_real_timestamps(
                        recording_id=recording_key,
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
                    # The stochastic assertion sizes its tolerance from the
                    # trace's fps; the manual assertion takes no tolerance.
                    extra = (
                        {"fps": per_recording.by_trace_fps[trace_key]}
                        if use_stochastic
                        else {}
                    )
                    assert_ts(
                        recording_id=recording_key,
                        trace_key=trace_key,
                        timestamps=timestamps,
                        expected_timestamps=expected[trace_key],
                        failures=trace_failures,
                        durations=durations,
                        **extra,
                    )

            if trace_failures:
                all_failures.append(
                    RecordingFailures(
                        recording_id=recording_key,
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


def _ffprobe_video_stream(video_path: Path) -> dict | None:
    """Return the first video stream's ffprobe info, or None if unavailable.

    Returns None (rather than failing) when ffprobe is not installed, so the
    file-existence checks still run on hosts without ffprobe.
    """
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,nb_read_frames",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    return streams[0] if streams else None


def assert_lossy_only_video_artifacts(min_trace_count: int = 1) -> None:
    """Assert every RGB video trace on disk is a single lossy H.264 video.

    For a recording made with ``nc.Codec.H264_MEDIUM`` the daemon writes only
    ``lossy.mp4`` (libx264) and no ``lossless.mp4``. For every ``RGB_IMAGES``
    trace directory under the recordings root this verifies:

    - ``lossy.mp4`` exists and ``lossless.mp4`` does NOT,
    - (when ffprobe is available) the video is H.264 and its frame count matches
      the per-frame ``trace.json`` sidecar.

    Works identically for the Python and Rust daemons (both write the same
    on-disk artefact layout).

    Args:
        min_trace_count: Minimum number of RGB trace directories expected.
    """
    recordings_root = get_daemon_recordings_root_path()
    assert recordings_root.exists(), f"recordings root missing: {recordings_root}"

    trace_dirs = [
        trace_dir
        for recording_dir in sorted(recordings_root.iterdir())
        if recording_dir.is_dir()
        for rgb_dir in [recording_dir / "RGB_IMAGES"]
        if rgb_dir.is_dir()
        for trace_dir in sorted(rgb_dir.iterdir())
        if trace_dir.is_dir()
    ]
    assert len(trace_dirs) >= min_trace_count, (
        f"expected at least {min_trace_count} RGB trace dir(s), "
        f"found {len(trace_dirs)} under {recordings_root}"
    )

    for trace_dir in trace_dirs:
        lossy_path = trace_dir / "lossy.mp4"
        lossless_path = trace_dir / "lossless.mp4"
        assert lossy_path.is_file(), f"missing lossy.mp4 in {trace_dir}"
        assert (
            not lossless_path.exists()
        ), f"lossy-only recording must not write lossless.mp4: {lossless_path}"

        stream = _ffprobe_video_stream(lossy_path)
        if stream is None:
            continue
        assert (
            stream.get("codec_name") == "h264"
        ), f"{lossy_path} should be H.264, got {stream.get('codec_name')!r}"

        trace_json = trace_dir / TRACE_JSON_NAME
        if trace_json.is_file():
            expected_frames = len(json.loads(trace_json.read_text(encoding="utf-8")))
            nb_read_frames = stream.get("nb_read_frames")
            if nb_read_frames is not None and expected_frames > 0:
                assert int(nb_read_frames) == expected_frames, (
                    f"{lossy_path} has {nb_read_frames} frames, "
                    f"trace.json expects {expected_frames}"
                )
