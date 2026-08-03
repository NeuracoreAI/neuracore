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
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    MIN_ENCODED_BYTES_PER_PIXEL,
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

    The on-disk directory and the traces join key are both the integer
    ``recording_index``.
    """
    return [
        (str(recording_index), recording_index)
        for recording_index in result.recording_indexes
    ]


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


def _assert_timestamps_match(
    *,
    recording_id: str,
    trace_key: str,
    timestamps: list[float],
    expected_timestamps: list[float],
    failures: list[TraceFailure],
    durations: dict[str, float],
    unknowable_timestamps: frozenset[float] = frozenset(),
) -> None:
    """Assert all timestamps exactly match the expected list (no tolerance).

    Applies to both phase modes: the producer emitted this exact sequence, so
    random-phase offsets need no tolerance window of their own.

    *unknowable_timestamps* are frames a producer logging across the recording
    lifecycle emitted while one of the recording's boundaries was passing, so
    neither side can say whether the daemon took them (see
    ``build_test_case_context._classify_boundary_frames``). They are removed
    from **both** lists up front — never permitted on one — and what remains is
    compared exactly, at both ends alike. Empty for every other case.

    Appends :class:`TraceFailure` instances to *failures* so the caller can
    aggregate traces that share the same failure body (e.g. all joints failing
    with the same mismatch pattern).
    """
    if unknowable_timestamps:
        timestamps = [ts for ts in timestamps if ts not in unknowable_timestamps]
        expected_timestamps = [
            ts for ts in expected_timestamps if ts not in unknowable_timestamps
        ]

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


def assert_disk_recording_properties(
    results: list[ContextResult],
) -> dict[str, float]:
    """Assert every on-disk trace holds exactly the timestamps that were logged.

    Producers emit a sequence precomputed by ``context_worker``, which also
    records it as ``result.expected_timestamps``.  Both phase modes are
    therefore checked the same way — exact equality, no tolerance.  A leaked
    wall-clock epoch, a dropped frame, or a reordered write all surface as a
    mismatch.

    Must be called **after** :func:`wait_for_all_traces_written` so that all
    trace files are fully flushed to disk.

    Args:
        results: Per-context results from the completed recording workload.

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

            per_recording = result.expected_timestamps.by_recording.get(recording_key)
            if per_recording is None:
                all_failures.append(
                    RecordingFailures(
                        recording_id=recording_key,
                        recording_error=(
                            f"no expected timestamps — known recording_index"
                            f" keys: {sorted(result.expected_timestamps.by_recording)}"
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
                _assert_timestamps_match(
                    recording_id=recording_key,
                    trace_key=trace_key,
                    timestamps=timestamps,
                    expected_timestamps=expected[trace_key],
                    failures=trace_failures,
                    durations=durations,
                    unknowable_timestamps=frozenset(
                        per_recording.by_trace_unknowable.get(trace_key, ())
                    ),
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


def _rgb_trace_dirs(
    recordings_root: Path, results: Iterable[ContextResult]
) -> list[Path]:
    """Return the ``RGB_IMAGES`` trace directories of the recordings in *results*.

    Scoped to *results* rather than walking the whole recordings root: under
    ``storage_state_action="preserve"`` the root still holds every earlier
    case's recordings, and those were made with a different codec and frame
    content, so judging this case against them is meaningless.
    """
    recording_dirs = sorted({
        recordings_root / recording_key
        for result in results
        for recording_key, _ in _result_recording_keys(result)
    })
    return [
        trace_dir
        for recording_dir in recording_dirs
        for rgb_dir in [recording_dir / "RGB_IMAGES"]
        if rgb_dir.is_dir()
        for trace_dir in sorted(rgb_dir.iterdir())
        if trace_dir.is_dir()
    ]


def assert_encoded_video_not_trivial(
    results: list[ContextResult], min_trace_count: int = 1
) -> None:
    """Assert the encoded lossless video carries a realistic amount of data.

    Guards the *frame content*, not the pipeline: solid-colour frames compress
    ~620:1 losslessly, so a regression that quietly reverted the synthetic camera
    frames to a flat fill would leave every other assertion in this suite passing
    while the video pipeline did almost no work.

    For every ``RGB_IMAGES`` trace, divides ``lossless.mp4``'s size by its frame
    count and pixel count and requires the result to exceed
    :data:`MIN_ENCODED_BYTES_PER_PIXEL`, which every resolution in this suite
    clears by at least 11x while flat frames fall at least 3x below it.

    Only meaningful for cases whose ``video_detail`` is ``DETAIL_REALISTIC`` and
    which write a lossless archive, so callers must gate on both.

    Args:
        results: Per-context results whose recordings scope the check.
        min_trace_count: Minimum number of RGB trace directories expected.
    """
    recordings_root = get_daemon_recordings_root_path()
    assert recordings_root.exists(), f"recordings root missing: {recordings_root}"

    trace_dirs = _rgb_trace_dirs(recordings_root, results)
    assert len(trace_dirs) >= min_trace_count, (
        f"expected at least {min_trace_count} RGB trace dir(s), "
        f"found {len(trace_dirs)} under {recordings_root}"
    )

    for trace_dir in trace_dirs:
        lossless_path = trace_dir / "lossless.mp4"
        assert lossless_path.is_file(), f"missing lossless.mp4 in {trace_dir}"

        stream = _ffprobe_video_stream(lossless_path)
        if stream is None:
            continue
        width, height = stream.get("width"), stream.get("height")
        nb_read_frames = stream.get("nb_read_frames")
        if not width or not height or not nb_read_frames:
            continue
        frame_count, pixels = int(nb_read_frames), int(width) * int(height)
        if frame_count <= 0 or pixels <= 0:
            continue

        encoded_bytes = lossless_path.stat().st_size
        bytes_per_pixel = encoded_bytes / frame_count / pixels
        assert bytes_per_pixel > MIN_ENCODED_BYTES_PER_PIXEL, (
            f"{lossless_path} encodes {bytes_per_pixel:.4f} bytes/pixel "
            f"({encoded_bytes} bytes, {frame_count} frames, {width}x{height}), "
            f"at or below the {MIN_ENCODED_BYTES_PER_PIXEL} floor — the logged "
            f"frames have regressed to near-trivial content and the video "
            f"pipeline is not being exercised"
        )


def assert_lossy_only_video_artifacts(
    results: list[ContextResult], min_trace_count: int = 1
) -> None:
    """Assert every RGB video trace on disk is a single lossy H.264 video.

    For a recording made with ``nc.Codec.H264_MEDIUM`` the daemon writes only
    ``lossy.mp4`` (libx264) and no ``lossless.mp4``. For every ``RGB_IMAGES``
    trace directory under the recordings root this verifies:

    - ``lossy.mp4`` exists and ``lossless.mp4`` does NOT,
    - (when ffprobe is available) the video is H.264 and its frame count matches
      the per-frame ``trace.json`` sidecar.

    Args:
        results: Per-context results whose recordings scope the check.
        min_trace_count: Minimum number of RGB trace directories expected.
    """
    recordings_root = get_daemon_recordings_root_path()
    assert recordings_root.exists(), f"recordings root missing: {recordings_root}"

    trace_dirs = _rgb_trace_dirs(recordings_root, results)
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
