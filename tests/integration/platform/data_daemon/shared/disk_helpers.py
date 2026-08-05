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
    DETAIL_FLAT,
    DETAIL_REALISTIC,
    FRAME_BYTE_LENGTH,
    FRAME_COLOR_CHANNELS,
    FRAME_GRID_SIZE,
    LOSSLESS_CONTENT_BYTES_PER_PIXEL,
    TRAILING_RGB_GAP_FRAME_TOLERANCE,
)
from tests.integration.platform.data_daemon.shared.test_case.frame_source import (
    frame_code_base,
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

LOSSY_VIDEO_NAME = "lossy.mp4"
"""Lossy H.264 video: a downscaled preview, or the only video in lossy-only mode."""

LOSSLESS_VIDEO_NAME = "lossless.mp4"
"""Bit-exact H.264 archive, written unless the case selects a lossy-only codec."""

VIDEO_FILENAMES = frozenset({LOSSY_VIDEO_NAME, LOSSLESS_VIDEO_NAME})
"""Every video file name a trace directory can hold."""


@dataclass(frozen=True, slots=True)
class VideoArtifact:
    """One video file a case's codec makes the daemon write, and what it proves.

    Attributes:
        filename: Name of the file within the trace directory.
        content_bytes_per_pixel: Encoded-size separator that decides whether this
            artefact's content matches the case's ``video_detail``, or ``None``
            when its size cannot decide that (see
            :data:`LOSSLESS_CONTENT_BYTES_PER_PIXEL`).
    """

    filename: str
    content_bytes_per_pixel: float | None


def expected_video_artifacts(case: DataDaemonTestCase) -> tuple[VideoArtifact, ...]:
    """Return the video artefacts *case*'s codec makes the daemon write.

    ``nc.Codec.H264_MEDIUM`` (``case.lossy_only``) writes a single
    full-resolution CRF-23 video and no archive.  The default codec writes the
    bit-exact archive plus a preview downscaled to 480 lines.

    Only the archive is judged on size: a lossy encode's bytes/pixel does not
    separate realistic frames from flat fill, so both lossy artefacts carry no
    separator and are checked structurally.  Frame content is generated upstream
    of codec selection, so the archive-writing cases in the same suite are what
    catch a frame source that regressed to flat fill.
    """
    if case.lossy_only:
        return (VideoArtifact(filename=LOSSY_VIDEO_NAME, content_bytes_per_pixel=None),)
    return (
        VideoArtifact(filename=LOSSY_VIDEO_NAME, content_bytes_per_pixel=None),
        VideoArtifact(
            filename=LOSSLESS_VIDEO_NAME,
            content_bytes_per_pixel=LOSSLESS_CONTENT_BYTES_PER_PIXEL,
        ),
    )


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


def collect_trace_timestamps_per_file(recording_dir: Path) -> dict[str, list[float]]:
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


def _assert_no_trailing_rgb_gap(
    *,
    trace_key: str,
    timestamps: list[float],
    expected_stop_timestamp: float | None,
    video_fps: int,
    failures: list[TraceFailure],
) -> None:
    """Assert an RGB trace's last on-disk frame isn't stranded before the stop.

    A tail video chunk silently orphaned at the recording-window boundary (see
    the dispatcher's per-producer flush-marker gating) truncates the trace
    without ever tripping :func:`_assert_timestamps_match` — that check only
    ever sees what actually reached disk, so a shortened sequence still
    compares equal to itself. This is deliberately immune to producer
    *under*-delivery (a slow or throttled camera legitimately logging fewer
    frames), which is why it compares the last on-disk timestamp against the
    recording's nominal capture-clock end rather than the expected frame
    count: only a chunk orphaned at the boundary strands frames an unbounded
    distance before a stop that has already happened.

    *expected_stop_timestamp* must be on the same capture clock as
    *timestamps* — see ``ContextResult.expected_video_stop_timestamp_by_recording``.
    It is deliberately not the wall-clock instant ``nc.stop_recording`` was
    called: that call's own ``timestamp`` argument, and therefore every
    per-frame capture timestamp compared here, is caller-supplied content on
    a clock the daemon and this harness never assume is wall-clock at all.
    """
    if expected_stop_timestamp is None or not timestamps:
        return
    if not trace_key.startswith("RGB_IMAGES/"):
        return
    gap_s = expected_stop_timestamp - max(timestamps)
    tolerance_s = TRAILING_RGB_GAP_FRAME_TOLERANCE / video_fps
    if gap_s > tolerance_s:
        failures.append(
            TraceFailure(
                trace_key=trace_key,
                body=(
                    f"last on-disk frame trails the recording's nominal end "
                    f"by {gap_s:.3f}s, more than the {tolerance_s:.3f}s "
                    f"tolerance ({TRAILING_RGB_GAP_FRAME_TOLERANCE} video "
                    f"frame interval(s) at {video_fps}fps) — a tail chunk "
                    f"may have been orphaned at the window boundary"
                ),
            )
        )


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

            trace_timestamps = collect_trace_timestamps_per_file(recording_dir)
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
                _assert_no_trailing_rgb_gap(
                    trace_key=trace_key,
                    timestamps=timestamps,
                    expected_stop_timestamp=(
                        result.expected_video_stop_timestamp_by_recording.get(
                            recording_key
                        )
                    ),
                    video_fps=result.video_fps,
                    failures=trace_failures,
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
    """Return the ``RGB_IMAGES`` trace directories of the recordings in *results*."""
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


def _assert_encoded_size_matches_detail(
    *, video_path: Path, stream: dict, detail: str, separator: float
) -> None:
    """Assert the encoded size sits where ``video_detail`` says it should."""
    width, height = stream.get("width"), stream.get("height")
    nb_read_frames = stream.get("nb_read_frames")
    if not width or not height or not nb_read_frames:
        return
    frame_count, pixels = int(nb_read_frames), int(width) * int(height)
    if frame_count <= 0 or pixels <= 0:
        return

    encoded_bytes = video_path.stat().st_size
    bytes_per_pixel = encoded_bytes / frame_count / pixels
    evidence = f"{encoded_bytes} bytes, {frame_count} frames, {width}x{height}"
    if detail == DETAIL_REALISTIC:
        assert bytes_per_pixel > separator, (
            f"{video_path} encodes {bytes_per_pixel:.4f} bytes/pixel ({evidence}),"
            f" at or below the {separator} floor — the logged frames have"
            f" regressed to near-trivial content and the video pipeline is not"
            f" being exercised"
        )
        return
    assert bytes_per_pixel < separator, (
        f"{video_path} encodes {bytes_per_pixel:.4f} bytes/pixel ({evidence}), at"
        f" or above the {separator} ceiling — a {DETAIL_FLAT}-detail case is"
        f" carrying real frame content, so it is not the cheap workload it is"
        f" calibrated as"
    )


def _assert_frame_count_matches_sidecar(
    *, video_path: Path, stream: dict, trace_dir: Path
) -> None:
    """Assert the video holds exactly one frame per ``trace.json`` entry."""
    trace_json = trace_dir / TRACE_JSON_NAME
    nb_read_frames = stream.get("nb_read_frames")
    if nb_read_frames is None or not trace_json.is_file():
        return
    expected_frames = len(json.loads(trace_json.read_text(encoding="utf-8")))
    if expected_frames == 0:
        return
    assert int(nb_read_frames) == expected_frames, (
        f"{video_path} has {nb_read_frames} frames, "
        f"trace.json expects {expected_frames}"
    )


def _assert_video_artifact(
    *, trace_dir: Path, artifact: VideoArtifact, detail: str
) -> None:
    """Assert one artefact exists, is well-formed, and carries the right content.

    Everything past existence needs ffprobe, so on a host without it this
    degrades to the file-existence check rather than failing.
    """
    video_path = trace_dir / artifact.filename
    assert video_path.is_file(), f"missing {artifact.filename} in {trace_dir}"

    stream = _ffprobe_video_stream(video_path)
    if stream is None:
        return
    assert (
        stream.get("codec_name") == "h264"
    ), f"{video_path} should be H.264, got {stream.get('codec_name')!r}"
    _assert_frame_count_matches_sidecar(
        video_path=video_path, stream=stream, trace_dir=trace_dir
    )

    if artifact.content_bytes_per_pixel is not None:
        _assert_encoded_size_matches_detail(
            video_path=video_path,
            stream=stream,
            detail=detail,
            separator=artifact.content_bytes_per_pixel,
        )


def assert_video_artifacts(
    results: list[ContextResult],
    case: DataDaemonTestCase,
    min_trace_count: int = 1,
) -> None:
    """Assert every RGB video trace on disk holds the artefacts *case* implies.

    One pass of the same checks covers all four (codec, detail) combinations —
    what differs between them is data, not code path:

    - **artefact set** — the lossy video is always written; the lossless archive
      only when the case keeps the default codec, and it must be *absent*
      otherwise, since a lossy-only recording that still wrote one would leave an
      archive on disk the daemon never registers or uploads.
    - **structure** — every artefact is H.264 and holds exactly as many frames as
      its per-frame ``trace.json`` sidecar.
    - **content** — an artefact whose size can decide the question (only the
      lossless archive; see :func:`expected_video_artifacts`) must encode on the
      side of its separator that ``case.video_detail`` claims.  See
      :func:`_assert_encoded_size_matches_detail` for why both directions are
      worth failing on.

    Args:
        results: Per-context results whose recordings scope the check.
        case: The active test case.  Its codec selects the artefact set, its
            ``video_detail`` selects the direction of the content assertions.
        min_trace_count: Minimum number of RGB trace directories expected.
    """
    recordings_root = get_daemon_recordings_root_path()
    assert recordings_root.exists(), f"recordings root missing: {recordings_root}"

    trace_dirs = _rgb_trace_dirs(recordings_root, results)
    assert len(trace_dirs) >= min_trace_count, (
        f"expected at least {min_trace_count} RGB trace dir(s), "
        f"found {len(trace_dirs)} under {recordings_root}"
    )

    artifacts = expected_video_artifacts(case)
    forbidden_names = VIDEO_FILENAMES - {artifact.filename for artifact in artifacts}

    for trace_dir in trace_dirs:
        for filename in sorted(forbidden_names):
            forbidden_path = trace_dir / filename
            assert not forbidden_path.exists(), (
                f"a {case.video_codec} recording must not write {filename}: "
                f"{forbidden_path}"
            )
        for artifact in artifacts:
            _assert_video_artifact(
                trace_dir=trace_dir, artifact=artifact, detail=case.video_detail
            )


def _decode_frame_codes(video_path: Path) -> list[int] | None:
    """Return the code painted into every frame of *video_path*, in stream order.

    Crops to the top-left grid before writing raw frames out, so only the bytes
    the code lives in cross the pipe — the decode still runs in full, but a
    1080p archive costs 48 bytes a frame to read back instead of 6 MB.

    Returns:
        One code per decoded frame, or ``None`` when ffmpeg is unavailable — the
        same degradation :func:`_ffprobe_video_stream` applies.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    raw = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(video_path),
            "-vf",
            f"crop={FRAME_GRID_SIZE}:{FRAME_GRID_SIZE}:0:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    grid_bytes = FRAME_GRID_SIZE * FRAME_GRID_SIZE * FRAME_COLOR_CHANNELS
    return [
        # Red channel of each grid pixel, row-major — the layout
        # ``encode_frame_number`` paints and ``decode_frame_number`` reads.
        int.from_bytes(
            raw[offset : offset + grid_bytes][0::FRAME_COLOR_CHANNELS][
                :FRAME_BYTE_LENGTH
            ],
            byteorder="big",
        )
        for offset in range(0, len(raw) - grid_bytes + 1, grid_bytes)
    ]


def _frame_code_failure(
    *, video_path: Path, camera_name: str, codes: list[int], expected: list[int]
) -> str:
    """Describe how a trace's decoded frame codes differ from what was painted."""
    if len(codes) != len(expected):
        return (
            f"{video_path} (camera {camera_name!r}) holds {len(codes)} frame(s),"
            f" trace.json expects {len(expected)}"
        )
    mismatches = [
        (index, actual, want)
        for index, (actual, want) in enumerate(zip(codes, expected))
        if actual != want
    ]
    examples = "; ".join(
        f"[{index}] actual={actual} expected={want}"
        for index, actual, want in mismatches[:3]
    )
    return (
        f"{video_path} (camera {camera_name!r}):"
        f" {len(mismatches)}/{len(codes)} frame code(s) mismatch — {examples}"
        + (f" (+ {len(mismatches) - 3} more)" if len(mismatches) > 3 else "")
    )


def assert_disk_frame_codes(
    results: list[ContextResult], case: DataDaemonTestCase
) -> None:
    """Assert the archive replays exactly the frames the producer painted.

    Every logged camera frame carries its own code in its top-left 4x4 pixel
    grid, and ``lossless.mp4`` is ``libx264rgb -qp 0``, so that grid survives the
    encode untouched.  Decoding it back turns the archive into a per-frame
    identity check: frame ``i`` must carry ``frame_code_base(...) + i``, in that
    order, for exactly as many frames as the trace's ``trace.json`` holds.

    This is the offline counterpart to the cloud pass's
    :func:`~assertions._assert_synced_camera_codes_are_sane`, and it is strict
    where that one cannot be.  The cloud pass reads a *synchronised* episode,
    where the synchroniser legitimately repeats a frame to fill a sync point, so
    it has to allow missing codes covered by duplicates.  The archive is the
    captured sequence itself, so a dropped, duplicated, or reordered frame has
    nowhere to hide here.

    A lossy-only case perturbs the very pixels the codes live in and writes no
    archive to read them from, so it is skipped — the same gate the cloud pass
    applies.

    Args:
        results: Per-context results whose recordings scope the check.
        case: The active test case; ``lossy_only`` decides whether codes survive.
    """
    if case.lossy_only:
        return

    from tests.integration.platform.data_daemon.shared.db_helpers import (
        fetch_all_traces,
    )

    recordings_root = get_daemon_recordings_root_path()

    for result in results:
        for recording_ordinal, recording_index in enumerate(result.recording_indexes):
            rgb_dir = recordings_root / str(recording_index) / "RGB_IMAGES"
            assert rgb_dir.is_dir(), f"no RGB_IMAGES traces on disk under {rgb_dir}"

            camera_by_trace = {
                str(row["trace_id"]): str(row.get("data_type_name") or "")
                for row in fetch_all_traces(
                    recording_index,
                    columns=["trace_id", "data_type", "data_type_name"],
                )
                if row.get("trace_id") and row.get("data_type") == "RGB_IMAGES"
            }

            for trace_dir in sorted(p for p in rgb_dir.iterdir() if p.is_dir()):
                camera_name = camera_by_trace.get(trace_dir.name)
                assert camera_name in result.camera_names, (
                    f"{trace_dir} maps to camera {camera_name!r}, which is not one"
                    f" of this context's cameras {result.camera_names}"
                )
                video_path = trace_dir / LOSSLESS_VIDEO_NAME
                trace_json = trace_dir / TRACE_JSON_NAME
                assert (
                    video_path.is_file()
                ), f"missing {LOSSLESS_VIDEO_NAME} in {trace_dir}"
                assert trace_json.is_file(), f"missing {TRACE_JSON_NAME} in {trace_dir}"

                codes = _decode_frame_codes(video_path)
                if codes is None:
                    return

                base_code = frame_code_base(
                    context_index=result.context_index,
                    recording_ordinal=recording_ordinal,
                    camera_index=result.camera_names.index(camera_name),
                )
                frame_count = len(json.loads(trace_json.read_text(encoding="utf-8")))
                expected = [base_code + index for index in range(frame_count)]
                assert codes == expected, _frame_code_failure(
                    video_path=video_path,
                    camera_name=camera_name,
                    codes=codes,
                    expected=expected,
                )


def assert_rgb_trace_respects_the_recording_boundary(
    recording_index: int,
    *,
    owed_timestamps: list[float],
    forbidden_before_start_timestamps: list[float],
    forbidden_after_stop_timestamps: list[float],
) -> None:
    """Assert the recording's RGB trace holds every frame it provably owns and
    nothing published outside its window, at either boundary.

    All three lists are capture stamps of frames the producer *reported*
    logging, as classified by
    ``build_test_case_context.classify_split_producer_frames``. None can be
    checked against a nominal ``fps * duration`` count, which measures the
    producer's throughput rather than the daemon's integrity: a cross-process
    producer's frame rate and the lag before its logging gate opens both move
    that number around by tens of frames.

    Called once the tail chunk should already have landed — see
    :func:`~runners.split_video_process_running`, which only returns once the
    producer's own writer flush barrier has completed and been acknowledged.

    A missing owed frame is one the daemon accepted the window for and then
    dropped, which at this boundary means an orphaned chunk. A present
    forbidden-before-start frame means a chunk that opened before this
    window's start was taken whole instead of being cut at it — the mirror,
    at the start boundary, of a present forbidden-after-stop frame, which
    means a chunk straddling the stop was taken whole instead of being cut
    there.
    """
    from tests.integration.platform.data_daemon.shared.db_helpers import (
        wait_for_written_rgb_trace,
    )

    assert owed_timestamps, (
        "the producer logged no frame provably inside the recording, so this "
        "run proves nothing about the boundary"
    )
    rgb_trace = wait_for_written_rgb_trace(recording_index)
    trace_json = (
        get_daemon_recordings_root_path()
        / str(recording_index)
        / "RGB_IMAGES"
        / rgb_trace["trace_id"]
        / "trace.json"
    )
    frames = json.loads(trace_json.read_text(encoding="utf-8"))
    on_disk = {float(frame["timestamp"]) for frame in frames}
    span = (
        f"{len(on_disk)} frame(s) on disk spanning "
        f"[{min(on_disk):.4f}, {max(on_disk):.4f}]"
    )

    missing = [stamp for stamp in owed_timestamps if stamp not in on_disk]
    assert not missing, (
        f"{len(missing)}/{len(owed_timestamps)} frame(s) logged inside the "
        f"recording never reached disk — a chunk was orphaned at the recording "
        f"boundary. First missing capture stamps: "
        f"{[round(stamp, 4) for stamp in missing[:5]]}; {span}"
    )

    kept_early = [
        stamp for stamp in forbidden_before_start_timestamps if stamp in on_disk
    ]
    assert not kept_early, (
        f"{len(kept_early)}/{len(forbidden_before_start_timestamps)} frame(s) "
        f"logged before the recording started reached disk — a chunk that "
        f"opened before this window's start was taken whole instead of being "
        f"cut at it. First such capture stamps: "
        f"{[round(stamp, 4) for stamp in kept_early[:5]]}; {span}"
    )

    kept_late = [stamp for stamp in forbidden_after_stop_timestamps if stamp in on_disk]
    assert not kept_late, (
        f"{len(kept_late)}/{len(forbidden_after_stop_timestamps)} frame(s) "
        f"logged after the recording stopped reached disk — a chunk "
        f"straddling the boundary was taken whole rather than cut at it. "
        f"First such capture stamps: "
        f"{[round(stamp, 4) for stamp in kept_late[:5]]}; {span}"
    )
