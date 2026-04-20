"""Shared configuration: dataclasses, utilities, and reporting.

Defines the ``DataDaemonTestCase`` dataclass, ``DataDaemonTestBatch`` for
grouping cases with shared infrastructure parameters, utility functions, and
analysis/reporting helpers.  Per-suite case lists live in each suite's
``test_cases.py``;
context-spec interpretation and recording workers live in
``build_test_case_context.py``.
"""

# cspell:ignore vardur
from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
        ContextResult,
    )

from neuracore.core.config.config_manager import get_config_manager
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    BASE_DATASET_READY_TIMEOUT_S,
    DURATION_MODE_FIXED,
    DURATION_MODE_VARIABLE,
    MAX_DATASET_READY_TIMEOUT_S,
    MODE_SEQUENTIAL,
    PRODUCER_SYNCHRONOUS,
    STOP_METHOD_CLI,
    STORAGE_STATE_EMPTY,
    TIMESTAMP_MODE_MANUAL,
    TIMESTAMP_MODE_REAL,
    StopMethod,
    StorageStateAction,
    TimestampMode,
)

logger = logging.getLogger(__name__)

SESSION_RUNS: list[dict[str, object]] = []

BASE_JOINT_NAMES = [
    "vx300s_left/waist",
    "vx300s_left/shoulder",
    "vx300s_left/elbow",
    "vx300s_left/forearm_roll",
    "vx300s_left/wrist_angle",
    "vx300s_left/wrist_rotate",
    "vx300s_left/left_finger",
    "vx300s_left/right_finger",
    "vx300s_right/waist",
    "vx300s_right/shoulder",
    "vx300s_right/elbow",
    "vx300s_right/forearm_roll",
    "vx300s_right/wrist_angle",
    "vx300s_right/wrist_rotate",
    "vx300s_right/left_finger",
    "vx300s_right/right_finger",
]

# Parameters that live at batch level and are propagated to each case.
_BATCH_PARAMS = frozenset({
    "kill_daemon_between_tests",
    "storage_state_action",
    "preserve_artifacts_per_test",
    "stop_method",
})


@dataclass(frozen=True)
class DataDaemonTestCase:
    """A single parametrised test case for the data-daemon integration suite.

    Each instance fully describes one combination of workload and daemon
    configuration parameters.  Hand-curated cases in each suite's ``test_cases.py``
    are constructed directly using shorthand defaults, typically grouped into a
    ``DataDaemonTestBatch`` that applies shared infrastructure parameters.

    Attributes:
        duration_sec: Wall-clock duration logged per individual recording, in
            seconds.  All frames are generated at ``fps`` Hz so the total
            expected frame count is ``fps * duration_sec``.
        parallel_contexts: Number of recording contexts that run concurrently.
            Each context owns an independent robot connection and cycles through
            its share of the total ``recording_count``.
        recording_count: Total number of recordings to produce across *all*
            parallel contexts. Work is distributed as evenly as possible:
            each context gets ``recording_count // parallel_contexts`` recordings,
            and the first ``recording_count % parallel_contexts`` contexts each
            get one additional recording.
        mode: Execution strategy for parallel contexts.  ``"sequential"``
            starts each context only after the previous one has finished;
            ``"staggered"`` starts contexts with a time offset so their active
            windows overlap.  Single-context cases always use ``"sequential"``.
            joint_count: Number of joint channels to log per frame.  Names are
            drawn from ``BASE_JOINT_NAMES`` and extended with synthetic names
            when the count exceeds the base list length.
        producer_channels: Thread-allocation strategy for data producers.
            ``"synchronous"`` logs all data types from a single thread in
            sequence; ``"per_thread"`` spawns one dedicated thread per data
            type so streams are written concurrently.
        video_count: Number of RGB camera streams to log per recording.  A
            value of ``0`` disables video entirely.
        image_width: Horizontal resolution of each camera frame in pixels.
            ``None`` when ``video_count`` is ``0``.
        image_height: Vertical resolution of each camera frame in pixels.
            ``None`` when ``video_count`` is ``0``.
        kill_daemon_between_tests: When ``True``, the daemon process is
            stopped and restarted before each test case so every case begins
            from a clean process state.  Set to ``False`` to keep the daemon
            alive across cases (faster but tests share daemon state).
        storage_state_action: Controls how the SQLite state database and
            recordings folder are handled between test cases.  ``"preserve"``
            leaves both untouched for post-mortem inspection; ``"empty"``
            truncates DB tables and clears recordings folder contents but keeps
            them on disk; ``"delete"`` removes both entirely.
        stop_method: Method used to stop the daemon process.  ``"cli"``
            (default) invokes the CLI stop command; ``"sigterm"`` sends SIGTERM
            directly; ``"sigkill"`` terminates immediately without giving the
            daemon a chance to flush buffers.
        preserve_artifacts_per_test: When ``True``, the recordings directory
            and DB file are copied to a timestamped artifact directory before
            cleanup so they can be inspected after the test run.  Implies
            ``storage_state_action == "preserve"`` for the active env paths.
        context_duration_mode: Controls per-recording duration within each
            context.  ``"fixed"`` makes every recording last exactly
            ``duration_sec`` seconds.  ``"variable"`` randomises the duration
            for each recording within a range around ``duration_sec``, which
            exercises the daemon's handling of recordings with unequal lengths.
        wait: When ``True``, recording contexts block until the daemon
            acknowledges the stop-recording call before returning.  When
            ``False`` the stop call is fire-and-forget, which exercises the
            daemon's ability to process uploads without an explicit client
            wait.  Cloud tests expand over both values; offline tests always
            use ``False``.
        joint_fps: Frame rate in Hz for joint data producers.  Determines the
            total expected joint frame count as ``joint_fps * duration_sec``.
        video_fps: Frame rate in Hz for video/camera producers.  Determines the
            total expected video frame count as ``video_fps * duration_sec``.
            Ignored when ``video_count`` is ``0``.
        timestamp_mode: Controls how timestamps are assigned to logged frames.
            ``"manual"`` (default) passes explicit monotonically-increasing
            timestamps computed from ``timestamp_start_s + frame_index / fps``.
            ``"real"`` omits the ``timestamp`` argument so the logging API uses
            the wall-clock time at the moment each frame is logged.

    Note:
        ``mode="staggered"`` and ``context_duration_mode="variable"``:
        Both are computed from the base ``duration_sec`` separately (rather than
        stagger being a function of the calculated duration variation).
        With a 50 % stagger and a 75 % duration floor, context 1's
        start is guaranteed to fall before context 0's end.
    """

    duration_sec: int = 5
    parallel_contexts: int = 1
    recording_count: int = 1
    mode: str = MODE_SEQUENTIAL
    joint_count: int = 10
    producer_channels: str = PRODUCER_SYNCHRONOUS
    video_count: int = 0
    image_width: int | None = None
    image_height: int | None = None
    kill_daemon_between_tests: bool = True
    storage_state_action: StorageStateAction = STORAGE_STATE_EMPTY
    stop_method: StopMethod = STOP_METHOD_CLI
    preserve_artifacts_per_test: bool = False
    context_duration_mode: str = DURATION_MODE_FIXED
    wait: bool = False
    joint_fps: int = 60
    video_fps: int = 60
    timestamp_mode: TimestampMode = TIMESTAMP_MODE_MANUAL

    @property
    def has_video(self) -> bool:
        """Return True when this case logs at least one camera stream."""
        return self.video_count > 0

    @property
    def expected_joint_frames(self) -> int:
        """Return expected joint frames: ``joint_fps * duration_sec``."""
        return self.joint_fps * self.duration_sec

    @property
    def expected_video_frames(self) -> int:
        """Return expected video frames: ``video_fps * duration_sec``."""
        return self.video_fps * self.duration_sec

    @property
    def recordings_per_context(self) -> int:
        """Return the base recordings assigned per context.

        Computed as ``recording_count // parallel_contexts``.
        Any remainder is distributed to the first contexts when specs are built.
        """
        return self.recording_count // self.parallel_contexts


@dataclass(frozen=True)
class DataDaemonTestBatch:
    """A named collection of test cases sharing common infrastructure parameters.

    Groups ``DataDaemonTestCase`` instances that should run under the same
    daemon lifecycle, storage, and artifact settings.  The batch-level params
    (``kill_daemon_between_tests``, ``storage_state_action``,
    ``preserve_artifacts_per_test``, ``stop_method``) are propagated to every
    case when :meth:`as_cases` is called, overriding any per-case values.

    Attributes:
        cases: The individual test case workload definitions.
        kill_daemon_between_tests: Propagated to every case; see
            ``DataDaemonTestCase.kill_daemon_between_tests``.
        storage_state_action: Propagated to every case; see
            ``DataDaemonTestCase.storage_state_action``.
        preserve_artifacts_per_test: Propagated to every case; see
            ``DataDaemonTestCase.preserve_artifacts_per_test``.
        stop_method: Propagated to every case; see
            ``DataDaemonTestCase.stop_method``.
            timestamp_mode: Optional batch-level override for timestamp mode. When
                unset, each case keeps its own ``timestamp_mode``.
    """

    cases: tuple[DataDaemonTestCase, ...]
    kill_daemon_between_tests: bool = True
    storage_state_action: StorageStateAction = STORAGE_STATE_EMPTY
    preserve_artifacts_per_test: bool = False
    stop_method: StopMethod = STOP_METHOD_CLI
    timestamp_mode: TimestampMode | None = None

    def as_cases(self) -> list[DataDaemonTestCase]:
        """Return cases with batch-level infrastructure params applied."""
        batch_overrides = {
            "kill_daemon_between_tests": self.kill_daemon_between_tests,
            "storage_state_action": self.storage_state_action,
            "preserve_artifacts_per_test": self.preserve_artifacts_per_test,
            "stop_method": self.stop_method,
        }
        if self.timestamp_mode is not None:
            batch_overrides["timestamp_mode"] = self.timestamp_mode
        return [
            DataDaemonTestCase(**{
                **{
                    f.name: getattr(c, f.name)
                    for f in fields(c)
                    if f.name not in _BATCH_PARAMS
                },
                **batch_overrides,
            })
            for c in self.cases
        ]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def case_id(case: DataDaemonTestCase) -> str:
    """Generate a short human-readable ID for a test case."""
    mode_short = "seq" if case.mode == MODE_SEQUENTIAL else "stag"
    channels_short = (
        "sync" if case.producer_channels == PRODUCER_SYNCHRONOUS else "threaded"
    )
    parts = [
        f"{case.duration_sec}s",
        f"{case.recording_count}recs",
        *(["variable"] if case.context_duration_mode == DURATION_MODE_VARIABLE else []),
        f"{case.parallel_contexts}ctx",
    ]
    if case.parallel_contexts > DataDaemonTestCase.parallel_contexts:
        parts.append(mode_short)
    parts.append(f"{case.joint_count}joints")

    if case.joint_fps != DataDaemonTestCase.joint_fps:
        parts.append(f"{case.joint_fps}hz")
    if case.has_video:
        parts.append(f"{case.video_count}cam")
        parts.append(f"{case.image_width}x{case.image_height}")
        if case.video_fps != DataDaemonTestCase.video_fps:
            parts.append(f"{case.video_fps}hz")
    parts.append(channels_short)
    if case.wait:
        parts.append("wait")
    if case.timestamp_mode == TIMESTAMP_MODE_REAL:
        parts.append("realtime")
    return "-".join(parts)


def case_ids(cases: Sequence[DataDaemonTestCase]) -> list[str]:
    """Generate stable pytest IDs and hyphen-suffix duplicates.

    Pytest auto-suffixes duplicate IDs without a separator. This helper keeps
    IDs readable by explicitly generating ``base-0``, ``base-1``, ... for
    duplicate base IDs while leaving unique IDs unchanged.
    """
    base_ids = [case_id(case) for case in cases]

    totals: dict[str, int] = {}
    for base in base_ids:
        totals[base] = totals.get(base, 0) + 1

    seen: dict[str, int] = {}
    resolved_ids: list[str] = []
    for base in base_ids:
        if totals[base] == 1:
            resolved_ids.append(base)
            continue
        suffix = seen.get(base, 0)
        resolved_ids.append(f"{base}-{suffix}")
        seen[base] = suffix + 1

    return resolved_ids


def has_configured_org() -> bool:
    """Check whether an organization is configured via env or saved config."""
    if os.environ.get("NEURACORE_ORG_ID"):
        return True
    try:
        return bool(get_config_manager().config.current_org_id)
    except Exception:  # noqa: BLE001
        return False


def joint_names_for_count(joint_count: int) -> list[str]:
    """Return a list of joint names of the requested length."""
    if joint_count <= len(BASE_JOINT_NAMES):
        return BASE_JOINT_NAMES[:joint_count]
    generated_names = list(BASE_JOINT_NAMES)
    for index in range(len(BASE_JOINT_NAMES), joint_count):
        generated_names.append(f"synthetic_joint_{index:02d}")
    return generated_names


def camera_names(video_count: int) -> list[str]:
    """Return a list of camera names for the given count."""
    return [f"camera_{index}" for index in range(video_count)]


def generate_joint_values(
    frame_index: int,
    fps: int,
    joint_names: list[str],
) -> dict[str, float]:
    """Generate deterministic sinusoidal joint values for a frame."""
    timestamp = frame_index / fps
    return {
        joint_name: math.sin(timestamp * (0.5 + (index * 0.25)))
        for index, joint_name in enumerate(joint_names)
    }


def case_timeout_seconds(case: DataDaemonTestCase) -> float:
    """Compute a reasonable timeout for waiting on a case to complete."""
    image_pixels = 0
    if (
        case.has_video
        and case.image_width is not None
        and case.image_height is not None
    ):
        image_pixels = case.video_count * case.image_width * case.image_height
    workload_units = (
        case.recording_count
        * case.duration_sec
        * (case.joint_count + max(1, image_pixels // 4096))
    )
    timeout_s = BASE_DATASET_READY_TIMEOUT_S + (workload_units * 0.2)
    if case.context_duration_mode == DURATION_MODE_VARIABLE:
        timeout_s *= 1.25
    return min(MAX_DATASET_READY_TIMEOUT_S, timeout_s)


# ---------------------------------------------------------------------------
# Analysis / reporting
# ---------------------------------------------------------------------------


def _format_timer_stats_line(label: str, stats: dict[str, float]) -> str:
    """Format a timer stats line for analysis output."""
    count = int(stats["count"])
    avg = stats["total"] / count if count > 0 else 0.0
    return f"    {label:<42}  {count:3}x" f"  avg={avg:.3f}s  max={stats['max']:.3f}s"


def log_run_analysis(
    *,
    case: DataDaemonTestCase,
    results: list[ContextResult],
    title: str | None = None,
    status: str | None = None,
    note: str | None = None,
    extra_sections: list[str] | None = None,
    include_in_session_summary: bool = True,
    disk_durations: dict[str, float] | None = None,
) -> str:
    """Log a detailed analysis of a test run for diagnostics."""

    def _format_recording_ids(recording_ids: list[str], *, max_items: int = 6) -> str:
        cleaned = [recording_id for recording_id in recording_ids if recording_id]
        if not cleaned:
            return "[none]"
        if len(cleaned) <= max_items:
            return ", ".join(cleaned)
        shown = ", ".join(cleaned[:max_items])
        return f"{shown}, ... (+{len(cleaned) - max_items} more)"

    separator = "=" * 64
    report_title = title or f"Run analysis: {case_id(case)}"
    lines = [separator, report_title, separator]

    if status is not None:
        lines.append(f"  Analysis status: {status}")
    if note is not None:
        lines.append(f"  {note}")

    lines += [
        f"  Case:          {case.recording_count} recordings x"
        f" {case.duration_sec}s  joints@{case.joint_fps}Hz",
        f"                 {case.joint_count} joints,"
        f" {case.producer_channels} channels",
    ]
    if case.has_video:
        lines.append(
            f"                 {case.video_count} camera(s)"
            f" @ {case.image_width}x{case.image_height}  video@{case.video_fps}Hz"
        )
    lines.append(
        f"  Total joint frames:  {case.recording_count * case.expected_joint_frames}"
    )
    if case.has_video:
        total_video_frames = case.recording_count * case.expected_video_frames
        lines.append(f"  Total video frames:  {total_video_frames}")

    if results:
        lines.append(f"\n  Dataset: {results[0].dataset_name!r}")
        lines.append("\n  Context wall times:")
        for result in sorted(results, key=lambda result: result.context_index):
            wall_s = result.wall_stopped_at - (result.wall_started_at or 0.0)
            recordings_per_context = len(result.recording_ids)
            avg_per_recording = (
                wall_s / recordings_per_context if recordings_per_context else 0.0
            )
            lines.append(
                f"    ctx[{result.context_index}]: {wall_s:.1f}s total,"
                f" {avg_per_recording:.1f}s avg per recording"
            )
            lines.append(
                "      recordings: " f"{_format_recording_ids(result.recording_ids)}"
            )
    else:
        lines.append(
            "\n  Context wall times: unavailable "
            "(run aborted before contexts completed)"
        )

    session_labels = sorted(Timer._stats.keys())
    if session_labels:
        lines.append("\n  Timer stats  (n / avg / max):")
        for label in session_labels:
            stats = Timer._stats[label]
            count = int(stats["count"])
            stats["total"] / count if count > 0 else 0.0
            lines.append(_format_timer_stats_line(label, Timer._stats[label]))

    if disk_durations:
        avg_duration_s = sum(disk_durations.values()) / len(disk_durations)
        lines.append(
            f"\n  Disk recording durations ({len(disk_durations)} recording(s)):"
        )
        for rec_id, dur_s in sorted(disk_durations.items()):
            lines.append(f"    {rec_id}: {dur_s:.3f}s")
        lines.append(f"    avg: {avg_duration_s:.3f}s")

    if extra_sections:
        lines.extend(extra_sections)

    if include_in_session_summary:
        SESSION_RUNS.append({
            "case_id": case_id(case),
            "dataset_name": results[0].dataset_name if results else None,
            "timer_stats": {
                label: dict(Timer._stats[label]) for label in session_labels
            },
            "context_results": [
                {
                    "context_index": result.context_index,
                    "wall_s": result.wall_stopped_at - (result.wall_started_at or 0.0),
                }
                for result in results
            ],
            **(
                {
                    "disk_durations": dict(disk_durations),
                    "avg_disk_duration_s": sum(disk_durations.values())
                    / len(disk_durations),
                }
                if disk_durations
                else {}
            ),
        })

    lines.append(separator)
    report = "\n".join(lines)
    logger.info(report)
    return report
