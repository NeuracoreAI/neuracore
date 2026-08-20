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
from tests.integration.platform.data_daemon.shared.process_control import (
    BUCKET_KEYS,
    LATENCY_BUCKETS_S,
    Timer,
    percentile_upper_bound,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    BASE_DATASET_READY_TIMEOUT_S,
    DETAIL_REALISTIC,
    DURATION_MODE_FIXED,
    DURATION_MODE_VARIABLE,
    LOG_PRESERVE,
    MAX_DATASET_READY_TIMEOUT_S,
    MODE_SEQUENTIAL,
    PACING_BURST_VIDEO,
    PACING_DEADLINE,
    PRODUCER_OLD_PER_THREAD,
    PRODUCER_PER_THREAD,
    PRODUCER_SYNCHRONOUS,
    STOP_METHOD_CLI,
    STORAGE_STATE_DELETE,
    STORAGE_STATE_EMPTY,
    DepthMode,
    LogAction,
    ProducerPacing,
    StopMethod,
    StorageStateAction,
    VideoDetail,
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
    "daemon_log_action",
    "preserve_artifacts_per_test",
    "stop_method",
})


def _unsupported_combination(case: DataDaemonTestCase) -> str | None:
    """Return why *case*'s parameters cannot run together, or None if they can."""
    if case.producer_pacing == PACING_BURST_VIDEO and not (
        case.has_video or case.has_depth
    ):
        return (
            f"producer_pacing={PACING_BURST_VIDEO!r} needs video_count > 0 or "
            "depth_count > 0: it clumps the video streams and there are none"
        )
    return None


@dataclass(frozen=True)
class DataDaemonTestCase:
    """A single parametrised test case for the data-daemon integration suite.

    Each instance fully describes one combination of workload and daemon
    configuration parameters.  Hand-curated cases in each suite's ``test_cases.py``
    are constructed directly using shorthand defaults, typically grouped into a
    ``DataDaemonTestBatch`` that applies shared infrastructure parameters.

    Attributes:
        duration_sec: Capture duration per individual recording, in seconds. All
            frames are generated at ``fps`` Hz so the total expected frame count
            is ``fps * duration_sec``. Under the default ``producer_pacing``
            it is wall-clock time too; a ``"saturate"`` case spends the same
            frame budget as fast as the transport allows.
        parallel_contexts: Number of recording contexts that run concurrently.
            Each context owns an independent robot connection and cycles through
            its share of the total ``recording_count``.
        recording_count: Total number of recordings to produce across *all*
            parallel contexts. Work is distributed as evenly as possible:
            each context gets ``recording_count // parallel_contexts`` recordings,
            and the first ``recording_count % parallel_contexts`` contexts each
            get one additional recording.
        mode: Timestamp layout across parallel contexts — *not* an execution
            order.  Every context in a multi-context case runs concurrently in
            a multiprocessing pool regardless of this value.  ``"sequential"``
            gives all contexts the same timestamp origin, so their recordings
            cover the same span; ``"staggered"`` offsets context *i* by
            ``duration_sec / 2 * i``, so consecutive contexts' spans partly
            overlap.  Single-context cases always use ``"sequential"``.
            joint_count: Number of joint channels to log per frame.  Names are
            drawn from ``BASE_JOINT_NAMES`` and extended with synthetic names
            when the count exceeds the base list length.
        producer_channels: How producer threads are allocated, and how long
            they live.  ``"per_thread"`` runs one thread per stream for the
            whole context lifetime — started before the first
            ``start_recording`` and stopped after the last ``stop_recording``,
            mid-loop at every boundary — mirroring a real camera that does not
            stop between recordings, and the only mode under which the daemon
            is shown what it does with frames logged as a window opens and
            closes.  The other two scope their threads to a single recording
            and join them before ``stop_recording``, so no frame is ever in
            flight when a boundary passes: ``"synchronous"`` (default) logs
            every data type from a single thread in sequence, and
            ``"old_per_thread"`` gives each stream its own thread so they race
            each other inside the recording.  Producer lifetime belongs on this
            axis rather than a separate flag because logging across recordings
            *requires* a thread per stream, so it is not independent of the
            allocation.
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
        random_phase: Controls *where* the explicitly-supplied timestamps sit,
            never *when* the producer runs — that is ``producer_pacing``'s
            business, and the two are independent.  ``False``
            (default) puts frames on an exact ``timestamp_start_s +
            frame_index / fps`` grid.  ``True`` offsets each frame by a
            deterministic pseudo-random amount within
            ``random_phase_jitter_window(fps)``, so the daemon sees
            non-uniformly spaced timestamps.
        skip: When ``True``, the case is skipped at collection time instead of
            being executed.  Lets unstable or not-yet-validated workloads stay
            in the suite (documented and discoverable) without running.  A
            batch with ``skip=True`` forces every case to skip regardless of
            this per-case value.
        video_codec: When set (e.g. ``"h264_medium"``), the case selects that
            global video codec via ``nc.set_video_encoding_options`` before
            recording, so RGB cameras upload a single lossy CRF-23 video and the
            lossless archive is dropped.  ``None`` keeps the default
            lossless+lossy behaviour.
        depth_count: Number of depth camera streams to log per recording. A
            value of ``0`` disables depth entirely. Depth cameras are
            independent streams from RGB cameras (distinct trace identities,
            see :func:`depth_camera_names`) but reuse ``image_width``,
            ``image_height``, ``video_fps``, and the RGB video timestamp
            schedule — depth intentionally adds no separate resolution or
            frame-rate knobs.
        depth_mode: The NumPy dtype depth frames are logged as — ``"float16"``
            or ``"float32"``. Ignored when ``depth_count`` is ``0``.
        video_detail: Pixel content of the synthetic camera frames — realistic
            costs full compression/encode, flat is a cheap solid fill; frame
            identity is embedded either way.
        producer_pacing: When each stream offers its next frame. Every value
            works under every producer lifetime; none of them decide what
            timestamp a frame carries (see ``random_phase``).

    Note:
        ``mode="staggered"`` and ``context_duration_mode="variable"``:
        Both are computed from the base ``duration_sec`` separately (rather than
        stagger being a function of the calculated duration variation).
        With a 50 % stagger and a 75 % duration floor, context 1's timestamp
        start is guaranteed to fall before context 0's timestamp end.
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
    daemon_log_action: LogAction = LOG_PRESERVE
    stop_method: StopMethod = STOP_METHOD_CLI
    preserve_artifacts_per_test: bool = False
    context_duration_mode: str = DURATION_MODE_FIXED
    wait: bool = False
    joint_fps: int = 60
    video_fps: int = 60
    random_phase: bool = False
    skip: bool = False
    video_codec: str | None = None
    depth_count: int = 0
    depth_mode: DepthMode = "float32"
    video_detail: VideoDetail = DETAIL_REALISTIC
    producer_pacing: ProducerPacing = PACING_DEADLINE

    def __post_init__(self) -> None:
        """Reject parameter combinations this case's shape cannot run."""
        problem = _unsupported_combination(self)
        if problem is not None:
            raise ValueError(problem)

    @property
    def has_video(self) -> bool:
        """Return True when this case logs at least one camera stream."""
        return self.video_count > 0

    @property
    def has_depth(self) -> bool:
        """Return True when this case logs at least one depth camera stream."""
        return self.depth_count > 0

    @property
    def lossy_only(self) -> bool:
        """Return True when the case drops the lossless archive for RGB video."""
        return self.video_codec == "h264_medium"

    @property
    def expected_joint_frames(self) -> int:
        """Return expected joint frames: ``joint_fps * duration_sec``."""
        return self.joint_fps * self.duration_sec

    @property
    def expected_video_frames(self) -> int:
        """Return expected video frames: ``video_fps * duration_sec``."""
        return self.video_fps * self.duration_sec

    @property
    def expected_depth_frames(self) -> int:
        """Return expected depth frames per camera.

        Depth cameras reuse the RGB video timestamp schedule (see
        :attr:`depth_count`), so this equals :attr:`expected_video_frames`.
        """
        return self.expected_video_frames

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
    ``daemon_log_action``, ``preserve_artifacts_per_test``, ``stop_method``)
    are propagated to every case via :meth:`as_cases`.

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
        skip: When ``True``, every case in the batch is skipped at collection
            time.  When ``False`` (default), each case keeps its own per-case
            ``skip`` value, so individual cases can still opt out.
        producer_channels: Workload override applied to every case when set;
            see ``DataDaemonTestCase.producer_channels``.  ``None`` (default)
            leaves each case's own value alone.  Lets a suite declare one
            producer model — e.g. ``"per_thread"`` — across its whole matrix
            instead of restating it on every case.
        producer_pacing: Workload override applied to every case when set; see
            ``DataDaemonTestCase.producer_pacing``.  ``None`` (default) leaves
            each case's own value alone.
    """

    cases: tuple[DataDaemonTestCase, ...]
    kill_daemon_between_tests: bool = True
    storage_state_action: StorageStateAction = STORAGE_STATE_DELETE
    daemon_log_action: LogAction = LOG_PRESERVE
    preserve_artifacts_per_test: bool = False
    stop_method: StopMethod = STOP_METHOD_CLI
    skip: bool = False
    producer_channels: str | None = None
    producer_pacing: ProducerPacing | None = None

    def as_cases(self) -> list[DataDaemonTestCase]:
        """Return cases with batch-level infrastructure params applied."""
        batch_overrides = {
            "kill_daemon_between_tests": self.kill_daemon_between_tests,
            "storage_state_action": self.storage_state_action,
            "daemon_log_action": self.daemon_log_action,
            "preserve_artifacts_per_test": self.preserve_artifacts_per_test,
            "stop_method": self.stop_method,
        }
        if self.skip:
            batch_overrides["skip"] = True
        # Workload overrides are opt-in; unset = keep case default.
        if self.producer_channels is not None:
            batch_overrides["producer_channels"] = self.producer_channels
        if self.producer_pacing is not None:
            batch_overrides["producer_pacing"] = self.producer_pacing
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
    parts = [
        f"{case.duration_sec}s",
        f"{case.recording_count}recs",
        *(["variable"] if case.context_duration_mode == DURATION_MODE_VARIABLE else []),
    ]
    if case.parallel_contexts > DataDaemonTestCase.parallel_contexts:
        parts.append(f"{case.parallel_contexts}ctx")
        parts.append(mode_short)
    parts.append(f"{case.joint_count}joints")
    if case.joint_fps != DataDaemonTestCase.joint_fps:
        parts.append(f"{case.joint_fps}hz")
    if case.has_video:
        parts.append(f"{case.video_count}cam")
        parts.append(f"{case.image_width}x{case.image_height}")
        if case.video_fps != DataDaemonTestCase.video_fps:
            parts.append(f"{case.video_fps}hz")
        if case.video_codec is not None:
            parts.append(case.video_codec)
        if case.video_detail != DataDaemonTestCase.video_detail:
            parts.append(f"{case.video_detail}frames")
    if case.has_depth:
        parts.append(f"{case.depth_count}depth")
        parts.append(case.depth_mode)
    if case.producer_channels == PRODUCER_OLD_PER_THREAD:
        parts.append("old-per-thread")
    elif case.producer_channels == PRODUCER_PER_THREAD:
        parts.append("per-thread")
    if case.producer_pacing != DataDaemonTestCase.producer_pacing:
        parts.append(case.producer_pacing)
    if case.random_phase:
        parts.append("random-phase")
    if case.wait:
        parts.append("wait")
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
    """Return a list of RGB camera names for the given count."""
    return [f"camera_{index}" for index in range(video_count)]


def depth_camera_names(depth_count: int) -> list[str]:
    """Return a list of depth camera names for the given count.

    Distinct from :func:`camera_names` — depth cameras are independent
    stream identities (``DEPTH_IMAGES/depth_camera_N`` traces) even though a
    depth-enabled case reuses the RGB spec's resolution and frame rate.
    """
    return [f"depth_camera_{index}" for index in range(depth_count)]


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
        (case.has_video or case.has_depth)
        and case.image_width is not None
        and case.image_height is not None
    ):
        camera_count = case.video_count + case.depth_count
        image_pixels = camera_count * case.image_width * case.image_height
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


MIN_SAMPLES_FOR_PERCENTILE = 100
"""Below this sample count a percentile is not reported — it would just be the max."""


def _percentile_text(stats: dict[str, float], quantile: float) -> str:
    """Render one percentile column, or empty when it would say nothing."""
    count = int(stats["count"])
    label = f"p{int(quantile * 100)}"
    if count < MIN_SAMPLES_FOR_PERCENTILE:
        return ""
    if not any(stats.get(key, 0.0) for key in BUCKET_KEYS):
        return f"{label}=n/a"
    bound = percentile_upper_bound(stats, quantile)
    if bound is None:
        return f"{label}>{LATENCY_BUCKETS_S[-1]:.3f}s"
    # A bucket edge, not an interpolated value — "<=" keeps that honest.
    return f"{label}<={min(bound, stats['max']):.3f}s"


def _format_timer_stats_line(label: str, stats: dict[str, float]) -> str:
    """Format a timer stats line for analysis output."""
    count = int(stats["count"])
    avg = stats["total"] / count if count > 0 else 0.0
    p95_text = _percentile_text(stats, 0.95)
    p99_text = _percentile_text(stats, 0.99)
    return (
        f"    {label:<42}  {count:3}x"
        f"  avg={avg:.3f}s  {p95_text:<12}  {p99_text:<12}"
        f"  max={stats['max']:.3f}s"
    )


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
    label_prefix: str | None = None,
    test_wall_s: float | None = None,
) -> str:
    """Log a detailed analysis of a test run for diagnostics."""

    display_case_id = (
        f"{label_prefix}-{case_id(case)}" if label_prefix else case_id(case)
    )
    separator = "=" * 64
    report_title = title or f"Run analysis: {display_case_id}"
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
    if case.has_depth:
        lines.append(
            f"                 {case.depth_count} depth camera(s)"
            f" @ {case.image_width}x{case.image_height} ({case.depth_mode})"
        )
    lines.append(
        f"  Total joint frames:  {case.recording_count * case.expected_joint_frames}"
    )
    if case.has_video:
        total_video_frames = case.recording_count * case.expected_video_frames
        lines.append(f"  Total video frames:  {total_video_frames}")
    if case.has_depth:
        total_depth_frames = case.recording_count * case.expected_depth_frames
        lines.append(f"  Total depth frames:  {total_depth_frames}")

    if test_wall_s is not None:
        lines.append(f"\n  Test wall time:  {test_wall_s:.1f}s")

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
            "case_id": display_case_id,
            "dataset_name": results[0].dataset_name if results else None,
            "test_wall_s": test_wall_s,
            "timer_stats": {
                label: dict(Timer._stats[label])
                for label in session_labels
                if not label.startswith("stop_daemon")
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
    logger.info("\n%s", report)
    return report
