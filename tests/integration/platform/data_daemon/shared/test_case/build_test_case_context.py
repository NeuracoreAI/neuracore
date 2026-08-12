"""Context-spec interpretation and recording worker logic.

Translates a ``DataDaemonTestCase`` into per-context worker specs, executes
the recording workload, and provides the context-mode assertion.
Configuration dataclasses and the matrix builder live in
``matrix_test_configs.py``; per-suite case lists live in ``test_cases.py``.
"""

from __future__ import annotations

import functools
import logging
import multiprocessing
import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

import neuracore as nc
from neuracore.core.streaming.recording_state_manager import RecordingStateManager
from tests.integration.platform.data_daemon.shared.auth import ensure_login
from tests.integration.platform.data_daemon.shared.process_control import (
    MAX_TIME_TO_LOG_S,
    Timer,
    init_worker_logging,
    relayed_worker_logs,
    surface_worker_errors,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
    camera_names,
    case_id,
    depth_camera_names,
    generate_joint_values,
    joint_names_for_count,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    BACKLOG_BACKOFF_BASE_S,
    BACKLOG_BACKOFF_MAX_S,
    BACKLOG_STALL_BUDGET_S,
    DATASET_POLL_INTERVAL_S,
    DURATION_MODE_VARIABLE,
    DURATION_VARIABLE_MAX_FACTOR,
    DURATION_VARIABLE_MIN_FACTOR,
    JOINT_KINDS,
    MAX_TIME_TO_START_S,
    MODE_STAGGERED,
    PACING_BURST_VIDEO,
    PACING_DEADLINE,
    PACING_SATURATE_WITH_BACKOFF,
    PER_THREAD_LOGGING_TAIL_S,
    PRODUCER_OLD_PER_THREAD,
    PRODUCER_PER_THREAD,
    PRODUCER_SYNCHRONOUS,
    STOP_RECORDING_NO_WAIT_SLA_S,
    STOP_RECORDING_OVERHEAD_PER_SEC,
    STOP_RECORDING_UPLOAD_SLA_PER_JOINT_SAMPLE_S,
    STOP_RECORDING_UPLOAD_SLA_PER_VIDEO_PIXEL_S,
    DepthMode,
    random_phase_jitter_window,
)
from tests.integration.platform.data_daemon.shared.test_case.frame_source import (
    encode_depth_frame,
    frame_code_base,
    make_camera_feed,
    preallocate_depth_buffer,
    prewarm_frame_bank,
)

logger = logging.getLogger(__name__)

CONTEXT_DURATION_RANDOM = random.Random(0)

# Stream discriminators feeding ``stream_phase_seed`` so a recording's joint and
# video streams draw independent phase offsets.
JOINT_STREAM = 0
VIDEO_STREAM = 1

# Producer pacing for the performance suites
LOG_LOOP_FREQUENCY_HZ = 120
LOG_LOOP_INTERVAL_S = 1.0 / LOG_LOOP_FREQUENCY_HZ

# The daemon's words for a refused frame; matched, not typed, because the bridge
# raises a bare RuntimeError.
_STALL_MESSAGE = "video logging stalled"

# Semantic trace data type each ``nc.log_joint_*`` call writes to. Derived from
# JOINT_KINDS so the two cannot drift apart.
_JOINT_DATA_TYPES = {kind: kind.upper() for kind in JOINT_KINDS}


def rgb_frame_code(
    *,
    context_index: int,
    recording_index: int,
    camera_index: int,
    frame_index: int,
) -> int:
    """Return the integer painted into a camera frame's pixels."""
    return (
        frame_code_base(
            context_index=context_index,
            recording_ordinal=recording_index,
            camera_index=camera_index,
        )
        + frame_index
    )


class EmittedFrame(NamedTuple):
    """One frame a producer logged, bracketed on the wall clock.

    A publish stamp is taken inside the ``log_*`` call, so a test only knows it
    lies somewhere in ``[emitted_at, completed_at]`` (see
    :func:`_classify_boundary_frames`).

    Attributes:
        timestamp: The value that reaches disk; the only field compared there.
        frame_index: Session-wide, never resets across recordings — recovers
            the painted frame code (see :func:`rgb_frame_code`).
        handle: Recording handle latched before the call, or ``None``. Reflects
            the local logging gate, not the daemon's actual window.
    """

    timestamp: float
    frame_index: int
    emitted_at: float
    completed_at: float
    handle: str | None


@dataclass(frozen=True, slots=True)
class TraceClassification:
    """What one recording requires of one trace's frames.

    The single verdict every classification rule reports in, so every consumer
    reads one object rather than unpacking a tuple, and a rule that grows a
    third answer grows a field here instead of a return position.

    The lists partition the frames the rule was given: a frame appears in
    exactly one.

    Attributes:
        owed: Frames the recording provably owns. Every one must reach disk.
        unknowable: Frames logged while a boundary was passing. The daemon's
            answer is correct either way, so they are dropped from *both* sides
            of the comparison rather than tolerated on one.
    """

    owed: list[EmittedFrame]
    unknowable: list[EmittedFrame] = field(default_factory=list)

    @property
    def owed_timestamps(self) -> list[float]:
        """The capture timestamps the on-disk trace must hold, in order."""
        return [frame.timestamp for frame in self.owed]

    @property
    def unknowable_timestamps(self) -> set[float]:
        """Capture timestamps allowed to be present or absent."""
        return {frame.timestamp for frame in self.unknowable}


@dataclass(frozen=True, slots=True)
class ObservedFrameCodes:
    """Camera frame codes one recording claimed, per camera name.

    Reported separately because a lifetime producer's frame index is
    session-wide, so codes can't be reconstructed from the recording ordinal.
    """

    inside: dict[str, list[int]]
    unknowable: dict[str, set[int]]


@dataclass(frozen=True, slots=True)
class RecordingControlBounds:
    """Wall-clock brackets around the two control calls that bound a recording.

    The daemon's window is ``[started_at_ns, stopped_at_ns)``, and both bounds
    are stamped on the publish clock *inside* the producer-side
    ``start_recording`` / ``stop_recording`` calls — never from the capture
    timestamp the caller passes. Neither instant is visible from here, but each
    is known to fall within the call that carries it, and the control thread can
    stamp both edges of that call on the same clock the producer threads use.

    Attributes:
        handle: The SDK recording handle for this recording, as the producer
            threads see it while it is current.
        start_called_at: Wall clock immediately before ``nc.start_recording``.
        start_returned_at: Wall clock immediately after it returned. The
            window's lower bound is somewhere in between.
        stop_called_at: Wall clock immediately before ``nc.stop_recording``.
        stop_returned_at: Wall clock immediately after it returned. The window's
            upper bound is somewhere in between.
    """

    handle: str | None
    start_called_at: float
    start_returned_at: float
    stop_called_at: float
    stop_returned_at: float


@dataclass(frozen=True, slots=True)
class RecordingExpectedTimestamps:
    """Expected timestamps per trace for one recording, keyed by semantic trace name.

    Produced during the recording loop (once the recording key is known) and
    consumed by :func:`~disk_helpers.assert_disk_recording_properties`
    to verify on-disk trace.json files match the timestamps that were logged.
    These are the *same lists* the producer emitted, so the assertion is exact
    equality regardless of ``random_phase``.

    Attributes:
        by_trace: Maps semantic trace key (e.g. ``"JOINT_POSITIONS"``,
            ``"camera_0"``) to the :class:`TraceClassification` the recording's
            boundary rule returned for it. One map rather than one per verdict,
            so the keys cannot drift apart and a new verdict needs no new field
            here.
    """

    by_trace: dict[str, TraceClassification]


@dataclass(frozen=True, slots=True)
class ContextExpectedTimestamps:
    """Expected timestamps for all recordings produced by one context worker.

    Attributes:
        by_recording: Maps the on-disk recording directory name to its
            :class:`RecordingExpectedTimestamps`. The directory name is the
            integer ``recording_index`` as a string.
    """

    by_recording: dict[str, RecordingExpectedTimestamps]


@dataclass(frozen=True, slots=True)
class ContextCaseSpec:
    duration_sec: int
    joint_count: int
    producer_channels: str
    video_count: int
    image_width: int | None
    image_height: int | None
    joint_fps: int
    video_fps: int
    wait: bool
    random_phase: bool
    video_detail: str
    producer_pacing: str
    depth_count: int = 0
    depth_mode: DepthMode = "float32"

    @property
    def stop_recording_sla_s(self) -> float:
        """Seconds allowed for the ``nc.stop_recording`` call.

        ``wait=False`` is fire-and-forget — the call never blocks on the
        upload pipeline — so it gets a flat constant. ``wait=True`` blocks
        until every trace has uploaded, so its budget is the sum of the
        joint-data and video-data upload costs: total joint samples
        (``duration_sec * joint_count * joint_fps``) and total video pixels
        across both RGB and depth cameras (``duration_sec * video_fps *
        (video_count + depth_count) * image_width * image_height``), each
        times an observed per-unit upload cost. Depth cameras reuse the RGB
        per-pixel upload constant as a first approximation — both are
        video-family traces that always keep a lossless archive, so their
        upload cost is comparable order-of-magnitude, though not necessarily
        identical. The budget is floored at the duration-based overhead so
        short or low-volume recordings keep a sane minimum.
        """
        if not self.wait:
            return STOP_RECORDING_NO_WAIT_SLA_S
        duration_floor = self.duration_sec * STOP_RECORDING_OVERHEAD_PER_SEC
        joint_budget = (
            self.duration_sec
            * self.joint_count
            * self.joint_fps
            * STOP_RECORDING_UPLOAD_SLA_PER_JOINT_SAMPLE_S
        )
        video_budget = 0.0
        camera_count = self.video_count + self.depth_count
        if camera_count and self.image_width and self.image_height:
            video_budget = (
                self.duration_sec
                * self.video_fps
                * camera_count
                * self.image_width
                * self.image_height
                * STOP_RECORDING_UPLOAD_SLA_PER_VIDEO_PIXEL_S
            )
        return max(duration_floor, joint_budget + video_budget)


@dataclass(frozen=True, slots=True)
class ContextResult:
    """Per-context result from a completed recording workload.

    Produced by :func:`context_worker` and consumed by assertion helpers
    and verification functions throughout the test suite.

    A recording is addressed by:

    - ``recording_ids`` — the cloud ``recording_id`` (TEXT) for each recording.
      These are what cloud verification (``verify_cloud_results``) matches
      against the dataset's ``recording.id``. The daemon mints them
      asynchronously, so an entry may be an empty string until the test
      resolves it (via ``resolve_cloud_recording_ids``) once online.

    - ``recording_indexes`` — the daemon-assigned local INTEGER
      ``recording_index`` for each recording, resolved from the source DB.
      These are the on-disk directory names and the daemon-DB join key.
    - ``source`` is the ``(robot_id, robot_instance)`` identity used to correlate
      a worker's recordings to daemon-minted ``recording_index`` values without
      relying on the local handle.
    """

    dataset_name: str
    recording_ids: list[str]
    robot_name: str
    joint_names: list[str]
    camera_names: list[str]
    joint_frame_count: int
    video_frame_count: int
    joint_fps: int
    video_fps: int
    duration_sec: int
    timestamp_start_s: float
    timestamp_end_s: float
    marker_names: list[str]
    has_video: bool
    context_index: int
    wall_started_at: float | None
    wall_stopped_at: float
    random_phase: bool
    expected_timestamps: ContextExpectedTimestamps = field(
        default_factory=lambda: ContextExpectedTimestamps(by_recording={})
    )
    timer_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    recording_indexes: list[int] = field(default_factory=list)
    source: tuple[str, int] = ("", 0)
    depth_camera_names: list[str] = field(default_factory=list)
    depth_frame_count: int = 0
    depth_mode: DepthMode = "float32"
    has_depth: bool = False
    observed_frame_codes: dict[str, ObservedFrameCodes] = field(default_factory=dict)
    """Painted camera frame codes per recording, keyed by ``recording_index``.

    Populated only for producers that outlive a recording, whose session-wide
    frame index makes the codes unpredictable from the ordinal.
    """
    expected_video_stop_timestamp_by_recording: dict[str, float] = field(
        default_factory=dict
    )
    """Nominal capture-clock upper bound of each recording's video.

    On the capture clock the frame timestamps use, not the wall clock the control
    calls carry. Bounds how far an RGB trace's last on-disk timestamp may trail
    the recording's end, which is the only way an orphaned tail chunk shows up:
    the exact-equality check only ever sees what reached disk.
    """


@dataclass(frozen=True, slots=True)
class ContextSpec:
    case: ContextCaseSpec
    context_index: int
    robot_name: str
    dataset_name: str
    recordings_per_context: int
    expected_joint_frames: int
    expected_video_frames: int
    timestamp_start_s: float
    timestamp_end_s: float
    assert_deadline: bool = False
    log_interval_s: float = 0.0


def build_context_specs(
    case: DataDaemonTestCase,
    dataset_name: str | None = None,
    assert_deadline: bool = False,
) -> list[ContextSpec]:
    """Build per-context worker specs for a matrix case.

    ``assert_deadline`` is the performance suites' marker: besides arming the
    ``Timer`` deadline assertions it imposes the fixed
    :data:`LOG_LOOP_FREQUENCY_HZ` inter-frame interval, so a performance case
    measures latency at a known offered rate rather than under whatever load
    the machine happens to sustain.  Independent of ``producer_pacing``, which
    decides only whether a stream waits for its deadline.
    """
    specs: list[ContextSpec] = []
    timestamp_stagger_s = case.duration_sec / 2.0
    base_recordings_per_context = case.recording_count // case.parallel_contexts
    recording_remainder = case.recording_count % case.parallel_contexts
    shared_dataset_name = (
        dataset_name or f"testing_dataset_{case_id(case)}_{uuid.uuid4().hex[:6]}"
    )

    for context_index in range(case.parallel_contexts):
        timestamp_start_s = 0.0
        if context_index > 0 and case.mode == MODE_STAGGERED:
            timestamp_start_s = float(timestamp_stagger_s * context_index)

        if case.context_duration_mode == DURATION_MODE_VARIABLE:
            context_duration_sec = max(
                1,
                min(
                    int(
                        case.duration_sec
                        * CONTEXT_DURATION_RANDOM.uniform(
                            DURATION_VARIABLE_MIN_FACTOR, DURATION_VARIABLE_MAX_FACTOR
                        )
                    ),
                    RecordingStateManager.MAX_RECORDING_DURATION_S,
                ),
            )
        else:
            context_duration_sec = case.duration_sec

        recordings_for_context = base_recordings_per_context + (
            1 if context_index < recording_remainder else 0
        )

        specs.append(
            ContextSpec(
                case=ContextCaseSpec(
                    duration_sec=context_duration_sec,
                    joint_count=case.joint_count,
                    producer_channels=case.producer_channels,
                    video_count=case.video_count,
                    image_width=case.image_width,
                    image_height=case.image_height,
                    joint_fps=case.joint_fps,
                    video_fps=case.video_fps,
                    wait=case.wait,
                    random_phase=case.random_phase,
                    video_detail=case.video_detail,
                    producer_pacing=case.producer_pacing,
                    depth_count=case.depth_count,
                    depth_mode=case.depth_mode,
                ),
                context_index=context_index,
                robot_name=f"matrix_robot_{uuid.uuid4().hex[:10]}",
                dataset_name=shared_dataset_name,
                recordings_per_context=recordings_for_context,
                expected_joint_frames=case.joint_fps * context_duration_sec,
                expected_video_frames=case.video_fps * context_duration_sec,
                timestamp_start_s=timestamp_start_s,
                timestamp_end_s=(
                    timestamp_start_s + context_duration_sec * recordings_for_context
                ),
                assert_deadline=assert_deadline,
                log_interval_s=LOG_LOOP_INTERVAL_S if assert_deadline else 0.0,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Recording worker functions
# ---------------------------------------------------------------------------


def _cleanup_test_worker_robot(robot: object | None) -> None:
    """Clean up temp dirs and recording context on a worker robot."""
    if robot is None:
        return

    temp_dir = getattr(robot, "_temp_dir", None)
    if temp_dir is not None:
        try:
            temp_dir.cleanup()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to cleanup worker robot temp dir", exc_info=True)
        finally:
            robot._temp_dir = None

    if hasattr(robot, "_daemon_recording_context"):
        robot._daemon_recording_context = None


def stream_phase_seed(context_index: int, recording_ordinal: int, stream: int) -> int:
    """Return a stable RNG seed for one recording's stream.

    Distinct per (context, recording, stream) so parallel contexts and the joint
    and video streams within a recording all draw independent phase offsets,
    while a rerun of the same case reproduces them exactly.
    """
    return (context_index * 1_000 + recording_ordinal) * 2 + stream


def precompute_timestamps(
    timestamp_start_s: float,
    frame_count: int,
    fps: int,
    random_phase: bool = False,
    seed: int = 0,
) -> list[float]:
    """Return the complete synthetic timestamp sequence for one stream.

    Frames sit on an exact ``timestamp_start_s + frame_index / fps`` grid.  With
    *random_phase* each frame is additionally offset by a pseudo-random amount
    within :func:`random_phase_jitter_window`, so the daemon sees non-uniformly
    spaced timestamps.  The offsets are drawn from a *seed*-derived RNG, which is
    what lets the caller hand the very same list to both the producer and the
    on-disk expectation.
    """
    if not random_phase:
        return [
            timestamp_start_s + frame_index / fps for frame_index in range(frame_count)
        ]

    rng = random.Random(seed)
    window = random_phase_jitter_window(fps)
    return [
        timestamp_start_s + frame_index / fps + rng.uniform(-window, window)
        for frame_index in range(frame_count)
    ]


def _should_pace(pacing: str, is_video: bool) -> bool:
    """Whether a stream sleeps to its wall-clock deadline under *pacing*.

    ``PACING_BURST_VIDEO`` un-paces every stream that feeds the daemon's video
    pipeline — depth cameras as well as RGB ones, since both backlog the same
    spool when the encoder cannot keep up.  Either saturating value un-paces
    everything; the daemon's spool cap is all that holds those back.
    """
    if pacing == PACING_DEADLINE:
        return True
    if pacing == PACING_BURST_VIDEO:
        return not is_video
    return False


def _should_backoff(pacing: str) -> bool:
    """Whether a rejected frame is retried rather than fatal under *pacing*."""
    return pacing == PACING_SATURATE_WITH_BACKOFF


def _await_frame_deadline(deadline: float, stop_event: threading.Event) -> bool:
    """Wait until *deadline*, unless *stop_event* fires first.

    Only the lifetime producer waits: it has no frame count to bound it, so its
    schedule is real time itself. The producers scoped to one recording are
    bounded by a fixed frame count and emit as fast as the transport allows.

    Returns:
        ``True`` when *stop_event* fired during the wait, meaning the caller's
        loop should stop; ``False`` otherwise.
    """
    remaining = deadline - time.time()
    if remaining <= 0:
        return False
    return stop_event.wait(remaining)


def _log_with_backlog_backoff(
    call: Callable[[], None], stop_event: threading.Event
) -> None:
    """Run one ``nc.log_*`` call, retrying while the daemon is backlogged.

    Wrapping the individual call rather than the whole emit keeps the retry
    exactly-once: a refused frame was never admitted, so re-sending it cannot
    duplicate one, and the stream's other channels are left alone.

    Raises:
        RuntimeError: the daemon's stall error, once one frame has absorbed
            :data:`BACKLOG_STALL_BUDGET_S` or *stop_event* fires mid backlog.
            The frame never landed, so it must surface rather than be dropped
            from the report.
    """
    delay = BACKLOG_BACKOFF_BASE_S
    stalled_for = 0.0
    while True:
        try:
            call()
            return
        except RuntimeError as exc:
            if _STALL_MESSAGE not in str(exc):
                raise
            stalled_for += delay
            if stalled_for >= BACKLOG_STALL_BUDGET_S or stop_event.wait(delay):
                raise
            delay = min(delay * 2, BACKLOG_BACKOFF_MAX_S)


@dataclass(frozen=True, slots=True)
class StreamPlan:
    """One producer stream: which channels it logs, and under which marker.

    The single description of how a workload decomposes into streams. Producers
    consume it through :class:`StreamEmitter` rather than re-deriving channel
    names, log functions or frame codes for themselves.
    """

    name: str
    fps: int
    channel_names: tuple[str, ...]
    marker_name: str | None = None
    # Which ``nc.log_joint_*`` calls one frame makes. Empty for camera streams.
    # Per-thread streams carry exactly one; the synchronous producer bundles
    # all three into a single stream sharing one marker.
    joint_kinds: tuple[str, ...] = ()
    camera_indexes: tuple[int, ...] = ()
    # Sample dtype for depth streams; ignored by every other kind.
    depth_mode: DepthMode = "float32"

    @property
    def is_rgb(self) -> bool:
        """Whether this stream logs RGB camera frames."""
        return self.name == "rgb"

    @property
    def is_depth(self) -> bool:
        """Whether this stream logs depth camera frames."""
        return self.name == "depth"

    @property
    def is_video(self) -> bool:
        """Whether this stream feeds the video pipeline rather than joint data.

        Both camera kinds run on the video timestamp schedule and are paced
        together, so the producers branch on this rather than on
        :attr:`is_rgb`.
        """
        return self.is_rgb or self.is_depth

    @property
    def trace_keys(self) -> list[str]:
        """Semantic trace keys one logged frame from this stream touches.

        Matches the ``data_type/data_type_name`` keys resolved from the DB in
        disk_helpers — one entry per named channel per data type, plus this
        stream's own ``CUSTOM_1D`` marker when it has one.
        """
        from neuracore_types.utils import validate_safe_name

        if self.is_rgb:
            data_types = ["RGB_IMAGES"]
        elif self.is_depth:
            data_types = ["DEPTH_IMAGES"]
        else:
            data_types = [_JOINT_DATA_TYPES[kind] for kind in self.joint_kinds]
        keys = [
            f"{data_type}/{validate_safe_name(name)}"
            for data_type in data_types
            for name in self.channel_names
        ]
        if self.marker_name is not None:
            keys.append(f"CUSTOM_1D/{validate_safe_name(self.marker_name)}")
        return keys


def _per_camera_plans(
    kind: str,
    camera_name_list: list[str],
    video_fps: int,
    *,
    depth_mode: DepthMode = "float32",
) -> list[StreamPlan]:
    """One stream per camera of *kind*, each carrying its own marker.

    Every producer but the synchronous one gives a camera its own thread, so it
    gets its own marker series too.
    """
    return [
        StreamPlan(
            name=kind,
            marker_name=f"marker_{camera_name}",
            fps=video_fps,
            channel_names=(camera_name,),
            camera_indexes=(camera_index,),
            depth_mode=depth_mode,
        )
        for camera_index, camera_name in enumerate(camera_name_list)
    ]


def _bundled_camera_plan(
    kind: str,
    camera_name_list: list[str],
    video_fps: int,
    *,
    depth_mode: DepthMode = "float32",
) -> StreamPlan | None:
    """One stream covering every camera of *kind*, or ``None`` if there are none.

    The synchronous producer logs a kind's cameras from one thread and emits a
    single marker series of its own, so these plans carry no marker.
    """
    if not camera_name_list:
        return None
    return StreamPlan(
        name=kind,
        fps=video_fps,
        channel_names=tuple(camera_name_list),
        camera_indexes=tuple(range(len(camera_name_list))),
        depth_mode=depth_mode,
    )


def build_stream_plans(
    *,
    joint_names: list[str],
    camera_name_list: list[str],
    depth_camera_name_list: list[str],
    depth_mode: DepthMode,
    joint_fps: int,
    video_fps: int,
) -> list[StreamPlan]:
    """Decompose a workload into one stream per camera and per joint data type."""
    return [
        *_per_camera_plans("rgb", camera_name_list, video_fps),
        *_per_camera_plans(
            "depth", depth_camera_name_list, video_fps, depth_mode=depth_mode
        ),
        *(
            StreamPlan(
                name=kind,
                marker_name=f"marker_{kind}",
                fps=joint_fps,
                channel_names=tuple(joint_names),
                joint_kinds=(kind,),
            )
            for kind in JOINT_KINDS
        ),
    ]


def build_synchronous_stream_plans(
    *,
    joint_names: list[str],
    camera_name_list: list[str],
    depth_camera_name_list: list[str],
    depth_mode: DepthMode,
    joint_fps: int,
    video_fps: int,
    marker_name: str,
) -> tuple[StreamPlan, StreamPlan | None, StreamPlan | None]:
    """Decompose a workload for the single-threaded producer.

    One stream per kind rather than per camera: the single thread logs every
    camera of a kind at each video timestamp. Only the joint stream carries the
    marker — the synchronous producer emits one marker series, not one per
    camera as the threaded producer does.
    """
    joint_plan = StreamPlan(
        name="joints",
        marker_name=marker_name,
        fps=joint_fps,
        channel_names=tuple(joint_names),
        joint_kinds=JOINT_KINDS,
    )
    video_plan = _bundled_camera_plan("rgb", camera_name_list, video_fps)
    depth_plan = _bundled_camera_plan(
        "depth", depth_camera_name_list, video_fps, depth_mode=depth_mode
    )
    return joint_plan, video_plan, depth_plan


def stream_plans_for_case(
    *,
    producer_channels: str,
    joint_names: list[str],
    camera_name_list: list[str],
    depth_camera_name_list: list[str],
    depth_mode: DepthMode,
    joint_fps: int,
    video_fps: int,
    marker_name: str,
) -> list[StreamPlan]:
    """Every stream the producer for *producer_channels* will actually run."""
    if producer_channels == PRODUCER_SYNCHRONOUS:
        return [
            plan
            for plan in build_synchronous_stream_plans(
                joint_names=joint_names,
                camera_name_list=camera_name_list,
                depth_camera_name_list=depth_camera_name_list,
                depth_mode=depth_mode,
                joint_fps=joint_fps,
                video_fps=video_fps,
                marker_name=marker_name,
            )
            if plan is not None
        ]
    return build_stream_plans(
        joint_names=joint_names,
        camera_name_list=camera_name_list,
        depth_camera_name_list=depth_camera_name_list,
        depth_mode=depth_mode,
        joint_fps=joint_fps,
        video_fps=video_fps,
    )


@dataclass(slots=True, eq=False)  # identity-keyed: one emitter per thread
class StreamEmitter:
    """Binds a :class:`StreamPlan` to one recording's logging calls.

    Owns every per-frame side effect a stream has, so producers cannot drift in
    what they log, how frames are identified, or how the calls are labelled for
    reporting.
    """

    plan: StreamPlan
    robot_name: str
    context_index: int
    recording_index: int
    # Cuts a backlog retry short, so it is needed whether or not one can happen.
    stop_event: threading.Event
    assert_deadline: bool = False
    # Set only by PACING_SATURATE_WITH_BACKOFF: retry a refused frame.
    backoff: bool = False
    feed: object | None = None
    depth_buffer: np.ndarray | None = None
    image_width: int | None = None
    image_height: int | None = None
    # Resolved once here rather than per frame: every producer reports each
    # emitted frame against them, and building them walks validate_safe_name.
    trace_keys: tuple[str, ...] = ()

    def prepare(self, image_width: int | None, image_height: int | None, detail: str):
        """Build this stream's frame source. Called on the stream's own thread."""
        self.image_width = image_width
        self.image_height = image_height
        self.trace_keys = tuple(self.plan.trace_keys)
        self.feed = make_camera_feed(
            self.plan.is_rgb, image_width, image_height, detail
        )
        self.depth_buffer = preallocate_depth_buffer(
            self.plan.is_depth, image_width, image_height, self.plan.depth_mode
        )

    def _log_video_frame(self, label: str, call: Callable[[], None]) -> None:
        """Log one video frame, timed, retrying it if the daemon is backlogged.

        The ``Timer`` sits *inside* the retry so each attempt is measured on its
        own; a refused attempt has its ``RuntimeError`` in flight, which the
        Timer skips its deadline assertion for.
        """

        def timed() -> None:
            with Timer(
                MAX_TIME_TO_LOG_S, label=label, assert_deadline=self.assert_deadline
            ):
                call()

        if self.backoff:
            _log_with_backlog_backoff(timed, self.stop_event)
        else:
            timed()

    def emit(self, frame_index: int, timestamp: float | None) -> None:
        """Log one frame's payload plus this stream's marker."""
        if self.plan.is_rgb:
            self._emit_rgb(frame_index, timestamp)
        elif self.plan.is_depth:
            self._emit_depth(frame_index, timestamp)
        else:
            self._emit_joints(frame_index, timestamp)
        self._emit_marker(frame_index, timestamp)

    def _emit_rgb(self, frame_index: int, timestamp: float | None) -> None:
        for camera_name, camera_index in zip(
            self.plan.channel_names, self.plan.camera_indexes
        ):
            frame_code = rgb_frame_code(
                context_index=self.context_index,
                recording_index=self.recording_index,
                camera_index=camera_index,
                frame_index=frame_index,
            )
            rgb_image = self.feed.render(frame_index, frame_code)
            self._log_video_frame(
                "nc.log_rgb",
                functools.partial(
                    nc.log_rgb,
                    camera_name,
                    rgb_image,
                    robot_name=self.robot_name,
                    timestamp=timestamp,
                ),
            )

    def _emit_depth(self, frame_index: int, timestamp: float | None) -> None:
        for camera_name, camera_index in zip(
            self.plan.channel_names, self.plan.camera_indexes
        ):
            # The same identity the RGB path paints into its pixels, here
            # seeding the depth pattern instead — see ``encode_depth_frame``.
            frame_code = (
                frame_code_base(
                    context_index=self.context_index,
                    recording_ordinal=self.recording_index,
                    camera_index=camera_index,
                )
                + frame_index
            )
            depth_image = encode_depth_frame(
                frame_code,
                self.image_width,
                self.image_height,
                self.plan.depth_mode,
                out=self.depth_buffer,
            )
            self._log_video_frame(
                "nc.log_depth",
                functools.partial(
                    nc.log_depth,
                    camera_name,
                    depth_image,
                    robot_name=self.robot_name,
                    timestamp=timestamp,
                ),
            )

    def _emit_joints(self, frame_index: int, timestamp: float | None) -> None:
        joint_values = generate_joint_values(
            frame_index, self.plan.fps, list(self.plan.channel_names)
        )
        for kind in self.plan.joint_kinds:
            log_fn_name = f"log_{kind}"
            with Timer(
                MAX_TIME_TO_LOG_S,
                label=f"nc.{log_fn_name}",
                assert_deadline=self.assert_deadline,
            ):
                getattr(nc, log_fn_name)(
                    joint_values, robot_name=self.robot_name, timestamp=timestamp
                )

    def _emit_marker(self, frame_index: int, timestamp: float | None) -> None:
        if self.plan.marker_name is None:
            return
        with Timer(
            MAX_TIME_TO_LOG_S,
            label="nc.log_custom_1d",
            assert_deadline=self.assert_deadline,
        ):
            nc.log_custom_1d(
                self.plan.marker_name,
                np.array([float(frame_index)], dtype=np.float32),
                robot_name=self.robot_name,
                timestamp=timestamp,
            )


def _run_stream_threads(
    emitters: list[StreamEmitter],
    worker: Callable[[StreamEmitter], None],
    *,
    error_label: str,
) -> None:
    """Run *worker* on one daemon thread per stream, then join and surface errors."""
    thread_errors: list[BaseException] = []

    def guarded(emitter: StreamEmitter) -> None:
        try:
            worker(emitter)
        except BaseException as exc:  # noqa: BLE001
            thread_errors.append(exc)

    threads = [
        threading.Thread(target=guarded, args=(emitter,), daemon=True)
        for emitter in emitters
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if thread_errors:
        raise RuntimeError(
            f"{error_label} producer failed: {thread_errors[0]}"
        ) from thread_errors[0]


@dataclass(frozen=True, slots=True)
class ProducerRequest:
    """Everything a producer needs to run, whatever that producer's lifetime.

    One shape for all three engines, so a case can swap producers without its
    caller learning anything about which one it got. Two fields carry the whole
    difference between them:

    - *duration_sec* bounds every stream at ``fps * duration_sec`` frames.
      ``None`` instead runs each stream until *stop_event* fires, which is what
      lets a producer outlive the recordings it logs across.
    - *recording_index* and *seed_ordinal* name the recording being logged. A
      producer that outlives every recording has none to name and passes ``0``
      for both: its frame index is session-wide and never resets, so frame
      codes stay unique without one.

    Attributes:
        robot: The connected robot handle, read once per frame for the SDK's
            local logging gate (see :attr:`EmittedFrame.handle`).
        robot_name: Name every ``nc.log_*`` call is made against.
        context_index: Index of the parallel context this producer runs in.
        recording_index: Recording ordinal that namespaces painted frame codes.
        seed_ordinal: Recording ordinal that seeds the random-phase offsets.
        plans: The streams to run, in the order the single-threaded producer
            should break ties between them.
        image_width: Camera frame width, or ``None`` when no camera streams.
        image_height: Camera frame height, or ``None`` when no camera streams.
        video_detail: Whether camera frames carry realistic content or flat fill.
        timestamp_start_s: Capture timestamp the first frame of every stream
            carries.
        random_phase: Whether to offset each timestamp within
            :func:`random_phase_jitter_window`.
        duration_sec: Seconds of frames each stream emits, or ``None`` to run
            until *stop_event*.
        pacing: How hard the streams may drive the SDK. Only the lifetime
            producer may carry a *rate* — every other producer is refused one
            when its case is built. Backlog tolerance every producer reads.
        stop_event: Set to ask every stream to stop at its next frame.
        assert_deadline: Arms the per-call ``Timer`` deadline assertions.
        log_interval_s: Fixed sleep after each frame, imposed by the
            performance suites so latency is measured at a known offered rate.
    """

    robot: object
    robot_name: str
    context_index: int
    recording_index: int
    seed_ordinal: int
    plans: tuple[StreamPlan, ...]
    image_width: int | None
    image_height: int | None
    video_detail: str
    timestamp_start_s: float
    random_phase: bool
    duration_sec: int | None
    pacing: str
    stop_event: threading.Event
    assert_deadline: bool = False  # only set by performance tests
    log_interval_s: float = 0.0  # only set by performance tests

    def frame_budget(self, plan: StreamPlan) -> int | None:
        """Frames *plan* emits, or ``None`` when it runs until stopped."""
        if self.duration_sec is None:
            return None
        return plan.fps * self.duration_sec


def stream_frame_schedule(
    plan: StreamPlan, request: ProducerRequest
) -> Iterator[tuple[int, float]]:
    """Yield ``(frame_index, timestamp)`` for one stream, in emission order.

    The single schedule every producer emits from, so which producer ran can
    never change what a stream was supposed to log. Frames sit on an exact
    ``timestamp_start_s + frame_index / fps`` grid, offset under
    ``random_phase`` by a *seed*-derived amount within
    :func:`random_phase_jitter_window` — the same sequence
    :func:`precompute_timestamps` builds for the same seed, which is what lets
    the on-disk expectation be stated independently of the producer.

    Streams of the same kind draw the same seed on purpose: every joint-kind
    stream walks one joint schedule and every camera stream one video schedule,
    so the joint data types stay aligned with each other and the cameras with
    each other, however many threads are running them.
    """
    stream = VIDEO_STREAM if plan.is_video else JOINT_STREAM
    rng = random.Random(
        stream_phase_seed(request.context_index, request.seed_ordinal, stream)
    )
    window = random_phase_jitter_window(plan.fps)
    budget = request.frame_budget(plan)
    frame_index = 0
    while budget is None or frame_index < budget:
        offset = rng.uniform(-window, window) if request.random_phase else 0.0
        yield frame_index, request.timestamp_start_s + frame_index / plan.fps + offset
        frame_index += 1


class FrameReport:
    """Every frame the producer logged, keyed by the traces it feeds.

    Filled from one thread per stream; read only after those threads join.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_trace: dict[str, list[EmittedFrame]] = {}

    def record(self, emitter: StreamEmitter, frame: EmittedFrame) -> None:
        """Attribute *frame* to every trace *emitter*'s stream writes."""
        with self._lock:
            for trace_key in emitter.trace_keys:
                self._by_trace.setdefault(trace_key, []).append(frame)

    def merge(self, other: dict[str, list[EmittedFrame]]) -> None:
        """Fold another run's frames in, keeping each trace in emission order."""
        with self._lock:
            for trace_key, frames in other.items():
                self._by_trace.setdefault(trace_key, []).extend(frames)

    def as_dict(self) -> dict[str, list[EmittedFrame]]:
        """Return the frames per trace, in emission order."""
        with self._lock:
            return dict(self._by_trace)


def _build_emitters(request: ProducerRequest) -> list[StreamEmitter]:
    """One emitter per stream, in the request's own plan order."""
    return [
        StreamEmitter(
            plan=plan,
            robot_name=request.robot_name,
            context_index=request.context_index,
            recording_index=request.recording_index,
            stop_event=request.stop_event,
            assert_deadline=request.assert_deadline,
            backoff=_should_backoff(request.pacing),
        )
        for plan in request.plans
    ]


def _emit_and_record(
    emitter: StreamEmitter,
    frame_index: int,
    timestamp: float,
    request: ProducerRequest,
    report: FrameReport,
) -> None:
    """Log one frame, recording everything the test can observe about the call."""
    # The gate the SDK checks again inside `emit`, read here so the caller can
    # tell a frame the gate had already refused from one it admitted.
    handle = request.robot.get_current_recording_id()
    # Brackets the publish stamp the daemon routes on, which is taken inside
    # the emit and is not observable from here.
    emitted_at = time.time()
    emitter.emit(frame_index, timestamp)
    report.record(
        emitter,
        EmittedFrame(
            timestamp=timestamp,
            frame_index=frame_index,
            emitted_at=emitted_at,
            completed_at=time.time(),
            handle=handle,
        ),
    )


def run_synchronous_logging(request: ProducerRequest) -> dict[str, list[EmittedFrame]]:
    """Log every stream from one thread, in nominal schedule order.

    Scoped to one recording: the request's frame budget bounds it and it
    returns before that recording stops. Depth cameras share the RGB video
    schedule, so they are logged alongside the RGB cameras at each video
    timestamp. One thread means no stream can outrun another — the streams
    interleave in the order their frames come due.
    """
    emitters = _build_emitters(request)
    for emitter in emitters:
        emitter.prepare(request.image_width, request.image_height, request.video_detail)

    report = FrameReport()
    schedules = {
        emitter: stream_frame_schedule(emitter.plan, request) for emitter in emitters
    }
    pending = {
        emitter: frame
        for emitter, schedule in schedules.items()
        if (frame := next(schedule, None)) is not None
    }

    while pending and not request.stop_event.is_set():
        due = min(pending, key=lambda emitter: pending[emitter][0] / emitter.plan.fps)
        frame_index, timestamp = pending[due]
        _emit_and_record(due, frame_index, timestamp, request, report)
        next_frame = next(schedules[due], None)
        if next_frame is None:
            del pending[due]
        else:
            pending[due] = next_frame
        if request.log_interval_s:
            time.sleep(request.log_interval_s)

    return report.as_dict()


def run_old_per_thread_logging(
    request: ProducerRequest,
) -> dict[str, list[EmittedFrame]]:
    """Run one thread per stream, each bursting through its own frame budget.

    The per-thread engine that predates producers running for the whole context
    lifetime, kept because it is the only one whose streams race each other
    *inside* a single recording and still stop cleanly at its edges: every
    thread is joined before the recording stops, so no frame is ever in flight
    when a window boundary passes.

    Every stream logs its frames back-to-back with no wall-clock wait, so this
    engine offers load as fast as the machine can generate it.
    """
    emitters = _build_emitters(request)
    report = FrameReport()
    barrier = threading.Barrier(len(emitters))

    def worker(emitter: StreamEmitter) -> None:
        barrier.wait()
        emitter.prepare(request.image_width, request.image_height, request.video_detail)
        for frame_index, timestamp in stream_frame_schedule(emitter.plan, request):
            if request.stop_event.is_set():
                break
            _emit_and_record(emitter, frame_index, timestamp, request, report)
            if request.log_interval_s:
                time.sleep(request.log_interval_s)

    _run_stream_threads(emitters, worker, error_label="Old per-thread")

    return report.as_dict()


def run_per_thread_logging(request: ProducerRequest) -> dict[str, list[EmittedFrame]]:
    """Run one thread per stream, for as long as the caller keeps them running.

    Mirrors real deployments where camera and proprioception loops run for the
    process lifetime: the threads start before the first ``nc.start_recording``
    and keep logging — on a session-wide, ever-increasing frame index — until
    the request's *stop_event* is set, regardless of how many recordings start
    and stop while they run.

    With no frame count to bound it, what each stream waits for is the request's
    ``pacing`` decision: its wall-clock deadline, as the camera it stands in for
    would, or nothing — leaving the daemon's spool cap to hold it back.

    Every frame is reported, whichever recording was current and even when none
    was: which of them belong to a recording is decided afterwards, from the
    wall-clock brackets around that recording's control calls (see
    :func:`_classify_boundary_frames`). Reporting only the frames logged during
    a recording would hide the ones the daemon must be shown to have *rejected*.

    Returns:
        Mapping of trace key -> every frame logged for it, in order.
    """
    emitters = _build_emitters(request)
    report = FrameReport()
    barrier = threading.Barrier(len(emitters))

    def worker(emitter: StreamEmitter) -> None:
        barrier.wait()
        emitter.prepare(request.image_width, request.image_height, request.video_detail)
        pace = _should_pace(request.pacing, emitter.plan.is_video)
        thread_wall_start = time.time()
        for frame_index, timestamp in stream_frame_schedule(emitter.plan, request):
            if request.stop_event.is_set():
                break
            frame_deadline = thread_wall_start + (frame_index / emitter.plan.fps)
            if pace and _await_frame_deadline(frame_deadline, request.stop_event):
                break
            _emit_and_record(emitter, frame_index, timestamp, request, report)
            if request.log_interval_s:
                time.sleep(request.log_interval_s)

    _run_stream_threads(emitters, worker, error_label="Per-thread")

    return report.as_dict()


class ProducerSession(ABC):
    """Runs a case's producer, whatever its lifetime, behind one interface.

    Lifetime is the only thing that differs at the call site: a producer scoped
    to a recording runs *inside* the window, while one that outlives every
    recording has to be started before the first and stopped after the last.
    Both are driven the same way here, so a caller states the recording
    protocol once and the case's ``producer_channels`` decides nothing about
    the shape of that code::

        session = make_producer_session(spec, robot=robot, marker_name=...)
        session.start()
        try:
            for ordinal in range(recordings):
                nc.start_recording(...)
                session.run_recording(ordinal)
                nc.stop_recording(...)
        finally:
            session.finish()
        report = session.report()

    The report is uniform too: every producer reports every frame it logged
    with the wall-clock brackets around the call, so one classification decides
    which recording owns which frame (see :func:`_classify_boundary_frames`).
    """

    def __init__(
        self, spec: ContextSpec, robot: object, plans: list[StreamPlan]
    ) -> None:
        self.spec = spec
        self.robot = robot
        self.plans = tuple(plans)
        self.stop_event = threading.Event()

    @property
    def marker_names(self) -> list[str]:
        """The ``CUSTOM_1D`` marker series this producer's streams write."""
        return [plan.marker_name for plan in self.plans if plan.marker_name is not None]

    def _request(
        self, *, recording_ordinal: int, duration_sec: int | None
    ) -> ProducerRequest:
        """Build the request for one run of this session's engine."""
        case = self.spec.case
        return ProducerRequest(
            robot=self.robot,
            robot_name=self.spec.robot_name,
            context_index=self.spec.context_index,
            recording_index=recording_ordinal,
            seed_ordinal=recording_ordinal,
            plans=self.plans,
            image_width=case.image_width,
            image_height=case.image_height,
            video_detail=case.video_detail,
            timestamp_start_s=(
                self.spec.timestamp_start_s + recording_ordinal * case.duration_sec
            ),
            random_phase=case.random_phase,
            duration_sec=duration_sec,
            pacing=case.producer_pacing,
            stop_event=self.stop_event,
            assert_deadline=self.spec.assert_deadline,
            log_interval_s=self.spec.log_interval_s,
        )

    @abstractmethod
    def frame_code_recording_index(self, recording_ordinal: int) -> int:
        """Recording ordinal *recording_ordinal*'s frames were painted under."""

    @abstractmethod
    def start(self) -> None:
        """Begin producing, if this producer starts before any recording."""

    @abstractmethod
    def run_recording(self, recording_ordinal: int) -> None:
        """Produce this recording's frames, and return once it may be stopped."""

    @abstractmethod
    def finish(self) -> None:
        """Stop producing and surface anything a producer thread raised."""

    @abstractmethod
    def report(self) -> dict[str, list[EmittedFrame]]:
        """Return every frame logged so far, keyed by trace."""


class BoundedProducerSession(ProducerSession):
    """A producer that lives and dies inside a single recording.

    Nothing runs before ``start_recording`` or after the engine returns, and
    the engine returns only once every thread it started has been joined — so
    the recording that was open at the time owns every frame reported, and no
    frame is ever in flight while a window boundary passes.
    """

    def __init__(
        self,
        spec: ContextSpec,
        robot: object,
        plans: list[StreamPlan],
        engine: Callable[[ProducerRequest], dict[str, list[EmittedFrame]]],
    ) -> None:
        super().__init__(spec, robot, plans)
        self._engine = engine
        self._report = FrameReport()
        self._runs: list[tuple[int, dict[str, list[EmittedFrame]]]] = []

    def frame_code_recording_index(self, recording_ordinal: int) -> int:
        """Its own ordinal: this producer's frame index restarts every recording."""
        return recording_ordinal

    def start(self) -> None:
        """Nothing to start: this producer only runs inside a recording."""

    def run_recording(self, recording_ordinal: int) -> None:
        """Log this recording's whole frame budget, then join every thread."""
        frames = self._engine(
            self._request(
                recording_ordinal=recording_ordinal,
                duration_sec=self.spec.case.duration_sec,
            )
        )
        self._runs.append((recording_ordinal, frames))
        self._report.merge(frames)

    def finish(self) -> None:
        """Check each run against the schedule it was supposed to emit.

        A bounded producer's schedule can be stated without running it, so it
        is — by :func:`recording_timestamps`, which shares nothing with the
        generator the producer emitted from. Checking here rather than in
        :meth:`run_recording` keeps the comparison out of the recording window
        it would otherwise hold open.
        """
        for recording_ordinal, frames in self._runs:
            joint_timestamps, video_timestamps = recording_timestamps(
                self.spec, recording_ordinal
            )
            for plan in self.plans:
                expected = video_timestamps if plan.is_video else joint_timestamps
                for trace_key in plan.trace_keys:
                    logged = [frame.timestamp for frame in frames.get(trace_key, ())]
                    assert logged == expected, (
                        f"{self.spec.case.producer_channels} producer logged "
                        f"{len(logged)} frames for {trace_key} in recording "
                        f"{recording_ordinal}, expected {len(expected)}"
                    )

    def report(self) -> dict[str, list[EmittedFrame]]:
        """Return every frame logged, across all this context's recordings."""
        return self._report.as_dict()


class LifetimeProducerSession(ProducerSession):
    """A producer that outlives every recording it logs across.

    Its threads are started before the first ``start_recording`` and stopped
    after the last ``stop_recording``, so they are mid-loop at every boundary —
    the whole point of it. The recordings simply open and close around a
    producer that never learns they exist.
    """

    def __init__(
        self, spec: ContextSpec, robot: object, plans: list[StreamPlan]
    ) -> None:
        super().__init__(spec, robot, plans)
        self._thread: threading.Thread | None = None
        self._frames: dict[str, list[EmittedFrame]] = {}
        self._error: BaseException | None = None

    def frame_code_recording_index(self, recording_ordinal: int) -> int:
        """Always ``0``: the frame index is session-wide and never resets, so
        codes stay unique on their own (see :func:`run_per_thread_logging`)."""
        return 0

    def start(self) -> None:
        """Start logging now, before any recording exists."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._frames = run_per_thread_logging(
                # No recording is current when these threads start, and none is
                # named while they run: the session-wide frame index keeps the
                # frame codes unique on its own.
                self._request(recording_ordinal=0, duration_sec=None)
            )
        except BaseException as exc:  # noqa: BLE001
            self._error = exc

    def run_recording(self, recording_ordinal: int) -> None:
        """Hold the window open: the producer is already running its own schedule."""
        time.sleep(self.spec.case.duration_sec)

    def finish(self) -> None:
        """Log past the last recording, then stop the threads and join them."""
        if self._thread is None:
            return
        try:
            # Keep logging after the last stop_recording exactly as a real
            # camera would, so the daemon is shown rejecting post-stop frames.
            time.sleep(PER_THREAD_LOGGING_TAIL_S)
        finally:
            self.stop_event.set()
            self._thread.join()
            self._thread = None
        if self._error is not None:
            raise RuntimeError(f"Per-thread producer failed: {self._error}") from (
                self._error
            )

    def report(self) -> dict[str, list[EmittedFrame]]:
        """Return every frame logged over the whole context lifetime."""
        return self._frames


def make_producer_session(
    spec: ContextSpec, *, robot: object, marker_name: str
) -> ProducerSession:
    """Build the producer session a case asked for.

    The one entry point to producing data: which engine runs, how many threads
    it uses and how long it lives are all decided here from
    ``producer_channels``, and nowhere else.

    *marker_name* names the single marker series the synchronous producer
    writes; the per-thread producers give each of their streams its own marker
    and ignore it.
    """
    case = spec.case
    plans = stream_plans_for_case(
        producer_channels=case.producer_channels,
        joint_names=joint_names_for_count(case.joint_count),
        camera_name_list=camera_names(case.video_count),
        depth_camera_name_list=depth_camera_names(case.depth_count),
        depth_mode=case.depth_mode,
        joint_fps=case.joint_fps,
        video_fps=case.video_fps,
        marker_name=marker_name,
    )
    if case.producer_channels == PRODUCER_PER_THREAD:
        return LifetimeProducerSession(spec, robot, plans)
    engine = (
        run_old_per_thread_logging
        if case.producer_channels == PRODUCER_OLD_PER_THREAD
        else run_synchronous_logging
    )
    return BoundedProducerSession(spec, robot, plans, engine)


def _classify_boundary_frames(
    frames: list[EmittedFrame],
    bounds: RecordingControlBounds,
) -> TraceClassification:
    """Split one trace's frames into those inside a recording and those unknowable.

    Neither boundary is directly observable, so each is bracketed between two
    clock readings: a frame is **inside** only if its whole ``log_*`` call ran
    between ``start_recording`` returning and ``stop_recording`` being called.
    It is **outside** if the call finished before ``start_recording`` was
    entered, or if the SDK's gate — a deliberate superset of the window — had
    already refused it after the stop; a gate *admission* proves nothing, since
    the gate opens first, so only a refusal is conclusive. Everything else
    straddled a bracket and is **unknowable**: the daemon's answer is correct
    either way, so those frames count on neither side.

    Returns:
        The recording's verdict on these frames, as whole frames rather than
        bare timestamps, since the cloud assertion also needs frame indexes.
    """
    inside: list[EmittedFrame] = []
    unknowable: list[EmittedFrame] = []
    for frame in frames:
        is_inside = (
            frame.emitted_at >= bounds.start_returned_at
            and frame.completed_at <= bounds.stop_called_at
        )
        is_outside = frame.completed_at <= bounds.start_called_at or (
            frame.emitted_at >= bounds.stop_called_at and frame.handle != bounds.handle
        )
        if is_inside:
            inside.append(frame)
        elif not is_outside:
            unknowable.append(frame)
    return TraceClassification(owed=inside, unknowable=unknowable)


def recording_timestamps(
    spec: ContextSpec, recording_index: int
) -> tuple[list[float], list[float]]:
    """Return the ``(joint, video)`` timestamp sequences for one recording.

    Shares nothing with the schedule producers emit from, so a bounded producer
    is checked against what it was supposed to log rather than against itself.
    """
    case = spec.case
    start_s = spec.timestamp_start_s + recording_index * case.duration_sec
    return (
        precompute_timestamps(
            start_s,
            spec.expected_joint_frames,
            case.joint_fps,
            random_phase=case.random_phase,
            seed=stream_phase_seed(spec.context_index, recording_index, JOINT_STREAM),
        ),
        precompute_timestamps(
            start_s,
            spec.expected_video_frames,
            case.video_fps,
            random_phase=case.random_phase,
            seed=stream_phase_seed(spec.context_index, recording_index, VIDEO_STREAM),
        ),
    )


def log_frames(
    spec: ContextSpec,
    *,
    robot: object,
    recording_index: int,
    marker_name: str,
) -> list[str]:
    """Log one recording's frames, for a caller driving that recording by hand.

    The whole session protocol for the single-recording case: start, run the
    one recording, and check what was emitted against
    :func:`recording_timestamps`. A case whose producer outlives its recordings
    has no single recording's worth of frames to log, so it is refused rather
    than quietly logging one from the wrong engine.

    Returns:
        The marker series the producer wrote.
    """
    if spec.case.producer_channels == PRODUCER_PER_THREAD:
        raise ValueError(
            f"log_frames needs a producer scoped to one recording, but "
            f"producer_channels={PRODUCER_PER_THREAD!r} runs for the whole "
            "context lifetime — drive it through make_producer_session instead"
        )
    session = make_producer_session(spec, robot=robot, marker_name=marker_name)
    session.start()
    session.run_recording(recording_index)
    session.finish()
    return session.marker_names


def _bind_worker_dataset(spec: ContextSpec) -> None:
    """Poll until the worker pool-shared dataset is visible to this worker.

    The timeout RuntimeError is raised inside the Timer block on purpose:
    ``Timer.__exit__`` skips its deadline assertion when an exception is in
    flight, so the real ``nc.get_dataset`` error (chained via ``last_error``)
    propagates instead of being masked by the Timer's own AssertionError.
    """
    last_error: Exception | None = None
    deadline = time.time() + MAX_TIME_TO_START_S
    with Timer(
        MAX_TIME_TO_START_S,
        label="nc.get_dataset",
        always_log=True,
        assert_deadline=spec.assert_deadline,
    ):
        while time.time() < deadline:
            try:
                nc.get_dataset(spec.dataset_name)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(DATASET_POLL_INTERVAL_S)

        raise RuntimeError(
            f"Timed out waiting for shared dataset '{spec.dataset_name}' to exist"
        ) from last_error


@surface_worker_errors
def _subprocess_context_worker(spec: ContextSpec) -> ContextResult:
    """Subprocess wrapper for context_worker used by multiprocessing.Pool.

    On Linux, Pool uses fork so workers inherit a copy of the parent's
    Timer._stats. Clearing it here ensures workers only capture their own
    timers and the parent's pre-fork timers (e.g. nc.login) are not
    double-counted when stats are merged back. Spawned workers (macOS)
    additionally re-authenticate, as they do not inherit the parent's
    in-process auth state.
    """
    multiprocessing.current_process().name = f"ctx-{spec.context_index}"
    Timer._stats.clear()
    ensure_login()
    return context_worker(spec)


def context_worker(spec: ContextSpec) -> ContextResult:
    """Execute recordings for a single parallel context."""
    from tests.integration.platform.data_daemon.shared.db_helpers import (
        wait_for_recording_index_for_source,
    )

    case = spec.case
    joint_name_list = joint_names_for_count(case.joint_count)
    camera_name_list = camera_names(case.video_count)
    depth_camera_name_list = depth_camera_names(case.depth_count)
    if camera_name_list:
        prewarm_frame_bank(case.video_detail, case.image_width, case.image_height)
    marker_names: list[str] = []
    recording_ids: list[str] = []
    recording_indexes: list[int] = []
    robot = None

    wall_started_at: float | None = None
    wall_stopped_at: float = 0.0

    try:
        _bind_worker_dataset(spec)
        with Timer(
            MAX_TIME_TO_START_S,
            label="nc.connect_robot",
            always_log=True,
            assert_deadline=spec.assert_deadline,
        ):
            robot = nc.connect_robot(spec.robot_name, overwrite=False)

        source: tuple[str, int] = (str(robot.id), int(robot.instance))

        expected_by_recording: dict[str, RecordingExpectedTimestamps] = {}
        expected_video_stop_timestamp_by_recording: dict[str, float] = {}
        bounds_by_disk_key: dict[str, RecordingControlBounds] = {}
        observed_frame_codes: dict[str, ObservedFrameCodes] = {}
        ordinal_by_disk_key: dict[str, int] = {}

        # One protocol for every producer: which engine runs, how many threads
        # it uses and whether it outlives these recordings are all the
        # session's business, not this loop's.
        session = make_producer_session(
            spec, robot=robot, marker_name="marker_synchronous"
        )
        marker_names = session.marker_names
        session.start()
        try:
            for recording_ordinal in range(spec.recordings_per_context):
                recording_capture_start_s = time.time()
                recording_capture_stop_s = recording_capture_start_s + case.duration_sec

                # Brackets the window's lower bound, which the daemon stamps
                # somewhere inside the call (see `RecordingControlBounds`).
                start_called_at = time.time()
                with Timer(
                    MAX_TIME_TO_START_S,
                    label="nc.start_recording",
                    always_log=True,
                    assert_deadline=spec.assert_deadline,
                ):
                    nc.start_recording(
                        robot_name=spec.robot_name, timestamp=recording_capture_start_s
                    )
                start_returned_at = time.time()
                if wall_started_at is None:
                    wall_started_at = start_returned_at

                previous_index = recording_indexes[-1] if recording_indexes else 0
                daemon_recording_index = wait_for_recording_index_for_source(
                    source[0],
                    source[1],
                    after_index=previous_index,
                    timeout_s=MAX_TIME_TO_START_S,
                )
                recording_indexes.append(daemon_recording_index)

                cloud_recording_id = robot.get_cloud_recording_id(timeout_s=0.0)
                recording_ids.append(str(cloud_recording_id or ""))

                disk_recording_key = str(daemon_recording_index)
                recording_handle = robot.get_current_recording_id()

                session.run_recording(recording_ordinal)

                # Brackets the window's upper bound, the mirror of the start.
                stop_called_at = time.time()
                with Timer(
                    case.stop_recording_sla_s,
                    label="nc.stop_recording",
                    always_log=True,
                    assert_deadline=spec.assert_deadline,
                ):
                    nc.stop_recording(
                        robot_name=spec.robot_name,
                        wait=case.wait,
                        timestamp=recording_capture_stop_s,
                    )
                wall_stopped_at = time.time()
                expected_video_stop_timestamp_by_recording[disk_recording_key] = (
                    spec.timestamp_start_s + (recording_ordinal + 1) * case.duration_sec
                )

                bounds_by_disk_key[disk_recording_key] = RecordingControlBounds(
                    handle=recording_handle,
                    start_called_at=start_called_at,
                    start_returned_at=start_returned_at,
                    stop_called_at=stop_called_at,
                    stop_returned_at=wall_stopped_at,
                )
                ordinal_by_disk_key[disk_recording_key] = recording_ordinal
        finally:
            # Always stop the producer — even when the loop above raised — so
            # no producer thread outlives this worker and confuses the
            # daemon-cleanup assertions that run next.
            session.finish()

        from neuracore_types.utils import validate_safe_name

        # Turns an RGB trace's classified frames back into painted codes.
        rgb_trace_cameras = {
            f"RGB_IMAGES/{validate_safe_name(camera)}": (camera, camera_index)
            for camera_index, camera in enumerate(camera_name_list)
        }
        # The producer reported every frame it logged; each recording claims
        # its own from the wall-clock brackets around its control calls. A
        # producer scoped to one recording logged all of its frames well inside
        # those brackets, so the same rule simply finds them all.
        report = session.report()
        for disk_key, bounds in bounds_by_disk_key.items():
            code_recording_index = session.frame_code_recording_index(
                ordinal_by_disk_key[disk_key]
            )
            by_trace: dict[str, TraceClassification] = {}
            codes_inside: dict[str, list[int]] = {}
            codes_unknowable: dict[str, set[int]] = {}
            for trace_key, frames in report.items():
                classification = _classify_boundary_frames(frames, bounds)
                by_trace[trace_key] = classification

                camera = rgb_trace_cameras.get(trace_key)
                if camera is None:
                    continue
                camera_name, camera_index = camera

                def _codes(
                    frames: list[EmittedFrame],
                    index: int = camera_index,
                    recording_index: int = code_recording_index,
                ):
                    return [
                        rgb_frame_code(
                            context_index=spec.context_index,
                            recording_index=recording_index,
                            camera_index=index,
                            frame_index=frame.frame_index,
                        )
                        for frame in frames
                    ]

                codes_inside[camera_name] = _codes(classification.owed)
                codes_unknowable[camera_name] = set(_codes(classification.unknowable))

            expected_by_recording[disk_key] = RecordingExpectedTimestamps(
                by_trace=by_trace
            )
            observed_frame_codes[disk_key] = ObservedFrameCodes(
                inside=codes_inside, unknowable=codes_unknowable
            )

        captured_timer_stats = {k: dict(v) for k, v in Timer._stats.items()}
        return ContextResult(
            dataset_name=spec.dataset_name,
            recording_ids=recording_ids,
            recording_indexes=recording_indexes,
            source=source,
            robot_name=spec.robot_name,
            joint_names=joint_name_list,
            camera_names=camera_name_list,
            joint_frame_count=spec.expected_joint_frames,
            video_frame_count=spec.expected_video_frames,
            joint_fps=case.joint_fps,
            video_fps=case.video_fps,
            duration_sec=case.duration_sec,
            timestamp_start_s=spec.timestamp_start_s,
            timestamp_end_s=spec.timestamp_end_s,
            marker_names=marker_names,
            has_video=bool(camera_name_list),
            context_index=spec.context_index,
            wall_started_at=wall_started_at,
            wall_stopped_at=wall_stopped_at,
            random_phase=case.random_phase,
            expected_timestamps=ContextExpectedTimestamps(
                by_recording=expected_by_recording
            ),
            timer_stats=captured_timer_stats,
            depth_camera_names=depth_camera_name_list,
            depth_frame_count=(
                spec.expected_video_frames if depth_camera_name_list else 0
            ),
            depth_mode=case.depth_mode,
            has_depth=bool(depth_camera_name_list),
            observed_frame_codes=observed_frame_codes,
            expected_video_stop_timestamp_by_recording=(
                expected_video_stop_timestamp_by_recording
            ),
        )
    except Exception:
        if robot is not None:
            try:
                if robot.is_recording():
                    nc.cancel_recording(robot_name=spec.robot_name)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to cancel active matrix recording for %s",
                    spec.robot_name,
                    exc_info=True,
                )
        raise
    finally:
        _cleanup_test_worker_robot(robot)


def run_case_contexts(
    case: DataDaemonTestCase,
    *,
    specs: list[ContextSpec] | None = None,
    wait_for_traces: bool = False,
) -> list[ContextResult]:
    """Run all parallel contexts for a matrix test case.

    Executes each context spec in-process (when parallel_contexts==1) or
    concurrently via a multiprocessing pool. The in-process path avoids pool
    overhead and simplifies debugging for single-context cases.

    Args:
        case: The test case defining parallelism level and context matrix.
        specs: Pre-built context specs to run. If None, built from ``case``
            via :func:`build_context_specs`.
        wait_for_traces: When ``True``, waits for all traces to be written to
            disk after running.

    Returns:
        List of result dicts from each context worker, one per spec.
    """
    ensure_login()

    if specs is None:
        specs = build_context_specs(case)

    if case.has_video:
        nc.set_video_encoding_options(
            nc.Codec(case.video_codec) if case.video_codec else nc.Codec.H264_LOSSLESS
        )

    if specs:
        with Timer(MAX_TIME_TO_START_S, label="nc.create_dataset", always_log=True):
            nc.create_dataset(specs[0].dataset_name)

    if not wait_for_traces:
        return _run_context_specs(case, specs)

    from tests.integration.platform.data_daemon.shared.db_helpers import (
        latching_trace_write_observer,
        wait_for_all_traces_written,
    )

    # The observer has to outlive the run as well as cover it: a recording that
    # finishes writing late is reaped during the wait below, not before it.
    with latching_trace_write_observer() as observed:
        results = _run_context_specs(case, specs)
        wait_for_all_traces_written(results=results, observed=observed)
    return results


def _run_context_specs(
    case: DataDaemonTestCase,
    specs: list[ContextSpec],
) -> list[ContextResult]:
    """Run *specs* in-process or across a pool, per the case's parallelism."""
    if case.parallel_contexts == 1:
        return [context_worker(specs[0])]

    with relayed_worker_logs() as log_queue:
        with multiprocessing.Pool(
            case.parallel_contexts,
            initializer=init_worker_logging,
            initargs=(log_queue, logging.getLogger().getEffectiveLevel()),
        ) as pool:
            results: list[ContextResult] = list(  # type: ignore[return-value]
                pool.map(_subprocess_context_worker, specs)
            )
    for result in results:
        Timer.merge_stats(result.timer_stats)
    return results


def create_testing_dataset_name(case: DataDaemonTestCase) -> str:
    """Create a unique dataset name for a test case."""
    return f"testing_dataset_{case_id(case)}_{uuid.uuid4().hex[:6]}"
