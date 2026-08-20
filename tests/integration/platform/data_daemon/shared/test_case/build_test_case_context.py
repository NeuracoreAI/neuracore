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
    DATASET_POLL_INTERVAL_S,
    DURATION_MODE_VARIABLE,
    DURATION_VARIABLE_MAX_FACTOR,
    DURATION_VARIABLE_MIN_FACTOR,
    JOINT_KINDS,
    MAX_TIME_TO_START_S,
    MODE_STAGGERED,
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
from tests.integration.platform.data_daemon.shared.test_case.pacing import (
    StreamPacer,
    pacer_for,
)

logger = logging.getLogger(__name__)

CONTEXT_DURATION_RANDOM = random.Random(0)

# Stream discriminators feeding ``stream_phase_seed`` so a recording's joint and
# video streams draw independent phase offsets.
JOINT_STREAM = 0
VIDEO_STREAM = 1

# Semantic trace data type each ``nc.log_joint_*`` call writes to. Derived from
# JOINT_KINDS so the two cannot drift apart.
_JOINT_DATA_TYPES = {kind: kind.upper() for kind in JOINT_KINDS}


class EmittedFrame(NamedTuple):
    """One frame a producer logged, bracketed on the wall clock.

    The daemon routes a sample by a publish stamp taken *inside* the ``log_*``
    call, so the test cannot observe that stamp — only that it lies somewhere in
    ``[emitted_at, completed_at]``. Both edges are kept because the two
    boundaries need opposite ones (see :func:`_classify_boundary_frames`).

    Every producer reports these, whatever its lifetime: a producer scoped to
    one recording brackets its frames well inside that recording's own control
    calls, so the same classification that a lifetime producer needs simply
    finds all of them inside.

    Attributes:
        timestamp: The frame's own capture timestamp — the value that reaches
            disk, and the only field the assertion compares.
        emitted_at: Wall clock immediately before the ``log_*`` call.
        completed_at: Wall clock immediately after it returned.
        handle: The SDK recording handle latched immediately before the call,
            or ``None`` when no recording was active. This is the local logging
            gate, not the daemon's window.
        deadline_breaches: ``nc.log_*`` calls this frame made that exceeded
            ``MAX_TIME_TO_LOG_S``. Only asserted on when this frame is inside.
    """

    timestamp: float
    emitted_at: float
    completed_at: float
    handle: str | None
    deadline_breaches: tuple[str, ...] = ()


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
            ``"camera_0"``) to the ordered list of expected timestamps for
            that trace within this recording.
        by_trace_unknowable: Maps the same semantic trace key to timestamps
            whose membership of this recording the test cannot determine —
            frames whose ``log_*`` call straddled one of the two control calls
            that carry the window's bounds (see
            :func:`_classify_boundary_frames`). The assertion drops them from
            both the expected and the on-disk list rather than tolerating them
            on one. Empty for producers that log strictly within one recording.
    """

    by_trace: dict[str, list[float]]
    by_trace_unknowable: dict[str, set[float]] = field(default_factory=dict)


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


def build_context_specs(
    case: DataDaemonTestCase,
    dataset_name: str | None = None,
    assert_deadline: bool = False,
) -> list[ContextSpec]:
    """Build per-context worker specs for a matrix case.

    ``assert_deadline`` arms the ``Timer`` deadline assertions; what rate the
    producer offers frames at is the case's own ``producer_pacing``.
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


@dataclass(frozen=True, slots=True)
class StreamPlan:
    """One producer stream: which channels it logs, and under which marker.

    Producers consume it through :class:`StreamEmitter` rather than re-deriving
    channel names, log functions or frame codes for themselves.
    """

    name: str
    fps: int
    channel_names: tuple[str, ...]
    marker_name: str | None = None
    # Per-thread streams hold one kind; the synchronous producer bundles all.
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

        Both camera kinds share the video timestamp schedule and are paced
        together, so producers branch on this rather than :attr:`is_rgb`.
        """
        return self.is_rgb or self.is_depth

    @property
    def trace_keys(self) -> list[str]:
        """Semantic trace keys one logged frame from this stream touches.

        Matches the ``data_type/data_type_name`` keys disk_helpers resolves from
        the DB, plus this stream's own ``CUSTOM_1D`` marker when it has one.
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
        *(
            StreamPlan(
                name="rgb",
                marker_name=f"marker_{camera_name}",
                fps=video_fps,
                channel_names=(camera_name,),
                camera_indexes=(camera_index,),
            )
            for camera_index, camera_name in enumerate(camera_name_list)
        ),
        *(
            StreamPlan(
                name="depth",
                marker_name=f"marker_{depth_camera_name}",
                fps=video_fps,
                channel_names=(depth_camera_name,),
                camera_indexes=(depth_camera_index,),
                depth_mode=depth_mode,
            )
            for depth_camera_index, depth_camera_name in enumerate(
                depth_camera_name_list
            )
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

    One stream per kind, not per camera, since a single thread logs every
    camera of a kind together; only the joint stream carries a marker.
    """
    joint_plan = StreamPlan(
        name="joints",
        marker_name=marker_name,
        fps=joint_fps,
        channel_names=tuple(joint_names),
        joint_kinds=JOINT_KINDS,
    )
    video_plan = (
        StreamPlan(
            name="rgb",
            fps=video_fps,
            channel_names=tuple(camera_name_list),
            camera_indexes=tuple(range(len(camera_name_list))),
        )
        if camera_name_list
        else None
    )
    depth_plan = (
        StreamPlan(
            name="depth",
            fps=video_fps,
            channel_names=tuple(depth_camera_name_list),
            camera_indexes=tuple(range(len(depth_camera_name_list))),
            depth_mode=depth_mode,
        )
        if depth_camera_name_list
        else None
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

    Owns every per-frame side effect, so producers cannot drift in what they
    log, how frames are identified, or how calls are labelled for reporting.
    """

    plan: StreamPlan
    robot_name: str
    context_index: int
    recording_index: int
    assert_deadline: bool = False
    feed: object | None = None
    depth_buffer: np.ndarray | None = None
    image_width: int | None = None
    image_height: int | None = None
    # Resolved once, not per frame: building them walks validate_safe_name.
    trace_keys: tuple[str, ...] = ()
    # Deadline breaches for the frame currently being emitted; drained by emit().
    _deadline_breaches: list[str] = field(default_factory=list)

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

    def _record_breach_if_slow(self, label: str, timer: Timer) -> None:
        """Record a deadline breach; the boundary classifier judges it later."""
        if self.assert_deadline and timer.interval >= MAX_TIME_TO_LOG_S:
            self._deadline_breaches.append(
                f"{label} took {timer.interval:.3f}s >= {MAX_TIME_TO_LOG_S:.3f}s"
            )

    def _timed_log(self, label: str, call: Callable[[], None]) -> None:
        """Make one ``nc.log_*`` call, timed under *label*."""
        with Timer(
            MAX_TIME_TO_LOG_S, label=label, assert_deadline=False, log_breaches=False
        ) as timer:
            call()
        self._record_breach_if_slow(label, timer)

    def emit(self, frame_index: int, timestamp: float | None) -> tuple[str, ...]:
        """Log one frame's payload plus this stream's marker.

        Returns:
            Labels of any ``nc.log_*`` calls that breached the deadline.
        """
        self._deadline_breaches = []
        if self.plan.is_rgb:
            self._emit_rgb(frame_index, timestamp)
        elif self.plan.is_depth:
            self._emit_depth(frame_index, timestamp)
        else:
            self._emit_joints(frame_index, timestamp)
        self._emit_marker(frame_index, timestamp)
        return tuple(self._deadline_breaches)

    def _emit_rgb(self, frame_index: int, timestamp: float | None) -> None:
        for camera_name, camera_index in zip(
            self.plan.channel_names, self.plan.camera_indexes
        ):
            # Frame codes must stay unique across contexts, recordings and
            # cameras: they are read back to identify frames.
            frame_code = (
                frame_code_base(
                    context_index=self.context_index,
                    recording_ordinal=self.recording_index,
                    camera_index=camera_index,
                )
                + frame_index
            )
            rgb_image = self.feed.render(frame_index, frame_code)
            self._timed_log(
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
            # Same identity the RGB path paints; here it seeds the depth pattern.
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
            self._timed_log(
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
            self._timed_log(
                f"nc.{log_fn_name}",
                functools.partial(
                    getattr(nc, log_fn_name),
                    joint_values,
                    robot_name=self.robot_name,
                    timestamp=timestamp,
                ),
            )

    def _emit_marker(self, frame_index: int, timestamp: float | None) -> None:
        if self.plan.marker_name is None:
            return
        self._timed_log(
            "nc.log_custom_1d",
            functools.partial(
                nc.log_custom_1d,
                self.plan.marker_name,
                np.array([float(frame_index)], dtype=np.float32),
                robot_name=self.robot_name,
                timestamp=timestamp,
            ),
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

    One shape for both engines, so a case can swap producers without its
    caller learning which one it got. Two fields carry the whole difference:
    *duration_sec* bounds each stream at ``fps * duration_sec`` frames, or runs
    it until *stop_event* fires; *recording_index* and *seed_ordinal* name the
    recording, and are ``0`` for a producer that outlives every recording.

    Attributes:
        robot: Connected robot handle, read once per frame for the SDK's local
            logging gate (see :attr:`EmittedFrame.handle`).
        recording_index: Recording ordinal that namespaces painted frame codes.
        seed_ordinal: Recording ordinal that seeds the random-phase offsets.
        plans: Streams to run, in the order the single-threaded producer breaks
            ties between them.
        timestamp_start_s: Capture timestamp the first frame of every stream
            carries.
        pacing: When each stream may offer its next frame, resolved per stream
            by :func:`pacer_for`. Independent of the producer's lifetime.
        log_interval_s: Fixed sleep after each frame, imposed by the performance
            suites so latency is measured at a known offered rate. Never applies
            to the video streams ``PACING_BURST_VIDEO`` un-paces.
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
    assert_deadline: bool = False

    def frame_budget(self, plan: StreamPlan) -> int | None:
        """Frames *plan* emits, or ``None`` when it runs until stopped."""
        if self.duration_sec is None:
            return None
        return plan.fps * self.duration_sec

    def pacer(self, plan: StreamPlan) -> StreamPacer:
        """The pacer this request's pacing gives *plan*."""
        return pacer_for(self.pacing, fps=plan.fps, is_video=plan.is_video)


def stream_frame_schedule(
    plan: StreamPlan, request: ProducerRequest
) -> Iterator[tuple[int, float]]:
    """Yield ``(frame_index, timestamp)`` for one stream, in emission order.

    Streams of the same kind share a seed, keeping them aligned with each other
    however many threads run them.
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
            assert_deadline=request.assert_deadline,
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
    # Read before the call, to distinguish a refused frame from an admitted one.
    handle = request.robot.get_current_recording_id()
    emitted_at = time.time()
    deadline_breaches = emitter.emit(frame_index, timestamp)
    report.record(
        emitter,
        EmittedFrame(
            timestamp=timestamp,
            emitted_at=emitted_at,
            completed_at=time.time(),
            handle=handle,
            deadline_breaches=deadline_breaches,
        ),
    )


def run_synchronous_logging(request: ProducerRequest) -> dict[str, list[EmittedFrame]]:
    """Log every stream from one thread, whichever stream is due next."""
    emitters = _build_emitters(request)
    for emitter in emitters:
        emitter.prepare(request.image_width, request.image_height, request.video_detail)

    report = FrameReport()
    pacers = {emitter: request.pacer(emitter.plan) for emitter in emitters}
    schedules = {
        emitter: stream_frame_schedule(emitter.plan, request) for emitter in emitters
    }
    pending = {
        emitter: frame
        for emitter, schedule in schedules.items()
        if (frame := next(schedule, None)) is not None
    }

    started_at = time.time()
    while pending and not request.stop_event.is_set():
        due = min(
            pending,
            key=lambda emitter: pacers[emitter].release_offset_s(pending[emitter][0]),
        )
        frame_index, timestamp = pending[due]
        if pacers[due].wait_until_due(started_at, frame_index, request.stop_event):
            break
        _emit_and_record(due, frame_index, timestamp, request, report)
        next_frame = next(schedules[due], None)
        if next_frame is None:
            del pending[due]
        else:
            pending[due] = next_frame

    return report.as_dict()


def run_threaded_logging(request: ProducerRequest) -> dict[str, list[EmittedFrame]]:
    """Run one thread per stream, each on its own pacer.

    Returns:
        Mapping of trace key -> every frame logged for it, in order.
    """
    emitters = _build_emitters(request)
    report = FrameReport()
    barrier = threading.Barrier(len(emitters))

    def worker(emitter: StreamEmitter) -> None:
        barrier.wait()
        emitter.prepare(request.image_width, request.image_height, request.video_detail)
        pacer = request.pacer(emitter.plan)
        started_at = time.time()
        for frame_index, timestamp in stream_frame_schedule(emitter.plan, request):
            if request.stop_event.is_set():
                break
            if pacer.wait_until_due(started_at, frame_index, request.stop_event):
                break
            _emit_and_record(emitter, frame_index, timestamp, request, report)

    _run_stream_threads(emitters, worker, error_label="Threaded")

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
        )

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
    """A producer that lives and dies inside a single recording: returns only
    once every thread has joined, so no frame is in flight at a boundary."""

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

        Checked here rather than in :meth:`run_recording`, to keep the
        comparison out of the recording window.
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
    """A producer that outlives every recording it logs across: threads start
    before the first ``start_recording`` and stop after the last, so it is
    mid-loop at every boundary."""

    def __init__(
        self, spec: ContextSpec, robot: object, plans: list[StreamPlan]
    ) -> None:
        super().__init__(spec, robot, plans)
        self._thread: threading.Thread | None = None
        self._frames: dict[str, list[EmittedFrame]] = {}
        self._error: BaseException | None = None

    def start(self) -> None:
        """Start logging now, before any recording exists."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._frames = run_threaded_logging(
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
            # Logs past the stop so the daemon is shown rejecting post-stop frames.
            time.sleep(PER_THREAD_LOGGING_TAIL_S)
        finally:
            self.stop_event.set()
            self._thread.join()
            self._thread = None
        if self._error is not None:
            raise RuntimeError(f"Lifetime producer failed: {self._error}") from (
                self._error
            )

    def report(self) -> dict[str, list[EmittedFrame]]:
        """Return every frame logged over the whole context lifetime."""
        return self._frames


def make_producer_session(
    spec: ContextSpec, *, robot: object, marker_name: str
) -> ProducerSession:
    """Build the producer session a case asked for: the one place
    ``producer_channels`` decides which engine runs. *marker_name* is used only
    by the synchronous producer; per-thread producers ignore it.
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
        run_threaded_logging
        if case.producer_channels == PRODUCER_OLD_PER_THREAD
        else run_synchronous_logging
    )
    return BoundedProducerSession(spec, robot, plans, engine)


def _classify_boundary_frames(
    frames: list[EmittedFrame],
    bounds: RecordingControlBounds,
) -> tuple[list[float], set[float]]:
    """Split one trace's frames into those inside a recording and those unknowable.

    A producer that logs across the recording lifecycle is mid-loop when each
    boundary passes, and neither boundary is directly observable: the daemon
    routes every sample by a publish stamp taken inside the ``log_*`` call, and
    stamps the window's own bounds inside the ``start_recording`` /
    ``stop_recording`` calls. The test resolves both the same way — by bracketing
    each unobservable instant between two it can measure on the same clock — and
    the same rule then decides every frame:

    - **Inside**: the frame's whole ``log_*`` call ran after ``start_recording``
      returned and finished before ``stop_recording`` was called, so its publish
      stamp cannot be anything but within the window. The start compares the
      emit interval's near edge and the stop its far edge; that opposition is
      the whole of the symmetry.
    - **Outside**: the call finished before ``start_recording`` was entered, so
      the window did not yet exist; or the SDK's local logging gate had already
      refused the frame at a point past the stop call. The gate is a deliberate
      superset of the window at both ends, which makes a stale read of it
      conclusive in exactly one direction at each end: a frame the gate *refused*
      was emitted after the gate closed, hence after the window closed, so it
      must not reach disk. A frame the gate *admitted* proves nothing at the
      start, because the gate opens before the window does.
    - **Unknowable**: everything else, i.e. the call straddled one of the two
      brackets. The daemon's answer for those frames is correct either way, so
      they are dropped from *both* sides of the comparison rather than tolerated
      on one.

    The ambiguous band is therefore measured, not guessed: it is as wide as the
    control call that carries the boundary, which is why a frame count could not
    size it — ``start_recording`` takes ~30 ms and a burst producer emits far
    more than a frame or two inside it.

    Returns:
        ``(inside timestamps in order, unknowable timestamps)``. Frames that are
        outside appear in neither, so the assertion requires them to be absent
        from disk.
    """
    inside: list[float] = []
    unknowable: set[float] = set()
    for frame in frames:
        is_inside = (
            frame.emitted_at >= bounds.start_returned_at
            and frame.completed_at <= bounds.stop_called_at
        )
        is_outside = frame.completed_at <= bounds.start_called_at or (
            frame.emitted_at >= bounds.stop_called_at and frame.handle != bounds.handle
        )
        if is_inside:
            inside.append(frame.timestamp)
        elif not is_outside:
            unknowable.add(frame.timestamp)
    return inside, unknowable


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

    Refuses a case whose producer outlives its recordings: it has no single
    recording's worth of frames to log.

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
        bounds_by_disk_key: dict[str, RecordingControlBounds] = {}

        session = make_producer_session(
            spec, robot=robot, marker_name="marker_synchronous"
        )
        marker_names = session.marker_names
        session.start()
        try:
            for recording_ordinal in range(spec.recordings_per_context):
                recording_capture_start_s = time.time()
                recording_capture_stop_s = recording_capture_start_s + case.duration_sec

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

                bounds_by_disk_key[disk_recording_key] = RecordingControlBounds(
                    handle=recording_handle,
                    start_called_at=start_called_at,
                    start_returned_at=start_returned_at,
                    stop_called_at=stop_called_at,
                    stop_returned_at=wall_stopped_at,
                )
        finally:
            # A surviving producer thread breaks the cleanup assertions.
            session.finish()

        report = session.report()
        for disk_key, bounds in bounds_by_disk_key.items():
            inside_by_trace: dict[str, list[float]] = {}
            unknowable_by_trace: dict[str, set[float]] = {}
            for trace_key, frames in report.items():
                inside, unknowable = _classify_boundary_frames(frames, bounds)
                inside_by_trace[trace_key] = inside
                unknowable_by_trace[trace_key] = unknowable

                breaching = [
                    frame
                    for frame in frames
                    if frame.deadline_breaches
                    and frame.emitted_at >= bounds.start_returned_at
                    and frame.completed_at <= bounds.stop_called_at
                ]
                assert not breaching, (
                    f"{trace_key} logged {len(breaching)} frame(s) inside "
                    f"recording {disk_key} that breached the logging deadline: "
                    f"{breaching[0].deadline_breaches}"
                )
            expected_by_recording[disk_key] = RecordingExpectedTimestamps(
                by_trace=inside_by_trace,
                by_trace_unknowable=unknowable_by_trace,
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

    # A late writer is reaped during the wait below, so observe across it.
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
