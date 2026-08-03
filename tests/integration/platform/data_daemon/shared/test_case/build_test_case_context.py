"""Context-spec interpretation and recording worker logic.

Translates a ``DataDaemonTestCase`` into per-context worker specs, executes
the recording workload, and provides the context-mode assertion.
Configuration dataclasses and the matrix builder live in
``matrix_test_configs.py``; per-suite case lists live in ``test_cases.py``.
"""

from __future__ import annotations

import logging
import multiprocessing
import random
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

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
    generate_joint_values,
    joint_names_for_count,
    resolve_producer_pacing,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DATASET_POLL_INTERVAL_S,
    DETAIL_REALISTIC,
    DURATION_MODE_VARIABLE,
    DURATION_VARIABLE_MAX_FACTOR,
    DURATION_VARIABLE_MIN_FACTOR,
    MAX_TIME_TO_START_S,
    MODE_STAGGERED,
    PACING_BURST_VIDEO,
    PACING_DEADLINE,
    PRODUCER_PER_THREAD,
    STOP_RECORDING_NO_WAIT_SLA_S,
    STOP_RECORDING_OVERHEAD_PER_SEC,
    STOP_RECORDING_UPLOAD_SLA_PER_JOINT_SAMPLE_S,
    STOP_RECORDING_UPLOAD_SLA_PER_VIDEO_PIXEL_S,
    random_phase_jitter_window,
)
from tests.integration.platform.data_daemon.shared.test_case.frame_source import (
    make_camera_feed,
    prewarm_frame_bank,
)

logger = logging.getLogger(__name__)

CONTEXT_DURATION_RANDOM = random.Random(0)

# Stream discriminators feeding ``stream_phase_seed`` so a recording's joint and
# video streams draw independent phase offsets.
JOINT_STREAM = 0
VIDEO_STREAM = 1

# Producer pacing for the performance suites
LOG_LOOP_FREQUENCY_HZ = 60
LOG_LOOP_INTERVAL_S = 1.0 / LOG_LOOP_FREQUENCY_HZ


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
    """

    by_trace: dict[str, list[float]]


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

    @property
    def stop_recording_sla_s(self) -> float:
        """Seconds allowed for the ``nc.stop_recording`` call.

        ``wait=False`` is fire-and-forget — the call never blocks on the
        upload pipeline — so it gets a flat constant. ``wait=True`` blocks
        until every trace has uploaded, so its budget is the sum of the
        joint-data and video-data upload costs: total joint samples
        (``duration_sec * joint_count * joint_fps``) and total video pixels
        (``duration_sec * video_fps * video_count * image_width *
        image_height``), each times an observed per-unit upload cost. The
        budget is floored at the duration-based overhead so short or
        low-volume recordings keep a sane minimum.
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
        if self.video_count and self.image_width and self.image_height:
            video_budget = (
                self.duration_sec
                * self.video_fps
                * self.video_count
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
    ``Timer`` deadline assertions it switches on producer pacing, so those suites
    emit at :data:`LOG_LOOP_FREQUENCY_HZ` instead of saturating the daemon's RGB
    spool.
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
                    producer_pacing=resolve_producer_pacing(case),
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


def _should_pace(pacing: str, is_rgb: bool) -> bool:
    """Whether a stream sleeps to its wall-clock deadline under *pacing*."""
    if pacing == PACING_DEADLINE:
        return True
    if pacing == PACING_BURST_VIDEO:
        return not is_rgb
    return False  # PACING_BURST_ALL


def _await_frame_deadline(deadline: float, *, pace: bool) -> None:
    """Wait until *deadline* when *pace* is set, otherwise return immediately."""
    if not pace:
        return
    remaining = deadline - time.time()
    if remaining > 0:
        time.sleep(remaining)


def log_synchronous_frames(
    *,
    robot_name: str,
    joint_timestamps: list[float],
    video_timestamps: list[float],
    recording_index: int,
    joint_names: list[str],
    camera_name_list: list[str],
    image_width: int | None,
    image_height: int | None,
    joint_fps: int,
    video_fps: int,
    marker_name: str,
    context_index: int,
    video_detail: str,
    pacing: str,
    assert_deadline: bool = False,  # only set by performance tests
    log_interval_s: float = 0.0,  # only set by performance tests
) -> None:
    """Log all joint and video frames for one recording synchronously.

    Both timestamp sequences are precomputed by the caller. Joint and video
    frames are interleaved in nominal-schedule order on a single thread, paced
    according to *pacing*, plus the fixed *log_interval_s* sleep after each
    iteration. Because there is only one thread, ``PACING_BURST_VIDEO`` has no
    meaning here and is resolved away by ``resolve_producer_pacing``.
    """
    joint_plan, video_plan = build_synchronous_stream_plans(
        joint_names=joint_names,
        camera_name_list=camera_name_list,
        joint_fps=joint_fps,
        video_fps=video_fps,
        marker_name=marker_name,
    )
    emitters = {
        plan: StreamEmitter(
            plan=plan,
            robot_name=robot_name,
            context_index=context_index,
            recording_index=recording_index,
            assert_deadline=assert_deadline,
        )
        for plan in (joint_plan, video_plan)
        if plan is not None
    }
    for emitter in emitters.values():
        emitter.prepare(image_width, image_height, video_detail)

    pace = _should_pace(pacing, is_rgb=False)
    recording_wall_start = time.time() if pace else 0.0
    timestamps = {joint_plan: joint_timestamps}
    if video_plan is not None:
        timestamps[video_plan] = video_timestamps
    indexes = dict.fromkeys(timestamps, 0)

    while any(indexes[plan] < len(timestamps[plan]) for plan in timestamps):
        due = min(
            (plan for plan in timestamps if indexes[plan] < len(timestamps[plan])),
            key=lambda plan: indexes[plan] / plan.fps,
        )
        frame_index = indexes[due]
        _await_frame_deadline(recording_wall_start + (frame_index / due.fps), pace=pace)
        emitters[due].emit(frame_index, timestamps[due][frame_index])
        indexes[due] += 1
        if log_interval_s:
            time.sleep(log_interval_s)


JOINT_KINDS = ("joint_positions", "joint_velocities", "joint_torques")

_JOINT_DATA_TYPES = {
    "joint_positions": "JOINT_POSITIONS",
    "joint_velocities": "JOINT_VELOCITIES",
    "joint_torques": "JOINT_TORQUES",
}


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

    @property
    def is_rgb(self) -> bool:
        """Whether this stream logs camera frames rather than joint data."""
        return self.name == "rgb"


def build_stream_plans(
    *,
    joint_names: list[str],
    camera_name_list: list[str],
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
    joint_fps: int,
    video_fps: int,
    marker_name: str,
) -> tuple[StreamPlan, StreamPlan | None]:
    """Decompose a workload for the single-threaded producer.

    One thread means a different decomposition to :func:`build_stream_plans`: all
    three joint types share a frame and a single marker, and every camera shows
    the same instant, so they share a frame too.
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
    return joint_plan, video_plan


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
    assert_deadline: bool = False
    feed: object | None = None

    def prepare(self, image_width: int | None, image_height: int | None, detail: str):
        """Build this stream's camera feed. Called on the stream's own thread."""
        self.feed = make_camera_feed(
            self.plan.is_rgb, image_width, image_height, detail
        )

    def emit(self, frame_index: int, timestamp: float | None) -> None:
        """Log one frame's payload plus this stream's marker."""
        if self.plan.is_rgb:
            self._emit_rgb(frame_index, timestamp)
        else:
            self._emit_joints(frame_index, timestamp)
        self._emit_marker(frame_index, timestamp)

    def _emit_rgb(self, frame_index: int, timestamp: float | None) -> None:
        for camera_name, camera_index in zip(
            self.plan.channel_names, self.plan.camera_indexes
        ):
            # Frame codes must stay unique across contexts, recordings and
            # cameras: they are read back to identify frames.
            frame_code = (
                (self.context_index * 1_000_000_000)
                + (self.recording_index * 10_000_000)
                + (camera_index * 100_000)
                + frame_index
            )
            rgb_image = self.feed.render(frame_index, frame_code)
            with Timer(
                MAX_TIME_TO_LOG_S,
                label="nc.log_rgb",
                assert_deadline=self.assert_deadline,
            ):
                nc.log_rgb(
                    camera_name,
                    rgb_image,
                    robot_name=self.robot_name,
                    timestamp=timestamp,
                )

    def _emit_joints(self, frame_index: int, timestamp: float | None) -> None:
        joint_values = generate_joint_values(
            frame_index, self.plan.fps, list(self.plan.channel_names)
        )
        for kind in self.plan.joint_kinds:
            # Resolved per call, not bound at import, so tests can monkeypatch nc.
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


def run_threaded_logging(
    *,
    robot_name: str,
    joint_timestamps: list[float],
    video_timestamps: list[float],
    recording_index: int,
    joint_fps: int,
    video_fps: int,
    context_index: int,
    joint_names: list[str],
    camera_name_list: list[str],
    image_width: int | None,
    image_height: int | None,
    video_detail: str,
    pacing: str,
    assert_deadline: bool = False,  # only set by performance tests
    log_interval_s: float = 0.0,  # only set by performance tests
) -> list[str]:
    """Run logging across multiple threads, one per data stream.

    Every joint-kind thread shares *joint_timestamps* and every camera thread
    shares *video_timestamps*, so the joint data types (and every camera) stay
    aligned exactly as they do in the synchronous producer. Each stream paces
    itself according to *pacing*, so ``PACING_BURST_VIDEO`` leaves the video
    threads free to outrun the joint threads, plus the fixed *log_interval_s*
    sleep after each frame.
    """
    plans = build_stream_plans(
        joint_names=joint_names,
        camera_name_list=camera_name_list,
        joint_fps=joint_fps,
        video_fps=video_fps,
    )
    emitters = [
        StreamEmitter(
            plan=plan,
            robot_name=robot_name,
            context_index=context_index,
            recording_index=recording_index,
            assert_deadline=assert_deadline,
        )
        for plan in plans
    ]
    barrier = threading.Barrier(len(emitters))

    def worker(emitter: StreamEmitter) -> None:
        barrier.wait()
        emitter.prepare(image_width, image_height, video_detail)
        fps = emitter.plan.fps
        pace = _should_pace(pacing, emitter.plan.is_rgb)
        thread_wall_start = time.time() if pace else 0.0
        timestamps = video_timestamps if emitter.plan.is_rgb else joint_timestamps
        for frame_index, timestamp in enumerate(timestamps):
            _await_frame_deadline(thread_wall_start + (frame_index / fps), pace=pace)
            emitter.emit(frame_index, timestamp)
            if log_interval_s:
                time.sleep(log_interval_s)

    _run_stream_threads(emitters, worker, error_label="Threaded")

    return [plan.marker_name for plan in plans]


def recording_timestamps(
    spec: ContextSpec, recording_index: int
) -> tuple[list[float], list[float]]:
    """Return the ``(joint, video)`` timestamp sequences for one recording.

    A pure function of ``(spec, recording_index)``, so :func:`context_worker`
    can call it to build the on-disk expectation and :func:`log_frames` can call
    it to decide what to emit, with no way for the two to drift apart.
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
    recording_index: int,
    marker_name: str,
) -> list[str]:
    """Log all frames for one recording, dispatching based on producer_channels.

    Timestamps come from :func:`recording_timestamps`, which the caller may also
    use to build the matching on-disk expectation.
    """
    joint_timestamps, video_timestamps = recording_timestamps(spec, recording_index)
    joint_name_list = joint_names_for_count(spec.case.joint_count)
    camera_name_list = camera_names(spec.case.video_count)

    if spec.case.producer_channels == PRODUCER_PER_THREAD:
        return run_threaded_logging(
            robot_name=spec.robot_name,
            joint_timestamps=joint_timestamps,
            video_timestamps=video_timestamps,
            recording_index=recording_index,
            joint_fps=spec.case.joint_fps,
            video_fps=spec.case.video_fps,
            context_index=spec.context_index,
            joint_names=joint_name_list,
            camera_name_list=camera_name_list,
            image_width=spec.case.image_width,
            image_height=spec.case.image_height,
            video_detail=spec.case.video_detail,
            pacing=spec.case.producer_pacing,
            assert_deadline=spec.assert_deadline,
            log_interval_s=spec.log_interval_s,
        )

    log_synchronous_frames(
        robot_name=spec.robot_name,
        joint_timestamps=joint_timestamps,
        video_timestamps=video_timestamps,
        recording_index=recording_index,
        joint_names=joint_name_list,
        camera_name_list=camera_name_list,
        image_width=spec.case.image_width,
        image_height=spec.case.image_height,
        joint_fps=spec.case.joint_fps,
        video_fps=spec.case.video_fps,
        marker_name=marker_name,
        context_index=spec.context_index,
        video_detail=spec.case.video_detail,
        pacing=spec.case.producer_pacing,
        assert_deadline=spec.assert_deadline,
        log_interval_s=spec.log_interval_s,
    )
    return [marker_name]


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
    if camera_name_list and case.video_detail == DETAIL_REALISTIC:
        # Render the frame bank before anything is timed. It costs ~3.5s at
        # 1080p, and the threaded producer's start barrier releases before each
        # thread builds its feed — so a lazy build inside the camera thread would
        # start the video stream seconds behind the joint streams, breaking the
        # exact RGB-frames-per-sync-point match cloud verification asserts.
        prewarm_frame_bank(case.image_width, case.image_height)
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

        for recording_ordinal in range(spec.recordings_per_context):
            recording_capture_start_s = time.time()
            recording_capture_stop_s = recording_capture_start_s + case.duration_sec

            # The very sequences log_frames will emit below.
            joint_ts, video_ts = recording_timestamps(spec, recording_ordinal)

            with Timer(
                MAX_TIME_TO_START_S,
                label="nc.start_recording",
                always_log=True,
                assert_deadline=spec.assert_deadline,
            ):
                nc.start_recording(
                    robot_name=spec.robot_name, timestamp=recording_capture_start_s
                )
            if wall_started_at is None:
                wall_started_at = time.time()

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

            # Map the two stream sequences onto every trace they feed, now that
            # the recording key is known. Trace keys use
            # "data_type/data_type_name" to match the semantic keys resolved
            # from the DB in disk_helpers. data_type_name is the storage name
            # produced by validate_safe_name (e.g. "vx300s_left\waist").
            from neuracore_types.utils import validate_safe_name

            by_trace: dict[str, list[float]] = {}
            for joint_name in joint_name_list:
                safe = validate_safe_name(joint_name)
                by_trace[f"JOINT_POSITIONS/{safe}"] = joint_ts
                by_trace[f"JOINT_VELOCITIES/{safe}"] = joint_ts
                by_trace[f"JOINT_TORQUES/{safe}"] = joint_ts
            for camera in camera_name_list:
                safe_cam = validate_safe_name(camera)
                by_trace[f"RGB_IMAGES/{safe_cam}"] = video_ts
            # CUSTOM_1D marker — name depends on producer_channels mode
            if case.producer_channels == PRODUCER_PER_THREAD:
                # One marker per joint data type thread
                for role_name in (
                    "joint_positions",
                    "joint_velocities",
                    "joint_torques",
                ):
                    safe_marker = validate_safe_name(f"marker_{role_name}")
                    by_trace[f"CUSTOM_1D/{safe_marker}"] = joint_ts
                for camera in camera_name_list:
                    safe_marker = validate_safe_name(f"marker_{camera}")
                    by_trace[f"CUSTOM_1D/{safe_marker}"] = video_ts
            else:
                safe_marker = validate_safe_name("marker_synchronous")
                by_trace[f"CUSTOM_1D/{safe_marker}"] = joint_ts
            expected_by_recording[disk_recording_key] = RecordingExpectedTimestamps(
                by_trace=by_trace,
            )

            current_marker_names = log_frames(
                spec,
                recording_index=recording_ordinal,
                marker_name="marker_synchronous",
            )
            if not marker_names:
                marker_names = current_marker_names

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

    if case.parallel_contexts == 1:
        results = [context_worker(specs[0])]
    else:
        with relayed_worker_logs() as log_queue:
            with multiprocessing.Pool(
                case.parallel_contexts,
                initializer=init_worker_logging,
                initargs=(log_queue, logging.getLogger().getEffectiveLevel()),
            ) as pool:
                results = list(  # type: ignore[return-value]
                    pool.map(_subprocess_context_worker, specs)
                )
        for result in results:
            Timer.merge_stats(result.timer_stats)

    if wait_for_traces:
        from tests.integration.platform.data_daemon.shared.db_helpers import (
            wait_for_all_traces_written,
        )

        wait_for_all_traces_written(results=results)

    return results


def create_testing_dataset_name(case: DataDaemonTestCase) -> str:
    """Create a unique dataset name for a test case."""
    return f"testing_dataset_{case_id(case)}_{uuid.uuid4().hex[:6]}"
