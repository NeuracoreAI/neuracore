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
from tests.integration.platform.data_daemon.shared.assertions import assert_context_mode
from tests.integration.platform.data_daemon.shared.auth import ensure_login
from tests.integration.platform.data_daemon.shared.macos_helpers import (
    set_thread_policy_for_macos,
)
from tests.integration.platform.data_daemon.shared.process_control import (
    MAX_TIME_TO_LOG_S,
    Timer,
    assert_on_schedule,
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
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    CONTINUOUS_LOGGING_TAIL_S,
    DATASET_POLL_INTERVAL_S,
    DETAIL_REALISTIC,
    DURATION_MODE_VARIABLE,
    DURATION_VARIABLE_MAX_FACTOR,
    DURATION_VARIABLE_MIN_FACTOR,
    MAX_TIME_TO_START_S,
    MODE_STAGGERED,
    PACING_BURST,
    PRODUCER_CONTINUOUS,
    PRODUCER_PER_THREAD,
    SCHEDULER_TOLERANCE_S,
    STOP_RECORDING_NO_WAIT_SLA_S,
    STOP_RECORDING_OVERHEAD_PER_SEC,
    STOP_RECORDING_UPLOAD_SLA_PER_JOINT_SAMPLE_S,
    STOP_RECORDING_UPLOAD_SLA_PER_VIDEO_PIXEL_S,
    TIMESTAMP_MODE_REAL,
    TIMESTAMP_MODE_STOCHASTIC,
    stochastic_jitter_window,
)
from tests.integration.platform.data_daemon.shared.test_case.frame_source import (
    make_camera_feed,
    prewarm_frame_bank,
)

logger = logging.getLogger(__name__)

CONTEXT_DURATION_RANDOM = random.Random(0)
STOCHASTIC_TIMESTAMP_RANDOM = random.Random(1)


@dataclass(frozen=True, slots=True)
class RecordingExpectedTimestamps:
    """Expected timestamps per trace for one recording, keyed by semantic trace name.

    Produced during the recording loop (once the recording key is known) and
    consumed by :func:`~disk_helpers.assert_disk_recording_properties`
    to verify on-disk trace.json files match the manually-supplied timestamps
    that were logged.

    Attributes:
        by_trace: Maps semantic trace key (e.g. ``"JOINT_POSITIONS"``,
            ``"camera_0"``) to the ordered list of expected timestamps for
            that trace within this recording.
        by_trace_fps: Maps the same semantic trace key to the producer fps for
            that trace, so the stochastic assertion can size its jitter window
            from the case's frame rate.
    """

    by_trace: dict[str, list[float]]
    by_trace_fps: dict[str, int]


@dataclass(frozen=True, slots=True)
class ContextExpectedTimestamps:
    """Expected timestamps for all recordings produced by one context worker.

    Attributes:
        by_recording: Maps the on-disk recording directory name to its
            :class:`RecordingExpectedTimestamps`. The directory name is the
            integer ``recording_index`` (as a string) under the Rust daemon, or
            the cloud ``recording_id`` under the legacy daemon.
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
    timestamp_mode: str
    video_detail: str
    video_pacing: str

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
      against the dataset's ``recording.id``. Under the legacy daemon
      ``nc.start_recording()`` returns this directly. Under the Rust daemon the
      daemon mints it asynchronously, so an entry may be an empty string until
      the test resolves it (via ``resolve_cloud_recording_ids``) once online.

    The remaining fields apply only under the Rust daemon (the daemon owns
    recording identity); they are left empty under the legacy daemon, which uses
    ``recording_ids`` for every correlation:

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
    timestamp_mode: str
    expected_timestamps: ContextExpectedTimestamps | None = None
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
    start_delay_s: float
    assert_deadline: bool = False


def build_context_specs(
    case: DataDaemonTestCase,
    dataset_name: str | None = None,
    assert_deadline: bool = False,
) -> list[ContextSpec]:
    """Build per-context worker specs for a matrix case."""
    specs: list[ContextSpec] = []
    timestamp_stagger_s = case.duration_sec / 2.0
    wall_stagger_s = 0.5
    base_recordings_per_context = case.recording_count // case.parallel_contexts
    recording_remainder = case.recording_count % case.parallel_contexts
    shared_dataset_name = (
        dataset_name or f"testing_dataset_{case_id(case)}_{uuid.uuid4().hex[:6]}"
    )

    for context_index in range(case.parallel_contexts):
        timestamp_start_s = 0.0
        start_delay_s = 0.0
        if context_index > 0 and case.mode == MODE_STAGGERED:
            timestamp_start_s = float(timestamp_stagger_s * context_index)
            start_delay_s = wall_stagger_s * context_index

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
                    timestamp_mode=case.timestamp_mode,
                    video_detail=case.video_detail,
                    video_pacing=case.video_pacing,
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
                start_delay_s=start_delay_s,
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


def get_jitter(use_stochastic_timestamps: bool, fps: int) -> float:
    if use_stochastic_timestamps:
        window = stochastic_jitter_window(fps)
        return STOCHASTIC_TIMESTAMP_RANDOM.uniform(-window, window)
    return 0.0


def precompute_timestamps(
    timestamp_start_s: float, frame_count: int, fps: int
) -> list[float]:
    """Return the complete synthetic timestamp sequence for one stream."""
    return [timestamp_start_s + frame_index / fps for frame_index in range(frame_count)]


def _await_frame_deadline(
    deadline: float, *, pace: bool, stop_event: threading.Event | None = None
) -> bool:
    """Wait until *deadline* when *pace* is set, otherwise return immediately.

    Shared by :func:`run_threaded_logging` (bounded, no *stop_event*) and
    :func:`run_continuous_logging` (unbounded, interruptible via *stop_event*).
    Returns ``True`` if *stop_event* fired while waiting, meaning the caller's
    loop should stop; always ``False`` otherwise.
    """
    if not pace:
        return False
    remaining = deadline - time.time()
    if remaining <= 0:
        return False
    if stop_event is not None:
        return stop_event.wait(remaining)
    time.sleep(remaining)
    return False


def log_synchronous_frames(
    *,
    robot_name: str,
    joint_frame_count: int,
    video_frame_count: int,
    recording_index: int,
    timestamp_start_s: float,
    joint_names: list[str],
    camera_name_list: list[str],
    image_width: int | None,
    image_height: int | None,
    joint_fps: int,
    video_fps: int,
    marker_name: str,
    context_index: int,
    video_detail: str,
    use_real_timestamps: bool = False,
    use_stochastic_timestamps: bool = False,
    assert_deadline: bool = False,  # only set by performance tests
) -> None:
    """Log all joint and video frames for one recording synchronously.

    Joint and video frames are interleaved in timestamp order. Manual timestamps
    are precomputed and emitted without wall-clock pacing; real and stochastic
    modes retain the deadline scheduler used by their timing assertions.
    """
    camera_feed = make_camera_feed(
        bool(camera_name_list), image_width, image_height, video_detail
    )

    pace_against_wall_clock = use_real_timestamps or use_stochastic_timestamps
    recording_wall_start = time.time() if pace_against_wall_clock else 0.0
    joint_timestamps = precompute_timestamps(
        timestamp_start_s, joint_frame_count, joint_fps
    )
    video_timestamps = precompute_timestamps(
        timestamp_start_s, video_frame_count, video_fps
    )
    joint_index = 0
    video_index = 0

    while joint_index < joint_frame_count or video_index < (
        video_frame_count if camera_name_list else 0
    ):
        joint_due = joint_index < joint_frame_count
        video_due = camera_name_list and video_index < video_frame_count
        # One jitter is shared by both deadlines/timestamps this iteration, so
        # size it to the tighter (higher-fps) window to stay within both.
        jitter = (
            get_jitter(
                use_stochastic_timestamps,
                max(joint_fps, video_fps) if camera_name_list else joint_fps,
            )
            if pace_against_wall_clock
            else 0.0
        )

        joint_deadline = (
            recording_wall_start + (joint_index / joint_fps) + jitter
            if joint_due
            else float("inf")
        )
        video_deadline = (
            recording_wall_start + (video_index / video_fps) + jitter
            if video_due
            else float("inf")
        )

        if joint_deadline <= video_deadline:
            if pace_against_wall_clock:
                remaining = joint_deadline - time.time()
                if remaining > 0:
                    time.sleep(remaining)
            if assert_deadline and use_stochastic_timestamps:
                assert_on_schedule(
                    joint_deadline, SCHEDULER_TOLERANCE_S, label="joint frame"
                )
            if use_real_timestamps:
                timestamp = None
            elif not use_stochastic_timestamps:
                timestamp = joint_timestamps[joint_index]
            else:
                intended = timestamp_start_s + (joint_index / joint_fps)
                timestamp = intended + jitter
            joint_values = generate_joint_values(joint_index, joint_fps, joint_names)
            with Timer(
                MAX_TIME_TO_LOG_S,
                label="nc.log_joint_positions",
                assert_deadline=assert_deadline,
            ):
                nc.log_joint_positions(
                    joint_values, robot_name=robot_name, timestamp=timestamp
                )
            with Timer(
                MAX_TIME_TO_LOG_S,
                label="nc.log_joint_velocities",
                assert_deadline=assert_deadline,
            ):
                nc.log_joint_velocities(
                    joint_values, robot_name=robot_name, timestamp=timestamp
                )
            with Timer(
                MAX_TIME_TO_LOG_S,
                label="nc.log_joint_torques",
                assert_deadline=assert_deadline,
            ):
                nc.log_joint_torques(
                    joint_values, robot_name=robot_name, timestamp=timestamp
                )
            with Timer(
                MAX_TIME_TO_LOG_S,
                label="nc.log_custom_1d",
                assert_deadline=assert_deadline,
            ):
                nc.log_custom_1d(
                    marker_name,
                    np.array([float(joint_index)], dtype=np.float32),
                    robot_name=robot_name,
                    timestamp=timestamp,
                )
            joint_index += 1
        else:
            if pace_against_wall_clock:
                remaining = video_deadline - time.time()
                if remaining > 0:
                    time.sleep(remaining)
            if assert_deadline and use_stochastic_timestamps:
                assert_on_schedule(
                    video_deadline, SCHEDULER_TOLERANCE_S, label="video frame"
                )
            if use_real_timestamps:
                timestamp = None
            elif not use_stochastic_timestamps:
                timestamp = video_timestamps[video_index]
            else:
                intended = timestamp_start_s + (video_index / video_fps)
                timestamp = intended + jitter

            for camera_index, camera_name in enumerate(camera_name_list):
                frame_code = (
                    (context_index * 1_000_000_000)
                    + (recording_index * 10_000_000)
                    + (camera_index * 100_000)
                    + video_index
                )
                # video_index, not camera_index: every camera shows the same
                # instant, distinguished only by its embedded frame code.
                rgb_image = camera_feed.render(video_index, frame_code)
                with Timer(
                    MAX_TIME_TO_LOG_S,
                    label="nc.log_rgb",
                    assert_deadline=assert_deadline,
                ):
                    nc.log_rgb(
                        camera_name,
                        rgb_image,
                        robot_name=robot_name,
                        timestamp=timestamp,
                    )
            video_index += 1


def build_thread_roles(
    *,
    joint_names: list[str],
    camera_name_list: list[str],
) -> list[dict[str, object]]:
    """Build role specs for per-thread logging."""
    roles: list[dict[str, object]] = []
    for camera_name in camera_name_list:
        roles.append({
            "role": "rgb",
            "camera_names": [camera_name],
            "marker_name": f"marker_{camera_name}",
        })
    for role_name in ("joint_positions", "joint_velocities", "joint_torques"):
        roles.append({
            "role": role_name,
            "joint_names": list(joint_names),
            "marker_name": f"marker_{role_name}",
        })
    return roles


_ROLE_DATA_TYPES = {
    "rgb": "RGB_IMAGES",
    "joint_positions": "JOINT_POSITIONS",
    "joint_velocities": "JOINT_VELOCITIES",
    "joint_torques": "JOINT_TORQUES",
}


def _role_trace_keys(role_spec: dict[str, object]) -> list[str]:
    """Return the semantic trace keys one logged frame from *role_spec* touches.

    Matches the ``data_type/data_type_name`` keys used elsewhere in this module
    (see ``context_worker``'s synthetic ``by_trace`` construction) — one entry
    per named channel the role owns (joints or its single camera), plus the
    role's own ``CUSTOM_1D`` marker.
    """
    from neuracore_types.utils import validate_safe_name

    role_name = str(role_spec["role"])
    data_type = _ROLE_DATA_TYPES[role_name]
    names = (
        role_spec["camera_names"] if role_name == "rgb" else role_spec["joint_names"]
    )
    return [
        *(f"{data_type}/{validate_safe_name(str(name))}" for name in names),
        f"CUSTOM_1D/{validate_safe_name(str(role_spec['marker_name']))}",
    ]


def _run_role_threads(
    roles: list[dict[str, object]],
    worker: Callable[[dict[str, object]], None],
    *,
    error_label: str,
) -> None:
    """Run *worker* on one daemon thread per role, then join and surface errors.

    Shared by :func:`run_threaded_logging` and :func:`run_continuous_logging`,
    which differ only in what their *worker* does per frame.
    """
    thread_errors: list[BaseException] = []

    def guarded(role_spec: dict[str, object]) -> None:
        try:
            worker(role_spec)
        except BaseException as exc:  # noqa: BLE001
            thread_errors.append(exc)

    threads = [
        threading.Thread(target=guarded, args=(role,), daemon=True) for role in roles
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
    joint_frame_count: int,
    video_frame_count: int,
    recording_index: int,
    timestamp_start_s: float,
    joint_fps: int,
    video_fps: int,
    context_index: int,
    joint_names: list[str],
    camera_name_list: list[str],
    image_width: int | None,
    image_height: int | None,
    video_detail: str,
    use_real_timestamps: bool = False,
    use_stochastic_timestamps: bool = False,
    assert_deadline: bool = False,  # only set by performance tests
    burst_video: bool = False,
) -> list[str]:
    """Run logging across multiple threads, one per data role.

    Manual timestamps are precomputed per role and emitted without wall-clock
    pacing. Real and stochastic modes keep their existing deadline scheduling.
    """
    roles = build_thread_roles(
        joint_names=joint_names, camera_name_list=camera_name_list
    )
    barrier = threading.Barrier(len(roles))

    def worker(role_spec: dict[str, object]) -> None:
        """Execute logging for a single thread role."""
        set_thread_policy_for_macos()  # set policy because new thread
        barrier.wait()
        role_name = str(role_spec["role"])
        marker_name = str(role_spec["marker_name"])
        is_rgb = role_name == "rgb"
        frame_count = video_frame_count if is_rgb else joint_frame_count
        fps = video_fps if is_rgb else joint_fps
        camera_feed = make_camera_feed(is_rgb, image_width, image_height, video_detail)
        pace_against_wall_clock = use_real_timestamps or use_stochastic_timestamps
        thread_wall_start = time.time() if pace_against_wall_clock else 0.0
        timestamps = precompute_timestamps(timestamp_start_s, frame_count, fps)
        burst = is_rgb and burst_video
        for frame_index in range(frame_count):
            jitter = (
                get_jitter(use_stochastic_timestamps, fps)
                if pace_against_wall_clock
                else 0.0
            )
            frame_deadline = thread_wall_start + (frame_index / fps) + jitter
            _await_frame_deadline(
                frame_deadline, pace=pace_against_wall_clock and not burst
            )
            if assert_deadline and use_stochastic_timestamps and not burst:
                assert_on_schedule(
                    frame_deadline,
                    SCHEDULER_TOLERANCE_S,
                    label=f"{role_name} frame",
                )
            if use_real_timestamps:
                timestamp = None
            elif not use_stochastic_timestamps:
                timestamp = timestamps[frame_index]
            else:
                intended = timestamp_start_s + (frame_index / fps)
                timestamp = intended + jitter
            if is_rgb:
                for camera_offset, camera_name in enumerate(role_spec["camera_names"]):
                    camera_id = str(camera_name)
                    camera_index = camera_name_list.index(camera_id) + camera_offset
                    frame_code = (
                        (context_index * 1_000_000_000)
                        + (recording_index * 10_000_000)
                        + (camera_index * 100_000)
                        + frame_index
                    )
                    rgb_image = camera_feed.render(frame_index, frame_code)
                    with Timer(
                        MAX_TIME_TO_LOG_S,
                        label="nc.log_rgb",
                        assert_deadline=assert_deadline,
                    ):
                        nc.log_rgb(
                            camera_id,
                            rgb_image,
                            robot_name=robot_name,
                            timestamp=timestamp,
                        )
            else:
                thread_joint_names = list(role_spec["joint_names"])
                joint_values = generate_joint_values(
                    frame_index, joint_fps, thread_joint_names
                )
                if role_name == "joint_positions":
                    with Timer(
                        MAX_TIME_TO_LOG_S,
                        label="nc.log_joint_positions",
                        assert_deadline=assert_deadline,
                    ):
                        nc.log_joint_positions(
                            joint_values,
                            robot_name=robot_name,
                            timestamp=timestamp,
                        )
                elif role_name == "joint_velocities":
                    with Timer(
                        MAX_TIME_TO_LOG_S,
                        label="nc.log_joint_velocities",
                        assert_deadline=assert_deadline,
                    ):
                        nc.log_joint_velocities(
                            joint_values,
                            robot_name=robot_name,
                            timestamp=timestamp,
                        )
                else:
                    with Timer(
                        MAX_TIME_TO_LOG_S,
                        label="nc.log_joint_torques",
                        assert_deadline=assert_deadline,
                    ):
                        nc.log_joint_torques(
                            joint_values,
                            robot_name=robot_name,
                            timestamp=timestamp,
                        )
            with Timer(
                MAX_TIME_TO_LOG_S,
                label="nc.log_custom_1d",
                assert_deadline=assert_deadline,
            ):
                nc.log_custom_1d(
                    marker_name,
                    np.array([float(frame_index)], dtype=np.float32),
                    robot_name=robot_name,
                    timestamp=timestamp,
                )

    _run_role_threads(roles, worker, error_label="Threaded")

    return [str(role["marker_name"]) for role in roles]


def run_continuous_logging(
    *,
    robot: object,
    robot_name: str,
    joint_names: list[str],
    camera_name_list: list[str],
    image_width: int | None,
    image_height: int | None,
    joint_fps: int,
    video_fps: int,
    video_detail: str,
    timestamp_start_s: float,
    use_stochastic_timestamps: bool,
    burst_video: bool,
    stop_event: threading.Event,
) -> dict[str, dict[str, list[float]]]:
    """Log continuously for the whole context lifetime, independent of any
    single recording.

    Mirrors real deployments where camera/proprioception loops run for the
    process lifetime: threads start before the first ``nc.start_recording()``
    and keep logging — paced by a session-wide, ever-increasing frame index —
    until *stop_event* is set, regardless of how many recordings start and stop
    while they run. Each frame reads which recording (if any) is active via
    ``robot.get_current_recording_id()`` before logging, so the caller can
    build expected-timestamp maps from what the SDK actually admitted rather
    than from a synthetic per-recording window.

    Returns:
        Mapping of local recording handle -> trace key -> ordered list of
        timestamps logged while that handle was current. Frames logged while
        no recording is active (handle is ``None``) are dropped.
    """
    roles = build_thread_roles(
        joint_names=joint_names, camera_name_list=camera_name_list
    )
    barrier = threading.Barrier(len(roles))
    report: dict[str, dict[str, list[float]]] = {}
    report_lock = threading.Lock()

    def worker(role_spec: dict[str, object]) -> None:
        set_thread_policy_for_macos()
        barrier.wait()
        role_name = str(role_spec["role"])
        marker_name = str(role_spec["marker_name"])
        is_rgb = role_name == "rgb"
        fps = video_fps if is_rgb else joint_fps
        camera_feed = make_camera_feed(is_rgb, image_width, image_height, video_detail)
        trace_keys = _role_trace_keys(role_spec)
        camera_id = str(role_spec["camera_names"][0]) if is_rgb else ""
        thread_joint_names = [] if is_rgb else list(role_spec["joint_names"])
        log_joint_fn = {
            "joint_positions": nc.log_joint_positions,
            "joint_velocities": nc.log_joint_velocities,
            "joint_torques": nc.log_joint_torques,
        }.get(role_name)
        burst = is_rgb and burst_video
        thread_wall_start = time.time()
        frame_index = 0
        while not stop_event.is_set():
            jitter = get_jitter(use_stochastic_timestamps, fps)
            frame_deadline = thread_wall_start + (frame_index / fps) + jitter
            if _await_frame_deadline(
                frame_deadline, pace=not burst, stop_event=stop_event
            ):
                break
            timestamp = timestamp_start_s + (frame_index / fps) + jitter

            handle = robot.get_current_recording_id()

            if is_rgb:
                rgb_image = camera_feed.render(frame_index, frame_index)
                with Timer(
                    MAX_TIME_TO_LOG_S, label="nc.log_rgb", assert_deadline=False
                ):
                    nc.log_rgb(
                        camera_id,
                        rgb_image,
                        robot_name=robot_name,
                        timestamp=timestamp,
                    )
            else:
                joint_values = generate_joint_values(
                    frame_index, fps, thread_joint_names
                )
                with Timer(
                    MAX_TIME_TO_LOG_S,
                    label=f"nc.{role_name}",
                    assert_deadline=False,
                ):
                    log_joint_fn(
                        joint_values, robot_name=robot_name, timestamp=timestamp
                    )
            with Timer(
                MAX_TIME_TO_LOG_S, label="nc.log_custom_1d", assert_deadline=False
            ):
                nc.log_custom_1d(
                    marker_name,
                    np.array([float(frame_index)], dtype=np.float32),
                    robot_name=robot_name,
                    timestamp=timestamp,
                )

            if handle is not None:
                with report_lock:
                    bucket = report.setdefault(handle, {})
                    for trace_key in trace_keys:
                        bucket.setdefault(trace_key, []).append(timestamp)

            frame_index += 1

    _run_role_threads(roles, worker, error_label="Continuous")

    return report


def log_frames(
    spec: ContextSpec,
    *,
    recording_index: int,
    marker_name: str,
) -> list[str]:
    """Log all frames for one recording, dispatching based on producer_channels.

    Derives timestamp mode and all frame parameters from *spec*.
    """
    use_real_timestamps = spec.case.timestamp_mode == TIMESTAMP_MODE_REAL
    use_stochastic_timestamps = spec.case.timestamp_mode == TIMESTAMP_MODE_STOCHASTIC
    recording_timestamp_start_s = (
        spec.timestamp_start_s + recording_index * spec.case.duration_sec
    )
    joint_name_list = joint_names_for_count(spec.case.joint_count)
    camera_name_list = camera_names(spec.case.video_count)

    if spec.case.producer_channels == PRODUCER_PER_THREAD:
        return run_threaded_logging(
            robot_name=spec.robot_name,
            joint_frame_count=spec.expected_joint_frames,
            video_frame_count=spec.expected_video_frames,
            recording_index=recording_index,
            timestamp_start_s=recording_timestamp_start_s,
            joint_fps=spec.case.joint_fps,
            video_fps=spec.case.video_fps,
            context_index=spec.context_index,
            joint_names=joint_name_list,
            camera_name_list=camera_name_list,
            image_width=spec.case.image_width,
            image_height=spec.case.image_height,
            video_detail=spec.case.video_detail,
            use_real_timestamps=use_real_timestamps,
            use_stochastic_timestamps=use_stochastic_timestamps,
            assert_deadline=spec.assert_deadline,
            burst_video=spec.case.video_pacing == PACING_BURST,
        )

    log_synchronous_frames(
        robot_name=spec.robot_name,
        joint_frame_count=spec.expected_joint_frames,
        video_frame_count=spec.expected_video_frames,
        recording_index=recording_index,
        timestamp_start_s=recording_timestamp_start_s,
        joint_names=joint_name_list,
        camera_name_list=camera_name_list,
        image_width=spec.case.image_width,
        image_height=spec.case.image_height,
        joint_fps=spec.case.joint_fps,
        video_fps=spec.case.video_fps,
        marker_name=marker_name,
        context_index=spec.context_index,
        video_detail=spec.case.video_detail,
        use_real_timestamps=use_real_timestamps,
        use_stochastic_timestamps=use_stochastic_timestamps,
        assert_deadline=spec.assert_deadline,
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
    double-counted when stats are merged back. The stochastic-timestamp RNG
    is reseeded per-context so parallel workers produce independent jitter
    sequences instead of replaying the parent's seed. Spawned workers
    (macOS) additionally re-authenticate, as they do not inherit the
    parent's in-process auth state.
    """
    multiprocessing.current_process().name = f"ctx-{spec.context_index}"
    Timer._stats.clear()
    STOCHASTIC_TIMESTAMP_RANDOM.seed(1 + spec.context_index)
    ensure_login()
    return context_worker(spec)


def context_worker(spec: ContextSpec) -> ContextResult:
    """Execute recordings for a single parallel context."""
    from neuracore.data_daemon.rust_selection import is_rust_daemon_enabled
    from tests.integration.platform.data_daemon.shared.db_helpers import (
        wait_for_recording_index_for_source,
    )

    use_rust = is_rust_daemon_enabled()
    set_thread_policy_for_macos()
    case = spec.case
    use_real_timestamps = case.timestamp_mode == TIMESTAMP_MODE_REAL
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

    if spec.start_delay_s > 0.0:
        time.sleep(spec.start_delay_s)

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

        expected_by_recording: dict[str, RecordingExpectedTimestamps] | None = (
            {} if not use_real_timestamps else None
        )

        continuous_stop_event: threading.Event | None = None
        continuous_thread: threading.Thread | None = None
        continuous_outcome: dict[str, object] = {}
        handle_to_disk_key: dict[str, str] = {}
        trace_key_to_fps: dict[str, int] = {}

        if case.producer_channels == PRODUCER_CONTINUOUS:
            roles = build_thread_roles(
                joint_names=joint_name_list, camera_name_list=camera_name_list
            )
            marker_names = [str(role["marker_name"]) for role in roles]
            for role in roles:
                fps = case.video_fps if role["role"] == "rgb" else case.joint_fps
                for trace_key in _role_trace_keys(role):
                    trace_key_to_fps[trace_key] = fps

            continuous_stop_event = threading.Event()

            def _run_continuous() -> None:
                try:
                    continuous_outcome["report"] = run_continuous_logging(
                        robot=robot,
                        robot_name=spec.robot_name,
                        joint_names=joint_name_list,
                        camera_name_list=camera_name_list,
                        image_width=case.image_width,
                        image_height=case.image_height,
                        joint_fps=case.joint_fps,
                        video_fps=case.video_fps,
                        video_detail=case.video_detail,
                        timestamp_start_s=spec.timestamp_start_s,
                        use_stochastic_timestamps=(
                            case.timestamp_mode == TIMESTAMP_MODE_STOCHASTIC
                        ),
                        burst_video=case.video_pacing == PACING_BURST,
                        stop_event=continuous_stop_event,
                    )
                except BaseException as exc:  # noqa: BLE001
                    continuous_outcome["error"] = exc

            continuous_thread = threading.Thread(target=_run_continuous, daemon=True)
            continuous_thread.start()

        try:
            for recording_ordinal in range(spec.recordings_per_context):
                recording_timestamp_start_s = (
                    spec.timestamp_start_s + recording_ordinal * case.duration_sec
                )
                recording_capture_start_s = None if use_real_timestamps else time.time()
                recording_capture_stop_s = (
                    None
                    if recording_capture_start_s is None
                    else recording_capture_start_s + case.duration_sec
                )

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

                if use_rust:
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
                else:
                    recording_id = str(robot.get_current_recording_id() or "")
                    recording_ids.append(recording_id)
                    disk_recording_key = recording_id

                if case.producer_channels == PRODUCER_CONTINUOUS:
                    handle = robot.get_current_recording_id()
                    if handle is not None:
                        handle_to_disk_key[handle] = disk_recording_key
                    time.sleep(case.duration_sec)

                # Build per-recording expected timestamps once the recording key is
                # known. Trace keys use "data_type/data_type_name" to match the
                # semantic keys resolved from the DB in disk_helpers. data_type_name is
                # the storage name produced by validate_safe_name (e.g.
                # "vx300s_left\waist" for joint names).
                elif expected_by_recording is not None:
                    from neuracore_types.utils import validate_safe_name

                    joint_ts = precompute_timestamps(
                        recording_timestamp_start_s,
                        spec.expected_joint_frames,
                        case.joint_fps,
                    )
                    video_ts = precompute_timestamps(
                        recording_timestamp_start_s,
                        spec.expected_video_frames,
                        case.video_fps,
                    )
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
                    by_trace_fps = {
                        trace_key: (
                            case.video_fps if timestamps is video_ts else case.joint_fps
                        )
                        for trace_key, timestamps in by_trace.items()
                    }
                    expected_by_recording[disk_recording_key] = (
                        RecordingExpectedTimestamps(
                            by_trace=by_trace,
                            by_trace_fps=by_trace_fps,
                        )
                    )

                if case.producer_channels != PRODUCER_CONTINUOUS:
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

            if continuous_stop_event is not None:
                # Give the still-running producer threads time to log a few
                # more frames after the last stop_recording, exactly as a real
                # camera would keep running past the end of a recording.
                time.sleep(CONTINUOUS_LOGGING_TAIL_S)
        finally:
            # Always stop and join the continuous producer threads — even when
            # the loop above raised — so no producer thread outlives this
            # worker and confuses the daemon-cleanup assertions that run next.
            if continuous_thread is not None:
                continuous_stop_event.set()
                continuous_thread.join()

        if continuous_thread is not None:
            continuous_error = continuous_outcome.get("error")
            if continuous_error is not None:
                raise RuntimeError(
                    f"Continuous producer failed: {continuous_error}"
                ) from continuous_error

            if expected_by_recording is not None:
                report: dict[str, dict[str, list[float]]] = continuous_outcome.get(
                    "report", {}
                )
                for handle, disk_key in handle_to_disk_key.items():
                    by_trace = report.get(handle, {})
                    expected_by_recording[disk_key] = RecordingExpectedTimestamps(
                        by_trace=by_trace,
                        by_trace_fps={
                            trace_key: trace_key_to_fps[trace_key]
                            for trace_key in by_trace
                        },
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
            timestamp_mode=case.timestamp_mode,
            expected_timestamps=(
                ContextExpectedTimestamps(by_recording=expected_by_recording)
                if expected_by_recording is not None
                else None
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
    assert_mode: bool = True,
    wait_for_traces: bool = False,
) -> list[ContextResult]:
    """Run all parallel contexts for a matrix test case.

    Executes each context spec either sequentially (when parallel_contexts==1)
    or concurrently via a multiprocessing pool. Sequential execution avoids
    pool overhead and simplifies debugging for single-context cases.

    Args:
        case: The test case defining parallelism level and context matrix.
        specs: Pre-built context specs to run. If None, built from ``case``
            via :func:`build_context_specs`.
        assert_mode: When ``True`` (default), calls :func:`assert_context_mode`
            after running to verify expected parallelization behaviour.
        wait_for_traces: When ``True``, waits for all traces to be written to
            disk after running (implies ``assert_mode``).

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

    if assert_mode or wait_for_traces:
        assert_context_mode(case, results)

    if wait_for_traces:
        from tests.integration.platform.data_daemon.shared.db_helpers import (
            wait_for_all_traces_written,
        )

        wait_for_all_traces_written(results=results)

    return results


def create_testing_dataset_name(case: DataDaemonTestCase) -> str:
    """Create a unique dataset name for a test case."""
    return f"testing_dataset_{case_id(case)}_{uuid.uuid4().hex[:6]}"
