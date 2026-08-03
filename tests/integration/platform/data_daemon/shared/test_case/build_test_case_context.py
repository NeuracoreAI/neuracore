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
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DATASET_POLL_INTERVAL_S,
    DURATION_MODE_VARIABLE,
    DURATION_VARIABLE_MAX_FACTOR,
    DURATION_VARIABLE_MIN_FACTOR,
    FRAME_BYTE_LENGTH,
    FRAME_COLOR_CHANNELS,
    FRAME_DEFAULT_FILL_VALUE,
    FRAME_GRID_SIZE,
    FRAME_HALF_DIVISOR,
    FRAME_MAX_COLOR_VALUE,
    MAX_TIME_TO_START_S,
    MODE_STAGGERED,
    PRODUCER_PER_THREAD,
    STOP_RECORDING_NO_WAIT_SLA_S,
    STOP_RECORDING_OVERHEAD_PER_SEC,
    STOP_RECORDING_UPLOAD_SLA_PER_JOINT_SAMPLE_S,
    STOP_RECORDING_UPLOAD_SLA_PER_VIDEO_PIXEL_S,
    random_phase_jitter_window,
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


def encode_frame_number(
    frame_num: int, width: int, height: int, out: np.ndarray | None = None
) -> np.ndarray:
    """Encode a frame number into the pixel data of a synthetic video frame.

    The 16-byte big-endian representation of ``frame_num`` is written into the
    top-left 4x4 grid of the image. For each pixel at ``(row, col)`` in that
    grid the byte value is mapped to the RGB channels as follows:

    - Red channel = ``byte_value``
    - Green channel = ``FRAME_MAX_COLOR_VALUE - byte_value``
    - Blue channel = ``byte_value // FRAME_HALF_DIVISOR``

    The remaining pixels are filled with :data:`FRAME_DEFAULT_FILL_VALUE`.

    Args:
        frame_num: The frame number to embed. Must fit in 16 bytes (i.e.
            less than ``2 ** 128``).
        width: Frame width in pixels.
        height: Frame height in pixels.
        out: If given, write into this preallocated ``(height, width, 3)``
            ``uint8`` array instead of allocating a new one, and skip the
            fill (every grid cell is always overwritten below, so a buffer
            filled once and reused is never left with stale pixels). Callers
            that pass ``out`` must never retain the returned array past the
            point where its contents may next be overwritten.

    Returns:
        A NumPy array with shape ``(height, width, 3)`` and dtype ``uint8``.
    """
    if out is None:
        img = np.zeros((height, width, FRAME_COLOR_CHANNELS), dtype=np.uint8)
        img.fill(FRAME_DEFAULT_FILL_VALUE)
    else:
        img = out

    frame_bytes = frame_num.to_bytes(FRAME_BYTE_LENGTH, byteorder="big")

    for row in range(FRAME_GRID_SIZE):
        for col in range(FRAME_GRID_SIZE):
            idx = row * FRAME_GRID_SIZE + col
            if idx < len(frame_bytes):
                pixel_value = frame_bytes[idx]
                img[row, col, 0] = pixel_value
                img[row, col, 1] = FRAME_MAX_COLOR_VALUE - pixel_value
                img[row, col, 2] = pixel_value // FRAME_HALF_DIVISOR

    return img


def preallocate_frame_buffer(
    should_allocate: bool, image_width: int | None, image_height: int | None
) -> np.ndarray | None:
    """Preallocate and fill a reusable frame buffer, or return ``None``.

    Callers that log video frames reuse a single buffer across iterations
    (via :func:`encode_frame_number`'s ``out`` param) instead of allocating a
    new one per frame, for a reduced memory footprint.

    Args:
        should_allocate: Whether this caller needs a buffer at all (e.g. the
            recording has cameras, or this thread's role is "rgb").
        image_width: Frame width in pixels, or ``None`` if not video.
        image_height: Frame height in pixels, or ``None`` if not video.

    Returns:
        A preallocated, pre-filled ``(image_height, image_width, 3)``
        ``uint8`` array, or ``None`` if ``should_allocate`` is ``False`` or
        either dimension is ``None``.
    """
    if not should_allocate or image_width is None or image_height is None:
        return None
    frame_buffer = np.zeros(
        (image_height, image_width, FRAME_COLOR_CHANNELS), dtype=np.uint8
    )
    frame_buffer.fill(FRAME_DEFAULT_FILL_VALUE)
    encode_frame_number(0, image_width, image_height, out=frame_buffer)
    return frame_buffer


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
    marker_name: str,
    context_index: int,
    assert_deadline: bool = False,  # only set by performance tests
    log_interval_s: float = 0.0,  # only set by performance tests
) -> None:
    """Log all joint and video frames for one recording synchronously.

    Both timestamp sequences are precomputed by the caller and emitted as fast
    as the transport allows, save for the fixed *log_interval_s* sleep after
    each iteration — the timestamps themselves are never paced to wall clock.
    Frames are interleaved in timestamp order so the daemon receives them the
    way a real producer would order them.
    """
    frame_buffer = preallocate_frame_buffer(
        bool(camera_name_list), image_width, image_height
    )

    joint_frame_count = len(joint_timestamps)
    video_frame_count = len(video_timestamps) if camera_name_list else 0
    joint_index = 0
    video_index = 0

    while joint_index < joint_frame_count or video_index < video_frame_count:
        next_joint = (
            joint_timestamps[joint_index]
            if joint_index < joint_frame_count
            else float("inf")
        )
        next_video = (
            video_timestamps[video_index]
            if video_index < video_frame_count
            else float("inf")
        )

        if next_joint <= next_video:
            timestamp = next_joint
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
            timestamp = next_video
            for camera_index, camera_name in enumerate(camera_name_list):
                frame_code = (
                    (context_index * 1_000_000_000)
                    + (recording_index * 10_000_000)
                    + (camera_index * 100_000)
                    + video_index
                )
                rgb_image = encode_frame_number(
                    frame_code, image_width, image_height, out=frame_buffer
                )
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

        if log_interval_s:
            time.sleep(log_interval_s)


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


def run_threaded_logging(
    *,
    robot_name: str,
    joint_timestamps: list[float],
    video_timestamps: list[float],
    recording_index: int,
    joint_fps: int,
    context_index: int,
    joint_names: list[str],
    camera_name_list: list[str],
    image_width: int | None,
    image_height: int | None,
    assert_deadline: bool = False,  # only set by performance tests
    log_interval_s: float = 0.0,  # only set by performance tests
) -> list[str]:
    """Run logging across multiple threads, one per data role.

    Every role emits the timestamp sequence its stream was given, as fast as the
    transport allows, save for the fixed *log_interval_s* sleep after each
    iteration — the timestamps themselves are never paced to wall clock.  Each
    role paces its own stream, so the rgb roles that feed the daemon's spool are
    capped independently of the joint roles.  All joint roles share the joint
    sequence, so the three joint data types stay aligned exactly as they do in
    the synchronous producer.
    """
    roles = build_thread_roles(
        joint_names=joint_names, camera_name_list=camera_name_list
    )
    barrier = threading.Barrier(len(roles))
    thread_errors: list[BaseException] = []

    def worker(role_spec: dict[str, object]) -> None:
        """Execute logging for a single thread role."""
        try:
            barrier.wait()
            role_name = str(role_spec["role"])
            marker_name = str(role_spec["marker_name"])
            is_rgb = role_name == "rgb"
            timestamps = video_timestamps if is_rgb else joint_timestamps
            frame_buffer = preallocate_frame_buffer(is_rgb, image_width, image_height)
            for frame_index, timestamp in enumerate(timestamps):
                if is_rgb:
                    for camera_offset, camera_name in enumerate(
                        role_spec["camera_names"]
                    ):
                        camera_id = str(camera_name)
                        camera_index = camera_name_list.index(camera_id) + camera_offset
                        frame_code = (
                            (context_index * 1_000_000_000)
                            + (recording_index * 10_000_000)
                            + (camera_index * 100_000)
                            + frame_index
                        )
                        rgb_image = encode_frame_number(
                            frame_code, image_width, image_height, out=frame_buffer
                        )
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
                if log_interval_s:
                    time.sleep(log_interval_s)
        except BaseException as exc:  # noqa: BLE001
            thread_errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(role,), daemon=True) for role in roles
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if thread_errors:
        raise RuntimeError(
            f"Threaded producer failed: {thread_errors[0]}"
        ) from thread_errors[0]

    return [str(role["marker_name"]) for role in roles]


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
            context_index=spec.context_index,
            joint_names=joint_name_list,
            camera_name_list=camera_name_list,
            image_width=spec.case.image_width,
            image_height=spec.case.image_height,
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
        marker_name=marker_name,
        context_index=spec.context_index,
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
