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
from tests.integration.platform.data_daemon.shared.assertions import assert_context_mode
from tests.integration.platform.data_daemon.shared.process_control import (
    MAX_TIME_TO_LOG_S,
    Timer,
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
    STOCHASTIC_JITTER_S,
    STOP_RECORDING_OVERHEAD_PER_SEC,
    TIMESTAMP_MODE_REAL,
    TIMESTAMP_MODE_STOCHASTIC,
)

logger = logging.getLogger(__name__)

CONTEXT_DURATION_RANDOM = random.Random(0)
STOCHASTIC_TIMESTAMP_RANDOM = random.Random(1)


def encode_frame_number(frame_num: int, width: int, height: int) -> np.ndarray:
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

    Returns:
        A NumPy array with shape ``(height, width, 3)`` and dtype ``uint8``.
    """
    img = np.zeros((height, width, FRAME_COLOR_CHANNELS), dtype=np.uint8)
    img.fill(FRAME_DEFAULT_FILL_VALUE)

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


@dataclass(frozen=True, slots=True)
class RecordingExpectedTimestamps:
    """Expected timestamps per trace for one recording, keyed by semantic trace name.

    Produced during the recording loop (once the recording ID is known) and
    consumed by :func:`~disk_helpers.assert_disk_recording_properties`
    to verify on-disk trace.json files match the manually-supplied timestamps
    that were logged.

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
        by_recording: Maps recording ID to its :class:`RecordingExpectedTimestamps`.
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


@dataclass(frozen=True, slots=True)
class ContextResult:
    """Per-context result from a completed recording workload.

    Produced by :func:`context_worker` and consumed by assertion helpers
    and verification functions throughout the test suite.
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
                int(
                    case.duration_sec
                    * CONTEXT_DURATION_RANDOM.uniform(
                        DURATION_VARIABLE_MIN_FACTOR, DURATION_VARIABLE_MAX_FACTOR
                    )
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


def get_jitter(use_stochastic_timestamps: bool) -> float:
    if use_stochastic_timestamps:
        return STOCHASTIC_TIMESTAMP_RANDOM.uniform(
            -STOCHASTIC_JITTER_S, STOCHASTIC_JITTER_S
        )
    return 0.0


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
    use_real_timestamps: bool = False,
    use_stochastic_timestamps: bool = False,
    assert_deadline: bool = False,  # only set by performance tests
) -> None:
    """Log all joint and video frames for one recording synchronously.

    Joint and video frames are interleaved in a single loop using a wall-clock
    deadline scheduler, so both streams advance together in time order.
    """
    timing_tolerance = STOCHASTIC_JITTER_S if use_stochastic_timestamps else None
    recording_wall_start = time.time()
    joint_index = 0
    video_index = 0

    while joint_index < joint_frame_count or video_index < (
        video_frame_count if camera_name_list else 0
    ):
        joint_due = joint_index < joint_frame_count
        video_due = camera_name_list and video_index < video_frame_count
        jitter = get_jitter(use_stochastic_timestamps)

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
            remaining = joint_deadline - time.time()
            if remaining > 0:
                time.sleep(remaining)
            if use_real_timestamps:
                timestamp = None
            else:
                intended = timestamp_start_s + (joint_index / joint_fps)
                timestamp = intended + jitter
            joint_values = generate_joint_values(joint_index, joint_fps, joint_names)
            with Timer(
                MAX_TIME_TO_LOG_S,
                label="nc.log_joint_positions",
                deadline=joint_deadline if assert_deadline else None,
                timing_tolerance=timing_tolerance,
            ):
                nc.log_joint_positions(
                    joint_values, robot_name=robot_name, timestamp=timestamp
                )
            with Timer(
                MAX_TIME_TO_LOG_S,
                label="nc.log_joint_velocities",
                deadline=joint_deadline if assert_deadline else None,
                timing_tolerance=timing_tolerance,
            ):
                nc.log_joint_velocities(
                    joint_values, robot_name=robot_name, timestamp=timestamp
                )
            with Timer(
                MAX_TIME_TO_LOG_S,
                label="nc.log_joint_torques",
                deadline=joint_deadline if assert_deadline else None,
                timing_tolerance=timing_tolerance,
            ):
                nc.log_joint_torques(
                    joint_values, robot_name=robot_name, timestamp=timestamp
                )
            with Timer(
                MAX_TIME_TO_LOG_S,
                label="nc.log_custom_1d",
                deadline=joint_deadline if assert_deadline else None,
                timing_tolerance=timing_tolerance,
            ):
                nc.log_custom_1d(
                    marker_name,
                    np.array([float(joint_index)], dtype=np.float32),
                    robot_name=robot_name,
                    timestamp=timestamp,
                )
            joint_index += 1
        else:
            remaining = video_deadline - time.time()
            if remaining > 0:
                time.sleep(remaining)
            if use_real_timestamps:
                timestamp = None
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
                rgb_image = encode_frame_number(frame_code, image_width, image_height)
                with Timer(
                    MAX_TIME_TO_LOG_S,
                    label="nc.log_rgb",
                    deadline=video_deadline if assert_deadline else None,
                    timing_tolerance=timing_tolerance,
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
    use_real_timestamps: bool = False,
    use_stochastic_timestamps: bool = False,
    assert_deadline: bool = False,  # only set by performance tests
) -> list[str]:
    """Run logging across multiple threads, one per data role."""
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
            frame_count = video_frame_count if is_rgb else joint_frame_count
            fps = video_fps if is_rgb else joint_fps
            thread_wall_start = time.time()
            jitter = get_jitter(use_stochastic_timestamps)
            timing_tolerance = (
                STOCHASTIC_JITTER_S if use_stochastic_timestamps else None
            )
            for frame_index in range(frame_count):
                frame_deadline = thread_wall_start + (frame_index / fps)
                remaining = frame_deadline - time.time()
                if remaining > 0:
                    time.sleep(remaining)
                if use_real_timestamps:
                    timestamp = None
                else:
                    intended = timestamp_start_s + (frame_index / fps)
                    timestamp = intended + jitter
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
                            frame_code, image_width, image_height
                        )
                        with Timer(
                            MAX_TIME_TO_LOG_S,
                            label="nc.log_rgb",
                            deadline=frame_deadline if assert_deadline else None,
                            timing_tolerance=timing_tolerance,
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
                            deadline=frame_deadline if assert_deadline else None,
                            timing_tolerance=timing_tolerance,
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
                            deadline=frame_deadline if assert_deadline else None,
                            timing_tolerance=timing_tolerance,
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
                            deadline=frame_deadline if assert_deadline else None,
                            timing_tolerance=timing_tolerance,
                        ):
                            nc.log_joint_torques(
                                joint_values,
                                robot_name=robot_name,
                                timestamp=timestamp,
                            )
                with Timer(
                    MAX_TIME_TO_LOG_S,
                    label="nc.log_custom_1d",
                    deadline=frame_deadline if assert_deadline else None,
                    timing_tolerance=timing_tolerance,
                ):
                    nc.log_custom_1d(
                        marker_name,
                        np.array([float(frame_index)], dtype=np.float32),
                        robot_name=robot_name,
                        timestamp=timestamp,
                    )
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
            use_real_timestamps=use_real_timestamps,
            use_stochastic_timestamps=use_stochastic_timestamps,
            assert_deadline=spec.assert_deadline,
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
        use_real_timestamps=use_real_timestamps,
        use_stochastic_timestamps=use_stochastic_timestamps,
        assert_deadline=spec.assert_deadline,
    )
    return [marker_name]


def _bind_worker_dataset(*, dataset_name: str, create_dataset: bool) -> nc.Dataset:
    """Ensure a worker is bound to the shared dataset before recording."""
    if create_dataset:
        with Timer(MAX_TIME_TO_START_S, label="nc.create_dataset", always_log=True):
            dataset = nc.create_dataset(dataset_name)
        with Timer(MAX_TIME_TO_START_S, label="nc.connect_dataset", always_log=True):
            return nc.connect_dataset(dataset)

    last_error: Exception | None = None
    deadline = time.time() + MAX_TIME_TO_START_S
    with Timer(MAX_TIME_TO_START_S, label="nc.connect_dataset", always_log=True):
        while time.time() < deadline:
            try:
                return nc.connect_dataset(dataset_name)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(DATASET_POLL_INTERVAL_S)

    raise RuntimeError(
        f"Timed out waiting for shared dataset '{dataset_name}' to exist"
    ) from last_error


def _subprocess_context_worker(spec: ContextSpec) -> ContextResult:
    """Subprocess wrapper for context_worker used by multiprocessing.Pool.

    On Linux, Pool uses fork so workers inherit a copy of the parent's
    Timer._stats. Clearing it here ensures workers only capture their own
    timers and the parent's pre-fork timers (e.g. nc.login) are not
    double-counted when stats are merged back.
    """
    Timer._stats.clear()
    return context_worker(spec)


def context_worker(spec: ContextSpec) -> ContextResult:
    """Execute recordings for a single parallel context."""
    case = spec.case
    use_real_timestamps = case.timestamp_mode == TIMESTAMP_MODE_REAL
    joint_name_list = joint_names_for_count(case.joint_count)
    camera_name_list = camera_names(case.video_count)
    marker_names: list[str] = []
    recording_ids: list[str] = []
    robot = None

    if spec.start_delay_s > 0.0:
        time.sleep(spec.start_delay_s)

    wall_started_at: float | None = None
    wall_stopped_at: float = 0.0

    try:
        dataset = _bind_worker_dataset(
            dataset_name=spec.dataset_name,
            create_dataset=spec.context_index == 0,
        )
        if spec.context_index == 0:
            with Timer(MAX_TIME_TO_START_S, label="nc.create_robot", always_log=True):
                nc.create_robot(spec.robot_name)
        with Timer(MAX_TIME_TO_START_S, label="nc.connect_robot", always_log=True):
            robot = nc.connect_robot(spec.robot_name)

        expected_by_recording: dict[str, RecordingExpectedTimestamps] | None = (
            {} if not use_real_timestamps else None
        )

        for recording_index in range(spec.recordings_per_context):
            recording_timestamp_start_s = (
                spec.timestamp_start_s + recording_index * case.duration_sec
            )

            with Timer(
                MAX_TIME_TO_START_S,
                label="nc.start_recording",
                always_log=True,
            ):
                dataset.start_recording(robot_name=spec.robot_name)
            if wall_started_at is None:
                wall_started_at = time.time()
            recording_id = str(robot.get_current_recording_id() or "")
            recording_ids.append(recording_id)

            # Build per-recording expected timestamps once the recording ID is known.
            # Keys use "data_type/data_type_name" to match the semantic keys resolved
            # from the DB in daemon_disk_helpers. data_type_name is the storage name
            # produced by validate_safe_name (e.g. "vx300s_left\waist" for joint names).
            if expected_by_recording is not None:
                from neuracore_types.utils import validate_safe_name

                joint_ts = [
                    recording_timestamp_start_s + i / case.joint_fps
                    for i in range(spec.expected_joint_frames)
                ]
                video_ts = [
                    recording_timestamp_start_s + i / case.video_fps
                    for i in range(spec.expected_video_frames)
                ]
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
                expected_by_recording[recording_id] = RecordingExpectedTimestamps(
                    by_trace=by_trace
                )

            current_marker_names = log_frames(
                spec,
                recording_index=recording_index,
                marker_name="marker_synchronous",
            )
            if not marker_names:
                marker_names = current_marker_names

            with Timer(
                case.duration_sec * STOP_RECORDING_OVERHEAD_PER_SEC,
                label="nc.stop_recording",
                always_log=True,
                assert_limit=False,
            ):
                dataset.stop_recording(robot_name=spec.robot_name, wait=case.wait)
            wall_stopped_at = time.time()

        captured_timer_stats = {k: dict(v) for k, v in Timer._stats.items()}
        return ContextResult(
            dataset_name=spec.dataset_name,
            recording_ids=recording_ids,
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
                    dataset.cancel_recording(robot_name=spec.robot_name)
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
    assert_deadline: bool = False,
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
        assert_deadline: Passed through to :func:`build_context_specs` when
            ``specs`` is ``None``.
        assert_mode: When ``True`` (default), calls :func:`assert_context_mode`
            after running to verify expected parallelization behaviour.
        wait_for_traces: When ``True``, waits for all traces to be written to
            disk after running (implies ``assert_mode``).

    Returns:
        List of result dicts from each context worker, one per spec.
    """
    if specs is None:
        specs = build_context_specs(case, assert_deadline=assert_deadline)

    if case.parallel_contexts == 1:
        results = [context_worker(specs[0])]
    else:
        with multiprocessing.Pool(case.parallel_contexts) as pool:
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
