"""Shared matrix configuration, case generation, and recording workers.

This module defines the test matrix dimensions, builds parametrized test
cases, and provides the recording worker functions used by both the
online and offline matrix test files.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np
from recording_playback_shared import (
    MATRIX_SESSION_RUNS,
    MAX_TIME_TO_LOG_S,
    Timer,
    encode_frame_number,
)

import neuracore as nc
from neuracore.core.config.config_manager import get_config_manager

logger = logging.getLogger(__name__)

MAX_TIME_TO_START_S = 20.0
STOP_RECORDING_OVERHEAD_PER_SEC = 0.5
BASE_DATASET_READY_TIMEOUT_S = 180.0
MAX_DATASET_READY_TIMEOUT_S = 3600.0
DATASET_POLL_INTERVAL_S = 0.25

MATRIX_RECORDING_COUNTS: tuple[int, ...] = (10,)

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


def _cleanup_test_worker_robot(robot: object | None) -> None:
    if robot is None:
        return

    temp_dir = getattr(robot, "_temp_dir", None)
    if temp_dir is not None:
        try:
            temp_dir.cleanup()
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to cleanup matrix worker robot temp dir", exc_info=True
            )
        finally:
            robot._temp_dir = None

    if hasattr(robot, "_daemon_recording_context"):
        robot._daemon_recording_context = None


@dataclass(frozen=True)
class MatrixDimensionConfig:
    """Configurable parameter arrays for matrix generation.

    Each field is a tuple of values to iterate over. The matrix builder
    expands all combinations of these values into individual test cases.
    """

    durations_sec: tuple[int, ...] = (60, 300)
    parallel_contexts_options: tuple[int, ...] = (1, 2)
    recording_counts: tuple[int, ...] = MATRIX_RECORDING_COUNTS
    joint_counts: tuple[int, ...] = (10, 30)
    producer_channel_options: tuple[str, ...] = ("synchronous", "per_thread")
    video_counts: tuple[int, ...] = (1, 6)
    image_dimensions: tuple[tuple[int, int], ...] = ((64, 64), (1920, 1080))
    kill_daemon_between_tests: bool = True
    state_db_action: str = "empty"


@dataclass(frozen=True)
class RecordingPlaybackMatrixCase:
    """A single parametrized matrix test case.

    The ``wait`` field is intentionally absent — it is a pipeline concern.
    Online tests parametrize it separately; offline tests always use False.
    """

    duration_sec: int
    parallel_contexts: int
    recording_count: int
    mode: str
    data_type: str
    joint_count: int
    producer_channels: str
    video_count: int
    image_width: int | None
    image_height: int | None
    kill_daemon_between_tests: bool = True
    state_db_action: str = "empty"

    @property
    def has_video(self) -> bool:
        """Whether this case includes video data."""
        return self.data_type == "all"

    @property
    def fps(self) -> int:
        """Frames per second for this case."""
        return 1

    @property
    def expected_frames(self) -> int:
        """Total expected frames per recording."""
        return self.fps * self.duration_sec

    @property
    def recordings_per_context(self) -> int:
        """Number of recordings each parallel context executes."""
        return self.recording_count // self.parallel_contexts


def build_matrix_cases(
    dimension_config: MatrixDimensionConfig | None = None,
) -> list[RecordingPlaybackMatrixCase]:
    """Build all matrix test cases from the given dimension config."""
    if dimension_config is None:
        dimension_config = MatrixDimensionConfig()

    cases: list[RecordingPlaybackMatrixCase] = []

    for duration_sec in dimension_config.durations_sec:
        for parallel_contexts in dimension_config.parallel_contexts_options:
            for recording_count in dimension_config.recording_counts:
                modes = ("sequential",)
                if parallel_contexts > 1:
                    modes = ("sequential", "staggered")
                for mode in modes:
                    for joint_count in dimension_config.joint_counts:
                        for (
                            producer_channels
                        ) in dimension_config.producer_channel_options:
                            cases.append(
                                RecordingPlaybackMatrixCase(
                                    duration_sec=duration_sec,
                                    parallel_contexts=parallel_contexts,
                                    recording_count=recording_count,
                                    mode=mode,
                                    data_type="joints_only",
                                    joint_count=joint_count,
                                    producer_channels=producer_channels,
                                    video_count=0,
                                    image_width=None,
                                    image_height=None,
                                    kill_daemon_between_tests=dimension_config.kill_daemon_between_tests,
                                    state_db_action=dimension_config.state_db_action,
                                )
                            )
                            for video_count in dimension_config.video_counts:
                                for (
                                    image_width,
                                    image_height,
                                ) in dimension_config.image_dimensions:
                                    if video_count == 6 and (
                                        image_width,
                                        image_height,
                                    ) == (1920, 1080):
                                        continue
                                    cases.append(
                                        RecordingPlaybackMatrixCase(
                                            duration_sec=duration_sec,
                                            parallel_contexts=parallel_contexts,
                                            recording_count=recording_count,
                                            mode=mode,
                                            data_type="all",
                                            joint_count=joint_count,
                                            producer_channels=producer_channels,
                                            video_count=video_count,
                                            image_width=image_width,
                                            image_height=image_height,
                                            kill_daemon_between_tests=dimension_config.kill_daemon_between_tests,
                                            state_db_action=dimension_config.state_db_action,
                                        )
                                    )
    return cases


def case_id(case: RecordingPlaybackMatrixCase) -> str:
    """Generate a short human-readable ID for a matrix case."""
    mode_short = "seq" if case.mode == "sequential" else "stag"
    data_short = "joints" if case.data_type == "joints_only" else "full"
    channels_short = "sync" if case.producer_channels == "synchronous" else "threaded"
    parts = [
        f"{case.duration_sec}s",
        f"{case.parallel_contexts}ctx",
    ]
    if len(MATRIX_RECORDING_COUNTS) > 1:
        parts.append(f"{case.recording_count}recs")
    parts += [
        mode_short,
        data_short,
        f"{case.joint_count}j",
        channels_short,
    ]
    if case.has_video:
        parts.append(f"{case.video_count}cam")
        parts.append(f"{case.image_width}x{case.image_height}")
    return "-".join(parts)


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


def case_timeout_seconds(case: RecordingPlaybackMatrixCase) -> float:
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
    return min(MAX_DATASET_READY_TIMEOUT_S, timeout_s)


def build_context_specs(
    case: RecordingPlaybackMatrixCase,
    *,
    wait: bool,
) -> list[dict[str, object]]:
    """Build per-context worker specs for a matrix case."""
    specs: list[dict[str, object]] = []
    timestamp_stagger_s = case.duration_sec / 2.0
    wall_stagger_s = 0.5
    shared_dataset_name = f"testing_dataset_{case_id(case)}_{uuid.uuid4().hex[:6]}"

    for context_index in range(case.parallel_contexts):
        timestamp_start_s = 0.0
        start_delay_s = 0.0
        if context_index > 0 and case.mode == "staggered":
            timestamp_start_s = float(timestamp_stagger_s * context_index)
            start_delay_s = wall_stagger_s * context_index

        specs.append({
            "case": {
                "duration_sec": case.duration_sec,
                "data_type": case.data_type,
                "joint_count": case.joint_count,
                "producer_channels": case.producer_channels,
                "video_count": case.video_count,
                "image_width": case.image_width,
                "image_height": case.image_height,
                "fps": case.fps,
                "wait": wait,
            },
            "context_index": context_index,
            "robot_name": f"matrix_robot_{uuid.uuid4().hex[:10]}",
            "dataset_name": shared_dataset_name,
            "recordings_per_context": case.recordings_per_context,
            "expected_frames": case.expected_frames,
            "timestamp_start_s": timestamp_start_s,
            "timestamp_end_s": (
                timestamp_start_s + case.duration_sec * case.recordings_per_context
            ),
            "start_delay_s": start_delay_s,
        })
    return specs


# ---------------------------------------------------------------------------
# Recording worker functions
# ---------------------------------------------------------------------------


def log_synchronous_frame(
    *,
    robot_name: str,
    frame_index: int,
    recording_index: int,
    timestamp: float,
    joint_names: list[str],
    camera_name_list: list[str],
    image_width: int | None,
    image_height: int | None,
    fps: int,
    marker_name: str,
    context_index: int,
) -> None:
    """Log a single synchronous frame of all data types."""
    for camera_index, camera_name in enumerate(camera_name_list):
        frame_code = (
            (context_index * 1_000_000_000)
            + (recording_index * 10_000_000)
            + (camera_index * 100_000)
            + frame_index
        )
        rgb_image = encode_frame_number(frame_code, image_width, image_height)
        with Timer(MAX_TIME_TO_LOG_S, label="matrix.nc.log_rgb"):
            nc.log_rgb(
                camera_name, rgb_image, robot_name=robot_name, timestamp=timestamp
            )

    joint_values = generate_joint_values(frame_index, fps, joint_names)
    with Timer(MAX_TIME_TO_LOG_S, label="matrix.nc.log_joint_positions"):
        nc.log_joint_positions(joint_values, robot_name=robot_name, timestamp=timestamp)
    with Timer(MAX_TIME_TO_LOG_S, label="matrix.nc.log_joint_velocities"):
        nc.log_joint_velocities(
            joint_values, robot_name=robot_name, timestamp=timestamp
        )
    with Timer(MAX_TIME_TO_LOG_S, label="matrix.nc.log_joint_torques"):
        nc.log_joint_torques(joint_values, robot_name=robot_name, timestamp=timestamp)
    with Timer(MAX_TIME_TO_LOG_S, label="matrix.nc.log_custom_1d"):
        nc.log_custom_1d(
            marker_name,
            np.array([float(frame_index)], dtype=np.float32),
            robot_name=robot_name,
            timestamp=timestamp,
        )


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
    frame_count: int,
    recording_index: int,
    timestamp_start_s: float,
    fps: int,
    context_index: int,
    joint_names: list[str],
    camera_name_list: list[str],
    image_width: int | None,
    image_height: int | None,
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
            for frame_index in range(frame_count):
                timestamp = timestamp_start_s + (frame_index / fps)
                if role_name == "rgb":
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
                        with Timer(MAX_TIME_TO_LOG_S, label="matrix.nc.log_rgb"):
                            nc.log_rgb(
                                camera_id,
                                rgb_image,
                                robot_name=robot_name,
                                timestamp=timestamp,
                            )
                else:
                    thread_joint_names = list(role_spec["joint_names"])
                    joint_values = generate_joint_values(
                        frame_index, fps, thread_joint_names
                    )
                    if role_name == "joint_positions":
                        with Timer(
                            MAX_TIME_TO_LOG_S, label="matrix.nc.log_joint_positions"
                        ):
                            nc.log_joint_positions(
                                joint_values,
                                robot_name=robot_name,
                                timestamp=timestamp,
                            )
                    elif role_name == "joint_velocities":
                        with Timer(
                            MAX_TIME_TO_LOG_S, label="matrix.nc.log_joint_velocities"
                        ):
                            nc.log_joint_velocities(
                                joint_values,
                                robot_name=robot_name,
                                timestamp=timestamp,
                            )
                    else:
                        with Timer(
                            MAX_TIME_TO_LOG_S, label="matrix.nc.log_joint_torques"
                        ):
                            nc.log_joint_torques(
                                joint_values,
                                robot_name=robot_name,
                                timestamp=timestamp,
                            )
                with Timer(MAX_TIME_TO_LOG_S, label="matrix.nc.log_custom_1d"):
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


def _bind_worker_dataset(*, dataset_name: str, create_dataset: bool) -> None:
    """Ensure a worker is bound to the shared dataset before recording."""
    if create_dataset:
        with Timer(
            MAX_TIME_TO_START_S, label="matrix.nc.create_dataset", always_log=True
        ):
            nc.create_dataset(dataset_name)
        return

    last_error: Exception | None = None
    deadline = time.time() + MAX_TIME_TO_START_S
    with Timer(MAX_TIME_TO_START_S, label="matrix.nc.get_dataset", always_log=True):
        while time.time() < deadline:
            try:
                nc.get_dataset(dataset_name)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(DATASET_POLL_INTERVAL_S)

    raise RuntimeError(
        f"Timed out waiting for shared dataset '{dataset_name}' to exist"
    ) from last_error


def matrix_context_worker(spec: dict[str, object]) -> dict[str, object]:
    """Execute recordings for a single parallel context."""
    case = dict(spec["case"])
    duration_sec = int(case["duration_sec"])
    fps = int(case["fps"])
    frame_count = int(spec["expected_frames"])
    joint_name_list = joint_names_for_count(int(case["joint_count"]))
    camera_name_list = camera_names(int(case["video_count"]))
    image_width = case["image_width"]
    image_height = case["image_height"]
    robot_name = str(spec["robot_name"])
    dataset_name = str(spec["dataset_name"])
    timestamp_start_s = float(spec["timestamp_start_s"])
    start_delay_s = float(spec["start_delay_s"])
    context_index = int(spec["context_index"])
    recordings_per_context = int(spec["recordings_per_context"])
    marker_names: list[str] = []
    recording_ids: list[str] = []
    robot = None

    if start_delay_s > 0.0:
        time.sleep(start_delay_s)

    wall_started_at: float | None = None
    wall_stopped_at: float = 0.0

    try:
        with Timer(MAX_TIME_TO_START_S, label="matrix.nc.login", always_log=True):
            nc.login()
        _bind_worker_dataset(
            dataset_name=dataset_name,
            create_dataset=context_index == 0,
        )
        with Timer(
            MAX_TIME_TO_START_S, label="matrix.nc.connect_robot", always_log=True
        ):
            robot = nc.connect_robot(robot_name, overwrite=False)

        for recording_index in range(recordings_per_context):
            recording_timestamp_start_s = (
                timestamp_start_s + recording_index * duration_sec
            )

            with Timer(
                MAX_TIME_TO_START_S,
                label="matrix.nc.start_recording",
                always_log=True,
            ):
                nc.start_recording(robot_name=robot_name)
            if wall_started_at is None:
                wall_started_at = time.time()
            recording_ids.append(str(robot.get_current_recording_id() or ""))

            if str(case["producer_channels"]) == "per_thread":
                current_marker_names = run_threaded_logging(
                    robot_name=robot_name,
                    frame_count=frame_count,
                    recording_index=recording_index,
                    timestamp_start_s=recording_timestamp_start_s,
                    fps=fps,
                    context_index=context_index,
                    joint_names=joint_name_list,
                    camera_name_list=camera_name_list,
                    image_width=image_width,
                    image_height=image_height,
                )
                if not marker_names:
                    marker_names = current_marker_names
            else:
                if not marker_names:
                    marker_names = ["marker_synchronous"]
                for frame_index in range(frame_count):
                    timestamp = recording_timestamp_start_s + (frame_index / fps)
                    log_synchronous_frame(
                        robot_name=robot_name,
                        frame_index=frame_index,
                        recording_index=recording_index,
                        timestamp=timestamp,
                        joint_names=joint_name_list,
                        camera_name_list=camera_name_list,
                        image_width=image_width,
                        image_height=image_height,
                        fps=fps,
                        marker_name=marker_names[0],
                        context_index=context_index,
                    )

            with Timer(
                duration_sec * STOP_RECORDING_OVERHEAD_PER_SEC,
                label="matrix.nc.stop_recording",
                always_log=True,
            ):
                nc.stop_recording(robot_name=robot_name, wait=bool(case["wait"]))
            wall_stopped_at = time.time()

        return {
            "dataset_name": dataset_name,
            "recording_ids": recording_ids,
            "robot_name": robot_name,
            "joint_names": joint_name_list,
            "camera_names": camera_name_list,
            "frame_count": frame_count,
            "fps": fps,
            "duration_sec": duration_sec,
            "timestamp_start_s": timestamp_start_s,
            "timestamp_end_s": float(spec["timestamp_end_s"]),
            "marker_names": marker_names,
            "has_video": bool(camera_name_list),
            "context_index": context_index,
            "wall_started_at": wall_started_at,
            "wall_stopped_at": wall_stopped_at,
            "data_type": str(case["data_type"]),
        }
    except Exception:
        if robot is not None:
            try:
                if robot.is_recording():
                    nc.cancel_recording(robot_name=robot_name)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to cancel active matrix recording for %s",
                    robot_name,
                    exc_info=True,
                )
        raise
    finally:
        _cleanup_test_worker_robot(robot)


def run_case_contexts(
    case: RecordingPlaybackMatrixCase,
    *,
    wait: bool,
    specs: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Run all parallel contexts for a matrix case."""
    if specs is None:
        specs = build_context_specs(case, wait=wait)

    if case.parallel_contexts == 1:
        return [matrix_context_worker(specs[0])]

    with multiprocessing.Pool(case.parallel_contexts) as pool:
        return list(pool.map(matrix_context_worker, specs))


def assert_context_mode(
    case: RecordingPlaybackMatrixCase, results: list[dict[str, object]]
) -> None:
    """Assert that context timing matches the expected mode."""
    if len(results) < 2:
        return

    ordered_results = sorted(results, key=lambda result: int(result["context_index"]))
    first = ordered_results[0]
    second = ordered_results[1]
    if case.mode == "sequential":
        assert float(first["timestamp_start_s"]) == float(second["timestamp_start_s"])
        assert float(first["timestamp_end_s"]) == float(second["timestamp_end_s"])
        assert float(first["wall_stopped_at"]) > float(second["wall_started_at"])
        return

    assert float(first["timestamp_end_s"]) > float(second["timestamp_start_s"])
    assert float(first["wall_stopped_at"]) > float(second["wall_started_at"])


def log_run_analysis(
    *,
    case: RecordingPlaybackMatrixCase,
    results: list[dict[str, object]],
) -> str:
    """Log a detailed analysis of a matrix run for diagnostics."""
    separator = "=" * 64
    lines = [separator, f"Run analysis: {case_id(case)}", separator]

    total_frames = case.recording_count * case.expected_frames
    lines += [
        f"  Case:          {case.recording_count} recordings x"
        f" {case.duration_sec}s @ {case.fps} fps",
        f"                 {case.joint_count} joints,"
        f" {case.producer_channels} channels",
    ]
    if case.has_video:
        lines.append(
            f"                 {case.video_count} camera(s)"
            f" @ {case.image_width}x{case.image_height}"
        )
    lines.append(f"  Total frames:  {total_frames}")

    if results:
        lines.append("\n  Context wall times:")
        for result in sorted(results, key=lambda result: int(result["context_index"])):
            wall_s = float(result["wall_stopped_at"]) - float(result["wall_started_at"])
            recordings_per_context = case.recordings_per_context
            avg_per_recording = (
                wall_s / recordings_per_context if recordings_per_context else 0.0
            )
            lines.append(
                f"    ctx[{result['context_index']}]: {wall_s:.1f}s total,"
                f" {avg_per_recording:.1f}s avg per recording"
            )
    else:
        lines.append(
            "\n  Context wall times: unavailable "
            "(run aborted before contexts completed)"
        )

    matrix_labels = sorted(
        label for label in Timer._stats if label.startswith("matrix.")
    )
    if matrix_labels:
        lines.append("\n  Timer stats  (n / avg / max / limit):")
        for label in matrix_labels:
            stats = Timer._stats[label]
            count = int(stats["count"])
            avg = stats["total"] / count if count > 0 else 0.0
            lines.append(
                f"    {label:<42}  {count:3}x"
                f"  avg={avg:.3f}s  max={stats['max']:.3f}s"
                f"  limit={stats['limit']:.3f}s"
            )

    MATRIX_SESSION_RUNS.append({
        "case_id": case_id(case),
        "timer_stats": {label: dict(Timer._stats[label]) for label in matrix_labels},
        "context_results": [
            {
                "context_index": int(result["context_index"]),
                "wall_s": float(result["wall_stopped_at"])
                - float(result["wall_started_at"]),
            }
            for result in results
        ],
    })

    lines.append(separator)
    report = "\n".join(lines)
    logger.info(report)
    return report


DEFAULT_DIMENSION_CONFIG = MatrixDimensionConfig()
MATRIX_CASES = build_matrix_cases(DEFAULT_DIMENSION_CONFIG)
