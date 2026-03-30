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
import pytest
from neuracore_types import DataType
from recording_playback_shared import (
    MATRIX_SESSION_RUNS,
    MAX_TIME_TO_LOG_S,
    Timer,
    decode_frame_number,
    encode_frame_number,
    wait_for_dataset_ready,
)

import neuracore as nc
from neuracore.core.config.config_manager import get_config_manager

MAX_TIME_TO_START_S = 20.0
STOP_RECORDING_OVERHEAD_PER_SEC = 0.5
BASE_DATASET_READY_TIMEOUT_S = 180.0
MAX_DATASET_READY_TIMEOUT_S = 3600.0

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class RecordingPlaybackMatrixCase:
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

    @property
    def has_video(self) -> bool:
        return self.data_type == "all"

    @property
    def fps(self) -> int:
        return 1

    @property
    def expected_frames(self) -> int:
        return self.fps * self.duration_sec

    @property
    def recordings_per_context(self) -> int:
        return self.recording_count // self.parallel_contexts


def _build_matrix_cases() -> list[RecordingPlaybackMatrixCase]:
    cases: list[RecordingPlaybackMatrixCase] = []

    for duration_sec in (60, 300):
        for parallel_contexts in (1, 2):
            for recording_count in MATRIX_RECORDING_COUNTS:
                modes = ("sequential",)
                if parallel_contexts > 1:
                    modes = ("sequential", "staggered")
                for mode in modes:
                    for joint_count in (10, 30):
                        for producer_channels in ("synchronous", "per_thread"):
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
                                )
                            )
                            for video_count in (1, 6):
                                for image_width, image_height in (
                                    (64, 64),
                                    (1920, 1080),
                                ):
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
                                        )
                                    )
    return cases


def _case_id(case: RecordingPlaybackMatrixCase) -> str:
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


MATRIX_CASES = _build_matrix_cases()


def _has_configured_org() -> bool:
    if os.environ.get("NEURACORE_ORG_ID"):
        return True
    try:
        return bool(get_config_manager().config.current_org_id)
    except Exception:  # noqa: BLE001
        return False


def _joint_names_for_count(joint_count: int) -> list[str]:
    if joint_count <= len(BASE_JOINT_NAMES):
        return BASE_JOINT_NAMES[:joint_count]
    generated_names = list(BASE_JOINT_NAMES)
    for index in range(len(BASE_JOINT_NAMES), joint_count):
        generated_names.append(f"synthetic_joint_{index:02d}")
    return generated_names


def _generate_joint_values(
    frame_index: int,
    fps: int,
    joint_names: list[str],
) -> dict[str, float]:
    timestamp = frame_index / fps
    return {
        joint_name: math.sin(timestamp * (0.5 + (index * 0.25)))
        for index, joint_name in enumerate(joint_names)
    }


def _case_timeout_seconds(case: RecordingPlaybackMatrixCase) -> float:
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


def _build_context_specs(case: RecordingPlaybackMatrixCase) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    timestamp_stagger_s = case.duration_sec / 2.0
    wall_stagger_s = 0.5
    shared_dataset_name = f"testing_dataset_{_case_id(case)}_{uuid.uuid4().hex[:6]}"

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


def _camera_names(video_count: int) -> list[str]:
    return [f"camera_{index}" for index in range(video_count)]


def _log_synchronous_frame(
    *,
    robot_name: str,
    frame_index: int,
    recording_index: int,
    timestamp: float,
    joint_names: list[str],
    camera_names: list[str],
    image_width: int | None,
    image_height: int | None,
    fps: int,
    marker_name: str,
    context_index: int,
) -> None:
    for camera_index, camera_name in enumerate(camera_names):
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

    joint_values = _generate_joint_values(frame_index, fps, joint_names)
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


def _build_thread_roles(
    *,
    joint_names: list[str],
    camera_names: list[str],
) -> list[dict[str, object]]:
    roles: list[dict[str, object]] = []
    for camera_name in camera_names:
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


def _run_threaded_logging(
    *,
    robot_name: str,
    frame_count: int,
    recording_index: int,
    timestamp_start_s: float,
    fps: int,
    context_index: int,
    joint_names: list[str],
    camera_names: list[str],
    image_width: int | None,
    image_height: int | None,
) -> list[str]:
    roles = _build_thread_roles(joint_names=joint_names, camera_names=camera_names)
    barrier = threading.Barrier(len(roles))
    thread_errors: list[BaseException] = []

    def worker(role_spec: dict[str, object]) -> None:
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
                        camera_index = camera_names.index(camera_id) + camera_offset
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
                    joint_values = _generate_joint_values(
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


def _matrix_context_worker(spec: dict[str, object]) -> dict[str, object]:
    case = dict(spec["case"])
    duration_sec = int(case["duration_sec"])
    fps = int(case["fps"])
    frame_count = int(spec["expected_frames"])
    joint_names = _joint_names_for_count(int(case["joint_count"]))
    camera_names = _camera_names(int(case["video_count"]))
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

    if start_delay_s > 0.0:
        time.sleep(start_delay_s)

    with Timer(MAX_TIME_TO_START_S, label="matrix.nc.login", always_log=True):
        nc.login()
    with Timer(MAX_TIME_TO_START_S, label="matrix.nc.connect_robot", always_log=True):
        robot = nc.connect_robot(robot_name, overwrite=False)
    with Timer(MAX_TIME_TO_START_S, label="matrix.nc.create_dataset", always_log=True):
        nc.create_dataset(dataset_name)

    wall_started_at: float | None = None
    wall_stopped_at: float = 0.0

    for recording_index in range(recordings_per_context):
        recording_timestamp_start_s = timestamp_start_s + recording_index * duration_sec

        with Timer(
            MAX_TIME_TO_START_S, label="matrix.nc.start_recording", always_log=True
        ):
            nc.start_recording(robot_name=robot_name)
        if wall_started_at is None:
            wall_started_at = time.time()
        recording_ids.append(str(robot.get_current_recording_id() or ""))

        if str(case["producer_channels"]) == "per_thread":
            current_marker_names = _run_threaded_logging(
                robot_name=robot_name,
                frame_count=frame_count,
                recording_index=recording_index,
                timestamp_start_s=recording_timestamp_start_s,
                fps=fps,
                context_index=context_index,
                joint_names=joint_names,
                camera_names=camera_names,
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
                _log_synchronous_frame(
                    robot_name=robot_name,
                    frame_index=frame_index,
                    recording_index=recording_index,
                    timestamp=timestamp,
                    joint_names=joint_names,
                    camera_names=camera_names,
                    image_width=image_width,
                    image_height=image_height,
                    fps=fps,
                    marker_name=marker_names[0],
                    context_index=context_index,
                )

        with Timer(
            duration_sec * STOP_RECORDING_OVERHEAD_PER_SEC + MAX_TIME_TO_START_S,
            label="matrix.nc.stop_recording",
            always_log=True,
        ):
            nc.stop_recording(robot_name=robot_name, wait=True)
        wall_stopped_at = time.time()

    return {
        "dataset_name": dataset_name,
        "recording_ids": recording_ids,
        "robot_name": robot_name,
        "joint_names": joint_names,
        "camera_names": camera_names,
        "frame_count": frame_count,
        "fps": fps,
        "duration_sec": duration_sec,
        "timestamp_start_s": timestamp_start_s,
        "timestamp_end_s": float(spec["timestamp_end_s"]),
        "marker_names": marker_names,
        "has_video": bool(camera_names),
        "context_index": context_index,
        "wall_started_at": wall_started_at,
        "wall_stopped_at": wall_stopped_at,
        "data_type": str(case["data_type"]),
    }


def _collect_episode_summary(synced_episode: object) -> dict[str, object]:
    summary: dict[str, object] = {
        "sync_points": 0,
        "rgb_counts": {},
        "frame_codes": {},
        "joint_position_counts": {},
        "joint_velocity_counts": {},
        "joint_torque_counts": {},
        "joint_position_values": [],
        "custom_counts": {},
    }

    for frame_index, sync_point in enumerate(  # type: ignore[call-overload]
        synced_episode
    ):
        summary["sync_points"] = int(summary["sync_points"]) + 1

        if DataType.RGB_IMAGES in sync_point.data:
            rgb_counts = dict(summary["rgb_counts"])
            frame_codes = dict(summary["frame_codes"])
            for camera_name, camera_data in sync_point[DataType.RGB_IMAGES].items():
                name = str(camera_name)
                rgb_counts[name] = rgb_counts.get(name, 0) + 1
                frame_codes.setdefault(name, []).append(
                    decode_frame_number(np.array(camera_data.frame))
                )
            summary["rgb_counts"] = rgb_counts
            summary["frame_codes"] = frame_codes

        if DataType.JOINT_POSITIONS in sync_point.data:
            counts = dict(summary["joint_position_counts"])
            for joint_name, joint_data in sync_point[DataType.JOINT_POSITIONS].items():
                name = str(joint_name)
                counts[name] = counts.get(name, 0) + 1
                summary["joint_position_values"].append(
                    (frame_index, name, float(joint_data.value))
                )
            summary["joint_position_counts"] = counts

        if DataType.JOINT_VELOCITIES in sync_point.data:
            counts = dict(summary["joint_velocity_counts"])
            for joint_name in sync_point[DataType.JOINT_VELOCITIES].keys():
                name = str(joint_name)
                counts[name] = counts.get(name, 0) + 1
            summary["joint_velocity_counts"] = counts

        if DataType.JOINT_TORQUES in sync_point.data:
            counts = dict(summary["joint_torque_counts"])
            for joint_name in sync_point[DataType.JOINT_TORQUES].keys():
                name = str(joint_name)
                counts[name] = counts.get(name, 0) + 1
            summary["joint_torque_counts"] = counts

        if DataType.CUSTOM_1D in sync_point.data:
            custom_counts = dict(summary["custom_counts"])
            for name in sync_point[DataType.CUSTOM_1D].keys():
                key = str(name)
                custom_counts[key] = custom_counts.get(key, 0) + 1
            summary["custom_counts"] = custom_counts

    return summary


def _verify_episode_summary(
    *,
    summary: dict[str, object],
    result: dict[str, object],
    case: RecordingPlaybackMatrixCase,
    recording_index: int,
) -> None:
    joint_names = list(result["joint_names"])
    frame_count = int(result["frame_count"])
    fps = int(result["fps"])

    for joint_name in joint_names:
        assert summary["joint_position_counts"].get(joint_name) == frame_count
        assert summary["joint_velocity_counts"].get(joint_name) == frame_count
        assert summary["joint_torque_counts"].get(joint_name) == frame_count

    for frame_index, joint_name, actual_value in summary["joint_position_values"]:
        joint_index = joint_names.index(joint_name)
        expected_value = math.sin((frame_index / fps) * (0.5 + (joint_index * 0.25)))
        assert abs(actual_value - expected_value) <= 1e-5

    expected_marker_names = list(result["marker_names"])
    for marker_name in expected_marker_names:
        assert summary["custom_counts"].get(marker_name) == frame_count

    if case.data_type == "joints_only":
        assert summary["rgb_counts"] == {}
        assert summary["frame_codes"] == {}
        return

    camera_names = list(result["camera_names"])
    for camera_index, camera_name in enumerate(camera_names):
        assert summary["rgb_counts"].get(camera_name) == frame_count
        expected_codes = [
            (int(result["context_index"]) * 1_000_000_000)
            + (recording_index * 10_000_000)
            + (camera_index * 100_000)
            + frame_index
            for frame_index in range(frame_count)
        ]
        assert summary["frame_codes"].get(camera_name) == expected_codes


def _verify_all_results(
    *,
    results: list[dict[str, object]],
    case: RecordingPlaybackMatrixCase,
) -> None:
    dataset_name = str(results[0]["dataset_name"])
    wait_for_dataset_ready(
        dataset_name,
        expected_recording_count=case.recording_count,
        timeout_s=_case_timeout_seconds(case),
        poll_interval_s=2.0,
    )

    dataset = nc.get_dataset(dataset_name)
    synced_dataset = dataset.synchronize()

    recording_lookup: dict[str, tuple[dict[str, object], int]] = {}
    for result in results:
        for rec_index, rec_id in enumerate(list(result["recording_ids"])):
            recording_lookup[str(rec_id)] = (result, rec_index)

    verified_ids: set[str] = set()

    for synced_episode in synced_dataset:
        recording_id = str(synced_episode.id)
        assert (
            recording_id in recording_lookup
        ), f"Unexpected recording {recording_id} in dataset '{dataset_name}'"
        result, recording_index = recording_lookup[recording_id]
        summary = _collect_episode_summary(synced_episode)
        _verify_episode_summary(
            summary=summary,
            result=result,
            case=case,
            recording_index=recording_index,
        )
        verified_ids.add(recording_id)

    missing = set(recording_lookup.keys()) - verified_ids
    assert not missing, f"Recordings not found in dataset '{dataset_name}': {missing}"


def _run_case_contexts(case: RecordingPlaybackMatrixCase) -> list[dict[str, object]]:
    specs = _build_context_specs(case)
    if case.parallel_contexts == 1:
        return [_matrix_context_worker(specs[0])]

    with multiprocessing.Pool(case.parallel_contexts) as pool:
        return list(pool.map(_matrix_context_worker, specs))


def _assert_context_mode(
    case: RecordingPlaybackMatrixCase, results: list[dict[str, object]]
) -> None:
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


def _log_run_analysis(
    *,
    case: RecordingPlaybackMatrixCase,
    results: list[dict[str, object]],
) -> None:
    separator = "=" * 64
    lines = [separator, f"Run analysis: {_case_id(case)}", separator]

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

    matrix_labels = sorted(
        label for label in Timer._stats if label.startswith("matrix.")
    )
    if matrix_labels:
        lines.append("\n  Timer stats  (n / avg / max):")
        for label in matrix_labels:
            stats = Timer._stats[label]
            count = int(stats["count"])
            avg = stats["total"] / count if count > 0 else 0.0
            lines.append(
                f"    {label:<42}  {count:3}x"
                f"  avg={avg:.3f}s  max={stats['max']:.3f}s"
            )

    MATRIX_SESSION_RUNS.append({
        "case_id": _case_id(case),
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
    logger.info("\n".join(lines))


@pytest.mark.parametrize("case", MATRIX_CASES, ids=_case_id)
def test_recording_and_playback_matrix(
    case: RecordingPlaybackMatrixCase,
    dataset_cleanup,
) -> None:
    for label in [key for key in Timer._stats if key.startswith("matrix.")]:
        del Timer._stats[label]
    nc.login()
    if not _has_configured_org():
        pytest.skip(
            "Recording/playback matrix tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )
    results = _run_case_contexts(case)
    dataset_cleanup(str(results[0]["dataset_name"]))
    _assert_context_mode(case, results)
    nc.login()
    _verify_all_results(results=results, case=case)
    _log_run_analysis(case=case, results=results)
