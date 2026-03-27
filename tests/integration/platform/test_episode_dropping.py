"""Regression tests for robot-like mixed-datatype recordings.

These tests are more targeted than the broad streaming tests. They cover the
class of issues where a recording with several different datatypes and a
robot-like joint schema could be dropped, become unreadable, or fail to
synchronize correctly.
"""

import sys
import uuid
from pathlib import Path

import numpy as np
from neuracore_types import DataType
from recording_playback_shared import wait_for_dataset_ready

import neuracore as nc

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent.parent.parent / "examples"))
# ruff: noqa: E402
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH

VX300S_JOINT_NAMES = [
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

LANGUAGE_NAME = "instruction"
LANGUAGE_TEXT = "Pick up the cube and pass it to the other robot"
CAMERA_NAME = "angle"


def make_joint_dict(frame_idx: int, offset: float = 0.0) -> dict[str, float]:
    return {
        joint_name: float(np.sin(frame_idx * 0.1 + i * 0.05 + offset))
        for i, joint_name in enumerate(VX300S_JOINT_NAMES)
    }


def make_target_dict(frame_idx: int) -> dict[str, float]:
    return {
        joint_name: float(np.cos(frame_idx * 0.1 + i * 0.03))
        for i, joint_name in enumerate(VX300S_JOINT_NAMES)
    }


def make_rgb(frame_idx: int, width: int = 64, height: int = 64) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = 40
    img[0, 0, 0] = frame_idx % 256
    img[0, 0, 1] = (frame_idx * 3) % 256
    img[0, 0, 2] = (frame_idx * 7) % 256
    return img


def collect_sync_points(dataset_name: str, timeout_s: float = 120.0):
    wait_for_dataset_ready(
        dataset_name,
        expected_recording_count=1,
        timeout_s=timeout_s,
        poll_interval_s=1.0,
    )

    dataset = nc.get_dataset(dataset_name)
    recordings = list(dataset)
    assert len(recordings) == 1

    synced_dataset = dataset.synchronize()
    synced_episodes = list(synced_dataset)
    assert len(synced_episodes) == 1

    sync_points = list(synced_episodes[0])
    assert sync_points, "Expected synchronized recording to contain sync points"

    return recordings, sync_points


def assert_expected_datatypes_present(
    sync_points, expected_datatypes: set[DataType]
) -> None:
    found_datatypes = set()

    for sync_point in sync_points:
        for data_type in expected_datatypes:
            if data_type in sync_point.data:
                found_datatypes.add(data_type)

    assert found_datatypes == expected_datatypes


def test_vx300s_mixed_datatypes_recording_is_not_dropped():
    robot_name = f"vx300s-mixed-robot-{uuid.uuid4().hex[:8]}"
    dataset_name = f"vx300s-mixed-dataset-{uuid.uuid4().hex[:8]}"

    nc.login()
    robot = nc.connect_robot(
        robot_name,
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )
    nc.create_dataset(dataset_name)

    nc.start_recording()
    recording_id = robot.get_current_recording_id()
    assert recording_id is not None

    sample_count = 12
    for frame_idx in range(sample_count):
        t = frame_idx * 0.05

        nc.log_joint_positions(make_joint_dict(frame_idx), timestamp=t)
        nc.log_joint_velocities(make_joint_dict(frame_idx, offset=1.0), timestamp=t)
        nc.log_joint_target_positions(make_target_dict(frame_idx), timestamp=t)
        nc.log_rgb(CAMERA_NAME, make_rgb(frame_idx), timestamp=t)
        nc.log_language(
            name=LANGUAGE_NAME,
            language=LANGUAGE_TEXT,
            timestamp=t,
        )
        nc.log_custom_1d(
            "frame_index",
            np.array([frame_idx], dtype=np.float32),
            timestamp=t,
        )

    nc.stop_recording(wait=True)

    recordings, sync_points = collect_sync_points(dataset_name)
    assert str(recordings[0].id) == str(recording_id)
    assert_expected_datatypes_present(
        sync_points,
        {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.RGB_IMAGES,
            DataType.LANGUAGE,
            DataType.CUSTOM_1D,
        },
    )


def test_vx300s_mixed_datatypes_wait_false_recording_is_not_dropped():
    robot_name = f"vx300s-mixed-nowait-robot-{uuid.uuid4().hex[:8]}"
    dataset_name = f"vx300s-mixed-nowait-dataset-{uuid.uuid4().hex[:8]}"

    nc.login()
    robot = nc.connect_robot(
        robot_name,
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )
    nc.create_dataset(dataset_name)

    nc.start_recording()
    recording_id = robot.get_current_recording_id()
    assert recording_id is not None

    sample_count = 20
    for frame_idx in range(sample_count):
        t = frame_idx * 0.04

        nc.log_joint_positions(make_joint_dict(frame_idx), timestamp=t)
        nc.log_joint_velocities(make_joint_dict(frame_idx, offset=0.75), timestamp=t)
        nc.log_joint_target_positions(make_target_dict(frame_idx), timestamp=t)
        nc.log_rgb(CAMERA_NAME, make_rgb(frame_idx), timestamp=t)
        nc.log_language(
            name=LANGUAGE_NAME,
            language=LANGUAGE_TEXT,
            timestamp=t,
        )
        nc.log_custom_1d(
            "frame_index",
            np.array([frame_idx], dtype=np.float32),
            timestamp=t,
        )

    nc.stop_recording(wait=False)

    recordings, sync_points = collect_sync_points(dataset_name, timeout_s=180.0)
    assert str(recordings[0].id) == str(recording_id)
    assert_expected_datatypes_present(
        sync_points,
        {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.RGB_IMAGES,
            DataType.LANGUAGE,
            DataType.CUSTOM_1D,
        },
    )


def test_vx300s_joint_schema_round_trips_through_sync():
    robot_name = f"vx300s-schema-robot-{uuid.uuid4().hex[:8]}"
    dataset_name = f"vx300s-schema-dataset-{uuid.uuid4().hex[:8]}"

    nc.login()
    nc.connect_robot(
        robot_name,
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )
    nc.create_dataset(dataset_name)

    nc.start_recording()

    for frame_idx in range(8):
        t = frame_idx * 0.1

        nc.log_joint_positions(make_joint_dict(frame_idx), timestamp=t)
        nc.log_joint_velocities(make_joint_dict(frame_idx, offset=0.5), timestamp=t)
        nc.log_joint_target_positions(make_target_dict(frame_idx), timestamp=t)
        nc.log_custom_1d(
            "frame_index",
            np.array([frame_idx], dtype=np.float32),
            timestamp=t,
        )

    nc.stop_recording(wait=True)

    _, sync_points = collect_sync_points(dataset_name)

    seen_joint_positions = False
    seen_joint_velocities = False
    seen_joint_targets = False

    for sync_point in sync_points:
        if DataType.JOINT_POSITIONS in sync_point.data:
            seen_joint_positions = True
            for joint_name in VX300S_JOINT_NAMES:
                assert joint_name in sync_point[DataType.JOINT_POSITIONS]

        if DataType.JOINT_VELOCITIES in sync_point.data:
            seen_joint_velocities = True
            for joint_name in VX300S_JOINT_NAMES:
                assert joint_name in sync_point[DataType.JOINT_VELOCITIES]

        if DataType.JOINT_TARGET_POSITIONS in sync_point.data:
            seen_joint_targets = True
            for joint_name in VX300S_JOINT_NAMES:
                assert joint_name in sync_point[DataType.JOINT_TARGET_POSITIONS]

    assert seen_joint_positions
    assert seen_joint_velocities
    assert seen_joint_targets


def test_vx300s_extended_datatypes_recording_survives():
    robot_name = f"vx300s-extended-robot-{uuid.uuid4().hex[:8]}"
    dataset_name = f"vx300s-extended-dataset-{uuid.uuid4().hex[:8]}"

    nc.login()
    robot = nc.connect_robot(
        robot_name,
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )
    nc.create_dataset(dataset_name)

    nc.start_recording()
    recording_id = robot.get_current_recording_id()
    assert recording_id is not None

    sample_count = 10
    for frame_idx in range(sample_count):
        t = frame_idx * 0.05

        nc.log_joint_positions(make_joint_dict(frame_idx), timestamp=t)
        nc.log_joint_velocities(make_joint_dict(frame_idx, offset=0.5), timestamp=t)
        nc.log_joint_target_positions(make_target_dict(frame_idx), timestamp=t)
        nc.log_joint_torques(make_joint_dict(frame_idx, offset=1.0), timestamp=t)
        nc.log_rgb(CAMERA_NAME, make_rgb(frame_idx), timestamp=t)
        nc.log_language(
            name=LANGUAGE_NAME,
            language=LANGUAGE_TEXT,
            timestamp=t,
        )
        nc.log_custom_1d(
            "frame_index",
            np.array([frame_idx], dtype=np.float32),
            timestamp=t,
        )
        nc.log_parallel_gripper_open_amount(
            name="gripper",
            value=0.5 + 0.01 * frame_idx,
            timestamp=t,
        )

    nc.stop_recording(wait=True)

    recordings, sync_points = collect_sync_points(dataset_name, timeout_s=180.0)
    assert str(recordings[0].id) == str(recording_id)
    assert_expected_datatypes_present(
        sync_points,
        {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.JOINT_TORQUES,
            DataType.RGB_IMAGES,
            DataType.LANGUAGE,
            DataType.CUSTOM_1D,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        },
    )
