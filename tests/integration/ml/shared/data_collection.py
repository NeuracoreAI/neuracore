"""Demonstration data collection helpers for ML integration tests."""

import logging
import os
import sys
import time

import numpy as np

import neuracore as nc
from neuracore.core.data.dataset import Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.join(_THIS_DIR, "..", "..", "..", "..", "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.append(_EXAMPLES_DIR)

# ruff: noqa: E402
from common.rollout_utils import rollout_policy
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH

from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_dataset_ready,
)
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running

logger = logging.getLogger(__name__)


def wait_for_dataset_recording_count(
    dataset_name: str,
    expected_recordings: int,
    timeout_seconds: int = 120,
    poll_seconds: int = 5,
) -> Dataset:
    deadline = time.time() + timeout_seconds
    last_count = None
    last_error = None

    while time.time() < deadline:
        try:
            dataset = nc.get_dataset(name=dataset_name)
            last_count = len(dataset)
            if last_count == expected_recordings:
                return dataset
            last_error = None
        except Exception as e:
            last_error = e

        time.sleep(poll_seconds)

    if last_error is not None:
        raise AssertionError(
            f"Dataset {dataset_name!r} did not become queryable within "
            f"{timeout_seconds} seconds; last error: {last_error}"
        )
    raise AssertionError(
        f"Dataset {dataset_name!r} had {last_count} recordings after "
        f"{timeout_seconds} seconds; expected {expected_recordings}"
    )


def collect_demo_data(
    robot_name: str,
    dataset_name: str,
    *,
    joint_names: tuple[str, ...] | list[str],
    gripper_names: list[str],
    language_label: str,
    nc_cam_name: str,
    pose_sensor_name: str,
    num_episodes: int = 3,
    instance_id: int = 0,
    episode_length_multiplier: int = 1,
    num_cameras: int = 1,
    frequency: float = 20,
    timestamp_jitter_frac: float = 0.05,
) -> Dataset:
    """Collect scripted demonstrations and log them to neuracore.

    Use different instances for different tests since they are run in parallel.
    Increase episode_length_multiplier to inflate episode length by repeating
    the rollout trajectory steps.
    Increase num_cameras to log multiple RGB streams per timestep.
    """
    assert (
        episode_length_multiplier >= 1
    ), f"episode_length_multiplier must be >= 1, got {episode_length_multiplier}"
    assert num_cameras >= 1, f"num_cameras must be >= 1, got {num_cameras}"

    with online_daemon_running():
        assert_exactly_one_daemon_pid()
        nc.connect_robot(
            robot_name=robot_name,
            instance=instance_id,
            urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
            overwrite=False,
        )
        dataset = nc.create_dataset(name=dataset_name)
        for ep_idx in range(num_episodes):
            logger.info(f"Collecting episode {ep_idx + 1}/{num_episodes}")
            action_traj = rollout_policy()
            expanded_action_traj = [
                action_dict
                for action_dict in action_traj
                for _ in range(episode_length_multiplier)
            ]
            nc.start_recording(robot_name=robot_name, instance=instance_id)
            t = time.time()
            timestamp_rng = np.random.default_rng(ep_idx)
            for frame_idx, action_dict in enumerate(expanded_action_traj):
                dt = 1.0 / frequency
                t += dt * float(
                    timestamp_rng.uniform(
                        1.0 - timestamp_jitter_frac, 1.0 + timestamp_jitter_frac
                    )
                )
                joint_positions = {
                    k: v for k, v in action_dict.items() if "gripper" not in k
                }
                joint_torques = {
                    name: float(0.01 * ((index + frame_idx) % 5))
                    for index, name in enumerate(joint_names)
                }
                joint_velocities = {
                    name: float(0.05 * ((index + frame_idx) % 7))
                    for index, name in enumerate(joint_names)
                }
                gripper_open_amounts = {
                    name: float(0.25 + 0.5 * ((frame_idx % 2) == 0))
                    for name in gripper_names
                }
                pose = np.array([0.1 + frame_idx * 0.001, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
                img = np.zeros((84, 84, 3), dtype=np.uint8)
                img.fill(50 + frame_idx % 200)

                nc.log_joint_positions(
                    positions=joint_positions,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_joint_target_positions(
                    target_positions=joint_positions,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_joint_velocities(
                    velocities=joint_velocities,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_joint_torques(
                    torques=joint_torques,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_parallel_gripper_open_amounts(
                    values=gripper_open_amounts,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_parallel_gripper_target_open_amounts(
                    values=gripper_open_amounts,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_pose(
                    name=pose_sensor_name,
                    pose=pose,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_language(
                    name=language_label,
                    language="pick and place",
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_rgb(
                    name=nc_cam_name,
                    rgb=img,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
            nc.stop_recording(wait=False, robot_name=robot_name, instance=instance_id)
            wait_for_dataset_ready(
                dataset_name,
                expected_recording_count=ep_idx + 1,
                timeout_s=500,
                poll_interval_s=5,
            )
            logger.info(
                f"Episode {ep_idx + 1} recorded ({len(expanded_action_traj)} frames)"
            )
    return dataset
