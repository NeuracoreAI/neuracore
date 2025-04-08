"""Upload ASU Table Top."""

import logging
import multiprocessing
import os
import time

import numpy as np
from robot_utils import Robot, RobotType

import neuracore as nc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ruff: noqa: E402
import tensorflow_datasets as tfds

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TFDS_DATASET_NAME = "asu_table_top_converted_externally_to_rlds"
NC_DATASET_NAME = "ASU Table Top"
SPLIT = "all"
FREQUENCY = 12.5
ROBOT = Robot(RobotType.UR5_2F85)

DOF = 6
JOINT_DIRECTION = np.array([1, 1, -1, 1, 1, 1])
JOINT_OFFSETS = np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0])
INVERTED_JOINTS = [
    "left_inner_finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
]


def _get_tf_dataset():
    return tfds.load(
        TFDS_DATASET_NAME,
        split=SPLIT,
        data_dir=os.getenv(
            "TFDS_DATA_DIR", os.path.expanduser("~/tensorflow_datasets")
        ),
    )


def _record_step(step: dict, timestamp: float) -> None:
    if "observation" not in step:
        raise ValueError("Observation does not contain 'state' field")
    observation = step["observation"]

    nc.log_rgb("image", observation["image"].numpy(), timestamp=timestamp)

    # [6x robot joint angles, 1x gripper position]
    state = observation["state"].numpy()
    joint_positions = (JOINT_DIRECTION * state[:DOF]) + JOINT_OFFSETS
    joint_positions_dict = dict(zip(ROBOT.joint_names[:DOF], joint_positions))
    # visual_joint_positions_dict = dict(zip(ROBOT.joint_names[DOF:], joint_positions))
    visual_joint_positions_dict = {}
    gripper_closed_amount = np.clip(state[-1], 0.0, 1.0).item()
    (lower, upper) = ROBOT.joint_limits["finger_joint"]
    v = float(lower + (upper - lower) * gripper_closed_amount)
    for jname, (lower, upper) in ROBOT.joint_limits.items():
        if jname in ROBOT.joint_names[DOF:]:
            visual_joint_positions_dict[jname] = v * (
                -1 if jname in INVERTED_JOINTS else 1
            )
    nc.log_joint_positions(
        positions=joint_positions_dict,
        additional_urdf_positions=visual_joint_positions_dict,
        timestamp=timestamp,
    )

    state_vel = observation["state_vel"].numpy().tolist()
    joint_velocities_dict = dict(zip(ROBOT.joint_names[:DOF], state_vel[:DOF]))

    nc.log_joint_velocities(velocities=joint_velocities_dict, timestamp=timestamp)

    gripper_open_amounts = {"gripper_open_amount": 1.0 - gripper_closed_amount}
    nc.log_gripper_data(
        open_amounts=gripper_open_amounts,
        timestamp=timestamp,
    )

    nc.log_language(
        language=step["language_instruction"].numpy().decode("utf-8"),
        timestamp=timestamp,
    )


def _process_episode_chunks(
    start_idx: int,
    end_idx: int,
) -> None:
    """Process a chunk of episodes. Runs within a process"""
    nc.login()
    # We will use the start_idx as the instance number
    nc.connect_robot(ROBOT.robot_info.name, shared=True, instance=start_idx)
    nc.get_dataset(NC_DATASET_NAME)
    ds = _get_tf_dataset()
    base_time = time.time()
    for episode in ds.skip(start_idx).take(end_idx - start_idx):
        steps = episode["steps"]
        nc.start_recording()
        for idx, step in enumerate(steps):
            timestamp = base_time + (idx / FREQUENCY)
            _record_step(step, timestamp)
        nc.stop_recording(wait=True)


def upload_asu_table_top():
    """Upload the Austin Buds dataset."""

    logger.info("Starting ASU Table Top dataset upload")
    nc.login()
    nc.connect_robot(
        robot_name=ROBOT.robot_info.name,
        urdf_path=ROBOT.robot_info.urdf_path,
        mjcf_path=ROBOT.robot_info.mjcf_path,
        # overwrite=True,
        shared=True,
    )
    nc.create_dataset(
        name=NC_DATASET_NAME,
        description=(
            "The robot interacts with a few objects on a table. "
            "It picks up, pushes forward, or rotates the objects."
        ),
        tags=["ur5", "table", "pushing", "picking", "rotating"],
        shared=True,
    )

    num_episodes = len(_get_tf_dataset())
    logger.info(f"Number of episodes: {num_episodes}")

    max_workers = max(2, int(os.cpu_count() * 0.8))
    logger.info(f"Using {max_workers} workers for uploading")
    multiprocessing.set_start_method("spawn", force=True)

    processes = []
    chunk_size = num_episodes // max_workers
    for i in range(max_workers):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < max_workers - 1 else num_episodes
        p = multiprocessing.Process(
            target=_process_episode_chunks, args=(start_idx, end_idx), daemon=False
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    upload_asu_table_top()
