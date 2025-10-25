"""Upload TACO Play."""

import logging
import multiprocessing
import os
import time

import numpy as np
import tensorflow_datasets as tfds
from robot_utils import Robot, RobotType

import neuracore as nc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TFDS_DATASET_NAME = "taco_play"
NC_DATASET_NAME = "Freiburg Franka Play - (TACO Play)"
SPLIT = "all"
FREQUENCY = 15.0
ROBOT = Robot(RobotType.FRANKA)

DOF = 7


def _get_tf_dataset():
    return tfds.load(
        TFDS_DATASET_NAME,
        split=SPLIT,
        data_dir=os.getenv(
            "TFDS_DATA_DIR", os.path.expanduser("~/tensorflow_datasets")
        ),
    )


def _record_step(step: dict, timestamp: float) -> None:
    obs = step["observation"]

    # RGB
    if "rgb_static" in obs:
        nc.log_rgb("rgb_static", obs["rgb_static"].numpy(), timestamp=timestamp)
    if "rgb_gripper" in obs:
        nc.log_rgb("rgb_gripper", obs["rgb_gripper"].numpy(), timestamp=timestamp)

    # Depth
    if "depth_static" in obs:
        d = np.nan_to_num(
            obs["depth_static"].numpy().astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        nc.log_depth("depth_static", d, timestamp=timestamp)
    if "depth_gripper" in obs:
        d = np.nan_to_num(
            obs["depth_gripper"].numpy().astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        nc.log_depth("depth_gripper", d, timestamp=timestamp)

    # Robot state: [0:3] pos, [3:6] euler,
    # [6] gripper width, [7:14] 7 joints, [14] gripper action
    robot_obs = obs["robot_obs"].numpy()
    joint_positions = robot_obs[7:14].tolist()
    joint_positions_dict = dict(zip(ROBOT.joint_names[:DOF], joint_positions))

    # Visual gripper joints
    gripper_width = float(robot_obs[6])
    open_amt = ROBOT.gripper_open_width_to_open_amount(
        gripper_joint_names=ROBOT.joint_names[DOF : DOF + 2],
        gripper_open_width=gripper_width,
    )
    visual_joints = {}
    for jname, (lower, upper) in ROBOT.joint_limits.items():
        if jname in ROBOT.joint_names[DOF:]:
            visual_joints[jname] = float(lower + (upper - lower) * open_amt)

    nc.log_joint_positions(
        positions=joint_positions_dict,
        additional_urdf_positions=visual_joints,
        timestamp=timestamp,
    )
    nc.log_gripper_data(
        open_amounts={"gripper_open_amount": open_amt},
        timestamp=timestamp,
    )

    # Language
    lang = None
    if "natural_language_instruction" in obs:
        lang = obs["natural_language_instruction"].numpy().decode("utf-8")
    elif "structured_language_instruction" in obs:
        lang = obs["structured_language_instruction"].numpy().decode("utf-8")
    if lang:
        nc.log_language(language=lang, timestamp=timestamp)


def _process_episode_chunks(start_idx: int, end_idx: int) -> None:
    """Process a chunk of episodes. Runs within a process."""
    nc.login()
    nc.connect_robot(ROBOT.robot_info.name, shared=True, instance=start_idx)
    nc.get_dataset(NC_DATASET_NAME)

    ds = _get_tf_dataset().skip(start_idx).take(end_idx - start_idx)

    for ep_idx, episode in enumerate(ds, start=start_idx):
        steps = episode["steps"]
        started = False
        try:
            # Per-episode instance avoids recorder collisions
            nc.connect_robot(ROBOT.robot_info.name, shared=True, instance=ep_idx)
            base_time = time.time()
            nc.start_recording()
            started = True

            for i, step in enumerate(steps):
                ts = base_time + (i / FREQUENCY)
                _record_step(step, ts)

        except Exception:
            logger.error(f"Episode {ep_idx} failed.", exc_info=True)
        finally:
            if started:
                try:
                    nc.stop_recording(wait=True)
                except Exception as e:
                    logger.warning(f"Episode {ep_idx} stop warning: {e}")


def upload_taco_play():
    """Upload the TACO Play dataset."""
    logger.info("Starting TACO Play dataset upload")
    nc.login()
    nc.connect_robot(
        robot_name=ROBOT.robot_info.name,
        urdf_path=ROBOT.robot_info.urdf_path,
        mjcf_path=ROBOT.robot_info.mjcf_path,
        shared=True,
    )
    nc.create_dataset(
        name=NC_DATASET_NAME,
        description=(
            "The robot interacts with toy blocks, it pick and places them,"
            "stacks them, "
            "unstacks them, opens drawers, sliding doors and turns on LED"
            "lights by pushing buttons."
        ),
        tags=["franka", "play", "teleop", "table", "multi-view", "depth", "language"],
        shared=True,
    )

    num_episodes = len(_get_tf_dataset())
    logger.info(f"Number of episodes: {num_episodes}")
    max_workers = 1

    logger.info(f"Using {max_workers} workers for uploading")
    multiprocessing.set_start_method("spawn", force=True)

    processes = []
    chunk_size = max(1, num_episodes // max_workers)
    for i in range(max_workers):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < max_workers - 1 else num_episodes
        if start_idx >= num_episodes:
            break
        p = multiprocessing.Process(
            target=_process_episode_chunks, args=(start_idx, end_idx), daemon=False
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    upload_taco_play()
