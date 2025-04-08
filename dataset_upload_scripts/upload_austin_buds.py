"""Upload Austin Buds."""

import logging
import multiprocessing
import os
import time

from robot_utils import Robot, RobotType

import neuracore as nc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ruff: noqa: E402
import tensorflow_datasets as tfds

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TFDS_DATASET_NAME = "austin_buds_dataset_converted_externally_to_rlds"
NC_DATASET_NAME = "Austin Buds"
SPLIT = "all"
FREQUENCY = 20.0
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
    if "observation" not in step:
        raise ValueError("Observation does not contain 'observation' field")
    observation = step["observation"]

    nc.log_rgb("image", observation["image"].numpy(), timestamp=timestamp)
    nc.log_rgb("wrist_image", observation["wrist_image"].numpy(), timestamp=timestamp)

    # State format: [
    #   7x robot joint angles,
    #   1x gripper position,
    #   16x robot end-effector homogeneous matrix
    # ]
    state = observation["state"].numpy()
    joint_positions = state[:DOF].tolist()
    joint_positions_dict = dict(zip(ROBOT.joint_names[:DOF], joint_positions))

    # Process gripper
    gripper_width = state[DOF]
    visual_joint_positions_dict = {}
    for jname, (lower, upper) in ROBOT.joint_limits.items():
        if jname in ROBOT.joint_names[DOF:]:
            gripper_open_amount = ROBOT.gripper_open_width_to_open_amount(
                gripper_joint_names=ROBOT.joint_names[DOF : DOF + 2],
                gripper_open_width=gripper_width,
            )
            v = float(lower + (upper - lower) * gripper_open_amount)
            visual_joint_positions_dict[jname] = v

    nc.log_joint_positions(
        positions=joint_positions_dict,
        additional_urdf_positions=visual_joint_positions_dict,
        timestamp=timestamp,
    )

    gripper_open_amount = ROBOT.gripper_open_width_to_open_amount(
        gripper_joint_names=ROBOT.joint_names[DOF : DOF + 2],
        gripper_open_width=gripper_width,
    )
    gripper_open_amounts = {"gripper_open_amount": gripper_open_amount}
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
    nc.connect_robot(ROBOT.robot_info.name, shared=True, instance=start_idx)
    nc.get_dataset(NC_DATASET_NAME)
    ds = _get_tf_dataset()
    base_time = time.time()
    for episode in ds.skip(start_idx).take(end_idx - start_idx):
        steps = episode["steps"]
        try:
            nc.start_recording()
            for idx, step in enumerate(steps):
                timestamp = base_time + (idx / FREQUENCY)
                _record_step(step, timestamp)
            nc.stop_recording()
        except Exception as e:
            logger.error(f"Error processing episode {start_idx} to {end_idx}: {str(e)}")


def upload_austin_buds():
    """Upload the Austin Buds dataset."""

    logger.info("Starting Austin Buds dataset upload")
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
            "The robot is trying to solve a long-horizon kitchen task by "
            "picking up pot, placing the pot in a plate, and push them "
            "together using a picked-up tool."
        ),
        tags=["franka", "kitchen", "pot", "plate", "pushing"],
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
    upload_austin_buds()
