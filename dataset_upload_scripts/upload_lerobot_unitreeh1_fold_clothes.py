"""Upload unitreeh1_fold_clothes dataset from lerobot to Neuracore."""

import logging
import multiprocessing
import os
import time

import numpy as np
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from robot_utils import Robot, RobotType

import neuracore as nc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LEROBOT_DATASET_NAME = "lerobot/unitreeh1_fold_clothes"
NC_DATASET_NAME = "UnitreeH1 Fold Clothes"
FREQUENCY = 50
ROBOT = Robot(RobotType.UNITREE_H1)


def _get_lerobot_dataset():
    """Get the LeRobot dataset and metadata"""
    # First fetch metadata to get info without downloading the entire dataset
    ds_meta = LeRobotDatasetMetadata(LEROBOT_DATASET_NAME)

    if ds_meta.fps != FREQUENCY:
        raise ValueError(
            f"Dataset FPS {ds_meta.fps} does not match expected FPS {FREQUENCY}"
        )

    logger.info(f"Dataset: {LEROBOT_DATASET_NAME}")
    logger.info(f"Total episodes: {ds_meta.total_episodes}")
    logger.info(f"Camera keys: {ds_meta.camera_keys}")

    # Load the actual dataset
    return LeRobotDataset(LEROBOT_DATASET_NAME), ds_meta


def _record_step(frame_data, timestamp: float, ds_meta) -> None:
    """Record a single frame to Neuracore"""
    # Log camera images
    for camera_key in ds_meta.camera_keys:
        if camera_key in frame_data:
            image = frame_data[camera_key].numpy()
            image = np.clip(image * 255, 0, 255).astype(
                np.uint8
            )  # Ensure image is in uint8 format
            image = np.transpose(
                image, (1, 2, 0)
            )  # Convert from [C, H, W] to [H, W, C]
            nc.log_rgb(camera_key, image, timestamp=timestamp)

    if "observation.state" in frame_data:
        state = frame_data["observation.state"]
        state = [float(v) for v in state.numpy()]
        assert len(state) == len(
            ROBOT.joint_names
        ), f"State len {len(state)} != joint names len {len(ROBOT.joint_names)}"
        joint_positions_dict = dict(zip(ROBOT.joint_names[: len(state)], state))
        nc.log_joint_positions(
            name="arm", positions=joint_positions_dict, timestamp=timestamp
        )

    # Log actions if available
    if "action" in frame_data:
        # TODO: We dont know what the actions represent, so skip
        frame_data["action"].numpy()

    # Log language instruction if available
    if frame_data["task"] and len(frame_data["task"]) > 0:
        nc.log_language(
            name="instruction", language=frame_data["task"], timestamp=timestamp
        )


def _process_episode_chunks(start_idx: int, end_idx: int) -> None:
    """Process a chunk of episodes. Runs within a process"""
    nc.login()
    # We will use the start_idx as the instance number
    dataset, ds_meta = _get_lerobot_dataset()
    nc.connect_robot(ROBOT.robot_info.name, shared=True, instance=start_idx)
    nc.get_dataset(NC_DATASET_NAME)

    base_time = time.time()

    for episode_idx in range(start_idx, end_idx):
        if episode_idx >= dataset.num_episodes:
            break

        # Get frame indices for this episode
        from_idx = dataset.episode_data_index["from"][episode_idx].item()
        to_idx = dataset.episode_data_index["to"][episode_idx].item()

        nc.start_recording()
        for frame_idx in range(from_idx, to_idx):
            frame_data = dataset[frame_idx]
            # Calculate timestamp based on index within episode
            relative_idx = frame_idx - from_idx
            timestamp = base_time + (relative_idx / FREQUENCY)
            _record_step(frame_data, timestamp, ds_meta)
        nc.stop_recording(wait=True)
        logger.info(f"Completed episode {episode_idx}")


def upload_unitreeh1_fold_clothes():
    """Upload the unitreeh1_fold_clothes dataset to Neuracore."""

    logger.info(f"Starting {LEROBOT_DATASET_NAME} dataset upload")

    # Get dataset info first to set up global variables
    dataset, ds_meta = _get_lerobot_dataset()

    # Login to Neuracore and create dataset
    nc.login()
    nc.connect_robot(
        robot_name=ROBOT.robot_info.name,
        urdf_path=ROBOT.robot_info.urdf_path,
        mjcf_path=ROBOT.robot_info.mjcf_path,
        shared=True,
    )

    # Create the dataset in Neuracore
    nc.create_dataset(
        name=NC_DATASET_NAME,
        description=(
            f"UnitreeH1 robot folding clothes. Imported from {LEROBOT_DATASET_NAME}."
            f" Contains {ds_meta.total_episodes} episodes of clothing manipulation."
        ),
        tags=["unitree", "humanoid", "folding", "clothes", "manipulation"],
        shared=True,
    )

    num_episodes = dataset.num_episodes
    logger.info(f"Number of episodes to upload: {num_episodes}")

    max_workers = max(2, int(os.cpu_count() * 0.8))
    logger.info(f"Using {max_workers} workers for uploading")
    multiprocessing.set_start_method("spawn", force=True)

    processes = []
    chunk_size = max(1, num_episodes // max_workers)
    for i in range(max_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_episodes)
        # If we've processed all episodes, don't start more workers
        if start_idx >= num_episodes:
            break
        p = multiprocessing.Process(
            target=_process_episode_chunks, args=(start_idx, end_idx), daemon=False
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    logger.info(f"Finished uploading {LEROBOT_DATASET_NAME} to Neuracore")


if __name__ == "__main__":
    upload_unitreeh1_fold_clothes()
