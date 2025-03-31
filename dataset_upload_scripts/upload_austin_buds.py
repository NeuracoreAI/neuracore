"""Upload Ausin Buds."""

import logging
import os

from neuracore.upload.converters.tfds_converter import TFDSConverter
from neuracore.upload.neura_uploader import DatasetInfo, NeuraUploader
from neuracore.upload.robot_utils import Robot, RobotType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _process_joint_positions(
    robot: Robot, observation: dict
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract and process joint positions from Austin Buds observation"""
    if "state" not in observation:
        raise ValueError("Observation does not contain 'state' field")
    # State format: [
    #   7x robot joint angles,
    #   1x gripper position,
    #   16x robot end-effector homogeneous matrix
    # ]
    state = observation["state"].numpy()
    joint_positions = state[:7].tolist()
    gripper_width = state[7]
    gripper_joint_positions_dict = robot.process_gripper_state(
        gripper_joint_names=robot.joint_names[7:9], gripper_open_width=gripper_width
    )
    joint_positions_dict = dict(zip(robot.joint_names[:7], joint_positions))
    return joint_positions_dict, gripper_joint_positions_dict


def _process_gripper_open_amounts(robot: Robot, observation: dict) -> list[float]:
    """Extract and process joint positions from Austin Buds observation"""
    if "state" not in observation:
        raise ValueError("Observation does not contain 'state' field")
    # State format: [
    #   7x robot joint angles,
    #   1x gripper position,
    #   16x robot end-effector homogeneous matrix
    # ]
    state = observation["state"].numpy()
    gripper_width = state[7]
    gripper_open_amount = robot.gripper_open_width_to_open_amount(
        gripper_joint_names=robot.joint_names[7:9], gripper_open_width=gripper_width
    )
    return [gripper_open_amount]


def upload_austin_buds():
    """Upload the Austin Buds dataset."""

    logger.info("Starting Austin Buds dataset upload")
    robot = Robot(RobotType.FRANKA)
    dataset_info = DatasetInfo(
        name="Austin Buds",
        description="The robot is trying to solve a long-horizon kitchen task by "
        "picking up pot, placing the pot in a plate, and push them "
        "together using a picked-up tool.",
        tags=["franka", "kitchen", "pot", "plate", "pushing"],
        frequency=20.0,
        visual_joint_names=robot.joint_names[7:],
    )
    robot._pinnochio_robot = None

    converter = TFDSConverter(
        tfds_name="austin_buds_dataset_converted_externally_to_rlds",
        robot=robot,
        dataset_info=dataset_info,
        process_joint_positions_fn=_process_joint_positions,
        process_gripper_open_amounts_fn=_process_gripper_open_amounts,
        rgb_keys=["image", "wrist_image"],
    )

    uploader = NeuraUploader(
        converter=converter,
        max_workers=max(2, os.cpu_count()),
        chunk_size=2,
        retry_attempts=3,
        verbose=True,
    )

    def report_progress(progress):
        logger.info(f"Upload progress: {progress:.1%}")

    dataset_id = uploader.upload(
        episodes_range=(0, 10), progress_callback=report_progress
    )

    logger.info(f"Upload complete! Dataset ID: {dataset_id}")
    return dataset_id


if __name__ == "__main__":
    upload_austin_buds()
