"""Upload ASU Table Top."""

import logging
import os

import numpy as np

from neuracore.upload.converters.tfds_converter import TFDSConverter
from neuracore.upload.neura_uploader import DatasetInfo, NeuraUploader
from neuracore.upload.robot_utils import Robot, RobotType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOF = 6
JOINT_DIRECTION = np.array([1, 1, -1, 1, 1, 1])
JOINT_OFFSETS = np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0])
INVERTED_JOINTS = [
    "left_inner_finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
]


def _process_joint_positions(
    robot: Robot, observation: dict
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract and process joint positions from Austin Buds observation"""
    if "state" not in observation:
        raise ValueError("Observation does not contain 'state' field")

    # [6x robot joint angles, 1x gripper position]
    state = observation["state"].numpy()
    joint_positions = (JOINT_DIRECTION * state[:DOF]) + JOINT_OFFSETS
    joint_positions_dict = dict(zip(robot.joint_names[:DOF], joint_positions))

    # Gripper
    gripper_closed_amount = np.clip(state[-1], 0.0, 1.0)
    (lower, upper) = robot.joint_limits["finger_joint"]
    v = float(lower + (upper - lower) * gripper_closed_amount)
    gripper_joint_positions_dict = {}
    for jname, (lower, upper) in robot.joint_limits.items():
        if jname in robot.joint_names[DOF:]:
            gripper_joint_positions_dict[jname] = v * (
                -1 if jname in INVERTED_JOINTS else 1
            )
    return joint_positions_dict, gripper_joint_positions_dict


def _process_gripper_open_amounts(robot: Robot, observation: dict) -> float:
    """Extract and process joint positions from Austin Buds observation"""
    if "state" not in observation:
        raise ValueError("Observation does not contain 'state' field")
    # [6x robot joint angles, 1x gripper position]
    state = observation["state"].numpy()
    gripper_closed_amount = np.clip(float(state[-1]), 0.0, 1.0)
    return [1.0 - gripper_closed_amount]


def upload_asu_table_top():
    """Upload the Austin Buds dataset."""

    logger.info("Starting ASU Table Top dataset upload")
    robot = Robot(RobotType.UR5_2F85)
    dataset_info = DatasetInfo(
        name="ASU Table Top",
        description=(
            "The robot interacts with a few objects on a table. "
            "It picks up, pushes forward, or rotates the objects."
        ),
        tags=["ur5", "table", "pushing", "picking", "rotating"],
        frequency=12.5,
        visual_joint_names=robot.joint_names[DOF:],
    )
    robot._pinnochio_robot = None

    converter = TFDSConverter(
        tfds_name="asu_table_top_converted_externally_to_rlds",
        robot=robot,
        dataset_info=dataset_info,
        process_joint_positions_fn=_process_joint_positions,
        process_gripper_open_amounts_fn=_process_gripper_open_amounts,
        rgb_keys=["image"],
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
    upload_asu_table_top()
