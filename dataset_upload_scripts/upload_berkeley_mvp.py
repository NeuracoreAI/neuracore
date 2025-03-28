"""Upload Berkeley MVP dataset."""

import logging
import os

from neuracore.upload.converters.tfds_converter import TFDSConverter
from neuracore.upload.neura_uploader import DatasetInfo, NeuraUploader
from neuracore.upload.robot_utils import Robot, RobotType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOF = 7


def _process_joint_positions(
    robot: Robot, observation: dict
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract and process joint positions from Berkeley MVP observation"""
    if "joint_pos" not in observation:
        raise ValueError("Observation does not contain 'joint_pos' field")
    if "gripper" not in observation:
        raise ValueError("Observation does not contain 'gripper' field")

    joint_positions = observation["joint_pos"].numpy().tolist()
    gripper_closed_amount = float(observation["gripper"].numpy())

    # Process arm joints
    joint_positions_dict = dict(zip(robot.joint_names[:DOF], joint_positions))

    # Process gripper joints
    gripper_joint_positions_dict = {}
    for idx, jname in enumerate(robot.joint_names[DOF:]):
        v = (
            robot.joint_limits[jname][0]
            + (robot.joint_limits[jname][1] - robot.joint_limits[jname][0])
            * gripper_closed_amount
        )
        gripper_joint_positions_dict[jname] = float(v)

    return joint_positions_dict, gripper_joint_positions_dict


def _process_gripper_open_amounts(robot: Robot, observation: dict) -> list[float]:
    """Extract gripper open amount from Berkeley MVP observation"""
    if "gripper" not in observation:
        raise ValueError("Observation does not contain 'gripper' field")

    gripper_closed_amount = float(observation["gripper"].numpy())
    return [1.0 - gripper_closed_amount]  # Convert to open amount


def upload_berkeley_mvp():
    """Upload the Berkeley MVP dataset."""

    logger.info("Starting Berkeley MVP dataset upload")
    robot = Robot(RobotType.XARM7)
    dataset_info = DatasetInfo(
        name="Berkeley MVP",
        description=(
            "Basic motor control tasks (reach, push, pick) on table top and "
            "toy environments (toy kitchen, toy fridge)."
        ),
        tags=["xarm7", "toy", "kitchen", "fridge"],
        frequency=5.0,
        visual_joint_names=robot.joint_names[DOF:],
    )
    robot._pinnochio_robot = None

    converter = TFDSConverter(
        tfds_name="berkeley_mvp_converted_externally_to_rlds",
        robot=robot,
        dataset_info=dataset_info,
        process_joint_positions_fn=_process_joint_positions,
        process_gripper_open_amounts_fn=_process_gripper_open_amounts,
        rgb_keys=["hand_image"],
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
    upload_berkeley_mvp()
