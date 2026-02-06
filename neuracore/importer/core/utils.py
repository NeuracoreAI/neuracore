"""Utility functions for importer."""

from neuracore_types.importer.config import VisualJointTypeConfig
from neuracore_types.importer.transform import Unnormalize
from neuracore_types.nc_data import DatasetImportConfig, DataType

from neuracore.core.robot import JointInfo
from neuracore.importer.core.exceptions import ConfigValidationError


def populate_robot_info(
    dataconfig: DatasetImportConfig, robot_info: dict[str, JointInfo]
) -> DatasetImportConfig:
    """Populate the dataset import config with the robot info."""
    for data_type, import_config in dataconfig.data_import_config.items():
        if (
            data_type == DataType.VISUAL_JOINT_POSITIONS
            and import_config.format.visual_joint_type == VisualJointTypeConfig.GRIPPER
        ):
            for item in import_config.mapping:
                if item.name is not None:
                    if item.name not in robot_info:
                        raise ConfigValidationError(
                            f"Joint {item.name} not found in robot model."
                        )
                    else:
                        for transform in item.transforms.transforms:
                            if type(transform) == Unnormalize:
                                joint_limit_lower = robot_info[item.name].limits.lower
                                joint_limit_upper = robot_info[item.name].limits.upper
                                if (
                                    joint_limit_lower is not None
                                    and joint_limit_upper is not None
                                ):
                                    transform.min = joint_limit_lower
                                    transform.max = joint_limit_upper
                                else:
                                    raise ConfigValidationError(
                                        f"Joint limits for {item.name} required to log "
                                        f"the visual joint positions from the gripper "
                                        f"open amounts but are not present in the "
                                        f"robot model."
                                    )
    return dataconfig
