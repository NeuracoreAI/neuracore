"""Utility functions for importer."""

import re

from neuracore_types.importer.config import VisualJointInputTypeConfig
from neuracore_types.importer.transform import Unnormalize
from neuracore_types.nc_data import DatasetImportConfig, DataType

from neuracore.core.robot import JointInfo
from neuracore.importer.core.exceptions import ConfigValidationError


def parse_storage_size(value: str) -> int:
    """Parse a human-readable storage size to bytes.

    Accepts a number followed by a unit (case-insensitive). Supported units:
    kb, mb, gb (binary: 1024-based). Optional space between number and unit.

    Examples: "10gb", "500 mb", "1kb", "2 GB"

    Raises:
        ValueError: If the string does not match the expected format.
    """
    value = value.strip()
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([kmg]b)$", value, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Invalid storage size '{value}'. "
            "Expected a number followed by kb, mb, or gb (e.g. 10gb, 500mb)."
        )
    num_str, unit = match.groups()
    num = float(num_str)
    if num <= 0:
        raise ValueError(f"Storage size must be positive, got '{value}'.")
    unit_factor = {"kb": 1024, "mb": 1024**2, "gb": 1024**3}
    return int(num * unit_factor[unit.lower()])


def populate_robot_info(
    dataconfig: DatasetImportConfig, robot_info: dict[str, JointInfo]
) -> DatasetImportConfig:
    """Populate the dataset import config with the robot info."""
    for data_type, import_config in dataconfig.data_import_config.items():
        if (
            data_type == DataType.VISUAL_JOINT_POSITIONS
            and import_config.format.visual_joint_input_type
            == VisualJointInputTypeConfig.GRIPPER
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
