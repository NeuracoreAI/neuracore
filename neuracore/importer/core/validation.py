"""Data validation functions for importer."""

from typing import Any

from neuracore_types import JointData
from neuracore_types.importer.config import (
    ImageConventionConfig,
    JointPositionTypeConfig,
    LanguageConfig,
    PoseConfig,
    RotationConfig,
)
from neuracore_types.importer.data_config import DataFormat
from neuracore_types.nc_data import DATA_TYPE_TO_NC_DATA_CLASS, DatasetImportConfig

from neuracore.core.robot import JointInfo
from neuracore.importer.core.exceptions import (
    ConfigValidationError,
    DataValidationError,
    DataValidationWarning,
)

JOINT_DATA_TYPES = [
    dt
    for dt in DATA_TYPE_TO_NC_DATA_CLASS
    if DATA_TYPE_TO_NC_DATA_CLASS[dt] == JointData
]


def validate_rgb_images(data: Any, format: DataFormat) -> None:
    """Validate RGB image data.

    Args:
        data: The RGB image data to validate.
        format: The data format configuration.

    Raises:
        DataValidationError: If the data does not match the expected format.
    """
    if len(data.shape) != 3:
        raise DataValidationError(
            f"RGB image data must have 3 dimensions. "
            f"Data of shape {data.shape} has {len(data.shape)} dimensions."
        )
    if format.image_convention == ImageConventionConfig.CHANNELS_LAST:
        if data.shape[2] != 3:
            raise DataValidationError(
                f"Config requires CHANNELS_LAST convention, "
                f"but data has shape {data.shape}"
            )
    elif format.image_convention == ImageConventionConfig.CHANNELS_FIRST:
        if data.shape[0] != 3:
            raise DataValidationError(
                f"Config requires CHANNELS_FIRST convention, "
                f"but data has shape {data.shape}"
            )
    if format.normalized_pixel_values:
        if data.min() < 0 or data.max() > 1:
            raise DataValidationWarning(
                f"Config requires normalized pixel values, "
                f"but data has values outside the range [0, 1]. "
                f"{data.min()} to {data.max()}"
            )
    else:
        if data.min() < 0 or data.max() > 255:
            raise DataValidationWarning(
                f"Config requires unnormalized pixel values, "
                f"but data has values outside the range [0, 255]. "
                f"{data.min()} to {data.max()}"
            )


def validate_depth_images(data: Any) -> None:
    """Validate depth image data.

    Args:
        data: The depth image data to validate.

    Raises:
        DataValidationError: If the data does not match the expected format.
    """
    if len(data.shape) != 2:
        raise DataValidationError(
            f"Depth image data must have 2 dimensions. "
            f"Data of shape {data.shape} has {len(data.shape)} dimensions."
        )


def validate_point_clouds(data: Any) -> None:
    """Validate point cloud data.

    Args:
        data: The point cloud data to validate.

    Raises:
        DataValidationError: If the data does not match the expected format.
    """
    if len(data.shape) != 2:
        raise DataValidationError(
            f"Point cloud data must have 2 dimensions. "
            f"Data of shape {data.shape} has {len(data.shape)} dimensions."
        )
    if data.shape[1] != 3:
        raise DataValidationError(
            f"Point cloud data must have 3 columns. "
            f"Data of shape {data.shape} has {data.shape[1]} columns."
        )


def validate_language(data: Any, format: DataFormat) -> None:
    """Validate language data.

    Args:
        data: The language data to validate.
        format: The data format configuration.

    Raises:
        DataValidationError: If the data does not match the expected format.
    """
    if format.language_type == LanguageConfig.STRING and not isinstance(data, str):
        raise DataValidationError(
            f"Config requires language type STRING, but data has type {type(data)}"
        )


def validate_joint_positions(
    data: Any, name: str, joint_info: dict[str, JointInfo]
) -> None:
    """Validate joint positions data.

    Args:
        data: The joint positions data to validate.
        name: The name of the joint.
        joint_info: The joint info to validate against.
    """
    if name not in joint_info:
        raise DataValidationError(f"Joint {name} not found in robot model.")

    if (
        joint_info[name].limits.lower is not None
        and data < joint_info[name].limits.lower
    ):
        raise DataValidationWarning(
            f"Position {data} is below the lower limit {joint_info[name].limits.lower}."
        )
    if (
        joint_info[name].limits.upper is not None
        and data > joint_info[name].limits.upper
    ):
        raise DataValidationWarning(
            f"Position {data} is above the upper limit {joint_info[name].limits.upper}."
        )


def validate_joint_velocities(
    data: Any, name: str, joint_info: dict[str, JointInfo]
) -> None:
    """Validate joint velocities data.

    Args:
        data: The joint velocities data to validate.
        name: The name of the joint.
        joint_info: The joint info to validate against.
    """
    if name not in joint_info:
        raise DataValidationError(f"Joint {name} not found in robot model.")

    if joint_info[name].limits.velocity is not None:
        if abs(data) > joint_info[name].limits.velocity:
            raise DataValidationWarning(
                f"Velocity {data} is outside the limits "
                f"{joint_info[name].limits.velocity}."
            )


def validate_joint_torques(
    data: Any, name: str, joint_info: dict[str, JointInfo]
) -> None:
    """Validate joint torques data.

    Args:
        data: The joint torques data to validate.
        name: The name of the joint.
        joint_info: The joint info to validate against.
    """
    if name not in joint_info:
        raise DataValidationError(f"Joint {name} not found in robot model.")

    if joint_info[name].limits.effort is not None:
        if abs(data) > joint_info[name].limits.effort:
            raise DataValidationWarning(
                f"Torque {data} is outside the limits {joint_info[name].limits.effort}."
            )


def validate_poses(data: Any, format: DataFormat) -> None:
    """Validate pose data.

    Args:
        data: The pose data to validate.
        format: The data format configuration.

    Raises:
        DataValidationError: If the data does not match the expected format.
    """
    if len(data.shape) != 1:
        raise DataValidationError(
            f"Pose data must have 1 dimension. "
            f"Data of shape {data.shape} has {len(data.shape)} dimensions."
        )
    if format.pose_type == PoseConfig.MATRIX and data.shape[0] != 16:
        raise DataValidationError(
            f"Config requires MATRIX pose type, expected shape [16], "
            f"but data has shape {data.shape}"
        )
    elif format.pose_type == PoseConfig.POSITION_ORIENTATION:
        if format.orientation.type == RotationConfig.QUATERNION:
            if data.shape[0] != 7:
                raise DataValidationError(
                    f"Config requires QUATERNION orientation type, "
                    f"expected shape [7], but data has shape {data.shape}"
                )
        elif (
            format.orientation.type == RotationConfig.EULER
            or format.orientation.type == RotationConfig.AXIS_ANGLE
        ):
            if data.shape[0] != 6:
                raise DataValidationError(
                    f"Config requires EULER orientation type, "
                    f"expected shape [6], but data has shape {data.shape}"
                )
        elif format.orientation.type == RotationConfig.MATRIX:
            if data.shape[0] != 9:
                raise DataValidationError(
                    f"Config requires MATRIX orientation type, "
                    f"expected shape [9], but data has shape {data.shape}"
                )


def validate_dataset_config_against_robot_model(
    dataset_config: DatasetImportConfig, joint_info: dict[str, JointInfo]
) -> None:
    """Validate dataset config against robot model.

    Ensure joint names in dataset config match robot model.

    Args:
        dataset_config: The dataset config to validate.
        joint_info: The joint info to validate against.

    Raises:
        ConfigValidationError: If joint names in config do not match robot model.
    """
    for data_type, import_config in dataset_config.data_import_config.items():
        if data_type in JOINT_DATA_TYPES:
            # No need to check joint names if converting from TCP
            if (
                import_config.format.joint_position_type
                == JointPositionTypeConfig.CUSTOM
            ):
                for item in import_config.mapping:
                    if item.name is not None:
                        if item.name not in joint_info:
                            raise ConfigValidationError(
                                f"Joint {item.name} not found in robot model."
                            )
