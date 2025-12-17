"""This module provides utilities for parsing and merging SynchronizedPoint data."""

<<<<<<< HEAD
from typing import cast

from neuracore_types import (
    DATA_TYPE_TO_NC_DATA_CLASS,
    DataType,
    NCData,
    NCDataUnion,
=======
from neuracore_types import (
    Custom1DData,
    DataType,
    DepthCameraData,
    EndEffectorPoseData,
    JointData,
    LanguageData,
    NCDataUnion,
    ParallelGripperOpenAmountData,
    PointCloudData,
    PoseData,
    RGBCameraData,
>>>>>>> 24731bf (get things working)
    RobotStreamTrack,
    SynchronizedPoint,
)
from pydantic import ValidationError

from neuracore.core.utils.image_string_encoder import ImageStringEncoder


def parse_sync_point(
    message_data: str, track_details: RobotStreamTrack
) -> SynchronizedPoint:
    """Parse a JSON message into a SynchronizedPoint based on track details.

    Args:
        message_data: The JSON string containing the data.
        track_details: RobotStreamTrack object describing the data.

    Returns:
        SynchronizedPoint: A SynchronizedPoint object containing the parsed data.

    Raises:
        ValueError: If the track data_type is unsupported or data validation fails.
    """
    try:
<<<<<<< HEAD
        data_type: DataType = track_details.data_type
        label: str = track_details.label

        # Get the appropriate data class from the mapping
        data_class: type[NCData] | None = DATA_TYPE_TO_NC_DATA_CLASS.get(data_type)
        if data_class is None:
            raise ValueError(f"Unsupported track data_type: {data_type}")

        # Parse the JSON data using the appropriate class
        data: NCData = data_class.model_validate_json(message_data)
        data = cast(NCDataUnion, data)

        # Decode image data
        if data_type in (DataType.RGB_IMAGES, DataType.DEPTH_IMAGES):
            data.frame = ImageStringEncoder.decode_image(data.frame)

        return SynchronizedPoint(
            timestamp=data.timestamp,
            data={data_type: {label: data}},
        )

=======
        dt = track_details.data_type
        label = track_details.label
        if dt in (
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.JOINT_TARGET_POSITIONS,
        ):
            joint_data = JointData.model_validate_json(message_data)
            return SynchronizedPoint(
                timestamp=joint_data.timestamp,
                data={dt: {label: joint_data}},
            )
        if dt == DataType.LANGUAGE:
            language_data = LanguageData.model_validate_json(message_data)
            return SynchronizedPoint(
                timestamp=language_data.timestamp,
                data={dt: {label: language_data}},
            )
        if dt == DataType.RGB_IMAGES:
            camera_data = RGBCameraData.model_validate_json(message_data)
            camera_data.frame = ImageStringEncoder.decode_image(camera_data.frame)
            camera_id = f"{dt.value}_{label}"
            return SynchronizedPoint(
                timestamp=camera_data.timestamp,
                data={dt: {camera_id: camera_data}},
            )
        if dt == DataType.DEPTH_IMAGES:
            camera_data = DepthCameraData.model_validate_json(message_data)
            camera_data.frame = ImageStringEncoder.decode_image(camera_data.frame)
            camera_id = f"{dt.value}_{label}"
            return SynchronizedPoint(
                timestamp=camera_data.timestamp,
                data={dt: {camera_id: camera_data}},
            )
        if dt == DataType.END_EFFECTOR_POSES:
            end_effector_poses = EndEffectorPoseData.model_validate_json(message_data)
            return SynchronizedPoint(
                timestamp=end_effector_poses.timestamp,
                data={dt: {label: end_effector_poses}},
            )
        if dt == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
            parallel_gripper_open_amounts = (
                ParallelGripperOpenAmountData.model_validate_json(message_data)
            )
            return SynchronizedPoint(
                timestamp=parallel_gripper_open_amounts.timestamp,
                data={dt: {label: parallel_gripper_open_amounts}},
            )
        if dt == DataType.POINT_CLOUDS:
            point_cloud = PointCloudData.model_validate_json(message_data)
            return SynchronizedPoint(
                timestamp=point_cloud.timestamp,
                data={dt: {label: point_cloud}},
            )
        if dt == DataType.CUSTOM_1D:
            custom_data = Custom1DData.model_validate_json(message_data)
            return SynchronizedPoint(
                timestamp=custom_data.timestamp,
                data={dt: {label: custom_data}},
            )
        if dt == DataType.POSES:
            pose_data = PoseData.model_validate_json(message_data)
            return SynchronizedPoint(
                timestamp=pose_data.timestamp,
                data={dt: {label: pose_data}},
            )
        raise ValueError(f"Unsupported track data_type: {dt}")
>>>>>>> 24731bf (get things working)
    except ValidationError:
        raise ValueError("Invalid or unsupported data")


def merge_sync_points(*args: SynchronizedPoint) -> SynchronizedPoint:
    """Merge multiple SynchronizedPoint objects into a single SynchronizedPoint.

    Properties with later timestamps  will override earlier data.
    The timestamp of the combined sync point will be that of the latest sync point.
    If no sync points are provided, an empty SynchronizedPoint is returned.

    Args:
        *args: Variable number of SynchronizedPoint objects to merge.

    Returns:
        SynchronizedPoint: A new SynchronizedPoint object containing the merged data.
    """
    if len(args) == 0:
        return SynchronizedPoint()
<<<<<<< HEAD

    # Sort by timestamp so that later points override earlier ones.
    sorted_points = sorted(args, key=lambda x: x.timestamp)

    merged_synced_data: dict[DataType, dict[str, NCDataUnion]] = {}

    for sync_point in sorted_points:
        for data_type, values in sync_point.data.items():
            if data_type not in merged_synced_data:
                merged_synced_data[data_type] = {}
            merged_synced_data[data_type].update(values)

    return SynchronizedPoint(
        timestamp=sorted_points[-1].timestamp,
=======
    # Sort by timestamp so that later points override earlier ones.
    sorted_points = sorted(args, key=lambda x: x.timestamp)
    merged_synced_data: dict[DataType, dict[str, NCDataUnion]] = {}
    merged_robot_id = None
    for sync_point in sorted_points:
        if sync_point.robot_id is not None:
            merged_robot_id = sync_point.robot_id
        for dt, values in sync_point.data.items():
            if dt not in merged_synced_data:
                merged_synced_data[dt] = {}
            merged_synced_data[dt].update(values)
    return SynchronizedPoint(
        timestamp=sorted_points[-1].timestamp,
        robot_id=merged_robot_id,
>>>>>>> 24731bf (get things working)
        data=merged_synced_data,
    )
