"""This module provides utilities for parsing and merging SyncPoint data."""

from typing import Any

from neuracore_types import (
    CameraData,
    CustomData,
    EndEffectorData,
    EndEffectorPoseData,
    JointData,
    LanguageData,
    ParallelGripperOpenAmountData,
    PointCloudData,
    PoseData,
    RobotStreamTrack,
    SyncPoint,
    TrackKind,
)
from pydantic import ValidationError

from neuracore.core.utils.image_string_encoder import ImageStringEncoder


def parse_sync_point(message_data: str, track_details: RobotStreamTrack) -> SyncPoint:
    """Parse a JSON message into a SyncPoint based on track details.

    Args:
        message_data: The JSON string containing the data.
        track_details: RobotStreamTrack object describing the data.

    Returns:
        SyncPoint: A SyncPoint object containing the parsed data.

    Raises:
        ValueError: If the track kind is unsupported or data validation fails.
    """
    try:
        if track_details.kind == TrackKind.JOINTS:
            joint_data = JointData.model_validate_json(message_data)
            return SyncPoint.model_validate(
                {track_details.label: joint_data, "timestamp": joint_data.timestamp}
            )
        if track_details.kind == TrackKind.LANGUAGE:
            language_data = LanguageData.model_validate_json(message_data)
            return SyncPoint(
                language_data=language_data, timestamp=language_data.timestamp
            )

        if track_details.kind in (TrackKind.DEPTH, TrackKind.RGB):
            camera_data = CameraData.model_validate_json(message_data)

            camera_data.frame = ImageStringEncoder.decode_image(camera_data.frame)

            camera_id = f"{track_details.kind.value}_{track_details.label}"
            return SyncPoint.model_validate({
                f"{track_details.kind.value}_images": {camera_id: camera_data},
                "timestamp": camera_data.timestamp,
            })
        if track_details.kind == TrackKind.GRIPPER:
            end_effectors = EndEffectorData.model_validate_json(message_data)
            return SyncPoint(
                end_effectors=end_effectors, timestamp=end_effectors.timestamp
            )
        if track_details.kind == TrackKind.END_EFFECTOR_POSE:
            end_effector_poses = EndEffectorPoseData.model_validate_json(message_data)
            return SyncPoint(
                end_effector_poses={track_details.label: end_effector_poses},
                timestamp=end_effector_poses.timestamp,
            )

        if track_details.kind == TrackKind.PARALLEL_GRIPPER_OPEN_AMOUNT:
            parallel_gripper_open_amounts = (
                ParallelGripperOpenAmountData.model_validate_json(message_data)
            )
            return SyncPoint(
                parallel_gripper_open_amounts={
                    track_details.label: parallel_gripper_open_amounts
                },
                timestamp=parallel_gripper_open_amounts.timestamp,
            )

        if track_details.kind == TrackKind.POINT_CLOUD:
            point_cloud = PointCloudData.model_validate_json(message_data)
            return SyncPoint(
                point_clouds={track_details.label: point_cloud},
                timestamp=point_cloud.timestamp,
            )

        if track_details.kind == TrackKind.CUSTOM:
            custom_data = CustomData.model_validate_json(message_data)
            return SyncPoint(
                custom_data={track_details.label: custom_data},
                timestamp=custom_data.timestamp,
            )

        if track_details.kind == TrackKind.POSE:
            pose_data = PoseData.model_validate_json(message_data)
            # This doesn't match the schema but it is what the sync data does
            return SyncPoint(poses=pose_data, timestamp=pose_data.timestamp)

        raise ValueError(f"Unsupported track kind: {track_details.kind}")
    except ValidationError:
        raise ValueError("Invalid or unsupported data")


def merge_sync_points(*args: SyncPoint) -> SyncPoint:
    """Merge multiple SyncPoint objects into a single SyncPoint.

    Properties with later timestamps  will override earlier data.
    The timestamp of the combined sync point will be that of the latest sync point.

    If no sync points are provided, an empty SyncPoint is returned.

    Args:
        *args: Variable number of SyncPoint objects to merge.

    Returns:
        SyncPoint: A new SyncPoint object containing the merged data.
    """
    if len(args) == 0:
        return SyncPoint()

    sorted_points = sorted(args, key=lambda x: x.timestamp)

    merged_sync_point_dict: dict[str, Any] = {}

    for sync_point in sorted_points:
        # Joint Positions, Velocities, Torques, Target Positions
        if sync_point.joint_positions is not None:
            if "joint_positions" not in merged_sync_point_dict:
                merged_sync_point_dict["joint_positions"] = {}
            merged_sync_point_dict["joint_positions"].update(sync_point.joint_positions)

        if sync_point.joint_velocities is not None:
            if "joint_velocities" not in merged_sync_point_dict:
                merged_sync_point_dict["joint_velocities"] = {}
            merged_sync_point_dict["joint_velocities"].update(
                sync_point.joint_velocities
            )
        if sync_point.joint_torques is not None:
            if "joint_torques" not in merged_sync_point_dict:
                merged_sync_point_dict["joint_torques"] = {}
            merged_sync_point_dict["joint_torques"].update(sync_point.joint_torques)

        if sync_point.joint_target_positions is not None:
            if "joint_target_positions" not in merged_sync_point_dict:
                merged_sync_point_dict["joint_target_positions"] = {}
            merged_sync_point_dict["joint_target_positions"].update(
                sync_point.joint_target_positions
            )

        # Camera Data
        if sync_point.rgb_images is not None:
            if "rgb_images" not in merged_sync_point_dict:
                merged_sync_point_dict["rgb_images"] = {}
            merged_sync_point_dict["rgb_images"].update(sync_point.rgb_images)

        if sync_point.depth_images is not None:
            if "depth_images" not in merged_sync_point_dict:
                merged_sync_point_dict["depth_images"] = {}
            merged_sync_point_dict["depth_images"].update(sync_point.depth_images)

        # Point Clouds
        if sync_point.point_clouds is not None:
            if "point_clouds" not in merged_sync_point_dict:
                merged_sync_point_dict["point_clouds"] = {}
            merged_sync_point_dict["point_clouds"].update(sync_point.point_clouds)

        # End Effector Data
        if sync_point.end_effectors is not None:
            if "end_effectors" not in merged_sync_point_dict:
                merged_sync_point_dict["end_effectors"] = {}
            merged_sync_point_dict["end_effectors"].update(sync_point.end_effectors)

        # End Effector Poses
        if sync_point.end_effector_poses is not None:
            if "end_effector_poses" not in merged_sync_point_dict:
                merged_sync_point_dict["end_effector_poses"] = {}
            merged_sync_point_dict["end_effector_poses"].update(
                sync_point.end_effector_poses
            )

        # Parallel Gripper Open Amounts
        if sync_point.parallel_gripper_open_amounts is not None:
            if "parallel_gripper_open_amount" not in merged_sync_point_dict:
                merged_sync_point_dict["parallel_gripper_open_amount"] = {}
            merged_sync_point_dict["parallel_gripper_open_amount"].update(
                sync_point.parallel_gripper_open_amounts
            )
        # Pose Data
        if sync_point.poses is not None:
            if "poses" not in merged_sync_point_dict:
                merged_sync_point_dict["poses"] = {}
            merged_sync_point_dict["poses"].update(sync_point.poses)

        # Language Data
        if sync_point.language_data is not None:
            if "language_data" not in merged_sync_point_dict:
                merged_sync_point_dict["language_data"] = {}
            merged_sync_point_dict["language_data"].update(sync_point.language_data)

        # Custom Data
        if sync_point.custom_data is not None:
            if "custom_data" not in merged_sync_point_dict:
                merged_sync_point_dict["custom_data"] = {}
            merged_sync_point_dict["custom_data"].update(sync_point.custom_data)

    return SyncPoint(**merged_sync_point_dict, timestamp=sorted_points[-1].timestamp)
