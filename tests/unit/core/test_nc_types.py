"""Tests for data ordering functionality.

This module contains tests to verify that the data ordering utilities
work correctly and ensure consistent ordering across sync points.
"""

import time

import numpy as np
import pytest

from neuracore.core.nc_types import (
    CameraData,
    CustomData,
    EndEffectorData,
    JointData,
    LanguageData,
    PointCloudData,
    PoseData,
    SyncedData,
    SyncPoint,
)


@pytest.fixture
def sync_point_unordered() -> SyncPoint:
    """Create a test sync point with deliberately unordered dictionary keys."""
    return SyncPoint(
        timestamp=time.time(),
        # Joint data with unordered keys
        joint_positions=JointData(
            values={"joint_2": 1.0, "joint_1": 0.5, "joint_3": 1.5},
            additional_values={"extra_2": 2.0, "extra_1": 1.0},
        ),
        joint_velocities=JointData(
            values={"joint_3": 0.1, "joint_1": 0.2, "joint_2": 0.3}
        ),
        # End effector data with unordered keys
        end_effectors=EndEffectorData(
            open_amounts={"gripper_2": 0.8, "gripper_1": 0.2}
        ),
        # Pose data with unordered keys
        poses=PoseData(
            pose={
                "pose_2": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                "pose_1": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            }
        ),
        # Camera data with unordered keys
        rgb_images={
            "camera_3": CameraData(frame=np.zeros((224, 224, 3))),
            "camera_1": CameraData(frame=np.zeros((224, 224, 3))),
            "camera_2": CameraData(frame=np.zeros((224, 224, 3))),
        },
        depth_images={
            "depth_2": CameraData(frame=np.zeros((224, 224))),
            "depth_1": CameraData(frame=np.zeros((224, 224))),
        },
        # Point cloud data with unordered keys
        point_clouds={
            "lidar_2": PointCloudData(points=[[1, 2, 3], [4, 5, 6]]),
            "lidar_1": PointCloudData(points=[[7, 8, 9]]),
        },
        # Custom data with unordered keys
        custom_data={
            "sensor_z": CustomData(data=[1, 2, 3]),
            "sensor_a": CustomData(data=[4, 5, 6]),
        },
        # Language data (no ordering needed)
        language_data=LanguageData(text="Pick up the object"),
    )


def _verify_sync_point_ordering(sync_point_unordered: SyncPoint):
    """Test that sync point ordering works correctly."""
    ordered_sync = sync_point_unordered.order()

    # Verify that all keys are now sorted
    assert ordered_sync.joint_positions is not None
    assert list(ordered_sync.joint_positions.values.keys()) == [
        "joint_1",
        "joint_2",
        "joint_3",
    ]
    assert ordered_sync.joint_velocities is not None
    assert list(ordered_sync.joint_velocities.values.keys()) == [
        "joint_1",
        "joint_2",
        "joint_3",
    ]
    assert ordered_sync.end_effectors is not None
    assert list(ordered_sync.end_effectors.open_amounts.keys()) == [
        "gripper_1",
        "gripper_2",
    ]
    assert ordered_sync.poses is not None
    assert list(ordered_sync.poses.pose.keys()) == ["pose_1", "pose_2"]
    assert ordered_sync.rgb_images is not None
    assert list(ordered_sync.rgb_images.keys()) == ["camera_1", "camera_2", "camera_3"]
    assert ordered_sync.depth_images is not None
    assert list(ordered_sync.depth_images.keys()) == ["depth_1", "depth_2"]
    assert ordered_sync.point_clouds is not None
    assert list(ordered_sync.point_clouds.keys()) == ["lidar_1", "lidar_2"]
    assert ordered_sync.custom_data is not None
    assert list(ordered_sync.custom_data.keys()) == ["sensor_a", "sensor_z"]

    # Verify joint additional values are sorted
    assert ordered_sync.joint_positions.additional_values is not None
    assert list(ordered_sync.joint_positions.additional_values.keys()) == [
        "extra_1",
        "extra_2",
    ]


def test_sync_point_ordering(sync_point_unordered: SyncPoint):
    """Test that sync point ordering works correctly."""
    _verify_sync_point_ordering(sync_point_unordered)


def test_synced_data_ordering(sync_point_unordered: SyncPoint):
    """Test that synced data ordering works for multiple sync points."""
    # Create multiple unordered sync points
    sync_points = [sync_point_unordered for _ in range(3)]

    # Modify timestamps to be different
    for i, sync_point in enumerate(sync_points):
        sync_point.timestamp = time.time() + i

    # Create synced data
    synced_data = SyncedData(
        frames=sync_points,
        start_time=sync_points[0].timestamp,
        end_time=sync_points[-1].timestamp,
        robot_id="robot1",
    )

    ordered_synced_data = synced_data.order()

    for frame in ordered_synced_data.frames:
        _verify_sync_point_ordering(frame)
