"""Tests for data ordering functionality.

This module contains tests to verify that the data ordering utilities
work correctly and ensure consistent ordering across sync points.
"""

import time
from typing import cast

import numpy as np
import pytest
from neuracore_types import (
    CameraData,
    CustomData,
    DataType,
    JointData,
    LanguageData,
    PointCloudData,
    PoseData,
    SynchronizedEpisode,
    SynchronizedPoint,
)


@pytest.fixture
def sync_point_unordered() -> SynchronizedPoint:
    """Create a test sync point with deliberately unordered dictionary keys."""
    return SynchronizedPoint(
        timestamp=time.time(),
        data={
            # Joint positions with unordered keys
            DataType.JOINT_POSITIONS: {
                "default": JointData(
                    values={"joint_2": 1.0, "joint_1": 0.5, "joint_3": 1.5},
                    additional_values={"extra_2": 2.0, "extra_1": 1.0},
                )
            },
            # Joint velocities with unordered keys
            DataType.JOINT_VELOCITIES: {
                "default": JointData(
                    values={"joint_3": 0.1, "joint_1": 0.2, "joint_2": 0.3}
                )
            },
            # Pose data with unordered keys
            DataType.POSES: {
                "pose_2": PoseData(pose=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]),
                "pose_1": PoseData(pose=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]),
            },
            # RGB camera data with unordered keys
            DataType.RGB_IMAGES: {
                "camera_3": CameraData(frame=np.zeros((224, 224, 3))),
                "camera_1": CameraData(frame=np.zeros((224, 224, 3))),
                "camera_2": CameraData(frame=np.zeros((224, 224, 3))),
            },
            # Depth camera data with unordered keys
            DataType.DEPTH_IMAGES: {
                "depth_2": CameraData(frame=np.zeros((224, 224))),
                "depth_1": CameraData(frame=np.zeros((224, 224))),
            },
            # Point cloud data with unordered keys
            DataType.POINT_CLOUDS: {
                "lidar_2": PointCloudData(
                    points=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
                ),
                "lidar_1": PointCloudData(
                    points=np.array([[7, 8, 9]], dtype=np.float16)
                ),
            },
            # Custom data with unordered keys
            DataType.CUSTOM: {
                "sensor_z": CustomData(data=[1, 2, 3]),
                "sensor_a": CustomData(data=[4, 5, 6]),
            },
            # Language data
            DataType.LANGUAGE: {"default": LanguageData(text="Pick up the object")},
        },
    )


def _verify_sync_point_ordering(sync_point_unordered: SynchronizedPoint):
    """Test that sync point ordering works correctly."""
    ordered_sync = sync_point_unordered.order()

    # Verify that all keys are now sorted
    assert DataType.JOINT_POSITIONS in ordered_sync.data
    jd = cast(JointData, list(ordered_sync[DataType.JOINT_POSITIONS].values())[0])
    assert list(jd.values.keys()) == [
        "joint_1",
        "joint_2",
        "joint_3",
    ]

    # Verify joint velocities ordering
    assert DataType.JOINT_VELOCITIES in ordered_sync.data
    jv = cast(JointData, list(ordered_sync[DataType.JOINT_VELOCITIES].values())[0])
    assert list(jv.values.keys()) == [
        "joint_1",
        "joint_2",
        "joint_3",
    ]

    # Verify poses ordering (both the dict keys and pose names should be sorted)
    assert DataType.POSES in ordered_sync.data
    assert list(ordered_sync[DataType.POSES].keys()) == ["pose_1", "pose_2"]

    # Verify RGB images ordering
    assert DataType.RGB_IMAGES in ordered_sync.data
    assert list(ordered_sync[DataType.RGB_IMAGES].keys()) == [
        "camera_1",
        "camera_2",
        "camera_3",
    ]

    # Verify depth images ordering
    assert DataType.DEPTH_IMAGES in ordered_sync.data
    assert list(ordered_sync[DataType.DEPTH_IMAGES].keys()) == ["depth_1", "depth_2"]

    # Verify point clouds ordering
    assert DataType.POINT_CLOUDS in ordered_sync.data
    assert list(ordered_sync[DataType.POINT_CLOUDS].keys()) == ["lidar_1", "lidar_2"]

    # Verify custom data ordering
    assert DataType.CUSTOM in ordered_sync.data
    assert list(ordered_sync[DataType.CUSTOM].keys()) == ["sensor_a", "sensor_z"]


def test_sync_point_ordering(sync_point_unordered: SynchronizedPoint):
    """Test that sync point ordering works correctly."""
    _verify_sync_point_ordering(sync_point_unordered)


def test_synced_data_ordering(sync_point_unordered: SynchronizedPoint):
    """Test that synced data ordering works for multiple sync points."""
    # Create multiple unordered sync points
    sync_points = [sync_point_unordered for _ in range(3)]

    # Modify timestamps to be different
    for i, sync_point in enumerate(sync_points):
        sync_point.timestamp = time.time() + i

    # Create synced data
    synced_data = SynchronizedEpisode(
        observations=sync_points,
        start_time=sync_points[0].timestamp,
        end_time=sync_points[-1].timestamp,
        robot_id="robot1",
    )

    ordered_synced_data = synced_data.order()

    for frame in ordered_synced_data.observations:
        _verify_sync_point_ordering(frame)
