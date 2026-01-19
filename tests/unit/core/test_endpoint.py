"""Tests for core endpoint functionality.

This module tests the DirectPolicy and other core endpoint classes.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from neuracore_types import (
    DATA_TYPE_TO_NC_DATA_CLASS,
    BatchedJointData,
    DataType,
    JointData,
    RGBCameraData,
    SynchronizedPoint,
)

from neuracore.core.endpoint import DirectPolicy


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model file path."""
    model_file = tmp_path / "model.nc.zip"
    model_file.touch()
    return model_file


@pytest.fixture
def mock_policy_inference():
    """Create a mock PolicyInference object."""
    mock_policy = MagicMock()
    mock_policy.model_input_order = {}
    mock_policy.model_output_order = {}
    mock_policy.return_value = {
        DataType.JOINT_TARGET_POSITIONS: {
            "joint1": BatchedJointData(value=torch.zeros((1, 3, 1))),
        }
    }
    return mock_policy


@pytest.fixture
def sample_sync_point_with_multiple_data_types():
    """Create a SynchronizedPoint with multiple data types."""
    return SynchronizedPoint(
        timestamp=1234567890.0,
        data={
            DataType.JOINT_POSITIONS: {
                "joint1": JointData(timestamp=1234567890.0, value=0.1),
                "joint2": JointData(timestamp=1234567890.0, value=0.2),
            },
            DataType.JOINT_VELOCITIES: {
                "joint1": JointData(timestamp=1234567890.0, value=0.01),
            },
            DataType.RGB_IMAGES: {
                "camera1": RGBCameraData(
                    timestamp=1234567890.0,
                    frame=np.zeros((100, 100, 3), dtype=np.uint8),
                    extrinsics=np.eye(4, dtype=np.float16),
                    intrinsics=np.eye(3, dtype=np.float16),
                ),
                "camera2": RGBCameraData(
                    timestamp=1234567890.0,
                    frame=np.ones((100, 100, 3), dtype=np.uint8),
                    extrinsics=np.eye(4, dtype=np.float16),
                    intrinsics=np.eye(3, dtype=np.float16),
                ),
            },
            DataType.JOINT_TORQUES: {
                "joint1": JointData(timestamp=1234567890.0, value=1.0),
            },
        },
    )


@patch("neuracore.ml.utils.policy_inference.PolicyInference")
def test_predict_filters_to_model_input_order_only(
    mock_policy_inference_class,
    mock_model_path,
    sample_sync_point_with_multiple_data_types,
):
    """Test _predict only includes data types from model_input_order."""
    # Setup: Model only expects JOINT_POSITIONS and RGB_IMAGES
    model_input_order = {
        DataType.JOINT_POSITIONS: ["joint1", "joint2"],
        DataType.RGB_IMAGES: ["camera1"],
    }
    model_output_order = {
        DataType.JOINT_TARGET_POSITIONS: ["joint1"],
    }

    # Create mock PolicyInference instance
    mock_policy = MagicMock()
    mock_policy.model_input_order = model_input_order
    mock_policy.return_value = {
        DataType.JOINT_TARGET_POSITIONS: {
            "joint1": BatchedJointData(value=torch.zeros((1, 3, 1))),
        }
    }
    mock_policy_inference_class.return_value = mock_policy

    # Create DirectPolicy
    policy = DirectPolicy(
        model_input_order=model_input_order,
        model_output_order=model_output_order,
        model_path=mock_model_path,
        org_id="test_org",
    )

    # Call _predict with sync point containing more data types than model expects
    policy._predict(sample_sync_point_with_multiple_data_types)

    # Verify that PolicyInference was called
    assert mock_policy.call_count == 1

    # Get the sync point that was passed to PolicyInference
    called_sync_point = mock_policy.call_args[0][0]

    # Verify only expected data types are present
    assert DataType.JOINT_POSITIONS in called_sync_point.data
    assert DataType.RGB_IMAGES in called_sync_point.data

    # Verify filtered out data types are NOT present
    assert DataType.JOINT_VELOCITIES not in called_sync_point.data
    assert DataType.JOINT_TORQUES not in called_sync_point.data

    # Verify the original sync point was mutated (data was filtered in place)
    assert DataType.JOINT_POSITIONS in sample_sync_point_with_multiple_data_types.data
    assert DataType.RGB_IMAGES in sample_sync_point_with_multiple_data_types.data
    assert (
        DataType.JOINT_VELOCITIES not in sample_sync_point_with_multiple_data_types.data
    )
    assert DataType.JOINT_TORQUES not in sample_sync_point_with_multiple_data_types.data

    # Verify the data content is preserved for filtered types
    assert (
        called_sync_point.data[DataType.JOINT_POSITIONS]["joint1"]
        == sample_sync_point_with_multiple_data_types.data[DataType.JOINT_POSITIONS][
            "joint1"
        ]
    )
    assert (
        called_sync_point.data[DataType.RGB_IMAGES]["camera1"]
        == sample_sync_point_with_multiple_data_types.data[DataType.RGB_IMAGES][
            "camera1"
        ]
    )


@patch("neuracore.ml.utils.policy_inference.PolicyInference")
@patch("neuracore.core.endpoint.get_latest_sync_point")
def test_predict_filters_when_sync_point_is_none(
    mock_get_latest_sync_point,
    mock_policy_inference_class,
    mock_model_path,
    sample_sync_point_with_multiple_data_types,
):
    """Test _predict filters as expected when sync_point is None."""
    # Setup: Model only expects JOINT_POSITIONS
    model_input_order = {
        DataType.JOINT_POSITIONS: ["joint1", "joint2"],
    }
    model_output_order = {
        DataType.JOINT_TARGET_POSITIONS: ["joint1"],
    }

    # Mock get_latest_sync_point to return sync point with multiple data types
    mock_get_latest_sync_point.return_value = sample_sync_point_with_multiple_data_types

    # Create mock PolicyInference instance
    mock_policy = MagicMock()
    mock_policy.model_input_order = model_input_order
    mock_policy.return_value = {
        DataType.JOINT_TARGET_POSITIONS: {
            "joint1": BatchedJointData(value=torch.zeros((1, 3, 1))),
        }
    }
    mock_policy_inference_class.return_value = mock_policy

    # Create DirectPolicy
    policy = DirectPolicy(
        model_input_order=model_input_order,
        model_output_order=model_output_order,
        model_path=mock_model_path,
        org_id="test_org",
    )

    # Call _predict with None sync point
    policy._predict(sync_point=None)

    # Verify get_latest_sync_point was called
    assert mock_get_latest_sync_point.call_count == 1

    # Verify that PolicyInference was called
    assert mock_policy.call_count == 1

    # Get the sync point that was passed to PolicyInference
    called_sync_point = mock_policy.call_args[0][0]

    # Verify only expected data types are present
    assert DataType.JOINT_POSITIONS in called_sync_point.data

    # Verify filtered out data types are NOT present
    assert DataType.JOINT_VELOCITIES not in called_sync_point.data
    assert DataType.RGB_IMAGES not in called_sync_point.data
    assert DataType.JOINT_TORQUES not in called_sync_point.data


@patch("neuracore.ml.utils.policy_inference.PolicyInference")
def test_predict_filters_when_sync_point_missing_expected_data_types(
    mock_policy_inference_class,
    mock_model_path,
):
    """Test _predict skips missing model-required data types gracefully."""
    # Setup: Model expects JOINT_POSITIONS and RGB_IMAGES
    model_input_order = {
        DataType.JOINT_POSITIONS: ["joint1"],
        DataType.RGB_IMAGES: ["camera1"],
    }
    model_output_order = {
        DataType.JOINT_TARGET_POSITIONS: ["joint1"],
    }

    # Create sync point with only JOINT_POSITIONS (missing RGB_IMAGES)
    sync_point = SynchronizedPoint(
        timestamp=1234567890.0,
        data={
            DataType.JOINT_POSITIONS: {
                "joint1": JointData(timestamp=1234567890.0, value=0.1),
            },
        },
    )

    # Create mock PolicyInference instance
    mock_policy = MagicMock()
    mock_policy.model_input_order = model_input_order
    mock_policy.return_value = {
        DataType.JOINT_TARGET_POSITIONS: {
            "joint1": BatchedJointData(value=torch.zeros((1, 3, 1))),
        }
    }
    mock_policy_inference_class.return_value = mock_policy

    # Create DirectPolicy
    policy = DirectPolicy(
        model_input_order=model_input_order,
        model_output_order=model_output_order,
        model_path=mock_model_path,
        org_id="test_org",
    )

    # Call _predict - should filter to only include JOINT_POSITIONS
    policy._predict(sync_point)

    # Get the sync point that was passed to PolicyInference
    called_sync_point = mock_policy.call_args[0][0]

    # Verify only available data type is present
    assert DataType.JOINT_POSITIONS in called_sync_point.data
    assert DataType.RGB_IMAGES not in called_sync_point.data


@patch("neuracore.ml.utils.policy_inference.PolicyInference")
def test_predict_filters_with_empty_model_input_order(
    mock_policy_inference_class,
    mock_model_path,
    sample_sync_point_with_multiple_data_types,
):
    """Test that _predict handles empty model_input_order correctly."""
    # Setup: Model expects no inputs (edge case)
    model_input_order = {}
    model_output_order = {
        DataType.JOINT_TARGET_POSITIONS: ["joint1"],
    }

    # Create mock PolicyInference instance
    mock_policy = MagicMock()
    mock_policy.model_input_order = model_input_order
    mock_policy.return_value = {
        DataType.JOINT_TARGET_POSITIONS: {
            "joint1": BatchedJointData(value=torch.zeros((1, 3, 1))),
        }
    }
    mock_policy_inference_class.return_value = mock_policy

    # Create DirectPolicy
    policy = DirectPolicy(
        model_input_order=model_input_order,
        model_output_order=model_output_order,
        model_path=mock_model_path,
        org_id="test_org",
    )

    # Call _predict
    policy._predict(sample_sync_point_with_multiple_data_types)

    # Get the sync point that was passed to PolicyInference
    called_sync_point = mock_policy.call_args[0][0]

    # Verify sync point has no data (all filtered out)
    assert len(called_sync_point.data) == 0


@patch("neuracore.ml.utils.policy_inference.PolicyInference")
def test_predict_filters_preserves_all_sensors_for_selected_data_types(
    mock_policy_inference_class,
    mock_model_path,
):
    """Test that filtering preserves all sensors/labels for selected data types."""
    # Setup: Model expects JOINT_POSITIONS with multiple joints
    model_input_order = {
        DataType.JOINT_POSITIONS: ["joint1", "joint2", "joint3"],
        DataType.RGB_IMAGES: ["camera1", "camera2"],
    }
    model_output_order = {
        DataType.JOINT_TARGET_POSITIONS: ["joint1"],
    }

    # Create sync point with multiple sensors per data type
    sync_point = SynchronizedPoint(
        timestamp=1234567890.0,
        data={
            DataType.JOINT_POSITIONS: {
                "joint1": JointData(timestamp=1234567890.0, value=0.1),
                "joint2": JointData(timestamp=1234567890.0, value=0.2),
                "joint3": JointData(timestamp=1234567890.0, value=0.3),
            },
            DataType.RGB_IMAGES: {
                "camera1": RGBCameraData(
                    timestamp=1234567890.0,
                    frame=np.zeros((100, 100, 3), dtype=np.uint8),
                    extrinsics=np.eye(4, dtype=np.float16),
                    intrinsics=np.eye(3, dtype=np.float16),
                ),
                "camera2": RGBCameraData(
                    timestamp=1234567890.0,
                    frame=np.ones((100, 100, 3), dtype=np.uint8),
                    extrinsics=np.eye(4, dtype=np.float16),
                    intrinsics=np.eye(3, dtype=np.float16),
                ),
            },
            DataType.JOINT_VELOCITIES: {
                "joint1": JointData(timestamp=1234567890.0, value=0.01),
            },
        },
    )

    # Create mock PolicyInference instance
    mock_policy = MagicMock()
    mock_policy.model_input_order = model_input_order
    mock_policy.return_value = {
        DataType.JOINT_TARGET_POSITIONS: {
            "joint1": BatchedJointData(value=torch.zeros((1, 3, 1))),
        }
    }
    mock_policy_inference_class.return_value = mock_policy

    # Create DirectPolicy
    policy = DirectPolicy(
        model_input_order=model_input_order,
        model_output_order=model_output_order,
        model_path=mock_model_path,
        org_id="test_org",
    )

    # Call _predict
    policy._predict(sync_point)

    # Get the sync point that was passed to PolicyInference
    called_sync_point = mock_policy.call_args[0][0]

    # Verify all sensors for selected data types are preserved
    assert len(called_sync_point.data[DataType.JOINT_POSITIONS]) == 3
    assert "joint1" in called_sync_point.data[DataType.JOINT_POSITIONS]
    assert "joint2" in called_sync_point.data[DataType.JOINT_POSITIONS]
    assert "joint3" in called_sync_point.data[DataType.JOINT_POSITIONS]

    assert len(called_sync_point.data[DataType.RGB_IMAGES]) == 2
    assert "camera1" in called_sync_point.data[DataType.RGB_IMAGES]
    assert "camera2" in called_sync_point.data[DataType.RGB_IMAGES]

    # Verify filtered out data type is not present
    assert DataType.JOINT_VELOCITIES not in called_sync_point.data


@patch("neuracore.ml.utils.policy_inference.PolicyInference")
@patch("neuracore.core.endpoint.get_latest_sync_point")
def test_predict_filters_multiple_streams_logged_scenario(
    mock_get_latest_sync_point,
    mock_policy_inference_class,
    mock_model_path,
):
    """Test filtering when multiple streams are logged but only a subset is selected.

    This simulates the real-world scenario where:
    1. Multiple data streams are logged (joint positions, velocities, images, etc.)
    2. Model only expects a subset (e.g., just joint positions and one camera)
    3. Only the expected subset should be passed to the model
    """
    # Setup: Model only expects JOINT_POSITIONS and one RGB camera
    model_input_order = {
        DataType.JOINT_POSITIONS: ["arm_joint1", "arm_joint2"],
        DataType.RGB_IMAGES: ["top_camera"],
    }
    model_output_order = {
        DataType.JOINT_TARGET_POSITIONS: ["arm_joint1", "arm_joint2"],
    }

    # Create sync point simulating multiple logged streams
    sync_point_with_all_streams = SynchronizedPoint(
        timestamp=1234567890.0,
        data={
            DataType.JOINT_POSITIONS: {
                "arm_joint1": JointData(timestamp=1234567890.0, value=0.1),
                "arm_joint2": JointData(timestamp=1234567890.0, value=0.2),
            },
            DataType.JOINT_VELOCITIES: {
                "arm_joint1": JointData(timestamp=1234567890.0, value=0.01),
            },
            DataType.JOINT_TORQUES: {
                "arm_joint1": JointData(timestamp=1234567890.0, value=1.0),
            },
            DataType.RGB_IMAGES: {
                "top_camera": RGBCameraData(
                    timestamp=1234567890.0,
                    frame=np.zeros((100, 100, 3), dtype=np.uint8),
                    extrinsics=np.eye(4, dtype=np.float16),
                    intrinsics=np.eye(3, dtype=np.float16),
                ),
                "side_camera": RGBCameraData(
                    timestamp=1234567890.0,
                    frame=np.ones((100, 100, 3), dtype=np.uint8),
                    extrinsics=np.eye(4, dtype=np.float16),
                    intrinsics=np.eye(3, dtype=np.float16),
                ),
            },
            DataType.DEPTH_IMAGES: {
                "top_camera": DATA_TYPE_TO_NC_DATA_CLASS[
                    DataType.DEPTH_IMAGES
                ].sample(),
            },
        },
    )

    # Mock get_latest_sync_point to return sync point with all streams
    mock_get_latest_sync_point.return_value = sync_point_with_all_streams

    # Create mock PolicyInference instance
    mock_policy = MagicMock()
    mock_policy.model_input_order = model_input_order
    mock_policy.return_value = {
        DataType.JOINT_TARGET_POSITIONS: {
            "arm_joint1": BatchedJointData(value=torch.zeros((1, 3, 1))),
            "arm_joint2": BatchedJointData(value=torch.zeros((1, 3, 1))),
        }
    }
    mock_policy_inference_class.return_value = mock_policy

    # Create DirectPolicy
    policy = DirectPolicy(
        model_input_order=model_input_order,
        model_output_order=model_output_order,
        model_path=mock_model_path,
        org_id="test_org",
    )

    # Call _predict with None (will use get_latest_sync_point)
    policy._predict(sync_point=None)

    # Verify get_latest_sync_point was called
    assert mock_get_latest_sync_point.call_count == 1

    # Verify that PolicyInference was called
    assert mock_policy.call_count == 1

    # Get the sync point that was passed to PolicyInference
    called_sync_point = mock_policy.call_args[0][0]

    # Verify only expected data types are present
    assert DataType.JOINT_POSITIONS in called_sync_point.data
    assert DataType.RGB_IMAGES in called_sync_point.data

    # Verify filtered out data types are NOT present
    assert DataType.JOINT_VELOCITIES not in called_sync_point.data
    assert DataType.JOINT_TORQUES not in called_sync_point.data
    assert DataType.DEPTH_IMAGES not in called_sync_point.data

    # Verify the original sync point was mutated
    assert DataType.JOINT_POSITIONS in sync_point_with_all_streams.data
    assert DataType.RGB_IMAGES in sync_point_with_all_streams.data
    assert DataType.JOINT_VELOCITIES not in sync_point_with_all_streams.data
    assert DataType.JOINT_TORQUES not in sync_point_with_all_streams.data
    assert DataType.DEPTH_IMAGES not in sync_point_with_all_streams.data

    # Verify all sensors for selected data types are preserved
    assert len(called_sync_point.data[DataType.JOINT_POSITIONS]) == 2
    assert "arm_joint1" in called_sync_point.data[DataType.JOINT_POSITIONS]
    assert "arm_joint2" in called_sync_point.data[DataType.JOINT_POSITIONS]

    assert (
        len(called_sync_point.data[DataType.RGB_IMAGES]) == 2
    )  # Both cameras preserved
    assert "top_camera" in called_sync_point.data[DataType.RGB_IMAGES]
    assert "side_camera" in called_sync_point.data[DataType.RGB_IMAGES]


@patch("neuracore.ml.utils.policy_inference.PolicyInference")
def test_predict_filters_with_single_data_type(
    mock_policy_inference_class,
    mock_model_path,
    sample_sync_point_with_multiple_data_types,
):
    """Test filtering when model only expects a single data type."""
    # Setup: Model only expects RGB_IMAGES
    model_input_order = {
        DataType.RGB_IMAGES: ["camera1", "camera2"],
    }
    model_output_order = {
        DataType.JOINT_TARGET_POSITIONS: ["joint1"],
    }

    # Create mock PolicyInference instance
    mock_policy = MagicMock()
    mock_policy.model_input_order = model_input_order
    mock_policy.return_value = {
        DataType.JOINT_TARGET_POSITIONS: {
            "joint1": BatchedJointData(value=torch.zeros((1, 3, 1))),
        }
    }
    mock_policy_inference_class.return_value = mock_policy

    # Create DirectPolicy
    policy = DirectPolicy(
        model_input_order=model_input_order,
        model_output_order=model_output_order,
        model_path=mock_model_path,
        org_id="test_org",
    )

    # Call _predict
    policy._predict(sample_sync_point_with_multiple_data_types)

    # Get the sync point that was passed to PolicyInference
    called_sync_point = mock_policy.call_args[0][0]

    # Verify only RGB_IMAGES is present
    assert DataType.RGB_IMAGES in called_sync_point.data
    assert len(called_sync_point.data) == 1

    # Verify all other data types are filtered out
    assert DataType.JOINT_POSITIONS not in called_sync_point.data
    assert DataType.JOINT_VELOCITIES not in called_sync_point.data
    assert DataType.JOINT_TORQUES not in called_sync_point.data
