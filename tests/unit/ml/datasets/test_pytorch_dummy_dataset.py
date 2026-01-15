"""Tests for PytorchDummyDataset.

This module provides comprehensive testing for the synthetic data generation
capabilities of PytorchDummyDataset, including multi-modal data generation,
proper tensor shapes, masking, and collation functionality.
"""

import pytest
import torch
from neuracore_types import DataType

from neuracore.ml import BatchedTrainingSamples
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset


class TestPytorchDummyDataset:
    """Test suite for PytorchDummyDataset."""

    @pytest.fixture
    def basic_robot_data_spec(self):
        """Basic robot data specs for testing."""
        return {
            "input_spec": {
                "robot_0": {
                    DataType.JOINT_POSITIONS: ["joint_0", "joint_1", "joint_2"],
                    DataType.RGB_IMAGES: ["camera_0"],
                }
            },
            "output_spec": {
                "robot_0": {
                    DataType.JOINT_TARGET_POSITIONS: ["joint_0", "joint_1", "joint_2"],
                }
            },
        }

    @pytest.fixture
    def all_data_types_robot_spec(self):
        """All supported data types for comprehensive testing."""
        return {
            "input_spec": {
                "robot_0": {
                    DataType.JOINT_POSITIONS: ["joint_0", "joint_1"],
                    DataType.JOINT_VELOCITIES: ["joint_0", "joint_1"],
                    DataType.JOINT_TORQUES: ["joint_0", "joint_1"],
                    DataType.RGB_IMAGES: ["camera_0", "camera_1"],
                    DataType.DEPTH_IMAGES: ["depth_camera_0"],
                    DataType.POINT_CLOUDS: ["pointcloud_0"],
                    DataType.END_EFFECTOR_POSES: ["ee_0"],
                    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: ["gripper_0"],
                    DataType.POSES: ["pose_0"],
                    # TODO: Language not currently supported in dummy dataset
                    # DataType.LANGUAGE: ["instruction"],
                    DataType.CUSTOM_1D: ["sensor_0", "sensor_1"],
                }
            },
            "output_spec": {
                "robot_0": {
                    DataType.JOINT_TARGET_POSITIONS: ["joint_0", "joint_1"],
                    DataType.RGB_IMAGES: ["camera_0"],
                }
            },
        }

    def test_initialization_basic(self, basic_robot_data_spec):
        """Test basic dataset initialization."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec=basic_robot_data_spec["input_spec"],
            output_robot_data_spec=basic_robot_data_spec["output_spec"],
            num_samples=50,
            num_episodes=10,
        )

        assert len(dataset) == 50
        assert dataset.num_samples == 50
        assert dataset.output_prediction_horizon == 5  # default value

    def test_initialization_all_data_types(self, all_data_types_robot_spec):
        """Test initialization with all supported data types."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec=all_data_types_robot_spec["input_spec"],
            output_robot_data_spec=all_data_types_robot_spec["output_spec"],
            num_samples=20,
            output_prediction_horizon=8,
        )

        assert len(dataset) == 20
        assert dataset.output_prediction_horizon == 8

        # Check dataset statistics are properly initialized
        stats = dataset.dataset_statistics
        assert DataType.JOINT_POSITIONS in stats
        assert len(stats[DataType.JOINT_POSITIONS]) == 2  # 2 joints
        assert DataType.RGB_IMAGES in stats
        assert len(stats[DataType.RGB_IMAGES]) == 2  # 2 cameras

    def test_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        # No data types
        with pytest.raises(ValueError):
            PytorchDummyDataset(
                input_robot_data_spec={}, output_robot_data_spec={}, num_samples=10
            )

    def test_dataset_length(self):
        """Test dataset length functionality."""
        for num_samples in [1, 10, 100]:
            dataset = PytorchDummyDataset(
                input_robot_data_spec={
                    "robot_0": {DataType.JOINT_POSITIONS: ["joint_0"]}
                },
                output_robot_data_spec={
                    "robot_0": {DataType.JOINT_TARGET_POSITIONS: ["joint_0"]}
                },
                num_samples=num_samples,
            )
            assert len(dataset) == num_samples

    def test_sample_generation_basic(self, basic_robot_data_spec):
        """Test basic sample generation."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec=basic_robot_data_spec["input_spec"],
            output_robot_data_spec=basic_robot_data_spec["output_spec"],
            num_samples=10,
        )

        sample = dataset[0]

        # Check sample structure
        assert isinstance(sample, BatchedTrainingSamples)
        assert sample.inputs is not None
        assert sample.outputs is not None
        assert sample.inputs_mask is not None
        assert sample.outputs_mask is not None

    def test_joint_data_generation(self):
        """Test joint data generation and properties."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec={
                "robot_0": {
                    DataType.JOINT_POSITIONS: ["joint_0", "joint_1", "joint_2"],
                    DataType.JOINT_VELOCITIES: ["joint_0", "joint_1"],
                }
            },
            output_robot_data_spec={
                "robot_0": {
                    DataType.JOINT_TARGET_POSITIONS: ["joint_0", "joint_1", "joint_2"]
                }
            },
            num_samples=5,
        )

        sample = dataset[0]

        # Test input joint positions
        assert DataType.JOINT_POSITIONS in sample.inputs
        assert len(sample.inputs[DataType.JOINT_POSITIONS]) == 3  # 3 joints
        assert DataType.JOINT_POSITIONS in sample.inputs_mask
        assert sample.inputs_mask[DataType.JOINT_POSITIONS].shape == (3,)
        assert torch.all(sample.inputs_mask[DataType.JOINT_POSITIONS] == 1.0)

        # Test input joint velocities
        assert DataType.JOINT_VELOCITIES in sample.inputs
        assert len(sample.inputs[DataType.JOINT_VELOCITIES]) == 2  # 2 joints

        # Test output joint target positions
        assert DataType.JOINT_TARGET_POSITIONS in sample.outputs
        assert len(sample.outputs[DataType.JOINT_TARGET_POSITIONS]) == 3
        assert DataType.JOINT_TARGET_POSITIONS in sample.outputs_mask

    def test_image_data_generation(self):
        """Test RGB and depth image generation."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec={
                "robot_0": {
                    DataType.RGB_IMAGES: ["camera_0", "camera_1"],
                    DataType.DEPTH_IMAGES: ["depth_0"],
                }
            },
            output_robot_data_spec={"robot_0": {DataType.RGB_IMAGES: ["camera_0"]}},
            num_samples=3,
        )

        sample = dataset[0]

        # Test RGB images
        assert DataType.RGB_IMAGES in sample.inputs
        assert len(sample.inputs[DataType.RGB_IMAGES]) == 2  # 2 cameras
        assert DataType.RGB_IMAGES in sample.inputs_mask
        assert sample.inputs_mask[DataType.RGB_IMAGES].shape == (2,)
        assert torch.all(sample.inputs_mask[DataType.RGB_IMAGES] == 1.0)

        # Test depth images
        assert DataType.DEPTH_IMAGES in sample.inputs
        assert len(sample.inputs[DataType.DEPTH_IMAGES]) == 1  # 1 depth camera

    def test_point_cloud_generation(self):
        """Test point cloud data generation."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec={
                "robot_0": {DataType.POINT_CLOUDS: ["pointcloud_0"]}
            },
            output_robot_data_spec={
                "robot_0": {DataType.JOINT_TARGET_POSITIONS: ["joint_0"]}
            },
            num_samples=3,
        )

        sample = dataset[0]

        assert DataType.POINT_CLOUDS in sample.inputs
        assert len(sample.inputs[DataType.POINT_CLOUDS]) == 1
        assert DataType.POINT_CLOUDS in sample.inputs_mask
        assert sample.inputs_mask[DataType.POINT_CLOUDS].shape == (1,)
        assert torch.all(sample.inputs_mask[DataType.POINT_CLOUDS] == 1.0)

    def test_end_effector_pose_data_generation(self):
        """Test end-effector pose data generation."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec={"robot_0": {DataType.END_EFFECTOR_POSES: ["ee_0"]}},
            output_robot_data_spec={"robot_0": {DataType.END_EFFECTOR_POSES: ["ee_0"]}},
            num_samples=3,
        )

        sample = dataset[0]

        # Test input end-effector poses
        assert DataType.END_EFFECTOR_POSES in sample.inputs
        assert len(sample.inputs[DataType.END_EFFECTOR_POSES]) == 1
        assert DataType.END_EFFECTOR_POSES in sample.inputs_mask
        assert sample.inputs_mask[DataType.END_EFFECTOR_POSES].shape == (1,)
        assert torch.all(sample.inputs_mask[DataType.END_EFFECTOR_POSES] == 1.0)

        # Test output end-effector poses
        assert DataType.END_EFFECTOR_POSES in sample.outputs
        assert len(sample.outputs[DataType.END_EFFECTOR_POSES]) == 1

    def test_parallel_gripper_open_amount_data_generation(self):
        """Test parallel gripper open amount data generation."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec={
                "robot_0": {DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: ["gripper_0"]}
            },
            output_robot_data_spec={
                "robot_0": {
                    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: ["gripper_0"]
                }
            },
            num_samples=3,
        )

        sample = dataset[0]

        # Test input parallel gripper open amounts
        assert DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in sample.inputs
        assert len(sample.inputs[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]) == 1
        assert DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in sample.inputs_mask
        assert sample.inputs_mask[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS].shape == (1,)
        assert torch.all(
            sample.inputs_mask[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS] == 1.0
        )

        # Test output parallel gripper target open amounts
        assert DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS in sample.outputs
        assert len(sample.outputs[DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS]) == 1

    def test_pose_data_generation(self):
        """Test pose data generation."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec={"robot_0": {DataType.POSES: ["pose_0", "pose_1"]}},
            output_robot_data_spec={"robot_0": {DataType.POSES: ["pose_0"]}},
            num_samples=3,
        )

        sample = dataset[0]

        assert DataType.POSES in sample.inputs
        assert len(sample.inputs[DataType.POSES]) == 2
        assert DataType.POSES in sample.inputs_mask
        assert sample.inputs_mask[DataType.POSES].shape == (2,)
        assert torch.all(sample.inputs_mask[DataType.POSES] == 1.0)

    def test_deterministic_behavior(self):
        """Test that the dataset generates deterministic data."""
        # Create two identical datasets
        dataset1 = PytorchDummyDataset(
            input_robot_data_spec={"robot_0": {DataType.JOINT_POSITIONS: ["joint_0"]}},
            output_robot_data_spec={
                "robot_0": {DataType.JOINT_TARGET_POSITIONS: ["joint_0"]}
            },
            num_samples=5,
        )

        dataset2 = PytorchDummyDataset(
            input_robot_data_spec={"robot_0": {DataType.JOINT_POSITIONS: ["joint_0"]}},
            output_robot_data_spec={
                "robot_0": {DataType.JOINT_TARGET_POSITIONS: ["joint_0"]}
            },
            num_samples=5,
        )

        sample1 = dataset1[0]
        sample2 = dataset2[0]

        # Structure should be the same
        assert len(sample1.inputs[DataType.JOINT_POSITIONS]) == len(
            sample2.inputs[DataType.JOINT_POSITIONS]
        )
        assert len(sample1.outputs[DataType.JOINT_TARGET_POSITIONS]) == len(
            sample2.outputs[DataType.JOINT_TARGET_POSITIONS]
        )

    def test_collate_fn_basic(self, basic_robot_data_spec):
        """Test basic collation functionality."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec=basic_robot_data_spec["input_spec"],
            output_robot_data_spec=basic_robot_data_spec["output_spec"],
            num_samples=10,
        )

        # Get multiple samples
        samples = [dataset[i] for i in range(3)]

        # Test collation
        batched = dataset.collate_fn(samples)

        assert isinstance(batched, BatchedTrainingSamples)
        assert batched.inputs is not None
        assert batched.outputs is not None
        assert batched.batch_size == 3

        # Check batch dimensions
        for joint_data in batched.inputs[DataType.JOINT_POSITIONS]:
            # Check that tensors have batch dimension
            for attr_name in vars(joint_data):
                attr_value = getattr(joint_data, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    assert attr_value.shape[0] == 3  # batch size

    def test_collate_fn_images(self):
        """Test collation with image data."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec={
                "robot_0": {DataType.RGB_IMAGES: ["camera_0", "camera_1"]}
            },
            output_robot_data_spec={"robot_0": {DataType.RGB_IMAGES: ["camera_0"]}},
            num_samples=5,
        )

        samples = [dataset[i] for i in range(2)]
        batched = dataset.collate_fn(samples)

        # Check that images are batched properly
        assert DataType.RGB_IMAGES in batched.inputs
        assert len(batched.inputs[DataType.RGB_IMAGES]) == 2  # 2 cameras

        # Each camera should have batch dimension
        for camera_data in batched.inputs[DataType.RGB_IMAGES]:
            for attr_name in vars(camera_data):
                attr_value = getattr(camera_data, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    assert attr_value.shape[0] == 2  # batch size

    def test_error_handling(self):
        """Test error handling in dataset operations."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec={"robot_0": {DataType.JOINT_POSITIONS: ["joint_0"]}},
            output_robot_data_spec={
                "robot_0": {DataType.JOINT_TARGET_POSITIONS: ["joint_0"]}
            },
            num_samples=5,
        )

        # Test index out of bounds
        with pytest.raises(IndexError):
            _ = dataset[10]  # Only 5 samples

        with pytest.raises(IndexError):
            _ = dataset[-10]  # Negative out of bounds

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single sample dataset
        dataset = PytorchDummyDataset(
            input_robot_data_spec={"robot_0": {DataType.JOINT_POSITIONS: ["joint_0"]}},
            output_robot_data_spec={
                "robot_0": {DataType.JOINT_TARGET_POSITIONS: ["joint_0"]}
            },
            num_samples=1,
        )
        assert len(dataset) == 1
        sample = dataset[0]
        assert sample is not None

        # Large prediction horizon
        dataset = PytorchDummyDataset(
            input_robot_data_spec={"robot_0": {DataType.JOINT_POSITIONS: ["joint_0"]}},
            output_robot_data_spec={
                "robot_0": {DataType.JOINT_TARGET_POSITIONS: ["joint_0"]}
            },
            num_samples=5,
            output_prediction_horizon=20,
        )
        sample = dataset[0]
        # Check that output has extended time dimension
        for joint_data in sample.outputs[DataType.JOINT_TARGET_POSITIONS]:
            for attr_name in vars(joint_data):
                attr_value = getattr(joint_data, attr_name)
                if isinstance(attr_value, torch.Tensor) and len(attr_value.shape) > 0:
                    # Time dimension should reflect prediction horizon
                    assert attr_value.shape[1] == 20

    @pytest.mark.parametrize("horizon", [1, 5, 10, 20])
    def test_different_prediction_horizons(self, horizon):
        """Test dataset with different prediction horizons."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec={"robot_0": {DataType.JOINT_POSITIONS: ["joint_0"]}},
            output_robot_data_spec={
                "robot_0": {DataType.JOINT_TARGET_POSITIONS: ["joint_0"]}
            },
            num_samples=3,
            output_prediction_horizon=horizon,
        )

        sample = dataset[0]
        # Check output time dimension reflects horizon
        for joint_data in sample.outputs[DataType.JOINT_TARGET_POSITIONS]:
            for attr_name in vars(joint_data):
                attr_value = getattr(joint_data, attr_name)
                if isinstance(attr_value, torch.Tensor) and len(attr_value.shape) > 0:
                    assert attr_value.shape[1] == horizon


class TestDatasetStatistics:
    """Test the dataset statistics generation in PytorchDummyDataset."""

    def test_dataset_statistics_initialization(self):
        """Test that dataset statistics are properly initialized."""
        dataset = PytorchDummyDataset(
            input_robot_data_spec={
                "robot_0": {
                    DataType.JOINT_POSITIONS: ["joint_0", "joint_1", "joint_2"],
                    DataType.RGB_IMAGES: ["camera_0"],
                }
            },
            output_robot_data_spec={
                "robot_0": {DataType.JOINT_TARGET_POSITIONS: ["joint_0", "joint_1"]}
            },
            num_samples=5,
        )

        stats = dataset.dataset_statistics

        # Check joint positions statistics
        assert DataType.JOINT_POSITIONS in stats
        assert len(stats[DataType.JOINT_POSITIONS]) == 3  # 3 joints

        # Check RGB images statistics
        assert DataType.RGB_IMAGES in stats
        assert len(stats[DataType.RGB_IMAGES]) == 1  # 1 camera

        # Check joint target positions statistics
        assert DataType.JOINT_TARGET_POSITIONS in stats
        assert len(stats[DataType.JOINT_TARGET_POSITIONS]) == 2  # 2 joints
