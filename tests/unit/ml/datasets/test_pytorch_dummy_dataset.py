"""Tests for PytorchDummyDataset.

This module provides comprehensive testing for the synthetic data generation
capabilities of PytorchDummyDataset, including multi-modal data generation,
proper tensor shapes, masking, and collation functionality.
"""

import pytest
import torch

from neuracore.core.nc_types import DataType
from neuracore.ml import BatchedTrainingSamples, MaskableData
from neuracore.ml.datasets.pytorch_dummy_dataset import PytorchDummyDataset


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for language data testing."""

    def tokenizer(texts):
        # Simple mock that returns fixed-size tensors
        batch_size = len(texts)
        seq_len = 10
        return (
            torch.randint(0, 1000, (batch_size, seq_len)),  # input_ids
            torch.ones(batch_size, seq_len),  # attention_mask
        )

    return tokenizer


class TestPytorchDummyDataset:
    """Test suite for PytorchDummyDataset."""

    @pytest.fixture
    def basic_data_types(self):
        """Basic data types for testing."""
        return {
            "input_types": [DataType.JOINT_POSITIONS, DataType.RGB_IMAGE],
            "output_types": [DataType.JOINT_TARGET_POSITIONS],
        }

    @pytest.fixture
    def all_data_types(self):
        """All supported data types for comprehensive testing."""
        return {
            "input_types": [
                DataType.JOINT_POSITIONS,
                DataType.JOINT_VELOCITIES,
                DataType.JOINT_TORQUES,
                DataType.RGB_IMAGE,
                DataType.DEPTH_IMAGE,
                DataType.POINT_CLOUD,
                DataType.END_EFFECTORS,
                DataType.END_EFFECTOR_POSES,
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
                DataType.POSES,
                DataType.LANGUAGE,
                DataType.CUSTOM,
            ],
            "output_types": [
                DataType.JOINT_TARGET_POSITIONS,
                DataType.RGB_IMAGE,
                DataType.END_EFFECTORS,
            ],
        }

    def test_initialization_basic(self, basic_data_types):
        """Test basic dataset initialization."""
        dataset = PytorchDummyDataset(
            input_data_types=basic_data_types["input_types"],
            output_data_types=basic_data_types["output_types"],
            num_samples=50,
            num_episodes=10,
        )

        assert len(dataset) == 50
        assert dataset.num_samples == 50
        assert dataset.num_recordings == 10
        assert dataset.output_prediction_horizon == 5  # default value

        # Check data types are properly set
        expected_types = set(
            basic_data_types["input_types"] + basic_data_types["output_types"]
        )
        assert dataset.data_types == expected_types

    def test_initialization_all_data_types(self, all_data_types, mock_tokenizer):
        """Test initialization with all supported data types."""
        dataset = PytorchDummyDataset(
            input_data_types=all_data_types["input_types"],
            output_data_types=all_data_types["output_types"],
            num_samples=20,
            output_prediction_horizon=8,
            tokenize_text=mock_tokenizer,
        )

        assert len(dataset) == 20
        assert dataset.output_prediction_horizon == 8
        assert dataset.tokenize_text is not None

        # Check dataset description is properly initialized
        desc = dataset.dataset_description
        assert desc.joint_positions.max_len == 6
        assert desc.rgb_images.max_len == 2
        assert desc.point_clouds.max_len == 1
        assert desc.end_effector_poses.max_len == 7
        assert desc.parallel_gripper_open_amounts.max_len == 1

    def test_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        # No data types
        with pytest.raises(ValueError):
            PytorchDummyDataset(
                input_data_types=[], output_data_types=[], num_samples=10
            )

        # Language without tokenizer
        with pytest.raises(ValueError, match="Failed to initialize tokenize_text"):
            dataset = PytorchDummyDataset(
                input_data_types=[DataType.LANGUAGE],
                output_data_types=[DataType.JOINT_TARGET_POSITIONS],
                num_samples=10,
            )
            # Trigger the error by loading a sample
            _ = dataset[0]

    def test_dataset_length(self):
        """Test dataset length functionality."""
        for num_samples in [1, 10, 100]:
            dataset = PytorchDummyDataset(
                input_data_types=[DataType.JOINT_POSITIONS],
                output_data_types=[DataType.JOINT_TARGET_POSITIONS],
                num_samples=num_samples,
            )
            assert len(dataset) == num_samples

    def test_sample_generation_basic(self, basic_data_types):
        """Test basic sample generation."""
        dataset = PytorchDummyDataset(
            input_data_types=basic_data_types["input_types"],
            output_data_types=basic_data_types["output_types"],
            num_samples=10,
        )

        sample = dataset[0]

        # Check sample structure
        assert isinstance(sample, BatchedTrainingSamples)
        assert sample.inputs is not None
        assert sample.outputs is not None
        assert sample.output_prediction_mask is not None

        # Check prediction mask shape
        assert sample.output_prediction_mask.shape == (
            dataset.output_prediction_horizon,
        )
        assert torch.all(sample.output_prediction_mask == 1.0)

    def test_joint_data_generation(self):
        """Test joint data generation and properties."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.JOINT_POSITIONS, DataType.JOINT_VELOCITIES],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            num_samples=5,
        )

        sample = dataset[0]

        # Test input joint positions
        assert sample.inputs.joint_positions is not None
        assert isinstance(sample.inputs.joint_positions, MaskableData)
        assert sample.inputs.joint_positions.data.shape == (
            6,
        )  # max_len from dataset description
        assert sample.inputs.joint_positions.mask.shape == (6,)
        assert torch.all(sample.inputs.joint_positions.mask == 1.0)

        # Test input joint velocities
        assert sample.inputs.joint_velocities is not None
        assert sample.inputs.joint_velocities.data.shape == (6,)

        # Test output joint target positions
        assert sample.outputs.joint_target_positions is not None
        assert sample.outputs.joint_target_positions.data.shape == (
            7,
        )  # different max_len
        assert sample.outputs.joint_target_positions.mask.shape == (7,)

    def test_image_data_generation(self):
        """Test RGB and depth image generation."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.RGB_IMAGE, DataType.DEPTH_IMAGE],
            output_data_types=[DataType.RGB_IMAGE],
            num_samples=3,
        )

        sample = dataset[0]

        # Test RGB images
        assert sample.inputs.rgb_images is not None
        rgb_data = sample.inputs.rgb_images.data
        rgb_mask = sample.inputs.rgb_images.mask

        assert rgb_data.shape == (2, 3, 224, 224)  # max_cameras=2, RGB channels
        assert rgb_mask.shape == (2,)
        assert torch.all(rgb_mask == 1.0)

        # Test depth images
        assert sample.inputs.depth_images is not None
        depth_data = sample.inputs.depth_images.data

        assert depth_data.shape == (2, 1, 224, 224)  # depth has 1 channel

    def test_point_cloud_generation(self):
        """Test point cloud data generation."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.POINT_CLOUD],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            num_samples=3,
        )

        sample = dataset[0]

        assert sample.inputs.point_clouds is not None
        pc_data = sample.inputs.point_clouds.data
        pc_mask = sample.inputs.point_clouds.mask

        assert pc_data.shape == (1, 1024, 3)  # 1 cloud, 1024 points, xyz
        assert pc_mask.shape == (1,)
        assert torch.all(pc_mask == 1.0)

    def test_end_effector_data_generation(self):
        """Test end-effector data generation."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.END_EFFECTORS],
            output_data_types=[DataType.END_EFFECTORS],
            num_samples=3,
        )

        sample = dataset[0]

        # Test input end-effectors
        assert sample.inputs.end_effectors is not None
        ee_data = sample.inputs.end_effectors.data
        ee_mask = sample.inputs.end_effectors.mask

        assert ee_data.shape == (2,)  # 2 end-effectors
        assert ee_mask.shape == (2,)
        assert torch.all(ee_mask == 1.0)

    def test_end_effector_pose_data_generation(self):
        """Test end-effector pose data generation."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.END_EFFECTOR_POSES],
            output_data_types=[DataType.END_EFFECTOR_POSES],
            num_samples=3,
        )

        sample = dataset[0]

        # Test input end-effector poses
        assert sample.inputs.end_effector_poses is not None
        assert isinstance(sample.inputs.end_effector_poses, MaskableData)
        assert sample.inputs.end_effector_poses.data.shape == (7,)
        assert sample.inputs.end_effector_poses.mask.shape == (7,)
        assert torch.all(sample.inputs.end_effector_poses.mask == 1.0)

        # Test output end-effector poses
        assert sample.outputs.end_effector_poses is not None
        assert sample.outputs.end_effector_poses.data.shape == (7,)

    def test_parallel_gripper_open_amount_data_generation(self):
        """Test parallel gripper open amount data generation."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS],
            output_data_types=[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS],
            num_samples=3,
        )

        sample = dataset[0]

        # Test input parallel gripper open amounts
        assert sample.inputs.parallel_gripper_open_amounts is not None
        assert isinstance(sample.inputs.parallel_gripper_open_amounts, MaskableData)
        assert sample.inputs.parallel_gripper_open_amounts.data.shape == (1,)
        assert sample.inputs.parallel_gripper_open_amounts.mask.shape == (1,)
        assert torch.all(sample.inputs.parallel_gripper_open_amounts.mask == 1.0)

        # Test output parallel gripper open amounts
        assert sample.outputs.parallel_gripper_open_amounts is not None
        assert sample.outputs.parallel_gripper_open_amounts.data.shape == (1,)

    def test_pose_data_generation(self):
        """Test pose data generation."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.POSES],
            output_data_types=[DataType.POSES],
            num_samples=3,
        )

        sample = dataset[0]

        assert sample.inputs.poses is not None
        pose_data = sample.inputs.poses.data
        pose_mask = sample.inputs.poses.mask

        assert pose_data.shape == (12,)  # 2 poses x 6DOF each
        assert pose_mask.shape == (12,)
        assert torch.all(pose_mask == 1.0)

    def test_language_data_generation(self, mock_tokenizer):
        """Test language data generation with tokenizer."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.LANGUAGE],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            num_samples=3,
            tokenize_text=mock_tokenizer,
        )

        sample = dataset[0]

        assert sample.inputs.language_tokens is not None
        token_data = sample.inputs.language_tokens.data
        token_mask = sample.inputs.language_tokens.mask

        assert token_data.shape == (1, 10)
        assert token_mask.shape == (1, 10)

    def test_custom_data_generation(self):
        """Test custom data generation."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.CUSTOM],
            output_data_types=[DataType.CUSTOM],
            num_samples=3,
        )

        sample = dataset[0]

        assert sample.inputs.custom_data is not None
        assert isinstance(sample.inputs.custom_data, dict)
        assert len(sample.inputs.custom_data) >= 2  # sensor_1, sensor_2

        # Check specific sensors
        assert "sensor_1" in sample.inputs.custom_data
        sensor1_data = sample.inputs.custom_data["sensor_1"]
        assert isinstance(sensor1_data, MaskableData)
        assert sensor1_data.data.shape == (10,)
        assert sensor1_data.mask.shape == (10,)

    def test_deterministic_behavior(self):
        """Test that the dataset generates deterministic data."""
        # Create two identical datasets
        dataset1 = PytorchDummyDataset(
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            num_samples=5,
        )

        dataset2 = PytorchDummyDataset(
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            num_samples=5,
        )

        # Note: Since the dataset uses random generation, we can't guarantee
        # deterministic behavior without setting seeds. This test documents
        # the current behavior and can be enhanced if deterministic behavior
        # is implemented.

        sample1 = dataset1[0]
        sample2 = dataset2[0]

        # Structure should be the same
        assert (
            sample1.inputs.joint_positions.data.shape
            == sample2.inputs.joint_positions.data.shape
        )
        assert (
            sample1.outputs.joint_target_positions.data.shape
            == sample2.outputs.joint_target_positions.data.shape
        )

    def test_collate_fn_basic(self, basic_data_types):
        """Test basic collation functionality."""
        dataset = PytorchDummyDataset(
            input_data_types=basic_data_types["input_types"],
            output_data_types=basic_data_types["output_types"],
            num_samples=10,
        )

        # Get multiple samples
        samples = [dataset[i] for i in range(3)]

        # Test collation
        batched = dataset.collate_fn(samples)

        assert isinstance(batched, BatchedTrainingSamples)
        assert batched.inputs is not None
        assert batched.outputs is not None

        # Check batch dimensions
        assert batched.inputs.joint_positions.data.shape[0] == 3  # batch size
        assert batched.outputs.joint_target_positions.data.shape[0] == 3

        # Check prediction mask
        assert batched.output_prediction_mask.shape == (
            3,
            dataset.output_prediction_horizon,
        )

    def test_collate_fn_images(self):
        """Test collation with image data."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.RGB_IMAGE],
            output_data_types=[DataType.RGB_IMAGE],
            num_samples=5,
        )

        samples = [dataset[i] for i in range(2)]
        batched = dataset.collate_fn(samples)

        # Check input images
        assert batched.inputs.rgb_images.data.shape == (
            2,
            2,
            3,
            224,
            224,
        )  # batch=2, cameras=2
        assert batched.inputs.rgb_images.mask.shape == (2, 2)

        # Check output images (should be expanded across prediction horizon)
        expected_shape = (2, dataset.output_prediction_horizon, 2, 3, 224, 224)
        assert batched.outputs.rgb_images.data.shape == expected_shape

    def test_collate_fn_point_clouds(self):
        """Test collation with point cloud data."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.POINT_CLOUD],
            output_data_types=[DataType.POINT_CLOUD],
            num_samples=5,
        )

        samples = [dataset[i] for i in range(2)]
        batched = dataset.collate_fn(samples)

        # Check point clouds
        assert batched.inputs.point_clouds.data.shape == (
            2,
            1,
            1024,
            3,
        )  # batch=2, clouds=1

        # Output should be expanded
        expected_shape = (2, dataset.output_prediction_horizon, 1, 1024, 3)
        assert batched.outputs.point_clouds.data.shape == expected_shape

    def test_collate_fn_custom_data(self):
        """Test collation with custom data."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.CUSTOM],
            output_data_types=[DataType.CUSTOM],
            num_samples=5,
        )

        samples = [dataset[i] for i in range(2)]
        batched = dataset.collate_fn(samples)

        # Check custom data structure
        assert isinstance(batched.inputs.custom_data, dict)
        assert isinstance(batched.outputs.custom_data, dict)

        # Check that custom data is properly batched
        for key in batched.inputs.custom_data:
            input_data = batched.inputs.custom_data[key]
            assert input_data.data.shape[0] == 2  # batch size

            if key in batched.outputs.custom_data:
                output_data = batched.outputs.custom_data[key]
                # Should be expanded across prediction horizon
                assert output_data.data.shape[0] == 2  # batch size
                assert output_data.data.shape[1] == dataset.output_prediction_horizon

    def test_collate_fn_language_data(self, mock_tokenizer):
        """Test collation with language data."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.LANGUAGE],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            num_samples=5,
            tokenize_text=mock_tokenizer,
        )

        samples = [dataset[i] for i in range(2)]
        batched = dataset.collate_fn(samples)

        assert batched.inputs.language_tokens.data.shape == (2, 10)
        assert batched.inputs.language_tokens.mask.shape == (2, 10)

    def test_error_handling(self):
        """Test error handling in dataset operations."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            num_samples=5,
        )

        # Test index out of bounds
        with pytest.raises(IndexError):
            _ = dataset[10]  # Only 5 samples

        # Test negative indexing (should work)
        sample = dataset[-1]
        assert sample is not None

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single sample dataset
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            num_samples=1,
        )
        assert len(dataset) == 1
        sample = dataset[0]
        assert sample is not None

        # Large prediction horizon
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            num_samples=5,
            output_prediction_horizon=20,
        )
        sample = dataset[0]
        assert sample.output_prediction_mask.shape == (20,)

    def test_memory_efficiency(self):
        """Test that the dataset doesn't consume excessive memory."""
        # This test ensures that creating a large dataset doesn't immediately
        # allocate memory for all samples (lazy loading)
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.RGB_IMAGE],
            output_data_types=[DataType.RGB_IMAGE],
            num_samples=1000,  # Large number
        )

        # Dataset creation should be fast and not allocate much memory
        assert len(dataset) == 1000

        # Only loading specific samples should create data
        sample = dataset[0]
        assert sample is not None

    @pytest.mark.parametrize("horizon", [1, 5, 10, 20])
    def test_different_prediction_horizons(self, horizon):
        """Test dataset with different prediction horizons."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            num_samples=3,
            output_prediction_horizon=horizon,
        )

        sample = dataset[0]
        assert sample.output_prediction_mask.shape == (horizon,)
        assert torch.all(sample.output_prediction_mask == 1.0)

    @pytest.mark.parametrize(
        "data_type",
        [
            DataType.JOINT_POSITIONS,
            DataType.RGB_IMAGE,
            DataType.POINT_CLOUD,
            DataType.END_EFFECTORS,
            DataType.POSES,
        ],
    )
    def test_single_data_type_datasets(self, data_type):
        """Test datasets with single data types."""
        # Skip language without tokenizer
        if data_type == DataType.LANGUAGE:
            pytest.skip("Language requires tokenizer")

        dataset = PytorchDummyDataset(
            input_data_types=[data_type],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],  # Safe fallback
            num_samples=3,
        )

        sample = dataset[0]
        assert sample is not None

        # Check that the specified data type is present
        if data_type == DataType.JOINT_POSITIONS:
            assert sample.inputs.joint_positions is not None
        elif data_type == DataType.RGB_IMAGE:
            assert sample.inputs.rgb_images is not None
        elif data_type == DataType.POINT_CLOUD:
            assert sample.inputs.point_clouds is not None
        elif data_type == DataType.END_EFFECTORS:
            assert sample.inputs.end_effectors is not None
        elif data_type == DataType.POSES:
            assert sample.inputs.poses is not None


class TestDatasetDescription:
    """Test the dataset description generation in PytorchDummyDataset."""

    def test_dataset_description_initialization(self):
        """Test that dataset description is properly initialized."""
        dataset = PytorchDummyDataset(
            input_data_types=[DataType.JOINT_POSITIONS, DataType.RGB_IMAGE],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            num_samples=5,
        )

        desc = dataset.dataset_description

        # Check joint positions
        assert desc.joint_positions.max_len == 6
        assert len(desc.joint_positions.mean) == 6
        assert len(desc.joint_positions.std) == 6
        assert all(std == 1.0 for std in desc.joint_positions.std)
        assert all(mean == 0.0 for mean in desc.joint_positions.mean)

        # Check RGB images
        assert desc.rgb_images.max_len == 2

        # Check joint target positions
        assert desc.joint_target_positions.max_len == 7

    def test_dataset_description_all_modalities(self, mock_tokenizer):
        """Test dataset description with all modalities."""
        all_input_types = [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.RGB_IMAGE,
            DataType.DEPTH_IMAGE,
            DataType.POINT_CLOUD,
            DataType.END_EFFECTORS,
            DataType.END_EFFECTOR_POSES,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.POSES,
            DataType.LANGUAGE,
        ]

        dataset = PytorchDummyDataset(
            input_data_types=all_input_types,
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            num_samples=5,
            tokenize_text=mock_tokenizer,
        )

        desc = dataset.dataset_description

        # Check all components are properly initialized
        assert desc.joint_positions.max_len > 0
        assert desc.joint_velocities.max_len > 0
        assert desc.joint_torques.max_len > 0
        assert desc.end_effector_states.max_len > 0
        assert desc.poses.max_len > 0
        assert desc.end_effector_poses.max_len > 0
        assert desc.parallel_gripper_open_amounts.max_len > 0
        assert desc.rgb_images.max_len > 0
        assert desc.depth_images.max_len > 0
        assert desc.point_clouds.max_len > 0
        assert desc.language.max_len > 0
