"""Tests for PytorchSynchronizedDataset.

This module provides comprehensive testing for the synchronized dataset
functionality including data loading, caching, multi-modal data processing,
error handling, and device management.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.nc_types import (
    CameraData,
    CustomData,
    DataItemStats,
    DatasetDescription,
    DataType,
    EndEffectorData,
    JointData,
    LanguageData,
    PointCloudData,
    PoseData,
    SyncPoint,
)
from neuracore.ml import BatchedTrainingSamples, MaskableData
from neuracore.ml.datasets.pytorch_synchronized_dataset import (
    PytorchSynchronizedDataset,
)


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for language data testing."""

    def tokenizer(texts):
        batch_size = len(texts)
        seq_len = 10
        return (
            torch.randint(0, 1000, (batch_size, seq_len)),  # input_ids
            torch.ones(batch_size, seq_len),  # attention_mask
        )

    return tokenizer


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataset_description():
    """Create a sample dataset description for testing."""
    return DatasetDescription(
        joint_positions=DataItemStats(mean=[0.0] * 6, std=[1.0] * 6, max_len=6),
        joint_velocities=DataItemStats(mean=[0.0] * 6, std=[1.0] * 6, max_len=6),
        joint_torques=DataItemStats(mean=[0.0] * 6, std=[1.0] * 6, max_len=6),
        joint_target_positions=DataItemStats(mean=[0.0] * 7, std=[1.0] * 7, max_len=7),
        end_effector_states=DataItemStats(mean=[0.0] * 2, std=[1.0] * 2, max_len=2),
        poses=DataItemStats(mean=[0.0] * 12, std=[1.0] * 12, max_len=12),
        max_num_rgb_images=2,
        max_num_depth_images=2,
        max_num_point_clouds=1,
        max_language_length=50,
        custom_data_stats={
            "sensor_1": DataItemStats(mean=[0.0] * 5, std=[1.0] * 5, max_len=5),
            "sensor_2": DataItemStats(mean=[0.0] * 3, std=[1.0] * 3, max_len=3),
        },
    )


@pytest.fixture
def mock_image():
    """Create a mock PIL Image for testing."""
    return Image.new("RGB", (224, 224), color="red")


@pytest.fixture
def mock_depth_image():
    """Create a mock depth image for testing."""
    return Image.new("L", (224, 224), color=128)


@pytest.fixture
def sample_sync_point(mock_image, mock_depth_image):
    """Create a sample SyncPoint with various data types."""
    return SyncPoint(
        timestamp=1234567890.0,
        joint_positions=JointData(
            values={"joint_1": 0.1, "joint_2": 0.2, "joint_3": 0.3}
        ),
        joint_velocities=JointData(
            values={"joint_1": 0.01, "joint_2": 0.02, "joint_3": 0.03}
        ),
        joint_torques=JointData(
            values={"joint_1": 1.0, "joint_2": 2.0, "joint_3": 3.0}
        ),
        joint_target_positions=JointData(
            values={"joint_1": 0.15, "joint_2": 0.25, "joint_3": 0.35, "joint_4": 0.45}
        ),
        end_effectors=EndEffectorData(
            open_amounts={"gripper_1": 0.5, "gripper_2": 0.8}
        ),
        poses={
            "end_effector": PoseData(
                pose={"end_effector": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]}
            ),
            "object": PoseData(pose={"object": [4.0, 5.0, 6.0, 0.1, 0.2, 0.3]}),
        },
        rgb_images={
            "camera_1": CameraData(frame=mock_image, timestamp=1234567890.0),
            "camera_2": CameraData(frame=mock_image, timestamp=1234567890.1),
        },
        depth_images={
            "depth_camera_1": CameraData(frame=mock_depth_image, timestamp=1234567890.0)
        },
        point_clouds={
            "lidar_1": PointCloudData(
                points=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
            )
        },
        language_data=LanguageData(text="Pick up the red block"),
        custom_data={
            "sensor_1": CustomData(data=[1.0, 2.0, 3.0, 4.0, 5.0]),
            "sensor_2": CustomData(data=[0.1, 0.2, 0.3]),
        },
    )


@pytest.fixture
def mock_synced_recording(sample_sync_point):
    """Create a mock SynchronizedRecording for testing."""
    # Create multiple sync points for sequence testing
    sync_points = [sample_sync_point] * 10  # 10 timesteps

    class MockSynchronizedRecording:
        def __init__(self):
            self.sync_points = sync_points

        def __len__(self):
            return len(self.sync_points)

        def __getitem__(self, idx):
            if isinstance(idx, int):
                return self.sync_points[idx]
            elif isinstance(idx, slice):
                return self.sync_points[idx.start : idx.stop : idx.step]
            else:
                raise TypeError(f"Invalid index type: {type(idx)}")

        def __iter__(self):
            return iter(self.sync_points)

    return MockSynchronizedRecording()


@pytest.fixture
def mock_synchronized_dataset(mock_synced_recording, sample_dataset_description):
    """Create a mock SynchronizedDataset for testing."""

    class MockSynchronizedDataset(SynchronizedDataset):
        def __init__(self):
            self.dataset_description = sample_dataset_description
            self._recordings = [mock_synced_recording] * 5  # 5 episodes

        @property
        def num_transitions(self):
            return 50  # 5 episodes * 10 timesteps each

        def __len__(self):
            return 5  # 5 episodes

        def __getitem__(self, idx):
            return self._recordings[idx]

    return MockSynchronizedDataset()


class TestPytorchSynchronizedDatasetInitialization:
    """Test dataset initialization and configuration."""

    def test_basic_initialization(self, mock_synchronized_dataset, temp_cache_dir):
        """Test basic dataset initialization."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS, DataType.RGB_IMAGE],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=5,
            cache_dir=temp_cache_dir,
        )

        assert dataset.synchronized_dataset == mock_synchronized_dataset
        assert dataset.input_data_types == [
            DataType.JOINT_POSITIONS,
            DataType.RGB_IMAGE,
        ]
        assert dataset.output_data_types == [DataType.JOINT_TARGET_POSITIONS]
        assert dataset.output_prediction_horizon == 5
        assert dataset.cache_dir == Path(temp_cache_dir)
        assert len(dataset) == 50  # num_transitions

    def test_initialization_with_all_data_types(
        self, mock_synchronized_dataset, mock_tokenizer, temp_cache_dir
    ):
        """Test initialization with all supported data types."""
        all_input_types = [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.RGB_IMAGE,
            DataType.DEPTH_IMAGE,
            DataType.POINT_CLOUD,
            DataType.END_EFFECTORS,
            DataType.POSES,
            DataType.LANGUAGE,
            DataType.CUSTOM,
        ]

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=all_input_types,
            output_data_types=[DataType.JOINT_TARGET_POSITIONS, DataType.END_EFFECTORS],
            output_prediction_horizon=8,
            cache_dir=temp_cache_dir,
            tokenize_text=mock_tokenizer,
        )

        assert dataset.tokenize_text == mock_tokenizer
        assert dataset.output_prediction_horizon == 8
        assert set(dataset.data_types) == set(
            all_input_types + [DataType.JOINT_TARGET_POSITIONS, DataType.END_EFFECTORS]
        )

    def test_initialization_invalid_synchronized_dataset(self, temp_cache_dir):
        """Test initialization with invalid synchronized dataset."""
        with pytest.raises(
            TypeError,
            match="synchronized_dataset must be an instance of SynchronizedDataset",
        ):
            PytorchSynchronizedDataset(
                synchronized_dataset="invalid",  # Not a SynchronizedDataset
                input_data_types=[DataType.JOINT_POSITIONS],
                output_data_types=[DataType.JOINT_TARGET_POSITIONS],
                output_prediction_horizon=5,
                cache_dir=temp_cache_dir,
            )

    def test_default_cache_dir(self, mock_synchronized_dataset):
        """Test initialization with default cache directory."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=5,
        )

        # Should create a default cache directory in temp
        assert dataset.cache_dir.name == "episodic_dataset_cache"
        assert dataset.cache_dir.exists()

    def test_cache_directory_cleanup(self, mock_synchronized_dataset, temp_cache_dir):
        """Test that existing cache directory is cleaned up."""
        # Create some files in the cache directory
        test_file = Path(temp_cache_dir) / "test_file.txt"
        test_file.write_text("test content")
        assert test_file.exists()

        # Initialize dataset (should clean up cache)
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=5,
            cache_dir=temp_cache_dir,
        )

        # Cache directory should exist but be empty
        assert dataset.cache_dir.exists()
        assert not test_file.exists()


class TestDataLoading:
    """Test data loading functionality."""

    @patch("neuracore.login")
    def test_load_sample_basic(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test basic sample loading."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS, DataType.RGB_IMAGE],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=2)

        assert isinstance(sample, BatchedTrainingSamples)
        assert sample.inputs is not None
        assert sample.outputs is not None
        assert sample.output_predicition_mask is not None

        # Check that login was called
        mock_login.assert_called_once()

    @patch("neuracore.login")
    def test_load_sample_with_caching(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test sample loading with caching."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None

            # First load - should create cache
            sample1 = dataset.load_sample(episode_idx=0, timestep=2)

            # Check cache file was created
            cache_file = dataset.cache_dir / "ep_0_frame_2.pt"
            assert cache_file.exists()

            # Second load - should use cache
            sample2 = dataset.load_sample(episode_idx=0, timestep=2)

            # Samples should be identical (loaded from cache)
            assert torch.equal(
                sample1.output_predicition_mask, sample2.output_predicition_mask
            )

    @patch("neuracore.login")
    def test_load_sample_memory_monitoring(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test memory monitoring during sample loading."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None

            # Load multiple samples to trigger memory check
            for i in range(105):  # Should trigger memory check at multiples of 100
                dataset._mem_check_counter = i
                dataset.load_sample(episode_idx=0, timestep=0)

            # Memory should be checked at least once
            assert mock_monitor.check_memory.call_count >= 1


class TestDataTypeProcessing:
    """Test processing of different data types."""

    @patch("neuracore.login")
    def test_joint_data_processing(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test joint data processing."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS, DataType.JOINT_VELOCITIES],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=0)

        # Check input joint positions
        assert sample.inputs.joint_positions is not None
        assert isinstance(sample.inputs.joint_positions, MaskableData)
        assert sample.inputs.joint_positions.data.shape == (
            6,
        )  # max_len from dataset description
        assert sample.inputs.joint_positions.mask.shape == (6,)

        # Check input joint velocities
        assert sample.inputs.joint_velocities is not None
        assert sample.inputs.joint_velocities.data.shape == (6,)

        # Check output joint target positions
        assert sample.outputs.joint_target_positions is not None
        assert sample.outputs.joint_target_positions.data.shape == (
            2,
            7,
        )  # prediction_horizon x max_len

    @patch("neuracore.login")
    def test_image_data_processing(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test RGB and depth image processing."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.RGB_IMAGE, DataType.DEPTH_IMAGE],
            output_data_types=[DataType.RGB_IMAGE],
            output_prediction_horizon=2,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=0)

        # Check RGB images
        assert sample.inputs.rgb_images is not None
        rgb_data = sample.inputs.rgb_images.data
        rgb_mask = sample.inputs.rgb_images.mask

        assert rgb_data.shape == (
            2,
            3,
            224,
            224,
        )  # max_cameras x channels x height x width
        assert rgb_mask.shape == (2,)

        # Check depth images
        assert sample.inputs.depth_images is not None
        depth_data = sample.inputs.depth_images.data

        assert depth_data.shape == (2, 1, 224, 224)  # depth has 1 channel

        # Check output RGB images
        assert sample.outputs.rgb_images is not None
        output_rgb_data = sample.outputs.rgb_images.data

        assert output_rgb_data.shape == (
            2,
            2,
            3,
            224,
            224,
        )  # prediction_horizon x cameras x channels x h x w

    @patch("neuracore.login")
    def test_point_cloud_processing(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test point cloud data processing."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.POINT_CLOUD],
            output_data_types=[DataType.POINT_CLOUD],
            output_prediction_horizon=2,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=0)

        # Check input point clouds
        assert sample.inputs.point_clouds is not None
        pc_data = sample.inputs.point_clouds.data
        pc_mask = sample.inputs.point_clouds.mask

        assert pc_data.shape == (1, 1024, 3)  # 1 cloud, 1024 points, xyz
        assert pc_mask.shape == (1, 1024)

        # Check output point clouds
        assert sample.outputs.point_clouds is not None
        output_pc_data = sample.outputs.point_clouds.data

        assert output_pc_data.shape == (
            2,
            1,
            1024,
            3,
        )  # prediction_horizon x clouds x points x xyz

    @patch("neuracore.login")
    def test_end_effector_processing(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test end-effector data processing."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.END_EFFECTORS],
            output_data_types=[DataType.END_EFFECTORS],
            output_prediction_horizon=2,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=0)

        # Check input end-effectors
        assert sample.inputs.end_effectors is not None
        ee_data = sample.inputs.end_effectors.data
        ee_mask = sample.inputs.end_effectors.mask

        assert ee_data.shape == (2,)  # max_len from dataset description
        assert ee_mask.shape == (2,)

        # Check output end-effectors
        assert sample.outputs.end_effectors is not None
        output_ee_data = sample.outputs.end_effectors.data

        assert output_ee_data.shape == (2, 2)  # prediction_horizon x max_len

    @patch("neuracore.login")
    def test_pose_processing(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test pose data processing."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.POSES],
            output_data_types=[DataType.POSES],
            output_prediction_horizon=2,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=0)

        # Check input poses
        assert sample.inputs.poses is not None
        pose_data = sample.inputs.poses.data
        pose_mask = sample.inputs.poses.mask

        assert pose_data.shape == (12,)  # max_len from dataset description
        assert pose_mask.shape == (12,)

        # Check output poses
        assert sample.outputs.poses is not None
        output_pose_data = sample.outputs.poses.data

        assert output_pose_data.shape == (2, 12)  # prediction_horizon x max_len

    @patch("neuracore.login")
    def test_language_processing(
        self, mock_login, mock_synchronized_dataset, mock_tokenizer, temp_cache_dir
    ):
        """Test language data processing."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.LANGUAGE],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
            cache_dir=temp_cache_dir,
            tokenize_text=mock_tokenizer,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=0)

        # Check language tokens
        assert sample.inputs.language_tokens is not None
        token_data = sample.inputs.language_tokens.data
        token_mask = sample.inputs.language_tokens.mask

        assert token_data.shape == (1, 10)  # batch_size=1, seq_len=10
        assert token_mask.shape == (1, 10)

    @patch("neuracore.login")
    def test_language_processing_without_tokenizer(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test language processing fails without tokenizer."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.LANGUAGE],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
            cache_dir=temp_cache_dir,
            # No tokenize_text provided
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None

            with pytest.raises(ValueError, match="Failed to initialize tokenize_text"):
                dataset.load_sample(episode_idx=0, timestep=0)

    @patch("neuracore.login")
    def test_custom_data_processing(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test custom data processing."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.CUSTOM],
            output_data_types=[DataType.CUSTOM],
            output_prediction_horizon=2,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=0)

        # Check input custom data
        assert sample.inputs.custom_data is not None
        assert isinstance(sample.inputs.custom_data, dict)
        assert len(sample.inputs.custom_data) >= 2  # sensor_1, sensor_2

        # Check specific sensors
        assert "sensor_1" in sample.inputs.custom_data
        sensor1_data = sample.inputs.custom_data["sensor_1"]
        assert isinstance(sensor1_data, MaskableData)

        # Check output custom data
        assert sample.outputs.custom_data is not None
        assert isinstance(sample.outputs.custom_data, dict)


class TestCacheManagement:
    """Test caching functionality."""

    @patch("neuracore.login")
    def test_cache_space_management(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test cache space management."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
            cache_dir=temp_cache_dir,
        )

        with patch.object(
            dataset.cache_manager, "ensure_space_available"
        ) as mock_ensure_space:
            mock_ensure_space.return_value = True

            with patch.object(dataset, "_memory_monitor") as mock_monitor:
                mock_monitor.check_memory.return_value = None
                dataset.load_sample(episode_idx=0, timestep=0)

            # Should check for space availability
            mock_ensure_space.assert_called()

    @patch("neuracore.login")
    def test_cache_space_low_warning(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test warning when cache space is low."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
            cache_dir=temp_cache_dir,
        )

        with patch.object(
            dataset.cache_manager, "ensure_space_available"
        ) as mock_ensure_space:
            mock_ensure_space.return_value = False  # Simulate low space

            with patch.object(dataset, "_memory_monitor") as mock_monitor:
                mock_monitor.check_memory.return_value = None

                with patch(
                    "neuracore.ml.datasets.pytorch_synchronized_dataset.logger"
                ) as mock_logger:
                    dataset.load_sample(episode_idx=0, timestep=0)

                    # Should log warning about low disk space
                    mock_logger.warning.assert_called_with(
                        "Low disk space. Some cache files were removed."
                    )

    def test_cache_file_naming(self, mock_synchronized_dataset, temp_cache_dir):
        """Test cache file naming convention."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
            cache_dir=temp_cache_dir,
        )

        # Test cache file path generation
        cache_file = dataset.cache_dir / "ep_5_frame_10.pt"
        expected_path = Path(temp_cache_dir) / "ep_5_frame_10.pt"

        assert str(cache_file) == str(expected_path)


class TestOutputPredictionMask:
    """Test output prediction mask creation."""

    def test_create_output_prediction_mask_full_horizon(
        self, mock_synchronized_dataset
    ):
        """Test mask creation when full horizon is available."""

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=5,
        )

        # Episode has enough timesteps
        mask = dataset._create_output_prediction_mask(
            episode_length=20, timestep=5, output_prediction_horizon=5
        )

        assert mask.shape == (5,)
        assert torch.all(mask == 1.0)

    def test_create_output_prediction_mask_partial_horizon(
        self, mock_synchronized_dataset
    ):
        """Test mask creation when only partial horizon is available."""

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=5,
        )

        # Episode ends before full horizon
        mask = dataset._create_output_prediction_mask(
            episode_length=8, timestep=5, output_prediction_horizon=5
        )

        assert mask.shape == (5,)
        # Only first 3 timesteps should be valid (5, 6, 7)
        expected_mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
        assert torch.equal(mask, expected_mask)

    def test_create_output_prediction_mask_at_episode_end(
        self, mock_synchronized_dataset
    ):
        """Test mask creation at episode end."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=5,
        )

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=5,
        )

        # At the very end of episode
        mask = dataset._create_output_prediction_mask(
            episode_length=10, timestep=9, output_prediction_horizon=5
        )

        assert mask.shape == (5,)
        # Only first timestep should be valid
        expected_mask = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        assert torch.equal(mask, expected_mask)


class TestHelperMethods:
    """Test helper methods for data processing."""

    def test_create_joint_maskable_input_data(self, mock_synchronized_dataset):
        """Test joint data maskable input creation."""
        # Create a minimal mock dataset just for testing the helper method

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
        )

        joint_data = JointData(values={"joint_1": 0.1, "joint_2": 0.2, "joint_3": 0.3})
        max_len = 6

        maskable_data = dataset._create_joint_maskable_input_data(joint_data, max_len)

        assert isinstance(maskable_data, MaskableData)
        assert maskable_data.data.shape == (max_len,)
        assert maskable_data.mask.shape == (max_len,)

        # First 3 elements should be valid, rest should be masked
        assert torch.equal(
            maskable_data.mask, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        )

    def test_create_joint_maskable_output_data(self, mock_synchronized_dataset):
        """Test joint data maskable output creation."""

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
        )

        joint_data_list = [
            JointData(values={"joint_1": 0.1, "joint_2": 0.2}),
            JointData(values={"joint_1": 0.3, "joint_2": 0.4}),
        ]
        max_len = 4

        maskable_data = dataset._create_joint_maskable_output_data(
            joint_data_list, max_len
        )

        assert isinstance(maskable_data, MaskableData)
        assert maskable_data.data.shape == (2, max_len)  # prediction_horizon x max_len
        assert maskable_data.mask.shape == (2, max_len)

    def test_create_end_effector_maskable_input_data(self, mock_synchronized_dataset):
        """Test end-effector maskable input creation."""

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.END_EFFECTORS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
        )

        ee_data = EndEffectorData(open_amounts={"gripper_1": 0.5, "gripper_2": 0.8})

        maskable_data = dataset._create_end_effector_maskable_input_data(ee_data)

        assert isinstance(maskable_data, MaskableData)
        assert maskable_data.data.shape == (2,)  # max_len from dataset description
        assert maskable_data.mask.shape == (2,)
        assert torch.all(maskable_data.mask == 1.0)

    def test_create_pose_maskable_input_data(self, mock_synchronized_dataset):
        """Test pose maskable input creation."""

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.POSES],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
        )

        poses = {
            "end_effector": PoseData(
                pose={"end_effector": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]}
            ),
            "object": PoseData(pose={"object": [4.0, 5.0, 6.0, 0.1, 0.2, 0.3]}),
        }

        maskable_data = dataset._create_pose_maskable_input_data(poses)

        assert isinstance(maskable_data, MaskableData)
        assert maskable_data.data.shape == (12,)  # max_len from dataset description
        assert maskable_data.mask.shape == (12,)
        assert torch.all(maskable_data.mask == 1.0)

    def test_create_point_cloud_maskable_input_data(self, mock_synchronized_dataset):
        """Test point cloud maskable input creation."""

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.POINT_CLOUD],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
        )

        point_clouds = {
            "lidar_1": PointCloudData(points=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        }

        maskable_data = dataset._create_point_cloud_maskable_input_data(point_clouds)

        assert isinstance(maskable_data, MaskableData)
        assert maskable_data.data.shape == (1, 1024, 3)  # 1 cloud, 1024 points, xyz
        assert maskable_data.mask.shape == (1, 1024)

    def test_create_custom_maskable_input_data(self, mock_synchronized_dataset):
        """Test custom data maskable input creation."""

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.CUSTOM],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
        )

        custom_data = {
            "sensor_1": CustomData(data=[1.0, 2.0, 3.0]),
            "sensor_2": CustomData(data="text_data"),
        }

        maskable_data_dict = dataset._create_custom_maskable_input_data(custom_data)

        assert isinstance(maskable_data_dict, dict)
        assert "sensor_1" in maskable_data_dict
        assert "sensor_2" in maskable_data_dict

        # Check sensor_1 with list data
        sensor1_data = maskable_data_dict["sensor_1"]
        assert isinstance(sensor1_data, MaskableData)
        assert sensor1_data.data.shape == (3,)
        assert torch.all(sensor1_data.mask == 1.0)

        # Check sensor_2 with non-list data
        sensor2_data = maskable_data_dict["sensor_2"]
        assert isinstance(sensor2_data, MaskableData)
        assert sensor2_data.data.shape == (1,)

    def test_create_camera_maskable_input_data(
        self, mock_synchronized_dataset, mock_image
    ):
        """Test camera maskable input creation."""

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.RGB_IMAGE],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
        )

        camera_data = [mock_image, mock_image]  # 2 cameras

        maskable_data = dataset._create_camera_maskable_input_data(
            camera_data, dataset.dataset_description.max_num_rgb_images
        )

        assert isinstance(maskable_data, MaskableData)
        assert maskable_data.data.shape == (
            2,
            3,
            224,
            224,
        )  # cameras x channels x h x w
        assert maskable_data.mask.shape == (2,)
        assert torch.all(maskable_data.mask == 1.0)


class TestDatasetIntegration:
    """Test dataset integration with PyTorch ecosystem."""

    @patch("neuracore.login")
    def test_getitem_method(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test __getitem__ method."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset[5]

        assert isinstance(sample, BatchedTrainingSamples)
        assert sample.inputs is not None
        assert sample.outputs is not None

    @patch("neuracore.login")
    def test_getitem_negative_index(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test __getitem__ with negative index."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None

            # Should work with negative indices
            sample = dataset[-1]
            assert isinstance(sample, BatchedTrainingSamples)

    def test_getitem_index_out_of_bounds(
        self, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test __getitem__ with out of bounds index."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        # Test positive out of bounds
        with pytest.raises(IndexError):
            _ = dataset[100]  # Only 50 samples

        # Test negative out of bounds
        with pytest.raises(IndexError):
            _ = dataset[-100]

    def test_len_method(self, mock_synchronized_dataset, temp_cache_dir):
        """Test __len__ method."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        assert len(dataset) == 50  # num_transitions from mock

    @patch("neuracore.login")
    def test_error_handling_in_getitem(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test error handling in __getitem__ method."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "load_sample") as mock_load_sample:
            mock_load_sample.side_effect = Exception("Load error")

            # Should propagate the error after max retries
            with pytest.raises(Exception, match="Load error"):
                _ = dataset[0]


class TestPerformanceAndOptimization:
    """Test performance and optimization features."""

    def test_memory_monitoring_initialization(
        self, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test memory monitor initialization."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        assert dataset._memory_monitor is not None
        assert hasattr(dataset._memory_monitor, "check_memory")

    @patch("neuracore.login")
    def test_parallel_downloads_environment_variable(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test setting parallel downloads environment variable."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            dataset.load_sample(episode_idx=0, timestep=0)

        # Should set parallel downloads to 0
        assert os.environ.get("NEURACORE_NUM_PARALLEL_VIDEO_DOWNLOADS") == "0"

    def test_login_state_management(self, mock_synchronized_dataset, temp_cache_dir):
        """Test login state management."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        # Initially not logged in
        assert not dataset._logged_in

        with patch("neuracore.login") as mock_login:
            with patch.object(dataset, "_memory_monitor") as mock_monitor:
                mock_monitor.check_memory.return_value = None

                # First call should trigger login
                dataset.load_sample(episode_idx=0, timestep=0)
                assert dataset._logged_in
                mock_login.assert_called_once()

                # Second call should not trigger login again
                mock_login.reset_mock()
                dataset.load_sample(episode_idx=0, timestep=1)
                mock_login.assert_not_called()


class TestDataTypeCompatibility:
    """Test compatibility between different data type combinations."""

    @patch("neuracore.login")
    def test_visual_and_joint_combination(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test combination of visual and joint data."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[
                DataType.RGB_IMAGE,
                DataType.JOINT_POSITIONS,
                DataType.DEPTH_IMAGE,
            ],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS, DataType.END_EFFECTORS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=0)

        # Check all input types are present
        assert sample.inputs.rgb_images is not None
        assert sample.inputs.depth_images is not None
        assert sample.inputs.joint_positions is not None

        # Check all output types are present
        assert sample.outputs.joint_target_positions is not None
        assert sample.outputs.end_effectors is not None

    @patch("neuracore.login")
    def test_all_modalities_combination(
        self, mock_login, mock_synchronized_dataset, mock_tokenizer, temp_cache_dir
    ):
        """Test combination of all data modalities."""
        all_types = [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.RGB_IMAGE,
            DataType.DEPTH_IMAGE,
            DataType.POINT_CLOUD,
            DataType.END_EFFECTORS,
            DataType.POSES,
            DataType.LANGUAGE,
            DataType.CUSTOM,
        ]

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=all_types,
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=2,
            cache_dir=temp_cache_dir,
            tokenize_text=mock_tokenizer,
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None
            sample = dataset.load_sample(episode_idx=0, timestep=0)

        # Check all modalities are present
        assert sample.inputs.joint_positions is not None
        assert sample.inputs.joint_velocities is not None
        assert sample.inputs.rgb_images is not None
        assert sample.inputs.depth_images is not None
        assert sample.inputs.point_clouds is not None
        assert sample.inputs.end_effectors is not None
        assert sample.inputs.poses is not None
        assert sample.inputs.language_tokens is not None
        assert sample.inputs.custom_data is not None


class TestErrorRecovery:
    """Test error recovery and robustness."""

    def test_error_count_tracking(self, mock_synchronized_dataset, temp_cache_dir):
        """Test error count tracking in parent class."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        # Initial error count should be 0
        assert dataset._error_count == 0
        assert dataset._max_error_count == 100

    @patch("neuracore.login")
    def test_cache_corruption_recovery(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test recovery from cache corruption."""
        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        # Create a corrupted cache file
        cache_file = dataset.cache_dir / "ep_0_frame_0.pt"
        cache_file.write_text("corrupted data")

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None

            # Should handle corrupted cache gracefully and regenerate
            with patch("torch.load") as mock_load:
                mock_load.side_effect = Exception("Corrupted cache")

                # Should fall back to regenerating the sample
                sample = dataset.load_sample(episode_idx=0, timestep=0)
                assert sample is not None


@pytest.mark.integration
class TestIntegrationWithPyTorchDataLoader:
    """Integration tests with PyTorch DataLoader."""

    @patch("neuracore.login")
    def test_dataloader_with_collate_fn(
        self, mock_login, mock_synchronized_dataset, temp_cache_dir
    ):
        """Test DataLoader with custom collate function."""
        from torch.utils.data import DataLoader

        dataset = PytorchSynchronizedDataset(
            synchronized_dataset=mock_synchronized_dataset,
            input_data_types=[DataType.JOINT_POSITIONS, DataType.RGB_IMAGE],
            output_data_types=[DataType.JOINT_TARGET_POSITIONS],
            output_prediction_horizon=3,
            cache_dir=temp_cache_dir,
        )

        dataloader = DataLoader(
            dataset, batch_size=3, shuffle=False, collate_fn=dataset.collate_fn
        )

        with patch.object(dataset, "_memory_monitor") as mock_monitor:
            mock_monitor.check_memory.return_value = None

            batch = next(iter(dataloader))
            assert isinstance(batch, BatchedTrainingSamples)
            assert len(batch) == 3

            # Check shapes are correct for batching
            assert batch.inputs.joint_positions.data.shape[0] == 3
            assert batch.inputs.rgb_images.data.shape[0] == 3
