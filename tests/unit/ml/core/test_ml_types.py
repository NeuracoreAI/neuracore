"""Comprehensive test suite for Neuracore data type support.

This module provides tests for all data types including ML data structures,
dataset classes, encoder modules, and end-to-end functionality.
"""

import pytest
import torch
from neuracore_types import DataType

from neuracore.ml.core.ml_types import BatchedInferenceInputs, MaskableData


class TestMaskableData:
    """Test suite for MaskableData class."""

    def test_init(self):
        """Test MaskableData initialization."""
        data = torch.randn(5, 10)
        mask = torch.ones(5, 10)
        maskable = MaskableData(data, mask)

        assert torch.equal(maskable.data, data)
        assert torch.equal(maskable.mask, mask)

    def test_to_device(self):
        """Test moving MaskableData to different device."""
        data = torch.randn(5, 10)
        mask = torch.ones(5, 10)
        maskable = MaskableData(data, mask)

        device = torch.device("cpu")
        moved = maskable.to(device)

        assert moved.data.device == device
        assert moved.mask.device == device


class TestBatchedInferenceSamples:
    """Test suite for BatchedInferenceSamples class."""

    def test_init_empty(self):
        """Test empty BatchedInferenceSamples initialization."""
        batch = BatchedInferenceInputs()

        assert len(batch.inputs) == 0

    def test_init_with_data(self):
        """Test BatchedInferenceSamples initialization with data."""
        joint_data = MaskableData(torch.randn(4, 6), torch.ones(4, 6))
        rgb_data = MaskableData(torch.randn(4, 2, 3, 224, 224), torch.ones(4, 2))

        batch = BatchedInferenceInputs(
            inputs={
                DataType.JOINT_POSITIONS: {"default": joint_data},
                DataType.RGB_IMAGES: {"camera_0": rgb_data},
            }
        )

        assert DataType.JOINT_POSITIONS in batch.inputs
        assert DataType.RGB_IMAGES in batch.inputs
        assert torch.equal(
            batch.inputs[DataType.JOINT_POSITIONS]["default"].data, joint_data.data
        )

    def test_len(self):
        """Test BatchedInferenceSamples length calculation."""
        joint_data = MaskableData(torch.randn(4, 6), torch.ones(4, 6))
        batch = BatchedInferenceInputs(
            inputs={DataType.JOINT_POSITIONS: {"default": joint_data}}
        )

        assert len(batch) == 4

    def test_len_empty_raises(self):
        """Test that empty BatchedInferenceSamples raises error for len."""
        batch = BatchedInferenceInputs()

        with pytest.raises(ValueError, match="No tensor found"):
            len(batch)

    def test_to_device(self):
        """Test moving BatchedInferenceSamples to device."""
        joint_data = MaskableData(torch.randn(4, 6), torch.ones(4, 6))
        custom_data = MaskableData(torch.randn(4, 10), torch.ones(4, 10))

        batch = BatchedInferenceInputs(
            inputs={
                DataType.JOINT_POSITIONS: {"default": joint_data},
                DataType.CUSTOM: {"sensor1": custom_data},
            }
        )

        device = torch.device("cpu")
        moved = batch.to(device)

        assert moved.inputs[DataType.JOINT_POSITIONS]["default"].data.device == device
        assert moved.inputs[DataType.CUSTOM]["sensor1"].data.device == device

    def test_init_all_data_types(self):
        """Test initialization with all supported data types."""
        batch_size = 4

        # Create sample data for each type
        joint_pos = MaskableData(torch.randn(batch_size, 6), torch.ones(batch_size, 6))
        joint_vel = MaskableData(torch.randn(batch_size, 6), torch.ones(batch_size, 6))
        joint_torq = MaskableData(torch.randn(batch_size, 6), torch.ones(batch_size, 6))
        joint_target = MaskableData(
            torch.randn(batch_size, 7), torch.ones(batch_size, 7)
        )
        gripper = MaskableData(torch.randn(batch_size, 2), torch.ones(batch_size, 2))
        poses = MaskableData(torch.randn(batch_size, 12), torch.ones(batch_size, 12))
        rgb_imgs = MaskableData(
            torch.randn(batch_size, 2, 3, 224, 224), torch.ones(batch_size, 2)
        )
        depth_imgs = MaskableData(
            torch.randn(batch_size, 2, 1, 224, 224), torch.ones(batch_size, 2)
        )
        point_clouds = MaskableData(
            torch.randn(batch_size, 1, 1024, 3), torch.ones(batch_size, 1)
        )
        lang_tokens = MaskableData(
            torch.randint(0, 1000, (batch_size, 50)), torch.ones(batch_size, 50)
        )
        custom = MaskableData(torch.randn(batch_size, 10), torch.ones(batch_size, 10))

        batch = BatchedInferenceInputs(
            inputs={
                DataType.JOINT_POSITIONS: {"default": joint_pos},
                DataType.JOINT_VELOCITIES: {"default": joint_vel},
                DataType.JOINT_TORQUES: {"default": joint_torq},
                DataType.JOINT_TARGET_POSITIONS: {"default": joint_target},
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {"default": gripper},
                DataType.POSES: {"default": poses},
                DataType.RGB_IMAGES: {"camera_0": rgb_imgs},
                DataType.DEPTH_IMAGES: {"camera_0": depth_imgs},
                DataType.POINT_CLOUDS: {"lidar_0": point_clouds},
                DataType.LANGUAGE: {"default": lang_tokens},
                DataType.CUSTOM: {"sensor1": custom},
            }
        )

        assert len(batch) == batch_size
        assert DataType.JOINT_POSITIONS in batch.inputs
        assert DataType.POSES in batch.inputs
        assert DataType.DEPTH_IMAGES in batch.inputs
        assert DataType.POINT_CLOUDS in batch.inputs
        assert "sensor1" in batch.inputs[DataType.CUSTOM]

    def test_combine_for_data_type(self):
        """Test combining multiple named inputs for a data type."""
        batch_size = 4

        # Create multiple cameras
        camera_1 = MaskableData(
            torch.randn(batch_size, 3, 224, 224), torch.ones(batch_size, 1)
        )
        camera_2 = MaskableData(
            torch.randn(batch_size, 3, 224, 224), torch.ones(batch_size, 1)
        )

        batch = BatchedInferenceInputs(
            inputs={
                DataType.RGB_IMAGES: {
                    "camera_0": camera_1,
                    "camera_1": camera_2,
                }
            }
        )

        # Combine all RGB images
        combined = batch.combine_for_data_type(DataType.RGB_IMAGES)

        assert isinstance(combined, MaskableData)
        # Data should be concatenated along last dimension
        assert (
            combined.data.shape[-1] == camera_1.data.shape[-1] + camera_2.data.shape[-1]
        )
        # Mask should be stacked
        assert combined.mask.shape[0] == 2  # Two cameras

    def test_combine_for_missing_data_type(self):
        """Test that combining a missing data type raises error."""
        batch = BatchedInferenceInputs(
            inputs={
                DataType.JOINT_POSITIONS: {
                    "default": MaskableData(torch.randn(4, 6), torch.ones(4, 6))
                }
            }
        )

        with pytest.raises(ValueError, match="No inputs found for data type"):
            batch.combine_for_data_type(DataType.RGB_IMAGES)
