"""Comprehensive test suite for Neuracore data type support.

This module provides tests for all data types including ML data structures,
dataset classes, encoder modules, and end-to-end functionality.
"""

import pytest
import torch

from neuracore.ml.core.ml_types import (
    BatchedData,
    BatchedInferenceSamples,
    MaskableData,
)


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


class TestBatchedData:
    """Test suite for BatchedData class."""

    def test_init_empty(self):
        """Test empty BatchedData initialization."""
        batch = BatchedData()

        assert batch.joint_positions is None
        assert batch.rgb_images is None
        assert batch.custom_data == {}

    def test_init_with_data(self):
        """Test BatchedData initialization with data."""
        joint_data = MaskableData(torch.randn(4, 6), torch.ones(4, 6))
        rgb_data = MaskableData(torch.randn(4, 2, 3, 224, 224), torch.ones(4, 2))

        batch = BatchedData(joint_positions=joint_data, rgb_images=rgb_data)

        assert batch.joint_positions is not None
        assert batch.rgb_images is not None
        assert torch.equal(batch.joint_positions.data, joint_data.data)

    def test_len(self):
        """Test BatchedData length calculation."""
        joint_data = MaskableData(torch.randn(4, 6), torch.ones(4, 6))
        batch = BatchedData(joint_positions=joint_data)

        assert len(batch) == 4

    def test_len_empty_raises(self):
        """Test that empty BatchedData raises error for len."""
        batch = BatchedData()

        with pytest.raises(ValueError, match="No tensor found"):
            len(batch)

    def test_to_device(self):
        """Test moving BatchedData to device."""
        joint_data = MaskableData(torch.randn(4, 6), torch.ones(4, 6))
        custom_data = {"sensor1": MaskableData(torch.randn(4, 10), torch.ones(4, 10))}

        batch = BatchedData(joint_positions=joint_data, custom_data=custom_data)

        device = torch.device("cpu")
        moved = batch.to(device)

        assert moved.joint_positions.data.device == device
        assert moved.custom_data["sensor1"].data.device == device


class TestBatchedInferenceSamples:
    """Test suite for BatchedInferenceSamples class."""

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
        end_eff = MaskableData(torch.randn(batch_size, 2), torch.ones(batch_size, 2))
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
        custom = {
            "sensor1": MaskableData(
                torch.randn(batch_size, 10), torch.ones(batch_size, 10)
            )
        }

        batch = BatchedInferenceSamples(
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            joint_torques=joint_torq,
            joint_target_positions=joint_target,
            end_effectors=end_eff,
            poses=poses,
            rgb_images=rgb_imgs,
            depth_images=depth_imgs,
            point_clouds=point_clouds,
            language_tokens=lang_tokens,
            custom_data=custom,
        )

        assert len(batch) == batch_size
        assert batch.joint_positions is not None
        assert batch.end_effectors is not None
        assert batch.poses is not None
        assert batch.depth_images is not None
        assert batch.point_clouds is not None
        assert "sensor1" in batch.custom_data
