"""Comprehensive test suite for normalizer module.

This module provides tests for all normalizer classes including
MeanStdNormalizer and MinMaxNormalizer, testing initialization,
normalization, unnormalization, and edge cases.
"""

import numpy as np
import pytest
import torch
from neuracore_types import DataItemStats

from neuracore.ml.algorithm_utils.normalizer import (
    MeanStdNormalizer,
    MinMaxNormalizer,
    Normalizer,
)


class TestNormalizer:
    """Test suite for base Normalizer class."""

    def test_init(self):
        """Test Normalizer initialization."""
        normalizer = Normalizer(name="test_normalizer")
        assert normalizer._name == "test_normalizer"

    def test_normalize_not_implemented(self):
        """Test that normalize raises NotImplementedError."""
        normalizer = Normalizer(name="test")
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            normalizer.normalize(torch.randn(5, 10))

    def test_unnormalize_not_implemented(self):
        """Test that unnormalize raises NotImplementedError."""
        normalizer = Normalizer(name="test")
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            normalizer.unnormalize(torch.randn(5, 10))


class TestMeanStdNormalizer:
    """Test suite for MeanStdNormalizer class."""

    @pytest.fixture
    def sample_stats(self):
        """Create sample DataItemStats with mean and std."""
        return DataItemStats(
            mean=np.array([0.0, 1.0, 2.0]),
            std=np.array([1.0, 2.0, 3.0]),
        )

    @pytest.fixture
    def multiple_stats(self):
        """Create multiple DataItemStats for combined normalization."""
        return [
            DataItemStats(mean=np.array([0.0, 1.0]), std=np.array([1.0, 2.0])),
            DataItemStats(mean=np.array([2.0, 3.0]), std=np.array([3.0, 4.0])),
        ]

    def test_init_with_statistics(self, sample_stats):
        """Test MeanStdNormalizer initialization with statistics."""
        normalizer = MeanStdNormalizer(name="test", statistics=[sample_stats])
        assert hasattr(normalizer, "test_mean")
        assert hasattr(normalizer, "test_std")
        assert torch.equal(normalizer.test_mean, torch.tensor([0.0, 1.0, 2.0]))
        assert torch.equal(normalizer.test_std, torch.tensor([1.0, 2.0, 3.0]))

    def test_init_without_statistics(self):
        """Test MeanStdNormalizer initialization without statistics raises error."""
        with pytest.raises(ValueError, match="Statistics are not provided"):
            MeanStdNormalizer(name="test", statistics=None)

    def test_init_with_empty_statistics(self):
        """Test MeanStdNormalizer initialization with empty statistics raises error."""
        with pytest.raises(ValueError, match="Statistics are not provided"):
            MeanStdNormalizer(name="test", statistics=[])

    def test_init_with_multiple_statistics(self, multiple_stats):
        """Test MeanStdNormalizer with multiple DataItemStats."""
        normalizer = MeanStdNormalizer(name="joint_states", statistics=multiple_stats)
        # Should combine means and stds
        expected_mean = torch.tensor([0.0, 1.0, 2.0, 3.0])
        expected_std = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.equal(normalizer.joint_states_mean, expected_mean)
        assert torch.equal(normalizer.joint_states_std, expected_std)

    def test_normalize(self, sample_stats):
        """Test normalization with mean/std."""
        normalizer = MeanStdNormalizer(name="test", statistics=[sample_stats])
        data = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        normalized = normalizer.normalize(data)

        # First sample: (0-0)/1, (1-1)/2, (2-2)/3 = [0, 0, 0]
        # Second sample: (1-0)/1, (2-1)/2, (3-2)/3 = [1, 0.5, 0.333...]
        expected = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, 1.0 / 3.0]])
        assert torch.allclose(normalized, expected, atol=1e-6)

    def test_normalize_with_small_std(self, sample_stats):
        """Test normalization handles small standard deviations."""
        # Modify stats to have very small std
        small_std_stats = DataItemStats(
            mean=np.array([0.0, 1.0, 2.0]),
            std=np.array([1.0, 1e-10, 3.0]),
        )
        normalizer = MeanStdNormalizer(name="test", statistics=[small_std_stats])
        data = torch.tensor([[0.0, 1.0, 2.0]])
        normalized = normalizer.normalize(data)

        # Should clamp std to 1e-8 minimum
        assert not torch.isinf(normalized).any()
        assert not torch.isnan(normalized).any()

    def test_unnormalize(self, sample_stats):
        """Test unnormalization with mean/std."""
        normalizer = MeanStdNormalizer(name="test", statistics=[sample_stats])
        normalized_data = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, 1.0 / 3.0]])
        unnormalized = normalizer.unnormalize(normalized_data)

        # Should recover original: data * std + mean
        # First: [0, 0, 0] * [1, 2, 3] + [0, 1, 2] = [0, 1, 2]
        # Second: [1, 0.5, 0.333] * [1, 2, 3] + [0, 1, 2] = [1, 2, 3]
        expected = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        assert torch.allclose(unnormalized, expected, atol=1e-6)

    def test_normalize_unnormalize_roundtrip(self, sample_stats):
        """Test that normalize and unnormalize are inverse operations."""
        normalizer = MeanStdNormalizer(name="test", statistics=[sample_stats])
        original_data = torch.tensor(
            [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]
        )

        normalized = normalizer.normalize(original_data)
        unnormalized = normalizer.unnormalize(normalized)

        assert torch.allclose(original_data, unnormalized, atol=1e-6)

    def test_device_handling(self, sample_stats):
        """Test that normalizer buffers move with model to device."""
        normalizer = MeanStdNormalizer(name="test", statistics=[sample_stats])
        device = torch.device("cpu")
        normalizer = normalizer.to(device)

        assert normalizer.test_mean.device == device
        assert normalizer.test_std.device == device

    def test_batch_processing(self, sample_stats):
        """Test normalization with different batch sizes."""
        normalizer = MeanStdNormalizer(name="test", statistics=[sample_stats])

        # Single sample
        single = torch.tensor([[0.0, 1.0, 2.0]])
        assert normalizer.normalize(single).shape == (1, 3)

        # Multiple samples
        batch = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        assert normalizer.normalize(batch).shape == (3, 3)

    def test_different_dimensions(self):
        """Test normalizer with different data dimensions."""
        # 1D stats
        stats_1d = DataItemStats(mean=np.array([5.0]), std=np.array([2.0]))
        normalizer_1d = MeanStdNormalizer(name="test", statistics=[stats_1d])
        data_1d = torch.tensor([[5.0], [7.0], [3.0]])
        normalized_1d = normalizer_1d.normalize(data_1d)
        assert normalized_1d.shape == (3, 1)
        assert torch.allclose(normalized_1d, torch.tensor([[0.0], [1.0], [-1.0]]))

        # High dimensional stats
        stats_high = DataItemStats(mean=np.array([0.0] * 10), std=np.array([1.0] * 10))
        normalizer_high = MeanStdNormalizer(name="test", statistics=[stats_high])
        data_high = torch.randn(5, 10)
        normalized_high = normalizer_high.normalize(data_high)
        assert normalized_high.shape == (5, 10)


class TestMinMaxNormalizer:
    """Test suite for MinMaxNormalizer class."""

    @pytest.fixture
    def sample_stats(self):
        """Create sample DataItemStats with min and max."""
        return DataItemStats(
            min=np.array([0.0, 1.0, 2.0]),
            max=np.array([2.0, 3.0, 4.0]),
        )

    @pytest.fixture
    def multiple_stats(self):
        """Create multiple DataItemStats for combined normalization."""
        return [
            DataItemStats(min=np.array([0.0, 1.0]), max=np.array([2.0, 3.0])),
            DataItemStats(min=np.array([2.0, 3.0]), max=np.array([4.0, 5.0])),
        ]

    def test_init_with_statistics(self, sample_stats):
        """Test MinMaxNormalizer initialization with statistics."""
        normalizer = MinMaxNormalizer(name="test", statistics=[sample_stats])
        assert hasattr(normalizer, "test_min")
        assert hasattr(normalizer, "test_max")
        assert torch.equal(normalizer.test_min, torch.tensor([0.0, 1.0, 2.0]))
        assert torch.equal(normalizer.test_max, torch.tensor([2.0, 3.0, 4.0]))

    def test_init_without_statistics(self):
        """Test MinMaxNormalizer initialization without statistics raises error."""
        with pytest.raises(ValueError, match="Statistics are not provided"):
            MinMaxNormalizer(name="test", statistics=None)

    def test_init_with_empty_statistics(self):
        """Test MinMaxNormalizer initialization with empty statistics raises error."""
        with pytest.raises(ValueError, match="Statistics are not provided"):
            MinMaxNormalizer(name="test", statistics=[])

    def test_init_with_multiple_statistics(self, multiple_stats):
        """Test MinMaxNormalizer with multiple DataItemStats."""
        normalizer = MinMaxNormalizer(name="actions", statistics=multiple_stats)
        # Should combine mins and maxs
        expected_min = torch.tensor([0.0, 1.0, 2.0, 3.0])
        expected_max = torch.tensor([2.0, 3.0, 4.0, 5.0])
        assert torch.equal(normalizer.actions_min, expected_min)
        assert torch.equal(normalizer.actions_max, expected_max)

    def test_normalize(self, sample_stats):
        """Test normalization with min/max scaling to [-1, 1]."""
        normalizer = MinMaxNormalizer(name="test", statistics=[sample_stats])
        data = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])

        normalized = normalizer.normalize(data)

        # Formula: 2.0 * (data - min) / (max - min) - 1.0
        # First: [0, 1, 2] -> [2*(0-0)/(2-0)-1, 2*(1-1)/(3-1)-1, 2*(2-2)/(4-2)-1]
        #        = [-1, -1, -1]
        # Second: [1, 2, 3] -> [2*(1-0)/(2-0)-1, 2*(2-1)/(3-1)-1, 2*(3-2)/(4-2)-1]
        #         = [0, 0, 0]
        # Third: [2, 3, 4] -> [2*(2-0)/(2-0)-1, 2*(3-1)/(3-1)-1, 2*(4-2)/(4-2)-1]
        #        = [1, 1, 1]
        expected = torch.tensor([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        assert torch.allclose(normalized, expected, atol=1e-6)

    def test_normalize_with_zero_range(self):
        """Test normalization handles zero range (min == max)."""
        # Stats where min == max (zero range)
        zero_range_stats = DataItemStats(
            min=np.array([1.0, 2.0, 3.0]),
            max=np.array([1.0, 2.0, 3.0]),  # Same as min
        )
        normalizer = MinMaxNormalizer(name="test", statistics=[zero_range_stats])
        data = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

        normalized = normalizer.normalize(data)

        # Should handle division by zero by clamping range to 1e-8
        # Result should be finite
        assert not torch.isinf(normalized).any()
        assert not torch.isnan(normalized).any()

    def test_unnormalize(self, sample_stats):
        """Test unnormalization with min/max."""
        normalizer = MinMaxNormalizer(name="test", statistics=[sample_stats])
        normalized_data = torch.tensor(
            [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        )

        unnormalized = normalizer.unnormalize(normalized_data)

        # Formula: (data + 1.0) / 2.0 * (max - min) + min
        # Should recover: [0, 1, 2], [1, 2, 3], [2, 3, 4]
        expected = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        assert torch.allclose(unnormalized, expected, atol=1e-6)

    def test_normalize_unnormalize_roundtrip(self, sample_stats):
        """Test that normalize and unnormalize are inverse operations."""
        normalizer = MinMaxNormalizer(name="test", statistics=[sample_stats])
        original_data = torch.tensor(
            [[0.5, 1.5, 2.5], [1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]
        )

        normalized = normalizer.normalize(original_data)
        unnormalized = normalizer.unnormalize(normalized)

        assert torch.allclose(original_data, unnormalized, atol=1e-6)

    def test_normalize_output_range(self, sample_stats):
        """Test that normalized output is in [-1, 1] range."""
        normalizer = MinMaxNormalizer(name="test", statistics=[sample_stats])
        # Test with values at boundaries and beyond
        data = torch.tensor(
            [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [-1.0, 0.0, 1.0], [3.0, 4.0, 5.0]]
        )

        normalized = normalizer.normalize(data)

        # Values within [min, max] should be in [-1, 1]
        # Values outside might be outside [-1, 1] but should be finite
        assert not torch.isinf(normalized).any()
        assert not torch.isnan(normalized).any()
        # First two rows should be in [-1, 1] since they're within bounds
        assert torch.all(normalized[0] >= -1.0 - 1e-6)
        assert torch.all(normalized[0] <= 1.0 + 1e-6)
        assert torch.all(normalized[1] >= -1.0 - 1e-6)
        assert torch.all(normalized[1] <= 1.0 + 1e-6)

    def test_device_handling(self, sample_stats):
        """Test that normalizer buffers move with model to device."""
        normalizer = MinMaxNormalizer(name="test", statistics=[sample_stats])
        device = torch.device("cpu")
        normalizer = normalizer.to(device)

        assert normalizer.test_min.device == device
        assert normalizer.test_max.device == device

    def test_batch_processing(self, sample_stats):
        """Test normalization with different batch sizes."""
        normalizer = MinMaxNormalizer(name="test", statistics=[sample_stats])

        # Single sample
        single = torch.tensor([[1.0, 2.0, 3.0]])
        assert normalizer.normalize(single).shape == (1, 3)

        # Multiple samples
        batch = torch.tensor([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        assert normalizer.normalize(batch).shape == (3, 3)

    def test_different_dimensions(self):
        """Test normalizer with different data dimensions."""
        # 1D stats
        stats_1d = DataItemStats(min=np.array([0.0]), max=np.array([10.0]))
        normalizer_1d = MinMaxNormalizer(name="test", statistics=[stats_1d])
        data_1d = torch.tensor([[5.0], [0.0], [10.0]])
        normalized_1d = normalizer_1d.normalize(data_1d)
        assert normalized_1d.shape == (3, 1)
        # 5.0 should normalize to 0, 0.0 to -1, 10.0 to 1
        assert torch.allclose(normalized_1d, torch.tensor([[0.0], [-1.0], [1.0]]))

        # High dimensional stats
        stats_high = DataItemStats(min=np.array([0.0] * 10), max=np.array([1.0] * 10))
        normalizer_high = MinMaxNormalizer(name="test", statistics=[stats_high])
        data_high = torch.rand(5, 10)
        normalized_high = normalizer_high.normalize(data_high)
        assert normalized_high.shape == (5, 10)


class TestNormalizerIntegration:
    """Integration tests for normalizer usage patterns."""

    def test_multiple_data_types_combined(self):
        """Test combining multiple data types in one normalizer."""
        # Simulate joint positions and velocities combined
        joint_positions = DataItemStats(
            mean=np.array([0.0, 1.0]), std=np.array([1.0, 2.0])
        )
        joint_velocities = DataItemStats(
            mean=np.array([0.0, -1.0]), std=np.array([0.5, 1.0])
        )

        normalizer = MeanStdNormalizer(
            name="joint_states", statistics=[joint_positions, joint_velocities]
        )

        # Should have combined 4 dimensions
        assert normalizer.joint_states_mean.shape == (4,)
        assert normalizer.joint_states_std.shape == (4,)

        # Test normalization
        data = torch.tensor([[0.0, 1.0, 0.0, -1.0]])  # [pos1, pos2, vel1, vel2]
        normalized = normalizer.normalize(data)
        # Should normalize each dimension according to its stats
        assert normalized.shape == (1, 4)

    def test_realistic_robot_joint_normalization(self):
        """Test with realistic robot joint statistics."""
        # Typical robot joint positions: -π to π
        joint_positions = DataItemStats(
            mean=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            std=np.array([1.57, 1.57, 1.57, 1.57, 1.57, 1.57]),  # ~π/2
        )

        normalizer = MeanStdNormalizer(name="joints", statistics=[joint_positions])

        # Simulate some joint positions
        joints = torch.tensor([[0.0, 1.57, -1.57, 0.785, -0.785, 0.0]])
        normalized = normalizer.normalize(joints)

        # Should normalize around 0 with std ~1.57
        assert normalized.shape == (1, 6)
        # Values near mean should normalize close to 0
        assert torch.abs(normalized[0, 0]) < 0.1  # 0.0 should normalize to ~0

    def test_action_normalization(self):
        """Test action normalization with target positions."""
        # Actions are typically in a smaller range
        actions = DataItemStats(
            min=np.array([-0.1, -0.1, -0.1]),
            max=np.array([0.1, 0.1, 0.1]),
        )

        normalizer = MinMaxNormalizer(name="actions", statistics=[actions])

        # Test action at center
        center_action = torch.tensor([[0.0, 0.0, 0.0]])
        normalized = normalizer.normalize(center_action)
        # Center should map to 0 in [-1, 1] range
        assert torch.allclose(normalized, torch.tensor([[0.0, 0.0, 0.0]]), atol=1e-6)

        # Test action at min
        min_action = torch.tensor([[-0.1, -0.1, -0.1]])
        normalized_min = normalizer.normalize(min_action)
        assert torch.allclose(
            normalized_min, torch.tensor([[-1.0, -1.0, -1.0]]), atol=1e-6
        )

        # Test action at max
        max_action = torch.tensor([[0.1, 0.1, 0.1]])
        normalized_max = normalizer.normalize(max_action)
        assert torch.allclose(
            normalized_max, torch.tensor([[1.0, 1.0, 1.0]]), atol=1e-6
        )
