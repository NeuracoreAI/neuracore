"""Normalizer module for standardizing data across algorithms.

This module provides Normalizer classes that handle both
mean/std and min/max normalization for multiple data types,
with support for PyTorch's register_buffer for proper device handling.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Normalizer(nn.Module, ABC):
    """Base normalizer class for multiple data types.

    This class provides the common interface for normalization.
    Subclasses implement specific normalization strategies.
    """

    def __init__(
        self,
        name: str,
    ) -> None:
        """Initialize a Normalizer with optional statistics.

        Args:
            name: Name of the normalizer.
            statistics: Optional list of DataItemStats objects.
        """
        super().__init__()
        self._name = name

    @abstractmethod
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize using a specific normalizer.

        Args:
            data: Input tensor to normalize.

        Returns:
            Normalized tensor.

        """

    @abstractmethod
    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Unnormalize using a specific normalizer.

        Args:
            data: Normalized tensor to unnormalize.

        Returns:
            Unnormalized tensor.

        """


class MeanStdNormalizer(Normalizer):
    """Mean/std normalization normalizer.

    This class manages normalization using mean and standard deviation
    for joint states (joint_positions, joint_velocities, joint_torques)
    and actions (target positions).
    It uses register_buffer to ensure statistics move with the model.

    Args:
        name: Name of the normalizer.
        statistics: Optional list of DataItemStats objects with .mean/.std attributes.

    Raises:
        ValueError: If statistics are not provided.
    """

    def __init__(
        self,
        name: str,
        statistics: list[Any],
    ) -> None:
        """Initialize a MeanStdNormalizer with optional statistics.

        Args:
            statistics: Optional list of DataItemStats objects with
                .mean/.std attributes.
            name: Name of the normalizer.

        Raises:
            ValueError: If statistics are not provided.
        """
        super().__init__(name=name)
        if statistics:
            combined_mean: list[float] = []
            combined_std: list[float] = []
            for s in statistics:
                combined_mean.extend(s.mean)
                combined_std.extend(s.std)
            self.register_buffer(
                f"{self._name}_mean", torch.tensor(combined_mean, dtype=torch.float32)
            )
            self.register_buffer(
                f"{self._name}_std", torch.tensor(combined_std, dtype=torch.float32)
            )
        else:
            raise ValueError(f"Statistics are not provided for {self._name}")

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize using mean/std normalization.

        Args:
            name: Name of the normalizer to use.

        Returns:
            Normalized tensor.
        """
        mean = getattr(self, f"{self._name}_mean")
        std = getattr(self, f"{self._name}_std")
        std = torch.clamp(std, min=1e-6)
        return (data - mean) / std

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Unnormalize using mean/std normalization.

        Args:
            data: Normalized tensor to unnormalize.

        Returns:
            Unnormalized tensor.

        Raises:
            ValueError: If mean or std is not found.
        """
        mean = getattr(self, f"{self._name}_mean")
        std = getattr(self, f"{self._name}_std")
        return data * std + mean


class MinMaxNormalizer(Normalizer):
    """Min/max normalization normalizer.

    This class manages normalization using min and max values
    for joint states (joint_positions, joint_velocities, joint_torques)
    and actions (target positions).
    It uses register_buffer to ensure statistics move with the model.

    Args:
        name: Name of the normalizer.
        statistics: Optional list of DataItemStats objects with .min/.max attributes.

    Raises:
        ValueError: If statistics are not provided.
    """

    def __init__(
        self,
        name: str,
        statistics: list[Any],
    ) -> None:
        """Initialize a MinMaxNormalizer with optional statistics.

        Args:
            statistics: Optional list of DataItemStats objects with
                .min/.max attributes.
            name: Name of the normalizer.

        Raises:
            ValueError: If statistics are not provided.
        """
        super().__init__(name=name)
        if statistics:
            combined_min: list[float] = []
            combined_max: list[float] = []
            for s in statistics:
                combined_min.extend(s.min)
                combined_max.extend(s.max)
            self.register_buffer(
                f"{self._name}_min", torch.tensor(combined_min, dtype=torch.float32)
            )
            self.register_buffer(
                f"{self._name}_max", torch.tensor(combined_max, dtype=torch.float32)
            )
        else:
            raise ValueError(f"Statistics are not provided for {self._name}")

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize using min/max normalization.

        Args:
            data: Input tensor to normalize.

        Returns:
            Normalized tensor (scaled to [-1, 1]).
        """
        min_val = getattr(self, f"{self._name}_min")
        max_val = getattr(self, f"{self._name}_max")
        range_val = max_val - min_val
        # Avoid division by zero
        range_val = torch.clamp(range_val, min=1e-6)
        # Scale to [-1, 1]
        return 2.0 * (data - min_val) / range_val - 1.0

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Unnormalize using min/max normalization.

        Args:
            data: Normalized tensor to unnormalize (expected to be in [-1, 1]).

        Returns:
            Unnormalized tensor.
        """
        min_val = getattr(self, f"{self._name}_min")
        max_val = getattr(self, f"{self._name}_max")
        range_val = max_val - min_val
        return (data + 1.0) / 2.0 * range_val + min_val
