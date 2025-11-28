"""Normalizer module for standardizing data across algorithms.

This module provides a Normalizer class that handles both
mean/std and min/max normalization for multiple data types,
with support for PyTorch's register_buffer for proper device handling.
"""

from typing import Any, List

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """A normalizer for multiple data types.

    This class manages normalization for joint states (joint_positions,
    joint_velocities, joint_torques) and actions (target positions).
    It uses register_buffer to ensure statistics move with the model.

    The normalization type is determined by the normalization_type parameter:
    "mean_std" for mean/std normalization, "min_max" for min/max normalization.
    """

    def __init__(self) -> None:
        """Initialize an empty Normalizer."""
        super().__init__()
        self._norm_types: dict[str, str] = {}

    def add_statistics(
        self,
        name: str,
        stats: List[Any],
        normalization_type: str = "mean_std",
    ) -> "Normalizer":
        """Add normalization by combining multiple DataItemStats objects.

        Args:
            name: Name/key for the normalizer.
            stats: List of objects with .mean/.std or .min/.max attributes.
            normalization_type: Either "mean_std" or "min_max".

        Returns:
            Self for method chaining.
        """
        self._norm_types[name] = normalization_type

        if normalization_type == "mean_std":
            combined_mean: List[float] = []
            combined_std: List[float] = []
            for s in stats:
                combined_mean.extend(s.mean)
                combined_std.extend(s.std)
            self.register_buffer(
                f"{name}_mean", torch.tensor(combined_mean, dtype=torch.float32)
            )
            self.register_buffer(
                f"{name}_std", torch.tensor(combined_std, dtype=torch.float32)
            )
        elif normalization_type == "min_max":
            combined_min: List[float] = []
            combined_max: List[float] = []
            for s in stats:
                combined_min.extend(s.min)
                combined_max.extend(s.max)
            self.register_buffer(
                f"{name}_min", torch.tensor(combined_min, dtype=torch.float32)
            )
            self.register_buffer(
                f"{name}_max", torch.tensor(combined_max, dtype=torch.float32)
            )
        else:
            raise ValueError(f"Unknown normalization_type: {normalization_type}")

        return self

    def normalize(self, name: str, data: torch.Tensor) -> torch.Tensor:
        """Normalize using a specific normalizer.

        Args:
            name: Name of the normalizer to use.
            data: Input tensor to normalize.

        Returns:
            Normalized tensor.
        """
        if name not in self._norm_types:
            raise KeyError(f"Normalizer '{name}' not found. Available: {self.keys()}")

        if self._norm_types[name] == "mean_std":
            mean = getattr(self, f"{name}_mean")
            std = getattr(self, f"{name}_std")
            return (data - mean) / std
        elif self._norm_types[name] == "min_max":
            min_val = getattr(self, f"{name}_min")
            max_val = getattr(self, f"{name}_max")
            range_val = max_val - min_val
            # Avoid division by zero
            range_val = torch.clamp(range_val, min=1e-8)
            # Scale to [-1, 1]
            return 2.0 * (data - min_val) / range_val - 1.0

    def unnormalize(self, name: str, data: torch.Tensor) -> torch.Tensor:
        """Unnormalize using a specific normalizer.

        Args:
            name: Name of the normalizer to use.
            data: Normalized tensor to unnormalize.

        Returns:
            Unnormalized tensor.

        Raises:
            KeyError: If normalizer with given name doesn't exist.
        """
        if name not in self._norm_types:
            raise KeyError(f"Normalizer '{name}' not found. Available: {self.keys()}")

        if self._norm_types[name] == "mean_std":
            mean = getattr(self, f"{name}_mean")
            std = getattr(self, f"{name}_std")
            return data * std + mean
        elif self._norm_types[name] == "min_max":
            min_val = getattr(self, f"{name}_min")
            max_val = getattr(self, f"{name}_max")
            range_val = max_val - min_val
            return (data + 1.0) / 2.0 * range_val + min_val
        else:
            raise ValueError(
                f"Unsupported normalization_type: {self._norm_types[name]}"
            )

    def __contains__(self, name: str) -> bool:
        """Check if a normalizer exists."""
        return name in self._norm_types

    def keys(self) -> list[str]:
        """Get all normalizer names."""
        return list(self._norm_types.keys())
