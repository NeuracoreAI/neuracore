"""Gaussian noise preprocessing method (training augmentation)."""

from __future__ import annotations

import torch
from neuracore_types import BatchedNCData, BatchedRGBData, DataType

from ..base import PreprocessingMethod


class GaussianNoise(PreprocessingMethod):
    """Add i.i.d. Gaussian noise to RGB frames.

    Noise ``std`` is in pixel units for float frames in ``[0, 255]``.
    """

    # randn_like over a full frame is expensive; far cheaper batched on the device
    on_cpu = False

    def __init__(
        self,
        std: float = 5.0,
        clip: bool = True,
    ) -> None:
        """Initialize Gaussian noise parameters.

        Args:
            std: Standard deviation of noise in pixel units (0–255 scale).
            clip: When True, clamp the result to ``[0, 255]``.
        """
        if std < 0.0:
            raise ValueError("gaussian_noise expects non-negative std.")
        self.std = std
        self.clip = clip

    @staticmethod
    def allowed_data_types() -> frozenset[DataType]:
        """Return data types supported by this method."""
        return frozenset({DataType.RGB_IMAGES})

    def __call__(self, data: BatchedNCData) -> BatchedNCData:
        """Add Gaussian noise to RGB frames.

        Args:
            data: Batched RGB data with float frames in ``[0, 255]``.

        Returns:
            The same batched data with noise added to ``frame``.
        """
        if not isinstance(data, BatchedRGBData):
            raise TypeError(
                f"Unsupported batched data type for gaussian_noise: {type(data)!r}"
            )

        if self.std == 0.0:
            return data

        frame = data.frame + torch.randn_like(data.frame) * self.std
        if self.clip:
            frame = frame.clamp(0.0, 255.0)
        data.frame = frame
        return data
