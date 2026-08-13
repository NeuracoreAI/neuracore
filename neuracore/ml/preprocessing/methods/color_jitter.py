"""Color jitter preprocessing method (training augmentation)."""

from __future__ import annotations

import torchvision.transforms as T
from neuracore_types import BatchedNCData, BatchedRGBData, DataType

from ..base import PreprocessingMethod


class ColorJitter(PreprocessingMethod):
    """Randomly jitter brightness, contrast, saturation, and hue of RGB frames.

    Expects float frames in ``[0, 255]`` (as produced by ``BatchedRGBData``),
    applies torchvision color jitter in ``[0, 1]``, then scales back.
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.05,
    ) -> None:
        """Initialize color jitter parameters.

        Args:
            brightness: Max relative brightness change; samples uniformly from
                ``[max(0, 1 - brightness), 1 + brightness]``.
            contrast: Max relative contrast change; same sampling as brightness.
            saturation: Max relative saturation change; same sampling as brightness.
            hue: Max hue shift in ``[-hue, hue]`` (must be in ``[0, 0.5]``).
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self._jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    @staticmethod
    def allowed_data_types() -> frozenset[DataType]:
        """Return data types supported by this method."""
        return frozenset({DataType.RGB_IMAGES})

    def __call__(self, data: BatchedNCData) -> BatchedNCData:
        """Apply color jitter to RGB frames.

        Args:
            data: Batched RGB data with float frames in ``[0, 255]``.

        Returns:
            The same batched data with color-jittered ``frame``.
        """
        if not isinstance(data, BatchedRGBData):
            raise TypeError(
                f"Unsupported batched data type for color_jitter: {type(data)!r}"
            )

        jittered = self._jitter(data.frame / 255.0)
        data.frame = (jittered * 255.0).clamp(0.0, 255.0)
        return data
