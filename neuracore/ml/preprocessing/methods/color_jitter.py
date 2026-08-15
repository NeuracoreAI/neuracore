"""Color jitter preprocessing method (training augmentation)."""

from __future__ import annotations

import torchvision.transforms as T
from neuracore_types import BatchedNCData, BatchedRGBData, DataType

from ..base import PreprocessingMethod, PreprocessingStage


class ColorJitter(PreprocessingMethod):
    """Randomly jitter brightness, contrast, saturation, and hue of RGB frames.

    Expects float frames in ``[0, 255]`` (as produced by ``BatchedRGBData``
    once it is on the device), applies torchvision color jitter in ``[0, 1]``,
    then scales back.
    """

    # Elementwise and shape-preserving, so it does not need to run before
    # collation. On the device it runs once per batch instead of once per
    # frame on a worker CPU shared with every other worker.
    stage = PreprocessingStage.DEVICE

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
        # torchvision's hue adjustment permutes channels internally and hands
        # back a non-contiguous view. This method changes colour, not layout,
        # so normalise it: downstream encoders flatten frames with .view, which
        # requires contiguity. Collation used to hide this by re-materialising
        # the tensor with torch.cat, but jitter now runs after collation.
        data.frame = (jittered * 255.0).clamp(0.0, 255.0).contiguous()
        return data
