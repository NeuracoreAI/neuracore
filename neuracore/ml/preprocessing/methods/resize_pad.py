"""Resize and pad preprocessing method."""

from __future__ import annotations

import torch
from neuracore_types import BatchedDepthData, BatchedNCData, BatchedRGBData, DataType

from ..base import PreprocessingMethod


class ResizePad(PreprocessingMethod):
    """Resize and pad RGB/depth tensors to fixed spatial size."""

    def __init__(self, size: list[int] | tuple[int, int] = (224, 224)) -> None:
        """Initialize resize target as (height, width)."""
        self.size = size

    @staticmethod
    def allowed_data_types() -> frozenset[DataType]:
        """Return data types supported by this method."""
        return frozenset({DataType.RGB_IMAGES, DataType.DEPTH_IMAGES})

    def __call__(self, data: BatchedNCData) -> BatchedNCData:
        """Resize while preserving aspect ratio, then center-pad to target size."""
        batched_data = data
        if len(self.size) != 2:
            raise ValueError("resize_pad expects size as [height, width].")
        target_h, target_w = int(self.size[0]), int(self.size[1])
        if target_h <= 0 or target_w <= 0:
            raise ValueError("resize_pad expects positive size values.")

        frame = batched_data.frame
        if frame.shape[-2:] == (target_h, target_w):
            return batched_data

        batch_size, time_steps, channels, src_h, src_w = frame.shape
        scale = min(target_h / src_h, target_w / src_w)
        resized_h = max(1, int(round(src_h * scale)))
        resized_w = max(1, int(round(src_w * scale)))

        reshaped = frame.reshape(batch_size * time_steps, channels, src_h, src_w)
        if isinstance(batched_data, BatchedRGBData):
            mode = "bilinear"
        elif isinstance(batched_data, BatchedDepthData):
            mode = "nearest"
        else:
            raise TypeError(
                f"Unsupported batched data type for resize_pad: {type(batched_data)!r}"
            )
        resized = torch.nn.functional.interpolate(
            reshaped,
            size=(resized_h, resized_w),
            mode=mode,
        )

        pad_h = target_h - resized_h
        pad_w = target_w - resized_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded = torch.nn.functional.pad(
            resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )
        batched_data.frame = padded.reshape(
            batch_size, time_steps, channels, target_h, target_w
        )
        return batched_data
