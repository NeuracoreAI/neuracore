"""Preprocessing method implementations."""

from __future__ import annotations

import torch
from neuracore_types import BatchedDepthData, BatchedNCData, BatchedRGBData


def resize_pad(
    batched_data: BatchedRGBData | BatchedDepthData,
    size: list[int] | tuple[int, int] = (224, 224),
) -> BatchedNCData:
    """Resize and pad a batched RGB/depth item to a fixed spatial size."""
    if len(size) != 2:
        raise ValueError("resize_pad expects size as [height, width].")
    target_h, target_w = int(size[0]), int(size[1])
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
    # NOTE: Keep interpolation mode tied to modality semantics:
    # - RGB uses bilinear interpolation for smooth image resizing.
    # - Depth uses nearest-neighbor interpolation to avoid
    #   inventing intermediate values.
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
        resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0
    )
    batched_data.frame = padded.reshape(
        batch_size, time_steps, channels, target_h, target_w
    )
    return batched_data
