"""Depth image utility functions for encoding and decoding depth images."""

import numpy as np

from neuracore.core.utils.cmaps import INFERNO_CMAP

MAX_DEPTH = 10.0
MAX_DEPTH_VISUALIZATION_MULTIPLIER = 1.5
DEPTH_PREVIEW_SHIFT_M = 1.0
DEPTH_PREVIEW_MIN_M = 0.05
DEPTH_PREVIEW_MAX_M = 6.55
DEPTH_PREVIEW_LEVELS = 256


def depth_to_rgb_storage(depth_img: np.ndarray) -> np.ndarray:
    """Convert a depth image (in meters) to an RGB image (uint8) for storage.

    This encodes depth values across all three channels to maximize precision.

    The encoding is done as follows:
    - Depth values are clipped to the range [0, MAX_DEPTH].
    - Depth values are normalized to the range [0, 1].
    - The normalized depth is scaled to a 24-bit range (0 to 2^24 - 1).
    - Each channel (R, G, B) is filled with the corresponding part of the depth value.
    - The resulting RGB image is uint8 with depth encoded across all channels.

    Args:
        depth_img: Depth image in meters as float32 with shape (H, W)

    Returns:
        rgb_img: uint8 RGB image with depth encoded across all channels
    """
    if len(depth_img.shape) != 2:
        raise ValueError("depth_img must be a 2D array with shape (H, W)")

    # Clip depths to the maximum range, convert to float32 to make sure scaling works
    clipped_depth = np.clip(depth_img, 0, MAX_DEPTH).astype(np.float32)

    # Normalize to 0-1 range
    normalized_depth = clipped_depth / MAX_DEPTH

    # Scale to 24-bit precision (8 bits per channel × 3 channels)
    depth_scaled = normalized_depth * (2**24 - 1)

    # Extract the contribution for each channel
    r = np.floor(depth_scaled / (256 * 256)).astype(np.uint8)
    g = np.floor((depth_scaled / 256) % 256).astype(np.uint8)
    b = np.floor(depth_scaled % 256).astype(np.uint8)

    # Stack channels to create RGB image
    rgb_img = np.stack([r, g, b], axis=-1)

    return rgb_img


def rgb_to_depth_storage(rgb_img: np.ndarray) -> np.ndarray:
    """Convert an RGB-encoded depth image from storage back to a depth image in meters.

    Decoding is done by reversing the encoding process used in depth_to_rgb_storage.

    Args:
        rgb_img: uint8 RGB image with depth encoded across channels

    Returns:
        depth_img: Depth image in meters as float32
    """
    # Convert back to original depth
    r, g, b = rgb_img[..., 0], rgb_img[..., 1], rgb_img[..., 2]

    depth_value = (
        r.astype(np.float32) * 256 * 256
        + g.astype(np.float32) * 256
        + b.astype(np.float32)
    )

    # Convert normalized values back to meters
    depth_img = (depth_value / (2**24 - 1)) * MAX_DEPTH

    return depth_img


def depth_to_rgb_visualization(
    depth_img: np.ndarray, max_depth: float = MAX_DEPTH
) -> np.ndarray:
    """Convert a depth image (in meters) to an RGB image (uint8) for visualization.

    This encodes depth values across all three channels using the inferno colormap.

    Args:
        depth_img: Depth image in meters as float32 with shape (H, W)

    Returns:
        rgb_img: uint8 RGB image with depth encoded across all channels
    """
    if len(depth_img.shape) != 2:
        raise ValueError("depth_img must be a 2D array with shape (H, W)")

    # Increase the max depth to accommodate scene changes
    max_depth = MAX_DEPTH_VISUALIZATION_MULTIPLIER * max_depth

    # Clip depths to the maximum range
    clipped_depth = np.clip(depth_img, 0, max_depth)

    # Mask for zero (invalid/missing) depth values
    zero_mask = depth_img == 0

    # Normalize to 0-1 range, invert so that closer depths are brighter
    normalized_depth = 1 - (clipped_depth / max_depth)

    # Apply inferno colormap
    indices = np.clip((normalized_depth * 255).astype(np.int32), 0, 255)
    rgb_img = (INFERNO_CMAP[indices] * 255).astype(np.uint8)

    rgb_img[zero_mask] = 0

    return rgb_img


def decode_depth_frame(payload: bytes) -> np.ndarray:
    """Decode one JPEG-XL depth frame.

    Args:
        payload: JPEG-XL codestream of the frame.

    Returns:
        The uint16 frame of sensor units.

    Raises:
        ValueError: If the frame decodes to another dtype.
    """
    import imagecodecs

    frame = np.asarray(imagecodecs.jpegxl_decode(payload))
    if frame.dtype != np.uint16:
        raise ValueError(f"Depth frame decodes as {frame.dtype}, expected uint16")
    return frame


def depth_to_log_gray(
    depth_m: np.ndarray,
    shift: float = DEPTH_PREVIEW_SHIFT_M,
    min_m: float = DEPTH_PREVIEW_MIN_M,
    max_m: float = DEPTH_PREVIEW_MAX_M,
    levels: int = DEPTH_PREVIEW_LEVELS,
) -> np.ndarray:
    """Map depth in meters to grey levels along a log curve.

    The curve is the one the data daemon writes into the depth viewer video.
    Zero, negative and non-finite depth map to level 0, the no-return level.
    Valid depth clips to min_m and max_m and maps to levels 1 to levels - 1
    along log(depth + shift), rounding half away from zero.

    Args:
        depth_m: Depth in meters of any shape.
        shift: Offset in meters added before the logarithm.
        min_m: Nearest depth that keeps its own level.
        max_m: Farthest depth that keeps its own level.
        levels: Number of grey levels including the no-return level.

    Returns:
        Grey levels as uint8 when levels fit in 8 bits, else uint16.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    low = np.log(min_m + shift)
    span = np.log(max_m + shift) - low
    clipped = np.clip(np.where(valid, depth, min_m), min_m, max_m)
    scaled = (np.log(clipped + shift) - low) / span * (levels - 2)
    whole = np.floor(scaled)
    rounded = whole + (scaled - whole >= 0.5)
    level = np.where(valid, 1 + rounded, 0)
    return level.astype(np.uint8 if levels <= 256 else np.uint16)


def log_gray_to_depth(
    level: np.ndarray,
    shift: float = DEPTH_PREVIEW_SHIFT_M,
    min_m: float = DEPTH_PREVIEW_MIN_M,
    max_m: float = DEPTH_PREVIEW_MAX_M,
    levels: int = DEPTH_PREVIEW_LEVELS,
) -> np.ndarray:
    """Map log-curve grey levels back to depth in meters.

    Args:
        level: Grey levels from depth_to_log_gray or the depth viewer video.
        shift: Offset in meters added before the logarithm.
        min_m: Nearest depth that keeps its own level.
        max_m: Farthest depth that keeps its own level.
        levels: Number of grey levels including the no-return level.

    Returns:
        Depth in meters as float32, with 0 where the level is 0.
    """
    level_f = np.asarray(level, dtype=np.float64)
    low = np.log(min_m + shift)
    span = np.log(max_m + shift) - low
    depth = np.exp(low + (level_f - 1) / (levels - 2) * span) - shift
    return np.where(level_f > 0, depth, 0.0).astype(np.float32)
