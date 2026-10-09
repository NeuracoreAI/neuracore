"""Depth frame decoding and the depth viewer log curve."""

import numpy as np

DEPTH_PREVIEW_SHIFT_M = 1.0
DEPTH_PREVIEW_MIN_M = 0.05
DEPTH_PREVIEW_MAX_M = 6.55
DEPTH_PREVIEW_LEVELS = 256


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
