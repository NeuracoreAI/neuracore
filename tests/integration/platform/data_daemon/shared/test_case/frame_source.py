"""Synthetic camera frame generation for data-daemon tests.

RGB frames use realistic, textured content rather than a flat fill, so they
exercise the daemon's video pipeline the way a real camera would; depth
frames use a cheap analytic pattern instead, since they only need to check
the depth-to-RGB round trip. Both build their content once into a bank and
paint a per-frame identity from :func:`frame_code_base`.
"""

from __future__ import annotations

import threading

import numpy as np

from neuracore.core.utils.depth_utils import MAX_DEPTH
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DEPTH_FRAME_BASE_FRACTION,
    DEPTH_FRAME_BASE_MODULUS,
    DEPTH_FRAME_COL_FRACTION,
    DEPTH_FRAME_FLOOR_FRACTION,
    DEPTH_FRAME_ROW_FRACTION,
    DETAIL_FLAT,
    FRAME_BYTE_LENGTH,
    FRAME_COLOR_CHANNELS,
    FRAME_DEFAULT_FILL_VALUE,
    FRAME_GRID_SIZE,
    FRAME_HALF_DIVISOR,
    FRAME_MAX_COLOR_VALUE,
    DepthMode,
)

# cspell:ignore copyto perlin recnum camnum framenum
BANK_FRAMES = 12
SENSOR_NOISE_SIGMA = 1.0
SCENE_SEED = 7
SCENE_OCTAVES = ((0.5, 70.0), (1 / 6, 40.0), (1 / 16, 20.0), (1 / 48, 9.0))
SCENE_EDGE_COUNT = 40
SCENE_EDGE_AMPLITUDE = 55
SCENE_BLUR_FRACTION = 1 / 160
SCENE_CHANNEL_GAINS = (1.0, 0.92, 0.85)
SCENE_CHANNEL_OFFSETS = (0.0, 10.0, 18.0)
SCENE_PAN_FRACTION = 1 / 8
SCENE_PAN_MIN = 4
OBJECT_HEIGHT_FRACTION = 1 / 9
OBJECT_WIDTH_FRACTION = 1 / 12
OBJECT_PATH_AMPLITUDE = 0.45
OBJECT_BLEND = 0.45
OBJECT_LEVEL = 140.0

_BANK_CACHE: dict[tuple[str, int, int], np.ndarray] = {}
_BANK_CACHE_LOCK = threading.Lock()


FRAME_CODE_CONTEXT_STRIDE = 1_000_000_000
FRAME_CODE_RECORDING_STRIDE = 10_000_000
FRAME_CODE_CAMERA_STRIDE = 100_000


def frame_code_base(
    *, context_index: int, recording_ordinal: int, camera_index: int
) -> int:
    """Return the code painted into the first frame of one camera's recording.

    Args:
        recording_ordinal: Position within the context, not the
            daemon-assigned recording index.
    """
    return (
        context_index * FRAME_CODE_CONTEXT_STRIDE
        + recording_ordinal * FRAME_CODE_RECORDING_STRIDE
        + camera_index * FRAME_CODE_CAMERA_STRIDE
    )


def encode_frame_number(frame_num: int, frame: np.ndarray) -> np.ndarray:
    """Paint ``frame_num`` into the top-left :data:`FRAME_GRID_SIZE` grid.

    Args:
        frame_num: Must fit in :data:`FRAME_BYTE_LENGTH` bytes.
    """
    frame_bytes = frame_num.to_bytes(FRAME_BYTE_LENGTH, byteorder="big")

    for row in range(FRAME_GRID_SIZE):
        for col in range(FRAME_GRID_SIZE):
            idx = row * FRAME_GRID_SIZE + col
            if idx < len(frame_bytes):
                pixel_value = frame_bytes[idx]
                frame[row, col, 0] = pixel_value
                frame[row, col, 1] = FRAME_MAX_COLOR_VALUE - pixel_value
                frame[row, col, 2] = pixel_value // FRAME_HALF_DIVISOR

    return frame


def _pan_range(width: int, height: int) -> int:
    """Return the pan travel, in pixels, for a frame of this size."""
    return max(SCENE_PAN_MIN, int(min(width, height) * SCENE_PAN_FRACTION))


def _box_blur(field: np.ndarray, radius: int) -> np.ndarray:
    """Box-blur a 2-D field with a summed-area table, preserving its shape."""
    if radius <= 1:
        return field
    height, width = field.shape
    integral = np.cumsum(np.cumsum(np.pad(field, ((1, 0), (1, 0))), axis=0), axis=1)
    blurred = (
        integral[radius:, radius:]
        - integral[:-radius, radius:]
        - integral[radius:, :-radius]
        + integral[:-radius, :-radius]
    ) / (radius * radius)
    return np.pad(
        blurred,
        ((0, height - blurred.shape[0]), (0, width - blurred.shape[1])),
        mode="edge",
    )


def build_scene(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Build one static scene: 1/f texture, lens blur, tint, and hard edges.

    Returns:
        A ``(height, width, 3)`` ``float32`` array, float so the bank can
        sample and blend it before quantising.
    """
    smaller_side = min(width, height)
    luminance = np.zeros((height, width), dtype=np.float32)
    for fraction, amplitude in SCENE_OCTAVES:
        scale = max(1, int(smaller_side * fraction))
        coarse = rng.random((height // scale + 2, width // scale + 2), dtype=np.float32)
        upsampled = np.repeat(np.repeat(coarse, scale, axis=0), scale, axis=1)
        luminance += amplitude * upsampled[:height, :width]

    luminance = _box_blur(luminance, max(1, int(smaller_side * SCENE_BLUR_FRACTION)))

    scene = np.stack(
        [
            luminance * gain + offset
            for gain, offset in zip(SCENE_CHANNEL_GAINS, SCENE_CHANNEL_OFFSETS)
        ],
        axis=-1,
    )

    for _ in range(SCENE_EDGE_COUNT):
        edge_height = int(rng.integers(max(2, height // 24), max(3, height // 4)))
        edge_width = int(rng.integers(max(2, width // 24), max(3, width // 4)))
        top = int(rng.integers(0, max(1, height - edge_height)))
        left = int(rng.integers(0, max(1, width - edge_width)))
        scene[top : top + edge_height, left : left + edge_width] += rng.integers(
            -SCENE_EDGE_AMPLITUDE, SCENE_EDGE_AMPLITUDE, FRAME_COLOR_CHANNELS
        ).astype(np.float32)

    return np.clip(scene, 0, FRAME_MAX_COLOR_VALUE)


def build_realistic_bank(width: int, height: int) -> np.ndarray:
    """Render :data:`BANK_FRAMES` frames of moving, textured, noisy content.

    Closes the pan on a cosine loop so the last-to-first step is ordinary,
    keeping a cycling bank free of encoder discontinuities.

    Returns:
        A contiguous ``(BANK_FRAMES, height, width, 3)`` ``uint8`` array.
    """
    rng = np.random.default_rng(SCENE_SEED)
    pan = _pan_range(width, height)
    # Room for the bilinear sample's second tap at a full-travel pan.
    scene = build_scene(width + pan + 2, height + pan + 2, rng)

    object_height = max(2, int(height * OBJECT_HEIGHT_FRACTION))
    object_width = max(2, int(width * OBJECT_WIDTH_FRACTION))

    frames = []
    for index in range(BANK_FRAMES):
        phase = 2 * np.pi * index / BANK_FRAMES
        offset_x = pan * 0.5 * (1 - np.cos(phase))
        # Offset the vertical phase so the path loops instead of running diagonally.
        offset_y = pan * 0.5 * (1 - np.cos(phase + 1.0))
        left, top = int(offset_x), int(offset_y)
        weight_x, weight_y = offset_x - left, offset_y - top

        top_left = scene[top : top + height, left : left + width]
        top_right = scene[top : top + height, left + 1 : left + 1 + width]
        bottom_left = scene[top + 1 : top + 1 + height, left : left + width]
        bottom_right = scene[top + 1 : top + 1 + height, left + 1 : left + 1 + width]
        frame = (
            top_left * (1 - weight_x) * (1 - weight_y)
            + top_right * weight_x * (1 - weight_y)
            + bottom_left * (1 - weight_x) * weight_y
            + bottom_right * weight_x * weight_y
        )

        object_top = int(
            (height - object_height) * (0.5 + OBJECT_PATH_AMPLITUDE * np.sin(phase))
        )
        object_left = int(
            (width - object_width) * (0.5 + OBJECT_PATH_AMPLITUDE * np.cos(phase))
        )
        patch = frame[
            object_top : object_top + object_height,
            object_left : object_left + object_width,
        ]
        frame[
            object_top : object_top + object_height,
            object_left : object_left + object_width,
        ] = (
            patch * OBJECT_BLEND + OBJECT_LEVEL
        )

        if SENSOR_NOISE_SIGMA > 0:
            frame = frame + rng.normal(
                0, SENSOR_NOISE_SIGMA, (height, width, FRAME_COLOR_CHANNELS)
            )

        frames.append(np.clip(frame, 0, FRAME_MAX_COLOR_VALUE).astype(np.uint8))

    return np.ascontiguousarray(np.stack(frames))


def build_flat_bank(width: int, height: int) -> np.ndarray:
    """Render the one solid-fill frame :data:`DETAIL_FLAT` cycles.

    A longer bank would only cost memory, since flat content never changes.

    Returns:
        A contiguous ``(1, height, width, 3)`` ``uint8`` array.
    """
    return np.full(
        (1, height, width, FRAME_COLOR_CHANNELS),
        FRAME_DEFAULT_FILL_VALUE,
        dtype=np.uint8,
    )


def build_frame_bank(detail: str, width: int, height: int) -> np.ndarray:
    """Build the bank of frames a feed at this detail level cycles through.

    Returns:
        A contiguous ``(frames, height, width, 3)`` ``uint8`` array holding
        at least one frame.
    """
    if detail == DETAIL_FLAT:
        return build_flat_bank(width, height)
    return build_realistic_bank(width, height)


def prewarm_frame_bank(detail: str, width: int | None, height: int | None) -> None:
    """Build and cache this case's frame bank, if not already built.

    Call this before any producer starts its wall-clock schedule: the start
    barrier releases before each thread allocates its buffer, so a lazy build
    on the camera thread would start the video stream behind the joint streams.
    """
    if width is None or height is None:
        return
    _cached_frame_bank(detail, width, height)


def _cached_frame_bank(detail: str, width: int, height: int) -> np.ndarray:
    """Return the shared frame bank for this detail and resolution, built once.

    Read-only once built, so every camera thread in the process shares one.
    """
    key = (detail, width, height)
    bank = _BANK_CACHE.get(key)
    if bank is not None:
        return bank
    with _BANK_CACHE_LOCK:
        bank = _BANK_CACHE.get(key)
        if bank is None:
            bank = build_frame_bank(detail, width, height)
            _BANK_CACHE[key] = bank
    return bank


class SyntheticCameraFeed:
    """A reusable frame buffer plus the content to fill it with.

    Callers must not retain a rendered frame past the next :meth:`render`
    call, since the same buffer is reused for every frame.
    """

    def __init__(self, width: int, height: int, detail: str) -> None:
        """Bind the *detail* level's bank and allocate the output buffer."""
        self._bank = _cached_frame_bank(detail, width, height)
        # A copy, not a view: the shared bank stays read-only.
        self._buffer = self._bank[0].copy()

    def render(self, tick: int, frame_code: int) -> np.ndarray:
        """Fill the buffer with the content for *tick* and paint *frame_code*.

        Returns:
            The buffer, valid until the next call.
        """
        np.copyto(self._buffer, self._bank[tick % self._bank.shape[0]])
        return encode_frame_number(frame_code, self._buffer)


def make_camera_feed(
    should_allocate: bool,
    image_width: int | None,
    image_height: int | None,
    detail: str,
) -> SyntheticCameraFeed | None:
    """Build a camera feed, or ``None`` if this caller logs no video."""
    if not should_allocate or image_width is None or image_height is None:
        return None
    return SyntheticCameraFeed(image_width, image_height, detail)


def encode_depth_frame(
    frame_num: int,
    width: int,
    height: int,
    mode: DepthMode,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Build a deterministic, non-zero, spatially non-uniform depth frame.

    Args:
        frame_num: Only ``frame_num % DEPTH_FRAME_BASE_MODULUS`` affects the
            output.
        out: Preallocated buffer already of *mode*'s dtype; callers passing
            it must not retain the returned array past the next overwrite.

    Returns:
        A ``(height, width)`` array of *mode*'s dtype.
    """
    dtype = np.float16 if mode == "float16" else np.float32
    base = (
        (frame_num % DEPTH_FRAME_BASE_MODULUS)
        / DEPTH_FRAME_BASE_MODULUS
        * (MAX_DEPTH * DEPTH_FRAME_BASE_FRACTION)
    )
    row_gradient = (np.arange(height, dtype=np.float32) / max(height - 1, 1)) * (
        MAX_DEPTH * DEPTH_FRAME_ROW_FRACTION
    )
    col_gradient = (np.arange(width, dtype=np.float32) / max(width - 1, 1)) * (
        MAX_DEPTH * DEPTH_FRAME_COL_FRACTION
    )
    floor = MAX_DEPTH * DEPTH_FRAME_FLOOR_FRACTION

    pattern = floor + base + row_gradient[:, None] + col_gradient[None, :]

    if out is None:
        return pattern.astype(dtype)
    out[:] = pattern.astype(dtype)
    return out


def preallocate_depth_buffer(
    should_allocate: bool,
    image_width: int | None,
    image_height: int | None,
    mode: DepthMode,
) -> np.ndarray | None:
    """Preallocate a reusable depth frame buffer, or return ``None``.

    Returns:
        A ``(image_height, image_width)`` array of *mode*'s dtype, or ``None``
        when this caller logs no depth or the case has no video.
    """
    if not should_allocate or image_width is None or image_height is None:
        return None
    dtype = np.float16 if mode == "float16" else np.float32
    return np.empty((image_height, image_width), dtype=dtype)
