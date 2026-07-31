"""Synthetic camera frame generation for data-daemon tests.

Camera frames logged by the test producers must cost the daemon's video pipeline
what a real camera costs it. A solid-colour frame does not: it compresses ~620:1
losslessly, so the SDK's PNG compression pool, the on-disk video spool, and the
``libx264rgb`` encoder all sit effectively idle and the suite cannot say anything
about real-time viability.

:data:`DETAIL_REALISTIC` frames therefore carry 1/f spatial texture, optics-like
blur, hard-edged objects, sub-pixel global motion and sensor noise, which together
land within a factor of ~1.2 of real 1080p footage's lossless bitrate.

All of that content is generated **once** into a small bank of frames
(:func:`build_frame_bank`) and each logged frame is a single copy out of that bank
(:meth:`SyntheticCameraFeed.render`), because the producers hold hard per-frame
wall-clock deadlines — as little as 8.3 ms at 120 fps. Generating content per
frame would blow those deadlines at 1080p; copying does not (~0.34 ms).

:data:`DETAIL_FLAT` reproduces the original solid-fill frames byte for byte, and
is used by the performance suites whose timing budgets are calibrated against
them.
"""

from __future__ import annotations

import threading

import numpy as np

from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DETAIL_REALISTIC,
    FRAME_BYTE_LENGTH,
    FRAME_COLOR_CHANNELS,
    FRAME_DEFAULT_FILL_VALUE,
    FRAME_GRID_SIZE,
    FRAME_HALF_DIVISOR,
    FRAME_MAX_COLOR_VALUE,
)

# cspell:ignore copyto
# Frames held in the pre-rendered bank. The producers cycle through it, so this
# is the motion loop's period rather than a frame budget. Kept small because the
# bank is resident for the whole process (75 MB at 1080p); large enough that the
# cycle is invisible to an encoder which, at ``-preset ultrafast``, references
# only the previous frame.
BANK_FRAMES = 12

# Per-pixel gaussian sensor noise, in 8-bit levels, drawn fresh for each bank
# frame. This is the single dial for how hard the encoder is worked: measured at
# 1920x1080 through the daemon's own ffmpeg arguments, 0.0 gives 11.3:1 lossless
# compression (66 Mbps at 15 fps), 0.4 gives 3.7:1 (200 Mbps) and 1.0 gives 2.8:1
# (269 Mbps). 1.0 is what a real sensor costs a lossless RGB encoder.
SENSOR_NOISE_SIGMA = 1.0

# Fixed seed: frame content must be identical across processes and runs, so that
# a failure reproduces and so that per-case artefact sizes are comparable.
SCENE_SEED = 7

# 1/f spatial texture: each octave is a random field at ``fraction`` of the
# smaller frame dimension, upsampled and summed with the paired amplitude.
# Fractions are relative to frame size so 64x64 and 1920x1080 get comparable
# structure instead of one flat block.
SCENE_OCTAVES = ((0.5, 70.0), (1 / 6, 40.0), (1 / 16, 20.0), (1 / 48, 9.0))

# Hard-edged rectangles laid over the texture, for the high-frequency edge
# structure real scenes have and smooth noise does not.
SCENE_EDGE_COUNT = 40
SCENE_EDGE_AMPLITUDE = 55

# Box-blur radius as a fraction of the smaller frame dimension. Stands in for
# lens blur and removes the upsampling's nearest-neighbour block edges, which are
# an artefact of how the octaves are built rather than real scene content.
SCENE_BLUR_FRACTION = 1 / 160

# Per-channel gain and offset applied to the shared luminance field, so the three
# planes are correlated (as a real sensor's are) without being identical.
SCENE_CHANNEL_GAINS = (1.0, 0.92, 0.85)
SCENE_CHANNEL_OFFSETS = (0.0, 10.0, 18.0)

# Global motion: the frame is sampled from a larger scene at a sub-pixel offset
# travelling a closed loop, so every consecutive pair of bank frames differs by a
# real motion step including across the wrap. Sub-pixel matters — an integer pan
# is exactly motion-compensated and the residual collapses to a quarter of the
# realistic bitrate.
SCENE_PAN_FRACTION = 1 / 8
SCENE_PAN_MIN = 4

# A foreground object tracking its own path across the frame, so the content is
# not pure global motion.
OBJECT_HEIGHT_FRACTION = 1 / 9
OBJECT_WIDTH_FRACTION = 1 / 12
OBJECT_PATH_AMPLITUDE = 0.45
OBJECT_BLEND = 0.45
OBJECT_LEVEL = 140.0

_BANK_CACHE: dict[tuple[int, int], np.ndarray] = {}
_BANK_CACHE_LOCK = threading.Lock()


def encode_frame_number(
    frame_num: int, width: int, height: int, out: np.ndarray | None = None
) -> np.ndarray:
    """Encode a frame number into the pixel data of a synthetic video frame.

    The 16-byte big-endian representation of ``frame_num`` is written into the
    top-left 4x4 grid of the image. For each pixel at ``(row, col)`` in that
    grid the byte value is mapped to the RGB channels as follows:

    - Red channel = ``byte_value``
    - Green channel = ``FRAME_MAX_COLOR_VALUE - byte_value``
    - Blue channel = ``byte_value // FRAME_HALF_DIVISOR``

    Only that grid is written. When ``out`` is omitted the rest of the frame is
    :data:`FRAME_DEFAULT_FILL_VALUE`; when ``out`` is given the rest of the frame
    is whatever the caller left there, which is how
    :meth:`SyntheticCameraFeed.render` paints the code over scene content.

    The grid is the only part of a frame the decode side reads back
    (``decode_frame_number``), so it must survive the round trip untouched. It
    does: the daemon's ``lossless.mp4`` is ``libx264rgb -qp 0``, and surrounding
    content cannot perturb it.

    Args:
        frame_num: The frame number to embed. Must fit in 16 bytes (i.e.
            less than ``2 ** 128``).
        width: Frame width in pixels.
        height: Frame height in pixels.
        out: If given, write into this preallocated ``(height, width, 3)``
            ``uint8`` array instead of allocating a new one. Callers that pass
            ``out`` must never retain the returned array past the point where
            its contents may next be overwritten.

    Returns:
        A NumPy array with shape ``(height, width, 3)`` and dtype ``uint8``.
    """
    if out is None:
        img = np.zeros((height, width, FRAME_COLOR_CHANNELS), dtype=np.uint8)
        img.fill(FRAME_DEFAULT_FILL_VALUE)
    else:
        img = out

    frame_bytes = frame_num.to_bytes(FRAME_BYTE_LENGTH, byteorder="big")

    for row in range(FRAME_GRID_SIZE):
        for col in range(FRAME_GRID_SIZE):
            idx = row * FRAME_GRID_SIZE + col
            if idx < len(frame_bytes):
                pixel_value = frame_bytes[idx]
                img[row, col, 0] = pixel_value
                img[row, col, 1] = FRAME_MAX_COLOR_VALUE - pixel_value
                img[row, col, 2] = pixel_value // FRAME_HALF_DIVISOR

    return img


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

    The result is the *background* the frame bank pans across, so it is built
    larger than a frame by the pan travel.

    Args:
        width: Scene width in pixels.
        height: Scene height in pixels.
        rng: Seeded generator, so the scene is reproducible.

    Returns:
        A ``(height, width, 3)`` ``float32`` array with values in ``[0, 255]``.
        Left as float so the bank can sample and blend it before quantising.
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


def build_frame_bank(width: int, height: int) -> np.ndarray:
    """Render :data:`BANK_FRAMES` frames of moving, textured, noisy content.

    Each frame samples :func:`build_scene` at a sub-pixel offset on a closed
    cosine loop, blends in a moving foreground object, and adds fresh sensor
    noise. Because the loop closes, the step from the last frame back to the
    first is an ordinary motion step, so cycling the bank never presents the
    encoder with a discontinuity.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        A contiguous ``(BANK_FRAMES, height, width, 3)`` ``uint8`` array.
    """
    rng = np.random.default_rng(SCENE_SEED)
    pan = _pan_range(width, height)
    # +1 for the bilinear sample's second tap, +1 so a pan of exactly ``pan``
    # still has that tap in bounds.
    scene = build_scene(width + pan + 2, height + pan + 2, rng)

    object_height = max(2, int(height * OBJECT_HEIGHT_FRACTION))
    object_width = max(2, int(width * OBJECT_WIDTH_FRACTION))

    frames = []
    for index in range(BANK_FRAMES):
        phase = 2 * np.pi * index / BANK_FRAMES
        offset_x = pan * 0.5 * (1 - np.cos(phase))
        # Offset the vertical phase so the path is a loop rather than a diagonal.
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


def prewarm_frame_bank(width: int | None, height: int | None) -> None:
    """Build and cache the frame bank for this resolution, if not already built.

    Call this before any producer starts its wall-clock schedule. Building the
    bank takes ~1.9 s at 1080p, and the threaded producer's start barrier
    releases *before* each thread allocates its buffer — so a lazy build inside
    the camera thread would start the video stream seconds behind the joint
    streams, and cloud verification asserts an exact frame-count match against
    the synchronised sync points.

    Args:
        width: Frame width in pixels, or ``None`` if the case has no video.
        height: Frame height in pixels, or ``None`` if the case has no video.
    """
    if width is None or height is None:
        return
    _cached_frame_bank(width, height)


def _cached_frame_bank(width: int, height: int) -> np.ndarray:
    """Return the shared frame bank for this resolution, building it once.

    The bank is read-only once built, so every camera thread in the process can
    share one and copy out of it into its own buffer.
    """
    key = (width, height)
    bank = _BANK_CACHE.get(key)
    if bank is not None:
        return bank
    with _BANK_CACHE_LOCK:
        bank = _BANK_CACHE.get(key)
        if bank is None:
            bank = build_frame_bank(width, height)
            _BANK_CACHE[key] = bank
    return bank


class SyntheticCameraFeed:
    """A reusable frame buffer plus the content to fill it with.

    Owns one output buffer for the lifetime of the producer that holds it, so no
    per-frame allocation happens on the logging path. The buffer is handed
    straight to ``nc.log_rgb``, which copies it synchronously, so the caller must
    not retain a returned frame past the next :meth:`render`.
    """

    def __init__(self, width: int, height: int, detail: str) -> None:
        """Allocate the output buffer and bind the content source.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            detail: :data:`DETAIL_REALISTIC` to render moving textured content,
                :data:`DETAIL_FLAT` for solid-fill frames.
        """
        self._width = width
        self._height = height
        self._buffer = np.zeros((height, width, FRAME_COLOR_CHANNELS), dtype=np.uint8)
        self._buffer.fill(FRAME_DEFAULT_FILL_VALUE)
        self._bank = (
            _cached_frame_bank(width, height) if detail == DETAIL_REALISTIC else None
        )

    def render(self, tick: int, frame_code: int) -> np.ndarray:
        """Fill the buffer with the content for *tick* and paint *frame_code*.

        Args:
            tick: The producer's frame index. Selects the bank frame, so all
                cameras logged for the same index show the same instant.
            frame_code: The integer to embed in the top-left 4x4 grid.

        Returns:
            The buffer, valid until the next call.
        """
        if self._bank is not None:
            np.copyto(self._buffer, self._bank[tick % self._bank.shape[0]])
        return encode_frame_number(
            frame_code, self._width, self._height, out=self._buffer
        )


def make_camera_feed(
    should_allocate: bool,
    image_width: int | None,
    image_height: int | None,
    detail: str,
) -> SyntheticCameraFeed | None:
    """Build a camera feed, or ``None`` if this caller logs no video.

    Args:
        should_allocate: Whether this caller needs a feed at all (e.g. the
            recording has cameras, or this thread's role is "rgb").
        image_width: Frame width in pixels, or ``None`` if not video.
        image_height: Frame height in pixels, or ``None`` if not video.
        detail: The case's ``video_detail``.

    Returns:
        A :class:`SyntheticCameraFeed`, or ``None`` if ``should_allocate`` is
        ``False`` or either dimension is ``None``.
    """
    if not should_allocate or image_width is None or image_height is None:
        return None
    return SyntheticCameraFeed(image_width, image_height, detail)
