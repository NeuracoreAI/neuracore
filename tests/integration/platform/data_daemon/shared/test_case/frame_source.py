"""Synthetic camera frame generation for data-daemon tests.

Frames logged by the test producers must cost the daemon's video pipeline what a
real camera costs it. A solid-colour frame does not: it compresses ~620:1
losslessly, leaving the SDK's PNG pool, the video spool and the ``libx264rgb``
encoder idle, so the suite can say nothing about real-time viability.

:data:`DETAIL_REALISTIC` frames therefore carry 1/f texture, optics-like blur,
hard edges, sub-pixel motion and sensor noise, landing within ~1.2x of real 1080p
footage's lossless bitrate. :data:`DETAIL_FLAT` reproduces the original solid-fill
frames byte for byte, for cases that only care about frame counts.

Both build their content **once** into a bank (:func:`build_frame_bank`); each
logged frame is one copy out of it with its frame code painted on
(:meth:`SyntheticCameraFeed.render`). Producers hold per-frame deadlines as tight
as 8.3 ms at 120 fps: a 1080p bank takes ~1.9 s to build (~160 ms a frame), a copy
out of it ~0.34 ms. Flat content is cheap either way, but shares the bank path so
a detail switch changes pixels and nothing else.
"""

from __future__ import annotations

import threading

import numpy as np

from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DETAIL_FLAT,
    FRAME_BYTE_LENGTH,
    FRAME_COLOR_CHANNELS,
    FRAME_DEFAULT_FILL_VALUE,
    FRAME_GRID_SIZE,
    FRAME_HALF_DIVISOR,
    FRAME_MAX_COLOR_VALUE,
)

# cspell:ignore copyto
# Frames in the realistic bank: the motion loop's period, not a frame budget.
# Small because the bank is process-resident (75 MB at 1080p); large enough that
# the cycle is invisible to an encoder which, at ``-preset ultrafast``, references
# only the previous frame. The flat bank holds one frame — it has no motion.
BANK_FRAMES = 12

# Per-pixel gaussian sensor noise in 8-bit levels, redrawn per bank frame. The
# single dial for how hard the encoder works: at 1920x1080 through the daemon's
# own ffmpeg arguments, 0.0 gives 11.3:1 lossless (66 Mbps at 15 fps), 0.4 gives
# 3.7:1 (200 Mbps), 1.0 gives 2.8:1 (269 Mbps) — what a real sensor costs.
SENSOR_NOISE_SIGMA = 1.0

# Fixed so content is identical across processes and runs: failures reproduce and
# per-case artefact sizes stay comparable.
SCENE_SEED = 7

# 1/f texture: each octave is a random field at ``fraction`` of the smaller frame
# dimension, upsampled and summed with its amplitude. Fractions are relative to
# frame size so 64x64 and 1920x1080 get comparable structure, not one flat block.
SCENE_OCTAVES = ((0.5, 70.0), (1 / 6, 40.0), (1 / 16, 20.0), (1 / 48, 9.0))

# Hard-edged rectangles over the texture: the high-frequency edge structure real
# scenes have and smooth noise does not.
SCENE_EDGE_COUNT = 40
SCENE_EDGE_AMPLITUDE = 55

# Box-blur radius as a fraction of the smaller frame dimension: stands in for lens
# blur, and removes the upsampling's block edges — an artefact of the octaves.
SCENE_BLUR_FRACTION = 1 / 160

# Gain and offset per channel on the shared luminance field, so the three planes
# are correlated as a real sensor's are without being identical.
SCENE_CHANNEL_GAINS = (1.0, 0.92, 0.85)
SCENE_CHANNEL_OFFSETS = (0.0, 10.0, 18.0)

# Global motion: frames are sampled from a larger scene at a sub-pixel offset on a
# closed loop, so every consecutive pair differs by a real motion step, wrap
# included. Sub-pixel matters — an integer pan is exactly motion-compensated and
# the residual collapses to a quarter of the realistic bitrate.
SCENE_PAN_FRACTION = 1 / 8
SCENE_PAN_MIN = 4

# A foreground object on its own path, so content is not pure global motion.
OBJECT_HEIGHT_FRACTION = 1 / 9
OBJECT_WIDTH_FRACTION = 1 / 12
OBJECT_PATH_AMPLITUDE = 0.45
OBJECT_BLEND = 0.45
OBJECT_LEVEL = 140.0

_BANK_CACHE: dict[tuple[str, int, int], np.ndarray] = {}
_BANK_CACHE_LOCK = threading.Lock()


# Decimal bands giving each (context, recording, camera) its own range of frame
# codes, so a decoded code says where the frame came from as well as its position:
# context 2's second recording's camera 1, frame 45, paints 2_010_100_045.
FRAME_CODE_CONTEXT_STRIDE = 1_000_000_000
FRAME_CODE_RECORDING_STRIDE = 10_000_000
FRAME_CODE_CAMERA_STRIDE = 100_000


def frame_code_base(
    *, context_index: int, recording_ordinal: int, camera_index: int
) -> int:
    """Return the code painted into the first frame of one camera's recording.

    Frame ``i`` carries ``frame_code_base(...) + i``. Producers paint from this and
    both the disk and cloud verification passes re-derive it, so the mapping lives
    here rather than at each site.

    Args:
        context_index: Index of the parallel recording context.
        recording_ordinal: Zero-based position within that context, *not* the
            daemon-assigned recording index.
        camera_index: Position of the camera in the case's camera list.
    """
    return (
        context_index * FRAME_CODE_CONTEXT_STRIDE
        + recording_ordinal * FRAME_CODE_RECORDING_STRIDE
        + camera_index * FRAME_CODE_CAMERA_STRIDE
    )


def encode_frame_number(frame_num: int, frame: np.ndarray) -> np.ndarray:
    """Paint ``frame_num`` into the top-left 4x4 grid of *frame*, in place.

    The 16-byte big-endian representation of ``frame_num`` fills the grid, one byte
    per pixel: red = ``byte``, green = ``FRAME_MAX_COLOR_VALUE - byte``, blue =
    ``byte // FRAME_HALF_DIVISOR``. Nothing else is written, so the rest of *frame*
    keeps the caller's content — that is how :meth:`SyntheticCameraFeed.render`
    paints the code over a bank frame.

    The grid is the only part the decode side reads back (``decode_frame_number``)
    and it survives the round trip untouched: the daemon's ``lossless.mp4`` is
    ``libx264rgb -qp 0``, so surrounding content cannot perturb it.

    Args:
        frame_num: Must fit in 16 bytes, i.e. below ``2 ** 128``.
        frame: The ``(height, width, 3)`` ``uint8`` array to paint into.

    Returns:
        *frame*, so callers can hand the painted frame straight on.
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

    This is the *background* the bank pans across, so callers build it larger than
    a frame by the pan travel. *rng* is seeded, so the scene is reproducible.

    Returns:
        A ``(height, width, 3)`` ``float32`` array in ``[0, 255]`` — float so the
        bank can sample and blend it before quantising.
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

    Each frame samples :func:`build_scene` at a sub-pixel offset on a closed cosine
    loop, blends in a moving foreground object, and adds fresh sensor noise. The
    loop closing makes the last-to-first step an ordinary one, so cycling the bank
    never hands the encoder a discontinuity.

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


def build_flat_bank(width: int, height: int) -> np.ndarray:
    """Render the one solid-fill frame :data:`DETAIL_FLAT` cycles.

    One frame rather than :data:`BANK_FRAMES` of them: flat content does not change
    from frame to frame, so a longer bank would only cost memory.

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
        A contiguous ``(frames, height, width, 3)`` ``uint8`` array holding at
        least one frame. The frame count is the detail level's business —
        :meth:`SyntheticCameraFeed.render` wraps its tick around whatever it is.
    """
    if detail == DETAIL_FLAT:
        return build_flat_bank(width, height)
    return build_realistic_bank(width, height)


def prewarm_frame_bank(detail: str, width: int | None, height: int | None) -> None:
    """Build and cache this case's frame bank, if not already built.

    Call this before any producer starts its wall-clock schedule. The realistic
    bank takes ~1.9 s at 1080p, and the threaded producer's start barrier releases
    *before* each thread allocates its buffer — so a lazy build inside the camera
    thread would start the video stream seconds behind the joint streams, and cloud
    verification asserts an exact frame-count match against the synchronised sync
    points. A flat bank costs nothing to build but is prewarmed the same way, so
    producer startup does not depend on which detail the case picked.

    *width* and *height* are ``None`` when the case has no video, and prewarming
    is then a no-op.
    """
    if width is None or height is None:
        return
    _cached_frame_bank(detail, width, height)


def _cached_frame_bank(detail: str, width: int, height: int) -> np.ndarray:
    """Return the shared frame bank for this detail and resolution, built once.

    Read-only once built, so every camera thread in the process shares one and
    copies out of it into its own buffer.
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

    Owns one buffer for the lifetime of the producer holding it, so the logging
    path allocates nothing per frame. The buffer goes straight to ``nc.log_rgb``,
    which copies it synchronously, so callers must not retain a rendered frame
    past the next :meth:`render`.
    """

    def __init__(self, width: int, height: int, detail: str) -> None:
        """Bind the *detail* level's bank and allocate the output buffer."""
        self._bank = _cached_frame_bank(detail, width, height)
        # A copy, not a view: the bank is shared across the process's camera
        # threads and stays read-only; the buffer is painted per frame.
        self._buffer = self._bank[0].copy()

    def render(self, tick: int, frame_code: int) -> np.ndarray:
        """Fill the buffer with the content for *tick* and paint *frame_code*.

        *tick* is the producer's frame index, so cameras logging the same index
        show the same instant; a flat bank holds one frame, so every tick selects
        it. *frame_code* is embedded in the top-left 4x4 grid.

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
    """Build a camera feed, or ``None`` if this caller logs no video.

    *should_allocate* is whether the caller needs a feed at all — the recording has
    cameras, or this thread's role is "rgb". Dimensions are ``None`` when the case
    has no video, which also yields ``None``.
    """
    if not should_allocate or image_width is None or image_height is None:
        return None
    return SyntheticCameraFeed(image_width, image_height, detail)
