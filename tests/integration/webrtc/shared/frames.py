"""Synthetic video frames with a counter + checksum embedded in the pixels.

The producer side keeps only bytes + track_id (frame metadata is deferred to
PR5), so order and integrity must travel *inside* the picture. We block-code a
monotonic 32-bit counter plus a 16-bit checksum into a header band of large
solid squares at the top of each frame. Solid blocks survive H.264's chroma
subsampling and deblocking, and decoding samples each block's centre to dodge
edge bleed — so the consumer can recover the counter from a *decoded* frame and
verify it was not corrupted.

The rest of the frame is a deterministic high-spatial-frequency texture (fine
diagonal stripes) XOR-ed with a per-counter moving gradient, so the encoder has
real, changing content to compress. The high-frequency detail is deliberate: it
makes the encoded bitrate strongly resolution-dependent (fine stripes survive at
full resolution but average toward grey when the congestion ladder downscales),
which is what gives the rungs real bitrate separation. A smooth gradient
compresses to almost nothing at *every* resolution, so the downscale ladder had
no rate lever and the constrained-link netem gate could not distinguish a fat
rung from a thin one (see reports/SPIKE-chrome-pframe.md §constrained-link).
"""

from __future__ import annotations

import zlib

import numpy as np

WIDTH = 640
HEIGHT = 480
CHANNELS = 3

# Header band geometry: a grid of solid BLOCK x BLOCK squares, one per bit.
_BLOCK = 40
_COLS = WIDTH // _BLOCK  # 16 blocks across
_HEADER_ROWS = 3  # -> 48 blocks available in the top 120 rows
_BITS = _COLS * _HEADER_ROWS  # 48
_COUNTER_BITS = 32
_CHECK_BITS = 16
assert _COUNTER_BITS + _CHECK_BITS == _BITS, "header band must hold counter+checksum"

_COUNTER_MASK = (1 << _COUNTER_BITS) - 1
_CHECK_MASK = (1 << _CHECK_BITS) - 1


def _checksum(counter: int) -> int:
    """16-bit checksum over the 32-bit counter."""
    return zlib.crc32(counter.to_bytes(4, "big")) & _CHECK_MASK


def _payload_bits(counter: int) -> list[int]:
    counter &= _COUNTER_MASK
    payload = (counter << _CHECK_BITS) | _checksum(counter)
    return [(payload >> (_BITS - 1 - i)) & 1 for i in range(_BITS)]


def _block_box(idx: int) -> tuple[int, int, int, int]:
    row, col = divmod(idx, _COLS)
    y0 = row * _BLOCK
    x0 = col * _BLOCK
    return y0, x0, y0 + _BLOCK, x0 + _BLOCK


def encode_frame(counter: int) -> np.ndarray:
    """Build a C-contiguous (H, W, 3) uint8 frame carrying ``counter``."""
    # High-frequency body: a static fine-stripe texture (period ~2px, so it
    # carries energy near Nyquist and shrinks sharply under downscale) XOR-ed with
    # a moving gradient (per-counter, so every frame changes and P-frames carry
    # real residual). Deterministic and reproducible.
    rows = np.arange(HEIGHT, dtype=np.uint32)[:, None]
    cols = np.arange(WIDTH, dtype=np.uint32)[None, :]
    stripes = (rows * 127 + cols * 127) % 256
    moving = (rows + cols + counter * 4) & 0xFF
    body = (stripes ^ moving).astype(np.uint8)
    frame = np.repeat(body[:, :, None], CHANNELS, axis=2)

    # Overwrite the header band with the block-coded counter + checksum.
    for idx, bit in enumerate(_payload_bits(counter)):
        y0, x0, y1, x1 = _block_box(idx)
        frame[y0:y1, x0:x1, :] = 255 if bit else 0

    return np.ascontiguousarray(frame)


def decode_frame(frame: np.ndarray) -> tuple[int, bool]:
    """Recover ``(counter, ok)`` from a decoded frame.

    ``ok`` is False when the embedded checksum does not match the recovered
    counter, i.e. the header band was corrupted in transit.
    """
    gray = frame[..., :CHANNELS].mean(axis=2) if frame.ndim == 3 else frame
    margin = _BLOCK // 4  # sample the central half of each block
    payload = 0
    for idx in range(_BITS):
        y0, x0, y1, x1 = _block_box(idx)
        patch = gray[y0 + margin : y1 - margin, x0 + margin : x1 - margin]
        payload = (payload << 1) | (1 if patch.mean() >= 127 else 0)
    counter = payload >> _CHECK_BITS
    check = payload & _CHECK_MASK
    return counter, check == _checksum(counter)


def parse_video_frame_event(event: dict) -> tuple[str | None, str | None, np.ndarray]:
    """Reshape a consumer ``on_frame`` event into ``(track_id, mid, array)``.

    Forward contract introduced by PR5 (extends the PR0 event schema): the
    consumer surfaces each decoded frame on its drainable queue as
    ``{"kind": "on_frame", "track_id", "mid", "data": bytes, "width",
    "height"}``. ``data`` is the decoded picture as 8-bit HxWx3; the block codec
    above is colour-order agnostic (black vs white blocks), so RGB or BGR both
    decode.
    """
    width = int(event.get("width", WIDTH))
    height = int(event.get("height", HEIGHT))
    array = np.frombuffer(event["data"], dtype=np.uint8)
    array = array[: width * height * CHANNELS].reshape(height, width, CHANNELS)
    return event.get("track_id"), event.get("mid"), array
