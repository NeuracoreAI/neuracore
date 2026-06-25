"""Unit tests for the synthetic frame codec (PR1 ``shared/frames.py``).

Formalises the self-checks the PR1 report noted: the embedded 32-bit counter
round-trips across the whole ``[0, 2**32)`` range, a scrambled header band is
flagged as corrupt, and the counter survives a simulated lossy (blur + quantise
+ noise) path the way it must survive H.264.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.integration.webrtc.shared import frames

# A spread across the 32-bit range: edges, powers of two, and an odd large value.
_COUNTERS = [
    0,
    1,
    255,
    256,
    65535,
    65536,
    1234567,
    2**31,
    2**32 - 2,
    2**32 - 1,
]


@pytest.mark.parametrize("counter", _COUNTERS)
def test_counter_round_trips_clean(counter: int) -> None:
    frame = frames.encode_frame(counter)
    # Contract for submit_frame: C-contiguous uint8 H x W x 3.
    assert frame.dtype == np.uint8
    assert frame.shape == (frames.HEIGHT, frames.WIDTH, frames.CHANNELS)
    assert frame.flags["C_CONTIGUOUS"]

    recovered, ok = frames.decode_frame(frame)
    assert ok, f"checksum failed for counter {counter}"
    assert recovered == counter


def test_scrambled_header_band_is_flagged_as_corrupt() -> None:
    frame = frames.encode_frame(987654)
    # Flip the header band (the top rows that carry the block-coded counter +
    # checksum) so the recovered counter no longer matches its checksum.
    header_rows = frames._HEADER_ROWS * frames._BLOCK
    corrupted = frame.copy()
    corrupted[:header_rows, :, :] = 255 - corrupted[:header_rows, :, :]

    _, ok = frames.decode_frame(corrupted)
    assert not ok, "corruption in the header band must be detected by the checksum"


def _lossy(frame: np.ndarray) -> np.ndarray:
    """A deterministic lossy transform standing in for an H.264 round trip:
    a 3x3 box blur, an 8-level quantisation, and a small fixed ripple."""
    blurred = frame.astype(np.float32)
    # Separable 3x3 box blur via shifts (cheap, no scipy dependency).
    for axis in (0, 1):
        blurred = (
            blurred + np.roll(blurred, 1, axis=axis) + np.roll(blurred, -1, axis=axis)
        ) / 3.0
    quantised = np.round(blurred / 32.0) * 32.0
    rows = np.arange(frame.shape[0])[:, None, None]
    ripple = (8.0 * np.sin(rows / 7.0)).astype(np.float32)
    return np.clip(quantised + ripple, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("counter", [0, 42, 4096, 1_000_000, 2**32 - 1])
def test_counter_survives_a_lossy_path(counter: int) -> None:
    frame = frames.encode_frame(counter)
    recovered, ok = frames.decode_frame(_lossy(frame))
    # The solid blocks + centre sampling are engineered to survive exactly this.
    assert ok, f"checksum failed through the lossy path for counter {counter}"
    assert recovered == counter


def test_parse_video_frame_event_reshapes_the_pr5_contract() -> None:
    counter = 314159
    frame = frames.encode_frame(counter)
    event = {
        "kind": "on_frame",
        "track_id": "cam0",
        "mid": "v0",
        "data": frame.tobytes(),
        "width": frames.WIDTH,
        "height": frames.HEIGHT,
    }
    track_id, mid, array = frames.parse_video_frame_event(event)
    assert (track_id, mid) == ("cam0", "v0")
    recovered, ok = frames.decode_frame(array)
    assert ok and recovered == counter
