"""Checks on the files the daemon uploads for each depth camera.

A depth camera uploads lossless.bin (lossless JPEG-XL codestreams back to
back), lossy.mp4 (an 8-bit log-curve H.264 viewer video) and trace.json (one
entry per frame, each holding depth_scale_m and the byte range of its frame in
lossless.bin).
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Sequence
from typing import Any

import av
import imagecodecs
import numpy as np
import requests
from neuracore_types.utils.name_utils import to_safe_name

from neuracore.core.data.recording import Recording
from neuracore.core.utils.depth_utils import depth_to_log_gray
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    DATA_TYPE_DEPTH_IMAGES,
    DEPTH_HOLE_SIZE,
    DEPTH_PREVIEW_MAX_CHROMA_OFFSET,
    DEPTH_PREVIEW_MEDIAN_LEVEL_TOLERANCE,
    DEPTH_SATURATED_VALUE,
    DepthMode,
)
from tests.integration.platform.data_daemon.shared.test_case.frame_source import (
    depth_scale_for_mode,
)

DEPTH_FRAMES_FILENAME = "lossless.bin"
DEPTH_PREVIEW_FILENAME = "lossy.mp4"
DEPTH_TRACE_FILENAME = "trace.json"
PACKED_DEPTH_VIDEO_FILENAME = "lossless.mp4"
DEPTH_PREVIEW_MAX_HEIGHT = 480
NEUTRAL_CHROMA = 128.0


def assert_depth_trace(trace: object, *, depth_scale_m: float) -> list[dict[str, Any]]:
    """Check a depth trace.json and return its frame entries.

    Args:
        trace: The parsed trace.json.
        depth_scale_m: The meters per unit every entry must declare.

    Returns:
        The frame entries, each holding depth_scale_m, offset and length.
    """
    assert (
        isinstance(trace, list) and trace
    ), f"Depth trace.json must be a non-empty array of frame entries, got {trace!r}"
    expected_offset = 0
    for index, entry in enumerate(trace):
        assert math.isclose(
            float(entry["depth_scale_m"]), depth_scale_m, rel_tol=1e-9
        ), (
            f"trace.json entry {index} depth_scale_m {entry['depth_scale_m']} does "
            f"not match {depth_scale_m}"
        )
        assert entry["offset"] == expected_offset and entry["length"] > 0, (
            f"trace.json entry {index} spans {entry['offset']}+{entry['length']}, "
            f"not a frame starting at byte {expected_offset}"
        )
        expected_offset += entry["length"]
    return trace


def assert_depth_frames(
    frames_file: bytes,
    *,
    trace: Sequence[dict[str, Any]],
    height: int,
    width: int,
) -> list[np.ndarray]:
    """Check a depth lossless.bin against its trace and return its decoded frames.

    Args:
        frames_file: The lossless.bin bytes.
        trace: The trace.json entries, in frame order.
        height: Expected frame height in pixels.
        width: Expected frame width in pixels.

    Returns:
        The decoded uint16 frames in trace order.
    """
    end = trace[-1]["offset"] + trace[-1]["length"]
    assert end == len(frames_file), (
        f"trace.json frames end at byte {end}, lossless.bin holds "
        f"{len(frames_file)} bytes"
    )
    frames = []
    for index, entry in enumerate(trace):
        payload = frames_file[entry["offset"] : entry["offset"] + entry["length"]]
        frame = imagecodecs.jpegxl_decode(payload)
        assert frame.dtype == np.uint16 and frame.shape == (
            height,
            width,
        ), f"Depth frame {index} decoded as {frame.dtype} {frame.shape}"
        holes = frame[height - DEPTH_HOLE_SIZE :, width - DEPTH_HOLE_SIZE :]
        assert not holes.any(), f"Depth frame {index} filled its no-return pixels"
        assert (
            frame[height - 1, 0] == DEPTH_SATURATED_VALUE
        ), f"Depth frame {index} lost its saturated pixel"
        frames.append(frame)
    return frames


def _chroma_offset(frame: av.VideoFrame) -> float:
    """Return the largest mean distance of the U or V plane from neutral grey."""
    planes = frame.reformat(format="yuv444p").to_ndarray()
    if planes.shape[-1] == 3:
        planes = np.moveaxis(planes, -1, 0)
    return max(abs(float(planes[index].mean()) - NEUTRAL_CHROMA) for index in (1, 2))


def assert_depth_preview(
    video: bytes,
    *,
    lossless_frames: Sequence[np.ndarray],
    depth_scale_m: float,
) -> None:
    """Check the depth viewer video against the frames in lossless.bin.

    Args:
        video: The lossy.mp4 bytes.
        lossless_frames: Decoded uint16 frames from lossless.bin.
        depth_scale_m: Meters per unit of the lossless frames.
    """
    level_errors: list[np.ndarray] = []
    chroma_offsets: list[float] = []
    decoded = 0
    with av.open(io.BytesIO(video)) as container:
        stream = container.streams.video[0]
        assert (
            stream.codec_context.name == "h264"
        ), f"Viewer video codec is {stream.codec_context.name!r}, not h264"
        assert (
            stream.codec_context.pix_fmt == "yuv420p"
        ), f"Viewer video pixel format is {stream.codec_context.pix_fmt!r}"
        assert stream.codec_context.height <= DEPTH_PREVIEW_MAX_HEIGHT
        for frame in container.decode(stream):
            if decoded < len(lossless_frames):
                luma = frame.to_ndarray(format="gray").astype(np.int16)
                source = lossless_frames[decoded]
                if luma.shape == source.shape:
                    expected = depth_to_log_gray(
                        source.astype(np.float64) * depth_scale_m
                    ).astype(np.int16)
                    level_errors.append(np.abs(luma - expected).ravel())
                chroma_offsets.append(_chroma_offset(frame))
            decoded += 1

    assert decoded == len(
        lossless_frames
    ), f"Viewer video holds {decoded} frames, lossless.bin {len(lossless_frames)}"
    assert max(chroma_offsets) <= DEPTH_PREVIEW_MAX_CHROMA_OFFSET, (
        "Viewer video carries colour, so it is not the greyscale log curve: "
        f"largest chroma offset {max(chroma_offsets):.1f}"
    )
    if level_errors:
        median_error = float(np.median(np.concatenate(level_errors)))
        assert median_error <= DEPTH_PREVIEW_MEDIAN_LEVEL_TOLERANCE, (
            f"Viewer video departs from the log curve by a median {median_error} "
            f"grey levels (tolerance {DEPTH_PREVIEW_MEDIAN_LEVEL_TOLERANCE})"
        )


def _assert_absent(recording: Recording, filepath: str) -> None:
    """Fail unless downloading filepath from the recording returns 404."""
    try:
        recording.download(filepath)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        assert status == 404, f"Download of {filepath} failed with {status}"
        return
    raise AssertionError(f"{filepath} exists, but depth uploads lossless.bin only")


def verify_depth_uploads(
    *,
    recording: Recording,
    camera_names: Sequence[str],
    height: int,
    width: int,
    depth_mode: DepthMode,
) -> None:
    """Check every file each depth camera of a recording uploaded.

    Args:
        recording: The uploaded recording.
        camera_names: Depth camera names the producer logged.
        height: Logged frame height in pixels.
        width: Logged frame width in pixels.
        depth_mode: The depth input form the producer logged with.
    """
    depth_scale_m = depth_scale_for_mode(depth_mode)
    for camera_name in camera_names:
        prefix = f"{DATA_TYPE_DEPTH_IMAGES}/{to_safe_name(camera_name)}"
        trace = assert_depth_trace(
            json.loads(recording.download(f"{prefix}/{DEPTH_TRACE_FILENAME}")),
            depth_scale_m=depth_scale_m,
        )
        frames = assert_depth_frames(
            recording.download(f"{prefix}/{DEPTH_FRAMES_FILENAME}"),
            trace=trace,
            height=height,
            width=width,
        )
        assert_depth_preview(
            recording.download(f"{prefix}/{DEPTH_PREVIEW_FILENAME}"),
            lossless_frames=frames,
            depth_scale_m=depth_scale_m,
        )
        _assert_absent(recording, f"{prefix}/{PACKED_DEPTH_VIDEO_FILENAME}")
