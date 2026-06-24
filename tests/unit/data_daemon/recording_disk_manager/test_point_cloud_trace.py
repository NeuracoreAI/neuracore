"""Tests for PointCloudTrace encoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from neuracore_types import PointCloudData
from neuracore_types.nc_data.point_cloud_data import (
    POINT_CLOUD_TRACE_BIN_FILENAME,
    decode_point_cloud_frame,
    encode_point_cloud_frame_parts,
)

from neuracore.data_daemon.recording_encoding_disk_manager.encoding import (
    point_cloud_trace,
)


def encode_wire_frame(data: PointCloudData) -> bytes:
    header, metadata_json, points_view, rgb_view = encode_point_cloud_frame_parts(data)
    parts: list[bytes | memoryview] = [header, metadata_json, points_view]
    if rgb_view is not None:
        parts.append(rgb_view)
    return b"".join(parts)


def test_point_cloud_trace_writes_bin_and_metadata_array(tmp_path: Path) -> None:
    trace = point_cloud_trace.PointCloudTrace(output_dir=tmp_path / "trace")
    frame_a = encode_wire_frame(
        PointCloudData(
            timestamp=1.0,
            points=np.array([[1.0, 2.0, 3.0]], dtype=np.float16),
        )
    )
    frame_b = encode_wire_frame(
        PointCloudData(
            timestamp=2.0,
            points=np.array([[4.0, 5.0, 6.0]], dtype=np.float16),
        )
    )
    trace.add_payload(frame_a)
    trace.add_payload(frame_b)
    trace.finish()

    bin_bytes = (tmp_path / "trace" / POINT_CLOUD_TRACE_BIN_FILENAME).read_bytes()
    index = json.loads((tmp_path / "trace" / "trace.json").read_text(encoding="utf-8"))

    assert isinstance(index, list)
    assert len(index) == 2
    assert index[0]["frame_idx"] == 0
    assert index[0]["offset"] == 0
    assert index[0]["length"] == len(frame_a)
    assert "points" not in index[0]
    assert index[1]["frame_idx"] == 1
    assert index[1]["offset"] == len(frame_a)
    assert index[1]["length"] == len(frame_b)
    assert bin_bytes == frame_a + frame_b

    decoded_a = decode_point_cloud_frame(
        bin_bytes[index[0]["offset"] : index[0]["offset"] + index[0]["length"]]
    )
    assert decoded_a.timestamp == 1.0


def test_point_cloud_trace_rejects_invalid_payload(tmp_path: Path) -> None:
    trace = point_cloud_trace.PointCloudTrace(output_dir=tmp_path / "trace")
    with pytest.raises(ValueError):
        trace.add_payload(b'{"type":"PointCloudData"}')
