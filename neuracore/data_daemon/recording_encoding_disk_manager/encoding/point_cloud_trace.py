"""Persist binary point cloud wire frames and a JSON metadata trace."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from neuracore_types.nc_data.point_cloud_data import (
    POINT_CLOUD_TRACE_BIN_FILENAME,
    decode_point_cloud_wire_metadata,
)

TRACE_INDEX_FILE = "trace.json"


class PointCloudTrace:
    """Write point cloud wire frames to trace.bin and metadata to trace.json."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        *,
        bin_filename: str = POINT_CLOUD_TRACE_BIN_FILENAME,
        index_filename: str = TRACE_INDEX_FILE,
    ) -> None:
        """Initialise the point cloud trace writer."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bin_path = self.output_dir / bin_filename
        self.index_path = self.output_dir / index_filename
        self._frames: list[dict[str, Any]] = []
        self._offset = 0
        self._bin_file = open(self.bin_path, "wb")

    def add_payload(self, payload: bytes) -> None:
        """Validate and append one wire frame to trace.bin."""
        if not payload:
            return

        metadata, _ = decode_point_cloud_wire_metadata(payload)
        self._bin_file.write(payload)
        self._frames.append({
            "type": "PointCloudData",
            "timestamp": metadata["timestamp"],
            "extrinsics": metadata.get("extrinsics"),
            "intrinsics": metadata.get("intrinsics"),
            "offset": self._offset,
            "length": len(payload),
        })
        self._offset += len(payload)

    def finish(self) -> None:
        """Write the metadata JSON trace file and close trace.bin."""
        for frame_idx, frame_meta in enumerate(self._frames):
            frame_meta["frame_idx"] = frame_idx

        self._bin_file.flush()
        self._bin_file.close()
        self.index_path.write_text(
            json.dumps(self._frames, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
