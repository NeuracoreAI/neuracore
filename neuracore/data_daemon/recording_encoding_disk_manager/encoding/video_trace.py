"""Consumes a mixed stream of metadata (JSON) and RGB frames (raw bytes)."""

from __future__ import annotations

import json
import pathlib
import struct
import time
from typing import Any

import numpy as np

from neuracore.data_daemon.recording_encoding_disk_manager.encoding.disk_video_encoder import (  # noqa: E501
    DiskVideoEncoder,
)

PTS_FRACT = 1000000
CHUNK_MULTIPLE = 256 * 1024
MB_CHUNK = 4 * CHUNK_MULTIPLE
CHUNK_SIZE = 64 * MB_CHUNK

LOSSY_VIDEO_NAME = "lossy.mp4"
LOSSLESS_VIDEO_NAME = "lossless.mp4"
TRACE_FILE = "trace.json"


class VideoTrace:
    """Write RGB payloads to MP4 outputs and persist associated metadata."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        *,
        chunk_size: int = CHUNK_SIZE,
        lossy_name: str = LOSSY_VIDEO_NAME,
        lossless_name: str = LOSSLESS_VIDEO_NAME,
    ) -> None:
        """Initialise a video trace writer.

        Args:
            output_dir: Directory where output files are written.
            chunk_size: Buffered write chunk size for video outputs.
            lossy_name: Filename for the lossy MP4 output.
            lossless_name: Filename for the lossless MP4 output.

        Returns:
            None
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lossy_path = self.output_dir / lossy_name
        self.lossless_path = self.output_dir / lossless_name
        self.trace_path = self.output_dir / TRACE_FILE

        self.chunk_size = chunk_size

        self.width: int | None = None
        self.height: int | None = None

        self._lossless_encoder: DiskVideoEncoder | None = None
        self._lossy_encoder: DiskVideoEncoder | None = None

        self._frame_metadata: list[dict[str, Any]] = []
        self._frame_index = 0

    def add_payload(self, payload: bytes) -> None:
        """Consume a payload that is either JSON metadata or raw RGB frame bytes.

        The payload can be in one of three formats:
        1. Pure JSON metadata (backward compatibility)
        2. Pure raw RGB frame bytes
        3. Combined packet: [4-byte metadata_len][JSON metadata][frame_bytes]

        Args:
            payload: Incoming payload bytes.

        Returns:
            None
        """
        parsed = self._try_parse_json(payload)
        if parsed is not None:
            self._handle_metadata(parsed)
            return

        if self._try_handle_combined_packet(payload):
            return

        self._handle_frame_bytes(payload)

    def _try_handle_combined_packet(self, payload: bytes) -> bool:
        """Try to parse payload as combined [4B len][JSON][frame] format.

        Args:
            payload: Incoming payload bytes.

        Returns:
            True if the payload was successfully handled as a combined packet,
            False otherwise.
        """
        if len(payload) < 4:
            return False

        metadata_len = struct.unpack("<I", payload[:4])[0]
        if metadata_len == 0 or metadata_len > len(payload) - 4:
            return False

        json_bytes = payload[4 : 4 + metadata_len]
        parsed = self._try_parse_json(json_bytes)
        if parsed is None:
            return False

        self._handle_metadata(parsed)

        frame_bytes = payload[4 + metadata_len :]
        if len(frame_bytes) > 0:
            self._handle_frame_bytes(frame_bytes)

        return True

    def _try_parse_json(self, payload: bytes) -> Any | None:
        """Attempt to parse a payload as JSON.

        Args:
            payload: Incoming payload bytes.

        Returns:
            Parsed JSON object if parsing succeeds, otherwise None.
        """
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    def _handle_metadata(self, obj: Any) -> None:
        """Handle a decoded metadata object and store it.

        Args:
            obj: Decoded JSON object.

        Returns:
            None
        """
        if isinstance(obj, list):
            for item in obj:
                self._handle_metadata(item)
            return

        if not isinstance(obj, dict):
            return

        width_value = obj.get("width")
        height_value = obj.get("height")
        if isinstance(width_value, int) and isinstance(height_value, int):
            if self.width is None and self.height is None:
                self.width, self.height = width_value, height_value

        obj["frame"] = None
        self._frame_metadata.append(obj)

    def _ensure_encoders(self) -> None:
        """Create encoders if required, after width/height has been learned.

        Returns:
            None

        Raises:
            RuntimeError: If width/height has not been provided via metadata.
        """
        if self.width is None or self.height is None:
            raise RuntimeError(
                "VideoTrace needs width/height before frames. Send metadata first."
            )

        if self._lossless_encoder is None:
            self._lossless_encoder = DiskVideoEncoder(
                filepath=self.lossless_path,
                width=self.width,
                height=self.height,
                codec="libx264",
                pixel_format="yuv444p10le",
                codec_context_options={"qp": "0", "preset": "ultrafast"},
                chunk_size=self.chunk_size,
            )

        if self._lossy_encoder is None:
            self._lossy_encoder = DiskVideoEncoder(
                filepath=self.lossy_path,
                width=self.width,
                height=self.height,
                codec="libx264",
                pixel_format="yuv420p",
                codec_context_options=None,
                chunk_size=self.chunk_size,
            )

    def _handle_frame_bytes(self, frame_bytes: bytes) -> None:
        """Validate and encode a raw RGB frame payload.

        Args:
            frame_bytes: Raw RGB frame bytes.

        Returns:
            None
        """
        self._ensure_encoders()

        if self.width is None or self.height is None:
            raise RuntimeError(
                "VideoTrace missing width/height after encoder initialisation"
            )
        if self._lossless_encoder is None or self._lossy_encoder is None:
            raise RuntimeError(
                "VideoTrace encoders unexpectedly None after initialisation"
            )

        expected_size = self.width * self.height * 3
        if len(frame_bytes) != expected_size:
            raise ValueError(
                f"Unexpected frame size: got={len(frame_bytes)} "
                f"expected={expected_size}"
            )

        np_frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )

        timestamp_value = self._get_frame_timestamp()
        self._lossless_encoder.add_frame(timestamp=timestamp_value, np_frame=np_frame)
        self._lossy_encoder.add_frame(timestamp=timestamp_value, np_frame=np_frame)

        self._frame_index += 1

    def _get_frame_timestamp(self) -> float:
        """Resolve the timestamp for the current frame.

        Returns:
            Timestamp in seconds for the current frame.
        """
        ts: Any = None
        if self._frame_index < len(self._frame_metadata):
            ts = self._frame_metadata[self._frame_index].get("timestamp")
        if not isinstance(ts, (int, float)):
            ts = time.time()
        return float(ts)

    def finish(self) -> None:
        """Finalise encoders and write the metadata JSON trace file.

        Returns:
            None
        """
        if self._lossless_encoder is not None:
            self._lossless_encoder.finish()
        if self._lossy_encoder is not None:
            self._lossy_encoder.finish()

        for i, frame_meta in enumerate(self._frame_metadata):
            frame_meta["frame_idx"] = i
            frame_meta["frame"] = None

        self.trace_path.write_text(
            json.dumps(self._frame_metadata, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )

        self._frame_metadata = []
