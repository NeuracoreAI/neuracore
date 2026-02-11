"""Video trace writer.

Consumes a mixed stream of metadata (JSON) and RGB frames (raw bytes), writing
both lossy and lossless MP4 outputs plus a metadata trace file.
"""

from __future__ import annotations

import io
import pathlib
import threading
import time
from fractions import Fraction

import numpy as np

PTS_FRACT = 1000000
CHUNK_MULTIPLE = 256 * 1024
MB_CHUNK = 4 * CHUNK_MULTIPLE
CHUNK_SIZE = 64 * MB_CHUNK

LOSSY_VIDEO_NAME = "lossy.mp4"
LOSSLESS_VIDEO_NAME = "lossless.mp4"
TRACE_FILE = "trace.json"


class DiskVideoEncoder:
    """Encode frames into an MP4 container buffered in memory.

    Output bytes are flushed to disk in chunks.
    """

    def __init__(
        self,
        *,
        filepath: pathlib.Path,
        width: int,
        height: int,
        codec: str,
        pixel_format: str,
        codec_context_options: dict[str, str] | None,
        chunk_size: int = CHUNK_SIZE,
    ) -> None:
        """Initialise an on-disk MP4 encoder.

        Args:
            filepath: Output file path.
            width: Frame width in pixels.
            height: Frame height in pixels.
            codec: FFmpeg codec name (e.g. "libx264").
            pixel_format: Pixel format for the stream (e.g. "yuv420p").
            codec_context_options: Codec options passed into PyAV codec context.
            chunk_size: Buffered write chunk size.

        Returns:
            None
        """
        self.width = width
        self.height = height
        self.codec = codec
        self.pixel_format = pixel_format
        self.codec_context_options = codec_context_options
        self.chunk_size = chunk_size
        self.container_format = "mp4"

        self._fh = open(filepath, "wb")

        import av

        self._av = av

        self.buffer = io.BytesIO()
        self.container = self._av.open(
            self.buffer,
            mode="w",
            format=self.container_format,
            options={"movflags": "frag_keyframe+empty_moov"},
        )

        self.stream = self.container.add_stream(self.codec, rate=PTS_FRACT)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = self.pixel_format
        if self.codec_context_options is not None:
            self.stream.codec_context.options = self.codec_context_options

        self.stream.time_base = Fraction(1, PTS_FRACT)

        self.first_timestamp: float | None = None
        self.last_pts: int | None = None

        self.upload_buffer = bytearray()
        self.last_write_position = 0
        self._last_progress_update_timer = 0.0
        self._lock = threading.Lock()
        self._finished = False

    def add_frame(self, *, timestamp: float, np_frame: np.ndarray) -> None:
        """Encode a single frame at the provided timestamp.

        Args:
            timestamp: Frame timestamp in seconds.
            np_frame: RGB frame as a NumPy array.

        Returns:
            None
        """
        with self._lock:
            if self._finished:
                return

            pts = self._compute_pts(timestamp=timestamp)

            frame = self._av.VideoFrame.from_ndarray(np_frame, format="rgb24")
            frame = frame.reformat(format=self.pixel_format)
            frame.pts = pts

            for packet in self.stream.encode(frame):
                self.container.mux(packet)

            self._stage_and_flush_if_needed()

    def _compute_pts(self, *, timestamp: float) -> int:
        """Compute monotonic PTS for a timestamp.

        Args:
            timestamp: Timestamp in seconds.

        Returns:
            Integer PTS for the stream time base.
        """
        if self.first_timestamp is None:
            self.first_timestamp = timestamp

        relative_time = timestamp - self.first_timestamp
        pts = int(relative_time * PTS_FRACT)

        if self.last_pts is not None and pts <= self.last_pts:
            pts = self.last_pts + 1
        self.last_pts = pts

        return pts

    def _stage_and_flush_if_needed(self) -> None:
        """Stage pending bytes and flush any full chunks to disk.

        Returns:
            None
        """
        current_position = self.buffer.tell()
        pending_bytes = current_position - self.last_write_position
        if pending_bytes >= self.chunk_size:
            self._stage_pending_bytes(
                current_position=current_position,
                pending_bytes=pending_bytes,
            )
            self._flush_full_chunks()

    def _stage_pending_bytes(
        self,
        *,
        current_position: int,
        pending_bytes: int,
    ) -> None:
        """Move pending container bytes into the upload buffer.

        Args:
            current_position: Current position in the in-memory container buffer.
            pending_bytes: Bytes produced since `last_write_position`.

        Returns:
            None
        """
        self.buffer.seek(self.last_write_position)
        chunk_bytes = self.buffer.read(pending_bytes)
        self.upload_buffer.extend(chunk_bytes)
        self.last_write_position = current_position
        self.buffer.seek(current_position)

    def _flush_full_chunks(self) -> None:
        """Write any full chunks from the upload buffer to disk.

        Returns:
            None
        """
        while len(self.upload_buffer) >= self.chunk_size:
            chunk = bytes(self.upload_buffer[: self.chunk_size])
            del self.upload_buffer[: self.chunk_size]
            self._fh.write(chunk)

        self._compact_buffer_if_needed()

        now = time.time()
        if now - self._last_progress_update_timer >= 30.0:
            self._last_progress_update_timer = now

    def _compact_buffer_if_needed(self) -> None:
        """Compact the in-memory container buffer to avoid unbounded growth.

        Returns:
            None
        """
        if self.last_write_position <= 0:
            return

        if self.last_write_position < (self.chunk_size * 4):
            return

        self.buffer.seek(self.last_write_position)
        remaining = self.buffer.read()

        self.buffer = io.BytesIO()
        self.buffer.write(remaining)

        self.last_write_position = 0
        self.buffer.seek(len(remaining))

    def finish(self) -> None:
        """Finalise encoding, flush remaining bytes, and close the output file.

        Returns:
            None
        """
        with self._lock:
            if self._finished:
                return
            self._finished = True

            for packet in self.stream.encode(None):
                self.container.mux(packet)

            self.container.close()

            current_position = self.buffer.tell()
            pending_bytes = current_position - self.last_write_position
            if pending_bytes > 0:
                self._stage_pending_bytes(
                    current_position=current_position,
                    pending_bytes=pending_bytes,
                )

            if self.upload_buffer:
                self._fh.write(bytes(self.upload_buffer))
                self.upload_buffer = bytearray()

            self._fh.flush()
            self._fh.close()
