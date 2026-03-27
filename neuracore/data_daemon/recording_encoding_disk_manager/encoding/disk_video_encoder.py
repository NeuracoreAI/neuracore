"""Video trace writer.

Consumes a mixed stream of metadata (JSON) and RGB frames (raw bytes), writing
both lossy and lossless MP4 outputs plus a metadata trace file.
"""

from __future__ import annotations

import io
import logging
import os
import pathlib
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from fractions import Fraction

import numpy as np

PTS_FRACT = 1000000
CHUNK_MULTIPLE = 256 * 1024
MB_CHUNK = 4 * CHUNK_MULTIPLE
CHUNK_SIZE = 64 * MB_CHUNK
FFMPEG_PIPE_WRITE_CHUNK_SIZE = 1024 * 1024

LOSSY_VIDEO_NAME = "lossy.mp4"
LOSSLESS_VIDEO_NAME = "lossless.mp4"
TRACE_FILE = "trace.json"

logger = logging.getLogger(__name__)
_FFMPEG_AVAILABLE: bool | None = None


def _is_ffmpeg_available() -> bool:
    """Return whether ffmpeg CLI is available on the host."""
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is None:
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            _FFMPEG_AVAILABLE = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            _FFMPEG_AVAILABLE = False
    return _FFMPEG_AVAILABLE


class BaseDiskVideoEncoder(ABC):
    """Abstract video encoder interface used by VideoTrace."""

    @abstractmethod
    def add_frame(self, *, timestamp: float, np_frame: np.ndarray) -> None:
        """Encode a frame."""

    @abstractmethod
    def finish(self) -> None:
        """Flush and close encoder resources."""


class PyAVDiskVideoEncoder(BaseDiskVideoEncoder):
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
            # Memory spike 4k videos
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

        NOTE: Must compact in place: the container holds a reference to this buffer
        and keeps writing to it. Replacing self.buffer would orphan that output and
        drop all subsequent frames (lossless hits this often → fewer frames).
        """
        if self.last_write_position <= 0:
            return

        if self.last_write_position < (self.chunk_size * 4):
            return

        self.buffer.seek(self.last_write_position)
        remaining = self.buffer.read()
        self.buffer.seek(0)
        self.buffer.write(remaining)
        self.buffer.truncate(len(remaining))
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


class FfmpegDiskVideoEncoder(BaseDiskVideoEncoder):
    """Encode frames by piping raw RGB frames into ffmpeg CLI."""

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
        """Initialise an ffmpeg CLI encoder process.

        Args:
            filepath: Output file path.
            width: Frame width in pixels.
            height: Frame height in pixels.
            codec: FFmpeg codec name (e.g. "libx264").
            pixel_format: Pixel format for the stream (e.g. "yuv420p").
            codec_context_options: Codec options passed to ffmpeg.
            chunk_size: Buffered write chunk size (kept for API parity).

        Returns:
            None
        """
        self.width = width
        self.height = height
        self.codec = codec
        self.pixel_format = pixel_format
        self.codec_context_options = codec_context_options
        self.chunk_size = chunk_size
        self._lock = threading.Lock()
        self._finished = False

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-avioflags",
            "direct",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(PTS_FRACT),
            "-i",
            "pipe:0",
            "-c:v",
            self.codec,
            "-pix_fmt",
            self.pixel_format,
        ]
        for key, value in (self.codec_context_options or {}).items():
            cmd.extend([f"-{key}", value])
        cmd.extend([
            "-movflags",
            "frag_keyframe+empty_moov",
            "-flush_packets",
            "1",
            str(filepath),
        ])

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def add_frame(self, *, timestamp: float, np_frame: np.ndarray) -> None:
        """Write a single RGB frame to ffmpeg stdin."""
        del timestamp
        with self._lock:
            if self._finished:
                return
            if self._proc.stdin is None:
                raise RuntimeError("ffmpeg encoder stdin is not available")
            if np_frame.dtype != np.uint8:
                frame = np_frame.astype(np.uint8, copy=False)
            else:
                frame = np_frame
            if frame.shape != (self.height, self.width, 3):
                raise ValueError(
                    "Unexpected frame shape for ffmpeg encoder: "
                    f"got={frame.shape} expected={(self.height, self.width, 3)}"
                )
            if not frame.flags.c_contiguous:
                frame = np.ascontiguousarray(frame)

            try:
                self._write_frame(frame)
            except BrokenPipeError as exc:
                stderr_msg = ""
                if self._proc.stderr is not None:
                    stderr_msg = self._proc.stderr.read().decode("utf-8", "ignore")
                raise RuntimeError(
                    f"ffmpeg encoder process closed stdin unexpectedly: {stderr_msg}"
                ) from exc

    def _write_frame(self, frame: np.ndarray) -> None:
        """Write a frame to ffmpeg without allocating a new bytes object."""
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg encoder stdin is not available")

        fd = self._proc.stdin.fileno()
        frame_view = memoryview(frame).cast("B")
        bytes_written = 0

        while bytes_written < len(frame_view):
            chunk_end = bytes_written + FFMPEG_PIPE_WRITE_CHUNK_SIZE
            wrote = os.write(fd, frame_view[bytes_written:chunk_end])
            if wrote <= 0:
                raise BrokenPipeError("ffmpeg encoder write returned no progress")
            bytes_written += wrote

    def finish(self) -> None:
        """Close ffmpeg stdin and wait for process completion."""
        with self._lock:
            if self._finished:
                return
            self._finished = True

            if self._proc.stdin is not None:
                self._proc.stdin.close()

            stderr_output = b""
            if self._proc.stderr is not None:
                stderr_output = self._proc.stderr.read()
                self._proc.stderr.close()

            return_code = self._proc.wait()
            if return_code != 0:
                raise RuntimeError(
                    "ffmpeg encoder failed "
                    f"(exit_code={return_code}): "
                    f"{stderr_output.decode('utf-8', 'ignore')}"
                )


def DiskVideoEncoder(
    *,
    filepath: pathlib.Path,
    width: int,
    height: int,
    codec: str,
    pixel_format: str,
    codec_context_options: dict[str, str] | None,
    chunk_size: int = CHUNK_SIZE,
) -> BaseDiskVideoEncoder:
    """Create either an ffmpeg or PyAV encoder implementation.

    Args:
        filepath: Output file path.
        width: Frame width in pixels.
        height: Frame height in pixels.
        codec: FFmpeg codec name (e.g. "libx264").
        pixel_format: Pixel format for the stream (e.g. "yuv420p").
        codec_context_options: Codec options passed to the encoder backend.
        chunk_size: Buffered write chunk size.

    Returns:
        Concrete disk video encoder implementation.
    """
    if _is_ffmpeg_available():
        try:
            return FfmpegDiskVideoEncoder(
                filepath=filepath,
                width=width,
                height=height,
                codec=codec,
                pixel_format=pixel_format,
                codec_context_options=codec_context_options,
                chunk_size=chunk_size,
            )
        except Exception:
            logger.exception("ffmpeg encoder init failed, falling back to PyAV encoder")

    return PyAVDiskVideoEncoder(
        filepath=filepath,
        width=width,
        height=height,
        codec=codec,
        pixel_format=pixel_format,
        codec_context_options=codec_context_options,
        chunk_size=chunk_size,
    )
