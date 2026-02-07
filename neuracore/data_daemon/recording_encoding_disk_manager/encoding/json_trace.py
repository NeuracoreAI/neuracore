"""Streams a JSON array of frame objects to disk while buffering writes in chunks."""

from __future__ import annotations

import io
import json
import pathlib
from typing import Any

CHUNK_MULTIPLE = 256 * 1024
MB_CHUNK = 4 * CHUNK_MULTIPLE
CHUNK_SIZE = 64 * MB_CHUNK


class JsonTrace:
    """Write frames to a single JSON array file on disk."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        filename: str = "trace.json",
        chunk_size: int = CHUNK_SIZE,
    ) -> None:
        """Initialise the trace writer.

        Args:
            output_dir: Directory where the trace file will be written.
            filename: Output filename within `output_dir`.
            chunk_size: Minimum buffered size before writing to disk.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / filename

        self.chunk_size = chunk_size

        self.buffer = io.BytesIO()
        self.upload_buffer = bytearray()
        self.last_write_position = 0

        self._started = False
        self._first_entry = True

        self._fh = open(self.filepath, "wb")

    def add_frame(self, data_entry: dict[str, Any]) -> None:
        """Append one frame entry to the JSON array.

        Args:
            data_entry: JSON-serialisable dict representing a single frame.

        Returns:
            None
        """
        if not self._started:
            self.buffer.write(b"[")
            self._started = True

        if not self._first_entry:
            self.buffer.write(b",")
        else:
            self._first_entry = False

        entry_json = json.dumps(data_entry, separators=(",", ":"), ensure_ascii=False)
        self.buffer.write(entry_json.encode("utf-8"))

        current_position = self.buffer.tell()
        pending_bytes = current_position - self.last_write_position

        if pending_bytes >= self.chunk_size:
            self._stage_pending_bytes_for_upload(
                current_position=current_position, pending_bytes=pending_bytes
            )
            self._flush_full_chunks_to_disk()

    def _stage_pending_bytes_for_upload(
        self,
        *,
        current_position: int,
        pending_bytes: int,
    ) -> None:
        """Move buffered bytes into the upload buffer.

        Args:
            current_position: Current buffer position.
            pending_bytes: Number of bytes since the last staged position.

        Returns:
            None
        """
        self.buffer.seek(self.last_write_position)
        chunk_bytes = self.buffer.read(pending_bytes)
        self.upload_buffer.extend(chunk_bytes)
        self.last_write_position = current_position
        self.buffer.seek(current_position)

    def _flush_full_chunks_to_disk(self) -> None:
        """Write any full chunks from the upload buffer to disk.

        Returns:
            None
        """
        while len(self.upload_buffer) >= self.chunk_size:
            chunk = bytes(self.upload_buffer[: self.chunk_size])
            del self.upload_buffer[: self.chunk_size]
            self._fh.write(chunk)

        self._compact_buffer_if_needed()

    def _compact_buffer_if_needed(self) -> None:
        """Compact the in-memory buffer to avoid unbounded growth.

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
        """Finalise the JSON array and close the file handle.

        Returns:
            None
        """
        if not self._started:
            self._fh.write(b"[]")
            self._fh.flush()
            self._fh.close()
            return

        self.buffer.write(b"]")

        current_position = self.buffer.tell()
        pending_bytes = current_position - self.last_write_position
        if pending_bytes > 0:
            self._stage_pending_bytes_for_upload(
                current_position=current_position, pending_bytes=pending_bytes
            )

        if self.upload_buffer:
            self._fh.write(bytes(self.upload_buffer))
            self.upload_buffer = bytearray()

        self._fh.flush()
        self._fh.close()
