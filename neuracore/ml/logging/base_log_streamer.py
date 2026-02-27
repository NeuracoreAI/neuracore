"""Shared background log streamer implementation."""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class BytesUploadStorageHandler(Protocol):
    """Protocol for storage handlers that support bytes uploads."""

    log_to_cloud: bool

    def upload_bytes(
        self,
        data: bytes,
        remote_filepath: str,
        content_type: str = "application/octet-stream",
    ) -> bool:
        """Upload raw bytes to remote storage."""
        ...

    def upload_file(
        self,
        local_path: Path,
        remote_filepath: str,
        content_type: str = "application/octet-stream",
    ) -> bool:
        """Upload a local file to remote storage."""
        ...


class BaseLogStreamer(ABC):
    """Tail a local log file and upload chunked log objects."""

    def __init__(
        self,
        storage_handler: BytesUploadStorageHandler,
        output_dir: Path,
        chunk_max_lines: int = 100,
        flush_interval_s: int = 2,
        max_buffer_bytes: int = 1024 * 1024,
    ) -> None:
        """Initialize shared streamer settings and mutable sync state."""
        self.storage_handler = storage_handler
        self.output_dir = output_dir
        self.chunk_max_lines = chunk_max_lines
        self.flush_interval_s = flush_interval_s
        self.max_buffer_bytes = max_buffer_bytes

        self._stop_event = threading.Event()
        self._sync_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._file_offset = 0
        self._line_buffer: list[str] = []
        self._buffer_bytes = 0
        self._chunk_index = 1
        self._total_lines_uploaded = 0
        self._total_bytes_uploaded = 0
        self._first_chunk_upload_ts: float | None = None
        self._last_chunk_upload_ts: float | None = None

    @property
    @abstractmethod
    def _log_path(self) -> Path:
        """Path of the local log file to stream."""

    @abstractmethod
    def _chunk_remote_path(self, chunk_index: int) -> str:
        """Build cloud path for a chunk index."""

    @abstractmethod
    def _index_remote_path(self) -> str:
        """Build cloud path for index manifest."""

    @abstractmethod
    def _retry_warning_message(self) -> str:
        """Warning text for retry logs."""

    def _before_sync_locked(self, final_sync: bool) -> None:
        """Hook for subclasses to run pre-sync work (for example Hydra uploads)."""

    def start(self) -> None:
        """Start the background streaming loop."""
        if not self.storage_handler.log_to_cloud:
            return
        if self._sync_thread is not None:
            return
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

    def close(self) -> None:
        """Stop streaming and perform a final upload flush."""
        if self._sync_thread is not None:
            self._stop_event.set()
            self._sync_thread.join(timeout=30)
        with self._lock:
            self._sync_once_locked(final_sync=True)

    def _sync_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                self._sync_once_locked(final_sync=False)
            self._stop_event.wait(self.flush_interval_s)

    def _sync_once_locked(self, final_sync: bool) -> None:
        self._before_sync_locked(final_sync=final_sync)
        self._read_new_log_lines()
        self._flush_available_chunks_locked(force_partial=True)
        if final_sync:
            self._upload_index_manifest()

    def _read_new_log_lines(self) -> None:
        if not self._log_path.exists() or not self._log_path.is_file():
            return
        if self._buffer_bytes >= self.max_buffer_bytes:
            return

        file_size = self._log_path.stat().st_size
        if file_size < self._file_offset:
            self._file_offset = 0

        with open(self._log_path, encoding="utf-8", errors="replace") as f:
            f.seek(self._file_offset)
            while self._buffer_bytes < self.max_buffer_bytes:
                position_before_line = f.tell()
                line = f.readline()
                if line == "":
                    break
                line_bytes = len(line.encode("utf-8"))
                if self._buffer_bytes + line_bytes > self.max_buffer_bytes:
                    f.seek(position_before_line)
                    break
                self._line_buffer.append(line)
                self._buffer_bytes += line_bytes
            self._file_offset = f.tell()

    def _flush_available_chunks_locked(self, force_partial: bool) -> None:
        while len(self._line_buffer) >= self.chunk_max_lines:
            uploaded = self._flush_chunk_with_retry_locked(self.chunk_max_lines)
            if not uploaded:
                return

        if force_partial and self._line_buffer:
            self._flush_chunk_with_retry_locked(len(self._line_buffer))

    def _flush_chunk_with_retry_locked(self, line_count: int) -> bool:
        for attempt in range(2):
            if self._flush_chunk_locked(line_count=line_count):
                return True
            if attempt == 0:
                logger.warning(self._retry_warning_message())
        return False

    def _flush_chunk_locked(self, line_count: int) -> bool:
        if not self._line_buffer:
            return True
        chunk_lines = self._line_buffer[:line_count]
        content = "".join(
            line if line.endswith(("\n", "\r")) else f"{line}\n" for line in chunk_lines
        )
        payload = content.encode("utf-8")
        consumed_bytes = sum(len(line.encode("utf-8")) for line in chunk_lines)
        remote_path = self._chunk_remote_path(self._chunk_index)
        uploaded = self.storage_handler.upload_bytes(
            data=payload,
            remote_filepath=remote_path,
            content_type="text/plain",
        )
        if not uploaded:
            return False

        now = time.time()
        if self._first_chunk_upload_ts is None:
            self._first_chunk_upload_ts = now
        self._last_chunk_upload_ts = now
        self._total_lines_uploaded += len(chunk_lines)
        self._total_bytes_uploaded += len(payload)
        self._chunk_index += 1
        del self._line_buffer[:line_count]
        self._buffer_bytes = max(0, self._buffer_bytes - consumed_bytes)
        return True

    def _upload_index_manifest(self) -> None:
        manifest = {
            "total_chunks": self._chunk_index - 1,
            "total_lines_uploaded": self._total_lines_uploaded,
            "total_bytes_uploaded": self._total_bytes_uploaded,
            "first_chunk_upload_timestamp": self._first_chunk_upload_ts,
            "last_chunk_upload_timestamp": self._last_chunk_upload_ts,
        }
        self.storage_handler.upload_bytes(
            data=json.dumps(manifest, sort_keys=True).encode("utf-8"),
            remote_filepath=self._index_remote_path(),
            content_type="application/json",
        )
