"""Single-producer/single-consumer buffer used by the data daemon.

Local buffers keep the existing in-process byte-ring behavior. Shared buffers
wrap ``zerobuffer-ipc`` so producer/daemon transport stays behind a small local
abstraction instead of leaking the third-party API across the codebase.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import timedelta

logger = logging.getLogger(__name__)

_SHARED_METADATA_SIZE = 4096
_WRITE_TIMEOUT = timedelta(days=1)
_OPEN_SHARED_TIMEOUT_S = 5.0
_OPEN_SHARED_RETRY_SLEEP_S = 0.01

try:
    from zerobuffer import BufferConfig as _SharedBufferConfig
    from zerobuffer import Frame as _SharedFrame
    from zerobuffer import Reader as _SharedReader
    from zerobuffer import Writer as _SharedWriter
    from zerobuffer.exceptions import (
        WriterAlreadyConnectedException as _SharedWriterAlreadyConnectedException,
        WriterDeadException as _SharedWriterDeadException,
        ZeroBufferException as _SharedZeroBufferException,
    )

    _SHARED_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised in envs without deps
    _SharedBufferConfig = None
    _SharedFrame = None
    _SharedReader = None
    _SharedWriter = None
    _SharedWriterAlreadyConnectedException = Exception
    _SharedWriterDeadException = Exception
    _SharedZeroBufferException = Exception
    _SHARED_IMPORT_ERROR = exc


class RingBuffer:
    """Circular byte buffer with an optional zerobuffer-backed shared mode.

    Local instances behave like the original byte ring. Shared instances expose
    the same ``write/read/peek/available`` API but internally move complete
    packets as zerobuffer frames.
    """

    def __init__(
        self,
        size: int = 1024,
        *,
        _shared_name: str | None = None,
        _shared_reader: _SharedReader | None = None,
        _shared_writer: _SharedWriter | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("RingBuffer size must be > 0")
        if _shared_reader is not None and _shared_writer is not None:
            raise ValueError("RingBuffer shared endpoint must be reader or writer")

        self.size = int(size)
        self.buffer = (
            None
            if _shared_reader is not None or _shared_writer is not None
            else bytearray(self.size)
        )
        self._shared_name = _shared_name
        self._shared_reader = _shared_reader
        self._shared_writer = _shared_writer
        self._shared_frame: _SharedFrame | None = None
        self._shared_frame_view: memoryview | None = None
        self._shared_frame_offset = 0
        self._read_index = 0
        self._write_index = 0
        self._lock = threading.RLock()
        self._not_full = threading.Condition(self._lock)

    @classmethod
    def supports_shared_transport(cls) -> bool:
        """Return True when zerobuffer shared transport is available."""
        return _SharedReader is not None and _SharedWriter is not None

    @classmethod
    def create_shared(
        cls,
        size: int,
        *,
        name: str | None = None,
    ) -> RingBuffer:
        """Create the daemon-owned shared buffer endpoint.

        ``zerobuffer`` requires the reader/owner side to create the shared
        resources first, so this method is used by the daemon when it processes
        ``OPEN_RING_BUFFER``.
        """
        cls._require_shared_support()
        shared_name = name or f"neuracore-ring-buffer-{uuid.uuid4().hex}"
        reader = _SharedReader(
            shared_name,
            config=_SharedBufferConfig(
                metadata_size=_SHARED_METADATA_SIZE,
                payload_size=int(size),
            ),
        )
        return cls(
            size=int(size),
            _shared_name=shared_name,
            _shared_reader=reader,
        )

    @classmethod
    def open_shared(
        cls,
        name: str,
        size: int,
    ) -> RingBuffer:
        """Open the producer-side writer endpoint for an existing shared buffer."""
        cls._require_shared_support()

        deadline = time.monotonic() + _OPEN_SHARED_TIMEOUT_S
        while True:
            try:
                writer = _SharedWriter(name)
                writer.write_timeout = _WRITE_TIMEOUT
                return cls(
                    size=int(size),
                    _shared_name=name,
                    _shared_writer=writer,
                )
            except (
                _SharedZeroBufferException,
                _SharedWriterAlreadyConnectedException,
            ) as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Timed out opening shared ring buffer '{name}'"
                    ) from exc
                time.sleep(_OPEN_SHARED_RETRY_SLEEP_S)

    @property
    def shared_name(self) -> str | None:
        """Return the shared transport name, if any."""
        return self._shared_name

    @property
    def read_pos(self) -> int:
        """Current logical read position within the local buffer or frame."""
        if self._shared_reader is None and self._shared_writer is None:
            with self._lock:
                return self._read_index % self.size
        return self._shared_frame_offset

    @property
    def write_pos(self) -> int:
        """Current logical write position within the local buffer."""
        if self._shared_reader is None and self._shared_writer is None:
            with self._lock:
                return self._write_index % self.size
        return 0

    @property
    def used(self) -> int:
        """Return currently occupied bytes."""
        return self.available()

    def available(self) -> int:
        """Return the number of bytes currently available to read."""
        if self._shared_reader is None and self._shared_writer is None:
            with self._lock:
                return self._write_index - self._read_index

        if not self._ensure_shared_frame_loaded():
            return 0
        view = self._shared_frame_view
        if view is None:
            return 0
        return len(view) - self._shared_frame_offset

    def write(self, data: bytes | bytearray | memoryview) -> None:
        """Write bytes into the buffer, blocking until space is available."""
        view = data if isinstance(data, memoryview) else memoryview(data)
        if view.ndim != 1 or view.itemsize != 1 or view.format != "B":
            view = view.cast("B")
        length = len(view)
        if length == 0:
            return
        if length > self.size:
            raise ValueError("chunk exceeds ring buffer capacity")

        if self._shared_writer is None:
            if self._shared_reader is not None:
                raise RuntimeError("Shared ring buffer is read-only")
            with self._not_full:
                while self.size - (self._write_index - self._read_index) < length:
                    self._not_full.wait(timeout=0.1)
                self._write_into_storage(self._write_index % self.size, view)
                self._write_index += length
            return

        frame_buffer = self._shared_writer.get_frame_buffer(length)
        frame_buffer[:] = view
        self._shared_writer.commit_frame()

    def read(self, size: int) -> bytes | None:
        """Read and consume exactly ``size`` bytes, or return None if unavailable."""
        if size < 0:
            raise ValueError("size must be >= 0")
        if size == 0:
            return b""

        if self._shared_reader is None and self._shared_writer is None:
            with self._not_full:
                available = self._write_index - self._read_index
                if available < size:
                    return None
                result = self._read_from_storage(self._read_index % self.size, size)
                self._read_index += size
                if self._read_index == self._write_index:
                    self._read_index = 0
                    self._write_index = 0
                self._not_full.notify_all()
                return result

        if not self._ensure_shared_frame_loaded():
            return None

        view = self._shared_frame_view
        if view is None:
            return None
        remaining = len(view) - self._shared_frame_offset
        if remaining < size:
            return None

        start = self._shared_frame_offset
        end = start + size
        result = bytes(view[start:end])
        self._shared_frame_offset = end
        if self._shared_frame_offset == len(view):
            self._release_shared_frame()
        return result

    def peek(self, size: int) -> bytes | None:
        """Read exactly ``size`` bytes without consuming them."""
        if size < 0:
            raise ValueError("size must be >= 0")
        if size == 0:
            return b""

        if self._shared_reader is None and self._shared_writer is None:
            with self._lock:
                available = self._write_index - self._read_index
                if available < size:
                    return None
                return self._read_from_storage(self._read_index % self.size, size)

        if not self._ensure_shared_frame_loaded():
            return None

        view = self._shared_frame_view
        if view is None:
            return None
        remaining = len(view) - self._shared_frame_offset
        if remaining < size:
            return None

        start = self._shared_frame_offset
        return bytes(view[start : start + size])

    def close(self) -> None:
        """Close local handles to the underlying storage."""
        self._release_shared_frame()
        if self._shared_writer is not None:
            self._shared_writer.close()
            self._shared_writer = None
        if self._shared_reader is not None:
            self._shared_reader.close()
            self._shared_reader = None
        self.buffer = None

    def unlink(self) -> None:
        """Release underlying storage when possible.

        ``zerobuffer`` performs cleanup when the reader/owner endpoint closes, so
        this method intentionally stays as a no-op wrapper hook.
        """

    @classmethod
    def _require_shared_support(cls) -> None:
        if cls.supports_shared_transport():
            return
        detail = ""
        if _SHARED_IMPORT_ERROR is not None:
            detail = f": {_SHARED_IMPORT_ERROR}"
        raise RuntimeError(f"Shared ring transport unavailable{detail}")

    def _ensure_shared_frame_loaded(self) -> bool:
        if self._shared_frame_view is not None:
            return True
        if self._shared_reader is None:
            return False
        try:
            frame = self._shared_reader.read_frame(timeout=0.0)
        except _SharedWriterDeadException:
            return False
        except Exception as exc:
            logger.debug("Shared ring read unavailable: %s", exc)
            return False
        if frame is None:
            return False
        self._shared_frame = frame
        self._shared_frame_view = frame.data
        self._shared_frame_offset = 0
        return True

    def _release_shared_frame(self) -> None:
        frame = self._shared_frame
        self._shared_frame = None
        self._shared_frame_view = None
        self._shared_frame_offset = 0
        if frame is not None:
            frame.dispose()

    def _read_from_storage(self, start: int, size: int) -> bytes:
        storage = self._storage()
        first = min(size, self.size - start)
        first_slice = bytes(storage[start : start + first])
        if first == size:
            return first_slice
        second = size - first
        return first_slice + bytes(storage[:second])

    def _write_into_storage(self, start: int, data: memoryview) -> None:
        storage = self._storage()
        length = len(data)
        first = min(length, self.size - start)
        storage[start : start + first] = data[:first]
        if first < length:
            storage[: length - first] = data[first:]

    def _storage(self) -> bytearray:
        if self.buffer is None:
            raise RuntimeError("Ring buffer storage is closed")
        return self.buffer
