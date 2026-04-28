"""Shared zerobuffer transport used by video producers and the daemon."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import timedelta

from neuracore.data_daemon.lifecycle.runtime_recovery import (
    ensure_shared_memory_capacity,
    shared_memory_required_bytes,
)

logger = logging.getLogger(__name__)

_SHARED_METADATA_SIZE = 4096
_WRITE_TIMEOUT = timedelta(days=1)
_OPEN_SHARED_TIMEOUT_S = 5.0
_OPEN_SHARED_RETRY_SLEEP_S = 0.01

try:
    from zerobuffer import BufferConfig as _SharedBufferConfig
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
    _SharedReader = None
    _SharedWriter = None
    _SharedWriterAlreadyConnectedException = Exception
    _SharedWriterDeadException = Exception
    _SharedZeroBufferException = Exception
    _SHARED_IMPORT_ERROR = exc


class RingBuffer:
    """Shared zerobuffer transport for packet-sized producer/daemon exchange."""

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
        if (_shared_reader is None) == (_shared_writer is None):
            raise ValueError("RingBuffer requires exactly one shared endpoint")

        self.size = int(size)
        self._shared_name = _shared_name
        self._shared_reader = _shared_reader
        self._shared_writer = _shared_writer

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
        ensure_shared_memory_capacity(
            shared_memory_required_bytes(
                int(size),
                metadata_size=_SHARED_METADATA_SIZE,
            )
        )
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
        """Current logical read position within the shared buffer."""
        return 0

    @property
    def write_pos(self) -> int:
        """Current logical write position within the shared buffer."""
        return 0

    @property
    def used(self) -> int:
        """Return currently occupied bytes.

        Shared transport is frame-based, so byte occupancy is not exposed here.
        """
        return 0

    def is_shared_transport(self) -> bool:
        """Return True when this ring buffer uses shared zerobuffer transport."""
        return True

    def write(self, data: bytes | bytearray | memoryview) -> None:
        """Write one complete packet as one zerobuffer frame."""
        view = data if isinstance(data, memoryview) else memoryview(data)
        if view.ndim != 1 or view.itemsize != 1 or view.format != "B":
            view = view.cast("B")
        length = len(view)
        if length == 0:
            return
        if length > self.size:
            raise ValueError("chunk exceeds ring buffer capacity")
        if self._shared_writer is None:
            raise RuntimeError("Shared ring buffer is read-only")
        try:
            frame_buffer = self._shared_writer.get_frame_buffer(length)
            frame_buffer[:] = view
            self._shared_writer.commit_frame()
        except Exception as exc:
            raise RuntimeError(
                "Failed to write shared ring packet "
                f"(length={length}, capacity={self.size}, shared_name={self._shared_name})"
            ) from exc

    def read_frame_packet(self) -> bytes | None:
        """Read exactly one committed zerobuffer frame as a packet."""
        if self._shared_reader is None:
            raise RuntimeError("Shared ring buffer is write-only")
        try:
            frame = self._shared_reader.read_frame(timeout=0.0)
        except _SharedWriterDeadException:
            return None
        except Exception as exc:
            logger.debug("Shared ring read unavailable: %s", exc)
            return None
        if frame is None:
            return None
        try:
            return bytes(frame.data)
        finally:
            frame.dispose()

    def close(self) -> None:
        """Close local handles to the underlying storage."""
        if self._shared_writer is not None:
            self._shared_writer.close()
            self._shared_writer = None
        if self._shared_reader is not None:
            self._shared_reader.close()
            self._shared_reader = None

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
