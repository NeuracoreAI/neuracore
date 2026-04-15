"""Single-producer/single-consumer ring buffer used by the data daemon."""

from __future__ import annotations

import mmap
import os
import struct
import threading
import time

_READ_INDEX_OFFSET = 0
_WRITE_INDEX_OFFSET = 8
_SHARED_INDEX_FORMAT = "!Q"
_SHARED_HEADER_SIZE = 16
_FULL_WAIT_SLEEP_S = 0.0005


class RingBuffer:
    """Circular byte buffer with optional shared-memory backing.

    Local buffers use a bytearray plus a condition variable. Shared buffers use a
    memfd-backed mmap and two monotonic counters:

    - read_index: bytes consumed by the single consumer
    - write_index: bytes committed by the single producer

    This avoids cross-process lost updates because the producer only mutates the
    write counter and the consumer only mutates the read counter.
    """

    def __init__(
        self,
        size: int = 1024,
        *,
        _mapping: mmap.mmap | None = None,
        _fd: int | None = None,
        _shared_name: str | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("RingBuffer size must be > 0")

        self.size = int(size)
        self.buffer = bytearray(self.size) if _mapping is None else None
        self._mapping = _mapping
        self._fd = _fd
        self._shared_name = _shared_name
        self._read_index = 0
        self._write_index = 0
        self._lock = threading.RLock()
        self._not_full = threading.Condition(self._lock)

    @classmethod
    def create_shared(cls, size: int) -> "RingBuffer":
        """Create a RAM-backed shared ring buffer using memfd."""
        fd = os.memfd_create(
            "neuracore-ring-buffer",
            flags=getattr(os, "MFD_CLOEXEC", 0),
        )
        total_size = _SHARED_HEADER_SIZE + int(size)
        os.ftruncate(fd, total_size)
        mapping = mmap.mmap(fd, total_size, access=mmap.ACCESS_WRITE)
        struct.pack_into(_SHARED_INDEX_FORMAT, mapping, _READ_INDEX_OFFSET, 0)
        struct.pack_into(_SHARED_INDEX_FORMAT, mapping, _WRITE_INDEX_OFFSET, 0)
        shared_name = f"/proc/{os.getpid()}/fd/{fd}"
        return cls(
            size=int(size),
            _mapping=mapping,
            _fd=fd,
            _shared_name=shared_name,
        )

    @classmethod
    def open_shared(cls, name: str, size: int) -> "RingBuffer":
        """Open an existing shared ring buffer from its procfs fd path."""
        fd = os.open(name, os.O_RDWR)
        total_size = _SHARED_HEADER_SIZE + int(size)
        mapping = mmap.mmap(fd, total_size, access=mmap.ACCESS_WRITE)
        return cls(
            size=int(size),
            _mapping=mapping,
            _fd=fd,
            _shared_name=name,
        )

    @property
    def shared_name(self) -> str | None:
        """Return the shared-memory attachment string, if any."""
        return self._shared_name

    @property
    def read_pos(self) -> int:
        """Current logical read position within the ring."""
        if self._mapping is None:
            with self._lock:
                return self._read_index % self.size
        return self._shared_read_index() % self.size

    @property
    def write_pos(self) -> int:
        """Current logical write position within the ring."""
        if self._mapping is None:
            with self._lock:
                return self._write_index % self.size
        return self._shared_write_index() % self.size

    @property
    def used(self) -> int:
        """Return currently occupied bytes."""
        return self.available()

    def available(self) -> int:
        """Return the number of bytes currently available to read."""
        if self._mapping is None:
            with self._lock:
                return self._write_index - self._read_index
        write_index = self._shared_write_index()
        read_index = self._shared_read_index()
        return max(0, write_index - read_index)

    def write(self, data: bytes | bytearray | memoryview) -> None:
        """Write bytes into the ring buffer, blocking until space is available."""
        view = data if isinstance(data, memoryview) else memoryview(data)
        if view.ndim != 1 or view.itemsize != 1 or view.format != "B":
            view = view.cast("B")
        length = len(view)
        if length == 0:
            return
        if length > self.size:
            raise ValueError("chunk exceeds ring buffer capacity")

        if self._mapping is None:
            with self._not_full:
                while self.size - (self._write_index - self._read_index) < length:
                    self._not_full.wait(timeout=0.1)
                self._write_into_storage(self._write_index % self.size, view)
                self._write_index += length
            return

        while True:
            read_index = self._shared_read_index()
            write_index = self._shared_write_index()
            if self.size - (write_index - read_index) >= length:
                self._write_into_storage(write_index % self.size, view)
                self._shared_store_write_index(write_index + length)
                return
            time.sleep(_FULL_WAIT_SLEEP_S)

    def read(self, size: int) -> bytes | None:
        """Read and consume exactly `size` bytes, or return None if unavailable."""
        if size < 0:
            raise ValueError("size must be >= 0")
        if size == 0:
            return b""

        if self._mapping is None:
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

        read_index = self._shared_read_index()
        write_index = self._shared_write_index()
        if write_index - read_index < size:
            return None
        result = self._read_from_storage(read_index % self.size, size)
        self._shared_store_read_index(read_index + size)
        return result

    def peek(self, size: int) -> bytes | None:
        """Read exactly `size` bytes without consuming them."""
        if size < 0:
            raise ValueError("size must be >= 0")
        if size == 0:
            return b""

        if self._mapping is None:
            with self._lock:
                available = self._write_index - self._read_index
                if available < size:
                    return None
                return self._read_from_storage(self._read_index % self.size, size)

        read_index = self._shared_read_index()
        write_index = self._shared_write_index()
        if write_index - read_index < size:
            return None
        return self._read_from_storage(read_index % self.size, size)

    def close(self) -> None:
        """Close local handles to the underlying storage."""
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def unlink(self) -> None:
        """Release underlying storage when possible.

        Memfd-backed shared buffers are reference-counted by open file descriptors,
        so there is nothing explicit to unlink here.
        """

    def _shared_read_index(self) -> int:
        mapping = self._require_mapping()
        return struct.unpack_from(
            _SHARED_INDEX_FORMAT,
            mapping,
            _READ_INDEX_OFFSET,
        )[0]

    def _shared_write_index(self) -> int:
        mapping = self._require_mapping()
        return struct.unpack_from(
            _SHARED_INDEX_FORMAT,
            mapping,
            _WRITE_INDEX_OFFSET,
        )[0]

    def _shared_store_read_index(self, value: int) -> None:
        mapping = self._require_mapping()
        struct.pack_into(_SHARED_INDEX_FORMAT, mapping, _READ_INDEX_OFFSET, value)

    def _shared_store_write_index(self, value: int) -> None:
        mapping = self._require_mapping()
        struct.pack_into(_SHARED_INDEX_FORMAT, mapping, _WRITE_INDEX_OFFSET, value)

    def _require_mapping(self) -> mmap.mmap:
        if self._mapping is None:
            raise RuntimeError("Shared ring buffer is closed")
        return self._mapping

    def _data_offset(self) -> int:
        return 0 if self._mapping is None else _SHARED_HEADER_SIZE

    def _read_from_storage(self, start: int, size: int) -> bytes:
        storage = self._storage()
        base = self._data_offset()
        first = min(size, self.size - start)
        first_slice = bytes(storage[base + start : base + start + first])
        if first == size:
            return first_slice
        second = size - first
        return first_slice + bytes(storage[base : base + second])

    def _write_into_storage(self, start: int, data: memoryview) -> None:
        storage = self._storage()
        base = self._data_offset()
        length = len(data)
        first = min(length, self.size - start)
        storage[base + start : base + start + first] = data[:first]
        if first < length:
            storage[base : base + (length - first)] = data[first:]

    def _storage(self) -> bytearray | mmap.mmap:
        mapping = self._mapping
        if mapping is not None:
            return mapping
        return self.buffer
