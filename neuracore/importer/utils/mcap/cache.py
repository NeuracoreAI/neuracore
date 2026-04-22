"""Disk cache format for preprocessed MCAP messages.

Preprocessing writes transformed events to disk so replay/logging can be done in a
separate pass. This avoids keeping large datasets in memory and enables TTL-safe
session rotation during logging.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np

from neuracore.importer.core.exceptions import ImportError

# cspell:ignore packb unpackb
try:
    import msgpack

    HAS_MSGPACK = True
except Exception:  # noqa: BLE001
    msgpack = None
    HAS_MSGPACK = False


_MODE = Literal["rb", "wb", "ab"]
_NDARRAY_SENTINEL = "__neuracore_ndarray__"


@dataclass(frozen=True, slots=True)
class CachedMessage:
    """One transformed event ready for replay logging."""

    data_type: str
    name: str
    timestamp: float
    log_time_ns: int
    transformed_data: Any
    source_topic: str = ""


def _require_msgpack() -> None:
    if HAS_MSGPACK and msgpack is not None:
        return
    raise ImportError(
        "MCAP cache serialization requires msgpack. "
        "Install with `pip install neuracore[import]` or `pip install msgpack`."
    )


def _pack_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            _NDARRAY_SENTINEL: True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tobytes(order="C"),
        }
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, tuple):
        return [_pack_value(item) for item in value]
    if isinstance(value, list):
        return [_pack_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _pack_value(item) for key, item in value.items()}
    return value


def _unpack_value(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get(_NDARRAY_SENTINEL):
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(dim) for dim in value["shape"])
            buffer = value["data"]
            return np.frombuffer(buffer, dtype=dtype).reshape(shape)
        return {str(key): _unpack_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack_value(item) for item in value]
    return value


class MessageCache:
    """Read/write a length-prefixed msgpack message cache file."""

    def __init__(self, cache_path: Path, mode: _MODE) -> None:
        """Initialize a cache file handle wrapper for the given mode."""
        _require_msgpack()
        self.cache_path = cache_path
        self.mode = mode
        self._handle: BinaryIO | None = None

    def __enter__(self) -> MessageCache:
        """Open the cache file and return this context manager instance."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.cache_path.open(self.mode)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close the cache file handle on context exit."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write_message(self, msg: CachedMessage) -> None:
        """Append one cached message to disk."""
        if self._handle is None or "r" in self.mode:
            raise RuntimeError("MessageCache is not open for writing.")
        if msgpack is None:
            raise RuntimeError("msgpack is unavailable")

        payload = {
            "data_type": msg.data_type,
            "name": msg.name,
            "timestamp": msg.timestamp,
            "log_time_ns": msg.log_time_ns,
            "source_topic": msg.source_topic,
            "transformed_data": _pack_value(msg.transformed_data),
        }
        blob = msgpack.packb(payload, use_bin_type=True)
        self._handle.write(struct.pack("<Q", len(blob)))
        self._handle.write(blob)

    def read_messages(self) -> Iterator[CachedMessage]:
        """Yield cached messages in write order."""
        if self._handle is None or "r" not in self.mode:
            raise RuntimeError("MessageCache is not open for reading.")
        if msgpack is None:
            raise RuntimeError("msgpack is unavailable")

        while True:
            size_buf = self._handle.read(8)
            if not size_buf:
                break
            if len(size_buf) != 8:
                raise ImportError("Corrupt cache record size header.")

            size = struct.unpack("<Q", size_buf)[0]
            blob = self._handle.read(size)
            if len(blob) != size:
                raise ImportError("Corrupt cache record payload.")

            try:
                payload = msgpack.unpackb(blob, raw=False)
            except Exception as exc:  # noqa: BLE001
                raise ImportError(
                    f"Corrupt cache record msgpack payload: {exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise ImportError("Corrupt cache record body.")

            yield CachedMessage(
                data_type=str(payload["data_type"]),
                name=str(payload["name"]),
                timestamp=float(payload["timestamp"]),
                log_time_ns=int(payload["log_time_ns"]),
                transformed_data=_unpack_value(payload.get("transformed_data")),
                source_topic=str(payload.get("source_topic", "")),
            )

    @classmethod
    def count_messages(cls, cache_path: Path) -> int:
        """Count cached messages without materializing all payloads."""
        count = 0
        with cls(cache_path, mode="rb") as cache:
            for _ in cache.read_messages():
                count += 1
        return count
