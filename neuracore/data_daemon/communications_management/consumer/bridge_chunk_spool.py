"""Disk-backed chunk spool for shared-slot completion work."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree

_DEFAULT_SEGMENT_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ChunkSpoolRef:
    """Location metadata for one spooled chunk payload."""

    segment_id: int
    offset: int
    length: int


class BridgeChunkSpool:
    """Persist large chunk payloads outside the Python heap."""

    def __init__(
        self,
        root: Path,
        *,
        segment_max_bytes: int = _DEFAULT_SEGMENT_MAX_BYTES,
    ) -> None:
        """Initialize the spool root and first writable segment."""
        self._root = Path(root)
        self._segment_max_bytes = max(1, int(segment_max_bytes))
        self._lock = threading.Lock()
        self._segment_refcounts: dict[int, int] = {}
        self._current_segment_id = 0
        self._current_segment_size = 0
        self._reset_root()
        self._current_segment_path = self._segment_path(self._current_segment_id)

    def append(self, chunk: bytes | bytearray | memoryview) -> ChunkSpoolRef:
        """Copy one chunk from shared memory into the spool and return its ref."""
        chunk_view = chunk if isinstance(chunk, memoryview) else memoryview(chunk)
        chunk_len = len(chunk_view)

        with self._lock:
            if (
                self._current_segment_size > 0
                and self._current_segment_size + chunk_len > self._segment_max_bytes
            ):
                self._rotate_segment_locked()

            offset = self._current_segment_size
            self._current_segment_path.parent.mkdir(parents=True, exist_ok=True)
            with self._current_segment_path.open("ab") as handle:
                handle.write(chunk_view)

            self._current_segment_size += chunk_len
            self._segment_refcounts[self._current_segment_id] = (
                self._segment_refcounts.get(self._current_segment_id, 0) + 1
            )

            return ChunkSpoolRef(
                segment_id=self._current_segment_id,
                offset=offset,
                length=chunk_len,
            )

    def materialize(self, refs: list[ChunkSpoolRef]) -> bytes:
        """Read a completed trace payload back into one contiguous bytes object."""
        total_bytes = sum(ref.length for ref in refs)
        payload = bytearray(total_bytes)
        cursor = 0

        for ref in refs:
            next_cursor = cursor + ref.length
            self.read_into(ref, memoryview(payload)[cursor:next_cursor])
            cursor = next_cursor

        return bytes(payload)

    def read_into(self, ref: ChunkSpoolRef, target: memoryview) -> None:
        """Read one spooled chunk into a caller-provided buffer."""
        if len(target) != ref.length:
            raise ValueError(
                f"Target length {len(target)} does not match ref length {ref.length}"
            )

        with self._segment_path(ref.segment_id).open("rb") as handle:
            handle.seek(ref.offset)
            bytes_read = handle.readinto(target)
        if bytes_read != ref.length:
            raise RuntimeError(
                "Failed to read the expected number of bytes from the chunk spool"
            )

    def release(self, ref: ChunkSpoolRef) -> None:
        """Release one chunk reference and delete fully drained old segments."""
        with self._lock:
            remaining = self._segment_refcounts.get(ref.segment_id, 0) - 1
            if remaining > 0:
                self._segment_refcounts[ref.segment_id] = remaining
                return

            self._segment_refcounts.pop(ref.segment_id, None)
            if ref.segment_id != self._current_segment_id:
                self._segment_path(ref.segment_id).unlink(missing_ok=True)

    def cleanup(self) -> None:
        """Remove all spool files for the current daemon session."""
        with self._lock:
            self._segment_refcounts.clear()
        if self._root.exists():
            rmtree(self._root, ignore_errors=True)

    def _segment_path(self, segment_id: int) -> Path:
        return self._root / f"segment-{segment_id:06d}.chunk"

    def _rotate_segment_locked(self) -> None:
        previous_segment_id = self._current_segment_id
        self._current_segment_id += 1
        self._current_segment_size = 0
        self._current_segment_path = self._segment_path(self._current_segment_id)
        self._current_segment_path.parent.mkdir(parents=True, exist_ok=True)
        self._current_segment_path.touch(exist_ok=True)
        if previous_segment_id not in self._segment_refcounts:
            self._segment_path(previous_segment_id).unlink(missing_ok=True)

    def _reset_root(self) -> None:
        if self._root.exists():
            rmtree(self._root, ignore_errors=True)
        self._root.mkdir(parents=True, exist_ok=True)
