"""Disk-backed bridge spool for complete messages awaiting RDM enqueue."""

from __future__ import annotations

import json
import logging
import struct
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree

from neuracore_types import DataType

from neuracore.data_daemon.models import CompleteMessage

_HEADER_FORMAT = "<II"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)
_DEFAULT_SEGMENT_MAX_BYTES = 64 * 1024 * 1024

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SpoolEntry:
    """Location metadata for a spooled message record."""

    segment_id: int
    offset: int
    length: int
    recording_id: str


class BridgeSpool:
    """Minimal disk-backed FIFO spool for complete messages."""

    def __init__(
        self,
        root: Path,
        *,
        segment_max_bytes: int = _DEFAULT_SEGMENT_MAX_BYTES,
    ) -> None:
        self._root = Path(root)
        self._segment_max_bytes = max(1, int(segment_max_bytes))
        self._entries: deque[_SpoolEntry] = deque()
        self._queued_messages = 0
        self._queued_bytes = 0
        self._pending_by_recording: dict[str, int] = {}
        self._pending_by_segment: dict[int, int] = {}
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

        self._reset_root()
        self._current_segment_id = 0
        self._current_segment_size = 0
        self._current_segment_path = self._segment_path(self._current_segment_id)
        self._current_segment_path.touch()

    def append(self, message: CompleteMessage) -> int:
        """Persist a message to disk and enqueue it for forwarding."""
        record = self._serialize(message)
        record_len = len(record)

        with self._not_empty:
            self._root.mkdir(parents=True, exist_ok=True)
            self._current_segment_path.parent.mkdir(parents=True, exist_ok=True)
            self._current_segment_path.touch(exist_ok=True)
            if (
                self._current_segment_size > 0
                and self._current_segment_size + record_len > self._segment_max_bytes
            ):
                self._rotate_segment()

            offset = self._current_segment_size
            with self._current_segment_path.open("ab") as handle:
                handle.write(record)

            self._current_segment_size += record_len
            self._entries.append(
                _SpoolEntry(
                    segment_id=self._current_segment_id,
                    offset=offset,
                    length=record_len,
                    recording_id=message.recording_id,
                )
            )
            self._queued_messages += 1
            self._queued_bytes += record_len
            self._pending_by_recording[message.recording_id] = (
                self._pending_by_recording.get(message.recording_id, 0) + 1
            )
            self._pending_by_segment[self._current_segment_id] = (
                self._pending_by_segment.get(self._current_segment_id, 0) + 1
            )
            self._not_empty.notify()
            return record_len

    def wait_for_item(self, timeout_s: float | None = None) -> bool:
        """Block until at least one item is queued or timeout expires."""
        with self._not_empty:
            if self._entries:
                return True
            self._not_empty.wait(timeout=timeout_s)
            return bool(self._entries)

    def peek(self) -> CompleteMessage | None:
        """Return the oldest queued message without acknowledging it."""
        with self._not_empty:
            while self._entries:
                entry = self._entries[0]
                try:
                    return self._read_entry(entry)
                except FileNotFoundError:
                    logger.warning(
                        "Bridge spool segment missing for segment_id=%s; dropping stale head entry",
                        entry.segment_id,
                    )
                    self._drop_entry_locked(self._entries.popleft())
            return None

    def ack(self) -> None:
        """Acknowledge the oldest queued message and advance the head."""
        with self._not_empty:
            if not self._entries:
                return

            self._drop_entry_locked(self._entries.popleft())

    def pending_count_for_recording(self, recording_id: str) -> int:
        """Return the number of queued records for a recording."""
        with self._lock:
            return int(self._pending_by_recording.get(str(recording_id), 0))

    def snapshot(self) -> dict[str, object]:
        """Return a lightweight snapshot of spool state."""
        with self._lock:
            return {
                "queued_messages": self._queued_messages,
                "queued_bytes": self._queued_bytes,
                "pending_by_recording": dict(self._pending_by_recording),
                "current_segment_id": self._current_segment_id,
            }

    def cleanup(self) -> None:
        """Remove all spool files for the current daemon session."""
        with self._not_empty:
            self._entries.clear()
            self._queued_messages = 0
            self._queued_bytes = 0
            self._pending_by_recording.clear()
            self._pending_by_segment.clear()
        if self._root.exists():
            rmtree(self._root, ignore_errors=True)

    def _drop_entry_locked(self, entry: _SpoolEntry) -> None:
        """Remove one queued entry while holding the spool lock."""
        self._queued_messages = max(0, self._queued_messages - 1)
        self._queued_bytes = max(0, self._queued_bytes - entry.length)

        recording_count = self._pending_by_recording.get(entry.recording_id, 0) - 1
        if recording_count > 0:
            self._pending_by_recording[entry.recording_id] = recording_count
        else:
            self._pending_by_recording.pop(entry.recording_id, None)

        segment_count = self._pending_by_segment.get(entry.segment_id, 0) - 1
        if segment_count > 0:
            self._pending_by_segment[entry.segment_id] = segment_count
            return

        self._pending_by_segment.pop(entry.segment_id, None)
        if entry.segment_id != self._current_segment_id:
            self._segment_path(entry.segment_id).unlink(missing_ok=True)

    def _segment_path(self, segment_id: int) -> Path:
        return self._root / f"segment-{segment_id:06d}.spool"

    def _rotate_segment(self) -> None:
        previous_segment_id = self._current_segment_id
        self._current_segment_id += 1
        self._current_segment_size = 0
        self._current_segment_path = self._segment_path(self._current_segment_id)
        self._current_segment_path.parent.mkdir(parents=True, exist_ok=True)
        self._current_segment_path.touch(exist_ok=True)
        if previous_segment_id not in self._pending_by_segment:
            self._segment_path(previous_segment_id).unlink(missing_ok=True)

    def _reset_root(self) -> None:
        if self._root.exists():
            rmtree(self._root, ignore_errors=True)
        self._root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _serialize(message: CompleteMessage) -> bytes:
        metadata = {
            "producer_id": message.producer_id,
            "trace_id": message.trace_id,
            "recording_id": message.recording_id,
            "dataset_id": message.dataset_id,
            "dataset_name": message.dataset_name,
            "robot_name": message.robot_name,
            "robot_id": message.robot_id,
            "data_type": message.data_type.value,
            "data_type_name": message.data_type_name,
            "robot_instance": message.robot_instance,
            "received_at": message.received_at,
            "final_chunk": message.final_chunk,
        }
        metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        return (
            struct.pack(_HEADER_FORMAT, len(metadata_bytes), len(message.data))
            + metadata_bytes
            + message.data
        )

    def _read_entry(self, entry: _SpoolEntry) -> CompleteMessage:
        with self._segment_path(entry.segment_id).open("rb") as handle:
            handle.seek(entry.offset)
            header = handle.read(_HEADER_SIZE)
            if len(header) != _HEADER_SIZE:
                raise RuntimeError("Failed to read bridge spool record header")
            metadata_len, data_len = struct.unpack(_HEADER_FORMAT, header)
            metadata_bytes = handle.read(metadata_len)
            if len(metadata_bytes) != metadata_len:
                raise RuntimeError("Failed to read bridge spool record metadata")
            data = handle.read(data_len)
            if len(data) != data_len:
                raise RuntimeError("Failed to read bridge spool record payload")

        metadata = json.loads(metadata_bytes.decode("utf-8"))
        return CompleteMessage(
            producer_id=str(metadata["producer_id"]),
            trace_id=str(metadata["trace_id"]),
            recording_id=str(metadata["recording_id"]),
            dataset_id=metadata.get("dataset_id"),
            dataset_name=metadata.get("dataset_name"),
            robot_name=metadata.get("robot_name"),
            robot_id=metadata.get("robot_id"),
            data_type=DataType(str(metadata["data_type"])),
            data_type_name=str(metadata.get("data_type_name") or ""),
            robot_instance=int(metadata.get("robot_instance") or 0),
            received_at=str(metadata["received_at"]),
            data=data,
            final_chunk=bool(metadata.get("final_chunk")),
        )
