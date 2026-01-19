"""Utilities for disk usage tracking and storage policy enforcement."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


def scan_used_bytes(root_path: Path) -> int:
    """Scan total bytes used under a directory.

    Args:
        root_path: Directory to scan.

    Returns:
        Total bytes used by files under root_path.
    """
    if not root_path.exists():
        return 0

    total_bytes = 0
    for file_path in root_path.rglob("*"):
        try:
            if file_path.is_file():
                total_bytes += file_path.stat().st_size
        except FileNotFoundError:
            continue
    return total_bytes


def scan_dir_bytes(directory_path: Path) -> int:
    """Scan total bytes used under a directory (including nested files).

    Args:
        directory_path: Directory to scan.

    Returns:
        Total bytes used by files under directory_path.
    """
    return scan_used_bytes(directory_path)


def get_free_bytes(path: Path) -> int:
    """Get free bytes for the filesystem that contains the given path.

    Args:
        path: Path on the target filesystem.

    Returns:
        Free bytes available on the filesystem.
    """
    try:
        return shutil.disk_usage(path).free
    except FileNotFoundError:
        path.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(path).free


@dataclass(frozen=True)
class StoragePolicy:
    """Storage policy for local writes.

    Args:
        storage_limit_bytes: Maximum bytes the daemon may use under the recordings root.
        min_free_disk_bytes: Minimum free bytes to keep available on the filesystem.
        refresh_seconds: How frequently to reconcile the used-bytes estimate via a scan.
    """

    storage_limit_bytes: int | None
    min_free_disk_bytes: int
    refresh_seconds: float


class StorageBudget:
    """Thread-safe storage budget tracker with periodic reconciliation.

    This class tracks an estimated "used bytes" value and supports:
      - Reserving budget before writes (fast path).
      - Releasing budget on deletes/aborts.
      - Periodic scan reconciliation to correct drift.

    Args:
        recordings_root: Root directory whose usage is tracked.
        policy: Storage policy configuration.
    """

    def __init__(self, recordings_root: Path, policy: StoragePolicy) -> None:
        """Initialise a storage budget tracker.

        Args:
            recordings_root: Root directory used to store recordings.
            policy: Storage policy configuration for limits and refresh behaviour.
        """
        self._recordings_root = recordings_root
        self._policy = policy
        self._lock = Lock()
        self._used_bytes = scan_used_bytes(recordings_root)
        self._last_refresh = time.monotonic()

    def refresh_if_stale(self) -> None:
        """Refresh estimated used bytes via a directory scan if stale.

        Returns:
            None
        """
        refresh_seconds = self._policy.refresh_seconds
        if refresh_seconds <= 0:
            return

        now = time.monotonic()
        with self._lock:
            if now - self._last_refresh < refresh_seconds:
                return
            self._used_bytes = scan_used_bytes(self._recordings_root)
            self._last_refresh = now

    def has_free_disk_for_write(self, bytes_to_write: int) -> bool:
        """Check filesystem free space safety margin.

        Args:
            bytes_to_write: Bytes about to be written.

        Returns:
            True if filesystem has enough free bytes after the write plus safety margin.
        """
        free_bytes = get_free_bytes(self._recordings_root)
        return free_bytes >= (bytes_to_write + self._policy.min_free_disk_bytes)

    def reserve(self, bytes_to_write: int) -> bool:
        """Reserve storage budget for an upcoming write.

        Args:
            bytes_to_write: Bytes to reserve.

        Returns:
            True if budget was reserved, False if it would exceed the limit.
        """
        storage_limit_bytes = self._policy.storage_limit_bytes
        if storage_limit_bytes is None:
            return True

        self.refresh_if_stale()

        with self._lock:
            if self._used_bytes + bytes_to_write > storage_limit_bytes:
                return False
            self._used_bytes += bytes_to_write
            return True

    def release(self, bytes_to_release: int) -> None:
        """Release storage budget after deleting data.

        Args:
            bytes_to_release: Bytes to release.

        Returns:
            None
        """
        storage_limit_bytes = self._policy.storage_limit_bytes
        if storage_limit_bytes is None:
            return

        with self._lock:
            self._used_bytes = max(0, self._used_bytes - bytes_to_release)

    def is_over_limit(self) -> bool:
        """Check whether the current estimated usage is over the configured limit.

        Returns:
            True if over the limit, otherwise False.
        """
        storage_limit_bytes = self._policy.storage_limit_bytes
        if storage_limit_bytes is None:
            return False

        self.refresh_if_stale()
        with self._lock:
            return self._used_bytes > storage_limit_bytes
