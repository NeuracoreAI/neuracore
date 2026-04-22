"""Recording session manager for MCAP replay logging.

Neuracore backend recording sessions expire after roughly 5 minutes (300s).
This manager rotates sessions using Neuracore's shared recording timing
constants so importer behavior stays aligned with core recording expiry rules.
By default it rotates slightly before Neuracore's warning threshold to reduce
warning log noise and keep safety margin before the hard expiry.
"""

from __future__ import annotations

import logging
import time

import neuracore as nc
from neuracore.importer.core.exceptions import ImportError

from .config import (
    DEFAULT_BACKEND_RECORDING_TTL_SECONDS,
    DEFAULT_SESSION_ROTATION_SECONDS,
)


class RecordingSession:
    """Manage active recording lifecycle with TTL-aware rotation."""

    BACKEND_TTL_SECONDS = DEFAULT_BACKEND_RECORDING_TTL_SECONDS
    ROTATION_INTERVAL = DEFAULT_SESSION_ROTATION_SECONDS

    def __init__(
        self,
        dataset_name: str,
        *,
        logger: logging.Logger,
        rotation_interval_seconds: int = ROTATION_INTERVAL,
    ) -> None:
        """Create a recording-session controller for the target dataset."""
        self._dataset_name = dataset_name
        self._logger = logger
        self._rotation_interval_seconds = max(1, rotation_interval_seconds)
        self._session_start_time = 0.0
        self._active = False
        self._session_count = 0
        self._active_dataset_id: str | None = None

    @property
    def session_count(self) -> int:
        """Return the number of recording sessions started."""
        return self._session_count

    def ensure_active(self) -> None:
        """Start or rotate the recording session as needed."""
        if not self._active or not nc.is_recording():
            self._start_session()
            return

        elapsed = time.monotonic() - self._session_start_time
        if elapsed >= self._rotation_interval_seconds:
            self._rotate_session()

    def stop(self) -> None:
        """Stop the active recording session if one exists."""
        if nc.is_recording():
            nc.stop_recording(wait=True)
        self._active = False

    def _start_session(self) -> None:
        """Start a new recording session with retries."""
        self._refresh_dataset_context()

        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                if nc.is_recording():
                    nc.stop_recording(wait=True)
                nc.start_recording()
                self._session_start_time = time.monotonic()
                self._active = True
                self._session_count += 1
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < 3:
                    time.sleep(0.5)

        raise ImportError(f"Failed to start recording session: {last_error}")

    def _rotate_session(self) -> None:
        """Rotate the active recording session."""
        self.stop()
        self._start_session()

    def _refresh_dataset_context(self) -> None:
        """Resolve and cache the output dataset context before recording."""
        if self._active_dataset_id:
            try:
                dataset = nc.get_dataset(id=self._active_dataset_id)
                self._active_dataset_id = dataset.id
                return
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Cached dataset id '%s' unavailable (%s); resolving by name '%s'.",
                    self._active_dataset_id,
                    exc,
                    self._dataset_name,
                )

        last_error: Exception | None = None
        for attempt in range(1, 9):
            try:
                dataset = nc.get_dataset(self._dataset_name)
                self._active_dataset_id = dataset.id
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < 8:
                    time.sleep(0.25)

        raise ImportError(
            "Unable to resolve active dataset context for recording "
            f"(name='{self._dataset_name}', id='{self._active_dataset_id}'): "
            f"{last_error}"
        )
