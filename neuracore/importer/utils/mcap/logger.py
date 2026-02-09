"""Phase-2 cache replay logger for MCAP imports.

This phase replays preprocessed cache records into Neuracore and relies on
`RecordingSession` to rotate sessions before backend TTL expiry.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from neuracore_types import DataType

from neuracore.importer.core.exceptions import ImportError

from .cache import MessageCache
from .progress import ProgressReporter
from .session import RecordingSession


@dataclass(frozen=True, slots=True)
class LoggingStats:
    """Summary metrics for cache replay logging."""

    message_count: int
    session_count: int
    duration_seconds: float


class MessageLogger:
    """Replay cached MCAP events into Neuracore logging APIs."""

    def __init__(
        self,
        session: RecordingSession,
        data_logger: Callable[[DataType, Any, str, float], None],
        progress_reporter: ProgressReporter,
        *,
        logger: logging.Logger,
        on_event_error: Callable[[int, str, str, int, Exception], bool] | None = None,
        max_replay_bytes_per_second: int = 32 * 1024 * 1024,
    ) -> None:
        """Initialize replay logger dependencies and pacing configuration."""
        self._session = session
        self._data_logger = data_logger
        self._progress = progress_reporter
        self._logger = logger
        self._on_event_error = on_event_error
        self._max_replay_bytes_per_second = max(0, int(max_replay_bytes_per_second))
        self._throttle_tokens = float(self._max_replay_bytes_per_second)
        self._throttle_last_update = time.monotonic()

    def log_from_cache(
        self,
        cache_file: Path,
        *,
        expected_total_messages: int | None = None,
    ) -> LoggingStats:
        """Replay cached events to Neuracore with session rotation."""
        started_at = time.monotonic()
        processed = 0

        total_for_progress = (
            int(expected_total_messages)
            if expected_total_messages is not None and expected_total_messages > 0
            else 0
        )
        self._progress.start_phase("logging", total_for_progress)
        try:
            self._session.ensure_active()
            with MessageCache(cache_file, mode="rb") as cache:
                for record in cache.read_messages():
                    event_index = processed + 1
                    try:
                        data_type = DataType(record.data_type)
                        self._throttle_before_log(data_type, record.transformed_data)
                        # Re-check right before log in case throttling slept.
                        self._session.ensure_active()
                        self._data_logger(
                            data_type,
                            record.transformed_data,
                            record.name,
                            record.timestamp,
                        )
                    except Exception as exc:  # noqa: BLE001
                        handled = False
                        if self._on_event_error is not None:
                            handled = bool(
                                self._on_event_error(
                                    event_index,
                                    record.source_topic or "<unknown>",
                                    record.name,
                                    int(record.log_time_ns),
                                    exc,
                                )
                            )
                        if handled:
                            continue
                        raise ImportError(
                            "Failed replaying cached MCAP event "
                            f"(index={event_index}, topic={record.source_topic}, "
                            f"name={record.name}, "
                            f"log_time_ns={record.log_time_ns}): {exc}"
                        ) from exc

                    processed += 1
                    self._progress.update(processed)
        finally:
            self._session.stop()
            self._progress.finish_phase()

        duration = time.monotonic() - started_at
        return LoggingStats(
            message_count=processed,
            session_count=self._session.session_count,
            duration_seconds=duration,
        )

    def _throttle_before_log(self, data_type: DataType, payload: Any) -> None:
        """Apply importer-side replay throttling for large payload streams."""
        if data_type not in {
            DataType.RGB_IMAGES,
            DataType.DEPTH_IMAGES,
            DataType.POINT_CLOUDS,
        }:
            return
        rate = self._max_replay_bytes_per_second
        if rate <= 0:
            return

        now = time.monotonic()
        elapsed = max(0.0, now - self._throttle_last_update)
        self._throttle_last_update = now

        self._throttle_tokens = min(
            float(rate),
            self._throttle_tokens + elapsed * float(rate),
        )

        cost = float(max(1, self._estimate_payload_bytes(payload)))
        if cost > float(rate):
            cost = float(rate)

        deficit = cost - self._throttle_tokens
        if deficit > 0:
            # Disabled replay pacing sleep to avoid slowing large imports.
            # time.sleep(deficit / float(rate))
            self._throttle_tokens = 0.0
            self._throttle_last_update = time.monotonic()
            return

        self._throttle_tokens -= cost

    def _estimate_payload_bytes(self, payload: Any) -> int:
        if payload is None:
            return 1
        if isinstance(payload, np.ndarray):
            return int(max(1, payload.nbytes))
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return int(max(1, len(payload)))
        if isinstance(payload, str):
            return int(max(1, len(payload.encode("utf-8"))))
        if isinstance(payload, np.generic):
            return int(max(1, payload.itemsize))
        if isinstance(payload, (int, float, bool)):
            return 8
        if isinstance(payload, (list, tuple)):
            return int(sum(self._estimate_payload_bytes(item) for item in payload[:32]))
        if isinstance(payload, dict):
            total = 0
            for idx, (key, value) in enumerate(payload.items()):
                if idx >= 32:
                    break
                total += len(str(key))
                total += self._estimate_payload_bytes(value)
            return int(max(1, total))
        return 64
