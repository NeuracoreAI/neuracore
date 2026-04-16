"""Event logger for daemon performance analysis.

Enabled when the daemon is launched with --debug or NDD_DEBUG=true.
Output is written alongside the daemon database file.
"""

import csv
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from neuracore.data_daemon.event_emitter import Emitter

logger = logging.getLogger(__name__)

_ALL_EVENTS = [
    value for name, value in vars(Emitter).items() if not name.startswith("_")
]


class EventLogger:
    """Logs daemon events with timestamps to a CSV file for performance analysis."""

    def __init__(self, output_path: Path) -> None:
        """Initialize the event logger.

        Args:
            output_path: Path to write the CSV event log.
        """
        self._start_time = time.time()
        self._lock = threading.Lock()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(output_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp_abs", "timestamp", "event_name"])
        self._file.flush()
        logger.info("EventLogger writing to %s", output_path)

    def attach(self, emitter: Emitter) -> None:
        """Subscribe to all known emitter events.

        Args:
            emitter: The daemon event emitter to subscribe to.
        """
        for event_name in _ALL_EVENTS:
            emitter.on(event_name, self._make_handler(event_name))

    def _make_handler(self, event_name: str) -> Callable[..., None]:
        def handler(*args: object, **kwargs: object) -> None:
            self._log(event_name)

        return handler

    def _log(self, event_name: str) -> None:
        now = time.time()
        elapsed = now - self._start_time
        with self._lock:
            try:
                self._writer.writerow([f"{now:.6f}", f"{elapsed:.6f}", event_name])
                self._file.flush()
            except Exception:
                logger.exception("Failed to write event log row for %s", event_name)

    def close(self) -> None:
        """Flush and close the CSV file."""
        with self._lock:
            try:
                self._file.close()
            except Exception:
                logger.exception("Failed to close event log file")
