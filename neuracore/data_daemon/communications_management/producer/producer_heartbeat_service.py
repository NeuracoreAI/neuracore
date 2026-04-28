"""Heartbeat service for producer channels."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class ProducerHeartbeatService:
    """Own the producer heartbeat thread and lifecycle."""

    def __init__(
        self,
        *,
        interval_s: float,
        send_heartbeat: Callable[[], None],
    ) -> None:
        """Configure the heartbeat interval and callback."""
        self._interval_s = interval_s
        self._send_heartbeat = send_heartbeat
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the heartbeat loop if it is not already running."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="producer-channel-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 1.0) -> None:
        """Stop the heartbeat loop and wait briefly for shutdown."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
            self._thread = None

    def get_stats(self) -> dict[str, bool]:
        """Return a lightweight snapshot of heartbeat state."""
        thread = self._thread
        return {
            "heartbeat_thread_alive": (
                thread.is_alive() if thread is not None else False
            )
        }

    @property
    def stop_event(self) -> threading.Event:
        """Expose the stop event for compatibility and test control."""
        return self._stop_event

    def _heartbeat_loop(self) -> None:
        self._send_heartbeat()

        while not self._stop_event.wait(self._interval_s):
            try:
                self._send_heartbeat()
            except Exception as exc:
                logger.warning("Heartbeat failed: %s", exc)
