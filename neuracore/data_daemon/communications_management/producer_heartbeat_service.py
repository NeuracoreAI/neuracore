"""Heartbeat service for producer channels."""

from __future__ import annotations

import atexit
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
    ) -> None:
        """Configure the heartbeat interval and callback."""
        self._interval_s = interval_s
        self._heartbeat_listeners = set[Callable[[], None]]()
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

    def register_heartbeat_listener(self, listener: Callable[[], None]) -> None:
        """Register a listener for heartbeat events.

        Args:
            listener: The listener to register.
        """
        self._heartbeat_listeners.add(listener)

    def unregister_heartbeat_listener(self, listener: Callable[[], None]) -> None:
        """Unregister a listener for heartbeat events.

        Args:
            listener: The listener to unregister.
        """
        self._heartbeat_listeners.discard(listener)

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
        while not self._stop_event.is_set():
            for listener in self._heartbeat_listeners:
                try:
                    listener()
                except Exception as exc:
                    logger.warning("Heartbeat failed: %s", exc)
            if self._stop_event.wait(self._interval_s):
                break


_producer_heartbeat_service: ProducerHeartbeatService | None = None


def get_producer_heartbeat_service() -> ProducerHeartbeatService:
    """Return the producer heartbeat service singleton.

    Returns:
        The producer heartbeat service singleton.
    """
    global _producer_heartbeat_service
    if _producer_heartbeat_service is None:
        # 10 seconds maximum
        _producer_heartbeat_service = ProducerHeartbeatService(interval_s=5.0)
        _producer_heartbeat_service.start()
        atexit.register(_producer_heartbeat_service.stop)
    return _producer_heartbeat_service
