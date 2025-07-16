"""JSON data source for streaming structured data over WebRTC.

This module provides a rate-limited JSON data source that can publish
state updates to WebRTC data channels with automatic frequency throttling
to prevent overwhelming the network connection.
"""

import asyncio
import concurrent.futures
import json
import time
from asyncio import AbstractEventLoop
from typing import Optional

from pyee.asyncio import AsyncIOEventEmitter

from neuracore.core.streaming.event_loop_utils import get_running_loop
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager

MAXIMUM_EVENT_FREQUENCY_HZ = 10
MINIMUM_EVENT_DELTA = 1 / MAXIMUM_EVENT_FREQUENCY_HZ


class JSONSource(AsyncIOEventEmitter):
    """Event-driven JSON data source with rate limiting for WebRTC streaming.

    Provides a mechanism to publish JSON state updates with automatic
    throttling to prevent network congestion. Extends AsyncIOEventEmitter
    to support event-driven architecture.
    """

    STATE_UPDATED_EVENT = "STATE_UPDATED_EVENT"

    def __init__(
        self,
        mid: str,
        stream_enabled: EnabledManager,
        loop: Optional[AbstractEventLoop] = None,
    ):
        """Initialize the JSON source.

        Args:
            mid: Media identifier for this data source
            stream_enabled: Manager for controlling streaming state
            loop: Optional event loop. If not provided, uses current loop
        """
        self.loop = loop or get_running_loop()
        super().__init__(self.loop)
        self.mid = mid
        self._last_state: Optional[dict] = None
        self._last_update_time: float = 0
        self.submit_task: Optional[concurrent.futures.Future[None]] = None
        self.stream_enabled = stream_enabled

    def publish(self, state: dict) -> None:
        """Publish a state update to all listeners.

        Schedules the state update for emission with rate limiting.
        If streaming is disabled, the update is ignored.

        Args:
            state: Dictionary containing the state data to publish
        """
        if self.stream_enabled.is_disabled():
            return
        self._last_state = state
        if self.submit_task is None or self.submit_task.done():
            self.submit_task = asyncio.run_coroutine_threadsafe(
                self._submit_event(), self.loop
            )

    def get_last_state(self) -> Optional[str]:
        """Get the last published state as a JSON string.

        Returns:
            str: JSON-serialized last state, or None if no state exists
        """
        if not self._last_state:
            return None
        return json.dumps(self._last_state)

    async def _submit_event(self) -> None:
        """Submit an event with rate limiting.

        Internal method that handles the actual event emission with
        frequency throttling. Ensures updates don't exceed the maximum
        frequency to prevent network congestion.
        """
        remaining_time = self._last_update_time + MINIMUM_EVENT_DELTA - time.time()

        if remaining_time > 0:
            await asyncio.sleep(remaining_time)
        if self._last_state is None:
            return
        if self.stream_enabled.is_disabled():
            return

        message = json.dumps(self._last_state)
        self._last_update_time = float(time.time())
        self.emit(self.STATE_UPDATED_EVENT, message)
