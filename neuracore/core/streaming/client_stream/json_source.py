import asyncio
import json
import time
from asyncio import AbstractEventLoop
from typing import Optional

from pyee.asyncio import AsyncIOEventEmitter

from neuracore.core.streaming.client_stream.stream_enabled import EnabledManager

MAXIMUM_EVENT_FREQUENCY_HZ = 10
MINIMUM_EVENT_DELTA = 1 / MAXIMUM_EVENT_FREQUENCY_HZ


class JSONSource(AsyncIOEventEmitter):
    STATE_UPDATED_EVENT = "STATE_UPDATED_EVENT"

    def __init__(
        self, mid: str, stream_enabled: EnabledManager, loop: AbstractEventLoop = None
    ):
        super().__init__(loop)
        self.mid = mid
        self._last_state: Optional[dict] = None
        self._last_update_time = 0
        self.submit_task = None
        self.stream_enabled = stream_enabled

    def publish(self, state: dict):
        """Publish an update to all listeners"""
        if not self.stream_enabled.is_enabled():
            return
        self._last_state = state
        if self.submit_task is None or self.submit_task.done():
            self.submit_task = asyncio.run_coroutine_threadsafe(
                self._submit_event(), self._loop
            )

    def get_last_state(self) -> str | None:
        if not self._last_state:
            return None
        return json.dumps(self._last_state)

    async def _submit_event(self):
        """Submit an event to the server"""

        remaining_time = self._last_update_time + MINIMUM_EVENT_DELTA - time.time()

        if remaining_time > 0:
            await asyncio.sleep(remaining_time)
        if self._last_state is None:
            return
        if not self.stream_enabled.is_enabled():
            return

        message = json.dumps(self._last_state)
        self._last_update_time = time.time()
        self.emit(self.STATE_UPDATED_EVENT, message)
