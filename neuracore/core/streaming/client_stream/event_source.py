from asyncio import AbstractEventLoop
import asyncio
import json
from typing import Optional
from pyee.asyncio import AsyncIOEventEmitter


MAXIMUM_EVENT_FREQUENCY_HZ = 30
MINIMUM_EVENT_DELTA = 1 / MAXIMUM_EVENT_FREQUENCY_HZ


class EventSource(AsyncIOEventEmitter):
    def __init__(self, mid: str, loop: AbstractEventLoop = None):
        super().__init__(loop)
        self.mid = mid
        self._last_event: Optional[dict] = None
        self.submit_task = None

    def publish(self, event: dict):
        """Publish an event to all listeners"""
        self._last_event = event
        
        if self.submit_task is None or self.submit_task.done():
            self.submit_task = asyncio.run_coroutine_threadsafe(
                self._submit_event(), self._loop
            )

    async def _submit_event(self):
        """Submit an event to the server"""
        await asyncio.sleep(MINIMUM_EVENT_DELTA)
        if self._last_event is None:
            return

        message = json.dumps(self._last_event)
        await self.emit("event", message)
