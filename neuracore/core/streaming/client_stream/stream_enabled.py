import threading
from asyncio import AbstractEventLoop

from pyee.asyncio import AsyncIOEventEmitter

from neuracore.core.const import LIVE_DATA_ENABLED


class StreamEnabled(AsyncIOEventEmitter):

    STREAMING_STOPPED = "STREAMING_STOPPED"

    def __init__(self, loop: AbstractEventLoop | None = None):
        super().__init__(loop)
        self.streaming = LIVE_DATA_ENABLED
        self.lock = threading.Lock()

    def is_streaming(self) -> bool:
        """Check if streaming is enabled"""
        with self.lock:
            return self.streaming

    def stop_streaming(self) -> None:
        """Stop streaming"""
        with self.lock:
            if not self.streaming:
                return
            self.streaming = False
            self.emit(self.STREAMING_STOPPED)
            self.remove_all_listeners()
