import threading
from asyncio import AbstractEventLoop

from pyee.asyncio import AsyncIOEventEmitter


class EnabledManager(AsyncIOEventEmitter):

    DISABLED = "DISABLED"

    def __init__(self, initial_state: bool, loop: AbstractEventLoop | None = None):
        super().__init__(loop)
        self._is_enabled = initial_state
        self.lock = threading.Lock()

    def is_enabled(self) -> bool:
        """Check if streaming is enabled"""
        with self.lock:
            return self._is_enabled

    def disable(self) -> None:
        """Stop streaming"""
        with self.lock:
            if not self._is_enabled:
                return
            self._is_enabled = False
            self.emit(self.DISABLED)
            self.remove_all_listeners()
