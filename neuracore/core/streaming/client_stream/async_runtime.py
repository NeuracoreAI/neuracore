import asyncio
import threading
from concurrent.futures import Future
from typing import Optional


class AsyncRuntime:
    _instance: Optional["AsyncRuntime"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AsyncRuntime, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if not self._initialized:
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            self._initialized = True

    def _run_loop(self):
        """Run the event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_coroutine(self, coro, timeout=None):
        """Run coroutine safely from sync code."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=timeout)

    def submit_coroutine(self, coro) -> Future:
        """Fire-and-forget coroutine from sync code."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def shutdown(self):
        """Stop the event loop and join the thread."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()

    @classmethod
    def get_instance(cls) -> "AsyncRuntime":
        """Get the singleton instance of AsyncRuntime."""
        if cls._instance is None:
            return cls()
        return cls._instance
