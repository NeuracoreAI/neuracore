"""Event loop utilities for asyncio operations.

This module provides utilities for managing asyncio event loops across
different execution contexts, ensuring that async operations can be
performed even when no event loop is currently running.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

_neuracore_async_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def get_running_loop() -> asyncio.AbstractEventLoop:
    """Gets the neuracore running event loop.

    Returns:
        asyncio.AbstractEventLoop: The running event loop

    Note:
        The created event loop runs in a daemon thread, which means it will
        be automatically terminated when the main program exits.
    """
    global _neuracore_async_loop
    if _neuracore_async_loop is not None:
        return _neuracore_async_loop

    with _loop_lock:
        if _neuracore_async_loop is not None:
            return _neuracore_async_loop
        _neuracore_async_loop = asyncio.new_event_loop()
        # Limit the number of threads to a reasonable number
        executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="nc-async-executor"
        )
        _neuracore_async_loop.set_default_executor(executor)
        threading.Thread(
            target=lambda: _neuracore_async_loop.run_forever(),
            name="nc-async-loop",
            daemon=True,
        ).start()
        asyncio.set_event_loop(_neuracore_async_loop)
    return _neuracore_async_loop
