"""Event loop utilities for asyncio operations.

This module provides utilities for managing asyncio event loops across
different execution contexts, ensuring that async operations can be
performed even when no event loop is currently running.
"""

import asyncio
import threading


def get_running_loop() -> asyncio.AbstractEventLoop:
    """Get the running event loop or create a new one if none exists.

    Attempts to get the currently running event loop. If no loop is running,
    creates a new event loop, sets it as the current loop, and starts it
    in a daemon thread to keep it running in the background.

    Returns:
        asyncio.AbstractEventLoop: The running event loop

    Note:
        The created event loop runs in a daemon thread, which means it will
        be automatically terminated when the main program exits.
    """
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        threading.Thread(target=lambda: loop.run_forever(), daemon=True).start()
        return loop
