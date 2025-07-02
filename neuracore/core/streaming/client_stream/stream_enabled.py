"""Thread-safe enabled state manager for streaming operations.

This module provides a thread-safe manager for controlling enabled/disabled
state with event emission capabilities. Used to coordinate streaming state
across multiple components and threads.
"""

import threading
from asyncio import AbstractEventLoop
from typing import Optional

from pyee.asyncio import AsyncIOEventEmitter


class EnabledManager(AsyncIOEventEmitter):
    """Thread-safe manager for enabled/disabled state with event notifications.

    Provides a thread-safe way to manage boolean state with automatic event
    emission when the state changes. Extends AsyncIOEventEmitter to support
    event-driven architecture across async and sync contexts.
    """

    DISABLED = "DISABLED"

    def __init__(self, initial_state: bool, loop: Optional[AbstractEventLoop] = None):
        """Initialize the enabled manager.

        Args:
            initial_state: Initial enabled/disabled state
            loop: Optional event loop for async event emission
        """
        super().__init__(loop)
        self._is_enabled = initial_state
        self.lock = threading.Lock()

    def is_enabled(self) -> bool:
        """Check if the manager is in enabled state.

        Returns:
            bool: True if enabled, False if disabled
        """
        with self.lock:
            return self._is_enabled

    def disable(self) -> None:
        """Disable the manager and emit notification.

        Thread-safely disables the manager, emits a DISABLED event,
        and removes all event listeners to prevent memory leaks.
        If already disabled, this is a no-op.
        """
        with self.lock:
            if not self._is_enabled:
                return
            self._is_enabled = False
            self.emit(self.DISABLED)
            self.remove_all_listeners()
