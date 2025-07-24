"""Thread-safe enabled state manager for streaming operations.

This module provides a thread-safe manager for controlling enabled/disabled
state with event emission capabilities. Used to coordinate streaming state
across multiple components and threads.
"""

import threading
from asyncio import AbstractEventLoop
from typing import Optional

from pyee.asyncio import AsyncIOEventEmitter

from neuracore.core.streaming.event_loop_utils import get_running_loop


class EnabledManager(AsyncIOEventEmitter):
    """Thread-safe manager for enabled/disabled state with event notifications.

    Provides a thread-safe way to manage boolean state with automatic event
    emission when the state changes. Extends AsyncIOEventEmitter to support
    event-driven architecture across async and sync contexts.
    """

    DISABLED = "DISABLED"

    def __init__(
        self, initial_state: bool, loop: Optional[AbstractEventLoop] = None
    ) -> None:
        """Initialize the enabled manager.

        Args:
            initial_state: Initial enabled/disabled state
            loop: Optional event loop for async event emission
        """
        self.loop = loop or get_running_loop()
        super().__init__(loop=self.loop)
        self._is_enabled = initial_state
        self.lock = threading.Lock()

    def is_enabled(self) -> bool:
        """Check if the manager is in enabled state.

        Returns:
            bool: True if enabled, False if disabled
        """
        with self.lock:
            return self._is_enabled and not self.loop.is_closed()

    def is_disabled(self) -> bool:
        """Check if the manager is in disabled state.

        Returns:
            bool: True if disabled, False if enabled
        """
        return not self.is_enabled()

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

    @staticmethod
    def all_enabled(
        *managers: "EnabledManager", loop: Optional[AbstractEventLoop] = None
    ) -> "EnabledManager":
        """Constructs a new enabled manager.

        the new manager is only enabled if and only if all the provided managers
        are enabled.

        Args:
            *managers: The managers to check.

        Returns:
            A new enabled manager that is only enabled if all the provided
                managers are enabled.
        """
        enabled_count = sum(1 for manager in managers if manager.is_enabled())
        if enabled_count < len(managers):
            return EnabledManager(initial_state=False, loop=loop)

        new_manager = EnabledManager(initial_state=True, loop=loop)

        for manager in managers:
            manager.add_listener(EnabledManager.DISABLED, lambda: new_manager.disable())

        return new_manager

    @staticmethod
    def any_enabled(
        *managers: "EnabledManager", loop: Optional[AbstractEventLoop] = None
    ) -> "EnabledManager":
        """Constructs a new enabled manager.

        The new manager is only enabled if and only if any of the provided
        managers are enabled.

        Args:
            *managers: The managers to check.

        Returns:
            A new enabled manager that is only enabled if any of the provided
                managers are enabled.
        """
        enabled_count = sum(1 for manager in managers if manager.is_enabled())
        if enabled_count == 0:
            return EnabledManager(initial_state=False, loop=loop)

        new_manager = EnabledManager(initial_state=True, loop=loop)

        def on_disable() -> None:
            nonlocal enabled_count
            enabled_count += 1
            if enabled_count <= 0:
                new_manager.disable()

        for manager in managers:
            manager.add_listener(EnabledManager.DISABLED, on_disable)

        return new_manager

    @classmethod
    def derived_manger(
        cls, manager: "EnabledManager", loop: Optional[AbstractEventLoop] = None
    ) -> "EnabledManager":
        """Constructs a new enabled manager that is derived from another.

        Args:
            manager: The manager to derive from.

        Returns:
            A new enabled manager that is enabled if the provided manager is
                enabled.
        """
        return cls.all_enabled(manager, loop=loop)
