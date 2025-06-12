"""Global singleton for managing Neuracore session state.

This module provides a singleton class that maintains global state across
the Neuracore session, including active robot connections, dataset information,
and validation status.
"""

from typing import Optional

from ..core.robot import Robot


class GlobalSingleton(object):
    """Singleton class for managing global Neuracore session state.

    This class ensures that only one instance exists throughout the application
    lifecycle and maintains critical session information including the currently
    active robot, dataset ID, and version validation status. The singleton pattern
    ensures consistent state management across all Neuracore modules.

    Attributes:
        _instance: Class-level singleton instance reference.
        _has_validated_version: Whether version compatibility has been verified
            with the Neuracore server.
        _active_robot: Currently active robot instance, used as the default
            for operations when no specific robot is specified.
        _active_dataset_id: ID of the currently active dataset that new
            recordings will be associated with.
    """

    _instance = None
    _has_validated_version = False
    _active_robot: Optional[Robot] = None
    _active_dataset_id: Optional[str] = None

    def __new__(cls) -> "GlobalSingleton":
        """Create or return the singleton instance.

        Ensures only one instance of GlobalSingleton exists throughout the
        application lifecycle. Subsequent calls to the constructor will
        return the same instance.

        Returns:
            The singleton instance of GlobalSingleton.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
