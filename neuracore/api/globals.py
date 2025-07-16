"""Global singleton for managing Neuracore session state.

This module provides a singleton class that maintains global state across
the Neuracore session, including active robot connections, dataset information,
and validation status.
"""

from typing import Optional

from neuracore.core.robot import Robot
from neuracore.core.utils.singleton_metaclass import SingletonMetaclass


class GlobalSingleton(metaclass=SingletonMetaclass):
    """Singleton class for managing global Neuracore session state.

    This class ensures that only one instance exists throughout the application
    lifecycle and maintains critical session information including the currently
    active robot, dataset ID, and version validation status. The singleton pattern
    ensures consistent state management across all Neuracore modules.

    Attributes:
        _has_validated_version: Whether version compatibility has been verified
            with the Neuracore server.
        _active_robot: Currently active robot instance, used as the default
            for operations when no specific robot is specified.
        _active_dataset_id: ID of the currently active dataset that new
            recordings will be associated with.
    """

    _has_validated_version = False
    _active_robot: Optional[Robot] = None
    _active_dataset_id: Optional[str] = None
