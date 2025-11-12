"""Base class for client streaming manager.

This module provides the basic interface for a streaming manager's functionality.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from neuracore_types import HandshakeMessage, OpenConnectionDetails

from neuracore.core.streaming.p2p.enabled_manager import EnabledManager

logger = logging.getLogger(__name__)


class BaseP2PStreamManager(ABC):
    """Base class for managing WebRTC streaming connections relating to a robot."""

    @property
    @abstractmethod
    def enabled_manager(self) -> EnabledManager:
        """Get the enabled manager for this streaming manager.

        Returns:
            EnabledManager: determines wether this streaming manager is enabled
        """
        raise NotImplementedError("enabled_manager not implemented")

    @abstractmethod
    async def create_new_connection(
        self,
        remote_stream_id: str,
        connection_id: str,
        connection_details: OpenConnectionDetails,
    ) -> EnabledManager:
        """Create a new peer-to-peer connection to a remote stream.

        Args:
            remote_stream_id: ID of the remote stream to connect to
            connection_id: Unique identifier for this connection
            connection_details: The describes the type of connection to establish.

        Returns:
            The enabled manager for this connection.
        """
        raise NotImplementedError("create_new_connection not implemented")

    @abstractmethod
    async def remove_connection(self, connection_id: str) -> None:
        """Remove a peer-to-peer connection.

        Args:
            connection_id: ID of the connection to end.
        """
        raise NotImplementedError("remove_connection not implemented")

    async def on_stream_resurrected(self) -> None:
        """Optional handler for when the stream has been resurrected."""
        pass

    @abstractmethod
    async def on_message(self, message: HandshakeMessage) -> None:
        """Handle a signalling message for one of the manager's connections.

        Args:
            message: The message to handle.
        """
        raise NotImplementedError("on_message not implemented")

    def close(self) -> None:
        """Close all connections and streams gracefully.

        Disables streaming, closes all P2P connections, stops video tracks,
        and cleans up resources.
        """
        self.enabled_manager.disable()


class ManagerType(str, Enum):
    """Enumerates the different kinds of message consumer."""

    CONSUMER = "CONSUMER"
    PROVIDER = "PROVIDER"


class BaseStreamManagerOrchestrator(ABC):
    """Base class for a object that creates and manages stream managers."""

    @abstractmethod
    def get_manager(
        self, type: ManagerType, robot_id: str, robot_instance: int
    ) -> BaseP2PStreamManager:
        """Get or create a stream manager.

        Args:
            robot_id: Unique identifier for the robot
            robot_instance: Instance number of the robot

        Returns:
            BaseP2PStreamManager: The stream manager instance.
        """
        raise NotImplementedError("get_manager not implemented")

    @abstractmethod
    def remove_manager(
        self, robot_id: str, robot_instance: int, type: Optional[ManagerType] = None
    ) -> None:
        """Remove a manager for a specific robot instance.

        Args:
            robot_id: Unique identifier for the robot.
            robot_instance: Instance number of the robot.
            type: The type of manager to remove. If None, removes both.
        """
        raise NotImplementedError("get_manager not implemented")
