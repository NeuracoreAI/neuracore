"""Client streaming manager for real-time robot data streaming.

This module provides WebRTC-based peer-to-peer streaming capabilities for robot
sensor data including video feeds and JSON event streams. It handles signaling,
connection management, and automatic reconnection with exponential backoff.
"""

import asyncio
import logging
from typing import Dict, Optional

from aiohttp import ClientSession
from neuracore_types import (
    AvailableRobotInstance,
    HandshakeMessage,
    MessageType,
    OpenConnectionDetails,
    SyncPoint,
)

from neuracore.core.auth import Auth, get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.streaming.p2p.base_p2p_connection_manager import (
    BaseP2PStreamManager,
)
from neuracore.core.streaming.p2p.consumer.consumer_connection import (
    PierToPierConsumerConnection,
)
from neuracore.core.streaming.p2p.consumer.ice_models import IceConfig
from neuracore.core.streaming.p2p.consumer.sync_point_parser import merge_sync_points
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.provider.global_live_data_enabled import (
    global_consume_live_data_manager,
)

logger = logging.getLogger(__name__)


class ClientConsumerStreamManager(BaseP2PStreamManager):
    """Manages WebRTC streaming connections for robot sensor data.

    Handles peer-to-peer connections, signaling, video tracks, and JSON data streams
    with automatic reconnection and proper cleanup.
    """

    def __init__(
        self,
        robot_id: str,
        robot_instance: int,
        local_stream_id: str,
        client_session: ClientSession,
        loop: asyncio.AbstractEventLoop,
        org_id: Optional[str] = None,
        auth: Optional[Auth] = None,
    ):
        """Initialize the client streaming manager.

        Args:
            robot_id: Unique identifier for the robot
            robot_instance: Instance number of the robot
            local_stream_id: Unique identifier of this recipient.
            client_session: HTTP client session for API requests
            loop: Event loop for async operations
            org_id: The unique identifier for the organization. If not provided,
                defaults to the current org.
            auth: Authentication object. If not provided, uses default auth
        """
        self.robot_id = robot_id
        self.robot_instance = robot_instance
        self.local_stream_id = local_stream_id
        self.loop = loop
        self.client_session = client_session
        self._current_track_information: Optional[AvailableRobotInstance] = None
        self.org_id = org_id or get_current_org()
        self.auth = auth or get_auth()
        self.streaming = EnabledManager.derived_manger(
            global_consume_live_data_manager, loop=loop
        )
        self.streaming.add_listener(EnabledManager.DISABLED, self._on_close)
        self.connections: Dict[str, PierToPierConsumerConnection] = {}
        self.ice_config: Optional[IceConfig] = None

    @property
    def enabled_manager(self) -> EnabledManager:
        """Get the enabled manager for this streaming manager.

        Returns:
            EnabledManager: determines wether this streaming manager is enabled
        """
        return self.streaming

    @property
    def current_track_information(self) -> Optional[AvailableRobotInstance]:
        """Get the current track information for this robot instance.

        Returns:
            AvailableRobotInstance: The tracks that are currently provided by
                this robot instance.
        """
        return self._current_track_information

    @current_track_information.setter
    def current_track_information(
        self, current_track_information: Optional[AvailableRobotInstance]
    ) -> None:
        """Set the current track information for this robot instance.

        Args:
            current_track_information: The tracks that are currently provided by
                this robot instance.
        """
        self._current_track_information = current_track_information
        if current_track_information is None:
            return

        for connection in self.connections.values():
            connection.expected_tracks = current_track_information.tracks.get(
                connection.remote_stream_id, []
            )

    async def _get_ice_config(self) -> IceConfig:
        """Get or create the ICE configuration for WebRTC connections.

        Fetches the ICE server configuration from the signaling server if not
        already available and caches it.

        Returns:
            IceConfig: The ICE configuration containing STUN/TURN server details.
        """
        if self.ice_config is not None:
            return self.ice_config

        response = await self.client_session.get(
            f"{API_URL}/org/{self.org_id}/signalling/turn_servers/{self.local_stream_id}",
            headers=self.auth.get_headers(),
        )
        self.ice_config = IceConfig.model_validate(await response.json())

        return self.ice_config

    def get_latest_data(self) -> SyncPoint:
        """Gets a sync point consisting of the latest data from all connections.

        Returns:
            The latest data from all connections.
        """
        sync_points = (
            connection.get_latest_data() for connection in self.connections.values()
        )

        return merge_sync_points(*sync_points)

    def all_remote_nodes_connected(self) -> bool:
        """Get wether all the remote nodes are connected.

        Returns:
            True if all remote nodes are connected, False otherwise.
        """
        if self.current_track_information is None:
            return False

        connection_stream_map = {
            connection.remote_stream_id: connection
            for connection in self.connections.values()
        }

        return all(
            stream_id in connection_stream_map
            and connection_stream_map[stream_id].fully_connected()
            for stream_id in self.current_track_information.tracks.keys()
            if stream_id != self.local_stream_id
        )

    def num_remote_nodes(self) -> int:
        """Get the number of remote nodes that should be connected.

        Based on the current information of other nodes get the number of
        remote nodes that should be connected. for this robot.

        Returns:
            The number of remote nodes that should be connected.
        """
        if self.current_track_information is None:
            return 0

        return sum(
            1
            for stream_id in self.current_track_information.tracks.keys()
            if stream_id != self.local_stream_id
        )

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
        expected_tracks = []
        if self.current_track_information is not None:
            expected_tracks = self.current_track_information.tracks.get(
                remote_stream_id, []
            )

        connection = PierToPierConsumerConnection(
            connection_id=connection_id,
            local_stream_id=self.local_stream_id,
            remote_stream_id=remote_stream_id,
            connection_details=connection_details,
            client_session=self.client_session,
            expected_tracks=expected_tracks,
            ice_config=await self._get_ice_config(),
            org_id=self.org_id,
            loop=self.loop,
            auth=self.auth,
        )

        @connection.enabled_manager.on(EnabledManager.DISABLED)
        def on_close() -> None:
            self.connections.pop(connection_id, None)

        self.connections[connection_id] = connection

        return connection.enabled_manager

    async def remove_connection(self, connection_id: str) -> None:
        """Remove a peer-to-peer connection.

        Args:
             connection_id: ID of the connection to end.
        """
        connection = self.connections.pop(connection_id, None)
        if connection is None:
            return

        connection.close()

    async def on_message(self, message: HandshakeMessage) -> None:
        """Handle a signalling message for one of the manager's connections.

        Args:
            message: The message to handle.
        """
        connection = self.connections.get(message.connection_id, None)
        if not connection:
            raise ValueError(f"Connection not found for id: {message.connection_id}")

        if message.type == MessageType.ICE_CANDIDATE:
            await connection.on_ice(message.data)
        elif message.type == MessageType.SDP_OFFER:
            await connection.on_offer(message.data)
        else:
            logger.warning(f"Unsupported message type: {message.type}")

    def _on_close(self) -> None:
        """Internal cleanup method called when streaming is disabled."""
        for connection in self.connections.values():
            connection.close()

        self.connections.clear()
