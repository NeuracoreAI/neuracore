"""Client streaming manager for real-time robot data streaming.

This module provides WebRTC-based peer-to-peer streaming capabilities for robot
sensor data including video feeds and JSON event streams. It handles signaling,
connection management, and automatic reconnection with exponential backoff.
"""

import asyncio
import logging

from aiohttp import ClientSession
from neuracore_types import (
    AvailableRobotInstance,
    HandshakeMessage,
    MessageType,
    OpenConnectionDetails,
    SynchronizedPoint,
)

from neuracore.core.streaming.p2p.base_p2p_connection_manager import (
    BaseP2PStreamManager,
)
from neuracore.core.streaming.p2p.consumer.consumer_connection import (
    PeerToPeerConsumerConnection,
)
from neuracore.core.streaming.p2p.consumer.ice_models import IceConfig
from neuracore.core.streaming.p2p.consumer.sync_point_parser import merge_sync_points
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.provider.global_live_data_enabled import (
    get_consume_live_data_enabled_manager,
)
from neuracore.core.streaming.p2p.signalling_endpoint import SignallingTarget

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
        client_session: ClientSession,
        loop: asyncio.AbstractEventLoop,
    ):
        """Initialize the client streaming manager.

        Args:
            robot_id: Unique identifier for the robot
            robot_instance: Instance number of the robot
            client_session: HTTP client session for API requests
            loop: Event loop for async operations
        """
        self.robot_id = robot_id
        self.robot_instance = robot_instance
        self.loop = loop
        self.client_session = client_session
        self._current_track_information: AvailableRobotInstance | None = None
        self.streaming = EnabledManager.derived_manger(
            get_consume_live_data_enabled_manager(), loop=loop
        )
        self.streaming.add_listener(EnabledManager.DISABLED, self._on_close)
        self.connections: dict[str, PeerToPeerConsumerConnection] = {}
        # Each signalling stream has its own TURN credentials.
        self.ice_configs: dict[str, IceConfig] = {}

    @property
    def enabled_manager(self) -> EnabledManager:
        """Get the enabled manager for this streaming manager.

        Returns:
            EnabledManager: determines wether this streaming manager is enabled
        """
        return self.streaming

    @property
    def current_track_information(self) -> AvailableRobotInstance | None:
        """Get the current track information for this robot instance.

        Returns:
            AvailableRobotInstance: The tracks that are currently provided by
                this robot instance.
        """
        return self._current_track_information

    @current_track_information.setter
    def current_track_information(
        self, current_track_information: AvailableRobotInstance | None
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

    async def _get_ice_config(self, target: SignallingTarget) -> IceConfig:
        """Get, once per signalling stream, the ICE configuration.

        Returns:
            IceConfig: The ICE configuration containing STUN/TURN server details.
        """
        ice_config = self.ice_configs.get(target.stream_id)
        if ice_config is not None:
            return ice_config

        base_url, headers = await target.endpoint()
        # The response is read within a context manager so the underlying
        # connection is released back to the session pool.
        async with self.client_session.get(
            f"{base_url}/turn_servers/{target.stream_id}",
            headers=headers,
        ) as response:
            ice_config = IceConfig.model_validate(await response.json())
        self.ice_configs[target.stream_id] = ice_config
        return ice_config

    def get_latest_data(self) -> SynchronizedPoint:
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

        return len(self.current_track_information.tracks)

    async def create_new_connection(
        self,
        remote_stream_id: str,
        connection_id: str,
        connection_details: OpenConnectionDetails,
        target: SignallingTarget,
    ) -> EnabledManager:
        """Create a new peer-to-peer connection to a remote stream.

        Args:
            remote_stream_id: ID of the remote stream to connect to
            connection_id: Unique identifier for this connection
            connection_details: The describes the type of connection to establish.
            target: The stream that signals the connection.

        Returns:
            The enabled manager for this connection.
        """
        expected_tracks = []
        if self.current_track_information is not None:
            expected_tracks = self.current_track_information.tracks.get(
                remote_stream_id, []
            )

        connection = PeerToPeerConsumerConnection(
            connection_id=connection_id,
            local_stream_id=target.stream_id,
            remote_stream_id=remote_stream_id,
            connection_details=connection_details,
            client_session=self.client_session,
            expected_tracks=expected_tracks,
            ice_config=await self._get_ice_config(target),
            loop=self.loop,
            signalling=target.endpoint,
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
