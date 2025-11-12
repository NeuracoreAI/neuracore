"""Client streaming manager for real-time robot data streaming.

This module provides WebRTC-based peer-to-peer streaming capabilities for robot
sensor data including video feeds and JSON event streams. It handles signaling,
connection management, and automatic reconnection with exponential backoff.
"""

import asyncio
import logging
from collections import defaultdict
from concurrent.futures import Future
from typing import Dict, Optional, Tuple

from aiohttp import ClientSession
from neuracore_types import (
    AvailableRobotCapacityUpdate,
    AvailableRobotInstance,
    HandshakeMessage,
    MessageType,
    OpenConnectionRequest,
    RobotInstanceIdentifier,
    VideoFormat,
)
from typing_extensions import TypeAlias

from neuracore.core.auth import Auth, get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.streaming.base_sse_consumer import (
    BaseSSEConsumer,
    EventSourceConfig,
)
from neuracore.core.streaming.event_loop_utils import get_running_loop
from neuracore.core.streaming.p2p.consumer.client_consumer_stream_manager import (
    ClientConsumerStreamManager,
)
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.stream_manager_orchestrator import (
    StreamManagerOrchestrator,
)
from neuracore.core.utils.background_coroutine_tracker import BackgroundCoroutineTracker

logger = logging.getLogger(__name__)


# Mapping each robot instance to its nodes and a hash over it's connections
InstanceStreamMap: TypeAlias = Dict[RobotInstanceIdentifier, Dict[str, int]]

# Mapping each robot to its connections to each node
InstanceConnectionMap: TypeAlias = Dict[
    RobotInstanceIdentifier, Dict[str, Tuple[str, int]]
]


class OrgNodesManager(BaseSSEConsumer):
    """Manages the connections to other nodes in an organization.

    Listens to updates in the available nodes to connect to and informs robot
    consumers.
    """

    def __init__(
        self,
        org_id: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        enabled_manager: Optional[EnabledManager] = None,
        background_coroutine_tracker: Optional[BackgroundCoroutineTracker] = None,
        client_session: Optional[ClientSession] = None,
        auth: Optional[Auth] = None,
        stream_manager_orchestrator: Optional[StreamManagerOrchestrator] = None,
    ) -> None:
        """Initialize the organization node manager.

        Args:
            org_id: the organization to receive streaming information from. If n
                ot provided defaults to the current org.
            loop: the event loop to run on. Defaults to the running loop if not
                provided.
            enabled_manager: The enabled manager for whether this should be
                consuming. Defaults to a new enabled manger if not provided.
            background_coroutine_tracker: The storage for background tasks
                scheduled on receiving events. Defaults to a new tracker if not
                provided.
            client_session: The http session to use. Defaults to a new session
                if not provided.
            auth: The auth instance used to connect to the signalling server or
                defaults to the global auth provider if not provided.
            stream_manager_orchestrator: the factory used to create stream managers

        """
        super().__init__(
            loop=loop,
            enabled_manager=enabled_manager,
            background_coroutine_tracker=background_coroutine_tracker,
            client_session=client_session,
        )
        self.stream_manager_orchestrator = (
            stream_manager_orchestrator or StreamManagerOrchestrator()
        )

        self.org_id = org_id or get_current_org()
        self.auth = auth or get_auth()

        self.consumers: Dict[RobotInstanceIdentifier, ClientConsumerStreamManager] = {}

        self.connections: InstanceConnectionMap = defaultdict(dict)

        self.last_nodes: InstanceStreamMap = defaultdict(dict)

        self.last_update: Optional[AvailableRobotCapacityUpdate] = None

    def get_sse_client_config(self) -> EventSourceConfig:
        """Used to configure the event client to consume events from the server.

        Returns:
            the configuration to be used to connect to the client
        """
        return EventSourceConfig(
            url=f"{API_URL}/org/{self.org_id}/signalling/available_robots",
            request_options={
                "headers": self.auth.get_headers(),
            },
        )

    async def on_message(self, message_data: str) -> None:
        """The main handler for when the stream receives a message.

        Args:
            message_data: The raw string data of the message

        """
        robot_update = AvailableRobotCapacityUpdate.model_validate_json(message_data)
        new_nodes = self.get_instance_stream_map(robot_update)
        self._apply_stream_changes(self.last_nodes, new_nodes)
        self.last_update = robot_update
        self.last_nodes = new_nodes

        for consumer in self.consumers:
            track_info = self._get_last_robot_tracks(
                robot_id=consumer.robot_id,
                robot_instance=consumer.robot_instance,
            )
            self.consumers[consumer].current_track_information = track_info

    def _get_last_robot_tracks(
        self, robot_id: str, robot_instance: int
    ) -> Optional[AvailableRobotInstance]:
        if self.last_update is None:
            return None

        robot = next(
            (robot for robot in self.last_update.robots if robot.robot_id == robot_id),
            None,
        )
        if robot is None:
            return None

        return robot.instances.get(robot_instance, None)

    def get_robot_consumer(
        self, robot_id: str, robot_instance: int
    ) -> ClientConsumerStreamManager:
        """Get or create a consumer for a specific robot instance.

        Args:
            robot_id: Unique identifier for the robot
            robot_instance: Instance number of the robot

        Returns:
            The consumer for this robot instance
        """
        key = RobotInstanceIdentifier(robot_id=robot_id, robot_instance=robot_instance)
        if key in self.consumers:
            return self.consumers[key]

        self.consumers[key] = self.stream_manager_orchestrator.get_consumer_manager(
            robot_id=robot_id, robot_instance=robot_instance
        )

        self.consumers[key].current_track_information = self._get_last_robot_tracks(
            robot_id=robot_id, robot_instance=robot_instance
        )

        if self.last_nodes is None:
            return self.consumers[key]

        for remote_stream_id, track_hash in self.last_nodes[key].items():
            self.background_tracker.submit_background_coroutine(
                self._create_new_connection(
                    manager=self.consumers[key],
                    track_hash=track_hash,
                    connection_request=OpenConnectionRequest(
                        from_id=self.stream_manager_orchestrator.signalling_consumer.local_stream_id,
                        to_id=remote_stream_id,
                        robot_id=robot_id,
                        robot_instance=robot_instance,
                        video_format=VideoFormat.NEURACORE_CUSTOM,
                    ),
                )
            )

        return self.consumers[key]

    def get_instance_stream_map(
        self,
        update: AvailableRobotCapacityUpdate,
    ) -> InstanceStreamMap:
        """Reduces the availability update down to just the available nodes.

        Args:
            update: the availability update for robots in the organization.

        Returns:
            InstanceStreamMap: a mapping from each robot to its set of stream ids.
        """
        local_stream_id = (
            self.stream_manager_orchestrator.signalling_consumer.local_stream_id
        )

        instance_stream_map: InstanceStreamMap = defaultdict(dict)
        for robot in update.robots:
            for instance in robot.instances.values():
                nodes = {}
                for steam_id, track_list in instance.tracks.items():
                    if steam_id == local_stream_id:
                        continue
                    track_hash = hash(tuple(sorted(track.id for track in track_list)))
                    nodes[steam_id] = track_hash

                instance_stream_map[
                    RobotInstanceIdentifier(
                        robot_id=robot.robot_id, robot_instance=instance.robot_instance
                    )
                ] = nodes

        return instance_stream_map

    async def _create_new_connection(
        self,
        track_hash: int,
        manager: ClientConsumerStreamManager,
        connection_request: OpenConnectionRequest,
    ) -> None:
        """Create a new connection for a specific robot instance.

        Requests a connection from the signalling server and registers it with
        the signalling consumer.

        Args:
            track_hash: a hash over the tracks this connection is setup to handle.
            manager: the consumer manager that will be responsible for this new
                connection.
            connection_request: the details of the connection to be opened.
        """
        response = await self.client_session.post(
            url=f"{API_URL}/org/{self.org_id}/signalling/connection?signature=temp",
            json=connection_request.model_dump(mode="json"),
            headers=self.auth.get_headers(),
        )
        response.raise_for_status()
        connection_message = HandshakeMessage.model_validate(await response.json())
        assert connection_message.type == MessageType.OPEN_CONNECTION

        robot_identifier = RobotInstanceIdentifier(
            robot_id=connection_request.robot_id,
            robot_instance=connection_request.robot_instance,
        )
        existing_connection = self.connections[robot_identifier].pop(
            connection_request.to_id, None
        )
        if existing_connection is not None:
            self.background_tracker.submit_background_coroutine(
                self.consumers[robot_identifier].remove_connection(
                    existing_connection[0]
                )
            )

        self.connections[robot_identifier][connection_request.to_id] = (
            connection_message.connection_id,
            track_hash,
        )

        signalling_consumer = self.stream_manager_orchestrator.signalling_consumer

        connection_enabled = await signalling_consumer.create_new_connection(
            message=connection_message, manager=manager
        )

        @connection_enabled.on(EnabledManager.DISABLED)
        async def on_close() -> None:
            current_connection_details = self.connections[robot_identifier].get(
                connection_request.to_id, None
            )
            if current_connection_details is None:
                return
            connection_id, _ = current_connection_details
            if connection_message.connection_id == connection_id:
                self.connections[robot_identifier].pop(connection_request.to_id, None)

    def _apply_stream_changes(
        self, old: InstanceStreamMap, current: InstanceStreamMap
    ) -> None:
        """Apply the changes in the available nodes between versions.

        this is done by adding or removing connections in the relevant managers.

        Args:
            old: the available nodes before the change
            current: the new nodes available for connection.
        """
        local_stream_id = (
            self.stream_manager_orchestrator.signalling_consumer.local_stream_id
        )

        for robot, manager in self.consumers.items():
            robot_id, robot_instance = robot
            old_streams = set(self.connections[robot].keys())
            current_streams = set(current[robot].keys())

            removed_streams = old_streams - current_streams
            added_streams = current_streams - old_streams
            existing_streams = current_streams & old_streams

            for removed_stream in removed_streams:
                (connection_id, _) = self.connections[robot].pop(removed_stream)

                self.background_tracker.submit_background_coroutine(
                    self.consumers[robot].remove_connection(connection_id)
                )

            for added_stream in added_streams:
                self.background_tracker.submit_background_coroutine(
                    self._create_new_connection(
                        manager=manager,
                        track_hash=current[robot][added_stream],
                        connection_request=OpenConnectionRequest(
                            from_id=local_stream_id,
                            to_id=added_stream,
                            robot_id=robot_id,
                            robot_instance=robot_instance,
                            video_format=VideoFormat.NEURACORE_CUSTOM,
                        ),
                    )
                )

            for existing_stream in existing_streams:
                old_track_hash = old[robot][existing_stream]
                current_track_hash = current[robot][existing_stream]
                if old_track_hash == current_track_hash:
                    continue

                (connection_id, _) = self.connections[robot].pop(existing_stream)
                self.background_tracker.submit_background_coroutine(
                    self.consumers[robot].remove_connection(connection_id)
                )
                self.background_tracker.submit_background_coroutine(
                    self._create_new_connection(
                        manager=manager,
                        track_hash=current[robot][existing_stream],
                        connection_request=OpenConnectionRequest(
                            from_id=local_stream_id,
                            to_id=existing_stream,
                            robot_id=robot_id,
                            robot_instance=robot_instance,
                            video_format=VideoFormat.NEURACORE_CUSTOM,
                        ),
                    )
                )

    def _remove_robot_consumer(self, robot_id: str, robot_instance: int) -> None:
        """Remove a consumer for a specific robot instance.

        Args:
            robot_id: Unique identifier for the robot
            robot_instance: Instance number of the robot
        """
        key = RobotInstanceIdentifier(robot_id=robot_id, robot_instance=robot_instance)
        if key not in self.consumers:
            return

        self.consumers[key].close()
        del self.consumers[key]

    async def close_robot_consumers(self) -> None:
        """Close all active peer-to-peer connections for this org."""
        pass

    def _on_close(self) -> None:
        """Close all connections and streams gracefully.

        Disables streaming, closes all P2P connections, stops video tracks,
        and cleans up resources.
        """
        super()._on_close()
        for robot_consumer in self.consumers.values():
            robot_consumer.close()

        self.consumers.clear()


_org_node_managers: Dict[str, Future[OrgNodesManager]] = {}


async def _create_org_nodes_manager(org_id: str) -> OrgNodesManager:
    """Create a new organization node manager instance.

    Args:
        org_id: Unique identifier for the organization.

    Returns:
        The new node manager for this organization
    """
    return OrgNodesManager(
        org_id=org_id,
        loop=asyncio.get_event_loop(),
    )


def get_org_nodes_manager(org_id: str) -> OrgNodesManager:
    """Get or create a streaming manager for a specific robot instance.

    Uses a singleton pattern to ensure only one manager exists per organization.
    Thread-safe and handles event loop coordination.

    Args:
        org_id: Unique identifier for the organization

    Returns:
        The node manager for this organization
    """
    if org_id not in _org_node_managers:
        # This needs to be run in the event loop thread
        # otherwise we will get a 'RuntimeError: no running event loop'
        manager = asyncio.run_coroutine_threadsafe(
            _create_org_nodes_manager(org_id), get_running_loop()
        )
        _org_node_managers[org_id] = manager
    return _org_node_managers[org_id].result()
