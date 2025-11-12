"""Signalling events consumer for real-time robot data streaming.

This module provides a consumer for routing signalling events to the appropriate
connection manager.
"""

import asyncio
from typing import Optional
from uuid import uuid4

from aiohttp import ClientSession
from neuracore_types import (
    HandshakeMessage,
    MessageType,
    OpenConnectionDetails,
    StreamAliveResponse,
)

from neuracore.core.auth import Auth, get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.const import API_URL
from neuracore.core.streaming.base_sse_consumer import (
    BaseSSEConsumer,
    EventSourceConfig,
)
from neuracore.core.streaming.p2p.base_p2p_connection_manager import (
    BaseP2PStreamManager,
    BaseStreamManagerOrchestrator,
    ManagerType,
)
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.utils.background_coroutine_tracker import BackgroundCoroutineTracker

MAX_QUEUED_MESSAGES = 100


class SignallingEventsConsumer(BaseSSEConsumer):
    """The class routes signalling events to the appropriate connection manager.

    This is a singleton class and is reasonable for all events from both the
      consumption and production side.
    """

    def __init__(
        self,
        manager_factory: BaseStreamManagerOrchestrator,
        org_id: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        enabled_manager: Optional[EnabledManager] = None,
        background_coroutine_tracker: Optional[BackgroundCoroutineTracker] = None,
        client_session: Optional[ClientSession] = None,
        auth: Optional[Auth] = None,
    ):
        """Initialise the signalling events consumer.

        Args:
            manager_factory: the factory used to instantiate new managers on
                connection request.
            org_id: the organization to receive signalling from. If not provided
                defaults to the current org.
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
        """
        super().__init__(
            loop=loop,
            enabled_manager=enabled_manager,
            background_coroutine_tracker=background_coroutine_tracker,
            client_session=client_session,
        )
        self.manager_factory = manager_factory
        self.org_id = org_id or get_current_org()
        self.local_stream_id = uuid4().hex
        self.connection_managers: dict[str, BaseP2PStreamManager] = {}
        self.message_queue: list[HandshakeMessage] = []
        self.auth = auth or get_auth()

    def get_sse_client_config(self) -> EventSourceConfig:
        """Used to configure the event client to consume events from the server.

        Returns:
            the configuration to be used to connect to the client
        """
        return EventSourceConfig(
            url=f"{API_URL}/org/{self.org_id}/signalling/notifications/{self.local_stream_id}",
            request_options={
                "headers": self.auth.get_headers(),
            },
        )

    def queue_message(self, message: HandshakeMessage) -> None:
        """Queue a message for a connection we dont have yet.

        This helps avoid missing messages when they are received out of order.

        Args:
            message: the message we don't have a recipient
                connection for yet.
        """
        self.message_queue.append(message)
        if len(self.message_queue) > MAX_QUEUED_MESSAGES:
            self.message_queue = self.message_queue[-MAX_QUEUED_MESSAGES:]

    def pop_queued_messages(self, connection_id: str) -> list[HandshakeMessage]:
        """Get all of the queued messages for a connection.

        Args:
            connection_id (str): the new connection that there may be messages
                for.

        Returns:
            List of the messages that arrived before the connection was opened.
        """
        queued_messages = [
            message
            for message in self.message_queue
            if message.connection_id == connection_id
        ]
        self.message_queue = [
            message
            for message in self.message_queue
            if message.connection_id != connection_id
        ]
        return queued_messages

    async def on_heartbeat(self) -> None:
        """Handle heartbeat events by keeping the connection alive."""
        response = await self.client_session.post(
            f"{API_URL}/org/{self.org_id}/signalling/alive/{self.local_stream_id}",
            headers=self.auth.get_headers(),
            data="pong",
        )
        response.raise_for_status()
        response = await response.json()
        response = StreamAliveResponse.model_validate(response)

        if response.resurrected:
            for manager in self.connection_managers.values():
                if manager.enabled_manager.is_disabled():
                    continue
                await manager.on_stream_resurrected()

    async def create_new_connection(
        self, message: HandshakeMessage, manager: BaseP2PStreamManager
    ) -> EnabledManager:
        """Add a connection manager listing to route signalling messages.

        Args:
            message: The message describing the connection to open
            manager: The manager responsible for this connection

        Raises:
            ValueError: if the message type is not OPEN_CONNECTION

        Returns:
            The enabled manager for this connection.
        """
        if message.type != MessageType.OPEN_CONNECTION:
            raise ValueError("Message type must be OPEN_CONNECTION")

        if manager.enabled_manager.is_disabled():
            return manager.enabled_manager

        connection_details = OpenConnectionDetails.model_validate_json(message.data)

        connection_enabled = await manager.create_new_connection(
            remote_stream_id=message.from_id,
            connection_id=message.connection_id,
            connection_details=connection_details,
        )

        @connection_enabled.on(EnabledManager.DISABLED)
        def on_close() -> None:
            self.connection_managers.pop(message.connection_id, None)

        self.connection_managers[message.connection_id] = manager

        queued_messages = self.pop_queued_messages(message.connection_id)
        for queued_message in queued_messages:
            self.background_tracker.submit_background_coroutine(
                manager.on_message(queued_message)
            )

        return connection_enabled

    def remove_connection(self, connection_id: str) -> None:
        """Remove a connection manager listing to route signalling messages.

        Args:
            connection_id: the connection to no longer consider
        """
        self.connection_managers.pop(connection_id, None)

    async def on_message(self, message_data: str) -> None:
        """The main handler for when the stream receives a message.

        Args:
            message_data: The raw string data of the message
        """
        message = HandshakeMessage.model_validate_json(message_data)
        if message.to_id != self.local_stream_id:
            raise ValueError("Handshake message to incorrect node recipient")

        if message.type == MessageType.OPEN_CONNECTION:
            connection_details = OpenConnectionDetails.model_validate_json(message.data)
            manager = self.manager_factory.get_manager(
                type=ManagerType.PROVIDER,
                robot_id=connection_details.robot_id,
                robot_instance=connection_details.robot_instance,
            )
            await self.create_new_connection(message, manager)
            return

        connection_manager = self.connection_managers.get(message.connection_id, None)

        if connection_manager is None:
            self.queue_message(message)
            return
        if connection_manager.enabled_manager.is_disabled():
            self.connection_managers.pop(message.connection_id, None)
            return
        await connection_manager.on_message(message)
