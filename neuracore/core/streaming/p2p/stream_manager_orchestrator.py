"""Stream manager factory for WebRTC connections.

This module provides a factory for creating and managing WebRTC stream managers
for both consuming and providing data streams from robot instances. It also
integrates with the signalling server to route messages.
"""

import asyncio
from typing import Optional

from aiohttp import ClientSession, ClientTimeout
from neuracore_types import RobotInstanceIdentifier

from neuracore.core.auth import Auth, get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.core.streaming.event_loop_utils import get_running_loop
from neuracore.core.streaming.p2p.base_p2p_connection_manager import (
    BaseP2PStreamManager,
    BaseStreamManagerOrchestrator,
    ManagerType,
)
from neuracore.core.streaming.p2p.consumer.client_consumer_stream_manager import (
    ClientConsumerStreamManager,
)
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.provider.client_provider_stream_manager import (
    ClientProviderStreamManager,
)
from neuracore.core.streaming.p2p.provider.global_live_data_enabled import (
    global_consume_live_data_manager,
    global_provide_live_data_manager,
)
from neuracore.core.streaming.p2p.signalling_events_consumer import (
    SignallingEventsConsumer,
)
from neuracore.core.utils.singleton_metaclass import SingletonMetaclass


class StreamManagerOrchestrator(
    BaseStreamManagerOrchestrator, metaclass=SingletonMetaclass
):
    """A singleton for creating stream managers for different robot instances.

    This class ensures that for each robot instance, there is a single
    `ClientConsumerStreamManager` and `ClientProviderStreamManager` instance,
    facilitating the management of WebRTC connections for both consuming
    and providing data streams. It also manages the `SignallingEventsConsumer`
    which routes incoming signalling messages to the correct manager.
    """

    def __init__(
        self,
        org_id: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        client_session: Optional[ClientSession] = None,
        auth: Optional[Auth] = None,
    ):
        """Initialize the stream manager factory.

        Args:
            org_id: the organization to receive signalling from. If not provided
                defaults to the current org.
            loop: the event loop to run on. Defaults to the running loop if not
                provided.
            client_session: The http session to use. Defaults to a new session
                if not provided.
            auth: The auth instance used to connect to the signalling server or
                defaults to the global auth provider if not provided.
        """
        self.provider_managers: dict[
            RobotInstanceIdentifier, ClientProviderStreamManager
        ] = {}
        self.consumer_managers: dict[
            RobotInstanceIdentifier, ClientConsumerStreamManager
        ] = {}

        self.org_id = org_id or get_current_org()
        self.auth = auth or get_auth()
        self.loop = loop or get_running_loop()
        self.client_session = client_session or ClientSession(
            timeout=ClientTimeout(sock_read=None, total=None), loop=self.loop
        )

        self.signalling_consumer = SignallingEventsConsumer(
            manager_factory=self,
            org_id=self.org_id,
            loop=self.loop,
            auth=self.auth,
            client_session=self.client_session,
            enabled_manager=EnabledManager.any_enabled(
                global_provide_live_data_manager, global_consume_live_data_manager
            ),
        )

    def get_manager(
        self, type: ManagerType, robot_id: str, robot_instance: int
    ) -> BaseP2PStreamManager:
        """Get or create a stream manager.

        Args:
            type: The type of manager to retrieve.
            robot_id: Unique identifier for the robot.
            robot_instance: Instance number of the robot.

        Returns:
            BaseP2PStreamManager: The stream manager instance.

        Raises:
            ValueError: If an unknown manager type is requested.
        """
        if type == ManagerType.CONSUMER:
            return self.get_consumer_manager(robot_id, robot_instance)
        elif type == ManagerType.PROVIDER:
            return self.get_provider_manager(robot_id, robot_instance)
        else:
            raise ValueError(f"Unknown manager type: {type}")

    def remove_manager(
        self, robot_id: str, robot_instance: int, type: Optional[ManagerType] = None
    ) -> None:
        """Remove a manager for a specific robot instance.

        Args:
            robot_id: Unique identifier for the robot.
            robot_instance: Instance number of the robot.
            type: The type of manager to remove. If None, removes both.
        """
        key = RobotInstanceIdentifier(robot_id=robot_id, robot_instance=robot_instance)
        if type is None or type == ManagerType.CONSUMER:
            if key in self.consumer_managers:
                self.consumer_managers[key].close()
                del self.consumer_managers[key]
        if type is None or type == ManagerType.PROVIDER:
            if key in self.provider_managers:
                self.provider_managers[key].close()
                del self.provider_managers[key]

    def get_consumer_manager(
        self, robot_id: str, robot_instance: int
    ) -> ClientConsumerStreamManager:
        """Get or create a consumer manager for a specific robot instance.

        Uses a singleton pattern to ensure only one manager exists per robot instance

        Args:
            robot_id: Unique identifier for the robot
            robot_instance: Instance number of the robot

        Returns:
            ClientConsumerStreamManager: The consumer manager for this robot instance
        """
        key = RobotInstanceIdentifier(robot_id=robot_id, robot_instance=robot_instance)
        if key not in self.consumer_managers:
            self.consumer_managers[key] = ClientConsumerStreamManager(
                local_stream_id=self.signalling_consumer.local_stream_id,
                robot_id=robot_id,
                robot_instance=robot_instance,
                org_id=self.org_id,
                client_session=self.client_session,
                loop=self.loop,
                auth=self.auth,
            )
            self.consumer_managers[key].enabled_manager.add_listener(
                EnabledManager.DISABLED,
                lambda: self.remove_manager(
                    robot_id, robot_instance, ManagerType.CONSUMER
                ),
            )

        return self.consumer_managers[key]

    def get_provider_manager(
        self, robot_id: str, robot_instance: int
    ) -> ClientProviderStreamManager:
        """Get or create a provider manager for a specific robot instance.

        Uses a singleton pattern to ensure only one manager exists per robot instance

        Args:
            robot_id: Unique identifier for the robot
            robot_instance: Instance number of the robot

        Returns:
            ClientProviderStreamManager: The provider manager for this robot instance
        """
        key = RobotInstanceIdentifier(robot_id=robot_id, robot_instance=robot_instance)
        if key not in self.provider_managers:
            self.provider_managers[key] = ClientProviderStreamManager(
                local_stream_id=self.signalling_consumer.local_stream_id,
                robot_id=robot_id,
                robot_instance=robot_instance,
                org_id=self.org_id,
                client_session=self.client_session,
                loop=self.loop,
                auth=self.auth,
            )
            self.provider_managers[key].enabled_manager.add_listener(
                EnabledManager.DISABLED,
                lambda: self.remove_manager(
                    robot_id, robot_instance, ManagerType.PROVIDER
                ),
            )

        return self.provider_managers[key]
