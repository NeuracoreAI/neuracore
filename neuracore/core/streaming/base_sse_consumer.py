"""This module provides the base class for consumers of server sent events."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from aiohttp import ClientSession, ClientTimeout
from aiohttp.client import _RequestOptions
from aiohttp_sse_client.client import EventSource

from neuracore.core.const import (
    STREAMING_MAXIMUM_BACKOFF_TIME_S,
    STREAMING_MINIMUM_BACKOFF_TIME_S,
)
from neuracore.core.streaming.event_loop_utils import get_running_loop
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.utils.background_coroutine_tracker import BackgroundCoroutineTracker

logger = logging.getLogger(__name__)


@dataclass
class EventSourceConfig:
    """The configuration to use when connecting to an event source.

    Attributes:
        url: The url to connect to to get the sse stream
        method: The method to use when connecting to the stream. Defaults to
            GET.
        request_options: any additional options such as headers to send when
            creating the connection
    """

    url: str
    method: str = "GET"
    request_options: _RequestOptions = field(default_factory=_RequestOptions)


class BaseSSEConsumer(ABC):
    """The base class for Server sent events consumers.

    This class handles reconnection after errors and debouncing as well as
    handling events in a non-blocking fashion.
    """

    def __init__(
        self,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        enabled_manager: Optional[EnabledManager] = None,
        background_coroutine_tracker: Optional[BackgroundCoroutineTracker] = None,
        client_session: Optional[ClientSession] = None,
    ):
        """Initialise the server sent events consumer.

        Args:
            loop: the event loop to run on. Defaults to the running loop if not
                provided.
            enabled_manager: The enabled manager for whether this should be
                consuming. Defaults to a new enabled manger if not provided.
            background_coroutine_tracker: The storage for background tasks
                scheduled on receiving events. Defaults to a new tracker if not
                provided.
            client_session: The http session to use. Defaults to a new session
                if not provided.
        """
        super().__init__()

        self.loop = loop or get_running_loop()
        self.enabled_manager = enabled_manager or EnabledManager(True, loop=self.loop)
        self.enabled_manager.add_listener(EnabledManager.DISABLED, self._on_close)
        self.client_session = client_session or ClientSession(
            timeout=ClientTimeout(sock_read=None, total=None), loop=self.loop
        )
        self.background_tracker = (
            background_coroutine_tracker or BackgroundCoroutineTracker(loop=self.loop)
        )
        self.signalling_stream_future = asyncio.run_coroutine_threadsafe(
            self._message_received_loop(), self.loop
        )

    @abstractmethod
    def get_sse_client_config(self) -> EventSourceConfig:
        """Used to configure the event client to consume events from the server.

        Returns:
            the configuration to be used to connect to the client
        """
        raise NotImplementedError("get_sse_client must be implemented")

    async def on_heartbeat(self) -> None:
        """An optional handler for heartbeat events."""
        pass

    async def on_end(self) -> None:
        """An optional handler for end events.

        If not overridden this method will disable the reconnection
        """
        self.enabled_manager.disable()

    @abstractmethod
    async def on_message(self, message_data: str) -> None:
        """The main handler for when the stream receives a message.

        Args:
            message_data: The raw string data of the message

        """
        raise NotImplementedError("on_message must be implemented")

    def _on_close(self) -> None:
        """Internal method to cleanup resources once the consumer has been closed."""
        self.background_tracker.stop_background_coroutines()
        self.signalling_stream_future.cancel()

    def close(self) -> None:
        """Disables the current enable manger, therefore cleaning up resources."""
        self.enabled_manager.disable()

    async def _message_received_loop(self) -> None:
        backoff_time = STREAMING_MINIMUM_BACKOFF_TIME_S

        while self.enabled_manager.is_enabled():
            try:
                await asyncio.sleep(backoff_time)
                backoff_time = min(STREAMING_MAXIMUM_BACKOFF_TIME_S, backoff_time * 2)
                event_source_config = self.get_sse_client_config()

                async with EventSource(
                    url=event_source_config.url,
                    option={"method": event_source_config.method},
                    session=self.client_session,
                    reconnection_time=timedelta(seconds=0.1),
                    **event_source_config.request_options,
                ) as event_source:
                    async for event in event_source:
                        backoff_time = max(
                            STREAMING_MINIMUM_BACKOFF_TIME_S, backoff_time / 2
                        )
                        if self.enabled_manager.is_disabled():
                            return
                        if event.type == "heartbeat":
                            self.background_tracker.submit_background_coroutine(
                                self.on_heartbeat()
                            )

                            continue

                        if event.type == "end":
                            self.background_tracker.submit_background_coroutine(
                                self.on_end()
                            )
                            if self.enabled_manager.is_disabled():
                                return
                            continue

                        if event.type == "data":
                            self.background_tracker.submit_background_coroutine(
                                self.on_message(event.data)
                            )
                            continue

                        logger.warning(
                            f'Unsupported SSE event type "{event.type}" received'
                        )
            except Exception as e:
                logger.warning(f"Streaming signalling error: {e}")
