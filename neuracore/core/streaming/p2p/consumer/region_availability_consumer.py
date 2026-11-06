"""One signalling region's robot list stream."""

import asyncio
from collections.abc import Callable

from aiohttp import ClientSession

from neuracore.core.streaming.base_sse_consumer import (
    BaseSSEConsumer,
    EventSourceConfig,
)
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.signalling_session import SignallingSession


class RegionAvailabilityConsumer(BaseSSEConsumer):
    """Streams one region's ``available_robots`` events to a callback."""

    def __init__(
        self,
        region_url: str,
        org_id: str,
        session: SignallingSession,
        on_event: Callable[[str, str], None],
        on_drop: Callable[[str], None],
        loop: asyncio.AbstractEventLoop,
        client_session: ClientSession,
        enabled_manager: EnabledManager,
    ) -> None:
        """Start streaming the region's list.

        Args:
            region_url: Canonical URL of the region.
            org_id: Organization whose robots to list.
            session: Organization session whose token the region accepts.
            on_event: Called with the region URL and each event's data.
            on_drop: Called with the region URL when the stream drops.
            loop: Event loop of the stream.
            client_session: HTTP session of the stream.
            enabled_manager: Whether the stream runs.
        """
        self.region_url = region_url
        self.org_id = org_id
        self.session = session
        self.on_event = on_event
        self.on_drop = on_drop
        super().__init__(
            loop=loop, enabled_manager=enabled_manager, client_session=client_session
        )

    def get_sse_client_config(self) -> EventSourceConfig:
        """Return the stream URL; ``prepare_sse_client_config`` adds the token."""
        return EventSourceConfig(
            url=f"{self.region_url}/org/{self.org_id}/signalling/available_robots"
        )

    async def prepare_sse_client_config(self) -> EventSourceConfig:
        """Return the stream URL with the organization token."""
        return EventSourceConfig(
            url=self.get_sse_client_config().url,
            request_options={"headers": await self.session.headers()},
        )

    async def on_reconnect(self) -> None:
        """Report the drop; the region's robots stay listed for a while."""
        self.on_drop(self.region_url)

    async def on_message(self, message_data: str) -> None:
        """Pass the event on."""
        self.on_event(self.region_url, message_data)
