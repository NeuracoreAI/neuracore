"""The client gets, renews and presents signalling tokens, and registers robots."""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import Any

import pytest
import pytest_asyncio
from aiohttp import ClientSession

from neuracore.core.exceptions import AuthenticationError
from neuracore.core.streaming.base_sse_consumer import (
    BaseSSEConsumer,
    EventSourceConfig,
)
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.signalling_events_consumer import (
    SignallingEventsConsumer,
)

pytestmark = pytest.mark.asyncio

ORG_ID = "org-1"
ROBOT_ID = "robot-1"
STREAM_ID = "robot-stream-1"
EVENT_SECONDS = 2.0


class Clock:
    """A UTC clock that tests move by hand."""

    def __init__(self, now: datetime) -> None:
        """Start at ``now``."""
        self.now = now

    def __call__(self) -> datetime:
        """Return the current time."""
        return self.now


@pytest_asyncio.fixture
async def client_session() -> AsyncIterator[ClientSession]:
    async with ClientSession() as session:
        yield session


@pytest.fixture
def clock(issued_at: datetime) -> Clock:
    return Clock(issued_at + timedelta(seconds=1))


@pytest.fixture
def robot_session(
    regional: Any,
    backend: Any,
    auth: Any,
    client_session: ClientSession,
    clock: Clock,
) -> Any:
    _, api_url = backend
    module = regional("signalling_session")
    return module.SignallingSession(
        api_url=f"{api_url}/api",
        org_id=ORG_ID,
        auth=auth,
        client_session=client_session,
        robot_id=ROBOT_ID,
        robot_instance=0,
        clock=clock,
    )


def _consumer(session: Any, client_session: ClientSession) -> SignallingEventsConsumer:
    """A consumer with only the parts that heartbeats use."""
    consumer = SignallingEventsConsumer.__new__(SignallingEventsConsumer)
    consumer.client_session = client_session
    consumer.org_id = ORG_ID
    consumer.local_stream_id = STREAM_ID
    consumer.connection_managers = {}
    consumer.signalling_session = session
    consumer.enabled_manager = EnabledManager(True, loop=asyncio.get_running_loop())
    consumer._decided = asyncio.Event()
    consumer._failed_over = False
    consumer._decided.set()
    return consumer


async def test_token_is_renewed_and_sent_on_heartbeat(
    robot_session: Any,
    backend: Any,
    region: Any,
    client_session: ClientSession,
    clock: Clock,
    issued_at: datetime,
) -> None:
    """Given a robot session registered in a region,
    when ``renew_after`` passes and the next heartbeat goes out,
    then the client got a new token from the backend, and the region
    received the new token on that heartbeat.
    """
    fake_backend, _ = backend
    fake_region, region_url = region
    await robot_session.register(region_url, STREAM_ID)
    consumer = _consumer(robot_session, client_session)
    await consumer.on_heartbeat()
    first = fake_region.heartbeats[-1].authorization

    clock.now = issued_at + timedelta(minutes=11)
    await consumer.on_heartbeat()

    renewed = fake_region.heartbeats[-1].authorization
    assert renewed != first
    assert renewed == f"Bearer token-{len(fake_backend.calls)}"
    assert "register_epoch" not in fake_backend.calls[-1].body


async def test_registration_requests_a_register_token(
    robot_session: Any, backend: Any, region: Any
) -> None:
    """Given a robot session for robot instance 0,
    when the robot registers in a region,
    then it asks the backend for a token with ``register_epoch: true`` and the
    robot instance, and presents that token to the region.
    """
    fake_backend, _ = backend
    fake_region, region_url = region

    result = await robot_session.register(region_url, STREAM_ID)

    assert fake_backend.calls[-1].body == {
        "robot_id": ROBOT_ID,
        "robot_instance": 0,
        "register_epoch": True,
    }
    registration = fake_region.registrations[-1]
    assert registration.authorization == f"Bearer token-{len(fake_backend.calls)}"
    assert registration.body == {
        "robot_id": ROBOT_ID,
        "robot_instance": 0,
        "stream_id": STREAM_ID,
    }
    assert result.canonical_url == region_url


async def test_headers_reuse_the_token_until_renewal(
    robot_session: Any, backend: Any, region: Any
) -> None:
    """Given a registered robot session,
    when it asks for headers before ``renew_after``,
    then it asks the backend for no token: its headers carry the token it has.
    """
    fake_backend, _ = backend
    _, region_url = region
    await robot_session.register(region_url, STREAM_ID)
    calls = len(fake_backend.calls)

    headers = await robot_session.headers()

    assert len(fake_backend.calls) == calls
    assert headers["Authorization"] == f"Bearer token-{calls}"


async def test_reconnect_registers_again(
    robot_session: Any,
    backend: Any,
    region: Any,
    client_session: ClientSession,
) -> None:
    """Given a registered robot session,
    when its event stream reconnects,
    then the robot registers again with a new register token.
    """
    fake_backend, _ = backend
    fake_region, region_url = region
    await robot_session.register(region_url, STREAM_ID)
    consumer = _consumer(robot_session, client_session)

    await consumer.on_reconnect()

    assert len(fake_region.registrations) == 2
    assert fake_backend.calls[-1].body["register_epoch"] is True


async def test_lower_epoch_rejection_is_reported(
    robot_session: Any, region: Any, regional: Any
) -> None:
    """Given a region that holds a higher epoch for the robot instance,
    when the robot registers,
    then ``register`` raises ``LowerEpochError``.
    """
    fake_region, region_url = region
    fake_region.registration_status = 409
    fake_region.registration_body = {"detail": "lower_epoch"}

    with pytest.raises(regional("signalling_session").LowerEpochError):
        await robot_session.register(region_url, STREAM_ID)


class _RecordingConsumer(BaseSSEConsumer):
    """A consumer of the fake region's event stream."""

    def __init__(self, url: str, closes: list[str], **kwargs: Any) -> None:
        self._url = url
        self.closes = closes
        self.configs = 0
        self.reconnects = 0
        super().__init__(**kwargs)

    async def get_sse_client_config(self) -> EventSourceConfig:
        self.configs += 1
        return EventSourceConfig(url=self._url)

    async def on_reconnect(self) -> None:
        self.reconnects += 1

    async def on_message(self, message_data: str) -> None:
        pass

    async def on_close(self, reason: str) -> None:
        self.closes.append(reason)


async def test_close_event_reconnects_and_end_event_stops(
    region: Any, client_session: ClientSession
) -> None:
    """Given a stream that sends ``close`` with reason ``expired``,
    when the event arrives,
    then the client reports the reason and opens the stream again; and after
    an ``end`` event with reason ``revoked``, it stops.
    """
    fake_region, region_url = region
    url = f"{region_url}/org/{ORG_ID}/signalling/notifications/{STREAM_ID}"
    fake_region.events = [("close", {"reason": "expired"})]
    # The region keeps the stream open: only the close event can end it.
    fake_region.hold_open = True
    closes: list[str] = []
    consumer = _RecordingConsumer(
        url, closes, loop=asyncio.get_running_loop(), client_session=client_session
    )
    try:

        async def reopened() -> None:
            while fake_region.event_streams < 2:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(reopened(), EVENT_SECONDS)
        assert closes[0] == "expired"
        assert consumer.enabled_manager.is_enabled()

        fake_region.hold_open = False
        fake_region.events = [("end", {"reason": "revoked"})]

        async def stopped() -> None:
            while consumer.enabled_manager.is_enabled():
                await asyncio.sleep(0.05)

        await asyncio.wait_for(stopped(), EVENT_SECONDS * 4)
    finally:
        consumer.close()


async def test_resurrected_heartbeat_registers_again(
    robot_session: Any,
    backend: Any,
    region: Any,
    client_session: ClientSession,
) -> None:
    """Given a registered robot whose region lost its state,
    when a heartbeat answers ``resurrected: true``,
    then the robot registers again in that region with a new register token.
    """
    fake_backend, _ = backend
    fake_region, region_url = region
    await robot_session.register(region_url, STREAM_ID)
    fake_region.resurrected = True
    consumer = _consumer(robot_session, client_session)

    await consumer.on_heartbeat()

    assert len(fake_region.registrations) == 2
    assert fake_backend.calls[-1].body["register_epoch"] is True


async def test_stream_end_reconnects_through_the_consumer(
    region: Any, client_session: ClientSession
) -> None:
    """Given a region that ends each stream with no event,
    when the stream ends,
    then the consumer reconnects with a new config, after ``on_reconnect``.
    """
    fake_region, region_url = region
    url = f"{region_url}/org/{ORG_ID}/signalling/notifications/{STREAM_ID}"
    consumer = _RecordingConsumer(
        url, [], loop=asyncio.get_running_loop(), client_session=client_session
    )
    try:

        async def reopened() -> None:
            while fake_region.event_streams < 3:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(reopened(), EVENT_SECONDS)
        assert consumer.reconnects >= 2
        assert consumer.configs >= 3
    finally:
        consumer.close()


async def test_lower_epoch_on_reconnect_stops_signalling(
    robot_session: Any, region: Any, client_session: ClientSession
) -> None:
    """Given a registered robot whose stream was registered at a newer epoch since,
    when the robot reconnects and the region answers ``lower_epoch``,
    then the consumer stops instead of registering with a newer epoch.
    """
    fake_region, region_url = region
    await robot_session.register(region_url, STREAM_ID)
    consumer = _consumer(robot_session, client_session)
    fake_region.registration_status = 409
    fake_region.registration_body = {"detail": "lower_epoch"}

    await consumer.on_reconnect()

    assert consumer.enabled_manager.is_disabled()


async def test_refused_api_key_raises_authentication_error(
    robot_session: Any, backend: Any
) -> None:
    """Given a backend that refuses the API key,
    when the session asks for headers,
    then it raises ``AuthenticationError``, which ends the consumer.
    """
    fake_backend, _ = backend
    fake_backend.status = 401

    with pytest.raises(AuthenticationError):
        await robot_session.headers()


async def test_concurrent_renewals_request_one_token(
    robot_session: Any, backend: Any, clock: Clock, issued_at: datetime
) -> None:
    """Given a session past ``renew_after``,
    when two callers ask for headers at once,
    then the session asks the backend for one token.
    """
    fake_backend, _ = backend
    await robot_session.headers()
    clock.now = issued_at + timedelta(minutes=11)
    fake_backend.issued_at = clock.now

    await asyncio.gather(robot_session.headers(), robot_session.headers())

    assert len(fake_backend.calls) == 2
