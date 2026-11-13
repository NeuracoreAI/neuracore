"""A robot's track and message calls go where its stream registered."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from aiohttp import ClientResponseError, ClientSession
from neuracore_types import DataType, MessageType, RobotStreamTrack

from neuracore.core.streaming.p2p import signalling_events_consumer
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.provider.client_provider_stream_manager import (
    ClientProviderStreamManager,
)
from neuracore.core.streaming.p2p.provider.provider_connection import (
    PeerToPeerProviderConnection,
)
from neuracore.core.streaming.p2p.signalling_events_consumer import (
    SignallingEventsConsumer,
)
from neuracore.core.streaming.p2p.signalling_session import SignallingSession

ORG_ID = "org-1"
ROBOT_ID = "robot-1"
STREAM_ID = "robot-stream-1"


@pytest_asyncio.fixture
async def client_session() -> AsyncIterator[ClientSession]:
    async with ClientSession() as session:
        yield session


def _consumer(
    session: SignallingSession | None, regions: list[str], auth: Any
) -> SignallingEventsConsumer:
    """A consumer with only the parts that the first connection uses."""
    consumer = SignallingEventsConsumer.__new__(SignallingEventsConsumer)
    consumer.org_id = ORG_ID
    consumer.local_stream_id = STREAM_ID
    consumer.signalling_session = session
    consumer.enabled_manager = EnabledManager(True, loop=asyncio.get_running_loop())
    consumer._decided = asyncio.Event()
    consumer.enabled_manager.add_listener(
        EnabledManager.DISABLED, consumer._decided.set
    )
    consumer._failed_over = False

    async def list_regions() -> list[str]:
        return regions

    consumer.regions = list_regions
    return consumer


def _robot_session(api_url: str, auth: Any, client_session: ClientSession) -> Any:
    return SignallingSession(
        api_url=f"{api_url}/api",
        org_id=ORG_ID,
        auth=auth,
        client_session=client_session,
        robot_id=ROBOT_ID,
        robot_instance=0,
    )


@pytest.mark.asyncio
async def test_endpoint_waits_for_registration_then_names_the_region(
    backend: Any, region: Any, auth: Any, client_session: ClientSession
) -> None:
    _, api_url = backend
    fake_region, region_url = region
    fake_region.registration_body["canonical_url"] = region_url
    consumer = _consumer(
        _robot_session(api_url, auth, client_session), [region_url], auth
    )
    endpoint = asyncio.ensure_future(consumer.endpoint())
    await asyncio.sleep(0)
    assert not endpoint.done()

    with patch.object(signalling_events_consumer, "SIGNALLING_URL", None):
        await consumer.get_sse_client_config()
    base_url, headers = await endpoint

    assert base_url == f"{region_url}/org/{ORG_ID}/signalling"
    assert headers["Authorization"].startswith("Bearer token-")


@pytest.mark.asyncio
async def test_endpoint_fails_without_regions(
    backend: Any, auth: Any, client_session: ClientSession
) -> None:
    _, api_url = backend
    consumer = _consumer(_robot_session(api_url, auth, client_session), [], auth)

    with pytest.raises(RuntimeError):
        await consumer.get_sse_client_config()

    with pytest.raises(RuntimeError):
        await asyncio.wait_for(consumer.endpoint(), timeout=1)


@pytest.mark.asyncio
async def test_endpoint_fails_once_the_stream_closed_undecided(
    auth: Any, client_session: ClientSession
) -> None:
    async def no_stream(self: Any) -> None:
        return None

    with patch.object(SignallingEventsConsumer, "_message_received_loop", no_stream):
        consumer = SignallingEventsConsumer(
            manager_factory=Any,
            org_id=ORG_ID,
            loop=asyncio.get_running_loop(),
            client_session=client_session,
            signalling_session=MagicMock(),
        )
    consumer.close()

    with pytest.raises(RuntimeError):
        await asyncio.wait_for(consumer.endpoint(), timeout=1)


class _Response:
    """A response that can be awaited or used as a context manager."""

    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __await__(self) -> Any:
        async def itself() -> "_Response":
            return self

        return itself().__await__()

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise ClientResponseError(MagicMock(), (), status=self.status)


class Recorder:
    """A client session double that records posts."""

    def __init__(self, status: int = 200) -> None:
        self.posts: list[tuple[str, dict[str, str]]] = []
        self.status = status

    def post(self, url: str, headers: dict[str, str], **_: Any) -> _Response:
        self.posts.append((url, headers))
        return _Response(self.status)


async def _region_endpoint() -> tuple[str, dict[str, str]]:
    return "https://region.example.com/org/org-1/signalling", {"Authorization": "x"}


@pytest.mark.asyncio
async def test_tracks_go_to_the_robots_endpoint() -> None:
    manager = ClientProviderStreamManager.__new__(ClientProviderStreamManager)
    manager.client_session = Recorder()
    manager.signalling = _region_endpoint
    track = RobotStreamTrack(
        robot_id=ROBOT_ID,
        robot_instance=0,
        stream_id=STREAM_ID,
        data_type=DataType.RGB_IMAGES,
        label="front",
        mid="0",
    )

    await manager._post_track_batch([track])

    assert manager.client_session.posts == [(
        "https://region.example.com/org/org-1/signalling/track/batch",
        {"Authorization": "x"},
    )]


@pytest.mark.asyncio
async def test_messages_go_to_the_robots_endpoint() -> None:
    connection = PeerToPeerProviderConnection.__new__(PeerToPeerProviderConnection)
    connection.client_session = Recorder()
    connection.signalling = _region_endpoint
    connection.id = "connection-1"
    connection.local_stream_id = STREAM_ID
    connection.remote_stream_id = "viewer-1"

    await connection.send_handshake_message(MessageType.SDP_OFFER, "v=0")

    assert connection.client_session.posts == [(
        "https://region.example.com/org/org-1/signalling/message/submit",
        {"Authorization": "x"},
    )]


@pytest.mark.asyncio
async def test_new_connections_use_the_robots_endpoint() -> None:
    manager = ClientProviderStreamManager.__new__(ClientProviderStreamManager)
    manager.signalling = _region_endpoint
    manager.local_stream_id = STREAM_ID
    manager.org_id = ORG_ID
    manager.client_session = Recorder()
    manager.auth = None
    manager.loop = asyncio.get_running_loop()
    manager.tracks = []
    manager.event_source_cache = {}
    manager.connections = {}
    manager.background_tracker = MagicMock()
    module = "neuracore.core.streaming.p2p.provider.client_provider_stream_manager"

    with patch(f"{module}.PeerToPeerProviderConnection") as connection_class:
        await manager.create_new_connection(
            "viewer-1", "connection-1", MagicMock(), target=MagicMock()
        )

    assert connection_class.call_args.kwargs["signalling"] is _region_endpoint


def _track() -> RobotStreamTrack:
    return RobotStreamTrack(
        robot_id=ROBOT_ID,
        robot_instance=0,
        stream_id=STREAM_ID,
        data_type=DataType.RGB_IMAGES,
        label="front",
        mid="0",
    )


@pytest.mark.asyncio
async def test_a_refused_track_batch_raises() -> None:
    manager = ClientProviderStreamManager.__new__(ClientProviderStreamManager)
    manager.client_session = Recorder(status=409)
    manager.signalling = _region_endpoint

    with pytest.raises(ClientResponseError):
        await manager._post_track_batch([_track()])


@pytest.mark.asyncio
async def test_calls_wait_through_a_failover_and_tracks_move_with_it(
    backend: Any, region: Any, auth: Any, client_session: ClientSession
) -> None:
    """Given a robot registered in a region that fails on reconnect,
    when it registers again through the shared URL,
    then calls made in between wait for the new region, and the robot's
    tracks are submitted there.
    """
    _, api_url = backend
    fake_region, region_url = region
    session = _robot_session(api_url, auth, client_session)
    session.base_url = "https://failed-region.example.com"
    consumer = _consumer(session, [region_url], auth)
    consumer._decided.set()
    manager = MagicMock()
    resubmitted = asyncio.Event()

    async def on_stream_resurrected() -> None:
        resubmitted.set()

    manager.on_stream_resurrected = on_stream_resurrected
    consumer.manager_factory = MagicMock(get_manager=MagicMock(return_value=manager))
    consumer.background_tracker = MagicMock(
        submit_background_coroutine=asyncio.ensure_future
    )

    with pytest.raises(Exception):
        await consumer.on_reconnect()
    waiting = asyncio.ensure_future(consumer.endpoint())
    await asyncio.sleep(0)
    assert not waiting.done()

    fake_region.registration_body["canonical_url"] = region_url
    with patch.object(signalling_events_consumer, "SIGNALLING_URL", None):
        await consumer.get_sse_client_config()

    base_url, _ = await asyncio.wait_for(waiting, timeout=1)
    assert base_url == f"{region_url}/org/{ORG_ID}/signalling"
    await asyncio.wait_for(resubmitted.wait(), timeout=1)
