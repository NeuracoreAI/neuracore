"""Robots register through the shared endpoint when the organization has regions."""

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import pytest_asyncio
from aiohttp import ClientResponseError, ClientSession

from neuracore.core.const import shared_signalling_url
from neuracore.core.streaming.p2p import signalling_events_consumer
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.signalling_events_consumer import (
    SignallingEventsConsumer,
)
from neuracore.core.streaming.p2p.signalling_session import (
    LowerEpochError,
    SignallingSession,
)
from neuracore.core.streaming.p2p.stream_manager_orchestrator import (
    StreamManagerOrchestrator,
)
from neuracore.core.utils.singleton_metaclass import SingletonMetaclass

ORG_ID = "org-1"
ROBOT_ID = "robot-1"
STREAM_ID = "robot-stream-1"


@pytest.mark.parametrize(
    ("api_url", "expected"),
    [
        ("https://api.neuracore.com/api", "https://signalling.neuracore.com"),
        (
            "https://staging.api.neuracore.com/api",
            "https://staging.signalling.neuracore.com",
        ),
        ("http://localhost:8000/api", None),
        ("https://example.com/api", None),
    ],
)
def test_shared_signalling_url_follows_the_api_host(
    api_url: str, expected: str | None
) -> None:
    assert shared_signalling_url(api_url) == expected


@pytest_asyncio.fixture
async def client_session() -> AsyncIterator[ClientSession]:
    async with ClientSession() as session:
        yield session


def _session(
    api_url: str, auth: Any, client_session: ClientSession, robot_instance: int | None
) -> SignallingSession:
    return SignallingSession(
        api_url=f"{api_url}/api",
        org_id=ORG_ID,
        auth=auth,
        client_session=client_session,
        robot_id=ROBOT_ID,
        robot_instance=robot_instance,
    )


def _consumer(
    session: SignallingSession, regions: list[str], client_session: ClientSession
) -> SignallingEventsConsumer:
    """A consumer with only the parts that the first connection uses."""
    consumer = SignallingEventsConsumer.__new__(SignallingEventsConsumer)
    consumer.client_session = client_session
    consumer.org_id = ORG_ID
    consumer.local_stream_id = STREAM_ID
    consumer.signalling_session = session
    consumer.enabled_manager = EnabledManager(True, loop=asyncio.get_running_loop())
    consumer._decided = asyncio.Event()
    consumer._failed_over = False

    async def list_regions() -> list[str]:
        return regions

    consumer.regions = list_regions
    return consumer


@pytest.mark.asyncio
async def test_first_connection_registers_through_the_shared_endpoint(
    backend: Any, region: Any, auth: Any, client_session: ClientSession
) -> None:
    """Given an organization with regions,
    when a robot's stream first connects,
    then it registers through the shared URL and streams from the region's
    canonical URL.
    """
    _, api_url = backend
    fake_region, region_url = region
    fake_region.registration_body["canonical_url"] = region_url
    consumer = _consumer(
        _session(api_url, auth, client_session, 0),
        ["https://other-region.example.com"],
        client_session,
    )

    with patch.object(signalling_events_consumer, "SIGNALLING_URL", region_url):
        config = await consumer.get_sse_client_config()

    assert [r.body["stream_id"] for r in fake_region.registrations] == [STREAM_ID]
    assert config.url == (
        f"{region_url}/org/{ORG_ID}/signalling/notifications/{STREAM_ID}"
    )


@pytest.mark.asyncio
async def test_first_region_is_used_without_a_shared_endpoint(
    backend: Any, region: Any, auth: Any, client_session: ClientSession
) -> None:
    _, api_url = backend
    fake_region, region_url = region
    fake_region.registration_body["canonical_url"] = region_url
    consumer = _consumer(
        _session(api_url, auth, client_session, 0), [region_url], client_session
    )

    with patch.object(signalling_events_consumer, "SIGNALLING_URL", None):
        await consumer.get_sse_client_config()

    assert len(fake_region.registrations) == 1


@pytest.mark.asyncio
async def test_organization_without_regions_stops_signalling(
    backend: Any, region: Any, auth: Any, client_session: ClientSession
) -> None:
    _, api_url = backend
    fake_region, region_url = region
    consumer = _consumer(_session(api_url, auth, client_session, 0), [], client_session)

    with (
        patch.object(signalling_events_consumer, "SIGNALLING_URL", region_url),
        pytest.raises(RuntimeError),
    ):
        await consumer.get_sse_client_config()

    assert fake_region.registrations == []
    assert consumer.enabled_manager.is_disabled()


@pytest.mark.asyncio
async def test_a_viewer_session_never_registers(
    backend: Any, region: Any, auth: Any, client_session: ClientSession
) -> None:
    _, api_url = backend
    fake_region, region_url = region
    consumer = _consumer(
        _session(api_url, auth, client_session, None), [region_url], client_session
    )

    with (
        patch.object(signalling_events_consumer, "SIGNALLING_URL", None),
        pytest.raises(RuntimeError),
    ):
        await consumer.get_sse_client_config()

    assert fake_region.registrations == []


@pytest.mark.asyncio
async def test_lower_epoch_on_first_registration_stops_signalling(
    backend: Any, region: Any, auth: Any, client_session: ClientSession
) -> None:
    _, api_url = backend
    fake_region, region_url = region
    fake_region.registration_status = 409
    fake_region.registration_body = {"detail": "lower_epoch"}
    consumer = _consumer(
        _session(api_url, auth, client_session, 0), [region_url], client_session
    )

    with (
        patch.object(signalling_events_consumer, "SIGNALLING_URL", None),
        pytest.raises(LowerEpochError),
    ):
        await consumer.get_sse_client_config()

    assert consumer.enabled_manager.is_disabled()


@pytest.fixture
def orchestrator() -> Any:
    """An orchestrator whose consumers and managers are recorded doubles."""
    consumers: list[MagicMock] = []

    def make_consumer(**kwargs: Any) -> MagicMock:
        consumer = MagicMock()
        consumer.local_stream_id = f"stream-{len(consumers)}"
        consumer.kwargs = kwargs
        consumers.append(consumer)
        return consumer

    module = "neuracore.core.streaming.p2p.stream_manager_orchestrator"
    with (
        patch.dict(SingletonMetaclass._instances, {}, clear=True),
        patch(f"{module}.SignallingEventsConsumer", side_effect=make_consumer),
        patch(f"{module}.ClientProviderStreamManager", side_effect=MagicMock),
        patch(f"{module}.ClientSession", MagicMock()),
    ):
        orchestrator = StreamManagerOrchestrator(
            org_id=ORG_ID, loop=MagicMock(), auth=MagicMock()
        )
        yield orchestrator, consumers


def test_each_provided_robot_instance_has_its_own_stream(orchestrator: Any) -> None:
    orchestrator, consumers = orchestrator

    first = orchestrator.get_provider_manager(ROBOT_ID, 0)
    second = orchestrator.get_provider_manager(ROBOT_ID, 1)

    robot_0, robot_1 = consumers
    assert robot_0.kwargs["signalling_session"].robot_instance == 0
    assert robot_1.kwargs["signalling_session"].robot_instance == 1
    assert robot_0.kwargs["regions"] == orchestrator.region_endpoints
    assert first.local_stream_id == robot_0.local_stream_id
    assert first.signalling == robot_0.endpoint
    assert second.local_stream_id == robot_1.local_stream_id
    assert first is orchestrator.get_provider_manager(ROBOT_ID, 0)
    assert first is not second


def test_removing_a_provider_closes_its_stream(orchestrator: Any) -> None:
    orchestrator, consumers = orchestrator
    orchestrator.get_provider_manager(ROBOT_ID, 0)

    orchestrator.remove_manager(ROBOT_ID, 0)

    consumers[0].close.assert_called_once()


@pytest.mark.asyncio
async def test_regions_are_asked_once(orchestrator: Any) -> None:
    orchestrator, _ = orchestrator
    orchestrator._org_session = MagicMock()

    async def headers() -> dict[str, str]:
        orchestrator._org_session.region_endpoints = ["https://a.example.com"]
        return {}

    orchestrator._org_session.headers = MagicMock(side_effect=headers)

    assert await orchestrator.region_endpoints() == ["https://a.example.com"]
    assert await orchestrator.region_endpoints() == ["https://a.example.com"]
    orchestrator._org_session.headers.assert_called_once()


@pytest.mark.asyncio
async def test_a_failed_registration_again_returns_to_the_shared_endpoint(
    backend: Any, region: Any, auth: Any, client_session: ClientSession
) -> None:
    """Given a robot registered in a region that now fails,
    when it registers again on reconnect,
    then the next connection registers through the shared URL.
    """
    _, api_url = backend
    fake_region, region_url = region
    fake_region.registration_status = 503
    fake_region.registration_body = {"detail": "redis_unavailable"}
    session = _session(api_url, auth, client_session, 0)
    session.base_url = region_url
    consumer = _consumer(session, [region_url], client_session)

    with pytest.raises(ClientResponseError):
        await consumer.on_reconnect()

    assert session.base_url is None


def _org_session_answering(orchestrator: Any, status: int) -> MagicMock:
    session = MagicMock()
    session.region_endpoints = []

    async def headers() -> dict[str, str]:
        raise ClientResponseError(MagicMock(), (), status=status)

    session.headers = MagicMock(side_effect=headers)
    orchestrator._org_session = session
    return session


@pytest.mark.asyncio
async def test_a_backend_without_the_token_route_has_no_regions(
    orchestrator: Any,
) -> None:
    orchestrator, _ = orchestrator
    _org_session_answering(orchestrator, 404)

    assert await orchestrator.region_endpoints() == []


@pytest.mark.asyncio
async def test_a_failed_region_request_is_asked_again(orchestrator: Any) -> None:
    orchestrator, _ = orchestrator
    session = _org_session_answering(orchestrator, 503)

    for _ in range(2):
        with pytest.raises(ClientResponseError):
            await orchestrator.region_endpoints()

    assert session.headers.call_count == 2


def test_a_stopped_robot_stream_removes_its_provider(orchestrator: Any) -> None:
    orchestrator, consumers = orchestrator
    orchestrator.get_provider_manager(ROBOT_ID, 0)
    robot_consumer = consumers[0]
    (_, on_disabled), _ = robot_consumer.enabled_manager.add_listener.call_args

    on_disabled()

    assert orchestrator.provider_managers == {}
    assert orchestrator.provider_consumers == {}


def test_streams_follow_their_live_data_settings() -> None:
    """A viewer's stream follows consume; a robot's stream follows provide."""
    module = "neuracore.core.streaming.p2p.stream_manager_orchestrator"
    consume, provide = MagicMock(name="consume"), MagicMock(name="provide")
    with (
        patch.dict(SingletonMetaclass._instances, {}, clear=True),
        patch(f"{module}.SignallingEventsConsumer", MagicMock()),
        patch(f"{module}.ClientProviderStreamManager", MagicMock()),
        patch(f"{module}.ClientSession", MagicMock()),
        patch(f"{module}.get_consume_live_data_enabled_manager", return_value=consume),
        patch(f"{module}.get_provide_live_data_enabled_manager", return_value=provide),
        patch(f"{module}.EnabledManager") as enabled_manager,
    ):
        orchestrator = StreamManagerOrchestrator(
            org_id=ORG_ID, loop=MagicMock(), auth=MagicMock()
        )
        orchestrator.get_provider_manager(ROBOT_ID, 0)
        orchestrator.viewer_consumer("https://region.example.com", ROBOT_ID)

    assert enabled_manager.derived_manger.call_args_list == [
        call(provide, loop=orchestrator.loop),
        call(consume, loop=orchestrator.loop),
    ]


def test_own_stream_ids_include_every_robot_stream(orchestrator: Any) -> None:
    orchestrator, _ = orchestrator
    orchestrator.get_provider_manager(ROBOT_ID, 0)
    orchestrator.get_provider_manager(ROBOT_ID, 1)

    assert orchestrator.own_stream_ids() == {"stream-0", "stream-1"}


def _stoppable(loop: Any) -> MagicMock:
    """A double whose close disables a real EnabledManager, as the real ones do."""
    double = MagicMock()
    double.enabled_manager = EnabledManager(True, loop=loop)
    double.close = MagicMock(side_effect=double.enabled_manager.disable)
    return double


def test_teardown_with_real_enabled_managers_finishes() -> None:
    """Closing streams and managers whose listeners close each other returns."""
    loop = asyncio.new_event_loop()
    module = "neuracore.core.streaming.p2p.stream_manager_orchestrator"
    consumers: list[MagicMock] = []

    def make_consumer(**_: Any) -> MagicMock:
        consumer = _stoppable(loop)
        consumer.local_stream_id = f"stream-{len(consumers)}"
        consumers.append(consumer)
        return consumer

    def make_manager(**_: Any) -> MagicMock:
        return _stoppable(loop)

    with (
        patch.dict(SingletonMetaclass._instances, {}, clear=True),
        patch(f"{module}.SignallingEventsConsumer", side_effect=make_consumer),
        patch(f"{module}.ClientProviderStreamManager", side_effect=make_manager),
        patch(f"{module}.ClientSession", MagicMock()),
    ):
        orchestrator = StreamManagerOrchestrator(
            org_id=ORG_ID, loop=MagicMock(), auth=MagicMock()
        )
        orchestrator.get_provider_manager(ROBOT_ID, 0)
        orchestrator.get_provider_manager(ROBOT_ID, 1)
        # A robot stream that stops by itself, as on a lower epoch.
        consumers[0].close()
        done = threading.Event()

        def close() -> None:
            orchestrator.close()
            done.set()

        threading.Thread(target=close, daemon=True).start()
        assert done.wait(2)

    assert orchestrator.provider_managers == {}
    assert orchestrator.provider_consumers == {}
    loop.close()


@pytest.mark.asyncio
async def test_a_failed_heartbeat_registration_keeps_the_region(
    backend: Any, region: Any, auth: Any, client_session: ClientSession
) -> None:
    """The stream is still open in its region, so its heartbeats stay there."""
    _, api_url = backend
    fake_region, region_url = region
    fake_region.registration_status = 503
    fake_region.registration_body = {"detail": "redis_unavailable"}
    session = _session(api_url, auth, client_session, 0)
    session.base_url = region_url
    consumer = _consumer(session, [region_url], client_session)

    with pytest.raises(ClientResponseError):
        await consumer._register_again()

    assert session.base_url == region_url


@pytest.mark.asyncio
async def test_robot_authentication_error_closes_only_that_stream(
    auth: Any, client_session: ClientSession
) -> None:
    """Given a robot stream and a viewer stream,
    when the robot's stream is refused,
    then only that stream closes; the viewer's refusal closes the orchestrator.
    """
    robot_session = SignallingSession(
        api_url="http://unused",
        org_id=ORG_ID,
        auth=auth,
        client_session=client_session,
        robot_id=ROBOT_ID,
        robot_instance=0,
    )
    robot_consumer = _consumer(robot_session, [], client_session)
    robot_consumer.manager_factory = MagicMock()

    robot_consumer.on_authentication_error()

    assert robot_consumer.enabled_manager.is_disabled()
    robot_consumer.manager_factory.close.assert_not_called()

    viewer_session = SignallingSession(
        api_url="http://unused",
        org_id=ORG_ID,
        auth=auth,
        client_session=client_session,
        robot_id=ROBOT_ID,
    )
    viewer_consumer = _consumer(viewer_session, [], client_session)
    viewer_consumer.manager_factory = MagicMock()

    viewer_consumer.on_authentication_error()

    viewer_consumer.manager_factory.close.assert_called_once_with()
