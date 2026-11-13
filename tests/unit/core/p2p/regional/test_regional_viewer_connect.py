"""A Python viewer connects to a robot through the source that lists it."""

import asyncio
import json
from collections import defaultdict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from neuracore_types import (
    AvailableRobot,
    AvailableRobotCapacityInit,
    AvailableRobotInstance,
    MessageType,
    OpenConnectionRequest,
    RobotInstanceIdentifier,
    VideoFormat,
)

from neuracore.core.streaming.base_sse_consumer import (
    BaseSSEConsumer,
    EventSourceConfig,
)
from neuracore.core.streaming.p2p.consumer.client_consumer_stream_manager import (
    ClientConsumerStreamManager,
)
from neuracore.core.streaming.p2p.consumer.consumer_connection import (
    PeerToPeerConsumerConnection,
)
from neuracore.core.streaming.p2p.consumer.org_nodes_manager import OrgNodesManager
from neuracore.core.streaming.p2p.consumer.regional_availability import (
    RegionalAvailability,
)
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.signalling_endpoint import SignallingTarget
from neuracore.core.streaming.p2p.signalling_events_consumer import (
    SignallingEventsConsumer,
)
from neuracore.core.streaming.p2p.stream_manager_orchestrator import (
    StreamManagerOrchestrator,
)
from neuracore.core.utils.singleton_metaclass import SingletonMetaclass

ORG_ID = "org-1"
ROBOT_ID = "robot-1"
REGION_A = "https://region-a.example.com"
REGION_B = "https://region-b.example.com"
KEY = RobotInstanceIdentifier(robot_id=ROBOT_ID, robot_instance=0)


class FakeStream:
    """A signalling stream with an endpoint and connection routing."""

    def __init__(self, name: str) -> None:
        self.local_stream_id = f"{name}-stream"
        self.base_url = f"https://{name}.example.com/org/{ORG_ID}/signalling"
        self.created: list[Any] = []
        self.waited_open = False

    async def wait_open(self) -> None:
        self.waited_open = True

    async def endpoint(self) -> tuple[str, dict[str, str]]:
        return self.base_url, {"Authorization": self.local_stream_id}

    async def create_new_connection(self, message: Any, manager: Any) -> Any:
        self.created.append((message, manager))
        enabled = MagicMock()
        enabled.on = lambda _event: lambda handler: handler
        return enabled


def _context(response: Any) -> Any:
    """An async context manager that yields ``response``."""
    context = MagicMock()

    async def enter() -> Any:
        return response

    async def leave(*_: Any) -> None:
        return None

    context.__aenter__ = lambda _self: enter()
    context.__aexit__ = lambda _self, *a: leave()
    return context


class Answers:
    """A client session double: answers each connection post in turn."""

    def __init__(self, *answers: tuple[int, dict[str, Any]]) -> None:
        self.answers = list(answers)
        self.posts: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> Any:
        self.posts.append((url, json, headers))
        status, body = self.answers.pop(0)
        response = MagicMock(status=status)

        async def read(content_type: Any = None) -> dict[str, Any]:
            return body

        response.json = read
        return _context(response)


def _open_message(stream_id: str) -> dict[str, Any]:
    return {
        "from_id": "robot-stream",
        "to_id": stream_id,
        "data": json.dumps({}),
        "connection_id": "connection-1",
        "type": MessageType.OPEN_CONNECTION.value,
        "id": "message-1",
    }


def _manager(session: Answers, owners: list[Any]) -> tuple[OrgNodesManager, Any]:
    """A nodes manager whose list shows each owner in turn."""
    manager = OrgNodesManager.__new__(OrgNodesManager)
    manager.client_session = session
    manager.connections = defaultdict(dict)
    manager.consumers = {KEY: MagicMock()}
    manager.background_tracker = MagicMock()
    manager.availability = MagicMock(spec=RegionalAvailability)
    manager.availability.owner_of = MagicMock(side_effect=owners + owners[-1:] * 5)

    async def wait_for_update(*_: Any) -> bool:
        return True

    manager.availability.wait_for_update = wait_for_update
    orchestrator = MagicMock()
    viewers: dict[str, FakeStream] = {}
    orchestrator.viewer_consumer = lambda region, robot_id: viewers.setdefault(
        region, FakeStream(region.split("//")[1].split(".")[0])
    )
    manager.stream_manager_orchestrator = orchestrator
    return manager, viewers


def _request() -> OpenConnectionRequest:
    return OpenConnectionRequest(
        from_id="placeholder",
        to_id="robot-stream",
        robot_id=ROBOT_ID,
        robot_instance=0,
        video_format=VideoFormat.NEURACORE_CUSTOM,
    )


@pytest.mark.asyncio
async def test_a_regional_robot_is_connected_through_its_open_region_stream() -> None:
    session = Answers((200, _open_message("region-a-stream")))
    manager, viewers = _manager(session, [REGION_A])

    await manager._create_new_connection(1, MagicMock(), _request())

    viewer = viewers[REGION_A]
    ((url, body, headers),) = session.posts
    manager.availability.owner_of.assert_called_with(ROBOT_ID, 0, "robot-stream")
    assert viewer.waited_open
    assert url == f"{viewer.base_url}/connection?signature=temp"
    assert body["from_id"] == viewer.local_stream_id
    assert headers == {"Authorization": viewer.local_stream_id}
    assert len(viewer.created) == 1


@pytest.mark.asyncio
async def test_a_robot_that_moved_is_connected_where_the_list_shows_it() -> None:
    session = Answers(
        (409, {"detail": "not_present", "retryable": True}),
        (200, _open_message("region-b-stream")),
    )
    manager, viewers = _manager(session, [REGION_A, REGION_A, REGION_B])

    await manager._create_new_connection(1, MagicMock(), _request())

    assert [url.split("/org")[0] for url, _, _ in session.posts] == [
        "https://region-a.example.com",
        "https://region-b.example.com",
    ]
    assert viewers[REGION_A].created == []
    assert len(viewers[REGION_B].created) == 1


@pytest.mark.asyncio
async def test_a_robot_never_present_is_not_connected() -> None:
    not_present = (409, {"detail": "not_present", "retryable": True})
    session = Answers(not_present, not_present, not_present)
    manager, viewers = _manager(session, [REGION_A])

    await manager._create_new_connection(1, MagicMock(), _request())

    assert len(session.posts) == 3
    assert viewers[REGION_A].created == []


@pytest.mark.asyncio
async def test_an_unlisted_robot_is_not_connected() -> None:
    session = Answers()
    manager, _ = _manager(session, [None])

    await manager._create_new_connection(1, MagicMock(), _request())

    assert session.posts == []


@pytest.mark.asyncio
async def test_a_refused_connection_raises() -> None:
    session = Answers((403, {"detail": "forbidden"}))
    manager, _ = _manager(session, [REGION_A])

    with pytest.raises(RuntimeError):
        await manager._create_new_connection(1, MagicMock(), _request())


@pytest.fixture
def orchestrator() -> Any:
    module = "neuracore.core.streaming.p2p.stream_manager_orchestrator"
    streams: list[MagicMock] = []

    def make_stream(**kwargs: Any) -> MagicMock:
        stream = MagicMock()
        stream.local_stream_id = f"stream-{len(streams)}"
        stream.kwargs = kwargs
        stream.enabled_manager.is_disabled.return_value = False
        streams.append(stream)
        return stream

    with (
        patch.dict(SingletonMetaclass._instances, {}, clear=True),
        patch(f"{module}.SignallingEventsConsumer", side_effect=make_stream),
        patch(f"{module}.ClientSession", MagicMock()),
    ):
        yield StreamManagerOrchestrator(
            org_id=ORG_ID, loop=MagicMock(), auth=MagicMock()
        ), streams


def test_viewer_streams_are_one_per_region_and_robot(orchestrator: Any) -> None:
    orchestrator, _ = orchestrator

    first = orchestrator.viewer_consumer(REGION_A, ROBOT_ID)

    assert orchestrator.viewer_consumer(REGION_A, ROBOT_ID) is first
    assert orchestrator.viewer_consumer(REGION_B, ROBOT_ID) is not first
    session = first.kwargs["signalling_session"]
    assert (session.base_url, session.robot_id, session.robot_instance) == (
        REGION_A,
        ROBOT_ID,
        None,
    )
    assert first.local_stream_id in orchestrator.own_stream_ids()


def test_closing_closes_the_viewer_streams(orchestrator: Any) -> None:
    orchestrator, _ = orchestrator
    viewer = orchestrator.viewer_consumer(REGION_A, ROBOT_ID)

    orchestrator.close()

    viewer.close.assert_called_once()


def test_a_robots_viewer_streams_close_with_its_last_viewed_instance() -> None:
    manager = OrgNodesManager.__new__(OrgNodesManager)
    other = RobotInstanceIdentifier(robot_id=ROBOT_ID, robot_instance=1)
    manager.consumers = {KEY: MagicMock(), other: MagicMock()}
    manager.stream_manager_orchestrator = MagicMock()

    manager._remove_robot_consumer(ROBOT_ID, 0)
    manager.stream_manager_orchestrator.close_viewer_consumers.assert_not_called()
    manager._remove_robot_consumer(ROBOT_ID, 1)

    manager.stream_manager_orchestrator.close_viewer_consumers.assert_called_once_with(
        ROBOT_ID
    )


def test_track_information_leaves_out_this_process_streams() -> None:
    manager = OrgNodesManager.__new__(OrgNodesManager)
    manager.stream_manager_orchestrator = MagicMock(
        own_stream_ids=MagicMock(return_value={"own-robot-stream"})
    )
    manager.merged_update = AvailableRobotCapacityInit(
        robots=[
            AvailableRobot(
                robot_id=ROBOT_ID,
                instances={
                    0: AvailableRobotInstance(
                        robot_instance=0,
                        tracks={"own-robot-stream": [], "remote-stream": []},
                        connections=0,
                    )
                },
            )
        ]
    )

    instance = manager._get_last_robot_tracks(ROBOT_ID, 0)

    assert instance is not None
    assert list(instance.tracks) == ["remote-stream"]


class TurnServers:
    """A client session double that answers TURN requests, slowly if asked."""

    def __init__(self) -> None:
        self.gets: list[str] = []
        self.slow = asyncio.Event()
        self.slow.set()

    def get(self, url: str, headers: dict[str, str]) -> Any:
        self.gets.append(url)
        response = MagicMock()

        async def read() -> dict[str, Any]:
            await self.slow.wait()
            return {"iceServers": []}

        response.json = read
        return _context(response)


def _consumer_manager(session: Any) -> ClientConsumerStreamManager:
    manager = ClientConsumerStreamManager.__new__(ClientConsumerStreamManager)
    manager.local_stream_id = "legacy-stream"
    manager.signalling = FakeStream("legacy").endpoint
    manager.ice_configs = {}
    manager._current_track_information = None
    manager.client_session = session
    manager.org_id = ORG_ID
    manager.loop = asyncio.get_running_loop()
    manager.auth = MagicMock()
    manager.connections = {}
    return manager


@pytest.mark.asyncio
async def test_turn_servers_are_asked_once_per_signalling_stream() -> None:
    session = TurnServers()
    manager = _consumer_manager(session)
    viewer = FakeStream("region-a")
    target = SignallingTarget(viewer.local_stream_id, viewer.endpoint)

    await manager._get_ice_config(target)
    await manager._get_ice_config(target)
    await manager._get_ice_config(
        SignallingTarget(manager.local_stream_id, manager.signalling)
    )

    assert session.gets == [
        f"{viewer.base_url}/turn_servers/{viewer.local_stream_id}",
        f"https://legacy.example.com/org/{ORG_ID}/signalling/turn_servers/legacy-stream",
    ]


@pytest.mark.asyncio
async def test_concurrent_connections_keep_their_own_signalling_stream() -> None:
    """A connection waiting for TURN is not moved by another one."""
    session = TurnServers()
    session.slow.clear()
    manager = _consumer_manager(session)
    region_a, region_b = FakeStream("region-a"), FakeStream("region-b")
    module = "neuracore.core.streaming.p2p.consumer.client_consumer_stream_manager"

    with patch(f"{module}.PeerToPeerConsumerConnection") as connection_class:
        first = asyncio.ensure_future(
            manager.create_new_connection(
                "robot-stream",
                "c1",
                MagicMock(),
                target=SignallingTarget(region_a.local_stream_id, region_a.endpoint),
            )
        )
        await asyncio.sleep(0)
        second = asyncio.ensure_future(
            manager.create_new_connection(
                "robot-stream",
                "c2",
                MagicMock(),
                target=SignallingTarget(region_b.local_stream_id, region_b.endpoint),
            )
        )
        await asyncio.sleep(0)
        session.slow.set()
        await asyncio.gather(first, second)

    made = {
        call.kwargs["connection_id"]: (
            call.kwargs["local_stream_id"],
            call.kwargs["signalling"],
        )
        for call in connection_class.call_args_list
    }
    assert made == {
        "c1": (region_a.local_stream_id, region_a.endpoint),
        "c2": (region_b.local_stream_id, region_b.endpoint),
    }


@pytest.mark.asyncio
async def test_a_signalling_stream_signals_the_connections_it_routes() -> None:
    stream = SignallingEventsConsumer.__new__(SignallingEventsConsumer)
    stream.local_stream_id = "region-a-stream"
    stream.connection_managers = {}
    stream.message_queue = []
    stream.background_tracker = MagicMock()
    manager = MagicMock()
    manager.enabled_manager.is_disabled.return_value = False
    enabled = EnabledManager(True, loop=asyncio.get_running_loop())

    async def create_new_connection(**_: Any) -> EnabledManager:
        return enabled

    manager.create_new_connection = MagicMock(side_effect=create_new_connection)
    message = MagicMock(
        type=MessageType.OPEN_CONNECTION,
        data=json.dumps({
            "robot_id": ROBOT_ID,
            "robot_instance": 0,
            "video_format": VideoFormat.NEURACORE_CUSTOM.value,
            "connection_token": "token",
        }),
        from_id="robot-stream",
        connection_id="c1",
    )

    await stream.create_new_connection(message, manager)

    target = manager.create_new_connection.call_args.kwargs["target"]
    assert target == SignallingTarget("region-a-stream", stream.endpoint)


@pytest.mark.asyncio
async def test_a_stream_is_open_once_it_has_opened() -> None:
    stream = SignallingEventsConsumer.__new__(SignallingEventsConsumer)
    stream._opened = asyncio.Event()
    waiting = asyncio.ensure_future(stream.wait_open())
    await asyncio.sleep(0)
    assert not waiting.done()

    await stream.on_open()

    await asyncio.wait_for(waiting, timeout=1)


@pytest.mark.asyncio
async def test_viewer_messages_go_to_the_connections_stream() -> None:
    connection = PeerToPeerConsumerConnection.__new__(PeerToPeerConsumerConnection)
    viewer = FakeStream("region-a")
    posts: list[tuple[str, dict[str, str]]] = []

    class Session:
        def post(self, url: str, headers: dict[str, str], json: Any) -> Any:
            posts.append((url, headers))
            return _context(MagicMock())

    connection.client_session = Session()
    connection.signalling = viewer.endpoint
    connection.id = "c1"
    connection.local_stream_id = viewer.local_stream_id
    connection.remote_stream_id = "robot-stream"

    await connection.send_handshake_message(MessageType.SDP_ANSWER, "v=0")

    assert posts == [(
        f"{viewer.base_url}/message/submit",
        {"Authorization": viewer.local_stream_id},
    )]


class OpenStream:
    """An event source that opens and then waits for events that never come."""

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    async def __aenter__(self) -> "OpenStream":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def __aiter__(self) -> "OpenStream":
        return self

    async def __anext__(self) -> Any:
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_the_stream_loop_reports_each_open() -> None:
    opened = asyncio.Event()

    class Stream(BaseSSEConsumer):
        async def get_sse_client_config(self) -> EventSourceConfig:
            return EventSourceConfig(url="https://region-a.example.com/events")

        async def on_open(self) -> None:
            opened.set()

        async def on_message(self, message_data: str) -> None:
            return None

    with patch("neuracore.core.streaming.base_sse_consumer.EventSource", OpenStream):
        stream = Stream(loop=asyncio.get_running_loop(), client_session=MagicMock())
        await asyncio.wait_for(opened.wait(), timeout=2)
        stream.close()
