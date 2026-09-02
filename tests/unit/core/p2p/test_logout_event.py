"""Tests for root streaming teardown after an explicit logout event."""

import asyncio
import threading
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest
from neuracore_types import RobotInstanceIdentifier
from pyee import EventEmitter

from neuracore.core.auth import Auth
from neuracore.core.exceptions import AuthenticationError
from neuracore.core.streaming import recording_state_manager as recording_module
from neuracore.core.streaming.p2p.consumer import org_nodes_manager as org_module
from neuracore.core.streaming.p2p.consumer.org_nodes_manager import OrgNodesManager
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.stream_manager_orchestrator import (
    StreamManagerOrchestrator,
)
from neuracore.core.streaming.recording_state_manager import (
    RecordingStateManager,
    TrackedRecording,
)
from neuracore.core.utils.singleton_metaclass import SingletonMetaclass


class FakeAuth(EventEmitter):
    """Minimal auth emitter used to isolate logout behavior."""

    def get_headers(self) -> dict[str, str]:
        return {}


class UnauthenticatedAuth(FakeAuth):
    """Auth double that reproduces a post-logout header request."""

    def get_headers(self) -> dict[str, str]:
        raise AuthenticationError("Not authenticated")


class FakeSession:
    """Client session double whose asynchronous closure can be observed."""

    def __init__(self) -> None:
        self.closed_event = threading.Event()

    async def close(self) -> None:
        self.closed_event.set()


class UnauthorizedEventSource:
    """SSE source double matching aiohttp-sse-client's HTTP 401 exception."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        raise ConnectionRefusedError("fetch https://stream.test failed: 401")

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        pass


@pytest.fixture
def nc_loop():
    """Run a dedicated streaming loop on a background thread."""
    loop = asyncio.new_event_loop()
    stopped = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()
        loop.close()
        stopped.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    stopped.wait(timeout=2)


def test_stream_orchestrator_tears_down_and_can_be_recreated(nc_loop) -> None:
    auth = FakeAuth()
    session = FakeSession()
    replacement_session = FakeSession()
    signalling_consumer = MagicMock()
    provider_manager = MagicMock()
    consumer_manager = MagicMock()

    with (
        patch.dict(SingletonMetaclass._instances, {}, clear=True),
        patch(
            "neuracore.core.streaming.p2p.stream_manager_orchestrator."
            "SignallingEventsConsumer",
            return_value=signalling_consumer,
        ),
        patch(
            "neuracore.core.streaming.p2p.stream_manager_orchestrator.ClientSession",
            side_effect=[session, replacement_session],
        ),
    ):
        manager = StreamManagerOrchestrator(
            org_id="org-1",
            loop=nc_loop,
            auth=auth,
        )
        manager.provider_managers[MagicMock()] = provider_manager
        manager.consumer_managers[MagicMock()] = consumer_manager

        auth.emit(Auth.LOGOUT_EVENT)

        replacement = StreamManagerOrchestrator(
            org_id="org-1",
            loop=nc_loop,
            auth=auth,
        )
        assert replacement is not manager

        assert session.closed_event.wait(timeout=2)
        signalling_consumer.close.assert_called_once_with()
        provider_manager.close.assert_called_once_with()
        consumer_manager.close.assert_called_once_with()
        assert manager._logout_listener not in auth.listeners(Auth.LOGOUT_EVENT)
        assert replacement._logout_listener in auth.listeners(Auth.LOGOUT_EVENT)

        nc_loop.call_soon_threadsafe(replacement.close)
        assert replacement_session.closed_event.wait(timeout=2)
        assert auth.listeners(Auth.LOGOUT_EVENT) == []


def test_org_nodes_manager_tears_down(nc_loop) -> None:
    auth = FakeAuth()
    session = FakeSession()
    enabled = EnabledManager(True, loop=nc_loop)
    consumer = MagicMock()
    with patch(
        "neuracore.core.streaming.base_sse_consumer.ClientSession",
        return_value=session,
    ):
        manager = OrgNodesManager(
            org_id="org-1",
            loop=nc_loop,
            enabled_manager=enabled,
            auth=auth,
            stream_manager_orchestrator=MagicMock(),
        )
    manager.consumers[MagicMock()] = consumer
    manager.connections[MagicMock()]["stream"] = ("connection", 1)

    manager_future: Future[OrgNodesManager] = Future()
    manager_future.set_result(manager)
    org_module._org_node_managers[manager.org_id] = manager_future

    auth.emit(Auth.LOGOUT_EVENT)
    assert manager.org_id not in org_module._org_node_managers
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), nc_loop).result(timeout=2)

    assert enabled.is_disabled()
    assert manager.signalling_stream_future.cancelled()
    assert session.closed_event.wait(timeout=2)
    consumer.close.assert_called_once_with()
    assert manager.connections == {}
    assert auth.listeners(Auth.LOGOUT_EVENT) == []


def test_sse_consumer_stops_on_authentication_error(nc_loop, caplog) -> None:
    auth = UnauthenticatedAuth()
    session = FakeSession()
    enabled = EnabledManager(True, loop=nc_loop)
    with patch(
        "neuracore.core.streaming.base_sse_consumer.ClientSession",
        return_value=session,
    ):
        manager = OrgNodesManager(
            org_id="org-1",
            loop=nc_loop,
            enabled_manager=enabled,
            auth=auth,
            stream_manager_orchestrator=MagicMock(),
        )

    async def wait_until_disabled() -> None:
        for _ in range(100):
            if enabled.is_disabled():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("SSE consumer did not stop after authentication failure")

    asyncio.run_coroutine_threadsafe(wait_until_disabled(), nc_loop).result(timeout=2)

    assert manager.signalling_stream_future.cancelled()
    assert session.closed_event.wait(timeout=2)
    assert not any(
        "Streaming signalling error" in record.message for record in caplog.records
    )


def test_signalling_http_401_tears_down_orchestrator(nc_loop, caplog) -> None:
    auth = FakeAuth()
    session = FakeSession()
    enabled = EnabledManager(True, loop=nc_loop)

    with (
        patch.dict(SingletonMetaclass._instances, {}, clear=True),
        patch.object(EnabledManager, "any_enabled", return_value=enabled),
        patch(
            "neuracore.core.streaming.p2p.stream_manager_orchestrator.ClientSession",
            return_value=session,
        ),
        patch(
            "neuracore.core.streaming.base_sse_consumer.EventSource",
            UnauthorizedEventSource,
        ),
    ):
        manager = StreamManagerOrchestrator(
            org_id="org-1",
            loop=nc_loop,
            auth=auth,
        )

        assert session.closed_event.wait(timeout=2)
        assert manager._closed
        assert manager._logout_listener not in auth.listeners(Auth.LOGOUT_EVENT)
        assert SingletonMetaclass._instances.get(StreamManagerOrchestrator) is None

    assert not any(
        "Streaming signalling error" in record.message for record in caplog.records
    )


def test_recording_manager_tears_down_owned_state(nc_loop) -> None:
    auth = FakeAuth()
    session = FakeSession()
    enabled = EnabledManager(True, loop=nc_loop)
    with patch(
        "neuracore.core.streaming.base_sse_consumer.ClientSession",
        return_value=session,
    ):
        manager = RecordingStateManager(
            org_id="org-1",
            loop=nc_loop,
            enabled_manager=enabled,
            auth=auth,
        )

    key = RobotInstanceIdentifier(robot_id="robot-1", robot_instance=0)
    manager.recording_robot_instances[key] = TrackedRecording("recording-1", 1.0)
    manager.active_dataset_ids[key] = "dataset-1"
    manager._drain_callbacks[key] = MagicMock()
    timer = MagicMock()
    manager._recording_timers["recording-1"] = [timer]

    manager_future: Future[RecordingStateManager] = Future()
    manager_future.set_result(manager)
    recording_module._recording_manager = manager_future

    auth.emit(Auth.LOGOUT_EVENT)
    assert recording_module._recording_manager is None

    assert session.closed_event.wait(timeout=2)
    assert enabled.is_disabled()
    assert manager.signalling_stream_future.cancelled()
    timer.cancel.assert_called_once_with()
    assert manager.recording_robot_instances == {}
    assert manager.active_dataset_ids == {}
    assert manager._drain_callbacks == {}
    assert auth.listeners(Auth.LOGOUT_EVENT) == []

    auth.emit(Auth.LOGOUT_EVENT)


def test_recording_manager_keeps_owned_state_when_the_stream_cannot_authenticate(
    nc_loop,
) -> None:
    """A rejected notification stream must not discard a local recording.

    `nc.stop_recording` reads this state to decide there is anything to stop, so
    clearing it on a stream failure leaves the stop unpublished and the daemon's
    window open. Contrast `test_recording_manager_tears_down_owned_state`, where
    an explicit logout clears it — that says the session is over, this does not.
    """
    auth = UnauthenticatedAuth()
    session = FakeSession()
    enabled = EnabledManager(True, loop=nc_loop)
    with patch(
        "neuracore.core.streaming.base_sse_consumer.ClientSession",
        return_value=session,
    ):
        manager = RecordingStateManager(
            org_id="org-1",
            loop=nc_loop,
            enabled_manager=enabled,
            auth=auth,
        )

    key = RobotInstanceIdentifier(robot_id="robot-1", robot_instance=0)
    manager.recording_robot_instances[key] = TrackedRecording("recording-1", 1.0)
    manager.active_dataset_ids[key] = "dataset-1"

    manager_future: Future[RecordingStateManager] = Future()
    manager_future.set_result(manager)
    recording_module._recording_manager = manager_future
    try:
        # The consumer loop returns once it gives up on the stream.
        manager.signalling_stream_future.result(timeout=2)

        assert manager.get_current_recording_id("robot-1", 0) == "recording-1"
        assert manager.is_recording("robot-1", 0)
        assert manager.active_dataset_ids[key] == "dataset-1"
        assert enabled.is_enabled()
        assert recording_module._recording_manager is manager_future

        # Logout is still the statement that clears it.
        auth.emit(Auth.LOGOUT_EVENT)
        assert session.closed_event.wait(timeout=2)
        assert manager.recording_robot_instances == {}
        assert recording_module._recording_manager is None
    finally:
        recording_module._recording_manager = None


@pytest.mark.parametrize(
    "manager_type",
    [StreamManagerOrchestrator, OrgNodesManager, RecordingStateManager],
)
def test_closed_loop_logout_does_not_leak_listener(manager_type) -> None:
    """One-shot listeners detach even when their manager loop is already closed."""
    auth = FakeAuth()
    manager = object.__new__(manager_type)
    manager.auth = auth
    manager.loop = MagicMock()
    manager.loop.is_closed.return_value = True

    if hasattr(manager, "_discard_cached_manager"):
        manager._discard_cached_manager = MagicMock()

    manager._logout_listener = manager._on_logout
    auth.once(Auth.LOGOUT_EVENT, manager._logout_listener)

    auth.emit(Auth.LOGOUT_EVENT)

    assert auth.listeners(Auth.LOGOUT_EVENT) == []
    manager.loop.call_soon_threadsafe.assert_not_called()
