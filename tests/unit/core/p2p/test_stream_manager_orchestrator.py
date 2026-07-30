"""Tests for the global P2P stream orchestrator lifecycle."""

import asyncio
import threading
from unittest.mock import Mock, patch

from neuracore.core.streaming.p2p.stream_manager_orchestrator import (
    StreamManagerOrchestrator,
)
from neuracore.core.utils.singleton_metaclass import SingletonMetaclass


def test_close_streaming_resources_closes_and_forgets_all_managers():
    """Shutdown closes signalling and every manager exactly once."""
    orchestrator = object.__new__(StreamManagerOrchestrator)
    orchestrator.signalling_consumer = Mock()
    consumer_manager = Mock()
    provider_manager = Mock()
    orchestrator.consumer_managers = {"consumer": consumer_manager}
    orchestrator.provider_managers = {"provider": provider_manager}

    orchestrator._close_streaming_resources()

    orchestrator.signalling_consumer.close.assert_called_once_with()
    consumer_manager.close.assert_called_once_with()
    provider_manager.close.assert_called_once_with()
    assert orchestrator.consumer_managers == {}
    assert orchestrator.provider_managers == {}


def test_shutdown_global_forgets_singleton_so_it_can_be_recreated():
    """A later login/connect must not receive the closed orchestrator."""
    previous_instance = SingletonMetaclass._instances.pop(
        StreamManagerOrchestrator, None
    )
    try:
        with patch.object(StreamManagerOrchestrator, "__init__", return_value=None):
            first = StreamManagerOrchestrator()
            first.shutdown = Mock()

            StreamManagerOrchestrator.shutdown_global()
            second = StreamManagerOrchestrator()

        first.shutdown.assert_called_once_with()
        assert second is not first
    finally:
        SingletonMetaclass._instances.pop(StreamManagerOrchestrator, None)
        if previous_instance is not None:
            SingletonMetaclass._instances[StreamManagerOrchestrator] = previous_instance


def test_shutdown_waits_for_owned_client_session_to_close():
    """Synchronous shutdown drains loop cleanup before returning to logout."""
    loop = asyncio.new_event_loop()
    loop_started = threading.Event()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop_started.set()
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    loop_started.wait(timeout=1.0)

    class ClientSessionStub:
        closed = False

        async def close(self) -> None:
            self.closed = True

    orchestrator = object.__new__(StreamManagerOrchestrator)
    orchestrator.loop = loop
    orchestrator._owns_client_session = True
    orchestrator.client_session = ClientSessionStub()
    orchestrator.signalling_consumer = Mock()
    orchestrator.consumer_managers = {}
    orchestrator.provider_managers = {}

    try:
        orchestrator.shutdown()
        assert orchestrator.client_session.closed is True
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=1.0)
        loop.close()
