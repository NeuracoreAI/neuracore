"""Tests for ConnectionManager."""

from __future__ import annotations

import asyncio

import aiohttp
import pytest
import pytest_asyncio

from neuracore.data_daemon.connection_management.connection_manager import (
    ConnectionManager,
)
from neuracore.data_daemon.event_emitter import Emitter, get_emitter


@pytest_asyncio.fixture
async def client_session():
    """Create an aiohttp session for testing."""
    session = aiohttp.ClientSession()
    yield session
    await session.close()


@pytest_asyncio.fixture
async def manager(client_session: aiohttp.ClientSession) -> ConnectionManager:
    """Create a ConnectionManager instance for testing."""
    return ConnectionManager(
        client_session=client_session,
        timeout=2.0,
        check_interval=1.0,
    )


@pytest.mark.asyncio
async def test_connection_manager_initializes_correctly(
    manager: ConnectionManager,
) -> None:
    """Test that ConnectionManager initializes with correct defaults."""
    assert manager._stopped is False
    assert manager._connection_task is None


@pytest.mark.asyncio
async def test_connection_manager_start_stop(manager: ConnectionManager) -> None:
    """Test basic start and stop functionality."""
    # Start manager
    await manager.start()
    assert manager._stopped is False
    assert manager._connection_task is not None
    assert not manager._connection_task.done()

    await asyncio.sleep(0.5)

    # Stop manager
    await manager.stop()
    assert manager._stopped is True


@pytest.mark.asyncio
async def test_connection_manager_emits_events_on_state_change(
    client_session: aiohttp.ClientSession,
) -> None:
    """Test that events are emitted when connection state changes."""
    received: list[bool] = []

    async def handler(is_connected: bool) -> None:
        received.append(is_connected)

    get_emitter().on(Emitter.IS_CONNECTED, handler)
    try:
        manager = ConnectionManager(
            client_session=client_session,
            timeout=2.0,
            check_interval=0.5,
        )

        await manager.start()
        await asyncio.sleep(2)
        await manager.stop()

        assert len(received) > 0
    finally:
        get_emitter().remove_listener(Emitter.IS_CONNECTED, handler)


@pytest.mark.asyncio
async def test_connection_manager_tracks_state_changes(
    client_session: aiohttp.ClientSession,
) -> None:
    """Test that connection state changes are tracked correctly."""
    received: list[bool] = []

    async def handler(is_connected: bool) -> None:
        received.append(is_connected)

    get_emitter().on(Emitter.IS_CONNECTED, handler)
    try:
        manager = ConnectionManager(
            client_session=client_session,
            timeout=2.0,
            check_interval=0.3,
        )

        connection_states = [True, True, False, False, True]
        state_index = [0]

        async def mock_check_connectivity() -> bool:
            state = connection_states[state_index[0] % len(connection_states)]
            state_index[0] += 1
            return state

        manager._check_connectivity = mock_check_connectivity

        await manager.start()
        await asyncio.sleep(2)
        await manager.stop()

        assert len(received) >= 2

        assert True in received
        assert False in received
    finally:
        get_emitter().remove_listener(Emitter.IS_CONNECTED, handler)


@pytest.mark.asyncio
async def test_connection_manager_is_connected_method(
    manager: ConnectionManager,
) -> None:
    """Test the is_connected() method returns current state."""
    current_state = manager.is_connected()
    assert isinstance(current_state, bool)

    await manager.start()
    await asyncio.sleep(1.5)

    current_state = manager.is_connected()
    assert isinstance(current_state, bool)

    await manager.stop()


@pytest.mark.asyncio
async def test_connection_manager_double_start_is_safe(
    manager: ConnectionManager,
) -> None:
    """Test that calling start twice is handled gracefully."""
    await manager.start()
    assert manager._stopped is False

    await manager.start()
    assert manager._stopped is False

    await manager.stop()


@pytest.mark.asyncio
async def test_connection_manager_stop_without_start_is_safe(
    manager: ConnectionManager,
) -> None:
    """Test that calling stop without start is handled gracefully."""
    assert manager._connection_task is None

    await manager.stop()

    assert manager._stopped is True


@pytest.mark.asyncio
async def test_connection_manager_get_available_bandwidth_returns_none(
    manager: ConnectionManager,
) -> None:
    """Test that get_available_bandwidth returns None (placeholder)."""
    bandwidth = manager.get_available_bandwidth()
    assert bandwidth is None


@pytest.mark.asyncio
async def test_connection_manager_stops_thread_on_stop(
    manager: ConnectionManager,
) -> None:
    """Test that the checking thread actually stops."""
    await manager.start()

    task = manager._connection_task
    assert task is not None
    assert not task.done()

    await manager.stop()

    await asyncio.sleep(0.5)

    assert task.done() or task.cancelled()


@pytest.mark.asyncio
async def test_connection_manager_handles_check_exceptions(
    client_session: aiohttp.ClientSession,
) -> None:
    """Test that exceptions in connectivity check are handled gracefully."""
    received: list[bool] = []

    async def handler(is_connected: bool) -> None:
        received.append(is_connected)

    get_emitter().on(Emitter.IS_CONNECTED, handler)
    try:
        manager = ConnectionManager(
            client_session=client_session,
            timeout=2.0,
            check_interval=0.3,
        )

        check_count = [0]

        async def mock_check_that_raises() -> bool:
            check_count[0] += 1
            if check_count[0] == 2:
                raise RuntimeError("Test exception")
            return True

        manager._check_connectivity = mock_check_that_raises

        await manager.start()
        await asyncio.sleep(1.5)
        await manager.stop()

        assert check_count[0] >= 3
    finally:
        get_emitter().remove_listener(Emitter.IS_CONNECTED, handler)


@pytest.mark.asyncio
async def test_connection_manager_only_emits_on_state_change(
    client_session: aiohttp.ClientSession,
) -> None:
    """Test that events are only emitted when state actually changes."""
    received: list[bool] = []

    async def handler(is_connected: bool) -> None:
        received.append(is_connected)

    get_emitter().on(Emitter.IS_CONNECTED, handler)
    try:
        manager = ConnectionManager(
            client_session=client_session,
            timeout=2.0,
            check_interval=0.3,
        )

        async def mock_check_always_true() -> bool:
            return True

        manager._check_connectivity = mock_check_always_true

        await manager.start()
        await asyncio.sleep(1.5)
        await manager.stop()
        assert len(received) <= 2

        if len(received) > 1:
            assert all(received[1:])
    finally:
        get_emitter().remove_listener(Emitter.IS_CONNECTED, handler)
