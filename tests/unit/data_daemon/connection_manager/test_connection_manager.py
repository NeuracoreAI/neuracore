"""Tests for ConnectionManager."""

from __future__ import annotations

import time

import pytest

from neuracore.data_daemon.connection_management.connection_manager import (
    ConnectionManager,
)
from neuracore.data_daemon.event_emitter import Emitter, emitter


@pytest.fixture
def manager() -> ConnectionManager:
    """Create a ConnectionManager instance for testing."""
    return ConnectionManager(
        timeout=2.0,
        check_interval=1.0,
    )


def test_connection_manager_initializes_correctly(manager: ConnectionManager) -> None:
    """Test that ConnectionManager initializes with correct defaults."""
    assert manager._running is False
    assert manager._checker_thread is None


def test_connection_manager_start_stop(manager: ConnectionManager) -> None:
    """Test basic start and stop functionality."""
    # Start manager
    manager.start()
    assert manager._running is True
    assert manager._checker_thread is not None
    assert manager._checker_thread.is_alive()

    time.sleep(0.5)

    # Stop manager
    manager.stop()
    assert manager._running is False


def test_connection_manager_emits_events_on_state_change() -> None:
    """Test that events are emitted when connection state changes."""
    received: list[bool] = []

    def handler(is_connected: bool) -> None:
        received.append(is_connected)

    emitter.on(Emitter.IS_CONNECTED, handler)
    try:
        manager = ConnectionManager(
            timeout=2.0,
            check_interval=0.5,
        )

        manager.start()
        time.sleep(2)
        manager.stop()

        assert len(received) > 0
    finally:
        emitter.remove_listener(Emitter.IS_CONNECTED, handler)


def test_connection_manager_tracks_state_changes() -> None:
    """Test that connection state changes are tracked correctly."""
    received: list[bool] = []

    def handler(is_connected: bool) -> None:
        received.append(is_connected)

    emitter.on(Emitter.IS_CONNECTED, handler)
    try:
        manager = ConnectionManager(
            timeout=2.0,
            check_interval=0.3,
        )

        connection_states = [True, True, False, False, True]
        state_index = [0]

        def mock_check_connectivity() -> bool:
            state = connection_states[state_index[0] % len(connection_states)]
            state_index[0] += 1
            return state

        manager._check_connectivity = mock_check_connectivity

        manager.start()
        time.sleep(2)
        manager.stop()

        assert len(received) >= 2

        assert True in received
        assert False in received
    finally:
        emitter.remove_listener(Emitter.IS_CONNECTED, handler)


def test_connection_manager_is_connected_method(manager: ConnectionManager) -> None:
    """Test the is_connected() method returns current state."""
    current_state = manager.is_connected()
    assert isinstance(current_state, bool)

    manager.start()
    time.sleep(1.5)

    current_state = manager.is_connected()
    assert isinstance(current_state, bool)

    manager.stop()


def test_connection_manager_double_start_is_safe(manager: ConnectionManager) -> None:
    """Test that calling start twice is handled gracefully."""
    manager.start()
    assert manager._running is True

    manager.start()
    assert manager._running is True

    manager.stop()


def test_connection_manager_stop_without_start_is_safe(
    manager: ConnectionManager,
) -> None:
    """Test that calling stop without start is handled gracefully."""
    assert manager._running is False

    manager.stop()

    assert manager._running is False


def test_connection_manager_get_available_bandwidth_returns_none(
    manager: ConnectionManager,
) -> None:
    """Test that get_available_bandwidth returns None (placeholder)."""
    bandwidth = manager.get_available_bandwidth()
    assert bandwidth is None


def test_connection_manager_stops_thread_on_stop(manager: ConnectionManager) -> None:
    """Test that the checking thread actually stops."""
    manager.start()

    thread = manager._checker_thread
    assert thread is not None
    assert thread.is_alive()

    manager.stop()

    time.sleep(2)

    assert thread.is_alive() is False


def test_connection_manager_handles_check_exceptions() -> None:
    """Test that exceptions in connectivity check are handled gracefully."""
    received: list[bool] = []

    def handler(is_connected: bool) -> None:
        received.append(is_connected)

    emitter.on(Emitter.IS_CONNECTED, handler)
    try:
        manager = ConnectionManager(timeout=2.0, check_interval=0.3)

        check_count = [0]

        def mock_check_that_raises() -> bool:
            check_count[0] += 1
            if check_count[0] == 2:
                raise RuntimeError("Test exception")
            return True

        manager._check_connectivity = mock_check_that_raises

        manager.start()
        time.sleep(1.5)
        manager.stop()

        assert check_count[0] >= 3
    finally:
        emitter.remove_listener(Emitter.IS_CONNECTED, handler)


def test_connection_manager_only_emits_on_state_change() -> None:
    """Test that events are only emitted when state actually changes."""
    received: list[bool] = []

    def handler(is_connected: bool) -> None:
        received.append(is_connected)

    emitter.on(Emitter.IS_CONNECTED, handler)
    try:
        manager = ConnectionManager(timeout=2.0, check_interval=0.3)

        manager._check_connectivity = True

        manager.start()
        time.sleep(1.5)
        manager.stop()
        assert len(received) <= 2

        if len(received) > 1:
            assert all(received[1:])
    finally:
        emitter.remove_listener(Emitter.IS_CONNECTED, handler)
