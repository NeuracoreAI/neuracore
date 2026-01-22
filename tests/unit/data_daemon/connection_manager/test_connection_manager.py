"""Tests for ConnectionManager."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from neuracore.data_daemon.connection_management.connection_manager import (
    ConnectionManager,
)
from neuracore.data_daemon.event_emitter import Emitter, emitter


class IsConnectedCapture:
    """Captures IS_CONNECTED events for testing."""

    def __init__(self) -> None:
        self.received: list[bool] = []

    def handler(self, is_connected: bool) -> None:
        self.received.append(is_connected)

    def __eq__(self, other: object) -> bool:
        return self.received == other


@pytest.fixture
def is_connected_capture():
    """Fixture that captures IS_CONNECTED events and cleans up after test."""
    capture = IsConnectedCapture()
    emitter.on(Emitter.IS_CONNECTED, capture.handler)
    yield capture
    emitter.remove_listener(Emitter.IS_CONNECTED, capture.handler)


@pytest.fixture
def manager_factory() -> Callable[..., ConnectionManager]:
    """Factory to create ConnectionManager instances with custom settings."""

    def _make(
        *, offline_mode: bool = False, timeout: float = 2.0, check_interval: float = 1.0
    ):
        return ConnectionManager(
            timeout=timeout,
            check_interval=check_interval,
            offline_mode=offline_mode,
        )

    return _make


def test_connection_manager_initializes_correctly(
    is_connected_capture, manager_factory
) -> None:
    """Test that ConnectionManager initializes with correct defaults."""
    manager = manager_factory()
    assert manager._running is False
    assert manager._checker_thread is None
    assert manager._offline_mode is False
    assert is_connected_capture == [False]


def test_connection_manager_initializes_offline_mode(
    is_connected_capture, manager_factory
) -> None:
    """Test that ConnectionManager in offline mode does not emit IS_CONNECTED."""
    manager = manager_factory(offline_mode=True)
    assert manager._running is False
    assert manager._checker_thread is None
    assert manager._offline_mode is True
    assert is_connected_capture == [False]


def test_connection_manager_offline_mode_start_does_nothing(manager_factory) -> None:
    """Test that start() in offline mode does not start the checker thread."""
    manager = manager_factory(offline_mode=True)
    manager.start()
    assert manager._running is False
    assert manager._checker_thread is None


def test_connection_manager_offline_mode_stop_does_nothing(manager_factory) -> None:
    """Test that stop() in offline mode does nothing."""
    manager = manager_factory(offline_mode=True)
    manager.stop()
    assert manager._running is False
    assert manager._checker_thread is None


def test_connection_manager_start_stop(manager_factory) -> None:
    """Test basic start and stop functionality."""
    manager = manager_factory()
    # Start manager
    manager.start()
    assert manager._running is True
    assert manager._checker_thread is not None
    assert manager._checker_thread.is_alive()

    time.sleep(0.5)

    # Stop manager
    manager.stop()
    assert manager._running is False


def test_connection_manager_emits_true_when_connected(
    is_connected_capture, manager_factory
) -> None:
    """Test that IS_CONNECTED event is emitted with True when connectivity succeeds."""
    manager = manager_factory(check_interval=0.1)

    manager._check_connectivity = lambda: True
    manager.start()
    time.sleep(0.5)

    assert is_connected_capture == [False, True]

    manager.stop()


def test_connection_manager_emits_false_when_disconnected(
    is_connected_capture, manager_factory
) -> None:
    """Test that IS_CONNECTED event is emitted with False when connectivity fails."""
    manager = manager_factory(check_interval=0.1)

    manager._check_connectivity = lambda: False
    manager.start()
    time.sleep(0.5)
    manager.stop()

    assert is_connected_capture == [False]


def test_connection_manager_offline_mode_never_emits_true(
    is_connected_capture, manager_factory
) -> None:
    """Test that offline mode never emits IS_CONNECTED as True."""
    manager = manager_factory(offline_mode=True, check_interval=0.1)

    manager._check_connectivity = lambda: True
    manager.start()
    time.sleep(0.5)
    manager.stop()

    assert is_connected_capture == [False]


def test_connection_manager_tracks_state_changes(
    is_connected_capture, manager_factory
) -> None:
    """Test that connection state changes are tracked correctly."""
    manager = manager_factory(check_interval=0.3)

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

    assert is_connected_capture == [False, True, False, True]


def test_connection_manager_is_connected_method(manager_factory) -> None:
    """Test the is_connected() method returns current state."""
    manager = manager_factory()
    current_state = manager.is_connected()
    assert isinstance(current_state, bool)

    manager.start()
    time.sleep(1.5)

    current_state = manager.is_connected()
    assert isinstance(current_state, bool)

    manager.stop()


def test_connection_manager_double_start_is_safe(manager_factory) -> None:
    """Test that calling start twice is handled gracefully."""
    manager = manager_factory()
    manager.start()
    assert manager._running is True

    manager.start()
    assert manager._running is True

    manager.stop()


def test_connection_manager_stop_without_start_is_safe(manager_factory) -> None:
    """Test that calling stop without start is handled gracefully."""
    manager = manager_factory()
    assert manager._running is False

    manager.stop()

    assert manager._running is False


def test_connection_manager_get_available_bandwidth_returns_none(
    manager_factory,
) -> None:
    """Test that get_available_bandwidth returns None (placeholder)."""
    manager = manager_factory()
    bandwidth = manager.get_available_bandwidth()
    assert bandwidth is None


def test_connection_manager_stops_thread_on_stop(manager_factory) -> None:
    """Test that the checking thread actually stops."""
    manager = manager_factory()
    manager.start()

    thread = manager._checker_thread
    assert thread is not None
    assert thread.is_alive()

    manager.stop(timeout=2.0)

    time.sleep(0.5)

    assert thread.is_alive() is False


def test_connection_manager_handles_check_exceptions(manager_factory) -> None:
    """Test that exceptions in connectivity check are handled gracefully."""
    manager = manager_factory(check_interval=0.3)

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
