"""Tests for provider connection event loop and cross-thread behavior."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest
from neuracore_types import OpenConnectionDetails, VideoFormat

from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.provider.json_source import JSONSource
from neuracore.core.streaming.p2p.provider.provider_connection import (
    PeerToPeerProviderConnection,
)
from neuracore.core.streaming.p2p.provider.video_source import VideoSource


def _make_connection_details() -> OpenConnectionDetails:
    return OpenConnectionDetails(
        connection_token="test-token",
        robot_id="test-robot",
        robot_instance=0,
        video_format=VideoFormat.NEURACORE_CUSTOM,
    )


@pytest.fixture
def nc_loop():
    """Event loop that simulates the neuracore async loop (runs in background)."""
    loop = asyncio.new_event_loop()
    done = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:
                pass
        done.set()

    t = threading.Thread(target=run, name="nc-async-loop", daemon=True)
    t.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    done.wait(timeout=2.0)


@pytest.fixture
def provider_connection(nc_loop):
    """Minimal provider connection using the nc_loop, with auth/org mocked."""
    with (
        patch(
            "neuracore.core.streaming.p2p.provider.provider_connection.get_current_org",
            return_value="test-org",
        ),
        patch(
            "neuracore.core.streaming.p2p.provider.provider_connection.get_auth",
            return_value=MagicMock(get_headers=lambda: {}),
        ),
    ):
        conn = PeerToPeerProviderConnection(
            connection_id="conn-1",
            local_stream_id="local-1",
            remote_stream_id="remote-1",
            connection_details=_make_connection_details(),
            client_session=None,
            org_id="test-org",
            loop=nc_loop,
            enabled_manager=EnabledManager(True, loop=nc_loop),
        )
        yield conn


@pytest.fixture
def json_source(nc_loop):
    """JSONSource that uses nc_loop (so it can be used from any thread)."""
    enabled = EnabledManager(True, loop=nc_loop)
    return JSONSource(mid="test-mid", stream_enabled=enabled, loop=nc_loop)


@pytest.fixture
def video_source(nc_loop):
    """VideoSource with streaming enabled, using nc_loop."""
    enabled = EnabledManager(True, loop=nc_loop)
    return VideoSource(stream_enabled=enabled, mid="video-0")


def test_caller_event_loop_not_overwritten_when_adding_event_source(
    provider_connection, json_source, nc_loop
):
    """
    Regression: calling add_event_source from a thread with its own loop must
    not overwrite that thread's event loop (no set_event_loop in provider).
    """
    main_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(main_loop)
        provider_connection.add_event_source(json_source)
        current = asyncio.get_event_loop()
        assert (
            current is main_loop
        ), "Calling thread's event loop must not be overwritten"
    finally:
        asyncio.set_event_loop(None)
        main_loop.close()


def test_caller_event_loop_not_overwritten_when_adding_video_source(
    provider_connection, video_source, nc_loop
):
    """
    Regression: calling add_video_source from a thread with its own loop must
    not overwrite that thread's event loop (no set_event_loop in provider).
    """
    main_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(main_loop)
        provider_connection.add_video_source(video_source)
        current = asyncio.get_event_loop()
        assert (
            current is main_loop
        ), "Calling thread's event loop must not be overwritten"
    finally:
        asyncio.set_event_loop(None)
        main_loop.close()


def test_add_event_source_from_thread_with_no_loop(
    provider_connection, json_source, nc_loop
):
    """
    add_event_source called from a thread that has no event loop must not
    raise (e.g. RuntimeError: no current event loop in thread).
    """
    result = {"done": False, "error": None}

    def from_thread_with_no_loop():
        try:
            provider_connection.add_event_source(json_source)
            result["done"] = True
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=from_thread_with_no_loop, name="no-loop-thread")
    t.start()
    t.join(timeout=5.0)
    assert t.is_alive() is False, "Thread should finish"
    assert (
        result["error"] is None
    ), f"add_event_source must not raise: {result['error']}"
    assert result["done"] is True


def test_add_event_source_from_main_with_own_loop(
    provider_connection, json_source, nc_loop
):
    """
    add_event_source from a thread that has its own (different) loop must not
    crash and must not overwrite that thread's loop.
    """
    main_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(main_loop)
        provider_connection.add_event_source(json_source)
        current = asyncio.get_event_loop()
        assert current is main_loop
    finally:
        asyncio.set_event_loop(None)
        main_loop.close()


def test_add_video_source_from_thread_with_no_loop(
    provider_connection, video_source, nc_loop
):
    """
    add_video_source from a thread with no event loop must not crash.
    """
    result = {"done": False, "error": None}

    def from_thread():
        try:
            provider_connection.add_video_source(video_source)
            result["done"] = True
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=from_thread, name="no-loop-thread")
    t.start()
    t.join(timeout=5.0)
    assert t.is_alive() is False
    assert (
        result["error"] is None
    ), f"add_video_source must not raise: {result['error']}"
    assert result["done"] is True


def test_add_video_source_from_thread_with_own_loop(
    provider_connection, video_source, nc_loop
):
    """
    add_video_source from a thread with its own loop must not overwrite
    that thread's event loop.
    """
    main_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(main_loop)
        provider_connection.add_video_source(video_source)
        current = asyncio.get_event_loop()
        assert current is main_loop
    finally:
        asyncio.set_event_loop(None)
        main_loop.close()


@pytest.mark.asyncio
async def test_add_video_and_event_source_from_connection_loop(
    provider_connection, video_source, json_source, nc_loop
):
    """
    Call add_video_source and add_event_source from a coroutine on the
    connection's loop (same-loop). Must complete without deadlock; connection
    schedules the work and send_offer (or awaiting _add_track_tasks) waits
    for it. Regression for create_new_connection calling add_* with no await.
    """

    async def add_sources_on_connection_loop():
        provider_connection.add_video_source(video_source)
        provider_connection.add_event_source(json_source)
        if provider_connection._add_track_tasks:
            await asyncio.gather(*provider_connection._add_track_tasks)

    future = asyncio.run_coroutine_threadsafe(add_sources_on_connection_loop(), nc_loop)
    await asyncio.wait_for(asyncio.wrap_future(future), timeout=2.0)

    senders = provider_connection.connection.getSenders()
    assert len(senders) >= 1, "Video track should be added to peer connection"
    assert json_source in provider_connection.event_sources


@pytest.mark.asyncio
async def test_connection_uses_provided_loop(nc_loop):
    """
    Connection must use the loop passed in, not the calling thread's loop.
    """
    with (
        patch(
            "neuracore.core.streaming.p2p.provider.provider_connection.get_current_org",
            return_value="test-org",
        ),
        patch(
            "neuracore.core.streaming.p2p.provider.provider_connection.get_auth",
            return_value=MagicMock(get_headers=lambda: {}),
        ),
    ):
        conn = PeerToPeerProviderConnection(
            connection_id="c1",
            local_stream_id="l1",
            remote_stream_id="r1",
            connection_details=_make_connection_details(),
            client_session=None,
            loop=nc_loop,
        )
    assert conn.loop is nc_loop
