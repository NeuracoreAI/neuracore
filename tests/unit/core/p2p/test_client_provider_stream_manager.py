"""Tests for the track batching in the client provider stream manager."""

import asyncio
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from neuracore_types import DataType, RobotStreamTrack

from neuracore.core.const import STREAM_API_URL
from neuracore.core.streaming.p2p.enabled_manager import EnabledManager
from neuracore.core.streaming.p2p.provider import client_provider_stream_manager
from neuracore.core.streaming.p2p.provider.client_provider_stream_manager import (
    TRACK_BATCH_MAX_SIZE,
    TRACK_BATCH_MAX_WAIT_S,
    ClientProviderStreamManager,
)

ORG_ID = "org-1"
ROBOT_ID = "robot-1"
STREAM_ID = "stream-1"
BATCH_URL = f"{STREAM_API_URL}/org/{ORG_ID}/signalling/track/batch"


class RecordingClientSession:
    """Client session stub that records the requests it is given."""

    def __init__(self) -> None:
        """Initialise the session with no recorded posts."""
        self.posts: list[tuple[str, Any]] = []

    async def post(
        self, url: str, headers: dict | None = None, json: Any = None
    ) -> None:
        """Record one post request."""
        self.posts.append((url, json))


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

    thread = threading.Thread(target=run, name="nc-async-loop", daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    done.wait(timeout=2.0)


@pytest.fixture
def client_session() -> RecordingClientSession:
    """Client session stub shared by the manager and the assertions."""
    return RecordingClientSession()


@pytest.fixture
def manager(
    nc_loop, client_session: RecordingClientSession, monkeypatch
) -> ClientProviderStreamManager:
    """Manager under test, isolated from the session global streaming state."""
    monkeypatch.setattr(
        client_provider_stream_manager,
        "get_provide_live_data_enabled_manager",
        lambda: EnabledManager(True, loop=nc_loop),
    )
    return ClientProviderStreamManager(
        robot_id=ROBOT_ID,
        robot_instance=0,
        local_stream_id=STREAM_ID,
        client_session=client_session,
        loop=nc_loop,
        org_id=ORG_ID,
        auth=MagicMock(get_headers=lambda: {}),
    )


def make_track(label: str) -> RobotStreamTrack:
    return RobotStreamTrack(
        robot_id=ROBOT_ID,
        robot_instance=0,
        stream_id=STREAM_ID,
        data_type=DataType.RGB_IMAGES,
        label=label,
        mid=label,
    )


def wait_for_posts(
    session: RecordingClientSession, count: int, timeout: float = 5.0
) -> list[tuple[str, Any]]:
    """Wait until the session has recorded at least `count` posts."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(session.posts) >= count:
            return session.posts
        time.sleep(0.01)
    raise AssertionError(f"expected {count} posts, got {len(session.posts)}")


def wait_past_flush() -> None:
    """Wait long enough for any scheduled flush to have run."""
    time.sleep(TRACK_BATCH_MAX_WAIT_S * 3)


def run_on_loop(loop: asyncio.AbstractEventLoop, coroutine) -> None:
    """Run a coroutine on the manager's loop and wait for it to finish."""
    asyncio.run_coroutine_threadsafe(coroutine, loop).result(timeout=5)


def labels_of(post: tuple[str, Any]) -> list[str]:
    return [track["label"] for track in post[1]["tracks"]]


def test_burst_of_sources_is_sent_as_one_batch(
    manager: ClientProviderStreamManager,
    client_session: RecordingClientSession,
    nc_loop,
):
    async def create_sources() -> None:
        manager.get_video_source("front", DataType.RGB_IMAGES, "front-rgb")
        manager.get_video_source("front", DataType.DEPTH_IMAGES, "front-depth")
        manager.get_json_source("joints", DataType.JOINT_POSITIONS, "joints")

    run_on_loop(nc_loop, create_sources())

    posts = wait_for_posts(client_session, 1)
    wait_past_flush()

    assert len(posts) == 1
    url, payload = posts[0]
    assert url == BATCH_URL
    assert [track["label"] for track in payload["tracks"]] == [
        "front",
        "front",
        "joints",
    ]


def test_track_submitted_after_a_flush_is_sent_in_a_later_batch(
    manager: ClientProviderStreamManager, client_session: RecordingClientSession
):
    manager.get_video_source("front", DataType.RGB_IMAGES, "front-rgb")
    wait_for_posts(client_session, 1)

    manager.get_json_source("joints", DataType.JOINT_POSITIONS, "joints")

    posts = wait_for_posts(client_session, 2)
    assert labels_of(posts[0]) == ["front"]
    assert labels_of(posts[1]) == ["joints"]


def test_resurrected_stream_resubmits_known_tracks_in_one_batch(
    manager: ClientProviderStreamManager,
    client_session: RecordingClientSession,
    nc_loop,
):
    for label in ("front", "joints"):
        track = make_track(label)
        manager.track_metadata[track.id] = track

    asyncio.run_coroutine_threadsafe(manager.on_stream_resurrected(), nc_loop).result(
        timeout=5
    )

    assert len(client_session.posts) == 1
    assert labels_of(client_session.posts[0]) == ["front", "joints"]


def test_resurrected_stream_chunks_tracks_over_the_batch_cap(
    manager: ClientProviderStreamManager,
    client_session: RecordingClientSession,
    nc_loop,
):
    track_count = TRACK_BATCH_MAX_SIZE + 50
    for index in range(track_count):
        track = make_track(f"sensor-{index}")
        manager.track_metadata[track.id] = track

    asyncio.run_coroutine_threadsafe(manager.on_stream_resurrected(), nc_loop).result(
        timeout=5
    )

    assert [len(payload["tracks"]) for _, payload in client_session.posts] == [
        TRACK_BATCH_MAX_SIZE,
        50,
    ]


def test_disabled_streaming_posts_nothing(
    manager: ClientProviderStreamManager, client_session: RecordingClientSession
):
    manager.enabled_manager.disable()

    manager.get_video_source("front", DataType.RGB_IMAGES, "front-rgb")
    wait_past_flush()

    assert client_session.posts == []


def test_pending_tracks_are_dropped_on_close(
    manager: ClientProviderStreamManager,
    client_session: RecordingClientSession,
    nc_loop,
):
    run_on_loop(nc_loop, manager.submit_track("0", DataType.RGB_IMAGES, "front"))
    assert manager._pending_tracks

    manager._on_close()
    wait_past_flush()

    assert client_session.posts == []
