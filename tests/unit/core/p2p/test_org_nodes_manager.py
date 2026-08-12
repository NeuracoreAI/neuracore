"""Tests for incremental robot availability signalling events."""

import json
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from neuracore.core.streaming.p2p.consumer.org_nodes_manager import OrgNodesManager


def _track(
    track_id: str,
    *,
    robot_id: str = "robot-1",
    robot_instance: int = 0,
    stream_id: str = "stream-1",
    label: str = "front_camera",
) -> dict:
    return {
        "robot_id": robot_id,
        "robot_instance": robot_instance,
        "stream_id": stream_id,
        "data_type": "RGB_IMAGES",
        "label": label,
        "mid": "v0",
        "id": track_id,
        "created_at": "2026-08-12T10:00:00Z",
    }


def _manager() -> OrgNodesManager:
    manager = OrgNodesManager.__new__(OrgNodesManager)
    manager.last_update = None
    manager.last_nodes = defaultdict(dict)
    manager.consumers = {}
    manager.stream_manager_orchestrator = SimpleNamespace(
        signalling_consumer=SimpleNamespace(local_stream_id="local-stream")
    )
    manager._apply_stream_changes = MagicMock()
    return manager


def _init_event(*tracks: dict, connections: int = 1) -> str:
    tracks_by_stream: dict[str, list[dict]] = defaultdict(list)
    for track in tracks:
        tracks_by_stream[track["stream_id"]].append(track)
    robots = []
    if tracks:
        robots = [{
            "robot_id": tracks[0]["robot_id"],
            "instances": {
                str(tracks[0]["robot_instance"]): {
                    "robot_instance": tracks[0]["robot_instance"],
                    "tracks": tracks_by_stream,
                    "connections": connections,
                }
            },
        }]
    return json.dumps({"type": "init", "robots": robots})


def _update_event(
    *,
    tracks_added: list[dict] | None = None,
    tracks_removed: list[dict] | None = None,
    connection_updates: list[dict] | None = None,
) -> str:
    return json.dumps({
        "type": "update",
        "tracks_added": tracks_added or [],
        "tracks_removed": tracks_removed or [],
        "connection_updates": connection_updates or [],
    })


@pytest.mark.asyncio
async def test_init_replaces_state_from_previous_connection() -> None:
    manager = _manager()
    await manager.on_message(_init_event(_track("stale-track")))

    await manager.on_message(_init_event(_track("fresh-track", robot_id="robot-2")))

    assert manager.last_update is not None
    assert [robot.robot_id for robot in manager.last_update.robots] == ["robot-2"]
    tracks = manager.last_update.robots[0].instances[0].tracks["stream-1"]
    assert [track.id for track in tracks] == ["fresh-track"]


@pytest.mark.asyncio
async def test_update_adds_replaces_and_deduplicates_tracks() -> None:
    manager = _manager()
    await manager.on_message(_init_event())
    added_track = _track("track-1")
    update = _update_event(
        tracks_added=[added_track],
        connection_updates=[
            {"robot_id": "robot-1", "robot_instance": 0, "connections": 2}
        ],
    )

    await manager.on_message(update)
    await manager.on_message(update)
    await manager.on_message(
        _update_event(tracks_added=[{**added_track, "label": "replacement"}])
    )

    assert manager.last_update is not None
    instance = manager.last_update.robots[0].instances[0]
    assert instance.connections == 2
    assert len(instance.tracks["stream-1"]) == 1
    assert instance.tracks["stream-1"][0].label == "replacement"


@pytest.mark.asyncio
async def test_update_removes_tracks_and_prunes_empty_containers() -> None:
    manager = _manager()
    await manager.on_message(_init_event(_track("track-1")))
    removal = {
        "robot_id": "robot-1",
        "robot_instance": 0,
        "stream_id": "stream-1",
        "track_id": "track-1",
    }

    await manager.on_message(_update_event(tracks_removed=[removal]))
    await manager.on_message(_update_event(tracks_removed=[removal]))

    assert manager.last_update is not None
    assert manager.last_update.robots == []
    assert manager.last_nodes == {}


@pytest.mark.asyncio
async def test_connection_updates_are_absolute() -> None:
    manager = _manager()
    await manager.on_message(_init_event(_track("track-1"), connections=3))
    update = _update_event(
        connection_updates=[
            {"robot_id": "robot-1", "robot_instance": 0, "connections": 1}
        ]
    )

    await manager.on_message(update)

    assert manager.last_update is not None
    assert manager.last_update.robots[0].instances[0].connections == 1


@pytest.mark.asyncio
async def test_update_before_init_is_ignored() -> None:
    manager = _manager()

    await manager.on_message(_update_event(tracks_added=[_track("track-1")]))

    assert manager.last_update is None
    manager._apply_stream_changes.assert_not_called()
