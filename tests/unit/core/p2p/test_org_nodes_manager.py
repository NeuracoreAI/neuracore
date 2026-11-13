"""Tests for incremental robot availability signalling events."""

import json
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

from neuracore_types import AvailableRobotCapacityInit

from neuracore.core.streaming.p2p.consumer.org_nodes_manager import OrgNodesManager
from neuracore.core.streaming.p2p.consumer.regional_availability import (
    RegionalAvailability,
)


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
    manager.last_nodes = defaultdict(dict)
    manager.consumers = {}
    manager.stream_manager_orchestrator = SimpleNamespace(
        signalling_consumer=SimpleNamespace(local_stream_id="local-stream"),
        own_stream_ids=lambda: {"local-stream", "own-robot-stream"},
    )
    manager._apply_stream_changes = MagicMock()
    manager.availability = RegionalAvailability(keep_seconds=25.0)
    manager.merged_update = None
    manager.enabled_manager = MagicMock(is_disabled=MagicMock(return_value=False))
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


def test_stream_map_skips_every_stream_of_this_process() -> None:
    manager = _manager()
    update = AvailableRobotCapacityInit.model_validate_json(
        _init_event(
            _track("own", stream_id="own-robot-stream"),
            _track("viewer", stream_id="local-stream"),
            _track("remote", stream_id="remote-stream"),
        )
    )

    stream_map = manager.get_instance_stream_map(update)

    assert list(stream_map[("robot-1", 0)]) == ["remote-stream"]
