"""A Python viewer lists robots from the legacy service and every region."""

import json
from collections import defaultdict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from neuracore.core.exceptions import AuthenticationError
from neuracore.core.streaming.p2p.consumer import org_nodes_manager as module
from neuracore.core.streaming.p2p.consumer.org_nodes_manager import (
    DROPPED_REGION_KEEP_S,
    OrgNodesManager,
)
from neuracore.core.streaming.p2p.consumer.region_availability_consumer import (
    RegionAvailabilityConsumer,
)
from neuracore.core.streaming.p2p.consumer.regional_availability import (
    RegionalAvailability,
)

ORG_ID = "org-1"
ROBOT_ID = "robot-1"
REGION = "https://region-a.example.com"


class Clock:
    """A settable monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _init(stream_id: str, epoch: int | None = None) -> str:
    track = {
        "robot_id": ROBOT_ID,
        "robot_instance": 0,
        "stream_id": stream_id,
        "data_type": "RGB_IMAGES",
        "label": "front",
        "mid": "v0",
        "id": f"{stream_id}-track",
        "created_at": "2026-11-06T10:00:00Z",
    }
    instance: dict[str, Any] = {
        "robot_instance": 0,
        "tracks": {stream_id: [track]},
        "connections": 0,
    }
    if epoch is not None:
        instance["epoch"] = epoch
    return json.dumps({
        "type": "init",
        "robots": [{"robot_id": ROBOT_ID, "instances": {"0": instance}}],
    })


def _manager(clock: Clock) -> OrgNodesManager:
    manager = OrgNodesManager.__new__(OrgNodesManager)
    manager.last_update = None
    manager.last_nodes = defaultdict(dict)
    manager.consumers = {}
    manager.stream_manager_orchestrator = MagicMock(
        own_stream_ids=MagicMock(return_value=set())
    )
    manager._apply_stream_changes = MagicMock()
    manager.availability = RegionalAvailability(
        keep_seconds=DROPPED_REGION_KEEP_S, clock=clock
    )
    manager.merged_update = None
    manager.enabled_manager = MagicMock(is_disabled=MagicMock(return_value=False))
    manager.region_consumers = []
    manager.org_id = ORG_ID
    manager.loop = MagicMock(is_closed=MagicMock(return_value=False))
    return manager


def _listed(manager: OrgNodesManager) -> list[str]:
    nodes = manager._apply_stream_changes.call_args.args[1]
    return list(nodes[(ROBOT_ID, 0)])


@pytest.mark.asyncio
async def test_a_regional_robot_is_listed_over_its_legacy_entry() -> None:
    manager = _manager(Clock())

    await manager.on_message(_init("legacy-stream"))
    manager._on_region_event(REGION, _init("regional-stream", epoch=5))

    assert _listed(manager) == ["regional-stream"]
    assert manager.availability.owner_of(ROBOT_ID, 0) == REGION


@pytest.mark.asyncio
async def test_a_dropped_region_falls_back_to_legacy_after_the_keep_time() -> None:
    clock = Clock()
    manager = _manager(clock)
    await manager.on_message(_init("legacy-stream"))
    manager._on_region_event(REGION, _init("regional-stream", epoch=5))

    manager._on_region_drop(REGION)
    (delay, republish), _ = manager.loop.call_later.call_args
    clock.now += DROPPED_REGION_KEEP_S - 1
    republish()
    assert _listed(manager) == ["regional-stream"]
    clock.now += 2
    republish()

    assert delay > DROPPED_REGION_KEEP_S
    assert _listed(manager) == ["legacy-stream"]


@pytest.mark.asyncio
async def test_each_region_gets_a_list_stream() -> None:
    manager = _manager(Clock())
    manager.client_session = MagicMock()

    async def regions() -> list[str]:
        return [REGION, "https://region-b.example.com"]

    manager.stream_manager_orchestrator.region_endpoints = regions
    with (
        patch.object(module, "RegionAvailabilityConsumer") as consumer_class,
        patch.object(module, "EnabledManager"),
    ):
        await manager._listen_to_regions()

    assert [c.kwargs["region_url"] for c in consumer_class.call_args_list] == [
        REGION,
        "https://region-b.example.com",
    ]
    assert consumer_class.call_args.kwargs["session"] is (
        manager.stream_manager_orchestrator.org_session
    )


@pytest.mark.asyncio
async def test_the_regions_are_asked_again_after_a_failure() -> None:
    manager = _manager(Clock())
    manager.client_session = MagicMock()
    answers: list[Any] = [RuntimeError("backend down"), [REGION]]

    async def regions() -> list[str]:
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    manager.stream_manager_orchestrator.region_endpoints = regions
    with (
        patch.object(module, "REGIONS_RETRY_MIN_S", 0),
        patch.object(module, "RegionAvailabilityConsumer") as consumer_class,
        patch.object(module, "EnabledManager"),
    ):
        await manager._listen_to_regions()

    assert [c.kwargs["region_url"] for c in consumer_class.call_args_list] == [REGION]


@pytest.mark.asyncio
async def test_no_region_streams_when_the_api_key_is_refused() -> None:
    manager = _manager(Clock())

    async def regions() -> list[str]:
        raise AuthenticationError("refused")

    manager.stream_manager_orchestrator.region_endpoints = regions
    with patch.object(module, "RegionAvailabilityConsumer") as consumer_class:
        await manager._listen_to_regions()

    consumer_class.assert_not_called()


@pytest.mark.asyncio
async def test_a_closed_manager_publishes_nothing() -> None:
    manager = _manager(Clock())
    await manager.on_message(_init("legacy-stream"))
    manager._apply_stream_changes.reset_mock()
    manager.enabled_manager.is_disabled.return_value = True

    manager._on_region_event(REGION, _init("regional-stream", epoch=5))

    manager._apply_stream_changes.assert_not_called()


def test_closing_closes_the_region_streams() -> None:
    manager = _manager(Clock())
    region_consumer = MagicMock()
    manager.region_consumers = [region_consumer]
    manager.connections = defaultdict(dict)
    manager.background_tracker = MagicMock()
    manager.signalling_stream_future = MagicMock()
    manager.auth = MagicMock(listeners=MagicMock(return_value=[]))
    manager._logout_listener = MagicMock()
    manager.loop = MagicMock(is_closed=MagicMock(return_value=True))

    manager._on_close()

    region_consumer.close.assert_called_once()
    assert manager.region_consumers == []


def _region_consumer() -> tuple[RegionAvailabilityConsumer, MagicMock, MagicMock]:
    consumer = RegionAvailabilityConsumer.__new__(RegionAvailabilityConsumer)
    consumer.region_url = REGION
    consumer.org_id = ORG_ID
    consumer.session = MagicMock()

    async def headers() -> dict[str, str]:
        return {"Authorization": "Bearer org-token"}

    consumer.session.headers = headers
    on_event, on_drop = MagicMock(), MagicMock()
    consumer.on_event = on_event
    consumer.on_drop = on_drop
    return consumer, on_event, on_drop


@pytest.mark.asyncio
async def test_region_stream_uses_the_org_token() -> None:
    consumer, _, _ = _region_consumer()

    config = await consumer.prepare_sse_client_config()

    assert config.url == f"{REGION}/org/{ORG_ID}/signalling/available_robots"
    assert config.request_options == {"headers": {"Authorization": "Bearer org-token"}}


@pytest.mark.asyncio
async def test_region_stream_reports_events_and_drops() -> None:
    consumer, on_event, on_drop = _region_consumer()

    await consumer.on_message("data")
    await consumer.on_reconnect()

    on_event.assert_called_once_with(REGION, "data")
    on_drop.assert_called_once_with(REGION)
