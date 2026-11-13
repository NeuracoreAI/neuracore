"""The client merges the robot lists of every region and connects to the owner."""

import asyncio
from typing import Any

import pytest
from neuracore_types import AvailableRobotCapacityInit, AvailableRobotCapacityUpdate

pytestmark = pytest.mark.asyncio

ORG_ID = "org-1"
ROBOT_ID = "robot-1"
REGION_A = "https://a.signalling.example"
REGION_B = "https://b.signalling.example"
KEEP_SECONDS = 25.0


class Clock:
    """A clock that tests move by hand."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current time."""
        return self.now


def _track(stream_id: str, track_id: str, robot_id: str = ROBOT_ID) -> dict[str, Any]:
    return {
        "robot_id": robot_id,
        "robot_instance": 0,
        "stream_id": stream_id,
        "data_type": "RGB_IMAGES",
        "label": "front_camera",
        "mid": "v0",
        "id": track_id,
        "created_at": "2026-10-30T10:00:00Z",
    }


def _init(
    epoch: int | None,
    stream_id: str,
    track_id: str,
    robot_id: str = ROBOT_ID,
    connections: int = 0,
    stream_epochs: bool = True,
) -> AvailableRobotCapacityInit:
    instance: dict[str, Any] = {
        "robot_instance": 0,
        "tracks": {stream_id: [_track(stream_id, track_id, robot_id)]},
        "connections": connections,
    }
    if epoch is not None:
        instance["epoch"] = epoch
        if stream_epochs:
            instance["stream_epochs"] = {stream_id: epoch}
    return AvailableRobotCapacityInit.model_validate({
        "type": "init",
        "robots": [{"robot_id": robot_id, "instances": {"0": instance}}],
    })


def _robots(snapshot: AvailableRobotCapacityInit) -> dict[str, Any]:
    return {robot.robot_id: robot for robot in snapshot.robots}


def _track_ids(snapshot: AvailableRobotCapacityInit, robot_id: str = ROBOT_ID) -> set:
    robot = _robots(snapshot)[robot_id]
    return {t.id for tracks in robot.instances[0].tracks.values() for t in tracks}


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def availability(regional: Any, clock: Clock) -> Any:
    module = regional("consumer.regional_availability")
    return module.RegionalAvailability(keep_seconds=KEEP_SECONDS, clock=clock)


async def test_list_opens_one_stream_for_each_region_endpoint(regional: Any) -> None:
    """Given a token response with three region endpoints,
    when the client builds its list streams,
    then it has one available_robots URL for each endpoint.
    """
    module = regional("consumer.regional_availability")
    endpoints = [REGION_A, REGION_B, "https://c.signalling.example"]

    urls = module.list_stream_urls(ORG_ID, endpoints)

    assert urls == [f"{e}/org/{ORG_ID}/signalling/available_robots" for e in endpoints]


async def test_merged_list_has_robots_from_every_region(availability: Any) -> None:
    """Given a robot in region A and another robot in region B,
    when both inits arrive,
    then the merged list has both robots.
    """
    availability.apply_init(REGION_A, _init(1, "s-a", "t-a"))
    availability.apply_init(REGION_B, _init(1, "s-b", "t-b", robot_id="robot-b"))

    assert set(_robots(availability.snapshot())) == {ROBOT_ID, "robot-b"}


async def test_stream_listed_by_two_regions_appears_once_with_higher_epoch(
    availability: Any,
) -> None:
    """Given one robot stream in region A at epoch 3 and in region B at
    epoch 4,
    when both inits arrive,
    then the merged list has it once, with region B's tracks, owned by B.
    """
    availability.apply_init(REGION_A, _init(3, "s-1", "t-a"))
    availability.apply_init(REGION_B, _init(4, "s-1", "t-b"))

    snapshot = availability.snapshot()
    assert len(snapshot.robots) == 1
    assert _track_ids(snapshot) == {"t-b"}
    assert availability.owner_of(ROBOT_ID, 0, "s-1") == REGION_B


async def test_instance_lists_every_stream_of_every_region(
    availability: Any,
) -> None:
    """Given two streams of one robot instance, one in region A at epoch 3
    and one in region B at epoch 4,
    when both inits arrive,
    then the instance has both streams, each owned by its region, and the
    connections of both regions.
    """
    availability.apply_init(REGION_A, _init(3, "s-a", "t-a", connections=1))
    availability.apply_init(REGION_B, _init(4, "s-b", "t-b", connections=2))

    instance = _robots(availability.snapshot())[ROBOT_ID].instances[0]
    assert _track_ids(availability.snapshot()) == {"t-a", "t-b"}
    assert availability.owner_of(ROBOT_ID, 0, "s-a") == REGION_A
    assert availability.owner_of(ROBOT_ID, 0, "s-b") == REGION_B
    assert instance.connections == 3
    assert instance.stream_epochs == {"s-a": 3, "s-b": 4}


async def test_region_without_stream_epochs_uses_the_instance_epoch(
    availability: Any,
) -> None:
    """Given region A of the first regional release, which sends only the
    instance epoch 4, and region B with the same stream at epoch 3,
    when both inits arrive,
    then region A owns the stream.
    """
    availability.apply_init(REGION_A, _init(4, "s-1", "t-a", stream_epochs=False))
    availability.apply_init(REGION_B, _init(3, "s-1", "t-b"))

    assert availability.owner_of(ROBOT_ID, 0, "s-1") == REGION_A


@pytest.mark.parametrize("order", [(REGION_B, REGION_A), (REGION_A, REGION_B)])
async def test_equal_epochs_pick_the_first_region_by_url(
    availability: Any, order: tuple[str, str]
) -> None:
    """Given one robot stream at the same epoch in regions B and A,
    when both inits arrive, in either order,
    then the region whose URL sorts first owns it.
    """
    tracks = {REGION_A: ("s-1", "t-a"), REGION_B: ("s-1", "t-b")}
    for source in order:
        availability.apply_init(source, _init(2, *tracks[source]))

    assert availability.owner_of(ROBOT_ID, 0, "s-1") == REGION_A
    assert _track_ids(availability.snapshot()) == {"t-a"}


async def test_updates_from_the_losing_region_are_ignored(availability: Any) -> None:
    """Given a robot stream owned by region B at epoch 4, still listed by
    region A at epoch 3,
    when region A adds a track to it,
    then the merged list keeps region B's tracks only.
    """
    availability.apply_init(REGION_A, _init(3, "s-1", "t-a"))
    availability.apply_init(REGION_B, _init(4, "s-1", "t-b"))

    availability.apply_update(
        REGION_A,
        AvailableRobotCapacityUpdate.model_validate({
            "type": "update",
            "tracks_added": [_track("s-1", "t-a2")],
            "epochs": {ROBOT_ID: {"0": 3}},
            "stream_epochs": {"s-1": 3},
        }),
    )

    assert _track_ids(availability.snapshot()) == {"t-b"}
    assert availability.owner_of(ROBOT_ID, 0, "s-1") == REGION_B


async def test_unreachable_region_leaves_the_list_working(availability: Any) -> None:
    """Given region A's stream that never connects, and region B with a robot,
    when region A drops,
    then the merged list still has region B's robot.
    """
    availability.apply_init(REGION_B, _init(1, "s-b", "t-b"))

    availability.drop_source(REGION_A)

    assert set(_robots(availability.snapshot())) == {ROBOT_ID}


async def test_dropped_region_keeps_its_robots_for_one_deadline(
    availability: Any, clock: Clock
) -> None:
    """Given a robot in region A,
    when region A's stream drops,
    then the robot stays in the merged list for one deadline, then leaves.
    """
    availability.apply_init(REGION_A, _init(1, "s-a", "t-a"))

    availability.drop_source(REGION_A)
    clock.now = KEEP_SECONDS - 1
    assert set(_robots(availability.snapshot())) == {ROBOT_ID}

    clock.now = KEEP_SECONDS + 1
    assert availability.snapshot().robots == []


async def test_reconnected_region_init_replaces_its_entries(availability: Any) -> None:
    """Given region A with a robot on track t-1,
    when region A reconnects and its new init has track t-2 only,
    then the merged list has t-2 and not t-1.
    """
    availability.apply_init(REGION_A, _init(1, "s-a", "t-1"))
    availability.drop_source(REGION_A)

    availability.apply_init(REGION_A, _init(1, "s-a", "t-2"))

    assert _track_ids(availability.snapshot()) == {"t-2"}


async def test_connect_goes_to_the_listed_region(
    availability: Any, regional: Any
) -> None:
    """Given a robot instance owned by region B,
    when the client connects,
    then it calls region B only.
    """
    module = regional("consumer.regional_availability")
    availability.apply_init(REGION_B, _init(1, "s-b", "t-b"))
    calls: list[str] = []

    async def open_connection(base_url: str) -> tuple[int, dict[str, Any]]:
        calls.append(base_url)
        return 200, {}

    await module.connect_to_listed_region(
        availability, ROBOT_ID, 0, "s-b", open_connection
    )

    assert calls == [REGION_B]


async def test_connect_retries_after_not_present(
    availability: Any, regional: Any
) -> None:
    """Given a robot instance listed in region A that has moved to region B,
    when region A answers 409 ``not_present`` and region B's init arrives,
    then the client connects again, in region B.
    """
    module = regional("consumer.regional_availability")
    availability.apply_init(REGION_A, _init(1, "s-a", "t-a"))
    calls: list[str] = []

    async def open_connection(base_url: str) -> tuple[int, dict[str, Any]]:
        calls.append(base_url)
        if base_url == REGION_A:
            asyncio.get_running_loop().call_later(
                0.05, availability.apply_init, REGION_B, _init(2, "s-a", "t-b")
            )
            return 409, {"detail": "not_present", "retryable": True}
        return 200, {}

    await module.connect_to_listed_region(
        availability, ROBOT_ID, 0, "s-a", open_connection, wait_seconds=1.0
    )

    assert calls == [REGION_A, REGION_B]


async def test_connect_reports_unavailable_after_three_tries(
    availability: Any, regional: Any
) -> None:
    """Given a robot instance whose region always answers ``not_present``,
    when the client connects,
    then it tries three times and raises ``RobotInstanceUnavailableError``.
    """
    module = regional("consumer.regional_availability")
    availability.apply_init(REGION_A, _init(1, "s-a", "t-a"))
    calls: list[str] = []

    async def open_connection(base_url: str) -> tuple[int, dict[str, Any]]:
        calls.append(base_url)
        return 409, {"detail": "not_present", "retryable": True}

    with pytest.raises(module.RobotInstanceUnavailableError):
        await module.connect_to_listed_region(
            availability, ROBOT_ID, 0, "s-a", open_connection, wait_seconds=0.05
        )

    assert calls == [REGION_A] * 3


async def test_snapshot_and_init_are_copies(availability: Any) -> None:
    """Given an applied init,
    when the caller changes the init or a snapshot,
    then the merged list does not change.
    """
    init = _init(1, "s-a", "t-a")
    availability.apply_init(REGION_A, init)

    init.robots[0].instances[0].tracks.clear()
    availability.snapshot().robots[0].instances[0].tracks.clear()

    assert _track_ids(availability.snapshot()) == {"t-a"}


async def test_snapshot_carries_the_updated_epoch(availability: Any) -> None:
    """Given a robot instance at epoch 1,
    when an update raises its epoch to 5,
    then the snapshot entry has epoch 5.
    """
    availability.apply_init(REGION_A, _init(1, "s-a", "t-a"))

    availability.apply_update(
        REGION_A,
        AvailableRobotCapacityUpdate.model_validate({
            "type": "update",
            "tracks_added": [_track("s-a", "t-a2")],
            "epochs": {ROBOT_ID: {"0": 5}},
            "stream_epochs": {"s-a": 5},
        }),
    )

    instance = _robots(availability.snapshot())[ROBOT_ID].instances[0]
    assert instance.epoch == 5
    assert instance.stream_epochs == {"s-a": 5}


async def test_connect_retries_at_once_when_the_list_moved_first(
    availability: Any, regional: Any
) -> None:
    """Given a robot instance listed in region A,
    when region B's init arrives before region A answers ``not_present``,
    then the client connects in region B without waiting.
    """
    module = regional("consumer.regional_availability")
    availability.apply_init(REGION_A, _init(1, "s-a", "t-a"))
    calls: list[str] = []

    async def open_connection(base_url: str) -> tuple[int, dict[str, Any]]:
        calls.append(base_url)
        if base_url == REGION_A:
            availability.apply_init(REGION_B, _init(2, "s-a", "t-b"))
            return 409, {"detail": "not_present", "retryable": True}
        return 200, {}

    await asyncio.wait_for(
        module.connect_to_listed_region(
            availability, ROBOT_ID, 0, "s-a", open_connection, wait_seconds=5.0
        ),
        1.0,
    )

    assert calls == [REGION_A, REGION_B]
