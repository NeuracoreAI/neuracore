"""Consumer-side reads of the regional signalling state."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import requests
from neuracore_types import DataType

import neuracore as nc
from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.robot import Robot

WAIT_SECONDS = 60
POLL_SECONDS = 0.5

Value = TypeVar("Value")


@dataclass(frozen=True)
class ListedStream:
    """One stream of a robot instance in the merged robot list.

    Attributes:
        stream_id: ID of the stream.
        region: Region URL that lists the stream, or the legacy source.
        epoch: Registration epoch of the stream in that region.
    """

    stream_id: str
    region: str
    epoch: int | None


def region_endpoints(org_id: str) -> list[str]:
    """Return the canonical URL of every signalling region of the organization."""
    response = requests.post(
        f"{API_URL}/org/{org_id}/signalling/token",
        json={},
        headers=get_auth().get_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return list(response.json()["region_endpoints"])


def listed_streams(robot: Robot) -> list[ListedStream]:
    """Return the streams that the merged robot list shows for the robot instance."""
    from neuracore.core.streaming.p2p.consumer.org_nodes_manager import (
        get_org_nodes_manager,
    )

    manager = get_org_nodes_manager(robot.org_id)

    async def read() -> list[ListedStream]:
        availability = manager.availability
        streams: list[ListedStream] = []
        for listed_robot in availability.snapshot().robots:
            if listed_robot.robot_id != robot.id:
                continue
            instance = listed_robot.instances.get(robot.instance)
            if instance is None:
                continue
            for stream_id in instance.tracks:
                owner = availability.owner_of(robot.id, robot.instance, stream_id)
                if owner is not None:
                    streams.append(
                        ListedStream(
                            stream_id=stream_id,
                            region=owner,
                            epoch=instance.stream_epochs.get(stream_id),
                        )
                    )
        return streams

    return asyncio.run_coroutine_threadsafe(read(), manager.loop).result(timeout=10)


def wait_until(poll: Callable[[], Value | None], description: str) -> Value:
    """Poll until a value is returned, and fail with the description on timeout."""
    deadline = time.monotonic() + WAIT_SECONDS
    while (value := poll()) is None:
        assert time.monotonic() < deadline, f"Timed out waiting for {description}."
        time.sleep(POLL_SECONDS)
    return value


def wait_for_streams(robot: Robot, count: int) -> list[ListedStream]:
    """Wait until the robot instance lists exactly the given number of streams."""

    def poll() -> list[ListedStream] | None:
        streams = listed_streams(robot)
        return streams if len(streams) == count else None

    return wait_until(poll, f"{count} listed stream(s)")


def wait_for_remote_nodes(robot: Robot, node_count: int) -> None:
    """Wait until the given number of remote nodes are connected to the robot."""

    def poll() -> bool | None:
        return (
            True
            if nc.check_remote_nodes_connected(
                num_remote_nodes=node_count,
                robot_name=robot.name,
                instance=robot.instance,
            )
            else None
        )

    wait_until(poll, f"{node_count} remote node(s) to connect")


def wait_for_joint_positions(robot: Robot, joint_names: list[str]) -> dict[str, float]:
    """Wait until the consumer has the named joint positions and return them."""

    def poll() -> dict[str, float] | None:
        sync_point = nc.get_latest_sync_point(
            robot_name=robot.name, instance=robot.instance, include_remote=True
        )
        if DataType.JOINT_POSITIONS not in sync_point.data:
            return None
        positions = sync_point[DataType.JOINT_POSITIONS]
        if not all(name in positions for name in joint_names):
            return None
        return {name: positions[name].value for name in joint_names}

    return wait_until(poll, f"joint positions {joint_names}")
