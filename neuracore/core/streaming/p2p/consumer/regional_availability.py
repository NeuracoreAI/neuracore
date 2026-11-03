"""The merged robot list of every signalling region.

Each region lists only the robots registered with it. The client streams the
list of every region and merges them per stream: a robot instance is the union
of its streams in every region. When two sources list one stream, the entry
with the higher stream epoch wins. Equal epochs go to the region whose URL
sorts first. A regional entry wins over a legacy one, which has no epoch.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from neuracore_types import (
    AvailableRobot,
    AvailableRobotCapacityInit,
    AvailableRobotCapacityUpdate,
    AvailableRobotInstance,
    RobotStreamTrack,
)

LEGACY_SOURCE = "legacy"
NOT_PRESENT = "not_present"
CONNECT_TRIES = 3
CONNECT_WAIT_SECONDS = 5.0

Key = tuple[str, int]
StreamKey = tuple[str, int, str]
OpenConnection = Callable[[str], Awaitable[tuple[int, dict[str, Any]]]]


class RobotInstanceUnavailableError(Exception):
    """No region accepted a connection to the robot instance."""


@dataclass
class _Source:
    """One source's listing: tracks per stream, stream epochs, connections."""

    tracks: dict[StreamKey, list[RobotStreamTrack]] = field(default_factory=dict)
    epochs: dict[str, int] = field(default_factory=dict)
    connections: dict[Key, int] = field(default_factory=dict)


@dataclass
class _Winner:
    source: str
    tracks: list[RobotStreamTrack]
    epoch: int | None


def list_stream_urls(org_id: str, region_endpoints: list[str]) -> list[str]:
    """Return one ``available_robots`` URL for each region endpoint."""
    return [
        f"{endpoint}/org/{org_id}/signalling/available_robots"
        for endpoint in region_endpoints
    ]


def _instance_epochs(stream_ids: Iterable[str], epoch: int | None) -> dict[str, int]:
    """Stream epochs of a region of the first regional release: the instance's."""
    return {} if epoch is None else dict.fromkeys(stream_ids, epoch)


class RegionalAvailability:
    """Merged robot list of all sources.

    A source is a region endpoint (the base URL that ``connect_to_listed_region``
    connects to), or ``LEGACY_SOURCE``.
    """

    def __init__(
        self, keep_seconds: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        """Initialize an empty list.

        Args:
            keep_seconds: How long a dropped source's entries stay listed.
            clock: Monotonic time in seconds.
        """
        self.keep_seconds = keep_seconds
        self.clock = clock
        self._sources: dict[str, _Source] = {}
        self._dropped_at: dict[str, float] = {}
        self._changed: dict[Key, asyncio.Event] = {}

    def apply_init(self, source: str, init: AvailableRobotCapacityInit) -> None:
        """Replace every entry of ``source``."""
        old = self._sources.get(source, _Source())
        new = _Source()
        for robot in init.robots:
            for index, instance in robot.instances.items():
                new.connections[(robot.robot_id, index)] = instance.connections
                for stream_id, tracks in instance.tracks.items():
                    new.tracks[(robot.robot_id, index, stream_id)] = [
                        t.model_copy(deep=True) for t in tracks
                    ]
                new.epochs.update(
                    instance.stream_epochs
                    or _instance_epochs(instance.tracks, instance.epoch)
                )
        self._sources[source] = new
        self._dropped_at.pop(source, None)
        self._notify(set(old.connections) | set(new.connections))

    def apply_update(self, source: str, update: AvailableRobotCapacityUpdate) -> None:
        """Apply a delta to the entries of ``source``."""
        own = self._sources.setdefault(source, _Source())
        touched: set[Key] = set()
        for track in update.tracks_added:
            key = (track.robot_id, track.robot_instance, track.stream_id)
            tracks = own.tracks.setdefault(key, [])
            tracks[:] = [t for t in tracks if t.id != track.id] + [track]
            touched.add(key[:2])
        for removed in update.tracks_removed:
            key = (removed.robot_id, removed.robot_instance, removed.stream_id)
            if key not in own.tracks:
                continue
            own.tracks[key] = [t for t in own.tracks[key] if t.id != removed.track_id]
            if not own.tracks[key]:
                del own.tracks[key]
                own.epochs.pop(removed.stream_id, None)
            touched.add(key[:2])
        for connection in update.connection_updates:
            instance_key = (connection.robot_id, connection.robot_instance)
            own.connections[instance_key] = connection.connections
            touched.add(instance_key)
        if update.stream_epochs:
            own.epochs.update(update.stream_epochs)
            touched |= {k[:2] for k in own.tracks if k[2] in update.stream_epochs}
        else:
            for robot_id, epochs in update.epochs.items():
                for index, epoch in epochs.items():
                    for key in own.tracks:
                        if key[:2] == (robot_id, index):
                            own.epochs[key[2]] = epoch
                            touched.add(key[:2])
        listed = {stream_key[:2] for stream_key in own.tracks}
        for instance_key in touched - listed:
            own.connections.pop(instance_key, None)
        self._notify(touched)

    def drop_source(self, source: str) -> None:
        """Mark that the stream of ``source`` dropped.

        Its entries stay listed for ``keep_seconds``, unless it reconnects.
        """
        self._dropped_at.setdefault(source, self.clock())

    def snapshot(self) -> AvailableRobotCapacityInit:
        """Return a copy of the winning streams of each robot instance."""
        live = self._live_sources()
        robots: dict[str, dict[int, AvailableRobotInstance]] = {}
        for (robot_id, index, stream_id), winner in self._winners().items():
            instances = robots.setdefault(robot_id, {})
            instance = instances.get(index)
            if instance is None:
                instance = instances[index] = AvailableRobotInstance(
                    robot_instance=index,
                    tracks={},
                    connections=sum(
                        self._sources[s].connections.get((robot_id, index), 0)
                        for s in live
                    ),
                )
            instance.tracks[stream_id] = [
                t.model_copy(deep=True) for t in winner.tracks
            ]
            if winner.epoch is not None:
                instance.stream_epochs[stream_id] = winner.epoch
                instance.epoch = max(instance.epoch or 0, winner.epoch)
        return AvailableRobotCapacityInit(
            robots=[
                AvailableRobot(robot_id=robot_id, instances=instances)
                for robot_id, instances in robots.items()
            ]
        )

    def owner_of(
        self, robot_id: str, robot_instance: int, stream_id: str
    ) -> str | None:
        """Return the source of the stream's winning entry, or None."""
        winner = self._winners().get((robot_id, robot_instance, stream_id))
        return None if winner is None else winner.source

    async def wait_for_update(
        self, robot_id: str, robot_instance: int, timeout: float
    ) -> bool:
        """Wait for a change of the robot instance's entries.

        Returns:
            True if a change came before the timeout.
        """
        event = self._changed.setdefault((robot_id, robot_instance), asyncio.Event())
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return True

    def _live_sources(self) -> list[str]:
        now = self.clock()
        return sorted(
            source
            for source in self._sources
            if now - self._dropped_at.get(source, now) <= self.keep_seconds
        )

    def _winners(self) -> dict[StreamKey, _Winner]:
        winners: dict[StreamKey, _Winner] = {}
        for source in self._live_sources():
            own = self._sources[source]
            for key, tracks in own.tracks.items():
                epoch = own.epochs.get(key[2])
                current = winners.get(key)
                if current is None or _rank(source, epoch) > _rank(
                    current.source, current.epoch
                ):
                    winners[key] = _Winner(source, tracks, epoch)
        return winners

    def _notify(self, keys: set[Key]) -> None:
        for key in keys:
            event = self._changed.pop(key, None)
            if event is not None:
                event.set()


def _rank(source: str, epoch: int | None) -> tuple[bool, int]:
    """Order entries: regional over legacy, then the higher epoch.

    Sources are visited in URL order and only a higher rank replaces the
    winner, so equal ranks go to the source that sorts first.
    """
    return (source != LEGACY_SOURCE, epoch or 0)


async def connect_to_listed_region(
    availability: RegionalAvailability,
    robot_id: str,
    robot_instance: int,
    stream_id: str,
    open_connection: OpenConnection,
    tries: int = CONNECT_TRIES,
    wait_seconds: float = CONNECT_WAIT_SECONDS,
) -> tuple[int, dict[str, Any]]:
    """Connect to the robot stream in the region that lists it.

    On 409 ``not_present`` the robot has moved: wait for the next list update
    of the instance, or ``wait_seconds``, then try the region the list shows.

    Args:
        availability: The merged robot list.
        robot_id: Robot to connect to.
        robot_instance: Robot instance to connect to.
        stream_id: Robot stream to connect to.
        open_connection: Opens the connection at a source (a region's base
            URL, or ``LEGACY_SOURCE``) and returns the status and JSON body.
        tries: Most connection attempts.
        wait_seconds: Longest wait for a list update between attempts.

    Returns:
        The status and body of the last attempt that was not ``not_present``.

    Raises:
        RobotInstanceUnavailableError: If no region lists the instance, or
            every attempt got ``not_present``.
    """
    for attempt in range(tries):
        owner = availability.owner_of(robot_id, robot_instance, stream_id)
        if owner is None:
            break
        status, body = await open_connection(owner)
        if status != 409 or body.get("detail") != NOT_PRESENT:
            return status, body
        moved = availability.owner_of(robot_id, robot_instance, stream_id) != owner
        if attempt + 1 < tries and not moved:
            await availability.wait_for_update(robot_id, robot_instance, wait_seconds)
    raise RobotInstanceUnavailableError(f"{robot_id}/{robot_instance}")
