"""The merged robot list of every signalling region.

Each region lists only the robots registered with it. The client streams the
list of every region and merges them: for each robot instance, the entry with
the highest epoch wins. Equal epochs go to the region whose URL sorts first.
A regional entry wins over a legacy one, which has no epoch.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from neuracore_types import (
    AvailableRobot,
    AvailableRobotCapacityInit,
    AvailableRobotCapacityUpdate,
    AvailableRobotInstance,
)

LEGACY_SOURCE = "legacy"
NOT_PRESENT = "not_present"
CONNECT_TRIES = 3
CONNECT_WAIT_SECONDS = 5.0

Key = tuple[str, int]
OpenConnection = Callable[[str], Awaitable[tuple[int, dict[str, Any]]]]


class RobotInstanceUnavailableError(Exception):
    """No region accepted a connection to the robot instance."""


@dataclass
class _Entry:
    """One source's listing of one robot instance."""

    epoch: int | None
    instance: AvailableRobotInstance


def list_stream_urls(org_id: str, region_endpoints: list[str]) -> list[str]:
    """Return one ``available_robots`` URL for each region endpoint."""
    return [
        f"{endpoint}/org/{org_id}/signalling/available_robots"
        for endpoint in region_endpoints
    ]


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
        self._entries: dict[str, dict[Key, _Entry]] = {}
        self._dropped_at: dict[str, float] = {}
        self._changed: dict[Key, asyncio.Event] = {}

    def apply_init(self, source: str, init: AvailableRobotCapacityInit) -> None:
        """Replace every entry of ``source``."""
        old = self._entries.get(source, {})
        entries = {
            (robot.robot_id, index): _Entry(
                instance.epoch, instance.model_copy(deep=True)
            )
            for robot in init.robots
            for index, instance in robot.instances.items()
        }
        self._entries[source] = entries
        self._dropped_at.pop(source, None)
        self._notify(set(old) | set(entries))

    def apply_update(self, source: str, update: AvailableRobotCapacityUpdate) -> None:
        """Apply a delta to the entries of ``source``."""
        entries = self._entries.setdefault(source, {})
        touched: set[Key] = set()
        for track in update.tracks_added:
            key = (track.robot_id, track.robot_instance)
            entry = entries.get(key)
            if entry is None:
                entry = _Entry(
                    None,
                    AvailableRobotInstance(
                        robot_instance=track.robot_instance, tracks={}, connections=0
                    ),
                )
                entries[key] = entry
            tracks = entry.instance.tracks.setdefault(track.stream_id, [])
            tracks[:] = [t for t in tracks if t.id != track.id] + [track]
            touched.add(key)
        for removed in update.tracks_removed:
            key = (removed.robot_id, removed.robot_instance)
            entry = entries.get(key)
            if entry is None:
                continue
            tracks = entry.instance.tracks.get(removed.stream_id, [])
            entry.instance.tracks[removed.stream_id] = [
                t for t in tracks if t.id != removed.track_id
            ]
            touched.add(key)
        for connection in update.connection_updates:
            key = (connection.robot_id, connection.robot_instance)
            entry = entries.get(key)
            if entry is not None:
                entry.instance.connections = connection.connections
                touched.add(key)
        for robot_id, epochs in update.epochs.items():
            for index, epoch in epochs.items():
                entry = entries.get((robot_id, index))
                if entry is not None:
                    entry.epoch = epoch
                    entry.instance.epoch = epoch
                    touched.add((robot_id, index))
        for key in touched:
            entry = entries[key]
            entry.instance.tracks = {
                s: t for s, t in entry.instance.tracks.items() if t
            }
            if not entry.instance.tracks:
                del entries[key]
        self._notify(touched)

    def drop_source(self, source: str) -> None:
        """Mark that the stream of ``source`` dropped.

        Its entries stay listed for ``keep_seconds``, unless it reconnects.
        """
        self._dropped_at.setdefault(source, self.clock())

    def snapshot(self) -> AvailableRobotCapacityInit:
        """Return a copy of the winning entry of each robot instance."""
        robots: dict[str, dict[int, AvailableRobotInstance]] = {}
        for (robot_id, index), (_, entry) in self._winners().items():
            robots.setdefault(robot_id, {})[index] = entry.instance.model_copy(
                deep=True
            )
        return AvailableRobotCapacityInit(
            robots=[
                AvailableRobot(robot_id=robot_id, instances=instances)
                for robot_id, instances in robots.items()
            ]
        )

    def owner_of(self, robot_id: str, robot_instance: int) -> str | None:
        """Return the source of the winning entry, or None."""
        winner = self._winners().get((robot_id, robot_instance))
        return None if winner is None else winner[0]

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
        return [
            source
            for source in self._entries
            if now - self._dropped_at.get(source, now) <= self.keep_seconds
        ]

    def _winners(self) -> dict[Key, tuple[str, _Entry]]:
        winners: dict[Key, tuple[str, _Entry]] = {}
        for source in sorted(self._live_sources()):
            for key, entry in self._entries[source].items():
                current = winners.get(key)
                if current is None or _rank(source, entry) > _rank(*current):
                    winners[key] = (source, entry)
        return winners

    def _notify(self, keys: set[Key]) -> None:
        for key in keys:
            event = self._changed.pop(key, None)
            if event is not None:
                event.set()


def _rank(source: str, entry: _Entry) -> tuple[bool, int]:
    """Order entries: regional over legacy, then the higher epoch.

    Sources are visited in URL order and only a higher rank replaces the
    winner, so equal ranks go to the source that sorts first.
    """
    return (source != LEGACY_SOURCE, entry.epoch or 0)


async def connect_to_listed_region(
    availability: RegionalAvailability,
    robot_id: str,
    robot_instance: int,
    open_connection: OpenConnection,
    tries: int = CONNECT_TRIES,
    wait_seconds: float = CONNECT_WAIT_SECONDS,
) -> tuple[int, dict[str, Any]]:
    """Connect to the robot instance in the region that lists it.

    On 409 ``not_present`` the robot has moved: wait for the next list update
    of the instance, or ``wait_seconds``, then try the region the list shows.

    Args:
        availability: The merged robot list.
        robot_id: Robot to connect to.
        robot_instance: Robot instance to connect to.
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
        owner = availability.owner_of(robot_id, robot_instance)
        if owner is None:
            break
        status, body = await open_connection(owner)
        if status != 409 or body.get("detail") != NOT_PRESENT:
            return status, body
        moved = availability.owner_of(robot_id, robot_instance) != owner
        if attempt + 1 < tries and not moved:
            await availability.wait_for_update(robot_id, robot_instance, wait_seconds)
    raise RobotInstanceUnavailableError(f"{robot_id}/{robot_instance}")
