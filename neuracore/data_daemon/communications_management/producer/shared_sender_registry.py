"""Shared producer sender runtimes keyed by daemon socket path."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

from ..shared_transport.communications_manager import CommunicationsManager
from .producer_channel_message_sender import ProducerChannelMessageSender


@dataclass
class SharedSenderHandle:
    """One acquired shared sender runtime plus a release callback."""

    comm: CommunicationsManager
    sender: ProducerChannelMessageSender
    release: Callable[[], None]


@dataclass
class _SharedSenderRuntime:
    comm: CommunicationsManager
    sender: ProducerChannelMessageSender
    refcount: int = 0


@dataclass
class _SharedSenderLane:
    runtime: _SharedSenderRuntime


@dataclass
class _SharedSenderGroup:
    lanes: list[_SharedSenderLane]
    binding_to_lane: dict[str, int]


class SharedSenderRegistry:
    """Reference-count shared sender runtimes by socket path."""

    _lock = threading.Lock()
    _groups: dict[tuple[str, int, str], _SharedSenderGroup] = {}

    @staticmethod
    def _create_runtime(socket_path: str, send_queue_maxsize: int) -> _SharedSenderRuntime:
        comm = CommunicationsManager(socket_path=socket_path)
        comm.create_producer_socket()
        sender = ProducerChannelMessageSender(
            comm=comm,
            send_queue_maxsize=send_queue_maxsize,
        )
        return _SharedSenderRuntime(comm=comm, sender=sender, refcount=0)

    @staticmethod
    def _desired_lane_count(binding_count: int) -> int:
        """Grow lane count dynamically with active bindings."""
        if binding_count <= 1:
            return 1
        return max(1, math.isqrt(binding_count - 1) + 1)

    @classmethod
    def acquire(
        cls,
        *,
        socket_path: str,
        send_queue_maxsize: int,
        binding_key: str,
        policy: str = "shared_single",
    ) -> SharedSenderHandle:
        key = (socket_path, int(send_queue_maxsize), policy)
        with cls._lock:
            group = cls._groups.get(key)
            if group is None:
                group = _SharedSenderGroup(lanes=[], binding_to_lane={})
                cls._groups[key] = group

            lane_index = group.binding_to_lane.get(binding_key)
            if lane_index is None:
                if policy == "json_message":
                    desired_lane_count = cls._desired_lane_count(
                        len(group.binding_to_lane) + 1
                    )
                else:
                    desired_lane_count = 1
                while len(group.lanes) < desired_lane_count:
                    group.lanes.append(
                        _SharedSenderLane(
                            runtime=cls._create_runtime(
                                socket_path=socket_path,
                                send_queue_maxsize=send_queue_maxsize,
                            )
                        )
                    )
                if policy == "json_message":
                    lane_index = min(
                        range(len(group.lanes)),
                        key=lambda idx: group.lanes[idx].runtime.refcount,
                    )
                else:
                    lane_index = 0
                group.binding_to_lane[binding_key] = lane_index

            lane = group.lanes[lane_index]
            lane.runtime.refcount += 1

        def _release() -> None:
            cls.release(
                socket_path=socket_path,
                send_queue_maxsize=send_queue_maxsize,
                binding_key=binding_key,
                policy=policy,
            )

        return SharedSenderHandle(
            comm=lane.runtime.comm,
            sender=lane.runtime.sender,
            release=_release,
        )

    @classmethod
    def release(
        cls,
        *,
        socket_path: str,
        send_queue_maxsize: int,
        binding_key: str,
        policy: str = "shared_single",
    ) -> None:
        key = (socket_path, int(send_queue_maxsize), policy)
        runtimes_to_close: list[_SharedSenderRuntime] = []
        with cls._lock:
            group = cls._groups.get(key)
            if group is None:
                return
            lane_index = group.binding_to_lane.pop(binding_key, None)
            if lane_index is None:
                return
            lane = group.lanes[lane_index]
            lane.runtime.refcount -= 1
            if not group.binding_to_lane:
                cls._groups.pop(key, None)
                runtimes_to_close = [lane.runtime for lane in group.lanes]

        for runtime in runtimes_to_close:
            runtime.sender.close(join_timeout_s=2.0)
            runtime.comm.cleanup_producer()
