"""Shared producer sender runtimes keyed by daemon socket path."""

from __future__ import annotations

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


class SharedSenderRegistry:
    """Reference-count shared sender runtimes by socket path."""

    _lock = threading.Lock()
    _runtimes: dict[tuple[str, int], _SharedSenderRuntime] = {}

    @classmethod
    def acquire(
        cls,
        *,
        socket_path: str,
        send_queue_maxsize: int,
    ) -> SharedSenderHandle:
        key = (socket_path, int(send_queue_maxsize))
        with cls._lock:
            runtime = cls._runtimes.get(key)
            if runtime is None:
                comm = CommunicationsManager(socket_path=socket_path)
                comm.create_producer_socket()
                sender = ProducerChannelMessageSender(
                    comm=comm,
                    send_queue_maxsize=send_queue_maxsize,
                )
                runtime = _SharedSenderRuntime(comm=comm, sender=sender, refcount=0)
                cls._runtimes[key] = runtime
            runtime.refcount += 1

        def _release() -> None:
            cls.release(socket_path=socket_path, send_queue_maxsize=send_queue_maxsize)

        return SharedSenderHandle(
            comm=runtime.comm,
            sender=runtime.sender,
            release=_release,
        )

    @classmethod
    def release(cls, *, socket_path: str, send_queue_maxsize: int) -> None:
        key = (socket_path, int(send_queue_maxsize))
        with cls._lock:
            runtime = cls._runtimes.get(key)
            if runtime is None:
                return
            runtime.refcount -= 1
            if runtime.refcount > 0:
                return
            cls._runtimes.pop(key, None)

        runtime.sender.close(join_timeout_s=2.0)
        runtime.comm.cleanup_producer()
