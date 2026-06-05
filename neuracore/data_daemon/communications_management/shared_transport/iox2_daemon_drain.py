"""Daemon-side iceoryx2 subscriber pool that drains all video channels.

Replaces the daemon half of the old fixed-slot video transport (allocation,
descriptor handling, and credit returns). One subscriber is created
per video channel on first contact; :meth:`Iox2DaemonDrain.drain_all` copies
each received frame's payload out of the zero-copy ring and hands the decoded
``(sequence_id, frame_index, metadata, chunk)`` to a callback before releasing
the sample back to the ring buffer.

Frames dropped under overload (DiscardData) leave gaps in the per-channel
``frame_index`` sequence; those gaps are counted and logged so silent loss is
observable.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from collections.abc import Callable
from typing import Protocol, cast

import iceoryx2
from iceoryx2 import Slice

from neuracore.data_daemon.communications_management.shared_transport.framing import (
    parse_video_transport_packet,
)
from neuracore.data_daemon.communications_management.shared_transport.iox2_video_transport import (  # noqa: E501
    FRAME_INDEX_KEY,
    FRAME_META_KEY,
    FRAME_SEQUENCE_KEY,
)
from neuracore.data_daemon.const import (
    IOX2_HISTORY_SIZE,
    IOX2_SERVICE_PREFIX,
    IOX2_SUBSCRIBER_BUFFER_SIZE,
)

logger = logging.getLogger(__name__)

# Callback signature: (channel_id, sequence_id, metadata_dict, chunk_bytes).
FrameCallback = Callable[[str, int, dict[str, object], bytes], None]

# Log a per-channel drop summary at most this often (number of drained frames).
_DROP_LOG_INTERVAL = 64


class _SamplePayload(Protocol):
    number_of_elements: int
    data_ptr: int


class _ReceivedSample(Protocol):
    def payload(self) -> _SamplePayload: ...

    def delete(self) -> None: ...


class _Subscriber(Protocol):
    def receive(self) -> _ReceivedSample | None: ...

    def delete(self) -> None: ...


class _ChannelSubscriber:
    """One channel's subscriber plus its drop-detection bookkeeping."""

    def __init__(self, subscriber: _Subscriber) -> None:
        self.subscriber = subscriber
        self.last_frame_index: int | None = None
        self.dropped_frames = 0
        self._frames_since_log = 0

    def note_frame_index(self, frame_index: int) -> int:
        """Update drop tracking for one received frame; return newly dropped count."""
        dropped = 0
        if (
            self.last_frame_index is not None
            and frame_index > self.last_frame_index + 1
        ):
            dropped = frame_index - self.last_frame_index - 1
            self.dropped_frames += dropped
        if self.last_frame_index is None or frame_index > self.last_frame_index:
            self.last_frame_index = frame_index
        self._frames_since_log += 1
        return dropped

    def should_log_drops(self) -> bool:
        """Return True roughly once per drop-log interval to bound log spam."""
        if self._frames_since_log >= _DROP_LOG_INTERVAL:
            self._frames_since_log = 0
            return True
        return False


class Iox2DaemonDrain:
    """Own all daemon-side iceoryx2 subscribers for video channels."""

    def __init__(self) -> None:
        """Create the daemon iceoryx2 node and empty subscriber pool."""
        self._lock = threading.Lock()
        self._node = iceoryx2.NodeBuilder.new().create(iceoryx2.ServiceType.Ipc)
        self._subscribers: dict[str, _ChannelSubscriber] = {}
        # Reclaim resources left behind by producer/daemon processes that exited
        # uncleanly in a previous run (best effort).
        try:
            self._node.try_cleanup_dead_nodes(
                iceoryx2.ServiceType.Ipc, self._node.config
            )
        except Exception:
            logger.debug("Iox2DaemonDrain: dead-node cleanup skipped", exc_info=True)

    def register_channel(self, channel_id: str) -> None:
        """Create a subscriber for ``channel_id`` if one does not already exist."""
        with self._lock:
            if channel_id in self._subscribers:
                return

        service_name = f"{IOX2_SERVICE_PREFIX}{channel_id}"
        try:
            service = (
                self._node.service_builder(iceoryx2.ServiceName.new(service_name))
                .publish_subscribe(Slice[ctypes.c_uint8])
                .subscriber_max_buffer_size(IOX2_SUBSCRIBER_BUFFER_SIZE)
                .enable_safe_overflow(True)
                .history_size(IOX2_HISTORY_SIZE)
                .open_or_create()
            )
            subscriber = (
                service.subscriber_builder()
                .buffer_size(IOX2_SUBSCRIBER_BUFFER_SIZE)
                .create()
            )
            typed_subscriber = cast(_Subscriber, subscriber)
        except Exception:
            logger.exception(
                "Iox2DaemonDrain: failed to register channel channel_id=%s",
                channel_id,
            )
            return

        with self._lock:
            # Re-check under the lock in case of a concurrent registration.
            if channel_id in self._subscribers:
                _delete_subscriber(typed_subscriber)
                return
            self._subscribers[channel_id] = _ChannelSubscriber(typed_subscriber)
        logger.info("Iox2DaemonDrain: registered subscriber service=%s", service_name)

    def is_registered(self, channel_id: str) -> bool:
        """Return whether a subscriber exists for the channel."""
        with self._lock:
            return channel_id in self._subscribers

    def drain_all(self, on_frame: FrameCallback) -> int:
        """Poll every subscriber and invoke ``on_frame`` for each received sample.

        Each sample's payload is copied out and the sample released immediately
        (returning the slot to the ring buffer). ``on_frame`` runs synchronously
        on the caller's thread; it should stay fast and offload heavy work.
        Returns the total number of frames drained.
        """
        with self._lock:
            items = list(self._subscribers.items())

        drained = 0
        for channel_id, channel_sub in items:
            drained += self._drain_channel(channel_id, channel_sub, on_frame)
        return drained

    def _drain_channel(
        self,
        channel_id: str,
        channel_sub: _ChannelSubscriber,
        on_frame: FrameCallback,
    ) -> int:
        drained = 0
        subscriber = channel_sub.subscriber
        while True:
            try:
                sample = subscriber.receive()
            except Exception:
                logger.exception(
                    "Iox2DaemonDrain: receive error channel_id=%s", channel_id
                )
                break
            if sample is None:
                break
            try:
                payload = sample.payload()
                raw = bytes(
                    (ctypes.c_uint8 * payload.number_of_elements).from_address(
                        payload.data_ptr
                    )
                )
            finally:
                sample.delete()

            try:
                self._dispatch_frame(channel_id, channel_sub, raw, on_frame)
                drained += 1
            except Exception:
                logger.exception(
                    "Iox2DaemonDrain: failed to process frame channel_id=%s",
                    channel_id,
                )
        return drained

    def _dispatch_frame(
        self,
        channel_id: str,
        channel_sub: _ChannelSubscriber,
        raw: bytes,
        on_frame: FrameCallback,
    ) -> None:
        envelope, chunk = parse_video_transport_packet(raw)
        sequence_id = _parse_int_field(envelope, FRAME_SEQUENCE_KEY)
        frame_index = _parse_int_field(envelope, FRAME_INDEX_KEY)
        metadata = _parse_metadata_field(envelope, FRAME_META_KEY)

        dropped = channel_sub.note_frame_index(frame_index)
        if dropped and channel_sub.should_log_drops():
            logger.warning(
                "Iox2DaemonDrain: dropped video frames channel_id=%s "
                "recent=%d total=%d (daemon overload / DiscardData)",
                channel_id,
                dropped,
                channel_sub.dropped_frames,
            )
        on_frame(channel_id, sequence_id, metadata, chunk)

    def dropped_frame_count(self, channel_id: str) -> int:
        """Return the total frames dropped for one channel (for observability)."""
        with self._lock:
            channel_sub = self._subscribers.get(channel_id)
            return channel_sub.dropped_frames if channel_sub is not None else 0

    def unregister_channel(self, channel_id: str) -> None:
        """Remove and close the subscriber for one channel."""
        with self._lock:
            channel_sub = self._subscribers.pop(channel_id, None)
        if channel_sub is None:
            return
        if channel_sub.dropped_frames:
            logger.warning(
                "Iox2DaemonDrain: channel_id=%s dropped %d video frames total",
                channel_id,
                channel_sub.dropped_frames,
            )
        _delete_subscriber(channel_sub.subscriber)
        logger.info("Iox2DaemonDrain: unregistered channel channel_id=%s", channel_id)

    def close(self) -> None:
        """Close all subscribers and the node."""
        with self._lock:
            channel_subs = list(self._subscribers.values())
            self._subscribers.clear()
        for channel_sub in channel_subs:
            _delete_subscriber(channel_sub.subscriber)
        try:
            del self._node
        except Exception:
            logger.warning("Iox2DaemonDrain: error closing node", exc_info=True)
        logger.info("Iox2DaemonDrain closed")


def _parse_int_field(envelope: dict[str, object], key: str) -> int:
    """Extract one integer field from a decoded frame envelope."""
    value = envelope[key]
    if not isinstance(value, int):
        raise TypeError(f"Frame envelope field {key!r} must be an int")
    return value


def _parse_metadata_field(envelope: dict[str, object], key: str) -> dict[str, object]:
    """Extract the per-chunk metadata mapping from a decoded frame envelope."""
    value = envelope[key]
    if not isinstance(value, dict):
        raise TypeError(f"Frame envelope field {key!r} must be a dict")
    return cast(dict[str, object], value)


def _delete_subscriber(subscriber: _Subscriber) -> None:
    """Best-effort release of one iceoryx2 subscriber."""
    try:
        subscriber.delete()
    except Exception:
        logger.warning("Iox2DaemonDrain: error closing subscriber", exc_info=True)
