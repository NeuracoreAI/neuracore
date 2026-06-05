"""iceoryx2-backed video frame transport for one producer channel.

Replaces the hand-rolled fixed-slot video transport. The producer owns one
iceoryx2 publisher per channel and publishes self-describing frame packets into
a lock-free, zero-copy ring buffer. Loan -> write -> send all happen on the
caller's thread; there is no background worker, queue, credit channel, or
watchdog. Backpressure is handled by iceoryx2: under daemon overload the oldest
buffered frames are overwritten (DiscardData semantics).

The channel ``sequence_id`` (reserved from the shared
:class:`ChannelSequenceAllocator`) travels inside each frame so the daemon can
preserve the same end-of-recording ordering and drop semantics as the ZMQ
control path. A separate per-publisher ``frame_index`` lets the daemon detect
and count frames dropped under overload.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from collections.abc import Mapping

import iceoryx2
from iceoryx2 import Slice

from neuracore.data_daemon.communications_management.sequence_allocator import (
    ChannelSequenceAllocator,
)
from neuracore.data_daemon.communications_management.shared_transport.framing import (
    PacketTooLarge,
    build_video_transport_packet,
)
from neuracore.data_daemon.const import (
    IOX2_HISTORY_SIZE,
    IOX2_MAX_FRAME_BYTES,
    IOX2_SERVICE_PREFIX,
    IOX2_SUBSCRIBER_BUFFER_SIZE,
)

logger = logging.getLogger(__name__)

# Metadata-envelope keys carried alongside the per-chunk metadata so the daemon
# can recover lifecycle ordering (``seq``) and detect dropped frames (``idx``).
FRAME_SEQUENCE_KEY = "seq"
FRAME_INDEX_KEY = "idx"
FRAME_META_KEY = "meta"


def build_frame_envelope(
    sequence_id: int,
    frame_index: int,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Wrap per-chunk metadata with the lifecycle sequence and frame index."""
    return {
        FRAME_SEQUENCE_KEY: sequence_id,
        FRAME_INDEX_KEY: frame_index,
        FRAME_META_KEY: metadata,
    }


class Iox2VideoTransport:
    """Producer-side iceoryx2 publisher for one video channel."""

    def __init__(
        self,
        channel_id: str,
        sequence_allocator: ChannelSequenceAllocator | None = None,
        max_frame_bytes: int = IOX2_MAX_FRAME_BYTES,
    ) -> None:
        """Create the iceoryx2 node, service, and publisher for one channel."""
        self._channel_id = channel_id
        self._service_name = f"{IOX2_SERVICE_PREFIX}{channel_id}"
        self._max_frame_bytes = int(max_frame_bytes)
        self._sequence_allocator = sequence_allocator or ChannelSequenceAllocator()
        self._state_lock = threading.Lock()
        self._healthy = True
        self._error: Exception | None = None
        self._frame_index = 0
        self._last_payload_sequence_number = 0

        # open_or_create lets the producer start before or after the daemon
        # without a coordination handshake.
        self._node = iceoryx2.NodeBuilder.new().create(iceoryx2.ServiceType.Ipc)
        service = (
            self._node.service_builder(iceoryx2.ServiceName.new(self._service_name))
            .publish_subscribe(Slice[ctypes.c_uint8])
            .subscriber_max_buffer_size(IOX2_SUBSCRIBER_BUFFER_SIZE)
            .enable_safe_overflow(True)
            .history_size(IOX2_HISTORY_SIZE)
            .open_or_create()
        )
        self._publisher = (
            service.publisher_builder()
            .initial_max_slice_len(self._max_frame_bytes)
            .max_loaned_samples(1)
            .create()
        )
        logger.info("Iox2VideoTransport created service=%s", self._service_name)

    @property
    def service_name(self) -> str:
        """Return the iceoryx2 service name for this channel."""
        return self._service_name

    def send_frame(
        self,
        metadata: Mapping[str, object],
        chunk: bytes | bytearray | memoryview,
        stop_cutoff_sequence_number: int | None = None,
    ) -> int | None:
        """Encode and publish one frame on the caller's thread.

        Reserves a channel sequence number, builds the self-describing packet
        (magic + header + JSON metadata envelope + raw chunk), loans an
        iceoryx2 slot, writes into it with a single memmove, and sends.

        Returns the reserved sequence number, or ``None`` if the frame is
        rejected by the stop cutoff. Raises :class:`PacketTooLarge` if the
        encoded frame exceeds the configured slot size. Returns ``None`` and
        marks the transport unhealthy if the iceoryx2 send fails.
        """
        sequence_number = self._sequence_allocator.reserve()
        if (
            stop_cutoff_sequence_number is not None
            and sequence_number > stop_cutoff_sequence_number
        ):
            return None

        if not self._healthy:
            return None

        with self._state_lock:
            frame_index = self._frame_index
            self._frame_index += 1

        envelope = build_frame_envelope(sequence_number, frame_index, metadata)
        packet = build_video_transport_packet(envelope, chunk)
        packet_length = len(packet)
        if packet_length > self._max_frame_bytes:
            raise PacketTooLarge(
                f"Frame {packet_length} bytes exceeds max {self._max_frame_bytes}"
            )

        try:
            sample = self._publisher.loan_slice_uninit(packet_length)
            ctypes.memmove(sample.payload().data_ptr, packet, packet_length)
            deliveries = sample.assume_init().send()
            if deliveries == 0:
                logger.debug(
                    "Iox2VideoTransport: no daemon subscriber yet service=%s",
                    self._service_name,
                )
        except Exception as exc:
            logger.exception(
                "Iox2VideoTransport send failed service=%s", self._service_name
            )
            with self._state_lock:
                self._healthy = False
                self._error = exc
            return None

        with self._state_lock:
            self._last_payload_sequence_number = max(
                self._last_payload_sequence_number, sequence_number
            )
        return sequence_number

    def update_connections(self) -> None:
        """Refresh publisher connections so late subscribers receive history.

        Called from the producer heartbeat so a daemon subscriber that registers
        after the producer has gone idle still receives buffered history frames.
        """
        if not self._healthy:
            return
        try:
            self._publisher.update_connections()
        except Exception:
            logger.debug(
                "Iox2VideoTransport: update_connections failed service=%s",
                self._service_name,
                exc_info=True,
            )

    def is_healthy(self) -> bool:
        """Return True while the transport can accept new video writes."""
        return self._healthy

    def get_last_reserved_sequence_number(self) -> int:
        """Return the most recently reserved channel sequence number."""
        return self._sequence_allocator.get_last_reserved_sequence_number()

    def get_last_payload_sequence_number(self) -> int:
        """Return the latest sequence number reserved for a published frame."""
        with self._state_lock:
            return self._last_payload_sequence_number

    def finish_recording_session(self) -> None:
        """Reset per-recording local state. The service/publisher persist."""
        with self._state_lock:
            self._last_payload_sequence_number = 0

    def close(self) -> None:
        """Drop the publisher and node."""
        try:
            self._publisher.delete()
        except Exception:
            logger.warning("Iox2VideoTransport: error closing publisher", exc_info=True)
        try:
            del self._node
        except Exception:
            logger.warning("Iox2VideoTransport: error closing node", exc_info=True)
        logger.info("Iox2VideoTransport closed service=%s", self._service_name)
