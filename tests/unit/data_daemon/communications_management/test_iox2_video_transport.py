"""Unit tests for the iceoryx2 producer-side video transport."""

from __future__ import annotations

import uuid

import pytest

from neuracore.data_daemon.communications_management.shared_transport.framing import (
    PacketTooLarge,
    parse_video_transport_packet,
)
from neuracore.data_daemon.communications_management.shared_transport.iox2_daemon_drain import (  # noqa: E501
    Iox2DaemonDrain,
)
from neuracore.data_daemon.communications_management.shared_transport.iox2_video_transport import (  # noqa: E501
    FRAME_INDEX_KEY,
    FRAME_META_KEY,
    FRAME_SEQUENCE_KEY,
    Iox2VideoTransport,
)


def _channel_id() -> str:
    return f"test-{uuid.uuid4().hex[:12]}"


def test_send_frame_round_trip() -> None:
    """A published frame is received and parses back to the original payload."""
    channel_id = _channel_id()
    transport = Iox2VideoTransport(channel_id)
    drain = Iox2DaemonDrain()
    try:
        drain.register_channel(channel_id)
        transport.update_connections()

        metadata = {"trace_id": "t1", "chunk_index": 0, "total_chunks": 1}
        chunk = b"hello-iceoryx2-payload"
        seq = transport.send_frame(metadata, chunk)
        assert seq is not None

        received: list[tuple[str, int, dict, bytes]] = []
        drain.drain_all(lambda *args: received.append(args))

        assert len(received) == 1
        got_channel, got_seq, got_meta, got_chunk = received[0]
        assert got_channel == channel_id
        assert got_seq == seq
        assert got_meta == metadata
        assert got_chunk == chunk
    finally:
        drain.close()
        transport.close()


def test_send_frame_too_large() -> None:
    """An oversized frame raises rather than being silently discarded."""
    channel_id = _channel_id()
    transport = Iox2VideoTransport(channel_id, max_frame_bytes=128)
    try:
        with pytest.raises(PacketTooLarge):
            transport.send_frame({"trace_id": "t"}, b"x" * 256)
    finally:
        transport.close()


def test_send_frame_no_subscriber_is_not_an_error() -> None:
    """Publishing with no daemon subscriber succeeds (frame just goes nowhere)."""
    channel_id = _channel_id()
    transport = Iox2VideoTransport(channel_id)
    try:
        seq = transport.send_frame({"trace_id": "t", "chunk_index": 0}, b"data")
        assert seq is not None
        assert transport.is_healthy()
    finally:
        transport.close()


def test_send_frame_respects_stop_cutoff() -> None:
    """Frames past the stop cutoff are rejected and return None."""
    channel_id = _channel_id()
    transport = Iox2VideoTransport(channel_id)
    try:
        first = transport.send_frame({"trace_id": "t"}, b"a")
        assert first is not None
        # The next reserved sequence will be first + 1, which is past the cutoff.
        rejected = transport.send_frame(
            {"trace_id": "t"}, b"b", stop_cutoff_sequence_number=first
        )
        assert rejected is None
    finally:
        transport.close()


def test_finish_recording_session_is_noop_for_service() -> None:
    """finish_recording_session keeps the publisher/service usable."""
    channel_id = _channel_id()
    transport = Iox2VideoTransport(channel_id)
    try:
        transport.send_frame({"trace_id": "t"}, b"a")
        transport.finish_recording_session()
        assert transport.is_healthy()
        assert transport.send_frame({"trace_id": "t2"}, b"b") is not None
    finally:
        transport.close()


def test_envelope_carries_sequence_and_index() -> None:
    """Each frame embeds its sequence id and a monotonic frame index."""
    channel_id = _channel_id()
    transport = Iox2VideoTransport(channel_id)
    drain = Iox2DaemonDrain()
    try:
        drain.register_channel(channel_id)
        transport.update_connections()
        transport.send_frame({"trace_id": "t", "chunk_index": 0}, b"a")
        transport.send_frame({"trace_id": "t", "chunk_index": 1}, b"b")

        indices: list[int] = []

        def collect(channel: str, seq: int, meta: dict, chunk: bytes) -> None:
            indices.append(seq)

        drain.drain_all(collect)
        assert indices == sorted(indices)
        assert len(indices) == 2
    finally:
        drain.close()
        transport.close()


def test_raw_packet_envelope_structure() -> None:
    """The on-wire packet wraps metadata with seq/idx/meta keys."""
    from neuracore.data_daemon.communications_management.shared_transport.framing import (  # noqa: E501
        build_video_transport_packet,
    )
    from neuracore.data_daemon.communications_management.shared_transport.iox2_video_transport import (  # noqa: E501
        build_frame_envelope,
    )

    envelope = build_frame_envelope(7, 3, {"trace_id": "t", "chunk_index": 0})
    packet = build_video_transport_packet(envelope, b"payload")
    parsed, chunk = parse_video_transport_packet(packet)
    assert parsed[FRAME_SEQUENCE_KEY] == 7
    assert parsed[FRAME_INDEX_KEY] == 3
    assert parsed[FRAME_META_KEY] == {"trace_id": "t", "chunk_index": 0}
    assert chunk == b"payload"
