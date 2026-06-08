"""Unit tests for the iceoryx2 daemon-side subscriber drain pool."""

from __future__ import annotations

import uuid

from neuracore.data_daemon.communications_management.shared_transport.iox2_daemon_drain import (  # noqa: E501
    Iox2DaemonDrain,
)
from neuracore.data_daemon.communications_management.shared_transport.iox2_video_transport import (  # noqa: E501
    Iox2VideoTransport,
)


def _channel_id() -> str:
    return f"test-{uuid.uuid4().hex[:12]}"


def test_register_and_drain() -> None:
    """A registered channel drains frames sent by its publisher."""
    channel_id = _channel_id()
    transport = Iox2VideoTransport(channel_id, channel_id)
    drain = Iox2DaemonDrain()
    try:
        drain.register_channel(channel_id)
        transport.update_connections()
        transport.send_frame({"trace_id": "t", "chunk_index": 0}, b"frame-0")

        seen: list[tuple[str, int, dict, bytes]] = []
        count = drain.drain_all(lambda *args: seen.append(args))
        assert count == 1
        assert seen[0][0] == channel_id
        assert seen[0][3] == b"frame-0"
    finally:
        drain.close()
        transport.close()


def test_drain_multiple_channels() -> None:
    """Two channels are drained independently."""
    channel_a = _channel_id()
    channel_b = _channel_id()
    transport_a = Iox2VideoTransport(channel_a, channel_a)
    transport_b = Iox2VideoTransport(channel_b, channel_b)
    drain = Iox2DaemonDrain()
    try:
        drain.register_channel(channel_a)
        drain.register_channel(channel_b)
        transport_a.update_connections()
        transport_b.update_connections()
        transport_a.send_frame({"trace_id": "a"}, b"aaa")
        transport_b.send_frame({"trace_id": "b"}, b"bbb")

        by_channel: dict[str, bytes] = {}
        drain.drain_all(lambda ch, seq, meta, chunk: by_channel.__setitem__(ch, chunk))
        assert by_channel == {channel_a: b"aaa", channel_b: b"bbb"}
    finally:
        drain.close()
        transport_a.close()
        transport_b.close()


def test_unregister_channel_stops_draining() -> None:
    """After unregister, frames on that channel are not drained."""
    channel_id = _channel_id()
    transport = Iox2VideoTransport(channel_id, channel_id)
    drain = Iox2DaemonDrain()
    try:
        drain.register_channel(channel_id)
        assert drain.is_registered(channel_id)
        drain.unregister_channel(channel_id)
        assert not drain.is_registered(channel_id)

        transport.send_frame({"trace_id": "t"}, b"x")
        count = drain.drain_all(lambda *args: None)
        assert count == 0
    finally:
        drain.close()
        transport.close()


def test_drain_empty_returns_zero() -> None:
    """Draining with no frames returns 0 and does not error."""
    channel_id = _channel_id()
    drain = Iox2DaemonDrain()
    try:
        drain.register_channel(channel_id)
        assert drain.drain_all(lambda *args: None) == 0
    finally:
        drain.close()


def test_drop_counter_detects_frame_index_gaps() -> None:
    """Gaps in the per-channel frame index are counted as drops."""
    channel_id = _channel_id()
    drain = Iox2DaemonDrain()
    try:
        drain.register_channel(channel_id)
        channel_sub = drain._subscribers[channel_id]
        channel_sub.note_frame_index(0)
        channel_sub.note_frame_index(1)
        # Frames 2 and 3 were dropped under overload.
        newly_dropped = channel_sub.note_frame_index(4)
        assert newly_dropped == 2
        assert drain.dropped_frame_count(channel_id) == 2
    finally:
        drain.close()
