import threading

import pytest

from neuracore.data_daemon.communications_management.producer.producer_channel import (
    ProducerChannel,
)
from neuracore.data_daemon.models import DataType, TraceTransportMetadata


class _FakeIox2Transport:
    def __init__(
        self, *, healthy: bool = True, seqs: list[int | None] | None = None
    ) -> None:
        self.healthy = healthy
        self.sent: list[tuple[dict, bytes]] = []
        self._seqs = seqs
        self.finish_recording_session_calls = 0
        self.close_calls = 0

    def send_frame(
        self,
        metadata: dict,
        chunk: bytes,
        stop_cutoff_sequence_number: int | None = None,
    ) -> int | None:
        self.sent.append((metadata, bytes(chunk)))
        if self._seqs is not None:
            return self._seqs.pop(0)
        return len(self.sent)

    def is_healthy(self) -> bool:
        return self.healthy

    def finish_recording_session(self) -> None:
        self.finish_recording_session_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_cleanup_producer_channel_finishes_recording_session() -> None:
    channel = object.__new__(ProducerChannel)
    transport = _FakeIox2Transport()
    wait_calls: list[int] = []
    end_trace_calls: list[str] = []

    channel._iox2_transport = transport
    # cleanup must flush the ZMQ sender's own last-enqueued sequence, NOT the
    # global stop cutoff (which is an iceoryx2 frame sequence for video channels
    # and would never be reported as sent over ZMQ).
    channel.get_last_enqueued_sequence_number = lambda: 12
    channel.wait_until_sequence_sent = (
        lambda sequence_number: wait_calls.append(sequence_number) or True
    )
    channel.end_trace = lambda: end_trace_calls.append("end")

    ProducerChannel.cleanup_producer_channel(
        channel,
        stop_cutoff_sequence_number=41,
        wait_for_transport_drain=True,
    )

    assert wait_calls == [12]
    assert end_trace_calls == ["end"]
    assert transport.finish_recording_session_calls == 1


def test_cleanup_producer_channel_raises_when_flush_not_sent() -> None:
    channel = object.__new__(ProducerChannel)
    transport = _FakeIox2Transport()
    end_trace_calls: list[str] = []

    channel._iox2_transport = transport
    channel.get_last_enqueued_sequence_number = lambda: 44
    channel.wait_until_sequence_sent = lambda sequence_number: False
    channel.end_trace = lambda: end_trace_calls.append("end")

    with pytest.raises(
        RuntimeError,
        match="Failed to send queued recording data before cleanup",
    ):
        ProducerChannel.cleanup_producer_channel(
            channel,
            stop_cutoff_sequence_number=44,
            wait_for_transport_drain=True,
        )

    assert end_trace_calls == []
    assert transport.finish_recording_session_calls == 0


def test_stop_producer_channel_closes_resources_in_order() -> None:
    channel = object.__new__(ProducerChannel)
    wait_calls: list[int] = []
    close_calls: list[str] = []

    channel._stop_heartbeat_service = lambda: close_calls.append("heartbeat")
    channel.get_last_enqueued_sequence_number = lambda: 12
    channel.wait_until_sequence_sent = (
        lambda sequence_number: wait_calls.append(sequence_number) or True
    )
    channel._close_iox2_transport = lambda: close_calls.append("transport")
    channel._stop_message_sender = lambda: close_calls.append("sender")
    channel._comm = type(
        "_Comm", (), {"cleanup_producer": lambda self: close_calls.append("comm")}
    )()

    ProducerChannel.stop_producer_channel(channel)

    assert wait_calls == [12]
    assert close_calls == ["heartbeat", "transport", "sender", "comm"]


def test_stop_producer_channel_raises_when_cutoff_not_sent_and_still_cleans_up() -> (
    None
):
    channel = object.__new__(ProducerChannel)
    close_calls: list[str] = []

    channel._stop_heartbeat_service = lambda: close_calls.append("heartbeat")
    channel.get_last_enqueued_sequence_number = lambda: 21
    channel.wait_until_sequence_sent = lambda sequence_number: False
    channel._get_message_sender_error = lambda: None
    channel._close_iox2_transport = lambda: close_calls.append("transport")
    channel._stop_message_sender = lambda: close_calls.append("sender")
    channel._comm = type(
        "_Comm", (), {"cleanup_producer": lambda self: close_calls.append("comm")}
    )()

    with pytest.raises(
        RuntimeError,
        match="Failed to send all enqueued messages before stopping producer channel",
    ):
        ProducerChannel.stop_producer_channel(channel)

    assert close_calls == ["heartbeat", "transport", "sender", "comm"]


def test_stop_producer_channel_swallows_cutoff_failure_after_sender_error() -> None:
    channel = object.__new__(ProducerChannel)
    close_calls: list[str] = []

    channel._stop_heartbeat_service = lambda: close_calls.append("heartbeat")
    channel.get_last_enqueued_sequence_number = lambda: 22
    channel.wait_until_sequence_sent = lambda sequence_number: False
    channel._get_message_sender_error = lambda: RuntimeError("boom")
    channel._close_iox2_transport = lambda: close_calls.append("transport")
    channel._stop_message_sender = lambda: close_calls.append("sender")
    channel._comm = type(
        "_Comm", (), {"cleanup_producer": lambda self: close_calls.append("comm")}
    )()

    ProducerChannel.stop_producer_channel(channel)

    assert close_calls == ["heartbeat", "transport", "sender", "comm"]


def test_end_trace_waits_before_clearing_trace_state() -> None:
    channel = object.__new__(ProducerChannel)
    send_calls: list[tuple[object, dict]] = []
    wait_calls: list[int] = []

    channel.trace_id = "trace-1"
    channel.recording_id = "recording-1"
    channel._send = lambda command, payload=None: (
        send_calls.append((command, payload or {})) or 55
    )
    channel.wait_until_sequence_sent = (
        lambda sequence_number: wait_calls.append(sequence_number) or True
    )

    ProducerChannel.end_trace(channel)

    assert len(send_calls) == 1
    assert wait_calls == [55]
    assert channel.trace_id is None
    assert channel.recording_id is None


def test_end_trace_keeps_trace_state_when_trace_end_not_sent() -> None:
    channel = object.__new__(ProducerChannel)

    channel.trace_id = "trace-1"
    channel.recording_id = "recording-1"
    channel._send = lambda command, payload=None: 56
    channel.wait_until_sequence_sent = lambda sequence_number: False

    with pytest.raises(
        RuntimeError,
        match="Failed to send TRACE_END before ending trace",
    ):
        ProducerChannel.end_trace(channel)

    assert channel.trace_id == "trace-1"
    assert channel.recording_id == "recording-1"


def _video_channel_for_send_tests(
    transport: _FakeIox2Transport,
) -> ProducerChannel:
    channel = object.__new__(ProducerChannel)
    channel._stop_cutoff_sequence_number = None
    channel.trace_id = "trace-1"
    channel.recording_id = "recording-1"
    channel.chunk_size = 1024
    channel._use_video_transport = True
    channel._iox2_transport = transport
    channel._recording_send_lock = threading.RLock()
    return channel


def _trace_metadata() -> TraceTransportMetadata:
    return TraceTransportMetadata(
        recording_id="recording-1",
        data_type=DataType.RGB_IMAGES,
        data_type_name="camera",
        robot_name="robot",
        dataset_name="dataset",
        robot_instance=0,
    )


def test_send_data_parts_iox2_publishes_each_chunk() -> None:
    transport = _FakeIox2Transport()
    channel = _video_channel_for_send_tests(transport)

    ProducerChannel._send_data_parts_iox2(
        channel,
        [memoryview(b"frame-bytes")],
        total_chunks=1,
        trace_metadata=_trace_metadata(),
    )

    assert len(transport.sent) == 1
    metadata, chunk = transport.sent[0]
    assert chunk == b"frame-bytes"
    assert metadata["trace_id"] == "trace-1"


def test_send_data_parts_iox2_stops_logging_when_transport_unhealthy() -> None:
    transport = _FakeIox2Transport(healthy=False, seqs=[None])
    channel = _video_channel_for_send_tests(transport)
    stop_calls: list[str] = []
    channel._stop_video_logging_after_failure = lambda: stop_calls.append("stop")

    with pytest.raises(RuntimeError, match="became unhealthy"):
        ProducerChannel._send_data_parts_iox2(
            channel,
            [memoryview(b"frame-bytes")],
            total_chunks=1,
            trace_metadata=_trace_metadata(),
        )

    assert stop_calls == ["stop"]


def test_send_data_parts_iox2_stops_on_cutoff_without_failure() -> None:
    # send_frame returns None but the transport is healthy => stop cutoff hit;
    # this must not be treated as a failure.
    transport = _FakeIox2Transport(healthy=True, seqs=[None])
    channel = _video_channel_for_send_tests(transport)
    stop_calls: list[str] = []
    channel._stop_video_logging_after_failure = lambda: stop_calls.append("stop")

    ProducerChannel._send_data_parts_iox2(
        channel,
        [memoryview(b"frame-bytes")],
        total_chunks=1,
        trace_metadata=_trace_metadata(),
    )

    assert stop_calls == []
