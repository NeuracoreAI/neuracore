import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from neuracore_types import DataType

from neuracore.data_daemon.communications_management.consumer import (
    bridge_chunk_spool as bridge_chunk_spool_module,
)
from neuracore.data_daemon.communications_management.consumer.completion_worker import (
    CompletionWorker,
)
from neuracore.data_daemon.communications_management.consumer.data_bridge import Daemon
from neuracore.data_daemon.communications_management.consumer.models import (
    ChannelState,
    CompletionChunkWork,
    TraceMetadataRegistrationRequest,
    TraceMetadataSnapshot,
    TransportMode,
    VideoFrameSequenceProgressRequest,
)
from neuracore.data_daemon.communications_management.producer.producer_channel import (
    ProducerChannel,
)
from neuracore.data_daemon.communications_management.shared_transport.framing import (
    PacketTooLarge,
)
from neuracore.data_daemon.communications_management.shared_transport.iox2_daemon_drain import (  # noqa: E501
    Iox2DaemonDrain,
)
from neuracore.data_daemon.const import HEARTBEAT_TIMEOUT_SECS
from neuracore.data_daemon.models import (
    BatchedJointDataItemPayload,
    BatchedJointDataPayload,
    CommandType,
    CompleteMessage,
    DataChunkPayload,
    MessageEnvelope,
    TraceTransportMetadata,
)

BridgeChunkSpool = bridge_chunk_spool_module.BridgeChunkSpool


def test_message_envelope_round_trip() -> None:
    payload = {"data_type": "rgb_image"}
    envelope = MessageEnvelope(
        producer_id="producer-123",
        command=CommandType.HEARTBEAT,
        payload=payload,
        sequence_number=3,
    )

    parsed = MessageEnvelope.from_bytes(envelope.to_bytes())

    assert parsed.producer_id == "producer-123"
    assert parsed.command == CommandType.HEARTBEAT
    assert parsed.payload == payload
    assert parsed.sequence_number == 3


def test_data_chunk_payload_round_trip() -> None:
    chunk = DataChunkPayload(
        channel_id="chan-1",
        recording_id="rec-1",
        trace_id="42",
        chunk_index=1,
        total_chunks=2,
        data_type_name="custom",
        dataset_id=None,
        dataset_name=None,
        robot_name=None,
        robot_id=None,
        robot_instance=3,
        data_type=DataType.CUSTOM_1D,
        data=b"payload-bytes",
    )

    restored = DataChunkPayload.from_dict(chunk.to_dict())

    assert restored == chunk


def test_batched_joint_data_payload_round_trip() -> None:
    payload = BatchedJointDataPayload(
        recording_id="rec-1",
        timestamp=123.456,
        dataset_id="dataset-1",
        dataset_name="dataset",
        robot_name="robot",
        robot_id="robot-1",
        robot_instance=3,
        data_type=DataType.JOINT_POSITIONS,
        items=[
            BatchedJointDataItemPayload(
                trace_id="trace-1",
                data_type_name="joint1",
                value=0.1,
            ),
            BatchedJointDataItemPayload(
                trace_id="trace-2",
                data_type_name="joint2",
                value=-0.2,
            ),
        ],
    )

    restored = BatchedJointDataPayload.from_dict(payload.to_dict())

    assert restored == payload


def test_complete_message_batch_record_round_trip() -> None:
    record = CompleteMessage.from_bytes(
        "prod",
        "rec-1",
        True,
        "trace",
        DataType.CUSTOM_1D,
        "custom_data",
        0,
        1,
        b"hello",
        None,
        None,
        None,
        None,
    )

    raw = record.to_batch_record()
    restored = CompleteMessage.iter_batch_records(raw)

    assert len(restored) == 1
    parsed = restored[0]
    assert parsed.producer_id == "prod"
    assert parsed.trace_id == "trace"
    assert parsed.recording_id == "rec-1"
    assert parsed.data_type == DataType.CUSTOM_1D
    assert parsed.data_type_name == "custom_data"
    assert parsed.robot_instance == 0
    assert datetime.fromisoformat(parsed.received_at)
    assert parsed.data == b"hello"
    assert parsed.final_chunk is True


class DummyComm:
    def __init__(self) -> None:
        self.messages = []
        self.cleaned = False
        self.socket_requested = False

    def create_producer_socket(self):
        self.socket_requested = True
        return object()

    def create_subscriber_socket(self):
        return None

    def send_message(self, message):
        self.messages.append(message)

    def cleanup_producer(self):
        self.cleaned = True


def _wait_for_envelopes(
    messages: list[MessageEnvelope], expected: int, timeout: float = 1.0
) -> None:
    """Wait for the producer sender thread to flush messages to the stub."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(messages) >= expected:
            return
        time.sleep(0.02)
    return


def _stub_producer_transport(monkeypatch) -> list[MessageEnvelope]:
    """Patch the producer ZMQ control channel to capture sent envelopes."""
    messages: list[MessageEnvelope] = []
    base = (
        "neuracore.data_daemon.communications_management.shared_transport."
        "communications_manager.CommunicationsManager"
    )

    monkeypatch.setattr(f"{base}.create_producer_socket", lambda self: None)
    monkeypatch.setattr(
        f"{base}.send_message", lambda self, message: messages.append(message)
    )
    monkeypatch.setattr(f"{base}.cleanup_producer", lambda self: None)
    return messages


def test_send_batched_joint_data_enqueues_expected_command(monkeypatch) -> None:
    messages = _stub_producer_transport(monkeypatch)
    producer = ProducerChannel(
        id="producer-joint-batch",
        recording_id="rec-1",
        data_type=DataType.JOINT_POSITIONS,
    )

    payload = BatchedJointDataPayload(
        recording_id="rec-1",
        timestamp=123.456,
        dataset_id="dataset-1",
        dataset_name="dataset",
        robot_name="robot",
        robot_id="robot-1",
        robot_instance=0,
        data_type=DataType.JOINT_POSITIONS,
        items=[
            BatchedJointDataItemPayload(
                trace_id="trace-1",
                data_type_name="joint1",
                value=0.25,
            )
        ],
    )

    try:
        producer.send_batched_joint_data(payload)
        _wait_for_envelopes(messages, expected=1)
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 1
    envelope = messages[0]
    assert envelope.command == CommandType.BATCHED_JOINT_DATA
    assert envelope.payload[CommandType.BATCHED_JOINT_DATA.value] == payload.to_dict()


def _drain_until(
    drain: Iox2DaemonDrain,
    received: list[tuple[dict, bytes]],
    expected: int,
    timeout: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout
    while len(received) < expected and time.monotonic() < deadline:
        drain.drain_all(
            lambda channel, seq, meta, chunk: received.append((meta, chunk))
        )
        time.sleep(0.02)


@pytest.mark.parametrize(
    "data_type",
    [DataType.RGB_IMAGES, DataType.DEPTH_IMAGES, DataType.POINT_CLOUDS],
)
def test_producer_routes_video_frames_over_iox2(monkeypatch, data_type) -> None:
    messages = _stub_producer_transport(monkeypatch)
    channel_id = f"vid-{uuid4().hex[:8]}"
    producer = ProducerChannel(
        id=channel_id,
        chunk_size=2,
        recording_id="rec-1",
        data_type=data_type,
    )
    drain = Iox2DaemonDrain()
    received: list[tuple[dict, bytes]] = []

    try:
        drain.register_channel(channel_id)
        producer.start_new_trace()
        producer.send_data(
            b"abcd",
            data_type=data_type,
            data_type_name="custom",
            robot_instance=2,
            robot_id="robot-1",
            robot_name="robot",
            dataset_id="dataset-1",
            dataset_name="dataset",
        )
        _drain_until(drain, received, expected=2)
    finally:
        drain.close()
        producer.stop_producer_channel()

    assert [chunk for _, chunk in received] == [b"ab", b"cd"]
    # First frame carries trace metadata, subsequent frames do not.
    assert received[0][0]["data_type"] == data_type.value
    assert "data_type" not in received[1][0]
    # Video frames must not travel over the ZMQ control channel.
    assert all(message.command != CommandType.DATA_CHUNK for message in messages)


def test_producer_video_chunks_across_multiple_frames(monkeypatch) -> None:
    _stub_producer_transport(monkeypatch)
    channel_id = f"vid-{uuid4().hex[:8]}"
    producer = ProducerChannel(
        id=channel_id,
        chunk_size=3,
        recording_id="rec-1",
        data_type=DataType.RGB_IMAGES,
    )
    drain = Iox2DaemonDrain()
    received: list[tuple[dict, bytes]] = []

    try:
        drain.register_channel(channel_id)
        producer.start_new_trace()
        # cspell:ignore cdef
        producer.send_data_parts(
            (b"ab", memoryview(b"cdef"), b"gh"),
            total_bytes=8,
            data_type=DataType.RGB_IMAGES,
            data_type_name="custom",
            robot_instance=2,
            robot_id="robot-1",
            robot_name="robot",
            dataset_id="dataset-1",
            dataset_name="dataset",
        )
        _drain_until(drain, received, expected=3)
    finally:
        drain.close()
        producer.stop_producer_channel()

    assert [chunk for _, chunk in received] == [b"abc", b"def", b"gh"]
    assert received[0][0]["recording_id"] == "rec-1"
    assert "recording_id" not in received[1][0]
    assert "recording_id" not in received[2][0]


def test_producer_send_data_parts_uses_socket_for_non_video(monkeypatch) -> None:
    messages = _stub_producer_transport(monkeypatch)
    producer = ProducerChannel(
        recording_id="rec-1",
        data_type=DataType.CUSTOM_1D,
    )

    try:
        producer.start_new_trace()
        producer.send_data(
            b"abcd",
            data_type=DataType.CUSTOM_1D,
            data_type_name="custom",
            robot_instance=2,
            robot_id="robot-1",
            robot_name="robot",
            dataset_id="dataset-1",
            dataset_name="dataset",
        )
        _wait_for_envelopes(messages, 1)
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 1
    envelope = messages[0]
    assert envelope.command == CommandType.DATA_CHUNK
    payload = DataChunkPayload.from_dict(envelope.payload["data_chunk"])
    assert payload.channel_id == producer.channel_id
    assert payload.recording_id == "rec-1"
    assert payload.trace_id == producer.trace_id
    assert payload.chunk_index == 0
    assert payload.total_chunks == 1
    assert payload.data_type == DataType.CUSTOM_1D
    assert payload.data == b"abcd"


def test_producer_video_rejects_oversized_frame(monkeypatch) -> None:
    _stub_producer_transport(monkeypatch)
    producer = ProducerChannel(
        chunk_size=16,
        recording_id="rec-1",
        max_frame_bytes=8,
        data_type=DataType.RGB_IMAGES,
    )

    try:
        producer.start_new_trace()
        with pytest.raises(PacketTooLarge):
            producer.send_data(
                b"abcdefgh",
                data_type=DataType.RGB_IMAGES,
                data_type_name="custom",
                robot_instance=2,
                robot_id="robot-1",
                dataset_id="dataset-1",
            )
    finally:
        producer.stop_producer_channel()


def test_producer_sequences_follow_enqueue_order_under_concurrent_senders(
    monkeypatch,
) -> None:
    messages = _stub_producer_transport(monkeypatch)
    producer = ProducerChannel(
        recording_id="rec-1",
        data_type=DataType.CUSTOM_1D,
    )

    first_put_entered = threading.Event()
    allow_first_put = threading.Event()
    second_send_finished = threading.Event()
    thread_errors: list[BaseException] = []

    real_put = producer._send_queue.put

    def blocked_put(item):
        if item is not None and not first_put_entered.is_set():
            first_put_entered.set()
            allow_first_put.wait(timeout=5.0)
        return real_put(item)

    monkeypatch.setattr(producer._send_queue, "put", blocked_put)

    def send_heartbeat(mark_done: threading.Event | None = None) -> None:
        try:
            producer.heartbeat()
        except BaseException as exc:  # pragma: no cover - surfaced below
            thread_errors.append(exc)
        finally:
            if mark_done is not None:
                mark_done.set()

    first_sender = threading.Thread(target=send_heartbeat, daemon=True)
    second_sender = threading.Thread(
        target=send_heartbeat,
        kwargs={"mark_done": second_send_finished},
        daemon=True,
    )

    try:
        first_sender.start()
        assert first_put_entered.wait(timeout=5.0) is True

        second_sender.start()
        time.sleep(0.1)

        assert producer.get_last_enqueued_sequence_number() == 1
        assert second_send_finished.is_set() is False

        allow_first_put.set()

        first_sender.join(timeout=5.0)
        second_sender.join(timeout=5.0)
        _wait_for_envelopes(messages, 2)
    finally:
        allow_first_put.set()
        producer.stop_producer_channel()

    assert thread_errors == []
    assert [message.sequence_number for message in messages] == [1, 2]


class DummyRecordingDiskManager:
    """Minimal recording disk manager for tests."""

    def __init__(self) -> None:
        self.messages = []

    def enqueue(self, msg):
        self.messages.append(msg)


def test_completion_worker_assembles_spooled_chunks(tmp_path) -> None:
    rdm = DummyRecordingDiskManager()
    chunk_spool = BridgeChunkSpool(tmp_path / "chunk-spool", segment_max_bytes=8)
    worker = CompletionWorker(
        chunk_spool=chunk_spool,
        recording_disk_manager=rdm,
        shard_count=1,
    )
    metadata = TraceTransportMetadata(
        recording_id="rec-1",
        data_type=DataType.RGB_IMAGES,
        data_type_name="camera_0",
        dataset_id="dataset-1",
        robot_id="robot-1",
        robot_instance=2,
    )

    try:
        worker.enqueue_chunk(
            CompletionChunkWork(
                producer_id="producer-1",
                trace_id="trace-1",
                recording_id="rec-1",
                chunk_index=0,
                total_chunks=2,
                sequence_number=1,
                chunk_spool=chunk_spool,
                chunk_spool_ref=chunk_spool.append(memoryview(b"ab")),
                trace_metadata=metadata,
                fallback_data_type=DataType.RGB_IMAGES,
            )
        )
        worker.enqueue_chunk(
            CompletionChunkWork(
                producer_id="producer-1",
                trace_id="trace-1",
                recording_id="rec-1",
                chunk_index=1,
                total_chunks=2,
                sequence_number=2,
                chunk_spool=chunk_spool,
                chunk_spool_ref=chunk_spool.append(memoryview(b"cd")),
                trace_metadata=metadata,
                fallback_data_type=DataType.RGB_IMAGES,
            )
        )

        deadline = time.monotonic() + 1.0
        while len(rdm.messages) < 1 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        worker.close()

    assert len(rdm.messages) == 1
    message = rdm.messages[0]
    assert message.trace_id == "trace-1"
    assert message.recording_id == "rec-1"
    assert message.data == b"abcd"
    assert message.data_type == DataType.RGB_IMAGES
    assert message.data_type_name == "camera_0"
    assert (tmp_path / "chunk-spool").exists() is False


def test_bridge_chunk_spool_append_recovers_from_stale_segment_size(tmp_path) -> None:
    chunk_spool = BridgeChunkSpool(tmp_path / "chunk-spool", segment_max_bytes=8)

    first_ref = chunk_spool.append(memoryview(b"ab"))
    chunk_spool._current_segment_size = 0

    second_ref = chunk_spool.append(memoryview(b"cd"))

    assert first_ref.offset == 0
    assert second_ref.offset == 2
    assert chunk_spool.materialize([first_ref, second_ref]) == b"abcd"


def test_bridge_chunk_spool_reuses_current_segment_handle_until_rotation(
    tmp_path,
) -> None:
    chunk_spool = BridgeChunkSpool(tmp_path / "chunk-spool", segment_max_bytes=3)

    first_handle = chunk_spool._current_segment_handle
    chunk_spool.append(memoryview(b"a"))
    chunk_spool.append(memoryview(b"b"))

    assert chunk_spool._current_segment_handle is first_handle

    chunk_spool.append(memoryview(b"cd"))

    assert chunk_spool._current_segment_handle is not first_handle
    assert first_handle.closed is True


def test_cleanup_removes_channel_without_heartbeat(emitter) -> None:
    daemon = Daemon(
        comm_manager=DummyComm(),
        recording_disk_manager=DummyRecordingDiskManager(),
        emitter=emitter,
    )
    channel = ChannelState(
        producer_id="stale",
        last_heartbeat=datetime.now(timezone.utc)
        - timedelta(seconds=HEARTBEAT_TIMEOUT_SECS + 1),
    )
    daemon.channels.add(channel)

    daemon._cleanup_expired_channels()

    assert daemon.channels.get("stale") is None


def test_cleanup_keeps_recent_channel(emitter) -> None:
    daemon = Daemon(
        comm_manager=DummyComm(),
        recording_disk_manager=DummyRecordingDiskManager(),
        emitter=emitter,
    )
    channel = ChannelState(
        producer_id="active",
        last_heartbeat=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    daemon.channels.add(channel)

    daemon._cleanup_expired_channels()

    assert daemon.channels.get("active") is channel


def test_heartbeat_missing_data_type_does_not_open_transport(
    caplog: pytest.LogCaptureFixture,
    emitter,
) -> None:
    daemon = Daemon(
        comm_manager=DummyComm(),
        recording_disk_manager=DummyRecordingDiskManager(),
        emitter=emitter,
    )

    with caplog.at_level(logging.WARNING):
        daemon.handle_message(
            MessageEnvelope(
                producer_id="missing-data-type",
                command=CommandType.HEARTBEAT,
                payload={},
            )
        )

    channel = daemon.channels.get("missing-data-type")
    assert channel is not None
    assert channel.transport_mode is TransportMode.NONE
    assert channel.data_type is None
    assert "missing required data_type" in caplog.text


def test_heartbeat_unknown_data_type_does_not_open_transport(
    caplog: pytest.LogCaptureFixture,
    emitter,
) -> None:
    daemon = Daemon(
        comm_manager=DummyComm(),
        recording_disk_manager=DummyRecordingDiskManager(),
        emitter=emitter,
    )

    with caplog.at_level(logging.WARNING):
        daemon.handle_message(
            MessageEnvelope(
                producer_id="unknown-data-type",
                command=CommandType.HEARTBEAT,
                payload={"data_type": "not-a-real-type"},
            )
        )

    channel = daemon.channels.get("unknown-data-type")
    assert channel is not None
    assert channel.transport_mode is TransportMode.NONE
    assert channel.data_type is None
    assert "carried unknown data_type" in caplog.text


def test_cleanup_keeps_stale_video_channel_with_pending_sequence(emitter) -> None:
    daemon = Daemon(
        comm_manager=DummyComm(),
        recording_disk_manager=DummyRecordingDiskManager(),
        emitter=emitter,
    )
    producer_id = "stale-video-producer"
    recording_id = "rec-1"
    trace_id = "trace-1"
    cutoff_sequence_number = 5

    channel = ChannelState(
        producer_id=producer_id,
        last_heartbeat=datetime.now(timezone.utc)
        - timedelta(seconds=HEARTBEAT_TIMEOUT_SECS + 1),
        trace_id=trace_id,
        last_sequence_number=cutoff_sequence_number,
    )
    channel.data_type = DataType.RGB_IMAGES
    channel.mark_video_transport_open()
    daemon.channels.add(channel)

    daemon._trace_lifecycle.register_trace(recording_id, trace_id)
    daemon._trace_lifecycle.register_trace_metadata(
        TraceMetadataRegistrationRequest(
            trace_id=trace_id,
            metadata=TraceMetadataSnapshot(
                data_type=DataType.RGB_IMAGES.value,
                data_type_name="camera_0",
            ),
        )
    )
    daemon._trace_lifecycle.handle_recording_stopped(
        MessageEnvelope(
            producer_id=None,
            command=CommandType.RECORDING_STOPPED,
            payload={
                "recording_stopped": {
                    "recording_id": recording_id,
                    "producer_stop_sequence_numbers": {
                        producer_id: cutoff_sequence_number,
                    },
                }
            },
        )
    )
    # A video frame at the cutoff is still pending spool processing.
    daemon._trace_lifecycle.mark_video_frame_sequence_pending(
        VideoFrameSequenceProgressRequest(
            producer_id=producer_id,
            sequence_number=cutoff_sequence_number,
        )
    )

    daemon._cleanup_expired_channels()

    assert daemon.channels.get(producer_id) is channel
    assert daemon._closed_producers.contains(producer_id) is False


def test_closed_producer_revived_on_heartbeat(emitter) -> None:
    daemon = Daemon(
        comm_manager=DummyComm(),
        recording_disk_manager=DummyRecordingDiskManager(),
        emitter=emitter,
    )
    producer_id = "reopened-producer"
    daemon._closed_producers.add(producer_id)

    heartbeat_calls: list[str] = []
    daemon._command_handlers[CommandType.HEARTBEAT] = (
        lambda channel, _message: heartbeat_calls.append(channel.producer_id)
    )

    # A non-heartbeat command from a closed producer is dropped.
    daemon.handle_message(
        MessageEnvelope(
            producer_id=producer_id,
            command=CommandType.TRACE_END,
            payload={"trace_end": {"trace_id": "t", "recording_id": "rec-1"}},
        )
    )

    assert daemon.channels.get(producer_id) is None
    assert heartbeat_calls == []
    assert daemon._closed_producers.contains(producer_id) is True

    # A heartbeat revives the channel.
    daemon.handle_message(
        MessageEnvelope(
            producer_id=producer_id,
            command=CommandType.HEARTBEAT,
            payload={"data_type": DataType.RGB_IMAGES.value},
        )
    )

    assert daemon.channels.get(producer_id) is not None
    assert daemon._closed_producers.contains(producer_id) is False
    assert heartbeat_calls == [producer_id]
