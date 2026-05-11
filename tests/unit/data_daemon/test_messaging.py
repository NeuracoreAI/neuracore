import json
import struct
import threading
import time
from datetime import datetime, timedelta, timezone
from multiprocessing.shared_memory import SharedMemory

import pytest
import zmq
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
    SharedSlotSequenceProgressRequest,
    TraceMetadataRegistrationRequest,
    TraceMetadataSnapshot,
)
from neuracore.data_daemon.communications_management.producer.producer_channel import (
    ProducerChannel,
)
from neuracore.data_daemon.communications_management.shared_transport import (
    shared_slot_transport as shared_slot_transport_module,
)
from neuracore.data_daemon.communications_management.shared_transport.models import (
    QueuedSharedSlotPacket,
)
from neuracore.data_daemon.communications_management.shared_transport.registry import (
    SharedSlotRegistry,
)
from neuracore.data_daemon.const import (
    DEFAULT_VIDEO_ACK_TIMEOUT_SECONDS,
    DEFAULT_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS,
    DEFAULT_VIDEO_SLOT_COUNT,
    HEARTBEAT_TIMEOUT_SECS,
    SHARED_MEMORY_RECORD_HEADER_FORMAT,
)
from neuracore.data_daemon.models import (
    BatchedJointDataItemPayload,
    BatchedJointDataPayload,
    CommandType,
    CompleteMessage,
    DataChunkPayload,
    MessageEnvelope,
    SharedSlotCreditReturn,
    SharedSlotDescriptor,
    SharedSlotOpenFailedModel,
    SharedSlotReadyModel,
    TraceTransportMetadata,
)


def _decode_shared_memory_write(packet: bytes) -> tuple[dict, bytes]:
    _magic, metadata_len, chunk_len = struct.unpack(
        SHARED_MEMORY_RECORD_HEADER_FORMAT,
        packet[: struct.calcsize(SHARED_MEMORY_RECORD_HEADER_FORMAT)],
    )
    metadata_start = struct.calcsize(SHARED_MEMORY_RECORD_HEADER_FORMAT)
    metadata_end = metadata_start + metadata_len
    metadata = json.loads(packet[metadata_start:metadata_end].decode("utf-8"))
    chunk = packet[metadata_end : metadata_end + chunk_len]
    return metadata, chunk


def _read_shared_slot_packet(envelope: MessageEnvelope) -> tuple[dict, bytes]:
    descriptor = SharedSlotDescriptor.from_dict(
        envelope.payload[CommandType.SHARED_SLOT_DESCRIPTOR.value]
    )
    shm = SharedMemory(name=descriptor.shm_name)
    try:
        packet = bytes(
            shm.buf[descriptor.offset : descriptor.offset + descriptor.length]
        )
    finally:
        shm.close()
    return parse_shared_frame_packet(packet)


def test_message_envelope_round_trip() -> None:
    payload = {
        "open_fixed_shared_slots": {
            "transport_mode": "FIXED_SHARED_SLOTS_DAEMON_OWNED",
            "control_endpoint": "ipc://test-message-round-trip",
            "slot_size": 2048,
            "slot_count": 16,
        }
    }
    envelope = MessageEnvelope(
        producer_id="producer-123",
        command=CommandType.OPEN_FIXED_SHARED_SLOTS,
        payload=payload,
    )

    parsed = MessageEnvelope.from_bytes(envelope.to_bytes())

    assert parsed.producer_id == "producer-123"
    assert parsed.command == CommandType.OPEN_FIXED_SHARED_SLOTS
    assert parsed.payload == payload


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
    assert parsed.dataset_id is None
    assert parsed.dataset_name is None
    assert parsed.robot_name is None
    assert parsed.robot_id is None
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


def _wait_for_messages(comm: DummyComm, expected: int, timeout: float = 1.0) -> None:
    """Wait for ProducerChannel's sender thread to flush messages to DummyComm."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(comm.messages) >= expected:
            return
        time.sleep(0.02)
    return


def _wait_for_envelopes(
    messages: list[MessageEnvelope], expected: int, timeout: float = 1.0
) -> None:
    """Wait for a stubbed producer transport to capture messages."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(messages) >= expected:
            return
        time.sleep(0.02)
    return


def _stub_producer_transport(monkeypatch) -> list[MessageEnvelope]:
    messages: list[MessageEnvelope] = []
    control_context = zmq.Context()
    control_sockets: dict[str, zmq.Socket] = {}
    control_endpoints: dict[str, str] = {}
    shared_memories: dict[str, SharedMemory] = {}

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.shared_transport.communications_manager.CommunicationsManager.create_producer_socket",
        lambda self: None,
    )

    def _send_message(_self, message):
        messages.append(message)

        if message.command == CommandType.OPEN_FIXED_SHARED_SLOTS:
            payload = message.payload["open_fixed_shared_slots"]
            control_endpoint = payload["control_endpoint"]
            shm_name = f"neuracore-slots-test-{len(shared_memories)}"
            shm = SharedMemory(
                name=shm_name,
                create=True,
                size=int(payload["slot_size"]) * int(payload["slot_count"]),
            )
            shared_memories[shm_name] = shm
            control_endpoints[str(message.producer_id)] = control_endpoint
            socket_obj = control_sockets.get(control_endpoint)
            if socket_obj is None:
                socket_obj = control_context.socket(zmq.PUSH)
                socket_obj.setsockopt(zmq.LINGER, 0)
                socket_obj.connect(control_endpoint)
                control_sockets[control_endpoint] = socket_obj
            socket_obj.send(
                MessageEnvelope(
                    producer_id=None,
                    command=CommandType.SHARED_SLOT_READY,
                    payload={
                        CommandType.SHARED_SLOT_READY.value: SharedSlotReadyModel(
                            shm_name=shm_name,
                            slot_size=int(payload["slot_size"]),
                            slot_count=int(payload["slot_count"]),
                        ).model_dump()
                    },
                ).to_bytes()
            )
            return

        if message.command == CommandType.SHARED_SLOT_DESCRIPTOR:
            descriptor = SharedSlotDescriptor.from_dict(
                message.payload[CommandType.SHARED_SLOT_DESCRIPTOR.value]
            )
            control_endpoint = control_endpoints.get(str(message.producer_id))
            if control_endpoint is None:
                raise RuntimeError(
                    "Missing control endpoint for shared-slot descriptor"
                )
            socket_obj = control_sockets[control_endpoint]
            socket_obj.send(
                MessageEnvelope(
                    producer_id=None,
                    command=CommandType.SHARED_SLOT_CREDIT_RETURN,
                    payload={
                        CommandType.SHARED_SLOT_CREDIT_RETURN.value: (
                            SharedSlotCreditReturn(
                                shm_name=descriptor.shm_name,
                                slot_id=descriptor.slot_id,
                                sequence_id=descriptor.sequence_id,
                            ).to_dict()
                        )
                    },
                ).to_bytes()
            )

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.shared_transport.communications_manager.CommunicationsManager.send_message",
        _send_message,
    )

    def _cleanup_producer(_self) -> None:
        for socket_obj in control_sockets.values():
            socket_obj.close(0)
        control_sockets.clear()
        for shm in shared_memories.values():
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        shared_memories.clear()
        control_context.term()

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.shared_transport.communications_manager.CommunicationsManager.cleanup_producer",
        _cleanup_producer,
    )

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


def test_producer_open_fixed_shared_slots_sends_payload(monkeypatch) -> None:
    messages = _stub_producer_transport(monkeypatch)
    producer = ProducerChannel(data_type=DataType.RGB_IMAGES)

    try:
        producer.open_fixed_shared_slots(slot_size=2048)
        _wait_for_envelopes(messages, 1)
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 1
    envelope = messages[0]
    assert envelope.command == CommandType.OPEN_FIXED_SHARED_SLOTS
    payload = envelope.payload["open_fixed_shared_slots"]
    assert payload["slot_size"] == 2048
    assert payload["slot_count"] == DEFAULT_VIDEO_SLOT_COUNT
    assert payload["control_endpoint"].startswith("ipc://")


def test_producer_send_data_parts_lazily_opens_shared_memory(
    monkeypatch,
) -> None:
    messages = _stub_producer_transport(monkeypatch)

    producer = ProducerChannel(
        chunk_size=2,
        recording_id="rec-1",
        shared_memory_size=2048,
        data_type=DataType.RGB_IMAGES,
    )

    try:
        producer.start_new_trace()
        producer.send_data(
            b"abcd",
            data_type=DataType.RGB_IMAGES,
            data_type_name="custom",
            robot_instance=2,
            robot_id="robot-1",
            robot_name="robot",
            dataset_id="dataset-1",
            dataset_name="dataset",
        )
        _wait_for_envelopes(messages, 3)
        first_metadata, first_chunk = _read_shared_slot_packet(messages[1])
        second_metadata, second_chunk = _read_shared_slot_packet(messages[2])
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 3
    assert messages[0].command == CommandType.OPEN_FIXED_SHARED_SLOTS
    assert messages[1].command == CommandType.SHARED_SLOT_DESCRIPTOR
    assert messages[2].command == CommandType.SHARED_SLOT_DESCRIPTOR
    assert first_chunk == b"ab"
    assert second_chunk == b"cd"


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


def test_producer_ensure_shared_memory_does_not_reannounce(
    monkeypatch,
) -> None:
    messages = _stub_producer_transport(monkeypatch)

    producer = ProducerChannel(
        recording_id="rec-1",
        shared_memory_size=2048,
        data_type=DataType.RGB_IMAGES,
    )

    try:
        producer.start_new_trace()
        producer.open_fixed_shared_slots(slot_size=2048)
        _wait_for_envelopes(messages, 1)
        control_endpoint = messages[0].payload["open_fixed_shared_slots"][
            "control_endpoint"
        ]
        trace_id = producer.trace_id

        producer.send_data(
            b"ab",
            data_type=DataType.RGB_IMAGES,
            data_type_name="custom",
            robot_instance=2,
            robot_id="robot-1",
            robot_name="robot",
            dataset_id="dataset-1",
            dataset_name="dataset",
        )
        _wait_for_envelopes(messages, 2)
        metadata, _chunk = _read_shared_slot_packet(messages[1])
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 2
    assert (
        messages[0].payload["open_fixed_shared_slots"]["control_endpoint"]
        == control_endpoint
    )
    assert metadata["trace_id"] == trace_id


def test_producer_send_data_parts_chunks_across_multiple_buffers(monkeypatch) -> None:
    messages = _stub_producer_transport(monkeypatch)

    producer = ProducerChannel(
        chunk_size=3,
        recording_id="rec-1",
        shared_memory_size=2048,
        data_type=DataType.RGB_IMAGES,
    )

    try:
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
        _wait_for_envelopes(messages, 4)
        packets = [_read_shared_slot_packet(packet) for packet in messages[1:]]
    finally:
        producer.stop_producer_channel()

    assert len(messages) == 4
    envelope = messages[0]
    assert envelope.command == CommandType.OPEN_FIXED_SHARED_SLOTS
    payload = envelope.payload["open_fixed_shared_slots"]
    assert payload["slot_size"] == 2048
    assert payload["control_endpoint"].startswith("ipc://")
    assert [chunk for _, chunk in packets] == [b"abc", b"def", b"gh"]
    assert packets[0][0]["recording_id"] == "rec-1"
    assert "recording_id" not in packets[1][0]
    assert "recording_id" not in packets[2][0]


def test_producer_shared_memory_rejects_oversized_packet(monkeypatch) -> None:
    monkeypatch.setenv("NDD_DEBUG", "true")
    _stub_producer_transport(monkeypatch)

    producer = ProducerChannel(
        chunk_size=16,
        recording_id="rec-1",
        shared_memory_size=8,
        data_type=DataType.RGB_IMAGES,
    )

    try:
        producer.start_new_trace()
        try:
            producer.send_data(
                b"abcdefgh",
                data_type=DataType.RGB_IMAGES,
                data_type_name="custom",
                robot_instance=2,
                robot_id="robot-1",
                dataset_id="dataset-1",
            )
        except PacketTooLarge:
            oversized = True
        else:
            oversized = False
    finally:
        producer.stop_producer_channel()

    assert oversized is True


def test_producer_sender_failure_stops_waiters(monkeypatch) -> None:
    monkeypatch.setenv("NDD_DEBUG", "true")
    sent = {"count": 0}
    control_context = zmq.Context()
    control_sockets: dict[str, zmq.Socket] = {}
    shared_memories: dict[str, SharedMemory] = {}

    def flaky_send(_self, message):
        sent["count"] += 1
        if message.command == CommandType.OPEN_FIXED_SHARED_SLOTS:
            payload = message.payload["open_fixed_shared_slots"]
            control_endpoint = payload["control_endpoint"]
            shm_name = "neuracore-slots-test-failure"
            shm = SharedMemory(
                name=shm_name,
                create=True,
                size=int(payload["slot_size"]) * int(payload["slot_count"]),
            )
            shared_memories[shm_name] = shm
            socket_obj = control_sockets.get(control_endpoint)
            if socket_obj is None:
                socket_obj = control_context.socket(zmq.PUSH)
                socket_obj.setsockopt(zmq.LINGER, 0)
                socket_obj.connect(control_endpoint)
                control_sockets[control_endpoint] = socket_obj
            socket_obj.send(
                MessageEnvelope(
                    producer_id=None,
                    command=CommandType.SHARED_SLOT_READY,
                    payload={
                        CommandType.SHARED_SLOT_READY.value: SharedSlotReadyModel(
                            shm_name=shm_name,
                            slot_size=int(payload["slot_size"]),
                            slot_count=int(payload["slot_count"]),
                        ).model_dump()
                    },
                ).to_bytes()
            )
            return
        if message.command == CommandType.SHARED_SLOT_DESCRIPTOR:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.shared_transport.communications_manager.CommunicationsManager.create_producer_socket",
        lambda self: None,
    )
    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.shared_transport.communications_manager.CommunicationsManager.send_message",
        flaky_send,
    )

    def _cleanup_producer(_self) -> None:
        for socket_obj in control_sockets.values():
            socket_obj.close(0)
        control_sockets.clear()
        for shm in shared_memories.values():
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        shared_memories.clear()
        control_context.term()

    monkeypatch.setattr(
        "neuracore.data_daemon.communications_management.shared_transport.communications_manager.CommunicationsManager.cleanup_producer",
        _cleanup_producer,
    )

    producer = ProducerChannel(
        chunk_size=2,
        recording_id="rec-1",
        shared_memory_size=2048,
        data_type=DataType.RGB_IMAGES,
    )

    try:
        producer.start_new_trace()
        producer.send_data(
            b"abcd",
            data_type=DataType.RGB_IMAGES,
            data_type_name="custom",
            robot_instance=2,
            robot_id="robot-1",
            dataset_id="dataset-1",
        )
        deadline = time.monotonic() + 1.0
        wait_result = True
        try:
            while time.monotonic() < deadline:
                try:
                    producer._send(CommandType.HEARTBEAT, {})
                except RuntimeError:
                    wait_result = False
                    break
                time.sleep(0.02)
        finally:
            if wait_result and time.monotonic() >= deadline:
                wait_result = True
    finally:
        producer.stop_producer_channel()

    assert sent["count"] >= 1
    assert wait_result is False


def test_shared_slot_video_worker_surfaces_background_failure() -> None:
    registry = SharedSlotRegistry(
        slot_size=2048,
        slot_count=2,
        ack_timeout_s=DEFAULT_VIDEO_ACK_TIMEOUT_SECONDS,
        allocate_timeout_s=DEFAULT_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS,
    )
    worker = SharedSlotVideoWorker.acquire(registry)

    def raise_worker_error(_item) -> None:
        raise RuntimeError("boom")

    worker._process_item = raise_worker_error  # type: ignore[method-assign]

    packet = QueuedSharedSlotPacket(
        producer_id="producer-1",
        sender=None,  # type: ignore[arg-type]
        metadata_bytes=b"{}",
        chunk=b"x",
        packet_length=1,
        sequence_number=1,
    )

    try:
        worker.enqueue_packet(packet=packet)
        deadline = time.monotonic() + 1.0
        while worker._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)

        with pytest.raises(RuntimeError, match="Shared-slot video worker failed"):
            worker.enqueue_packet(packet=packet)
    finally:
        SharedSlotVideoWorker.reset_shared_instance_for_tests()
        registry.close()


def test_shared_slot_transport_snapshots_chunk_before_background_handoff() -> None:
    transport = shared_slot_transport_module.SharedSlotVideoTransport()
    captured_packets: list[QueuedSharedSlotPacket] = []

    try:
        transport._worker.enqueue_packet = (  # type: ignore[method-assign]
            lambda *, packet: captured_packets.append(packet)
        )

        source = bytearray(b"frame-000")
        transport.enqueue_packet(
            producer_id="producer-1",
            sender=None,  # type: ignore[arg-type]
            metadata={"trace_id": "trace-1", "chunk_index": 0, "total_chunks": 1},
            chunk=memoryview(source),
        )

        source[:] = b"frame-999"

        assert len(captured_packets) == 1
        assert isinstance(captured_packets[0].chunk, bytes)
        assert captured_packets[0].chunk == b"frame-000"
    finally:
        transport.close()


def test_shared_slot_timeout_clock_starts_after_socket_send() -> None:
    registry = SharedSlotRegistry(
        slot_size=2048,
        slot_count=2,
        ack_timeout_s=0.01,
        allocate_timeout_s=0.01,
    )
    shm_name = f"test-credit-timeout-{time.time_ns()}"
    shm = SharedMemory(name=shm_name, create=True, size=2048 * 2)

    try:
        registry._apply_ready_message(
            SharedSlotReadyModel(
                shm_name=shm_name,
                slot_size=2048,
                slot_count=2,
            )
        )
        slot_id, _offset = registry.allocate_slot()
        sequence_id = registry.mark_in_flight(slot_id=slot_id, sequence_id=1)

        time.sleep(0.03)
        with registry._condition:
            registry._check_for_timeouts_locked()
        assert registry.is_healthy() is True
        assert sequence_id in registry._state.in_flight

        registry.mark_sent(sequence_id)
        time.sleep(0.03)
        with registry._condition:
            registry._check_for_timeouts_locked()
        assert registry.is_healthy() is False
    finally:
        registry.close()
        shm.close()
        shm.unlink()


def test_shared_slot_registry_runtime_starts_and_stops_cleanly() -> None:
    registry = SharedSlotRegistry(
        slot_size=2048,
        slot_count=2,
        ack_timeout_s=DEFAULT_VIDEO_ACK_TIMEOUT_SECONDS,
        allocate_timeout_s=DEFAULT_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS,
    )

    try:
        assert registry.control_endpoint.startswith("ipc://")
        assert registry._runtime.control_thread.is_alive()
        assert registry._runtime.watchdog_thread.is_alive()
    finally:
        registry.close()

    assert not registry._runtime.control_thread.is_alive()
    assert not registry._runtime.watchdog_thread.is_alive()


def test_shared_slot_ready_message_populates_free_slots() -> None:
    registry = SharedSlotRegistry(
        slot_size=2048,
        slot_count=3,
        ack_timeout_s=DEFAULT_VIDEO_ACK_TIMEOUT_SECONDS,
        allocate_timeout_s=DEFAULT_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS,
    )
    shm_name = f"test-ready-populates-free-slots-{time.time_ns()}"
    shm = SharedMemory(name=shm_name, create=True, size=2048 * 3)

    try:
        registry._apply_ready_message(
            SharedSlotReadyModel(
                shm_name=shm_name,
                slot_size=2048,
                slot_count=3,
            )
        )

        assert registry.is_ready() is True
        assert len(registry._state.free_slots) == 3
        assert registry.slot_size == 2048
        assert registry.slot_count == 3
    finally:
        registry.close()
        shm.close()
        shm.unlink()


def test_shared_slot_ready_message_adopts_daemon_slot_dimensions() -> None:
    registry = SharedSlotRegistry(
        slot_size=1024,
        slot_count=1,
        ack_timeout_s=DEFAULT_VIDEO_ACK_TIMEOUT_SECONDS,
        allocate_timeout_s=DEFAULT_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS,
    )
    shm_name = f"test-ready-adopts-slot-dimensions-{time.time_ns()}"
    shm = SharedMemory(name=shm_name, create=True, size=4096 * 4)

    try:
        registry._apply_ready_message(
            SharedSlotReadyModel(
                shm_name=shm_name,
                slot_size=4096,
                slot_count=4,
            )
        )

        assert registry.slot_size == 4096
        assert registry.slot_count == 4
        assert registry.slot_size * registry.slot_count == 4096 * 4
        assert len(registry._state.free_slots) == 4
    finally:
        registry.close()
        shm.close()
        shm.unlink()


def test_shared_slot_open_failure_message_surfaces_daemon_error() -> None:
    registry = SharedSlotRegistry(
        slot_size=2048,
        slot_count=3,
        ack_timeout_s=DEFAULT_VIDEO_ACK_TIMEOUT_SECONDS,
        allocate_timeout_s=DEFAULT_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS,
    )
    error_message = (
        "Not enough shared-memory for data throughput requirements. "
        "remaining=3.31MiB"
    )

    try:
        registry._process_control_message(
            MessageEnvelope(
                producer_id=None,
                command=CommandType.SHARED_SLOT_OPEN_FAILED,
                payload={
                    CommandType.SHARED_SLOT_OPEN_FAILED.value: (
                        SharedSlotOpenFailedModel(
                            error_message=error_message
                        ).model_dump()
                    )
                },
            )
        )

        assert registry.is_healthy() is False
        assert registry.is_ready() is False
        assert registry._state.unhealthy_reason == "open_failed"
        assert registry._state.failure_message == error_message

        with pytest.raises(RuntimeError, match="Not enough shared-memory"):
            registry.ensure_healthy()

        with pytest.raises(RuntimeError, match="Not enough shared-memory"):
            registry.allocate_slot()
    finally:
        registry.close()


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


def test_cleanup_keeps_stale_shared_memory_channel_with_pending_descriptor(
    emitter, monkeypatch
) -> None:
    daemon = Daemon(
        comm_manager=DummyComm(),
        recording_disk_manager=DummyRecordingDiskManager(),
        emitter=emitter,
    )
    producer_id = "stale-shm-producer"
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
    channel.mark_shared_slot_transport_open(
        control_endpoint="ipc://test-shared-slot-control",
        shm_name="neuracore-slots-test",
    )
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
    daemon._trace_lifecycle.mark_shared_slot_sequence_pending(
        SharedSlotSequenceProgressRequest(
            producer_id=producer_id,
            sequence_number=cutoff_sequence_number,
        )
    )

    cleanup_calls: list[str] = []
    monkeypatch.setattr(
        daemon._shared_slot_handler,
        "cleanup_channel_resources",
        lambda cleanup_channel: cleanup_calls.append(cleanup_channel.producer_id),
    )

    daemon._cleanup_expired_channels()

    assert daemon.channels.get(producer_id) is channel
    assert cleanup_calls == []
    assert daemon._closed_producers.contains(producer_id) is False


def test_closed_producer_drops_stale_messages_until_reopened(
    emitter, monkeypatch
) -> None:
    daemon = Daemon(
        comm_manager=DummyComm(),
        recording_disk_manager=DummyRecordingDiskManager(),
        emitter=emitter,
    )
    producer_id = "reopened-producer"
    daemon._closed_producers.add(producer_id)

    open_calls: list[str] = []
    heartbeat_calls: list[str] = []
    monkeypatch.setattr(
        daemon._shared_slot_handler,
        "handle_open",
        lambda channel, _payload: open_calls.append(channel.producer_id),
    )
    daemon._command_handlers[CommandType.HEARTBEAT] = (
        lambda channel, _message: heartbeat_calls.append(channel.producer_id)
    )

    daemon.handle_message(
        MessageEnvelope(
            producer_id=producer_id,
            command=CommandType.HEARTBEAT,
            payload={},
        )
    )

    assert daemon.channels.get(producer_id) is None
    assert heartbeat_calls == []
    assert daemon._closed_producers.contains(producer_id) is True

    daemon.handle_message(
        MessageEnvelope(
            producer_id=producer_id,
            command=CommandType.OPEN_FIXED_SHARED_SLOTS,
            payload={"open_fixed_shared_slots": {"control_endpoint": "ipc://test"}},
        )
    )

    assert daemon.channels.get(producer_id) is not None
    assert open_calls == [producer_id]
    assert daemon._closed_producers.contains(producer_id) is False

    daemon.handle_message(
        MessageEnvelope(
            producer_id=producer_id,
            command=CommandType.HEARTBEAT,
            payload={},
        )
    )

    assert heartbeat_calls == [producer_id]


BridgeChunkSpool = bridge_chunk_spool_module.BridgeChunkSpool
PacketTooLarge = shared_slot_transport_module.PacketTooLarge
SharedSlotVideoWorker = shared_slot_transport_module.SharedSlotVideoWorker
parse_shared_frame_packet = shared_slot_transport_module.parse_shared_frame_packet
