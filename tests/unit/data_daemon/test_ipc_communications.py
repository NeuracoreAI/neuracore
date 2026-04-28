import logging
import time
from collections.abc import Generator
from enum import Enum
from uuid import uuid4

import pytest
import zmq
from neuracore_types import DataType

import neuracore.data_daemon.const as const_module
import neuracore.data_daemon.communications_management.shared_transport.communications_manager as comms_module
from neuracore.data_daemon.communications_management.shared_transport.communications_manager import (
    CommunicationsManager,
)
from neuracore.data_daemon.communications_management.data_bridge import DataBridge
from neuracore.data_daemon.communications_management.producer.producer_channel import (
    ProducerChannel,
)
from neuracore.data_daemon.models import CommandType, DataChunkPayload, MessageEnvelope


class CaptureRDM:
    def __init__(self) -> None:
        self.enqueued = []

    def enqueue(self, message) -> None:
        self.enqueued.append(message)


@pytest.fixture(autouse=True)
def ipc_paths(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    base_dir = tmp_path / "ndd"
    socket_path = f"inproc://daemon-{uuid4().hex}"
    events_path = f"inproc://events-{uuid4().hex}"

    mpsa = monkeypatch.setattr

    mpsa(const_module, "BASE_DIR", base_dir)
    mpsa(const_module, "SOCKET_PATH", socket_path)
    mpsa(comms_module, "BASE_DIR", base_dir)
    mpsa(comms_module, "SOCKET_PATH", socket_path)

    yield

    for path in (socket_path, events_path):
        if hasattr(path, "unlink"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    try:
        base_dir.rmdir()
    except OSError:
        pass


@pytest.fixture
def zmq_context() -> Generator[zmq.Context, None, None]:
    context = zmq.Context.instance()
    yield context


def _drain_messages(
    daemon: DataBridge,
    comm: CommunicationsManager,
    expected: int,
    timeout: float = 2.0,
    until=None,
) -> None:
    poller = zmq.Poller()
    poller.register(comm._consumer_socket, zmq.POLLIN)
    received = 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        if received >= expected and (until is None or until()):
            return
        poll_timeout_ms = max(1, int(min(remaining, 0.05) * 1000))
        events = dict(poller.poll(poll_timeout_ms))
        if comm._consumer_socket in events:
            raw = comm.receive_raw()
            if raw is None:
                continue
            message = MessageEnvelope.from_bytes(raw)
            daemon.handle_message(message)
            received += 1
    assert received == expected
    if until is not None:
        assert until()


def test_daemon_singleton_socket_enforced(zmq_context: zmq.Context) -> None:
    daemon_comm = CommunicationsManager(context=zmq_context)
    daemon_comm.start_consumer()
    try:
        second_comm = CommunicationsManager(context=zmq_context)
        with pytest.raises(SystemExit) as exc:
            second_comm.start_consumer()
        assert exc.value.code == 1
    finally:
        daemon_comm.cleanup_daemon()
        if second_comm._consumer_socket is not None:
            second_comm._consumer_socket.close(0)


def test_create_producer_socket_returns_continues_without_daemon(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path = tmp_path / "ndd" / "management.sock"
    monkeypatch.setattr(const_module, "SOCKET_PATH", socket_path)
    monkeypatch.setattr(comms_module, "SOCKET_PATH", socket_path)

    comm = CommunicationsManager()
    assert not socket_path.exists()
    comm.create_producer_socket()
    assert isinstance(comm._producer_socket, zmq.Socket)
    assert (
        comm._producer_socket.getsockopt(zmq.IMMEDIATE) == 1
    ), "Socket should be setup in IMMEDIATE mode"
    comm.cleanup_producer()


def test_message_envelope_round_trip_bytes() -> None:
    envelope = MessageEnvelope(
        producer_id="producer-abc",
        command=CommandType.OPEN_FIXED_SHARED_SLOTS,
        payload={
            "open_fixed_shared_slots": {
                "transport_mode": "FIXED_SHARED_SLOTS_DAEMON_OWNED",
                "control_endpoint": "ipc://test-envelope-round-trip",
                "slot_size": 4096,
                "slot_count": 16,
            }
        },
    )

    parsed = MessageEnvelope.from_bytes(envelope.to_bytes())

    assert parsed == envelope


def test_pub_sub_recording_stopped_event(zmq_context: zmq.Context) -> None:
    daemon_comm = CommunicationsManager(context=zmq_context)
    daemon_comm.start_consumer()

    producer_comm = CommunicationsManager(context=zmq_context)
    producer_comm.create_producer_socket()

    assert producer_comm._producer_socket is not None
    time.sleep(0.05)

    payload = {"recording_stopped": {"recording_id": "rec-1"}}
    producer_comm.send_message(
        MessageEnvelope(
            producer_id=None,
            command=CommandType.RECORDING_STOPPED,
            payload=payload,
        )
    )

    poller = zmq.Poller()
    poller.register(daemon_comm._consumer_socket, zmq.POLLIN)
    events = dict(poller.poll(1000))
    assert daemon_comm._consumer_socket in events

    raw = daemon_comm.receive_raw()
    parsed = MessageEnvelope.from_bytes(raw)
    assert parsed.command == CommandType.RECORDING_STOPPED
    assert parsed.payload == payload

    producer_comm.cleanup_producer()
    daemon_comm.cleanup_daemon()


def test_large_payload_chunked_round_trip_over_ipc(
    zmq_context: zmq.Context, emitter
) -> None:
    daemon_comm = CommunicationsManager(context=zmq_context)
    daemon_comm.start_consumer()
    rdm = CaptureRDM()
    daemon = DataBridge(
        comm_manager=daemon_comm,
        recording_disk_manager=rdm,
        emitter=emitter,
    )

    producer = ProducerChannel(
        id="producer-large",
        context=zmq_context,
        chunk_size=16 * 1024,
        recording_id="rec-large",
        data_type=DataType.CUSTOM_1D,
    )

    producer.start_new_trace()

    payload = b"x" * 50_000
    producer.send_data(
        payload,
        data_type=DataType.CUSTOM_1D,
        data_type_name="custom",
        robot_instance=1,
        robot_id="robot-1",
        robot_name="robot",
        dataset_id="dataset-1",
        dataset_name="dataset",
    )
    _drain_messages(
        daemon,
        daemon_comm,
        expected=1,
        until=lambda: len(rdm.enqueued) == 1,
    )

    assert len(rdm.enqueued) == 1
    assert rdm.enqueued[0].data == payload

    producer.stop_producer_channel()
    daemon_comm.cleanup_daemon()


def test_two_producers_route_to_own_channels(zmq_context: zmq.Context, emitter) -> None:
    daemon_comm = CommunicationsManager(context=zmq_context)
    daemon_comm.start_consumer()
    rdm = CaptureRDM()
    daemon = DataBridge(
        comm_manager=daemon_comm,
        recording_disk_manager=rdm,
        emitter=emitter,
    )

    producer_a = ProducerChannel(
        id="producer-a",
        context=zmq_context,
        chunk_size=8,
        recording_id="rec-a",
        data_type=DataType.CUSTOM_1D,
    )
    producer_b = ProducerChannel(
        id="producer-b",
        context=zmq_context,
        chunk_size=8,
        recording_id="rec-b",
        data_type=DataType.CUSTOM_1D,
    )

    producer_a.start_new_trace()
    producer_b.start_new_trace()

    payload_a = b"payload-a"
    payload_b = b"payload-b"

    producer_a.send_data(
        payload_a,
        data_type=DataType.CUSTOM_1D,
        data_type_name="custom",
        robot_instance=1,
        robot_id="robot-a",
        dataset_id="dataset-a",
    )
    producer_b.send_data(
        payload_b,
        data_type=DataType.CUSTOM_1D,
        data_type_name="custom",
        robot_instance=2,
        robot_id="robot-b",
        dataset_id="dataset-b",
    )
    _drain_messages(
        daemon,
        daemon_comm,
        expected=2,
        until=lambda: len(rdm.enqueued) == 2,
    )

    by_producer = {msg.producer_id: msg.data for msg in rdm.enqueued}
    assert by_producer["producer-a"] == payload_a
    assert by_producer["producer-b"] == payload_b

    producer_a.stop_producer_channel()
    producer_b.stop_producer_channel()
    daemon_comm.cleanup_daemon()


def test_interleaved_chunks_reassemble_per_producer(
    zmq_context: zmq.Context, emitter
) -> None:
    daemon_comm = CommunicationsManager(context=zmq_context)
    daemon_comm.start_consumer()
    rdm = CaptureRDM()
    daemon = DataBridge(
        comm_manager=daemon_comm,
        recording_disk_manager=rdm,
        emitter=emitter,
    )

    producer_a_comm = CommunicationsManager(context=zmq_context)
    producer_b_comm = CommunicationsManager(context=zmq_context)

    producer_a_comm.create_producer_socket()
    producer_b_comm.create_producer_socket()

    def send_open(comm: CommunicationsManager, producer_id: str) -> None:
        comm.send_message(
            MessageEnvelope(
                producer_id=producer_id,
                command=CommandType.HEARTBEAT,
                payload={CommandType.HEARTBEAT.value: {}},
            ),
        )

    send_open(producer_a_comm, "producer-a")
    send_open(producer_b_comm, "producer-b")
    _drain_messages(daemon, daemon_comm, expected=2)

    payload_a = b"AAAAAA"
    payload_b = b"BBBBBB"
    chunks_a = [payload_a[:2], payload_a[2:4], payload_a[4:]]
    chunks_b = [payload_b[:2], payload_b[2:4], payload_b[4:]]

    def make_chunk(
        producer_id: str,
        recording_id: str,
        trace_id: str,
        idx: int,
        total: int,
        data: bytes,
    ):
        payload = DataChunkPayload(
            channel_id=producer_id,
            recording_id=recording_id,
            trace_id=trace_id,
            chunk_index=idx,
            total_chunks=total,
            data_type=DataType.CUSTOM_1D,
            data_type_name="custom",
            dataset_id="dataset",
            dataset_name=None,
            robot_name=None,
            robot_id="robot",
            robot_instance=1,
            data=data,
        )
        return MessageEnvelope(
            producer_id=producer_id,
            command=CommandType.DATA_CHUNK,
            payload={"data_chunk": payload.to_dict()},
        )

    interleaved = []
    for idx in range(3):
        interleaved.append((
            producer_a_comm,
            make_chunk("producer-a", "rec-a", "trace-a", idx, 3, chunks_a[idx]),
        ))
        interleaved.append((
            producer_b_comm,
            make_chunk("producer-b", "rec-b", "trace-b", idx, 3, chunks_b[idx]),
        ))

    for comm, envelope in interleaved:
        comm.send_message(envelope)

    _drain_messages(daemon, daemon_comm, expected=len(interleaved))

    by_producer = {msg.producer_id: msg.data for msg in rdm.enqueued}
    assert by_producer["producer-a"] == payload_a
    assert by_producer["producer-b"] == payload_b

    daemon_comm.cleanup_daemon()


def test_trace_id_required_on_send_data() -> None:
    """send_data() requires start_new_trace() to be called first."""
    producer = ProducerChannel(
        recording_id="rec-1",
        data_type=DataType.CUSTOM_1D,
    )

    with pytest.raises(ValueError, match="Trace ID required"):
        producer.send_data(
            b"data",
            data_type=DataType.CUSTOM_1D,
            data_type_name="custom",
            robot_instance=1,
            robot_id="robot",
            dataset_id="dataset",
        )

    producer.stop_producer_channel()


def test_recording_id_required_on_start_new_trace() -> None:
    """start_new_trace() requires recording_id to be set on init."""
    producer = ProducerChannel(
        data_type=DataType.CUSTOM_1D,
    )

    with pytest.raises(ValueError, match="recording_id is required"):
        producer.start_new_trace()

    producer.stop_producer_channel()


def test_unknown_command_logs_warning_and_continues(
    caplog: pytest.LogCaptureFixture,
    emitter,
) -> None:
    class FakeCommand(Enum):
        UNKNOWN = "unknown_command"

    daemon = DataBridge(
        comm_manager=CommunicationsManager(),
        recording_disk_manager=CaptureRDM(),
        emitter=emitter,
    )

    with caplog.at_level(logging.WARNING):
        daemon.handle_message(
            MessageEnvelope(
                producer_id="producer-1",
                command=FakeCommand.UNKNOWN,
                payload={},
            )
        )
    assert "Unknown command" in caplog.text

    daemon.handle_message(
        MessageEnvelope(
            producer_id="producer-1",
            command=CommandType.HEARTBEAT,
            payload={CommandType.HEARTBEAT.value: {}},
        )
    )
    assert daemon.channels.get("producer-1").is_open()


def test_garbage_messages_are_logged_and_daemon_survives(
    caplog: pytest.LogCaptureFixture,
    zmq_context: zmq.Context,
    monkeypatch: pytest.MonkeyPatch,
    emitter,
) -> None:
    daemon_comm = CommunicationsManager(context=zmq_context)
    daemon_comm.start_consumer()
    daemon = DataBridge(
        comm_manager=daemon_comm,
        recording_disk_manager=CaptureRDM(),
        emitter=emitter,
    )
    handled_messages: list[MessageEnvelope] = []

    original_handle_message = daemon.handle_message

    def _capture_handle_message(message: MessageEnvelope) -> None:
        handled_messages.append(message)
        original_handle_message(message)

    monkeypatch.setattr(daemon, "handle_message", _capture_handle_message)

    sender = zmq_context.socket(zmq.PUSH)
    sender.connect(str(const_module.SOCKET_PATH))

    with caplog.at_level(logging.ERROR):
        sender.send(b"{not-json")
        raw = daemon_comm._consumer_socket.recv()
        daemon.process_raw_message(raw)
        assert "Failed to parse incoming message bytes" in caplog.text

        sender.send(b'{"producer_id": "prod"}')
        raw = daemon_comm._consumer_socket.recv()
        daemon.process_raw_message(raw)

        sender.send(b'{"producer_id": "prod", "command": 123}')
        raw = daemon_comm._consumer_socket.recv()
        daemon.process_raw_message(raw)

    assert handled_messages == []

    sender.send(
        MessageEnvelope(
            producer_id="prod",
            command=CommandType.HEARTBEAT,
            payload={CommandType.HEARTBEAT.value: {}},
        ).to_bytes()
    )
    raw = daemon_comm._consumer_socket.recv()
    daemon.process_raw_message(raw)
    assert len(handled_messages) == 1
    assert daemon.channels.get("prod").is_open()

    sender.close(0)
    daemon_comm.cleanup_daemon()
