import base64
import logging
import math
import time
from enum import Enum
from uuid import uuid4

import pytest
import zmq
from neuracore_types import DataType

import neuracore.data_daemon.const as const_module
from neuracore.data_daemon.communications_management import (
    communications_manager as comms_module,
)
from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
)
from neuracore.data_daemon.communications_management.data_bridge import Daemon
from neuracore.data_daemon.communications_management.producer import Producer
from neuracore.data_daemon.models import CommandType, DataChunkPayload, MessageEnvelope


class CaptureRDM:
    def __init__(self) -> None:
        self.enqueued = []

    def enqueue(self, message) -> None:
        self.enqueued.append(message)


@pytest.fixture(autouse=True)
def ipc_paths(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path / "ndd"
    socket_path = f"inproc://daemon-{uuid4().hex}"
    events_path = f"inproc://events-{uuid4().hex}"

    mpsa = monkeypatch.setattr

    mpsa(const_module, "BASE_DIR", base_dir)
    mpsa(const_module, "SOCKET_PATH", socket_path)
    mpsa(const_module, "RECORDING_EVENTS_SOCKET_PATH", events_path)
    mpsa(comms_module, "BASE_DIR", base_dir)
    mpsa(comms_module, "SOCKET_PATH", socket_path)
    mpsa(comms_module, "RECORDING_EVENTS_SOCKET_PATH", events_path)

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
def zmq_context() -> zmq.Context:
    context = zmq.Context.instance()
    yield context


def _drain_messages(
    daemon: Daemon,
    comm: CommunicationsManager,
    expected: int,
    timeout: float = 2.0,
) -> None:
    poller = zmq.Poller()
    poller.register(comm.consumer_socket, zmq.POLLIN)
    received = 0
    deadline = time.monotonic() + timeout
    while received < expected and time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        events = dict(poller.poll(remaining * 1000))
        if comm.consumer_socket in events:
            message = comm.receive_message()
            daemon.handle_message(message)
            daemon._drain_channel_messages()
            received += 1
    assert received == expected


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
        if second_comm.consumer_socket is not None:
            second_comm.consumer_socket.close(0)


def test_create_producer_socket_returns_none_without_daemon(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path = tmp_path / "ndd" / "management.sock"
    monkeypatch.setattr(const_module, "SOCKET_PATH", socket_path)
    monkeypatch.setattr(comms_module, "SOCKET_PATH", socket_path)

    comm = CommunicationsManager()
    assert not socket_path.exists()
    assert comm.create_producer_socket() is None
    comm.cleanup_producer()


def test_message_envelope_round_trip_bytes() -> None:
    envelope = MessageEnvelope(
        producer_id="producer-abc",
        command=CommandType.OPEN_RING_BUFFER,
        payload={"open_ring_buffer": {"size": 4096}},
    )

    parsed = MessageEnvelope.from_bytes(envelope.to_bytes())

    assert parsed == envelope


def test_pub_sub_recording_stopped_event(zmq_context: zmq.Context) -> None:
    daemon_comm = CommunicationsManager(context=zmq_context)
    daemon_comm.start_publisher()

    producer_comm = CommunicationsManager(context=zmq_context)
    subscriber = producer_comm.create_subscriber_socket()

    assert subscriber is not None
    time.sleep(0.05)

    payload = {"recording_stopped": {"recording_id": "rec-1"}}
    daemon_comm.publish_message(
        MessageEnvelope(
            producer_id=None,
            command=CommandType.RECORDING_STOPPED,
            payload=payload,
        )
    )

    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    events = dict(poller.poll(1000))
    assert subscriber in events

    raw = subscriber.recv()
    parsed = MessageEnvelope.from_bytes(raw)
    assert parsed.command == CommandType.RECORDING_STOPPED
    assert parsed.payload == payload

    subscriber.close(0)
    producer_comm.cleanup_producer()
    daemon_comm.cleanup_daemon()


def test_large_payload_chunked_round_trip_over_ipc(zmq_context: zmq.Context) -> None:
    daemon_comm = CommunicationsManager(context=zmq_context)
    daemon_comm.start_consumer()
    rdm = CaptureRDM()
    daemon = Daemon(comm_manager=daemon_comm, recording_disk_manager=rdm)

    producer_comm = CommunicationsManager(context=zmq_context)
    producer = Producer(
        id="producer-large",
        comm_manager=producer_comm,
        chunk_size=16 * 1024,
        recording_id="rec-large",
    )

    producer.open_ring_buffer(size=128 * 1024)
    producer.start_new_trace()
    _drain_messages(daemon, daemon_comm, expected=1)

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

    expected_chunks = math.ceil(len(payload) / producer.chunk_size)
    _drain_messages(daemon, daemon_comm, expected=expected_chunks)

    assert len(rdm.enqueued) == 1
    encoded = rdm.enqueued[0].data
    assert base64.b64decode(encoded) == payload

    producer.stop_producer()
    daemon_comm.cleanup_daemon()


def test_two_producers_route_to_own_channels(zmq_context: zmq.Context) -> None:
    daemon_comm = CommunicationsManager(context=zmq_context)
    daemon_comm.start_consumer()
    rdm = CaptureRDM()
    daemon = Daemon(comm_manager=daemon_comm, recording_disk_manager=rdm)

    producer_a_comm = CommunicationsManager(context=zmq_context)
    producer_b_comm = CommunicationsManager(context=zmq_context)

    producer_a = Producer(
        id="producer-a",
        comm_manager=producer_a_comm,
        chunk_size=8,
        recording_id="rec-a",
    )
    producer_b = Producer(
        id="producer-b",
        comm_manager=producer_b_comm,
        chunk_size=8,
        recording_id="rec-b",
    )

    producer_a.open_ring_buffer(size=4096)
    producer_b.open_ring_buffer(size=4096)
    producer_a.start_new_trace()
    producer_b.start_new_trace()
    _drain_messages(daemon, daemon_comm, expected=2)

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

    expected = math.ceil(len(payload_a) / producer_a.chunk_size) + math.ceil(
        len(payload_b) / producer_b.chunk_size
    )
    _drain_messages(daemon, daemon_comm, expected=expected)

    by_producer = {msg.producer_id: base64.b64decode(msg.data) for msg in rdm.enqueued}
    assert by_producer["producer-a"] == payload_a
    assert by_producer["producer-b"] == payload_b

    producer_a.stop_producer()
    producer_b.stop_producer()
    daemon_comm.cleanup_daemon()


def test_interleaved_chunks_reassemble_per_producer(zmq_context: zmq.Context) -> None:
    daemon_comm = CommunicationsManager(context=zmq_context)
    daemon_comm.start_consumer()
    rdm = CaptureRDM()
    daemon = Daemon(comm_manager=daemon_comm, recording_disk_manager=rdm)

    producer_a_comm = CommunicationsManager(context=zmq_context)
    producer_b_comm = CommunicationsManager(context=zmq_context)

    producer_a_socket = producer_a_comm.create_producer_socket()
    producer_b_socket = producer_b_comm.create_producer_socket()
    assert producer_a_socket is not None
    assert producer_b_socket is not None

    def send_open(
        comm: CommunicationsManager, socket: zmq.Socket, producer_id: str
    ) -> None:
        comm.send_message(
            socket,
            MessageEnvelope(
                producer_id=producer_id,
                command=CommandType.OPEN_RING_BUFFER,
                payload={"open_ring_buffer": {"size": 4096}},
            ),
        )

    send_open(producer_a_comm, producer_a_socket, "producer-a")
    send_open(producer_b_comm, producer_b_socket, "producer-b")
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
            producer_a_socket,
            make_chunk("producer-a", "rec-a", "trace-a", idx, 3, chunks_a[idx]),
        ))
        interleaved.append((
            producer_b_comm,
            producer_b_socket,
            make_chunk("producer-b", "rec-b", "trace-b", idx, 3, chunks_b[idx]),
        ))

    for comm, socket, envelope in interleaved:
        comm.send_message(socket, envelope)

    _drain_messages(daemon, daemon_comm, expected=len(interleaved))

    by_producer = {msg.producer_id: base64.b64decode(msg.data) for msg in rdm.enqueued}
    assert by_producer["producer-a"] == payload_a
    assert by_producer["producer-b"] == payload_b

    producer_a_socket.close(0)
    producer_b_socket.close(0)
    producer_a_comm.cleanup_producer()
    producer_b_comm.cleanup_producer()
    daemon_comm.cleanup_daemon()


def test_trace_id_required_on_send_data() -> None:
    """send_data() requires start_new_trace() to be called first."""
    producer_comm = CommunicationsManager()
    producer = Producer(comm_manager=producer_comm, recording_id="rec-1")

    with pytest.raises(ValueError, match="Trace ID required"):
        producer.send_data(
            b"data",
            data_type=DataType.CUSTOM_1D,
            data_type_name="custom",
            robot_instance=1,
            robot_id="robot",
            dataset_id="dataset",
        )

    producer.stop_producer()


def test_recording_id_required_on_start_new_trace() -> None:
    """start_new_trace() requires recording_id to be set on init."""
    producer_comm = CommunicationsManager()
    producer = Producer(comm_manager=producer_comm)

    with pytest.raises(ValueError, match="recording_id is required"):
        producer.start_new_trace()

    producer.stop_producer()


def test_unknown_command_logs_warning_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeCommand(Enum):
        UNKNOWN = "unknown_command"

    daemon = Daemon(
        comm_manager=CommunicationsManager(), recording_disk_manager=CaptureRDM()
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
            command=CommandType.OPEN_RING_BUFFER,
            payload={"open_ring_buffer": {"size": 1024}},
        )
    )
    assert daemon.channels["producer-1"].ring_buffer is not None


def test_garbage_messages_are_logged_and_daemon_survives(
    caplog: pytest.LogCaptureFixture,
    zmq_context: zmq.Context,
) -> None:
    daemon_comm = CommunicationsManager(context=zmq_context)
    daemon_comm.start_consumer()
    daemon = Daemon(comm_manager=daemon_comm, recording_disk_manager=CaptureRDM())

    sender = zmq_context.socket(zmq.PUSH)
    sender.connect(str(const_module.SOCKET_PATH))

    with caplog.at_level(logging.ERROR):
        sender.send(b"{not-json")
        raw = daemon_comm.consumer_socket.recv()
        assert daemon.process_raw_message(raw) is False
        assert "Failed to parse incoming message bytes" in caplog.text

        sender.send(b'{"producer_id": "prod"}')
        raw = daemon_comm.consumer_socket.recv()
        assert daemon.process_raw_message(raw) is False

        sender.send(b'{"producer_id": "prod", "command": 123}')
        raw = daemon_comm.consumer_socket.recv()
        assert daemon.process_raw_message(raw) is False

    sender.send(
        MessageEnvelope(
            producer_id="prod",
            command=CommandType.OPEN_RING_BUFFER,
            payload={"open_ring_buffer": {"size": 1024}},
        ).to_bytes()
    )
    raw = daemon_comm.consumer_socket.recv()
    assert daemon.process_raw_message(raw) is True
    assert daemon.channels["prod"].ring_buffer is not None

    sender.close(0)
    daemon_comm.cleanup_daemon()
