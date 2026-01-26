from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest
import zmq
from neuracore_types import DataType

import neuracore.data_daemon.const as const_module
import neuracore.data_daemon.event_emitter as em_module
from neuracore.data_daemon.communications_management import (
    communications_manager as comms_module,
)
from neuracore.data_daemon.communications_management import (
    data_bridge as data_bridge_module,
)
from neuracore.data_daemon.communications_management import producer as producer_module
from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
)
from neuracore.data_daemon.communications_management.data_bridge import Daemon
from neuracore.data_daemon.communications_management.producer import (
    Producer,
    RecordingContext,
)
from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.event_loop_manager import EventLoopManager
from neuracore.data_daemon.recording_encoding_disk_manager import (
    recording_disk_manager as rdm_module,
)

TEST_HEARTBEAT_INTERVAL_SECS = 0.1
TEST_HEARTBEAT_TIMEOUT_SECS = 0.3

TEST_HEARTBEAT_INTERVAL_SECS = 0.1
TEST_HEARTBEAT_TIMEOUT_SECS = 0.3


def _wait_for(predicate, timeout: float, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _run_daemon_loop(
    daemon: Daemon,
    comm: CommunicationsManager,
    stop_event: threading.Event,
    ready_event: threading.Event,
    error_bucket: list[BaseException],
) -> None:
    try:
        comm.start_consumer()
        comm.start_publisher()
    except BaseException as exc:
        error_bucket.append(exc)
        ready_event.set()
        return
    ready_event.set()
    if comm.consumer_socket is None:
        raise RuntimeError("consumer socket not initialized")
    comm.consumer_socket.setsockopt(zmq.RCVTIMEO, 200)
    comm.consumer_socket.setsockopt(zmq.LINGER, 0)

    while not stop_event.is_set():
        msg = None
        try:
            msg = comm.receive_message()
        except zmq.Again:
            msg = None

        if msg is not None:
            daemon.handle_message(msg)

        daemon._cleanup_expired_channels()
        daemon._drain_channel_messages()

    comm.cleanup_daemon()


@pytest.fixture(autouse=True)
def ipc_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path / "ndd"
    socket_path = f"inproc://daemon-{uuid4().hex}"
    events_path = f"inproc://events-{uuid4().hex}"

    mpsa = monkeypatch.setattr

    mpsa(const_module, "BASE_DIR", base_dir)
    mpsa(const_module, "SOCKET_PATH", socket_path)
    mpsa(const_module, "RECORDING_EVENTS_SOCKET_PATH", events_path)
    mpsa(const_module, "HEARTBEAT_TIMEOUT_SECS", TEST_HEARTBEAT_TIMEOUT_SECS)
    mpsa(comms_module, "BASE_DIR", base_dir)
    mpsa(comms_module, "SOCKET_PATH", socket_path)
    mpsa(comms_module, "RECORDING_EVENTS_SOCKET_PATH", events_path)
    mpsa(data_bridge_module, "HEARTBEAT_TIMEOUT_SECS", TEST_HEARTBEAT_TIMEOUT_SECS)

    original_init = producer_module.Producer.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._heartbeat_interval = TEST_HEARTBEAT_INTERVAL_SECS

    mpsa(producer_module.Producer, "__init__", patched_init)

    yield


@pytest.fixture
def loop_manager():
    """EventLoopManager instance for tests."""
    em_module._emitter = None

    manager = EventLoopManager()
    manager.start()
    yield manager

    if manager.is_running():
        try:
            manager.stop()
        except RuntimeError:
            pass

    em_module._emitter = None


@pytest.fixture
def daemon_runtime(tmp_path: Path, loop_manager: EventLoopManager):
    recordings_root = tmp_path / "recordings"

    rdm = rdm_module.RecordingDiskManager(
        loop_manager=loop_manager,
        flush_bytes=1,
        recordings_root=str(recordings_root),
    )
    context = zmq.Context()
    comm = CommunicationsManager(context=context)
    daemon = Daemon(
        comm_manager=comm,
        recording_disk_manager=rdm,
    )

    stop_event = threading.Event()
    ready_event = threading.Event()
    error_bucket: list[BaseException] = []
    thread = threading.Thread(
        target=_run_daemon_loop,
        args=(daemon, comm, stop_event, ready_event, error_bucket),
        daemon=True,
    )
    thread.start()

    assert _wait_for(ready_event.is_set, timeout=0.5)
    if error_bucket:
        raise error_bucket[0]

    try:
        yield daemon, rdm, context
    finally:
        stop_event.set()
        thread.join(timeout=0.5)
        try:
            future = asyncio.run_coroutine_threadsafe(
                rdm.shutdown(), loop_manager.general_loop
            )
            future.result(timeout=1.0)
        except Exception:
            pass
        context.destroy(linger=0)


def test_zmq_socket_establishment_and_teardown() -> None:
    """Test ZMQ socket establishment and teardown.

    Verifies that creating and closing ZMQ sockets (producer and subscriber)
    works as expected. Also verifies that the CommunicationsManager's cleanup
    function properly closes the sockets and terminates the ZMQ context.
    """
    context = zmq.Context()
    comm = CommunicationsManager(context=context)
    comm.start_consumer()
    comm.start_publisher()

    producer_socket = comm.create_producer_socket()
    assert producer_socket is not None
    producer_socket.close(0)
    subscriber_socket = comm.create_subscriber_socket()
    assert subscriber_socket is not None
    subscriber_socket.close(0)

    comm.cleanup_daemon()

    assert comm.consumer_socket is None
    assert comm.publisher_socket is None
    context.term()


def test_zmq_commands_and_message_flow(daemon_runtime) -> None:
    """Test ZMQ command flow between producers and the daemon.

    Verifies that creating a trace, opening a ring buffer, sending data chunks,
    and stopping a trace works as expected. Also verifies that the
    CommunicationsManager's cleanup function properly closes the sockets and
    terminates the ZMQ context.
    """
    daemon, _, context = daemon_runtime

    producer_comm = CommunicationsManager(context=context)
    producer = Producer(comm_manager=producer_comm, chunk_size=16)
    recording_id = "rec-zmq-commands"

    producer.start_new_trace(recording_id=recording_id)
    producer.open_ring_buffer(size=2048)

    assert _wait_for(lambda: producer.producer_id in daemon.channels, timeout=0.5)
    channel = daemon.channels[producer.producer_id]
    assert channel.ring_buffer is not None
    assert channel.ring_buffer.size == 2048

    last_heartbeat = channel.last_heartbeat
    producer.heartbeat()
    assert _wait_for(
        lambda: daemon.channels[producer.producer_id].last_heartbeat > last_heartbeat,
        timeout=0.5,
    )

    payload = json.dumps({"seq": 1}).encode("utf-8")
    active_trace_id = producer.trace_id
    producer.send_data(
        payload,
        trace_id=producer.trace_id,
        recording_id=recording_id,
        data_type=DataType.CUSTOM_1D,
        data_type_name="custom",
        robot_instance=1,
        robot_id="robot-1",
        dataset_id="dataset-1",
    )
    producer.end_trace()

    trace_written: list[int] = []

    def on_trace_written(trace_id: str, _: str, bytes_written: int) -> None:
        if trace_id == active_trace_id:
            trace_written.append(bytes_written)

    get_emitter().on(Emitter.TRACE_WRITTEN, on_trace_written)
    try:
        assert _wait_for(lambda: trace_written, timeout=1)
    finally:
        get_emitter().remove_listener(Emitter.TRACE_WRITTEN, on_trace_written)

    recording_comm = CommunicationsManager(context=context)
    recording_context = RecordingContext(
        recording_id=recording_id, comm_manager=recording_comm
    )
    recording_context.stop_recording()
    if recording_context.socket is not None:
        recording_context.socket.close(0)
    recording_comm.cleanup_producer()

    assert _wait_for(lambda: recording_id in daemon._closed_recordings, timeout=0.5)

    producer.stop_producer()
    if producer.socket is not None:
        producer.socket.close(0)


def test_heartbeat_timeout_cleanup_and_partial_trace_finalization_and_crash_detection(
    daemon_runtime, tmp_path: Path
) -> None:
    """Tests the following:

    1. Heartbeat timeout cleanup: Producer does not send data chunks
        after the heartbeat timeout.
    2. Partial trace finalization: Daemon sends a final chunk after heartbeat
        timeout and the trace is written to disk.
    3. Crash detection: Daemon removes the channel after a heartbeat timeout
        and the trace is finalized.

    This test case starts a producer and a daemon, and then waits for the
    heartbeat timeout. It then verifies that the trace is written to disk
    and that the channel is removed from the daemon.
    """
    daemon, _, context = daemon_runtime

    producer_comm = CommunicationsManager(context=context)
    producer = Producer(comm_manager=producer_comm, chunk_size=16)
    recording_id = "rec-zmq-timeout"

    producer.start_new_trace(recording_id=recording_id)
    producer.open_ring_buffer(size=1024)
    producer.start_producer()

    assert _wait_for(lambda: producer.producer_id in daemon.channels, timeout=0.5)

    channel = daemon.channels[producer.producer_id]
    first_heartbeat = channel.last_heartbeat
    assert _wait_for(
        lambda: daemon.channels[producer.producer_id].last_heartbeat > first_heartbeat,
        timeout=0.5,
    )
    second_heartbeat = daemon.channels[producer.producer_id].last_heartbeat
    assert _wait_for(
        lambda: daemon.channels[producer.producer_id].last_heartbeat > second_heartbeat,
        timeout=0.5,
    )
    third_heartbeat = daemon.channels[producer.producer_id].last_heartbeat
    interval = (third_heartbeat - second_heartbeat).total_seconds()
    assert (
        TEST_HEARTBEAT_INTERVAL_SECS * 0.5
        <= interval
        <= TEST_HEARTBEAT_INTERVAL_SECS * 3
    )

    payload = json.dumps({"seq": 1}).encode("utf-8")
    producer.send_data(
        payload,
        trace_id=producer.trace_id,
        recording_id=recording_id,
        data_type=DataType.CUSTOM_1D,
        data_type_name="custom",
        robot_instance=1,
        robot_id="robot-1",
        dataset_id="dataset-1",
    )

    trace_written: list[int] = []
    active_trace_id = producer.trace_id

    def on_trace_written(trace_id: str, _: str, bytes_written: int) -> None:
        if trace_id == active_trace_id:
            trace_written.append(bytes_written)

    get_emitter().on(Emitter.TRACE_WRITTEN, on_trace_written)
    try:
        producer._stop_event.set()
        if hasattr(producer, "_heartbeat_thread"):
            producer._heartbeat_thread.join(timeout=1)
        if producer.socket is not None:
            producer.socket.close(0)

        start = time.monotonic()
        assert _wait_for(
            lambda: producer.producer_id not in daemon.channels,
            timeout=TEST_HEARTBEAT_TIMEOUT_SECS + 1,
        )
        elapsed = time.monotonic() - start
        assert elapsed <= TEST_HEARTBEAT_TIMEOUT_SECS + 1
        assert _wait_for(lambda: trace_written, timeout=1)
    finally:
        get_emitter().remove_listener(Emitter.TRACE_WRITTEN, on_trace_written)

    trace_dir = (
        tmp_path
        / "recordings"
        / recording_id
        / DataType.CUSTOM_1D.value
        / str(active_trace_id)
    )
    trace_file = trace_dir / "trace.json"
    assert trace_file.exists()


def test_socket_cleanup_on_disconnect(daemon_runtime) -> None:
    daemon, _, context = daemon_runtime

    producer_comm = CommunicationsManager(context=context)
    producer = Producer(comm_manager=producer_comm, chunk_size=16)
    producer.start_new_trace(recording_id="rec-disconnect")
    producer.open_ring_buffer(size=512)
    producer.start_producer()

    assert _wait_for(lambda: producer.producer_id in daemon.channels, timeout=0.5)

    producer._stop_event.set()
    if hasattr(producer, "_heartbeat_thread"):
        producer._heartbeat_thread.join(timeout=1)
    if producer.socket is not None:
        producer.socket.close(0)

    assert _wait_for(
        lambda: producer.producer_id not in daemon.channels,
        timeout=TEST_HEARTBEAT_TIMEOUT_SECS + 1,
    )
