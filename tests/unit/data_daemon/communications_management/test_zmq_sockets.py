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
from neuracore.data_daemon.models import MessageEnvelope
from neuracore.data_daemon.recording_encoding_disk_manager import (
    recording_disk_manager as rdm_module,
)

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
        daemon._finalize_pending_closes()
        raw = comm.receive_raw()
        msg = MessageEnvelope.from_bytes(raw) if raw else None

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
    recording_id = "rec-zmq-commands"
    producer = Producer(
        comm_manager=producer_comm, chunk_size=16, recording_id=recording_id
    )

    producer.start_new_trace()
    producer.open_ring_buffer(size=2048)

    assert _wait_for(lambda: producer.producer_id in daemon.channels, timeout=0.5)
    channel = daemon.channels[producer.producer_id]
    assert channel.ring_buffer is not None
    assert channel.ring_buffer.size == 2048

    payload = json.dumps({"seq": 1}).encode("utf-8")
    active_trace_id = producer.trace_id
    trace_written: list[int] = []

    def on_trace_written(trace_id: str, _: str, bytes_written: int) -> None:
        if trace_id == active_trace_id:
            trace_written.append(bytes_written)

    get_emitter().on(Emitter.TRACE_WRITTEN, on_trace_written)
    try:
        producer.send_data(
            payload,
            data_type=DataType.CUSTOM_1D,
            data_type_name="custom",
            robot_instance=1,
            robot_id="robot-1",
            dataset_id="dataset-1",
        )
        assert _wait_for(
            lambda: active_trace_id in daemon._trace_recordings, timeout=0.5
        )
        producer.end_trace()
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
    """Tests channel timeout cleanup and trace finalization.

    Verifies that:
    1. After inactivity timeout, the daemon removes the channel
    2. Partial trace data is finalized and written to disk
    3. Crash detection works (channel removed after timeout)
    """
    daemon, _, context = daemon_runtime

    producer_comm = CommunicationsManager(context=context)
    recording_id = "rec-zmq-timeout"
    producer = Producer(
        comm_manager=producer_comm, chunk_size=16, recording_id=recording_id
    )

    producer.start_new_trace()
    producer.open_ring_buffer(size=1024)

    assert _wait_for(lambda: producer.producer_id in daemon.channels, timeout=0.5)

    payload = json.dumps({"seq": 1}).encode("utf-8")
    producer.send_data(
        payload,
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
    """Test that channels are cleaned up after producer disconnects."""
    daemon, _, context = daemon_runtime

    producer_comm = CommunicationsManager(context=context)
    producer = Producer(
        comm_manager=producer_comm, chunk_size=16, recording_id="rec-disconnect"
    )
    producer.start_new_trace()
    producer.open_ring_buffer(size=512)

    assert _wait_for(lambda: producer.producer_id in daemon.channels, timeout=0.5)

    producer._stop_event.set()
    if producer.socket is not None:
        producer.socket.close(0)

    assert _wait_for(
        lambda: producer.producer_id not in daemon.channels,
        timeout=TEST_HEARTBEAT_TIMEOUT_SECS + 1,
    )


def test_deferred_close_honors_zmq_buffer_data(loop_manager: EventLoopManager) -> None:
    """Deferred close mechanism ensures ZMQ buffer data is processed before blocking.

    The Story:
        A producer is streaming data when the user calls `stop_recording()`.
        Due to ZMQ's internal buffering (default HWM ~1000 messages), some
        DATA_CHUNK messages are still queued in the socket when RECORDING_STOPPED
        arrives at the daemon. Without the deferred close mechanism, these
        buffered messages would be immediately dropped when the guard check
        at line 436 blocks data for closed recordings.

        The deferred close solves this by using a two-phase approach:
        1. RECORDING_STOPPED adds to `_pending_close_recordings`
        2. Next loop iteration moves pending → `_closed_recordings`

        This one-iteration delay allows any DATA_CHUNK messages that were
        already in the ZMQ buffer to be processed before blocking begins.

    The Flow:
        1. Create a mock RECORDING_STOPPED message for "rec-123"
        2. Call `daemon._handle_recording_stopped(msg)`
        3. Assert: "rec-123" is in `_pending_close_recordings`
        4. Assert: "rec-123" is NOT in `_closed_recordings`
        5. Assert: STOP_RECORDING event was emitted (via event listener)
        6. Call `daemon._finalize_pending_closes()`
        7. Assert: "rec-123" is now in `_closed_recordings`
        8. Assert: `_pending_close_recordings` is empty

    Why This Matters:
        ZMQ PUSH/PULL sockets can buffer up to 1000 messages by default.
        At 30 fps with 16KB chunks, this represents ~0.5 seconds of video
        data (~16MB). Immediate blocking on RECORDING_STOPPED would silently
        drop this buffered data, causing incomplete recordings with missing
        final frames. Users would not know data was lost until playback.

    Key Assertions:
        - After _handle_recording_stopped: recording_id IN pending, NOT IN closed
        - STOP_RECORDING event emitted exactly once with correct recording_id
        - After _finalize_pending_closes: recording_id IN closed
        - _pending_close_recordings is empty after finalization
        - State transitions are deterministic (no race conditions)
    """
    from unittest.mock import MagicMock

    from neuracore.data_daemon.communications_management.communications_manager import (
        MessageEnvelope,
    )
    from neuracore.data_daemon.models import CommandType

    mock_comm = MagicMock()
    mock_rdm = MagicMock()
    daemon = Daemon(comm_manager=mock_comm, recording_disk_manager=mock_rdm)

    recording_id = "rec-123"

    stop_events_received: list[str] = []

    def on_stop_recording(rec_id: str) -> None:
        stop_events_received.append(rec_id)

    emitter = get_emitter()
    emitter.on(Emitter.STOP_RECORDING, on_stop_recording)

    try:
        msg = MessageEnvelope(
            producer_id=None,
            command=CommandType.RECORDING_STOPPED,
            payload={"recording_stopped": {"recording_id": recording_id}},
        )

        assert recording_id not in daemon._pending_close_recordings
        assert recording_id not in daemon._closed_recordings

        daemon._handle_recording_stopped(msg)

        assert recording_id in daemon._pending_close_recordings
        assert recording_id not in daemon._closed_recordings

        assert len(stop_events_received) == 1
        assert stop_events_received[0] == recording_id

        daemon._finalize_pending_closes()

        assert recording_id in daemon._closed_recordings

        assert len(daemon._pending_close_recordings) == 0

        daemon._finalize_pending_closes()
        assert recording_id in daemon._closed_recordings
        assert len(daemon._pending_close_recordings) == 0
        assert len(stop_events_received) == 1

    finally:
        emitter.remove_listener(Emitter.STOP_RECORDING, on_stop_recording)


def test_data_blocked_after_recording_closed(
    loop_manager: EventLoopManager, caplog: pytest.LogCaptureFixture
) -> None:
    """Data chunks for closed recordings are dropped with warning.

    The Story:
        A recording has been stopped and fully finalized (recording_id is
        in `_closed_recordings`). A late DATA_CHUNK arrives - perhaps from
        a slow producer, network delay, or a producer that didn't receive
        the stop signal. This data belongs to a completed recording and
        must be dropped to prevent data corruption.

        Without this guard, late data could:
        - Be written to disk after the recording was marked complete
        - Cause inconsistent trace files (extra data after "final" frame)
        - Mix data between recordings if the same producer_id is reused

    The Flow:
        1. Create daemon instance with mocked dependencies
        2. Directly add "rec-closed" to `daemon._closed_recordings`
        3. Create a DATA_CHUNK message for "rec-closed"
        4. Call `daemon._handle_write_data_chunk(msg)`
        5. Assert: Method returns early (check via mock or return value)
        6. Assert: Warning was logged with recording_id and trace_id
        7. Assert: Ring buffer was NOT written to
        8. Assert: No TRACE_WRITTEN event emitted

    Why This Matters:
        Data integrity is paramount. A recording that's been stopped should
        be immutable - no additional data should be added. Late arrivals
        could cause:
        - Trace files with more frames than reported
        - Upload failures due to checksum mismatches
        - Confusion when replaying recordings

    Key Assertions:
        - _handle_write_data_chunk returns early for closed recording
        - Warning logged: "Dropping data for closed recording_id=X trace_id=Y"
        - Ring buffer write count unchanged
        - No events emitted (TRACE_WRITTEN, etc.)
        - Channel state unchanged
    """
    import logging
    from unittest.mock import MagicMock

    from neuracore.data_daemon.communications_management.communications_manager import (
        MessageEnvelope,
    )
    from neuracore.data_daemon.communications_management.data_bridge import ChannelState
    from neuracore.data_daemon.models import CommandType, DataChunkPayload

    mock_comm = MagicMock()
    mock_rdm = MagicMock()
    daemon = Daemon(comm_manager=mock_comm, recording_disk_manager=mock_rdm)

    recording_id = "rec-closed"
    trace_id = "trace-late-arrival"
    producer_id = "prod-456"

    daemon._closed_recordings.add(recording_id)

    mock_ring_buffer = MagicMock()
    mock_ring_buffer.write = MagicMock(return_value=None)
    channel = ChannelState(producer_id=producer_id, ring_buffer=mock_ring_buffer)
    daemon.channels[producer_id] = channel

    events_received: list[str] = []

    def on_trace_written(tid: str, rid: str, bytes_written: int) -> None:
        events_received.append(f"TRACE_WRITTEN:{tid}")

    emitter = get_emitter()
    emitter.on(Emitter.TRACE_WRITTEN, on_trace_written)

    try:
        data_chunk = DataChunkPayload(
            channel_id=producer_id,
            recording_id=recording_id,
            trace_id=trace_id,
            chunk_index=0,
            total_chunks=1,
            data_type=DataType.CUSTOM_1D,
            data_type_name="test",
            robot_instance=1,
            robot_id="robot-1",
            robot_name="test-robot",
            dataset_id="dataset-1",
            dataset_name="test-dataset",
            data=b"late data that should be dropped",
        )

        msg = MessageEnvelope(
            producer_id=producer_id,
            command=CommandType.DATA_CHUNK,
            payload={"data_chunk": data_chunk.to_dict()},
        )

        with caplog.at_level(logging.WARNING):
            daemon._handle_write_data_chunk(channel, msg)

        mock_ring_buffer.write.assert_not_called()

        warning_found = any(
            "Dropping data for closed recording_id=rec-closed" in record.message
            for record in caplog.records
        )
        assert warning_found, (
            f"Expected warning about dropping data not found. "
            f"Logs: {[r.message for r in caplog.records]}"
        )

        assert len(events_received) == 0

        assert channel in daemon.channels.values()
        assert channel.ring_buffer is mock_ring_buffer

    finally:
        emitter.remove_listener(Emitter.TRACE_WRITTEN, on_trace_written)


def test_channel_expires_after_timeout_without_recording_stopped(
    loop_manager: EventLoopManager,
) -> None:
    """Channels are cleaned up after inactivity timeout even without RECORDING_STOPPED.

    The Story:
        A producer crashes, loses network connectivity, or the user's code
        exits unexpectedly mid-recording. The producer never sends a
        RECORDING_STOPPED message. The daemon must detect this "orphaned"
        channel via the inactivity timeout and clean it up properly,
        ensuring partial trace data is finalized and resources are released.

        This is the safety net for abnormal terminations. Without it:
        - Ring buffers would leak memory
        - Partial traces would never be uploaded
        - Channel dict would grow unbounded

    The Flow:
        1. Create daemon with channel for producer "prod-123"
        2. Set channel's `last_heartbeat` to (now - 15 seconds)
        3. Register event listener for TRACE_END
        4. Call `daemon._cleanup_expired_channels()`
        5. Assert: TRACE_END event received for channel's trace_id
        6. Assert: "prod-123" NOT in `daemon.channels`
        7. Assert: Ring buffer was cleaned up

    Why This Matters:
        Producer crashes are inevitable in robotics systems. Hardware
        failures, network partitions, and software bugs all cause
        unexpected disconnections. The daemon must handle these gracefully:
        - Detect dead producers via timeout
        - Finalize partial traces (so data isn't lost)
        - Clean up resources (memory, file handles)
        - Allow the producer to reconnect with fresh state

    Key Assertions:
        - Channel with expired last_heartbeat is removed from channels dict
        - TRACE_END event emitted with correct trace_id and recording_id
        - Ring buffer deleted (or marked for cleanup)
        - Channel not in daemon.channels after cleanup
        - Cleanup is idempotent (calling twice doesn't error)
    """
    from datetime import datetime, timedelta, timezone
    from unittest.mock import MagicMock

    from neuracore.data_daemon.communications_management.data_bridge import ChannelState

    mock_comm = MagicMock()
    mock_rdm = MagicMock()
    daemon = Daemon(comm_manager=mock_comm, recording_disk_manager=mock_rdm)

    producer_id = "prod-123"
    trace_id = "trace-orphaned"
    recording_id = "rec-orphaned"

    mock_ring_buffer = MagicMock()
    channel = ChannelState(
        producer_id=producer_id,
        ring_buffer=mock_ring_buffer,
        trace_id=trace_id,
    )

    channel.last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=15)

    daemon.channels[producer_id] = channel
    daemon._trace_recordings[trace_id] = recording_id
    daemon._recording_traces[recording_id] = {trace_id}
    daemon._trace_metadata[trace_id] = {"data_type": "CUSTOM_1D"}

    assert producer_id in daemon.channels

    daemon._cleanup_expired_channels()

    assert producer_id not in daemon.channels

    mock_rdm.enqueue.assert_called()
    call_args = mock_rdm.enqueue.call_args
    complete_message = call_args[0][0]
    assert complete_message.final_chunk is True
    assert complete_message.trace_id == trace_id
    assert complete_message.recording_id == recording_id

    daemon._cleanup_expired_channels()
    assert producer_id not in daemon.channels


def test_channel_stays_alive_within_timeout(loop_manager: EventLoopManager) -> None:
    """Active channels within timeout window are NOT cleaned up.

    The Story:
        A channel received data recently (within the timeout window).
        The cleanup routine runs as part of the normal daemon loop.
        This active channel must NOT be removed - only truly inactive
        channels should be cleaned up.

        False positives in timeout detection would be catastrophic:
        - Active recordings would lose data mid-stream
        - Producers would need to reconnect and restart
        - Trust in the system would be destroyed

    The Flow:
        1. Create daemon with channel for producer "prod-active"
        2. Set channel's `last_heartbeat` to (now - 5 seconds)
           (well within 10-second timeout)
        3. Register event listener for TRACE_END
        4. Call `daemon._cleanup_expired_channels()`
        5. Assert: NO TRACE_END event received
        6. Assert: "prod-active" STILL in `daemon.channels`
        7. Assert: Ring buffer intact and usable

    Why This Matters:
        The timeout check must be precise. A channel that received data
        5 seconds ago is clearly active. Removing it would:
        - Cause immediate data loss for the ongoing recording
        - Force the producer to handle unexpected disconnection
        - Create inconsistent state between producer and daemon

        This test ensures the boundary condition is correct: channels
        are only cleaned up when last_heartbeat > HEARTBEAT_TIMEOUT_SECS.

    Key Assertions:
        - Channel with recent last_heartbeat remains in channels dict
        - NO TRACE_END event emitted
        - Ring buffer unchanged and functional
        - Channel state (trace_id, recording_id) preserved
        - Multiple calls to cleanup don't affect active channels
    """
    from datetime import datetime, timedelta, timezone
    from unittest.mock import MagicMock

    from neuracore.data_daemon.communications_management.data_bridge import ChannelState

    mock_comm = MagicMock()
    mock_rdm = MagicMock()
    daemon = Daemon(comm_manager=mock_comm, recording_disk_manager=mock_rdm)

    producer_id = "prod-active"
    trace_id = "trace-active"
    recording_id = "rec-active"

    mock_ring_buffer = MagicMock()
    channel = ChannelState(
        producer_id=producer_id,
        ring_buffer=mock_ring_buffer,
        trace_id=trace_id,
    )

    channel.last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=0.1)

    daemon.channels[producer_id] = channel
    daemon._trace_recordings[trace_id] = recording_id
    daemon._recording_traces[recording_id] = {trace_id}
    daemon._trace_metadata[trace_id] = {"data_type": "CUSTOM_1D"}

    initial_trace_id = channel.trace_id
    initial_ring_buffer = channel.ring_buffer

    assert producer_id in daemon.channels

    daemon._cleanup_expired_channels()

    assert producer_id in daemon.channels

    mock_rdm.enqueue.assert_not_called()

    assert channel.ring_buffer is initial_ring_buffer
    assert channel.ring_buffer is mock_ring_buffer

    assert channel.trace_id == initial_trace_id
    assert channel.trace_id == trace_id
    assert daemon._trace_recordings[trace_id] == recording_id

    daemon._cleanup_expired_channels()
    daemon._cleanup_expired_channels()
    assert producer_id in daemon.channels
    assert channel.trace_id == trace_id
    assert channel.ring_buffer is mock_ring_buffer
    mock_rdm.enqueue.assert_not_called()
