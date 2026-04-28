"""Tests for Daemon -> RecordingDiskManager integration."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from datetime import timedelta

import pytest
from neuracore_types import DataType

from neuracore.data_daemon.communications_management.data_bridge import (
    ChannelState,
    Daemon,
)
from neuracore.data_daemon.communications_management.ring_buffer import RingBuffer
from neuracore.data_daemon.const import (
    HEARTBEAT_TIMEOUT_SECS,
    SHARED_RING_RECORD_HEADER_FORMAT,
    SHARED_RING_RECORD_MAGIC,
)
from neuracore.data_daemon.models import CommandType, CompleteMessage, MessageEnvelope


@dataclass
class MockRDM:
    """Mock RecordingDiskManager to capture enqueued messages."""

    enqueued: list[CompleteMessage]

    def __init__(self) -> None:
        self.enqueued = []

    def enqueue(self, message: CompleteMessage) -> None:
        self.enqueued.append(message)


class MockComm:
    """Mock CommunicationsManager for testing."""

    def __init__(self) -> None:
        self.started_consumer = False
        self.started_publisher = False
        self.cleaned = False

    def start_consumer(self) -> None:
        self.started_consumer = True

    def start_publisher(self) -> None:
        self.started_publisher = True

    def cleanup_daemon(self) -> None:
        self.cleaned = True

    def receive_message(self) -> MessageEnvelope:
        raise KeyboardInterrupt()


def _create_daemon(
    emitter,
    *,
    comm_manager: MockComm | None = None,
    recording_disk_manager: object | None = None,
) -> tuple[Daemon, MockComm, object]:
    """Create a daemon with default test doubles for common test setup."""
    resolved_comm_manager = comm_manager or MockComm()
    resolved_recording_disk_manager = recording_disk_manager or MockRDM()
    daemon = Daemon(
        comm_manager=resolved_comm_manager,
        recording_disk_manager=resolved_recording_disk_manager,
        emitter=emitter,
    )
    return daemon, resolved_comm_manager, resolved_recording_disk_manager


class TestDaemonInit:
    """Tests for Daemon constructor dependency injection."""

    def test_daemon_accepts_required_params(self, emitter) -> None:
        """Daemon should accept recording_disk_manager and comm_manager."""
        daemon, _, mock_rdm = _create_daemon(emitter)

        assert daemon.recording_disk_manager is mock_rdm

    def test_daemon_requires_recording_disk_manager(self, emitter) -> None:
        """Daemon should require recording_disk_manager."""
        mock_comm = MockComm()

        with pytest.raises(TypeError):
            Daemon(comm_manager=mock_comm, emitter=emitter)


class TestOnCompleteMessage:
    """Tests for _on_complete_message() method."""

    def test_on_complete_message_enqueues_to_rdm(self, emitter) -> None:
        """_on_complete_message should construct CompleteMessage and enqueue to RDM."""
        daemon, _, mock_rdm = _create_daemon(emitter)

        channel = ChannelState(producer_id="test-producer")

        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-456",
            data_type=DataType.JOINT_POSITIONS,
            data=b"test-data",
            recording_id="rec-123",
            final_chunk=False,
        )

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.producer_id == "test-producer"
        assert msg.trace_id == "trace-456"
        assert msg.recording_id == "rec-123"
        assert msg.data_type == DataType.JOINT_POSITIONS
        assert msg.final_chunk is False

    def test_on_complete_message_with_final_chunk(self, emitter) -> None:
        """_on_complete_message should set final_chunk=True when specified."""
        daemon, _, mock_rdm = _create_daemon(emitter)

        channel = ChannelState(producer_id="test-producer")

        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-456",
            data_type=DataType.RGB_IMAGES,
            data=b"",
            recording_id="rec-123",
            final_chunk=True,
        )

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.final_chunk is True

    def test_on_complete_message_uses_trace_metadata(self, emitter) -> None:
        """_on_complete_message should use metadata from _trace_metadata."""
        daemon, _, mock_rdm = _create_daemon(emitter)

        # Register trace metadata
        daemon._trace_metadata["trace-456"] = {
            "dataset_id": "ds-001",
            "dataset_name": "test-dataset",
            "robot_name": "test-robot",
            "robot_id": "robot-001",
            "data_type_name": "custom_data",
            "robot_instance": 1,
        }

        channel = ChannelState(producer_id="test-producer")

        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-456",
            data_type=DataType.CUSTOM_1D,
            data=b"data",
            recording_id="rec-123",
            final_chunk=False,
        )

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.dataset_id == "ds-001"
        assert msg.dataset_name == "test-dataset"
        assert msg.robot_name == "test-robot"
        assert msg.robot_id == "robot-001"
        assert msg.data_type_name == "custom_data"
        assert msg.robot_instance == 1

    def test_on_complete_message_handles_missing_metadata(self, emitter) -> None:
        """_on_complete_message should handle missing metadata gracefully."""
        daemon, _, mock_rdm = _create_daemon(emitter)

        channel = ChannelState(producer_id="test-producer")

        # No metadata registered for this trace
        daemon._on_complete_message(
            channel=channel,
            trace_id="unknown-trace",
            data_type=DataType.CUSTOM_1D,
            data=b"data",
            recording_id="rec-123",
            final_chunk=False,
        )

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.dataset_id is None
        assert msg.dataset_name is None
        assert msg.robot_name is None
        assert msg.robot_id is None
        assert msg.data_type_name == ""
        assert msg.robot_instance == 0


class TestHandleEndTrace:
    """Tests for _handle_end_trace() method."""

    def test_handle_end_trace_sends_final_chunk_message(self, emitter) -> None:
        """_handle_end_trace should send final_chunk=True message to RDM."""
        daemon, _, mock_rdm = _create_daemon(emitter)

        channel = ChannelState(producer_id="test-producer")
        daemon.channels["test-producer"] = channel

        daemon._trace_metadata["trace-456"] = {
            "data_type": DataType.JOINT_POSITIONS.value,
            "dataset_id": "ds-001",
        }
        daemon._trace_recordings["trace-456"] = "rec-123"
        daemon._recording_traces["rec-123"] = {"trace-456"}

        message = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.TRACE_END,
            payload={
                "trace_end": {
                    "trace_id": "trace-456",
                    "recording_id": "rec-123",
                }
            },
        )

        daemon._handle_end_trace(channel, message)

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.trace_id == "trace-456"
        assert msg.final_chunk is True
        assert msg.data_type == DataType.JOINT_POSITIONS
        assert msg.data == b""

    def test_handle_end_trace_raises_for_unknown_data_type(self, emitter) -> None:
        """_handle_end_trace should raise ValueError for unknown data_type."""
        daemon, _, _ = _create_daemon(emitter)

        channel = ChannelState(producer_id="test-producer")
        daemon.channels["test-producer"] = channel

        # Register trace metadata with invalid data_type
        daemon._trace_metadata["trace-456"] = {
            "data_type": "INVALID_TYPE",
        }
        daemon._trace_recordings["trace-456"] = "rec-123"
        daemon._recording_traces["rec-123"] = {"trace-456"}

        message = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.TRACE_END,
            payload={
                "trace_end": {
                    "trace_id": "trace-456",
                    "recording_id": "rec-123",
                }
            },
        )

        with pytest.raises(ValueError, match="Unknown data_type"):
            daemon._handle_end_trace(channel, message)

    def test_handle_end_trace_raises_for_missing_metadata(self, emitter) -> None:
        """_handle_end_trace should raise ValueError if no metadata exists."""
        daemon, _, _ = _create_daemon(emitter)

        channel = ChannelState(producer_id="test-producer")
        daemon.channels["test-producer"] = channel

        # No metadata registered
        daemon._trace_recordings["trace-456"] = "rec-123"
        daemon._recording_traces["rec-123"] = {"trace-456"}

        message = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.TRACE_END,
            payload={
                "trace_end": {
                    "trace_id": "trace-456",
                    "recording_id": "rec-123",
                }
            },
        )

        with pytest.raises(ValueError, match="Missing data_type"):
            daemon._handle_end_trace(channel, message)

    def test_handle_end_trace_removes_trace_after_sending(self, emitter) -> None:
        """_handle_end_trace should remove trace from internal state after sending."""
        daemon, _, _ = _create_daemon(emitter)

        channel = ChannelState(producer_id="test-producer")
        daemon.channels["test-producer"] = channel

        daemon._trace_metadata["trace-456"] = {"data_type": DataType.CUSTOM_1D.value}
        daemon._trace_recordings["trace-456"] = "rec-123"
        daemon._recording_traces["rec-123"] = {"trace-456"}

        message = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.TRACE_END,
            payload={
                "trace_end": {
                    "trace_id": "trace-456",
                    "recording_id": "rec-123",
                }
            },
        )

        daemon._handle_end_trace(channel, message)

        assert "trace-456" not in daemon._trace_recordings
        assert "trace-456" not in daemon._trace_metadata

    def test_handle_end_trace_skips_if_missing_trace_id(self, emitter) -> None:
        """_handle_end_trace should skip if trace_id is missing."""
        daemon, _, mock_rdm = _create_daemon(emitter)

        channel = ChannelState(producer_id="test-producer")

        message = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.TRACE_END,
            payload={
                "trace_end": {
                    "recording_id": "rec-123",
                    # Missing trace_id
                }
            },
        )

        daemon._handle_end_trace(channel, message)

        assert len(mock_rdm.enqueued) == 0

    def test_handle_end_trace_skips_if_missing_recording_id(self, emitter) -> None:
        """_handle_end_trace skips if recording_id not in _trace_recordings."""
        daemon, _, mock_rdm = _create_daemon(emitter)

        channel = ChannelState(producer_id="test-producer")

        message = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.TRACE_END,
            payload={
                "trace_end": {
                    "trace_id": "trace-456",
                    # Missing recording_id and trace not in _trace_recordings
                }
            },
        )

        daemon._handle_end_trace(channel, message)

        assert len(mock_rdm.enqueued) == 0


class _FakeSharedFrame:
    def __init__(self, payload: bytes) -> None:
        self.data = memoryview(payload)

    def dispose(self) -> None:
        self.data.release()


class _FakeSharedReader:
    def __init__(self, packets: list[bytes]) -> None:
        self._packets = list(packets)

    def read_frame(self, timeout: float = 0.0) -> _FakeSharedFrame | None:
        del timeout
        if not self._packets:
            return None
        return _FakeSharedFrame(self._packets.pop(0))

    def close(self) -> None:
        return None


def _make_shared_ring_reader(*packets: bytes) -> RingBuffer:
    size = max((len(packet) for packet in packets), default=1)
    return RingBuffer(
        size=size,
        _shared_name="test-shared",
        _shared_reader=_FakeSharedReader(list(packets)),
    )


def _build_chunk_packet(
    trace_id: str,
    data_type: DataType,
    chunk_index: int,
    total_chunks: int,
    data: bytes,
) -> bytes:
    """Build one shared-ring packet in the expected format."""
    metadata = {
        "trace_id": trace_id,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    if chunk_index == 0:
        metadata.update(
            {
                "recording_id": "rec-123",
                "data_type": data_type.value,
                "data_type_name": data_type.value,
                "robot_instance": 0,
            }
        )
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return (
        struct.pack(
            SHARED_RING_RECORD_HEADER_FORMAT,
            SHARED_RING_RECORD_MAGIC,
            len(metadata_bytes),
            len(data),
        )
        + metadata_bytes
        + data
    )


class TestDrainChannelMessages:
    """Tests for _drain_channel_messages() method."""

    def test_drain_channel_messages_passes_data_type_to_on_complete(
        self, emitter
    ) -> None:
        """_drain_channel_messages should pass
        data_type from reader to _on_complete_message."""
        daemon, _, mock_rdm = _create_daemon(emitter)

        channel = ChannelState(producer_id="test-producer")
        daemon.channels["test-producer"] = channel

        # Register trace with recording_id (required for drain to work)
        daemon._trace_recordings["trace-789"] = "rec-123"
        daemon._recording_traces["rec-123"] = {"trace-789"}

        open_msg = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.OPEN_RING_BUFFER,
            payload={
                "open_ring_buffer": {
                    "size": 4096,
                    "shared_memory_name": "test-drain-type",
                }
            },
        )
        daemon._handle_open_ring_buffer(channel, open_msg)

        channel.set_ring_buffer(
            _make_shared_ring_reader(
                _build_chunk_packet(
                    trace_id="trace-789",
                    data_type=DataType.DEPTH_IMAGES,
                    chunk_index=0,
                    total_chunks=1,
                    data=b"image-data",
                )
            )
        )

        daemon._drain_channel_messages()

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.trace_id == "trace-789"
        assert msg.data_type == DataType.DEPTH_IMAGES

    def test_drain_channel_messages_handles_multi_chunk_message(self, emitter) -> None:
        """_drain_channel_messages should reassemble multi-chunk messages correctly."""
        daemon, _, mock_rdm = _create_daemon(emitter)

        channel = ChannelState(producer_id="test-producer")
        daemon.channels["test-producer"] = channel

        # Register trace with recording_id (required for drain to work)
        daemon._trace_recordings["trace-multi"] = "rec-123"
        daemon._recording_traces["rec-123"] = {"trace-multi"}

        open_msg = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.OPEN_RING_BUFFER,
            payload={
                "open_ring_buffer": {
                    "size": 4096,
                    "shared_memory_name": "test-drain-multi",
                }
            },
        )
        daemon._handle_open_ring_buffer(channel, open_msg)

        channel.set_ring_buffer(
            _make_shared_ring_reader(
                _build_chunk_packet(
                    trace_id="trace-multi",
                    data_type=DataType.JOINT_VELOCITIES,
                    chunk_index=0,
                    total_chunks=2,
                    data=b"part1",
                ),
                _build_chunk_packet(
                    trace_id="trace-multi",
                    data_type=DataType.JOINT_VELOCITIES,
                    chunk_index=1,
                    total_chunks=2,
                    data=b"part2",
                ),
            )
        )

        daemon._drain_channel_messages()
        assert len(mock_rdm.enqueued) == 0

        daemon._drain_channel_messages()

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.trace_id == "trace-multi"
        assert msg.data_type == DataType.JOINT_VELOCITIES

    def test_drain_channel_messages_registers_trace_from_shared_metadata(
        self, emitter
    ) -> None:
        """_drain_channel_messages should register traces from shared metadata."""
        daemon, _, mock_rdm = _create_daemon(emitter)

        channel = ChannelState(producer_id="test-producer")
        daemon.channels["test-producer"] = channel

        # Don't register trace up front; shared metadata should register it.

        open_msg = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.OPEN_RING_BUFFER,
            payload={
                "open_ring_buffer": {
                    "size": 4096,
                    "shared_memory_name": "test-drain-drop",
                }
            },
        )
        daemon._handle_open_ring_buffer(channel, open_msg)

        channel.set_ring_buffer(
            _make_shared_ring_reader(
                _build_chunk_packet(
                    trace_id="unregistered-trace",
                    data_type=DataType.DEPTH_IMAGES,
                    chunk_index=0,
                    total_chunks=1,
                    data=b"image-data",
                )
            )
        )

        daemon._drain_channel_messages()

        assert len(mock_rdm.enqueued) == 1
        assert daemon._trace_recordings["unregistered-trace"] == "rec-123"


class TestDataTypeHandling:
    """Tests for various DataType scenarios."""

    @pytest.mark.parametrize(
        "data_type",
        [
            DataType.RGB_IMAGES,
            DataType.DEPTH_IMAGES,
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.END_EFFECTOR_POSES,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.CUSTOM_1D,
        ],
    )
    def test_on_complete_message_handles_all_data_types(
        self, data_type: DataType, emitter
    ) -> None:
        """_on_complete_message should handle all DataType values."""
        daemon, _, mock_rdm = _create_daemon(emitter)

        channel = ChannelState(producer_id="test-producer")

        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-456",
            data_type=data_type,
            data=b"test-data",
            recording_id="rec-123",
            final_chunk=False,
        )

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.data_type == data_type


class TestExpiredChannelCleanup:
    """Tests for _cleanup_expired_channels() method."""

    def test_cleanup_expired_channels_sends_final_chunk(self, emitter) -> None:
        """Tests that _cleanup_expired_channels() sends a final_chunk message to
        RDM and removes the channel from daemon.channels.

        This test creates a channel with an active trace and
        sets its heartbeat to be expired.
        It then runs _cleanup_expired_channels() and verifies that a final_chunk message
        is sent to RDM and the channel is removed from daemon.channels.
        """
        daemon, _, mock_rdm = _create_daemon(emitter)

        channel = ChannelState(
            producer_id="expired-producer",
            trace_id="trace-expired",
        )
        channel.last_heartbeat = channel.last_heartbeat - timedelta(
            seconds=HEARTBEAT_TIMEOUT_SECS + 10
        )
        daemon.channels["expired-producer"] = channel

        daemon._trace_metadata["trace-expired"] = {
            "data_type": DataType.JOINT_POSITIONS.value,
        }
        daemon._trace_recordings["trace-expired"] = "rec-expired"
        daemon._recording_traces["rec-expired"] = {"trace-expired"}

        daemon._cleanup_expired_channels()

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.trace_id == "trace-expired"
        assert msg.final_chunk is True
        assert msg.data_type == DataType.JOINT_POSITIONS

        assert "expired-producer" not in daemon.channels

    def test_cleanup_expired_channels_no_trace_no_message(self, emitter) -> None:
        """_cleanup_expired_channels should not send message if channel has no trace."""
        daemon, _, mock_rdm = _create_daemon(emitter)

        channel = ChannelState(
            producer_id="expired-producer",
            trace_id=None,
        )
        channel.last_heartbeat = channel.last_heartbeat - timedelta(
            seconds=HEARTBEAT_TIMEOUT_SECS + 10
        )
        daemon.channels["expired-producer"] = channel

        daemon._cleanup_expired_channels()

        assert len(mock_rdm.enqueued) == 0

        assert "expired-producer" not in daemon.channels

    def test_cleanup_expired_channels_skips_active_channels(self, emitter) -> None:
        """Test _cleanup_expired_channels doesn't remove channels with recent heartbeat.

        This test creates a channel with a recent heartbeat and
        then runs _cleanup_expired_channels.
        It verifies that no message is sent to the RDM and
        that the channel is not removed.
        """
        daemon, _, mock_rdm = _create_daemon(emitter)

        channel = ChannelState(
            producer_id="active-producer",
            trace_id="trace-active",
        )
        daemon.channels["active-producer"] = channel

        # Register trace with recording
        daemon._trace_recordings["trace-active"] = "rec-active"
        daemon._recording_traces["rec-active"] = {"trace-active"}

        daemon._cleanup_expired_channels()

        assert len(mock_rdm.enqueued) == 0

        assert "active-producer" in daemon.channels


class TestRDMEnqueueErrorHandling:
    """Tests for error handling when RDM.enqueue() fails."""

    def test_on_complete_message_handles_enqueue_exception(self, emitter) -> None:
        """_on_complete_message should catch and log exceptions from RDM.enqueue()."""

        class FailingRDM:
            def enqueue(self, message: CompleteMessage) -> None:
                raise RuntimeError("Simulated RDM failure")

        failing_rdm = FailingRDM()

        daemon, _, _ = _create_daemon(
            emitter,
            recording_disk_manager=failing_rdm,
        )

        channel = ChannelState(producer_id="test-producer")

        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-456",
            data_type=DataType.CUSTOM_1D,
            data=b"test-data",
            recording_id="rec-123",
            final_chunk=False,
        )

    def test_daemon_continues_after_enqueue_failure(self, emitter) -> None:
        """Daemon should continue processing after RDM.enqueue() failure."""

        class FailOnceThenSucceedRDM:
            def __init__(self) -> None:
                self.call_count = 0
                self.enqueued: list[CompleteMessage] = []

            def enqueue(self, message: CompleteMessage) -> None:
                self.call_count += 1
                if self.call_count == 1:
                    raise RuntimeError("First call fails")
                self.enqueued.append(message)

        rdm = FailOnceThenSucceedRDM()

        daemon, _, _ = _create_daemon(
            emitter,
            recording_disk_manager=rdm,
        )

        channel = ChannelState(producer_id="test-producer")

        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-1",
            data_type=DataType.CUSTOM_1D,
            data=b"data-1",
            recording_id="rec-123",
            final_chunk=False,
        )

        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-2",
            data_type=DataType.CUSTOM_1D,
            data=b"data-2",
            recording_id="rec-123",
            final_chunk=False,
        )

        assert len(rdm.enqueued) == 1
        assert rdm.enqueued[0].trace_id == "trace-2"
