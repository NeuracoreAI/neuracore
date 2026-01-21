"""Tests for Daemon → RecordingDiskManager integration.

These tests verify the fixes made to connect the Daemon to RDM:
1. Daemon.__init__() accepts config_manager and instantiates RDM
2. _on_complete_message() constructs CompleteMessage and enqueues to RDM
3. _handle_end_trace() sends final_chunk=True message to RDM
4. data_type is properly passed through the chain
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from neuracore_types import DataType

from neuracore.data_daemon.communications_management.data_bridge import (
    ChannelState,
    Daemon,
)
from neuracore.data_daemon.communications_management.ring_buffer import RingBuffer
from neuracore.data_daemon.const import (
    CHUNK_HEADER_FORMAT,
    DATA_TYPE_FIELD_SIZE,
    TRACE_ID_FIELD_SIZE,
)
from neuracore.data_daemon.models import CommandType, CompleteMessage, MessageEnvelope
from tests.unit.data_daemon.helpers import MockConfigManager


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


# -----------------------------------------------------------------------------
# Test 1: Daemon.__init__() with dependency injection
# -----------------------------------------------------------------------------


class TestDaemonInit:
    """Tests for Daemon constructor dependency injection."""

    def test_daemon_accepts_config_manager(self, tmp_path: Any) -> None:
        """Daemon should accept config_manager parameter."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        assert daemon._config_manager is mock_config
        assert daemon.recording_disk_manager is mock_rdm

    def test_daemon_creates_default_config_manager_if_not_provided(self) -> None:
        """Daemon should create default ConfigManager if not provided."""
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            recording_disk_manager=mock_rdm,
        )

        assert daemon._config_manager is not None

    def test_daemon_creates_rdm_with_config_if_not_provided(
        self, tmp_path: Any
    ) -> None:
        """Daemon should create RDM using config_manager if RDM not provided."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()

        with patch(
            "neuracore.data_daemon.communications_management.data_bridge.RecordingDiskManager"
        ) as mock_rdm_class:
            mock_rdm_instance = MagicMock()
            mock_rdm_class.return_value = mock_rdm_instance

            daemon = Daemon(
                comm_manager=mock_comm,
                config_manager=mock_config,
            )

            mock_rdm_class.assert_called_once_with(mock_config)
            assert daemon.recording_disk_manager is mock_rdm_instance


# -----------------------------------------------------------------------------
# Test 2: _on_complete_message() constructs and enqueues CompleteMessage
# -----------------------------------------------------------------------------


class TestOnCompleteMessage:
    """Tests for _on_complete_message() method."""

    def test_on_complete_message_enqueues_to_rdm(self, tmp_path: Any) -> None:
        """_on_complete_message should construct CompleteMessage and enqueue to RDM."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")

        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-456",
            data_type=DataType.JOINT_POSITIONS,
            data=b"test-data",
            final_chunk=False,
        )

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.producer_id == "test-producer"
        assert msg.trace_id == "trace-456"
        assert msg.recording_id == "rec-123"
        assert msg.data_type == DataType.JOINT_POSITIONS
        assert msg.final_chunk is False

    def test_on_complete_message_with_final_chunk(self, tmp_path: Any) -> None:
        """_on_complete_message should set final_chunk=True when specified."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")

        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-456",
            data_type=DataType.RGB_IMAGES,
            data=b"",
            final_chunk=True,
        )

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.final_chunk is True

    def test_on_complete_message_uses_trace_metadata(self, tmp_path: Any) -> None:
        """_on_complete_message should use metadata from _trace_metadata."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        # Register trace metadata
        daemon._trace_metadata["trace-456"] = {
            "dataset_id": "ds-001",
            "dataset_name": "test-dataset",
            "robot_name": "test-robot",
            "robot_id": "robot-001",
        }

        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")

        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-456",
            data_type=DataType.CUSTOM_1D,
            data=b"data",
            final_chunk=False,
        )

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.dataset_id == "ds-001"
        assert msg.dataset_name == "test-dataset"
        assert msg.robot_name == "test-robot"
        assert msg.robot_id == "robot-001"

    def test_on_complete_message_handles_missing_metadata(self, tmp_path: Any) -> None:
        """_on_complete_message should handle missing metadata gracefully."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")

        # No metadata registered for this trace
        daemon._on_complete_message(
            channel=channel,
            trace_id="unknown-trace",
            data_type=DataType.CUSTOM_1D,
            data=b"data",
            final_chunk=False,
        )

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.dataset_id is None
        assert msg.dataset_name is None
        assert msg.robot_name is None
        assert msg.robot_id is None

    def test_on_complete_message_handles_empty_recording_id(
        self, tmp_path: Any
    ) -> None:
        """_on_complete_message should use empty string if recording_id is None."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id=None)

        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-456",
            data_type=DataType.CUSTOM_1D,
            data=b"data",
            final_chunk=False,
        )

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.recording_id == ""


# -----------------------------------------------------------------------------
# Test 3: _handle_end_trace() sends final_chunk=True message
# -----------------------------------------------------------------------------


class TestHandleEndTrace:
    """Tests for _handle_end_trace() method."""

    def test_handle_end_trace_sends_final_chunk_message(self, tmp_path: Any) -> None:
        """_handle_end_trace should send final_chunk=True message to RDM."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
        daemon.channels["test-producer"] = channel

        # Register trace metadata with data_type
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
        assert msg.data == ""  # Empty data encoded as base64

    def test_handle_end_trace_uses_custom_1d_for_unknown_data_type(
        self, tmp_path: Any
    ) -> None:
        """_handle_end_trace should default to CUSTOM_1D for unknown data_type."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
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

        daemon._handle_end_trace(channel, message)

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.data_type == DataType.CUSTOM_1D

    def test_handle_end_trace_uses_custom_1d_for_missing_metadata(
        self, tmp_path: Any
    ) -> None:
        """_handle_end_trace should default to CUSTOM_1D if no metadata exists."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
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

        daemon._handle_end_trace(channel, message)

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.data_type == DataType.CUSTOM_1D

    def test_handle_end_trace_removes_trace_after_sending(self, tmp_path: Any) -> None:
        """_handle_end_trace should remove trace from internal state after sending."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
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

        # Trace should be removed from internal state
        assert "trace-456" not in daemon._trace_recordings
        assert "trace-456" not in daemon._trace_metadata

    def test_handle_end_trace_skips_if_missing_trace_id(self, tmp_path: Any) -> None:
        """_handle_end_trace should skip if trace_id is missing."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")

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

    def test_handle_end_trace_skips_if_missing_recording_id(
        self, tmp_path: Any
    ) -> None:
        """_handle_end_trace should skip if recording_id is missing."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id=None)

        message = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.TRACE_END,
            payload={
                "trace_end": {
                    "trace_id": "trace-456",
                    # Missing recording_id and channel.recording_id is None
                }
            },
        )

        daemon._handle_end_trace(channel, message)

        assert len(mock_rdm.enqueued) == 0


# -----------------------------------------------------------------------------
# Test 4: _drain_channel_messages passes data_type correctly
# -----------------------------------------------------------------------------


def _write_chunk_to_ring_buffer(
    ring: RingBuffer,
    trace_id: str,
    data_type: DataType,
    chunk_index: int,
    total_chunks: int,
    data: bytes,
) -> None:
    """Helper to write a chunk to a ring buffer in the expected format."""
    trace_id_bytes = trace_id.encode("utf-8")
    trace_id_field = trace_id_bytes[:TRACE_ID_FIELD_SIZE].ljust(
        TRACE_ID_FIELD_SIZE, b"\x00"
    )
    data_type_bytes = data_type.value.encode("utf-8")
    data_type_field = data_type_bytes[:DATA_TYPE_FIELD_SIZE].ljust(
        DATA_TYPE_FIELD_SIZE, b"\x00"
    )
    header = struct.pack(
        CHUNK_HEADER_FORMAT,
        trace_id_field,
        data_type_field,
        chunk_index,
        total_chunks,
        len(data),
    )
    ring.write(header + data)


class TestDrainChannelMessages:
    """Tests for _drain_channel_messages() method."""

    def test_drain_channel_messages_passes_data_type_to_on_complete(
        self, tmp_path: Any
    ) -> None:
        """_drain_channel_messages passes data_type to _on_complete_message."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        # Create channel with ring buffer
        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
        daemon.channels["test-producer"] = channel

        # Handle OPEN_RING_BUFFER to initialize ring buffer
        open_msg = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.OPEN_RING_BUFFER,
            payload={"open_ring_buffer": {"size": 4096}},
        )
        daemon._handle_open_ring_buffer(channel, open_msg)

        # Write a complete message (single chunk) to ring buffer
        _write_chunk_to_ring_buffer(
            ring=channel.ring_buffer,
            trace_id="trace-789",
            data_type=DataType.DEPTH_IMAGES,
            chunk_index=0,
            total_chunks=1,
            data=b"image-data",
        )

        # Drain messages
        daemon._drain_channel_messages()

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.trace_id == "trace-789"
        assert msg.data_type == DataType.DEPTH_IMAGES

    def test_drain_channel_messages_handles_multi_chunk_message(
        self, tmp_path: Any
    ) -> None:
        """_drain_channel_messages should reassemble multi-chunk messages correctly."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")
        daemon.channels["test-producer"] = channel

        open_msg = MessageEnvelope(
            producer_id="test-producer",
            command=CommandType.OPEN_RING_BUFFER,
            payload={"open_ring_buffer": {"size": 4096}},
        )
        daemon._handle_open_ring_buffer(channel, open_msg)

        # Write first chunk
        _write_chunk_to_ring_buffer(
            ring=channel.ring_buffer,
            trace_id="trace-multi",
            data_type=DataType.JOINT_VELOCITIES,
            chunk_index=0,
            total_chunks=2,
            data=b"part1",
        )

        # First drain - should not produce a message yet (partial)
        daemon._drain_channel_messages()
        assert len(mock_rdm.enqueued) == 0

        # Write second chunk
        _write_chunk_to_ring_buffer(
            ring=channel.ring_buffer,
            trace_id="trace-multi",
            data_type=DataType.JOINT_VELOCITIES,
            chunk_index=1,
            total_chunks=2,
            data=b"part2",
        )

        # Second drain - now should produce the complete message
        daemon._drain_channel_messages()

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.trace_id == "trace-multi"
        assert msg.data_type == DataType.JOINT_VELOCITIES


# -----------------------------------------------------------------------------
# Test 5: Different DataType scenarios
# -----------------------------------------------------------------------------


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
        self, tmp_path: Any, data_type: DataType
    ) -> None:
        """_on_complete_message should handle all DataType values."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")

        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-456",
            data_type=data_type,
            data=b"test-data",
            final_chunk=False,
        )

        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.data_type == data_type


# -----------------------------------------------------------------------------
# Test 6: Expired channel cleanup
# -----------------------------------------------------------------------------


class TestExpiredChannelCleanup:
    """Tests for _cleanup_expired_channels() method."""

    def test_cleanup_expired_channels_sends_final_chunk(self, tmp_path: Any) -> None:
        """
        _cleanup_expired_channels sends final_chunk for expired channels.

        This covers the case where a channel has active traces.
        """
        from datetime import timedelta

        from neuracore.data_daemon.const import HEARTBEAT_TIMEOUT_SECS

        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        # Create a channel with an active trace
        channel = ChannelState(
            producer_id="expired-producer",
            recording_id="rec-expired",
            trace_id="trace-expired",
        )
        # Set heartbeat to be expired
        channel.last_heartbeat = channel.last_heartbeat - timedelta(
            seconds=HEARTBEAT_TIMEOUT_SECS + 10
        )
        daemon.channels["expired-producer"] = channel

        # Register trace metadata
        daemon._trace_metadata["trace-expired"] = {
            "data_type": DataType.JOINT_POSITIONS.value,
        }
        daemon._trace_recordings["trace-expired"] = "rec-expired"
        daemon._recording_traces["rec-expired"] = {"trace-expired"}

        # Run cleanup
        daemon._cleanup_expired_channels()

        # Should have sent a final_chunk message
        assert len(mock_rdm.enqueued) == 1
        msg = mock_rdm.enqueued[0]
        assert msg.trace_id == "trace-expired"
        assert msg.final_chunk is True
        assert msg.data_type == DataType.JOINT_POSITIONS

        # Channel should be removed
        assert "expired-producer" not in daemon.channels

    def test_cleanup_expired_channels_no_trace_no_message(self, tmp_path: Any) -> None:
        """_cleanup_expired_channels should not send message if channel has no trace."""
        from datetime import timedelta

        from neuracore.data_daemon.const import HEARTBEAT_TIMEOUT_SECS

        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        # Create a channel without an active trace
        channel = ChannelState(
            producer_id="expired-producer",
            recording_id=None,
            trace_id=None,
        )
        channel.last_heartbeat = channel.last_heartbeat - timedelta(
            seconds=HEARTBEAT_TIMEOUT_SECS + 10
        )
        daemon.channels["expired-producer"] = channel

        # Run cleanup
        daemon._cleanup_expired_channels()

        # No message should be sent
        assert len(mock_rdm.enqueued) == 0

        # Channel should still be removed
        assert "expired-producer" not in daemon.channels

    def test_cleanup_expired_channels_skips_active_channels(
        self, tmp_path: Any
    ) -> None:
        """
        _cleanup_expired_channels should not remove channels with recent heartbeat.
        """
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()
        mock_rdm = MockRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=mock_rdm,
        )

        # Create a channel with recent heartbeat (default is now)
        channel = ChannelState(
            producer_id="active-producer",
            recording_id="rec-active",
            trace_id="trace-active",
        )
        daemon.channels["active-producer"] = channel

        # Run cleanup
        daemon._cleanup_expired_channels()

        # No message should be sent
        assert len(mock_rdm.enqueued) == 0

        # Channel should NOT be removed
        assert "active-producer" in daemon.channels


# -----------------------------------------------------------------------------
# Test 7: Error handling in RDM.enqueue()
# -----------------------------------------------------------------------------


class TestRDMEnqueueErrorHandling:
    """Tests for error handling when RDM.enqueue() fails."""

    def test_on_complete_message_handles_enqueue_exception(self, tmp_path: Any) -> None:
        """_on_complete_message should catch and log exceptions from RDM.enqueue()."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()

        # Create a mock RDM that raises an exception
        class FailingRDM:
            def enqueue(self, message: CompleteMessage) -> None:
                raise RuntimeError("Simulated RDM failure")

        failing_rdm = FailingRDM()

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=failing_rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")

        # This should NOT raise - exception should be caught and logged
        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-456",
            data_type=DataType.CUSTOM_1D,
            data=b"test-data",
            final_chunk=False,
        )

        # If we get here without exception, the test passes

    def test_daemon_continues_after_enqueue_failure(self, tmp_path: Any) -> None:
        """Daemon should continue processing after RDM.enqueue() failure."""
        mock_config = MockConfigManager(path_to_store_record=str(tmp_path))
        mock_comm = MockComm()

        # Create a mock RDM that fails on first call, succeeds on second
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

        daemon = Daemon(
            comm_manager=mock_comm,
            config_manager=mock_config,
            recording_disk_manager=rdm,
        )

        channel = ChannelState(producer_id="test-producer", recording_id="rec-123")

        # First call - should fail silently
        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-1",
            data_type=DataType.CUSTOM_1D,
            data=b"data-1",
            final_chunk=False,
        )

        # Second call - should succeed
        daemon._on_complete_message(
            channel=channel,
            trace_id="trace-2",
            data_type=DataType.CUSTOM_1D,
            data=b"data-2",
            final_chunk=False,
        )

        # Only second message should be enqueued
        assert len(rdm.enqueued) == 1
        assert rdm.enqueued[0].trace_id == "trace-2"
