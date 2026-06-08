"""Unit tests for the robot-scoped producer coordinator."""

from __future__ import annotations

import time
from uuid import uuid4

import pytest
from neuracore_types import DataType

from neuracore.data_daemon.communications_management.producer.robot_producer_coordinator import (  # noqa: E501
    RobotProducerCoordinator,
    StreamPayload,
    _iter_chunk_views,
    _normalise_parts,
)
from neuracore.data_daemon.communications_management.shared_transport.iox2_daemon_drain import (  # noqa: E501
    Iox2DaemonDrain,
)
from neuracore.data_daemon.models import CommandType, DataChunkPayload


def _stub_socket(monkeypatch) -> list:
    """Patch the coordinator's ZMQ socket to capture sent envelopes."""
    messages: list = []
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


def _wait_for(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _register(coordinator, *, stream_name: str, data_type: DataType):
    return coordinator.register_stream_session(
        stream_name=stream_name,
        data_type=data_type,
        recording_id="rec-1",
        robot_instance=2,
        robot_id="robot-1",
        robot_name="robot",
        dataset_id="dataset-1",
        dataset_name="dataset",
    )


def test_iter_chunk_views_splits_across_parts() -> None:
    parts = _normalise_parts((b"ab", memoryview(b"cdef"), b"gh"))
    chunks = [bytes(chunk) for chunk in _iter_chunk_views(parts, 3)]
    assert chunks == [b"abc", b"def", b"gh"]


def test_non_video_stream_sends_single_data_chunk(monkeypatch) -> None:
    messages = _stub_socket(monkeypatch)
    coordinator = RobotProducerCoordinator(producer_id=f"robot-{uuid4().hex[:8]}")
    try:
        session = _register(
            coordinator, stream_name="custom", data_type=DataType.CUSTOM_1D
        )
        coordinator.enqueue_stream_payload(
            StreamPayload(session=session, parts=(memoryview(b"abcd"),), total_bytes=4)
        )
        assert _wait_for(
            lambda: any(m.command == CommandType.DATA_CHUNK for m in messages)
        )
    finally:
        coordinator.close()

    chunks = [m for m in messages if m.command == CommandType.DATA_CHUNK]
    assert len(chunks) == 1
    payload = DataChunkPayload.from_dict(chunks[0].payload["data_chunk"])
    assert payload.channel_id == coordinator.producer_id
    assert payload.recording_id == "rec-1"
    assert payload.trace_id == session.trace_id
    assert payload.chunk_index == 0
    assert payload.total_chunks == 1
    assert payload.data == b"abcd"
    assert payload.data_type == DataType.CUSTOM_1D


def test_batched_joint_enqueues_without_per_stream_channel(monkeypatch) -> None:
    from neuracore.data_daemon.models import (
        BatchedJointDataItemPayload,
        BatchedJointDataPayload,
    )

    messages = _stub_socket(monkeypatch)
    coordinator = RobotProducerCoordinator(producer_id=f"robot-{uuid4().hex[:8]}")
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
                trace_id="trace-1", data_type_name="joint1", value=0.25
            )
        ],
    )
    try:
        # No joint stream session is registered; batched joints go straight to
        # the coordinator's ordered data lane.
        coordinator.enqueue_batched_joint(payload)
        assert _wait_for(
            lambda: any(m.command == CommandType.BATCHED_JOINT_DATA for m in messages)
        )
    finally:
        coordinator.close()

    batches = [m for m in messages if m.command == CommandType.BATCHED_JOINT_DATA]
    assert len(batches) == 1
    assert batches[0].payload[CommandType.BATCHED_JOINT_DATA.value] == payload.to_dict()


def test_stop_recording_freezes_intake_and_sends_trace_end(monkeypatch) -> None:
    messages = _stub_socket(monkeypatch)
    coordinator = RobotProducerCoordinator(producer_id=f"robot-{uuid4().hex[:8]}")
    try:
        session = _register(
            coordinator, stream_name="custom", data_type=DataType.CUSTOM_1D
        )
        coordinator.enqueue_stream_payload(
            StreamPayload(session=session, parts=(memoryview(b"abcd"),), total_bytes=4)
        )
        assert _wait_for(
            lambda: any(m.command == CommandType.DATA_CHUNK for m in messages)
        )

        cutoff = coordinator.stop_recording(wait_for_drain=True)
        assert cutoff >= 1

        trace_ends = [m for m in messages if m.command == CommandType.TRACE_END]
        assert len(trace_ends) == 1
        assert trace_ends[0].payload["trace_end"]["recording_id"] == "rec-1"

        # New payloads are dropped while the coordinator is frozen.
        data_chunks_before = sum(
            1 for m in messages if m.command == CommandType.DATA_CHUNK
        )
        coordinator.enqueue_stream_payload(
            StreamPayload(session=session, parts=(memoryview(b"efgh"),), total_bytes=4)
        )
        time.sleep(0.1)
        data_chunks_after = sum(
            1 for m in messages if m.command == CommandType.DATA_CHUNK
        )
        assert data_chunks_after == data_chunks_before
        assert coordinator._heartbeat_service.stop_event.is_set()

        coordinator.register_stream_session(
            stream_name="custom",
            data_type=DataType.CUSTOM_1D,
            recording_id="rec-2",
            robot_instance=2,
            robot_id="robot-1",
            robot_name="robot",
            dataset_id="dataset-1",
            dataset_name="dataset",
        )
        assert not coordinator._heartbeat_service.stop_event.is_set()
    finally:
        coordinator.close()


@pytest.mark.parametrize(
    "data_type",
    [DataType.RGB_IMAGES, DataType.DEPTH_IMAGES, DataType.POINT_CLOUDS],
)
def test_video_stream_routes_over_iox2(monkeypatch, data_type) -> None:
    messages = _stub_socket(monkeypatch)
    coordinator = RobotProducerCoordinator(producer_id=f"robot-{uuid4().hex[:8]}")
    coordinator._video_chunk_size = 2
    drain = Iox2DaemonDrain()
    received: list[tuple[str, dict, bytes]] = []
    try:
        session = _register(coordinator, stream_name="camera_0", data_type=data_type)
        drain.register_channel(session.video_service_id)
        coordinator.enqueue_stream_payload(
            StreamPayload(session=session, parts=(memoryview(b"abcd"),), total_bytes=4)
        )
        _wait_for(
            lambda: drain.drain_all(
                lambda producer_id, seq, meta, chunk: received.append(
                    (producer_id, meta, chunk)
                )
            )
            or len(received) >= 2
        )
    finally:
        drain.close()
        coordinator.close()

    assert [chunk for _, _, chunk in received] == [b"ab", b"cd"]
    assert all(producer_id == coordinator.producer_id for producer_id, _, _ in received)
    # First frame carries trace metadata, subsequent frames do not.
    assert received[0][1]["data_type"] == data_type.value
    assert "data_type" not in received[1][1]
    # Video frames never travel over the ZMQ control lane.
    assert all(m.command != CommandType.DATA_CHUNK for m in messages)
