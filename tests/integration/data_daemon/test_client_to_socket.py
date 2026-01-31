"""Tests for client-to-socket data streaming.

Tests data serialization and transmission from the neuracore client
to the ZMQ socket, using a mock socket to capture sent bytes.
The mock_socket fixture is provided by conftest.py.
"""

import json
import logging
import uuid
import warnings
from collections.abc import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest

import neuracore as nc
from neuracore.data_daemon.models import CommandType, DataChunkPayload, MessageEnvelope
from tests.integration.data_daemon.helpers import DataTypeTestCase

logger = logging.getLogger(__name__)

TEST_ROBOT = "basic_test_robot"


@pytest.fixture
def stream_to_daemon() -> Generator[None, None, None]:
    """Fixture that handles robot setup and recording lifecycle.

    Setup:
        - Login to neuracore
        - Connect to test robot
        - Create a unique dataset
        - Start recording

    Teardown:
        - Stop recording (always, even on test failure)
        - Cancel recording if stop fails
    """
    dataset_name = f"test_dataset_{uuid.uuid4().hex[:8]}"
    recording_started = False

    # Setup
    logger.info("Setting up streaming test")
    nc.login()
    nc.connect_robot(TEST_ROBOT)
    nc.create_dataset(dataset_name)

    nc.start_recording()
    recording_started = True
    logger.info(f"Recording started for dataset: {dataset_name}")

    try:
        yield
    finally:
        logger.info("Tearing down streaming test")
        if recording_started:
            try:
                nc.stop_recording(wait=True)
                logger.info("Recording stopped successfully")
            except Exception as e:
                logger.warning(f"Error stopping recording: {e}")
                try:
                    nc.cancel_recording()
                    logger.info("Recording cancelled")
                except Exception as cancel_error:
                    logger.error(f"Failed to cancel recording: {cancel_error}")


def make_test_cases() -> list[DataTypeTestCase]:
    """Create test cases for all data types."""
    unit_quat_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    return [
        DataTypeTestCase(
            name="joint_positions",
            data_type="JOINT_POSITIONS",
            log_func=lambda timestamp: nc.log_joint_positions(
                positions={"joint1": 0.5}, timestamp=timestamp
            ),
        ),
        DataTypeTestCase(
            name="joint_velocities",
            data_type="JOINT_VELOCITIES",
            log_func=lambda timestamp: nc.log_joint_velocities(
                velocities={"joint1": 0.1}, timestamp=timestamp
            ),
        ),
        DataTypeTestCase(
            name="joint_torques",
            data_type="JOINT_TORQUES",
            log_func=lambda timestamp: nc.log_joint_torques(
                torques={"joint1": 0.2}, timestamp=timestamp
            ),
        ),
        DataTypeTestCase(
            name="joint_target_positions",
            data_type="JOINT_TARGET_POSITIONS",
            log_func=lambda timestamp: nc.log_joint_target_positions(
                target_positions={"joint1": 0.5}, timestamp=timestamp
            ),
        ),
        DataTypeTestCase(
            name="pose",
            data_type="POSES",
            log_func=lambda timestamp: nc.log_pose(
                name="test_pose", pose=unit_quat_pose, timestamp=timestamp
            ),
        ),
        DataTypeTestCase(
            name="end_effector_pose",
            data_type="END_EFFECTOR_POSES",
            log_func=lambda timestamp: nc.log_end_effector_pose(
                name="ee", pose=unit_quat_pose, timestamp=timestamp
            ),
        ),
        DataTypeTestCase(
            name="parallel_gripper",
            data_type="PARALLEL_GRIPPER_OPEN_AMOUNTS",
            log_func=lambda timestamp: nc.log_parallel_gripper_open_amount(
                name="gripper", value=0.5, timestamp=timestamp
            ),
        ),
        DataTypeTestCase(
            name="parallel_gripper_target",
            data_type="PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS",
            log_func=lambda timestamp: nc.log_parallel_gripper_target_open_amount(
                name="gripper", value=0.75, timestamp=timestamp
            ),
        ),
        DataTypeTestCase(
            name="language",
            data_type="LANGUAGE",
            log_func=lambda timestamp: nc.log_language(
                name="instruction", language="pick up the cup", timestamp=timestamp
            ),
        ),
        DataTypeTestCase(
            name="rgb_image",
            data_type="RGB_IMAGES",
            log_func=lambda timestamp: nc.log_rgb(
                name="camera",
                rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=timestamp,
            ),
        ),
        DataTypeTestCase(
            name="depth_image",
            data_type="DEPTH_IMAGES",
            log_func=lambda timestamp: nc.log_depth(
                name="depth_cam",
                depth=np.zeros((480, 640), dtype=np.float32),
                timestamp=timestamp,
            ),
        ),
        DataTypeTestCase(
            name="point_cloud",
            data_type="POINT_CLOUDS",
            log_func=lambda timestamp: nc.log_point_cloud(
                name="lidar",
                points=np.zeros((100, 3), dtype=np.float16),
                timestamp=timestamp,
            ),
        ),
        DataTypeTestCase(
            name="custom_1d",
            data_type="CUSTOM_1D",
            log_func=lambda timestamp: nc.log_custom_1d(
                name="sensor", data=np.array([1.0, 2.0, 3.0]), timestamp=timestamp
            ),
        ),
    ]


DATA_TYPE_TEST_CASES = make_test_cases()


class TestClientStreaming:
    """Minimal streaming tests."""

    @pytest.mark.parametrize(
        "test_case",
        DATA_TYPE_TEST_CASES,
        ids=[test_case.name for test_case in DATA_TYPE_TEST_CASES],
    )
    def test_stream_data_type_to_socket(
        self,
        test_case: DataTypeTestCase,
        mock_socket: MagicMock,
        stream_to_daemon: None,
    ) -> None:
        """Stream data and verify correct data_type and timestamp are sent.

        Captures raw bytes from socket.send() and deserializes to verify
        the wire format is correct.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_case.log_func(test_case.timestamp)

        raw_bytes_list = [call[0][0] for call in mock_socket.send.call_args_list]
        envelopes = [MessageEnvelope.from_bytes(raw) for raw in raw_bytes_list]

        data_chunk_envelopes = [
            envelope
            for envelope in envelopes
            if envelope.command == CommandType.DATA_CHUNK
        ]
        assert len(data_chunk_envelopes) >= 1, "No DATA_CHUNK messages sent"
        payloads = [
            DataChunkPayload.from_dict(envelope.payload["data_chunk"])
            for envelope in data_chunk_envelopes
        ]

        assert all(
            payload.data_type.value == test_case.data_type for payload in payloads
        ), f"Expected data_type {test_case.data_type} on all chunks"

        target_trace = payloads[0].trace_id
        trace_group = [
            payload for payload in payloads if payload.trace_id == target_trace
        ]
        assert len(trace_group) >= 1, "No chunks found for target trace_id"

        recording_id = trace_group[0].recording_id
        assert recording_id, "recording_id must be non-empty"
        assert all(
            payload.recording_id == recording_id for payload in trace_group
        ), "recording_id inconsistent across chunks"

        total_chunks = trace_group[0].total_chunks
        assert total_chunks == len(
            trace_group
        ), "total_chunks does not match chunk count"

        indices = sorted(payload.chunk_index for payload in trace_group)
        assert indices == list(range(total_chunks)), "chunk_index sequence incorrect"

        if test_case.data_type not in ("RGB_IMAGES", "DEPTH_IMAGES"):
            decoded_samples = [json.loads(payload.data) for payload in trace_group]
            assert all(
                sample.get("timestamp") == test_case.timestamp
                for sample in decoded_samples
            ), "timestamp mismatch"
