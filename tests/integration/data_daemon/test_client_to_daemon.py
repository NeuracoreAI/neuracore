"""Tests for client SDK to daemon communication (E2E).

Tests the full data flow from the neuracore client API through real ZMQ sockets
to the daemon:
- nc.log_*() API calls
- Real ZMQ socket communication
- Daemon receives and processes messages correctly
- All 13 DataTypes flow correctly

Uses the stream_to_daemon_with_capture fixture which provides:
- Real backend connection (nc.login, nc.connect_robot, nc.start_recording)
- Real daemon with CaptureRDM for payload verification
"""

from __future__ import annotations

import base64
import json
import logging
import warnings

import numpy as np
import pytest

import neuracore as nc
from tests.integration.data_daemon.conftest import DaemonRDMCapture, _wait_for
from tests.integration.data_daemon.helpers import DataTypeTestCase

logger = logging.getLogger(__name__)


def make_test_cases() -> list[DataTypeTestCase]:
    """Create test cases for all 13 data types."""
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
            name="parallel_gripper_open_amount",
            data_type="PARALLEL_GRIPPER_OPEN_AMOUNTS",
            log_func=lambda timestamp: nc.log_parallel_gripper_open_amount(
                name="gripper", value=0.5, timestamp=timestamp
            ),
        ),
        DataTypeTestCase(
            name="parallel_gripper_target_open_amount",
            data_type="PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS",
            log_func=lambda timestamp: nc.log_parallel_gripper_target_open_amount(
                name="gripper", value=0.75, timestamp=timestamp
            ),
            marks=(
                pytest.mark.xfail(
                    reason="Daemon 32-byte limit truncates data type name"
                ),
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
            name="custom_1d",
            data_type="CUSTOM_1D",
            log_func=lambda timestamp: nc.log_custom_1d(
                name="sensor", data=np.array([1.0, 2.0, 3.0]), timestamp=timestamp
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
    ]


DATA_TYPE_TEST_CASES = make_test_cases()


class TestClientToDaemon:
    """E2E tests for client SDK to daemon communication."""

    @pytest.mark.parametrize(
        "test_case",
        [pytest.param(tc, marks=tc.marks, id=tc.name) for tc in DATA_TYPE_TEST_CASES],
    )
    def test_log_data_type_to_daemon(
        self,
        test_case: DataTypeTestCase,
        stream_to_daemon_with_capture: DaemonRDMCapture,
    ) -> None:
        """Data logged via nc.log_*() arrives correctly at daemon."""
        stream_to_daemon_with_capture.capture.enqueued.clear()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_case.log_func(test_case.timestamp)

        assert _wait_for(
            lambda: len(stream_to_daemon_with_capture.capture.enqueued) > 0,
            timeout=5,
        ), f"No messages captured for {test_case.name}"

        captured_message = stream_to_daemon_with_capture.capture.enqueued[0]

        assert captured_message.data_type.value == test_case.data_type, (
            f"Expected data_type {test_case.data_type}, "
            f"got {captured_message.data_type.value}"
        )

        if test_case.data_type not in ("RGB_IMAGES", "DEPTH_IMAGES"):
            payload = base64.b64decode(captured_message.data)
            decoded = json.loads(payload.decode("utf-8"))
            assert decoded.get("timestamp") == test_case.timestamp, (
                f"Expected timestamp {test_case.timestamp}, "
                f"got {decoded.get('timestamp')}"
            )

    def test_multiple_logs_single_trace(
        self,
        stream_to_daemon_with_capture: DaemonRDMCapture,
    ) -> None:
        """Multiple logs from same recording should have same trace_id."""
        stream_to_daemon_with_capture.capture.enqueued.clear()

        for i in range(3):
            nc.log_joint_positions(
                positions={"joint1": float(i) * 0.1},
                timestamp=1234567890.0 + i,
            )

        assert _wait_for(
            lambda: len(stream_to_daemon_with_capture.capture.enqueued) >= 3,
            timeout=5,
        ), "Expected 3 messages"

        trace_ids = {
            msg.trace_id for msg in stream_to_daemon_with_capture.capture.enqueued
        }
        assert (
            len(trace_ids) == 1
        ), f"Expected single trace_id, got {len(trace_ids)}: {trace_ids}"

    def test_recording_id_present_on_all_messages(
        self,
        stream_to_daemon_with_capture: DaemonRDMCapture,
    ) -> None:
        """All captured messages should have non-empty recording_id."""
        stream_to_daemon_with_capture.capture.enqueued.clear()

        nc.log_joint_positions(positions={"joint1": 0.5}, timestamp=1234567890.0)
        nc.log_joint_velocities(velocities={"joint1": 0.1}, timestamp=1234567890.0)

        assert _wait_for(
            lambda: len(stream_to_daemon_with_capture.capture.enqueued) >= 2,
            timeout=5,
        ), "Expected 2 messages"

        for message in stream_to_daemon_with_capture.capture.enqueued:
            assert message.recording_id, "recording_id should be non-empty"

        recording_ids = {
            msg.recording_id for msg in stream_to_daemon_with_capture.capture.enqueued
        }
        assert (
            len(recording_ids) == 1
        ), f"All messages should have same recording_id, got {recording_ids}"
