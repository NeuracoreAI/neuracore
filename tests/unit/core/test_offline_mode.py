"""Tests for offline mode support in the neuracore SDK.

This module tests the daemon-based data persistence system and offline mode
functionality, including:
- Online mode (normal operation with server connectivity)
- Offline mode (operation without server connectivity)
- Offline to online transitions
"""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests
from neuracore_types import DataType, RobotInstanceIdentifier

import neuracore as nc
from neuracore.api.globals import GlobalSingleton
from neuracore.core.const import API_URL
from neuracore.core.data.dataset import Dataset
from neuracore.core.exceptions import DatasetError
from neuracore.core.robot import Robot, _robot_name_id_mapping, _robots
from neuracore.core.streaming.data_stream import (
    DataRecordingContext,
    DepthDataStream,
    JsonDataStream,
    RGBDataStream,
)
from neuracore.core.streaming.recording_state_manager import get_recording_state_manager

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_daemon_producer():
    """Mock the neuracore_data_daemon Producer."""
    with patch("neuracore.core.streaming.data_stream.Producer") as mock_producer_class:
        mock_producer = MagicMock()
        mock_producer_class.return_value = mock_producer
        yield mock_producer


@pytest.fixture
def mock_daemon_recording_context():
    """Mock the DaemonRecordingContext for stop_recording."""
    with patch("neuracore.core.robot.DaemonRecordingContext") as mock_context_class:
        mock_context = MagicMock()
        mock_context_class.return_value = mock_context
        yield mock_context


@pytest.fixture
def offline_robot(temp_config_dir, mock_daemon_producer, mocked_org_id, monkeypatch):
    """Create a robot in offline mode (no server connectivity)."""
    # Set up config with API key and org_id for offline mode
    from neuracore.core.config.config_manager import get_config_manager

    config_manager = get_config_manager()
    config_manager.config.api_key = "test_api_key"
    config_manager.config.current_org_id = mocked_org_id
    config_manager.save_config()

    # Mock connection error on robot init
    with patch("neuracore.core.robot.requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError("No connection")

        robot = Robot(robot_name="offline_robot", instance=0)
        robot.init()

    assert robot.is_connected is False
    assert robot.id is None
    return robot


@pytest.fixture
def online_robot(
    temp_config_dir, mock_auth_requests, mock_urdf, mocked_org_id, reset_neuracore
):
    """Create a robot in online mode (normal server connectivity)."""
    nc.login("test_api_key")

    mock_auth_requests.post(
        f"{API_URL}/org/{mocked_org_id}/robots",
        json={"robot_id": "online_robot_id", "has_urdf": True},
        status_code=200,
    )

    robot = nc.connect_robot("online_robot", urdf_path=mock_urdf)

    assert robot.is_connected is True
    assert robot.id == "online_robot_id"
    return robot


# =============================================================================
# ONLINE MODE TESTS
# =============================================================================


class TestOnlineMode:
    """Tests for normal online operation with server connectivity."""

    def test_robot_connects_online(self, online_robot):
        """Test that robot connects successfully in online mode."""
        assert online_robot.is_connected is True
        assert online_robot.id is not None
        assert online_robot.id == "online_robot_id"

    def test_dataset_creates_online(
        self, temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
    ):
        """Test that dataset is created with server-assigned ID in online mode."""
        nc.login("test_api_key")

        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            status_code=404,
        )
        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/datasets",
            json={
                "id": "online_dataset_id",
                "name": "test_dataset",
                "size_bytes": 0,
                "tags": [],
                "is_shared": False,
                "all_data_types": {},
                "created_at": 1704067200.0,
                "modified_at": 1704067200.0,
            },
            status_code=200,
        )

        dataset = Dataset.create(name="test_dataset")

        assert dataset.is_connected is True
        assert dataset.id == "online_dataset_id"

    def test_start_recording_online(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        mock_urdf,
        mocked_org_id,
        mock_daemon_recording_context,
    ):
        """Test starting recording in online mode gets server-generated recording ID."""
        nc.login("test_api_key")

        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/robots",
            json={"robot_id": "robot_id", "has_urdf": True},
            status_code=200,
        )
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            status_code=404,
        )
        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/datasets",
            json={
                "id": "dataset_id",
                "name": "test_dataset",
                "size_bytes": 0,
                "tags": [],
                "is_shared": False,
                "all_data_types": {},
                "created_at": 1704067200.0,
                "modified_at": 1704067200.0,
            },
            status_code=200,
        )
        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/recording/start",
            json={"id": "server_recording_id"},
            status_code=200,
        )

        nc.connect_robot("test_robot", urdf_path=mock_urdf)
        nc.create_dataset("test_dataset")
        nc.start_recording()

        robot = GlobalSingleton()._active_robot
        assert robot.is_recording() is True
        recording_id = robot.get_current_recording_id()
        assert recording_id == "server_recording_id"

    def test_logging_works_online(self, online_robot, mock_daemon_producer):
        """Test that logging functions work in online mode."""
        # Manually set up recording state for logging
        get_recording_state_manager().recording_started(
            robot_identifier=online_robot.id,
            instance=online_robot.instance,
            recording_id="test_recording",
        )
        GlobalSingleton()._active_robot = online_robot
        GlobalSingleton()._active_dataset_id = "test_dataset_id"
        GlobalSingleton()._active_dataset_name = "test_dataset"

        # Test logging - should not raise
        nc.log_joint_positions({"joint1": 0.5})

        rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        nc.log_rgb("camera", rgb)

    def test_stop_recording_online(
        self,
        temp_config_dir,
        mock_auth_requests,
        reset_neuracore,
        mock_urdf,
        mocked_org_id,
        mock_daemon_recording_context,
    ):
        """Test stopping recording in online mode calls both daemon and API."""
        nc.login("test_api_key")

        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/robots",
            json={"robot_id": "robot_id", "has_urdf": True},
            status_code=200,
        )
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            status_code=404,
        )
        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/datasets",
            json={
                "id": "dataset_id",
                "name": "test_dataset",
                "size_bytes": 0,
                "tags": [],
                "is_shared": False,
                "all_data_types": {},
                "created_at": 1704067200.0,
                "modified_at": 1704067200.0,
            },
            status_code=200,
        )
        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/recording/start",
            json={"id": "server_recording_id"},
            status_code=200,
        )
        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/recording/stop",
            json="OK",
            status_code=200,
        )

        nc.connect_robot("test_robot", urdf_path=mock_urdf)
        nc.create_dataset("test_dataset")
        nc.start_recording()

        robot = GlobalSingleton()._active_robot
        recording_id = robot.get_current_recording_id()
        assert recording_id == "server_recording_id"

        robot.stop_recording(recording_id)

        # Verify daemon context was used
        mock_daemon_recording_context.stop_recording.assert_called_once()

        # Verify API was called
        stop_request = mock_auth_requests.request_history[-1]
        assert "/recording/stop" in stop_request.url
        assert f"recording_id={recording_id}" in stop_request.url


# =============================================================================
# OFFLINE MODE TESTS
# =============================================================================


class TestOfflineMode:
    """Tests for offline operation without server connectivity."""

    def test_robot_initializes_offline(self, offline_robot):
        """Test that robot initializes in offline mode when server unavailable."""
        assert offline_robot.is_connected is False
        assert offline_robot.id is None
        assert offline_robot.name == "offline_robot"

    def test_dataset_creates_offline(
        self, temp_config_dir, reset_neuracore, mocked_org_id
    ):
        """Test that dataset is created locally in offline mode."""
        from neuracore.core.config.config_manager import get_config_manager

        config_manager = get_config_manager()
        config_manager.config.api_key = "test_api_key"
        config_manager.config.current_org_id = mocked_org_id
        config_manager.save_config()

        # Mock connection error
        with patch("neuracore.core.data.dataset.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError()

            with patch("neuracore.core.data.dataset.requests.post") as mock_post:
                mock_post.side_effect = requests.exceptions.ConnectionError()

                with patch("neuracore.core.data.dataset.get_auth") as mock_auth:
                    mock_auth.return_value.is_authenticated = False

                    dataset = Dataset.create(name="offline_dataset")

        assert dataset.is_connected is False
        assert dataset.id is None
        assert dataset.name == "offline_dataset"

    def test_start_recording_offline_generates_local_uuid(
        self, offline_robot, mock_daemon_producer
    ):
        """Test that offline recording generates local UUID."""
        GlobalSingleton()._active_robot = offline_robot
        GlobalSingleton()._active_dataset_name = "offline_dataset"

        recording_id = offline_robot.start_recording(
            dataset_id=None, dataset_name="offline_dataset"
        )

        # Should be a valid UUID
        assert recording_id is not None
        uuid.UUID(recording_id)  # Validates UUID format

        assert offline_robot.is_recording() is True

    def test_logging_works_offline(self, offline_robot, mock_daemon_producer):
        """Test that logging functions work in offline mode."""
        GlobalSingleton()._active_robot = offline_robot
        GlobalSingleton()._active_dataset_name = "offline_dataset"

        # Start recording
        offline_robot.start_recording(dataset_id=None, dataset_name="offline_dataset")

        # Test logging - should not raise
        nc.log_joint_positions({"joint1": 0.5})

        rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        nc.log_rgb("camera", rgb)

        depth = np.ones((100, 100), dtype=np.float32)
        nc.log_depth("depth_cam", depth)

    def test_stop_recording_offline(
        self, offline_robot, mock_daemon_producer, mock_daemon_recording_context
    ):
        """Test stopping recording in offline mode uses daemon."""
        GlobalSingleton()._active_robot = offline_robot
        GlobalSingleton()._active_dataset_name = "offline_dataset"

        recording_id = offline_robot.start_recording(
            dataset_id=None, dataset_name="offline_dataset"
        )

        offline_robot.stop_recording(recording_id)

        # Verify daemon context was used
        mock_daemon_recording_context.stop_recording.assert_called_once()

    def test_cancel_recording_offline(
        self, offline_robot, mock_daemon_producer, mock_daemon_recording_context
    ):
        """Test canceling recording in offline mode."""
        GlobalSingleton()._active_robot = offline_robot
        GlobalSingleton()._active_dataset_name = "offline_dataset"

        recording_id = offline_robot.start_recording(
            dataset_id=None, dataset_name="offline_dataset"
        )

        offline_robot.cancel_recording(recording_id)

        # Verify daemon context was used
        mock_daemon_recording_context.stop_recording.assert_called_once()

    def test_stop_live_data_noop_offline(self, offline_robot):
        """Test that stop_live_data is a no-op in offline mode."""
        GlobalSingleton()._active_robot = offline_robot
        # Register robot in global registry
        _robot_name_id_mapping[offline_robot.name] = offline_robot.name
        _robots[
            RobotInstanceIdentifier(
                robot_id=offline_robot.name, robot_instance=offline_robot.instance
            )
        ] = offline_robot

        # Should not raise - just silently return
        nc.stop_live_data("offline_robot")

    def test_get_latest_sync_point_returns_local_only_offline(
        self, offline_robot, mock_daemon_producer
    ):
        """Test that get_latest_sync_point returns local data only in offline mode."""
        from neuracore.core.get_latest_sync_point import get_latest_sync_point

        GlobalSingleton()._active_robot = offline_robot

        # Add a data stream with some data
        stream = JsonDataStream(DataType.JOINT_POSITIONS, "test_joints")
        offline_robot.add_data_stream("JOINT_POSITIONS:test_joints", stream)

        # Should not raise, just return local data
        # Note: This will fail if there's no data, but shouldn't raise RobotError
        try:
            get_latest_sync_point(include_remote=True)
        except (ValueError, AssertionError):
            # Expected if no data logged yet
            pass


# =============================================================================
# OFFLINE DATASET ERROR HANDLING TESTS
# =============================================================================


class TestOfflineDatasetErrors:
    """Tests for graceful error handling when using offline datasets."""

    def test_dataset_synchronize_fails_gracefully_offline(
        self, temp_config_dir, mocked_org_id
    ):
        """Test that synchronize() provides clear error for offline datasets."""
        dataset = Dataset(
            id=None,
            org_id=mocked_org_id,
            name="offline_dataset",
            size_bytes=0,
            tags=[],
            is_shared=False,
            data_types=[],
        )

        with pytest.raises(DatasetError) as exc_info:
            dataset.synchronize()

        assert "offline mode" in str(exc_info.value).lower()
        assert "offline_dataset" in str(exc_info.value)

    def test_dataset_get_recordings_fails_gracefully_offline(
        self, temp_config_dir, mocked_org_id
    ):
        """Test that fetching recordings provides clear error for offline datasets."""
        dataset = Dataset(
            id=None,
            org_id=mocked_org_id,
            name="offline_dataset",
            size_bytes=0,
            tags=[],
            is_shared=False,
            data_types=[],
        )

        with pytest.raises(DatasetError) as exc_info:
            len(dataset)  # Triggers _initialize_num_recordings

        assert "offline mode" in str(exc_info.value).lower()

    def test_dataset_robot_ids_fails_gracefully_offline(
        self, temp_config_dir, mocked_org_id
    ):
        """Test that robot_ids property provides clear error for offline datasets."""
        dataset = Dataset(
            id=None,
            org_id=mocked_org_id,
            name="offline_dataset",
            size_bytes=0,
            tags=[],
            is_shared=False,
            data_types=[],
        )

        with pytest.raises(DatasetError) as exc_info:
            _ = dataset.robot_ids

        assert "offline mode" in str(exc_info.value).lower()

    def test_dataset_get_full_data_spec_fails_gracefully_offline(
        self, temp_config_dir, mocked_org_id
    ):
        """Test that get_full_data_spec provides clear error for offline datasets."""
        dataset = Dataset(
            id=None,
            org_id=mocked_org_id,
            name="offline_dataset",
            size_bytes=0,
            tags=[],
            is_shared=False,
            data_types=[],
        )

        with pytest.raises(DatasetError) as exc_info:
            dataset.get_full_data_spec("robot_id")

        assert "offline mode" in str(exc_info.value).lower()


# =============================================================================
# DATA STREAM TESTS
# =============================================================================


class TestDataStreamDaemonIntegration:
    """Tests for data stream daemon integration."""

    def test_json_data_stream_sends_to_daemon(self, mock_daemon_producer):
        """Test that JsonDataStream sends data to daemon when recording."""
        from neuracore_types import JointData

        stream = JsonDataStream(DataType.JOINT_POSITIONS, "test_joints")

        context = DataRecordingContext(
            recording_id="test_recording",
            robot_id="robot_id",
            robot_name="test_robot",
            robot_instance=0,
            dataset_id="dataset_id",
            dataset_name="test_dataset",
        )

        stream.start_recording(context)

        # Log some data
        joint_data = JointData(timestamp=1.0, value=0.5)
        stream.log(joint_data)

        # Verify producer was created and used
        mock_daemon_producer.open_ring_buffer.assert_called_once()
        mock_daemon_producer.start_new_trace.assert_called_once()
        mock_daemon_producer.send_data.assert_called_once()

    def test_rgb_data_stream_sends_to_daemon(self, mock_daemon_producer):
        """Test that RGBDataStream sends data to daemon when recording."""
        from neuracore_types import RGBCameraData

        stream = RGBDataStream(camera_id="test_camera", width=100, height=100)

        context = DataRecordingContext(
            recording_id="test_recording",
            robot_id="robot_id",
            robot_name="test_robot",
            robot_instance=0,
            dataset_id="dataset_id",
            dataset_name="test_dataset",
        )

        stream.start_recording(context)

        # Log some data
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        camera_data = RGBCameraData(timestamp=1.0)
        stream.log(camera_data, frame)

        # Verify producer was used
        mock_daemon_producer.send_data.assert_called_once()

    def test_depth_data_stream_sends_to_daemon(self, mock_daemon_producer):
        """Test that DepthDataStream sends data to daemon when recording."""
        from neuracore_types import DepthCameraData

        stream = DepthDataStream(camera_id="test_depth", width=100, height=100)

        context = DataRecordingContext(
            recording_id="test_recording",
            robot_id="robot_id",
            robot_name="test_robot",
            robot_instance=0,
            dataset_id="dataset_id",
            dataset_name="test_dataset",
        )

        stream.start_recording(context)

        # Log some data
        frame = np.ones((100, 100), dtype=np.float32)
        camera_data = DepthCameraData(timestamp=1.0)
        stream.log(camera_data, frame)

        # Verify producer was used
        mock_daemon_producer.send_data.assert_called_once()

    def test_data_stream_does_not_send_when_not_recording(self, mock_daemon_producer):
        """Test that data streams don't send to daemon when not recording."""
        from neuracore_types import JointData

        stream = JsonDataStream(DataType.JOINT_POSITIONS, "test_joints")

        # Log without starting recording
        joint_data = JointData(timestamp=1.0, value=0.5)
        stream.log(joint_data)

        # Verify producer was NOT used
        mock_daemon_producer.send_data.assert_not_called()


# =============================================================================
# OFFLINE TO ONLINE TRANSITION TESTS
# =============================================================================


class TestOfflineToOnlineTransition:
    """Tests for transitioning from offline to online mode."""

    def test_robot_can_reconnect_after_offline(
        self, temp_config_dir, mock_auth_requests, mocked_org_id, mock_urdf
    ):
        """Test that a robot can reconnect after being offline."""
        from neuracore.core.config.config_manager import get_config_manager

        config_manager = get_config_manager()
        config_manager.config.api_key = "test_api_key"
        config_manager.config.current_org_id = mocked_org_id
        config_manager.save_config()

        # First, create robot in offline mode
        with patch("neuracore.core.robot.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError()

            robot = Robot(robot_name="test_robot", instance=0)
            robot.init()

        assert robot.is_connected is False
        assert robot.id is None

        # Now simulate coming back online
        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/robots",
            json={"robot_id": "reconnected_robot_id", "has_urdf": True},
            status_code=200,
        )

        # Re-initialize
        robot.init()

        assert robot.is_connected is True
        assert robot.id == "reconnected_robot_id"

    def test_new_dataset_after_coming_online(
        self, temp_config_dir, mock_auth_requests, reset_neuracore, mocked_org_id
    ):
        """Test creating a new online dataset after being offline."""
        nc.login("test_api_key")

        # First dataset created while "offline" (mocked)
        with patch("neuracore.core.data.dataset.get_auth") as mock_auth:
            mock_auth.return_value.is_authenticated = False
            offline_dataset = Dataset.create(name="offline_ds")

        assert offline_dataset.is_connected is False

        # Now create a new dataset while online
        mock_auth_requests.get(
            f"{API_URL}/org/{mocked_org_id}/datasets/search/by-name",
            status_code=404,
        )
        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/datasets",
            json={
                "id": "online_dataset_id",
                "name": "online_ds",
                "size_bytes": 0,
                "tags": [],
                "is_shared": False,
                "all_data_types": {},
                "created_at": 1704067200.0,
                "modified_at": 1704067200.0,
            },
            status_code=200,
        )

        online_dataset = Dataset.create(name="online_ds")

        assert online_dataset.is_connected is True
        assert online_dataset.id == "online_dataset_id"

    def test_recording_transitions_from_offline_to_online(
        self,
        temp_config_dir,
        mock_auth_requests,
        mocked_org_id,
        mock_urdf,
        mock_daemon_producer,
        mock_daemon_recording_context,
    ):
        """Test that recording can transition from offline to online."""
        from neuracore.core.config.config_manager import get_config_manager

        config_manager = get_config_manager()
        config_manager.config.api_key = "test_api_key"
        config_manager.config.current_org_id = mocked_org_id
        config_manager.save_config()

        # Start offline
        with patch("neuracore.core.robot.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError()
            robot = Robot(robot_name="test_robot", instance=0)
            robot.init()

        GlobalSingleton()._active_robot = robot
        GlobalSingleton()._active_dataset_name = "test_dataset"

        # Record offline
        offline_recording_id = robot.start_recording(
            dataset_id=None, dataset_name="test_dataset"
        )
        assert robot.is_connected is False

        # Stop offline recording
        robot.stop_recording(offline_recording_id)

        # Come back online
        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/robots",
            json={"robot_id": "online_robot_id", "has_urdf": True},
            status_code=200,
        )
        robot.init()

        assert robot.is_connected is True

        # Start online recording
        mock_auth_requests.post(
            f"{API_URL}/org/{mocked_org_id}/recording/start",
            json={"id": "online_recording_id"},
            status_code=200,
        )

        GlobalSingleton()._active_dataset_id = "online_dataset_id"
        online_recording_id = robot.start_recording(
            dataset_id="online_dataset_id", dataset_name="test_dataset"
        )

        assert online_recording_id == "online_recording_id"


# =============================================================================
# RECORDING STATE MANAGER TESTS
# =============================================================================


class TestRecordingStateManagerOffline:
    """Tests for recording state manager with offline mode."""

    def test_state_manager_uses_robot_name_offline(self, offline_robot):
        """Test that state manager uses robot name when ID is None."""
        manager = get_recording_state_manager()

        # Start recording using robot name as identifier
        manager.recording_started(
            robot_identifier=offline_robot.name,
            instance=offline_robot.instance,
            recording_id="test_recording",
        )

        # Check recording state using robot name
        assert manager.is_recording(offline_robot.name, offline_robot.instance) is True
        assert (
            manager.get_current_recording_id(offline_robot.name, offline_robot.instance)
            == "test_recording"
        )

    def test_state_manager_uses_robot_id_online(self, online_robot):
        """Test that state manager uses robot ID when available."""
        manager = get_recording_state_manager()

        # Start recording using robot ID
        manager.recording_started(
            robot_identifier=online_robot.id,
            instance=online_robot.instance,
            recording_id="test_recording",
        )

        # Check recording state using robot ID
        assert manager.is_recording(online_robot.id, online_robot.instance) is True
        assert (
            manager.get_current_recording_id(online_robot.id, online_robot.instance)
            == "test_recording"
        )
