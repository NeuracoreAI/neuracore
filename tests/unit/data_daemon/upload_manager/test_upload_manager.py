"""Tests for UploadManager.

Tests upload orchestration, event handling, progress tracking, and error handling.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from neuracore_types import DataType, RecordingDataTraceStatus

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.event_emitter import Emitter, emitter
from neuracore.data_daemon.models import TraceErrorCode, TraceStatus
from neuracore.data_daemon.upload_management.upload_manager import UploadManager


@pytest.fixture
def test_file(tmp_path: Path) -> Path:
    """Create a 1MB test file."""
    test_file = tmp_path / "test_video.mp4"
    test_file.write_bytes(b"X" * (1024 * 1024))
    return test_file


@pytest.fixture
def mock_auth():
    """Mock get_auth() globally for all tests."""
    with patch(
        "neuracore.data_daemon.upload_management.trace_manager.get_auth"
    ) as mock_get_auth:
        auth_instance = MagicMock()
        auth_instance.get_org_id.return_value = "test-org"
        auth_instance.get_headers.return_value = {"Authorization": "Bearer test-token"}
        mock_get_auth.return_value = auth_instance
        yield mock_get_auth


@pytest.fixture
def upload_manager() -> UploadManager:
    """Create and cleanup UploadManager instance."""
    config = DaemonConfig(num_threads=2)
    manager = UploadManager(config=config)
    try:
        yield manager
    finally:
        manager.shutdown(wait=True)


@pytest.fixture
def upload_manager_with_more_threads() -> UploadManager:
    """Create UploadManager with more threads for concurrent tests."""
    config = DaemonConfig(num_threads=4)
    manager = UploadManager(config=config)
    try:
        yield manager
    finally:
        manager.shutdown(wait=True)


@pytest.fixture(autouse=True)
def setup_test_env(mock_auth):
    """Setup test environment and cleanup after each test."""
    os.environ["NEURACORE_API_KEY"] = "test-key"
    os.environ["NEURACORE_ORG_ID"] = "test-org"

    yield

    os.environ.pop("NEURACORE_API_KEY", None)
    os.environ.pop("NEURACORE_ORG_ID", None)
    emitter.remove_all_listeners(Emitter.READY_FOR_UPLOAD)
    emitter.remove_all_listeners(Emitter.UPLOAD_COMPLETE)
    emitter.remove_all_listeners(Emitter.UPLOAD_FAILED)
    emitter.remove_all_listeners(Emitter.UPLOADED_BYTES)


def test_initialize_with_config() -> None:
    """Initialization with configuration."""
    config = DaemonConfig(num_threads=8)
    manager = UploadManager(config=config)

    try:
        assert manager._num_threads == 8
        assert manager._executor is not None
        assert manager._config == config
    finally:
        manager.shutdown(wait=False)


def test_subscribes_to_ready_for_upload_event(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Should subscribe to and responds to READY_FOR_UPLOAD events."""
    upload_called = []
    original = upload_manager._upload_single_trace
    upload_manager._upload_single_trace = (
        lambda *args: upload_called.append(True) or False
    )

    try:
        emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            str(test_file),
            "trace-1",
            DataType.RGB_IMAGES,
            "camera",
            "rec-1",
            0,
        )
        time.sleep(0.5)

        assert len(upload_called) == 1
    finally:
        upload_manager._upload_single_trace = original


def test_register_backend_trace_on_upload(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Registers trace with backend API."""
    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.post"
        ) as mock_post,
        patch("neuracore.data_daemon.upload_management.trace_manager.requests.put"),
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock backend trace registration
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "backend-trace-123"}
        mock_post.return_value.raise_for_status = MagicMock()

        # Mock uploader
        mock_instance = MagicMock()
        mock_instance.upload.return_value = (True, 1024 * 1024, None)
        MockUploader.return_value = mock_instance

        emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            str(test_file),
            "trace-1",
            DataType.RGB_IMAGES,
            "camera",
            "rec-1",
            0,
        )
        time.sleep(0.5)

        # Verify registration was called with correct parameters
        assert mock_post.called
        call_args = mock_post.call_args
        assert "rec-1" in call_args[0][0]
        assert call_args[1]["json"] == {"data_type": "RGB_IMAGES"}


def test_update_backend_with_upload_started(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Update backend with UPLOAD_STARTED status."""
    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.post"
        ) as mock_post,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.put"
        ) as mock_put,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock backend responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "backend-trace-123"}
        mock_post.return_value.raise_for_status = MagicMock()

        mock_put.return_value.status_code = 200
        mock_put.return_value.raise_for_status = MagicMock()

        # Mock uploader
        mock_instance = MagicMock()
        mock_instance.upload.return_value = (True, 1024 * 1024, None)
        MockUploader.return_value = mock_instance

        emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            str(test_file),
            "trace-1",
            DataType.RGB_IMAGES,
            "camera",
            "rec-1",
            0,
        )
        time.sleep(0.5)

        put_calls = [call[1]["json"] for call in mock_put.call_args_list]
        assert any(
            call.get("status") == RecordingDataTraceStatus.UPLOAD_STARTED
            for call in put_calls
        )


def test_update_backend_with_upload_complete(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Test UploadManager updates backend with UPLOAD_COMPLETE status and bytes."""
    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.post"
        ) as mock_post,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.put"
        ) as mock_put,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock backend responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "backend-trace-123"}
        mock_post.return_value.raise_for_status = MagicMock()

        mock_put.return_value.status_code = 200
        mock_put.return_value.raise_for_status = MagicMock()

        # Mock uploader
        mock_instance = MagicMock()
        mock_instance.upload.return_value = (True, 1024 * 1024, None)
        MockUploader.return_value = mock_instance

        emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            str(test_file),
            "trace-1",
            DataType.RGB_IMAGES,
            "camera",
            "rec-1",
            0,
        )
        time.sleep(0.5)

        put_calls = [call[1]["json"] for call in mock_put.call_args_list]
        complete_calls = [
            call
            for call in put_calls
            if call.get("status") == RecordingDataTraceStatus.UPLOAD_COMPLETE
        ]

        assert len(complete_calls) == 1
        assert complete_calls[0]["uploaded_bytes"] == 1024 * 1024
        assert complete_calls[0]["total_bytes"] == 1024 * 1024


def test_emits_upload_complete_event(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Manager should emit UPLOAD_COMPLETE event on successful upload."""
    completed = []

    def on_complete(trace_id: str, recording_id: str) -> None:
        completed.append((trace_id, recording_id))

    emitter.on(Emitter.UPLOAD_COMPLETE, on_complete)

    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.post"
        ) as mock_post,
        patch("neuracore.data_daemon.upload_management.trace_manager.requests.put"),
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock backend responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "backend-trace-123"}
        mock_post.return_value.raise_for_status = MagicMock()

        # Mock uploader
        mock_instance = MagicMock()
        mock_instance.upload.return_value = (True, 1024 * 1024, None)
        MockUploader.return_value = mock_instance

        emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            str(test_file),
            "trace-1",
            DataType.RGB_IMAGES,
            "camera",
            "rec-1",
            0,
        )
        time.sleep(0.5)

        assert len(completed) == 1
        assert completed[0] == ("trace-1", "rec-1")


def test_upload_manager_emits_uploaded_bytes_events(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Manager should emit UPLOADED_BYTES events during upload."""
    progress_events = []

    def on_progress(trace_id: str, bytes_uploaded: int) -> None:
        progress_events.append((trace_id, bytes_uploaded))

    emitter.on(Emitter.UPLOADED_BYTES, on_progress)

    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.post"
        ) as mock_post,
        patch("neuracore.data_daemon.upload_management.trace_manager.requests.put"),
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "backend-trace-123"}
        mock_post.return_value.raise_for_status = MagicMock()

        mock_instance = MagicMock()

        def mock_upload():
            callback = MockUploader.call_args[1]["progress_callback"]
            callback(256 * 1024)
            callback(256 * 1024)
            callback(512 * 1024)
            return (True, 1024 * 1024, None)

        mock_instance.upload.side_effect = mock_upload
        MockUploader.return_value = mock_instance

        emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            str(test_file),
            "trace-1",
            DataType.RGB_IMAGES,
            "camera",
            "rec-1",
            0,
        )
        time.sleep(0.5)

        assert len(progress_events) == 3
        assert progress_events[0] == ("trace-1", 256 * 1024)
        assert progress_events[1] == ("trace-1", 512 * 1024)
        assert progress_events[2] == ("trace-1", 1024 * 1024)


def test_upload_failure(upload_manager: UploadManager, test_file: Path) -> None:
    """Test UploadManager handles file upload failure."""
    failures = []

    def on_failure(trace_id, bytes_uploaded, status, error_code, error_message) -> None:
        failures.append(
            {"trace_id": trace_id, "error_code": error_code, "status": status}
        )

    emitter.on(Emitter.UPLOAD_FAILED, on_failure)

    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.post"
        ) as mock_post,
        patch("neuracore.data_daemon.upload_management.trace_manager.requests.put"),
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock backend responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "backend-trace-123"}
        mock_post.return_value.raise_for_status = MagicMock()

        # Mock uploader failure
        mock_instance = MagicMock()
        mock_instance.upload.return_value = (False, 512 * 1024, "Network error")
        MockUploader.return_value = mock_instance

        emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            str(test_file),
            "trace-1",
            DataType.RGB_IMAGES,
            "camera",
            "rec-1",
            0,
        )
        time.sleep(0.5)

        assert len(failures) == 1
        assert failures[0]["trace_id"] == "trace-1"
        assert failures[0]["error_code"] == TraceErrorCode.NETWORK_ERROR
        assert failures[0]["status"] == TraceStatus.WRITTEN


def test_file_not_found(upload_manager: UploadManager) -> None:
    """Test UploadManager handles missing file."""
    failures = []

    def on_failure(trace_id, bytes_uploaded, status, error_code, error_message) -> None:
        failures.append({"status": status, "error_code": error_code})

    emitter.on(Emitter.UPLOAD_FAILED, on_failure)

    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.post"
        ) as mock_post,
        patch("neuracore.data_daemon.upload_management.trace_manager.requests.put"),
    ):
        # Mock backend responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "backend-trace-123"}
        mock_post.return_value.raise_for_status = MagicMock()

        emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            "/nonexistent/file.mp4",
            "trace-1",
            DataType.RGB_IMAGES,
            "camera",
            "rec-1",
            0,
        )
        time.sleep(0.5)

        assert len(failures) == 1
        assert failures[0]["status"] == TraceStatus.FAILED
        assert failures[0]["error_code"] == TraceErrorCode.UPLOAD_FAILED


def test_upload_manager_passes_correct_parameters_to_uploader(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Test UploadManager passes correct parameters to ResumableFileUploader."""
    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.post"
        ) as mock_post,
        patch("neuracore.data_daemon.upload_management.trace_manager.requests.put"),
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "backend-trace-123"}
        mock_post.return_value.raise_for_status = MagicMock()

        mock_instance = MagicMock()
        mock_instance.upload.return_value = (True, 1024, None)
        MockUploader.return_value = mock_instance

        emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            str(test_file),
            "trace-1",
            DataType.RGB_IMAGES,
            "camera",
            "rec-1",
            512 * 1024,  # Resume from 512KB
        )
        time.sleep(0.5)

        # Verify constructor args
        assert MockUploader.called
        call_kwargs = MockUploader.call_args[1]
        assert call_kwargs["recording_id"] == "rec-1"
        assert call_kwargs["filepath"] == str(test_file)
        assert call_kwargs["content_type"] == "video/mp4"
        assert call_kwargs["bytes_uploaded"] == 512 * 1024
        assert "RGB_IMAGES/camera/" in call_kwargs["cloud_filepath"]
        assert call_kwargs["progress_callback"] is not None


def test_upload_manager_handles_concurrent_uploads(
    upload_manager_with_more_threads: UploadManager, test_file: Path
) -> None:
    """Test UploadManager handles multiple concurrent uploads."""
    completed = []

    def on_complete(trace_id: str, recording_id: str) -> None:
        completed.append(trace_id)

    emitter.on(Emitter.UPLOAD_COMPLETE, on_complete)

    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.post"
        ) as mock_post,
        patch("neuracore.data_daemon.upload_management.trace_manager.requests.put"),
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock backend responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "backend-trace-123"}
        mock_post.return_value.raise_for_status = MagicMock()

        # Mock uploader
        mock_instance = MagicMock()
        mock_instance.upload.return_value = (True, 1024, None)
        MockUploader.return_value = mock_instance

        # Submit 5 uploads
        for i in range(5):
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                str(test_file),
                f"trace-{i}",
                DataType.RGB_IMAGES,
                "camera",
                "rec-1",
                0,
            )

        time.sleep(1.0)

        assert len(completed) == 5
        assert set(completed) == {f"trace-{i}" for i in range(5)}


def test_upload_manager_updates_backend_periodically_during_upload(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Test UploadManager updates backend every 30 seconds during upload."""
    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.post"
        ) as mock_post,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.put"
        ) as mock_put,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.time"
        ) as mock_time,
    ):
        # Mock backend responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "backend-trace-123"}
        mock_post.return_value.raise_for_status = MagicMock()

        mock_put.return_value.status_code = 200
        mock_put.return_value.raise_for_status = MagicMock()

        # Mock time progression to trigger 30-second updates
        time_values = [0, 31, 62]
        mock_time.time.side_effect = time_values

        mock_instance = MagicMock()

        def mock_upload():
            callback = MockUploader.call_args[1]["progress_callback"]
            callback(512 * 1024)
            callback(512 * 1024)
            return (True, 1024 * 1024, None)

        mock_instance.upload.side_effect = mock_upload
        MockUploader.return_value = mock_instance

        emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            str(test_file),
            "trace-1",
            DataType.RGB_IMAGES,
            "camera",
            "rec-1",
            0,
        )
        time.sleep(0.5)

        # Verify multiple UPLOAD_STARTED updates (periodic progress)
        put_calls = [call[1]["json"] for call in mock_put.call_args_list]
        started_calls = [
            call
            for call in put_calls
            if call.get("status") == RecordingDataTraceStatus.UPLOAD_STARTED
        ]

        # Should have at least 2 progress updates
        assert len(started_calls) >= 2


def test_upload_manager_shutdown_waits_for_in_flight_uploads(test_file: Path) -> None:
    """Test UploadManager shutdown waits for in-flight uploads."""
    config = DaemonConfig(num_threads=2)
    upload_manager = UploadManager(config=config)

    upload_completed = []

    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.post"
        ) as mock_post,
        patch("neuracore.data_daemon.upload_management.trace_manager.requests.put"),
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock backend responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "backend-trace-123"}
        mock_post.return_value.raise_for_status = MagicMock()

        mock_instance = MagicMock()

        def slow_upload():
            time.sleep(0.3)  # Simulate slow upload
            upload_completed.append(True)
            return (True, 1024, None)

        mock_instance.upload.side_effect = slow_upload
        MockUploader.return_value = mock_instance

        emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            str(test_file),
            "trace-1",
            DataType.RGB_IMAGES,
            "camera",
            "rec-1",
            0,
        )

        upload_manager.shutdown(wait=True)

        assert len(upload_completed) == 1


def test_upload_manager_shutdown_unsubscribes_from_events(test_file: Path) -> None:
    """Test UploadManager shutdown unsubscribes from READY_FOR_UPLOAD events."""
    config = DaemonConfig(num_threads=2)
    upload_manager = UploadManager(config=config)

    upload_manager.shutdown()

    # Try to emit event after shutdown
    upload_called = []
    original = upload_manager._upload_single_trace
    upload_manager._upload_single_trace = lambda *args: upload_called.append(True)

    emitter.emit(
        Emitter.READY_FOR_UPLOAD,
        str(test_file),
        "trace-1",
        DataType.RGB_IMAGES,
        "camera",
        "rec-1",
        0,
    )
    time.sleep(0.3)

    assert len(upload_called) == 0
    upload_manager._upload_single_trace = original


def test_upload_manager_constructs_cloud_filepath_correctly(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Test UploadManager constructs correct cloud filepath."""
    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.requests.post"
        ) as mock_post,
        patch("neuracore.data_daemon.upload_management.trace_manager.requests.put"),
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock backend responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "backend-trace-123"}
        mock_post.return_value.raise_for_status = MagicMock()

        # Mock uploader
        mock_instance = MagicMock()
        mock_instance.upload.return_value = (True, 1024, None)
        MockUploader.return_value = mock_instance

        emitter.emit(
            Emitter.READY_FOR_UPLOAD,
            str(test_file),
            "trace-1",
            DataType.RGB_IMAGES,
            "camera_front",
            "rec-1",
            0,
        )
        time.sleep(0.5)

        # Verify cloud filepath format: data_type/data_type_name/filename
        call_kwargs = MockUploader.call_args[1]
        cloud_filepath = call_kwargs["cloud_filepath"]
        assert cloud_filepath.startswith("RGB_IMAGES/camera_front/")
        assert cloud_filepath.endswith("test_video.mp4")
