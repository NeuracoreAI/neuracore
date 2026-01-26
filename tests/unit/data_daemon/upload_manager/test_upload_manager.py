"""Tests for UploadManager.

Tests upload orchestration, event handling, progress tracking, and error handling.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
import pytest_asyncio
from neuracore_types import DataType, RecordingDataTraceStatus

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import TraceErrorCode, TraceStatus
from neuracore.data_daemon.upload_management.upload_manager import UploadManager


async def wait_for_uploads(upload_manager: UploadManager, timeout: float = 2.0):
    """Wait for all active uploads to complete."""
    start = asyncio.get_event_loop().time()
    while upload_manager._active_uploads:
        if asyncio.get_event_loop().time() - start > timeout:
            raise TimeoutError("Uploads did not complete in time")
        await asyncio.sleep(0.1)


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


@pytest_asyncio.fixture
async def client_session():
    """Create an aiohttp session for testing."""
    session = aiohttp.ClientSession()
    yield session
    await session.close()


@pytest_asyncio.fixture
async def upload_manager(client_session: aiohttp.ClientSession):
    """Create and cleanup UploadManager instance."""
    config = DaemonConfig(num_threads=2)
    manager = UploadManager(config=config, client_session=client_session)
    yield manager
    await manager.shutdown(wait=False)


@pytest.fixture(autouse=True)
def setup_test_env(mock_auth):
    """Setup test environment and cleanup after each test."""
    os.environ["NEURACORE_API_KEY"] = "test-key"
    os.environ["NEURACORE_ORG_ID"] = "test-org"

    yield

    os.environ.pop("NEURACORE_API_KEY", None)
    os.environ.pop("NEURACORE_ORG_ID", None)
    get_emitter().remove_all_listeners(Emitter.READY_FOR_UPLOAD)
    get_emitter().remove_all_listeners(Emitter.UPLOAD_COMPLETE)
    get_emitter().remove_all_listeners(Emitter.UPLOAD_FAILED)
    get_emitter().remove_all_listeners(Emitter.UPLOADED_BYTES)


@pytest.mark.asyncio
async def test_initialize_with_config(client_session: aiohttp.ClientSession) -> None:
    """Initialization with configuration."""
    config = DaemonConfig(num_threads=8)
    manager = UploadManager(config=config, client_session=client_session)

    try:
        assert manager._config == config
        assert manager._active_uploads == set()
        assert isinstance(manager._active_uploads, set)
    finally:
        await manager.shutdown(wait=False)


@pytest.mark.asyncio
async def test_subscribes_to_ready_for_upload_event(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Should subscribe to and respond to READY_FOR_UPLOAD events."""
    upload_called = []

    async def mock_upload(*args):
        upload_called.append(True)
        return False

    original = upload_manager._upload_single_trace
    upload_manager._upload_single_trace = mock_upload

    try:
        get_emitter().emit(
            Emitter.READY_FOR_UPLOAD,
            "trace-1",
            "rec-1",
            str(test_file),
            DataType.RGB_IMAGES,
            "camera",
            0,
        )
        await asyncio.sleep(0.5)

        assert len(upload_called) == 1
    finally:
        upload_manager._upload_single_trace = original


@pytest.mark.asyncio
async def test_register_backend_trace_on_upload(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Registers trace with backend API."""
    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._register_data_trace"
        ) as mock_register,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._update_data_trace"
        ) as mock_update,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock async backend trace registration
        mock_register.return_value = asyncio.Future()
        mock_register.return_value.set_result("backend-trace-123")

        mock_update.return_value = asyncio.Future()
        mock_update.return_value.set_result(None)

        # Mock uploader
        mock_instance = MagicMock()

        async def mock_upload():
            return (True, 1024 * 1024, None)

        mock_instance.upload = mock_upload
        MockUploader.return_value = mock_instance

        get_emitter().emit(
            Emitter.READY_FOR_UPLOAD,
            "trace-1",
            "rec-1",
            str(test_file),
            DataType.RGB_IMAGES,
            "camera",
            0,
        )
        await asyncio.sleep(0.5)

        # Verify registration was called
        assert mock_register.called
        call_args = mock_register.call_args
        assert call_args[0][0] == "rec-1"
        assert call_args[0][1] == DataType.RGB_IMAGES


@pytest.mark.asyncio
async def test_update_backend_with_upload_started(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Update backend with UPLOAD_STARTED status."""
    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._register_data_trace"
        ) as mock_register,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._update_data_trace"
        ) as mock_update,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock async responses
        mock_register.return_value = asyncio.Future()
        mock_register.return_value.set_result("backend-trace-123")

        mock_update.return_value = asyncio.Future()
        mock_update.return_value.set_result(None)

        # Mock uploader
        mock_instance = MagicMock()

        async def mock_upload():
            return (True, 1024 * 1024, None)

        mock_instance.upload = mock_upload
        MockUploader.return_value = mock_instance

        get_emitter().emit(
            Emitter.READY_FOR_UPLOAD,
            "trace-1",
            "rec-1",
            str(test_file),
            DataType.RGB_IMAGES,
            "camera",
            0,
        )
        await asyncio.sleep(0.5)

        # Check update calls
        update_calls = [call[0] for call in mock_update.call_args_list]
        assert any(
            call[2] == RecordingDataTraceStatus.UPLOAD_STARTED for call in update_calls
        )


@pytest.mark.asyncio
async def test_update_backend_with_upload_complete(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Test UploadManager updates backend with UPLOAD_COMPLETE status and bytes."""
    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._register_data_trace"
        ) as mock_register,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._update_data_trace"
        ) as mock_update,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock async responses
        mock_register.return_value = asyncio.Future()
        mock_register.return_value.set_result("backend-trace-123")

        mock_update.return_value = asyncio.Future()
        mock_update.return_value.set_result(None)

        # Mock uploader
        mock_instance = MagicMock()

        async def mock_upload():
            return (True, 1024 * 1024, None)

        mock_instance.upload = mock_upload
        MockUploader.return_value = mock_instance

        get_emitter().emit(
            Emitter.READY_FOR_UPLOAD,
            "trace-1",
            "rec-1",
            str(test_file),
            DataType.RGB_IMAGES,
            "camera",
            0,
        )
        await asyncio.sleep(0.5)

        # Find UPLOAD_COMPLETE call
        update_calls = mock_update.call_args_list
        complete_calls = [
            call
            for call in update_calls
            if call[0][2] == RecordingDataTraceStatus.UPLOAD_COMPLETE
        ]

        assert len(complete_calls) == 1
        # Check uploaded_bytes kwarg
        assert complete_calls[0][1]["uploaded_bytes"] == 1024 * 1024
        assert complete_calls[0][1]["total_bytes"] == 1024 * 1024


@pytest.mark.asyncio
async def test_emits_upload_complete_event(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Manager should emit UPLOAD_COMPLETE event on successful upload."""
    completed = []

    async def on_complete(trace_id: str, recording_id: str) -> None:
        completed.append((trace_id, recording_id))

    get_emitter().on(Emitter.UPLOAD_COMPLETE, on_complete)

    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._register_data_trace"
        ) as mock_register,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._update_data_trace"
        ) as mock_update,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock async responses
        mock_register.return_value = asyncio.Future()
        mock_register.return_value.set_result("backend-trace-123")

        mock_update.return_value = asyncio.Future()
        mock_update.return_value.set_result(None)

        # Mock uploader
        mock_instance = MagicMock()

        async def mock_upload():
            return (True, 1024 * 1024, None)

        mock_instance.upload = mock_upload
        MockUploader.return_value = mock_instance

        get_emitter().emit(
            Emitter.READY_FOR_UPLOAD,
            "trace-1",
            "rec-1",
            str(test_file),
            DataType.RGB_IMAGES,
            "camera",
            0,
        )
        await asyncio.sleep(0.5)

        assert len(completed) == 1
        assert completed[0] == ("trace-1", "rec-1")


@pytest.mark.asyncio
async def test_upload_manager_emits_uploaded_bytes_events(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Manager should emit UPLOADED_BYTES events during upload."""
    progress_events = []

    async def on_progress(trace_id: str, bytes_uploaded: int) -> None:
        progress_events.append((trace_id, bytes_uploaded))

    get_emitter().on(Emitter.UPLOADED_BYTES, on_progress)

    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._register_data_trace"
        ) as mock_register,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._update_data_trace"
        ) as mock_update,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock async responses
        mock_register.return_value = asyncio.Future()
        mock_register.return_value.set_result("backend-trace-123")

        mock_update.return_value = asyncio.Future()
        mock_update.return_value.set_result(None)

        mock_instance = MagicMock()

        async def mock_upload():
            callback = MockUploader.call_args[1]["progress_callback"]
            callback(256 * 1024)
            callback(256 * 1024)
            callback(512 * 1024)
            return (True, 1024 * 1024, None)

        mock_instance.upload = mock_upload
        MockUploader.return_value = mock_instance

        get_emitter().emit(
            Emitter.READY_FOR_UPLOAD,
            "trace-1",
            "rec-1",
            str(test_file),
            DataType.RGB_IMAGES,
            "camera",
            0,
        )

        await wait_for_uploads(upload_manager)
        await asyncio.sleep(0.2)

        assert len(progress_events) == 3
        assert progress_events[0] == ("trace-1", 256 * 1024)
        assert progress_events[1] == ("trace-1", 512 * 1024)
        assert progress_events[2] == ("trace-1", 1024 * 1024)


@pytest.mark.asyncio
async def test_upload_failure(upload_manager: UploadManager, test_file: Path) -> None:
    """Test UploadManager handles file upload failure."""
    failures = []

    async def on_failure(
        trace_id, bytes_uploaded, status, error_code, error_message
    ) -> None:
        failures.append(
            {"trace_id": trace_id, "error_code": error_code, "status": status}
        )

    get_emitter().on(Emitter.UPLOAD_FAILED, on_failure)

    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._register_data_trace"
        ) as mock_register,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._update_data_trace"
        ) as mock_update,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock async responses
        mock_register.return_value = asyncio.Future()
        mock_register.return_value.set_result("backend-trace-123")

        mock_update.return_value = asyncio.Future()
        mock_update.return_value.set_result(None)

        # Mock uploader failure
        mock_instance = MagicMock()

        async def mock_upload():
            return (False, 512 * 1024, "Network error")

        mock_instance.upload = mock_upload
        MockUploader.return_value = mock_instance

        get_emitter().emit(
            Emitter.READY_FOR_UPLOAD,
            "trace-1",
            "rec-1",
            str(test_file),
            DataType.RGB_IMAGES,
            "camera",
            0,
        )
        await asyncio.sleep(0.5)

        assert len(failures) == 1
        assert failures[0]["trace_id"] == "trace-1"
        assert failures[0]["error_code"] == TraceErrorCode.NETWORK_ERROR
        assert failures[0]["status"] == TraceStatus.WRITTEN


@pytest.mark.asyncio
async def test_file_not_found(upload_manager: UploadManager) -> None:
    """Test UploadManager handles missing file."""
    failures = []

    async def on_failure(
        trace_id, bytes_uploaded, status, error_code, error_message
    ) -> None:
        failures.append({"status": status, "error_code": error_code})

    get_emitter().on(Emitter.UPLOAD_FAILED, on_failure)

    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._register_data_trace"
        ) as mock_register,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._update_data_trace"
        ) as mock_update,
    ):
        # Mock async responses
        mock_register.return_value = asyncio.Future()
        mock_register.return_value.set_result("backend-trace-123")

        mock_update.return_value = asyncio.Future()
        mock_update.return_value.set_result(None)

        get_emitter().emit(
            Emitter.READY_FOR_UPLOAD,
            "trace-1",
            "rec-1",
            "/nonexistent/file.mp4",
            DataType.RGB_IMAGES,
            "camera",
            0,
        )
        await asyncio.sleep(0.5)

        assert len(failures) == 1
        assert failures[0]["status"] == TraceStatus.FAILED
        assert failures[0]["error_code"] == TraceErrorCode.UPLOAD_FAILED


@pytest.mark.asyncio
async def test_upload_manager_passes_correct_parameters_to_uploader(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Test UploadManager passes correct parameters to ResumableFileUploader."""
    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._register_data_trace"
        ) as mock_register,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._update_data_trace"
        ) as mock_update,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock async responses
        mock_register.return_value = asyncio.Future()
        mock_register.return_value.set_result("backend-trace-123")

        mock_update.return_value = asyncio.Future()
        mock_update.return_value.set_result(None)

        mock_instance = MagicMock()

        async def mock_upload():
            return (True, 1024, None)

        mock_instance.upload = mock_upload
        MockUploader.return_value = mock_instance

        get_emitter().emit(
            Emitter.READY_FOR_UPLOAD,
            "trace-1",
            "rec-1",
            str(test_file),
            DataType.RGB_IMAGES,
            "camera",
            512 * 1024,  # Resume from 512KB
        )
        await asyncio.sleep(0.5)

        # Verify constructor args
        assert MockUploader.called
        call_kwargs = MockUploader.call_args[1]
        assert call_kwargs["recording_id"] == "rec-1"
        assert call_kwargs["filepath"] == str(test_file)
        assert call_kwargs["content_type"] == "video/mp4"
        assert call_kwargs["bytes_uploaded"] == 512 * 1024
        assert "RGB_IMAGES/camera/" in call_kwargs["cloud_filepath"]
        assert call_kwargs["progress_callback"] is not None
        assert call_kwargs["client_session"] is not None


@pytest.mark.asyncio
async def test_upload_manager_handles_concurrent_uploads(
    client_session: aiohttp.ClientSession, test_file: Path
) -> None:
    """Test UploadManager handles multiple concurrent uploads."""
    config = DaemonConfig(num_threads=4)
    upload_manager = UploadManager(config=config, client_session=client_session)

    completed = []

    async def on_complete(trace_id: str, recording_id: str) -> None:
        completed.append(trace_id)

    get_emitter().on(Emitter.UPLOAD_COMPLETE, on_complete)

    try:
        with (
            patch(
                "neuracore.data_daemon.upload_management.trace_manager.TraceManager._register_data_trace"
            ) as mock_register,
            patch(
                "neuracore.data_daemon.upload_management.trace_manager.TraceManager._update_data_trace"
            ) as mock_update,
            patch(
                "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
            ) as MockUploader,
        ):
            # Mock async responses
            mock_register.return_value = asyncio.Future()
            mock_register.return_value.set_result("backend-trace-123")

            mock_update.return_value = asyncio.Future()
            mock_update.return_value.set_result(None)

            # Mock uploader
            mock_instance = MagicMock()

            async def mock_upload():
                return (True, 1024, None)

            mock_instance.upload = mock_upload
            MockUploader.return_value = mock_instance

            # Submit 5 uploads
            for i in range(5):
                get_emitter().emit(
                    Emitter.READY_FOR_UPLOAD,
                    f"trace-{i}",
                    "rec-1",
                    str(test_file),
                    DataType.RGB_IMAGES,
                    "camera",
                    0,
                )

            await asyncio.sleep(1.0)

            assert len(completed) == 5
            assert set(completed) == {f"trace-{i}" for i in range(5)}
    finally:
        await upload_manager.shutdown(wait=False)


@pytest.mark.asyncio
async def test_upload_manager_shutdown_waits_for_in_flight_uploads(
    client_session: aiohttp.ClientSession, test_file: Path
) -> None:
    """Test UploadManager shutdown waits for in-flight uploads."""
    config = DaemonConfig(num_threads=2)
    upload_manager = UploadManager(config=config, client_session=client_session)

    upload_completed = []

    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._register_data_trace"
        ) as mock_register,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._update_data_trace"
        ) as mock_update,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock async responses
        mock_register.return_value = asyncio.Future()
        mock_register.return_value.set_result("backend-trace-123")

        mock_update.return_value = asyncio.Future()
        mock_update.return_value.set_result(None)

        mock_instance = MagicMock()

        async def slow_upload():
            await asyncio.sleep(0.3)
            upload_completed.append(True)
            return (True, 1024, None)

        mock_instance.upload = slow_upload
        MockUploader.return_value = mock_instance

        get_emitter().emit(
            Emitter.READY_FOR_UPLOAD,
            "trace-1",
            "rec-1",
            str(test_file),
            DataType.RGB_IMAGES,
            "camera",
            0,
        )

        await asyncio.sleep(0.2)

        await upload_manager.shutdown(wait=True)

        assert len(upload_completed) == 1


@pytest.mark.asyncio
async def test_upload_manager_shutdown_unsubscribes_from_events(
    client_session: aiohttp.ClientSession, test_file: Path
) -> None:
    """Test UploadManager shutdown unsubscribes from READY_FOR_UPLOAD events."""
    config = DaemonConfig(num_threads=2)
    upload_manager = UploadManager(config=config, client_session=client_session)

    await upload_manager.shutdown()

    # Try to emit event after shutdown
    upload_called = []

    async def mock_upload(*args):
        upload_called.append(True)

    original = upload_manager._upload_single_trace
    upload_manager._upload_single_trace = mock_upload

    get_emitter().emit(
        Emitter.READY_FOR_UPLOAD,
        "trace-1",
        "rec-1",
        str(test_file),
        DataType.RGB_IMAGES,
        "camera",
        0,
    )
    await asyncio.sleep(0.3)

    assert len(upload_called) == 0
    upload_manager._upload_single_trace = original


@pytest.mark.asyncio
async def test_upload_manager_constructs_cloud_filepath_correctly(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Test UploadManager constructs correct cloud filepath."""
    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._register_data_trace"
        ) as mock_register,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._update_data_trace"
        ) as mock_update,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
    ):
        # Mock async responses
        mock_register.return_value = asyncio.Future()
        mock_register.return_value.set_result("backend-trace-123")

        mock_update.return_value = asyncio.Future()
        mock_update.return_value.set_result(None)

        # Mock uploader
        mock_instance = MagicMock()

        async def mock_upload():
            return (True, 1024, None)

        mock_instance.upload = mock_upload
        MockUploader.return_value = mock_instance

        get_emitter().emit(
            Emitter.READY_FOR_UPLOAD,
            "trace-1",
            "rec-1",
            str(test_file),
            DataType.RGB_IMAGES,
            "camera_front",
            0,
        )
        await asyncio.sleep(0.5)

        call_kwargs = MockUploader.call_args[1]
        cloud_filepath = call_kwargs["cloud_filepath"]
        assert cloud_filepath.startswith("RGB_IMAGES/camera_front/")
        assert cloud_filepath.endswith("test_video.mp4")


@pytest.mark.asyncio
async def test_upload_manager_updates_backend_periodically_during_upload(
    upload_manager: UploadManager, test_file: Path
) -> None:
    """Test UploadManager updates backend every 30 seconds during upload."""
    with (
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._register_data_trace"
        ) as mock_register,
        patch(
            "neuracore.data_daemon.upload_management.trace_manager.TraceManager._update_data_trace"
        ) as mock_update,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader,
        patch(
            "neuracore.data_daemon.upload_management.upload_manager.time"
        ) as mock_time,
    ):
        # Mock async responses
        mock_register.return_value = asyncio.Future()
        mock_register.return_value.set_result("backend-trace-123")

        mock_update.return_value = asyncio.Future()
        mock_update.return_value.set_result(None)

        # Mock time progression to trigger 30-second updates
        time_values = [0, 31, 62]
        mock_time.time.side_effect = time_values

        mock_instance = MagicMock()

        async def mock_upload():
            callback = MockUploader.call_args[1]["progress_callback"]
            callback(512 * 1024)
            callback(512 * 1024)
            return (True, 1024 * 1024, None)

        mock_instance.upload = mock_upload
        MockUploader.return_value = mock_instance

        get_emitter().emit(
            Emitter.READY_FOR_UPLOAD,
            "trace-1",
            "rec-1",
            str(test_file),
            DataType.RGB_IMAGES,
            "camera",
            0,
        )
        await asyncio.sleep(0.5)

        # Verify multiple UPLOAD_STARTED updates (periodic progress)
        update_calls = [call[0] for call in mock_update.call_args_list]
        started_calls = [
            call
            for call in update_calls
            if call[2] == RecordingDataTraceStatus.UPLOAD_STARTED
        ]

        # Should have at least 2 progress updates
        assert len(started_calls) >= 2
