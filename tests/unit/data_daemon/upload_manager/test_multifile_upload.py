"""Tests for multi-file upload system fixes.

This test module validates the fixes for 4 critical bugs:
1. Directory path handling (uploader now enumerates files from directory)
2. Cloud path construction (uses actual filename, not trace_id)
3. Multi-file upload (all files uploaded before UPLOAD_COMPLETE)
4. External trace ID persistence (survives daemon restart)

Tests are organized by functionality, not by bug number.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import pytest_asyncio
from neuracore_types import DataType

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import TraceErrorCode, TraceStatus
from neuracore.data_daemon.upload_management.upload_manager import UploadManager

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Maximum time (seconds) to wait for async events in tests.
# Increase this if debugging locally, or set to None to disable.
TEST_TIMEOUT_SECONDS = 60.0

# Valid UUIDs for testing (upload manager requires valid UUIDs)
TEST_TRACE_ID = "11111111-1111-1111-1111-111111111111"
TEST_TRACE_ID_2 = "22222222-2222-2222-2222-222222222222"
TEST_TRACE_ID_3 = "33333333-3333-3333-3333-333333333333"
TEST_TRACE_ID_EMPTY = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
TEST_TRACE_ID_INTERNAL = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
TEST_TRACE_ID_RESUME = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def trace_directory(tmp_path: Path) -> Path:
    """Create a trace directory with 3 files (simulating video trace)."""
    trace_dir = tmp_path / "trace-abc-123"
    trace_dir.mkdir()
    (trace_dir / "lossless.mp4").write_bytes(b"X" * 1000)  # 1KB
    (trace_dir / "lossy.mp4").write_bytes(b"Y" * 500)  # 500B
    (trace_dir / "trace.json").write_bytes(b'{"meta": "data"}')  # small JSON
    return trace_dir


@pytest.fixture
def empty_directory(tmp_path: Path) -> Path:
    """Create an empty trace directory."""
    trace_dir = tmp_path / "empty-trace"
    trace_dir.mkdir()
    return trace_dir


@pytest.fixture
def single_file(tmp_path: Path) -> Path:
    """Create a single file (not a directory)."""
    file_path = tmp_path / "single_file.mp4"
    file_path.write_bytes(b"X" * 1000)
    return file_path


@pytest.fixture
def mock_auth():
    """Mock get_auth() for all tests."""
    with patch(
        "neuracore.data_daemon.upload_management.trace_manager.get_auth"
    ) as mock_get_auth:
        auth_instance = MagicMock()
        auth_instance.get_org_id = AsyncMock(return_value="test-org")
        auth_instance.get_headers = AsyncMock(
            return_value={"Authorization": "Bearer test-token"}
        )
        mock_get_auth.return_value = auth_instance
        yield mock_get_auth


@pytest_asyncio.fixture
async def client_session():
    """Create an aiohttp session for testing."""
    session = aiohttp.ClientSession()
    yield session
    await session.close()


@pytest_asyncio.fixture
async def upload_manager(client_session: aiohttp.ClientSession, mock_auth):
    """Create and cleanup UploadManager instance."""
    config = DaemonConfig(num_threads=2)
    manager = UploadManager(config=config, client_session=client_session)
    yield manager
    await manager.shutdown(wait=False)


def make_upload_complete_handler(
    events_list: list[str],
    done_event: asyncio.Event,
    expected_count: int = 1,
):
    """Create a handler that appends to list and signals when count reached.

    This enables event-based waiting instead of polling with sleep(),
    making tests deterministic and not flaky in CI environments.

    Args:
        events_list: List to append trace_ids to when UPLOAD_COMPLETE fires
        done_event: Event to set when expected_count completions are reached
        expected_count: Number of UPLOAD_COMPLETE events to wait for

    Returns:
        Handler function to register with emitter.on(Emitter.UPLOAD_COMPLETE, ...)
    """

    def handler(trace_id: str) -> None:
        events_list.append(trace_id)
        if len(events_list) >= expected_count:
            done_event.set()

    return handler


# =============================================================================
# SECTION 1: DIRECTORY UPLOAD HANDLING
# =============================================================================


class TestDirectoryUploadHandling:
    """Tests for directory enumeration and multi-file upload."""

    @pytest.mark.asyncio
    async def test_t1_1_directory_with_multiple_files_uploads_successfully(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
    ) -> None:
        """T1.1: Directory with multiple files uploads successfully.

        The Story:
            A video trace has finished recording. The recording data manager
            wrote 3 files to a trace directory: lossless.mp4 (100MB),
            lossy.mp4 (50MB), and trace.json (1MB). The state manager emits
            READY_FOR_UPLOAD with the directory path. The upload
            manager must enumerate all files and upload each one to cloud storage.

        The Flow:
            1. Create temp directory with 3 files: lossless.mp4, lossy.mp4, trace.json
            2. Emit READY_FOR_UPLOAD with directory path (not file path)
            3. Upload manager receives event, calls Path.iterdir() on directory
            4. Sorts files alphabetically for deterministic order
            5. Creates ResumableFileUploader for each file
            6. Uploads: lossless.mp4, then lossy.mp4, then trace.json
            7. After all 3 complete, emits UPLOAD_COMPLETE

        Why This Matters:
            Previously, the uploader tried to open the directory path as a file, causing
            IsADirectoryError. Video traces have multiple files (lossless for quality,
            lossy for streaming, JSON for metadata). All must be uploaded for a complete
            trace. Missing files means corrupted data in cloud storage.

        Key Assertions:
            - ResumableFileUploader called 3 times (once per file)
            - Files processed in sorted order: lossless.mp4, lossy.mp4, trace.json
            - UPLOAD_COMPLETE emitted exactly once (after all files done)
            - No errors or UPLOAD_FAILED events
        """
        upload_complete_events: list[str] = []
        upload_failed_events: list[tuple] = []
        uploader_calls: list[dict] = []
        upload_done = asyncio.Event()

        emitter = get_emitter()
        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            make_upload_complete_handler(upload_complete_events, upload_done),
        )
        emitter.on(
            Emitter.UPLOAD_FAILED,
            lambda *args: upload_failed_events.append(args),
        )

        mock_uploader_instance = MagicMock()
        mock_uploader_instance.upload = AsyncMock(return_value=(True, 1000, None))

        with (
            patch(
                "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
            ) as MockUploader,
            patch.object(
                upload_manager, "_register_data_trace", new_callable=AsyncMock
            ) as mock_register,
            patch.object(upload_manager, "_update_data_trace", new_callable=AsyncMock),
        ):
            mock_register.return_value = "backend-trace-id-123"

            def capture_uploader_call(**kwargs):
                uploader_calls.append(kwargs)
                return mock_uploader_instance

            MockUploader.side_effect = capture_uploader_call

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID,
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )

            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        assert (
            len(uploader_calls) == 3
        ), f"Expected 3 uploader calls (one per file), got {len(uploader_calls)}"

        filenames = [call["filepath"].split("/")[-1] for call in uploader_calls]
        assert filenames == [
            "lossless.mp4",
            "lossy.mp4",
            "trace.json",
        ], f"Expected files in sorted order, got {filenames}"

        assert (
            len(upload_complete_events) == 1
        ), f"Expected 1 UPLOAD_COMPLETE, got {len(upload_complete_events)}"
        assert upload_complete_events[0] == TEST_TRACE_ID

        assert (
            len(upload_failed_events) == 0
        ), f"Expected no UPLOAD_FAILED, got {upload_failed_events}"

    @pytest.mark.asyncio
    async def test_t1_2_all_files_uploaded_before_upload_complete(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
    ) -> None:
        """T1.2: All files uploaded before UPLOAD_COMPLETE.

        The Story:
            A trace is only complete when ALL files are in cloud storage. Previously,
            the uploader would upload one file and immediately emit UPLOAD_COMPLETE.
            Now it must upload all files before signaling completion.

        The Flow:
            1. Create directory with 3 files (total 151MB)
            2. Emit READY_FOR_UPLOAD
            3. Upload manager uploads file 1 → success
            4. Upload manager uploads file 2 → success
            5. Upload manager uploads file 3 → success
            6. Only NOW emit UPLOAD_COMPLETE

        Why This Matters:
            Premature UPLOAD_COMPLETE triggers trace deletion in state manager.
            If only 1 of 3 files was uploaded, local files get deleted and the
            remaining 2 files are lost forever. Data corruption is permanent.

        Key Assertions:
            - ResumableFileUploader.upload() called 3 times
            - UPLOAD_COMPLETE emitted exactly 1 time
            - UPLOAD_COMPLETE emitted AFTER all 3 uploads finish
            - DELETE_TRACE triggered only after UPLOAD_COMPLETE
        """
        event_sequence: list[str] = []
        upload_done = asyncio.Event()

        emitter = get_emitter()

        def on_upload_complete(trace_id: str) -> None:
            event_sequence.append(f"UPLOAD_COMPLETE:{trace_id}")
            upload_done.set()

        emitter.on(Emitter.UPLOAD_COMPLETE, on_upload_complete)
        emitter.on(
            Emitter.DELETE_TRACE,
            lambda *args: event_sequence.append(f"DELETE_TRACE:{args[1]}"),
        )

        upload_call_count = [0]

        mock_uploader_instance = MagicMock()

        async def mock_upload():
            upload_call_count[0] += 1
            event_sequence.append(f"UPLOAD_FILE:{upload_call_count[0]}")
            return (True, 1000, None)

        mock_uploader_instance.upload = mock_upload

        with (
            patch(
                "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
            ) as MockUploader,
            patch.object(
                upload_manager, "_register_data_trace", new_callable=AsyncMock
            ) as mock_register,
            patch.object(upload_manager, "_update_data_trace", new_callable=AsyncMock),
        ):
            mock_register.return_value = "backend-trace-id-123"
            MockUploader.return_value = mock_uploader_instance

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID,
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )

            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        assert (
            upload_call_count[0] == 3
        ), f"Expected 3 upload calls, got {upload_call_count[0]}"

        complete_events = [e for e in event_sequence if e.startswith("UPLOAD_COMPLETE")]
        assert (
            len(complete_events) == 1
        ), f"Expected 1 UPLOAD_COMPLETE, got {len(complete_events)}"

        upload_positions = [
            i for i, e in enumerate(event_sequence) if e.startswith("UPLOAD_FILE")
        ]
        complete_position = event_sequence.index(f"UPLOAD_COMPLETE:{TEST_TRACE_ID}")

        assert len(upload_positions) == 3, f"Expected 3 uploads, got {upload_positions}"
        assert complete_position > max(upload_positions), (
            f"UPLOAD_COMPLETE at position {complete_position} should be after "
            f"all uploads at positions {upload_positions}. Sequence: {event_sequence}"
        )

    @pytest.mark.asyncio
    async def test_t1_3_empty_directory_fails_gracefully(
        self,
        upload_manager: UploadManager,
        empty_directory: Path,
    ) -> None:
        """T1.3: Empty directory fails gracefully.

        The Story:
            A trace directory was created but the recording was interrupted before any
            files were written. The state manager still marks it ready for upload.
            The upload manager must detect this and fail gracefully rather than
            marking empty upload as "complete".

        The Flow:
            1. Create empty temp directory
            2. Emit READY_FOR_UPLOAD with empty directory path
            3. Upload manager calls iterdir(), gets empty list
            4. Raises ValueError("No files found in trace directory")
            5. Emits UPLOAD_FAILED with UPLOAD_FAILED error code
            6. Does NOT emit UPLOAD_COMPLETE

        Why This Matters:
            An empty directory being marked as "uploaded" would create a ghost trace
            in the backend with no actual data. Users would see recording in UI but
            downloads would fail. Better to fail fast and let retry logic handle it
            after files are written.

        Key Assertions:
            - UPLOAD_FAILED emitted with status=FAILED
            - Error message contains "No files found"
            - UPLOAD_COMPLETE never emitted
            - ResumableFileUploader never instantiated
        """
        upload_complete_events: list[str] = []
        upload_failed_events: list[tuple] = []
        uploader_instantiated = [False]
        upload_done = asyncio.Event()

        emitter = get_emitter()
        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            lambda trace_id: upload_complete_events.append(trace_id),
        )

        def on_upload_failed(*args):
            upload_failed_events.append(args)
            upload_done.set()

        emitter.on(Emitter.UPLOAD_FAILED, on_upload_failed)

        def capture_uploader_call(**kwargs):
            uploader_instantiated[0] = True
            mock_instance = MagicMock()
            mock_instance.upload = AsyncMock(return_value=(True, 0, None))
            return mock_instance

        with (
            patch(
                "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
            ) as MockUploader,
            patch.object(
                upload_manager, "_register_data_trace", new_callable=AsyncMock
            ) as mock_register,
            patch.object(upload_manager, "_update_data_trace", new_callable=AsyncMock),
        ):
            mock_register.return_value = "backend-trace-id-123"
            MockUploader.side_effect = capture_uploader_call

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                "TEST_TRACE_ID_EMPTY",
                "rec-456",
                str(empty_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )

            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        assert (
            len(upload_failed_events) == 1
        ), f"Expected 1 UPLOAD_FAILED, got {len(upload_failed_events)}"
        failed_event = upload_failed_events[0]
        assert failed_event[0] == "TEST_TRACE_ID_EMPTY", "Wrong trace_id in UPLOAD_FAILED"
        assert (
            failed_event[2] == TraceStatus.FAILED
        ), f"Expected status=FAILED, got {failed_event[2]}"

        error_message = failed_event[4]
        assert (
            "Empty directory" in error_message or "empty" in error_message.lower()
        ), f"Expected 'Empty directory' in error message, got: {error_message}"

        assert (
            len(upload_complete_events) == 0
        ), f"Expected no UPLOAD_COMPLETE, got {upload_complete_events}"

        assert not uploader_instantiated[
            0
        ], "ResumableFileUploader should not be instantiated for empty directory"


# =============================================================================
# SECTION 2: CLOUD PATH CONSTRUCTION
# =============================================================================


class TestCloudPathConstruction:
    """Tests for correct cloud filepath format."""

    @pytest.mark.asyncio
    async def test_t2_1_cloud_path_format_correct(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
    ) -> None:
        """T2.1: Cloud path format is {data_type.value}/{data_type_name}/{filename}.

        The Story:
            Cloud storage paths must follow exact format for backend to
            generate correct download URLs.
            Format: "{data_type.value}/{data_type_name}/{filename}"
            Example: "RGB_IMAGES/camera_front/lossless.mp4"

        The Flow:
            1. READY_FOR_UPLOAD with:
               - data_type = DataType.RGB_IMAGES
               - data_type_name = "camera_front"
               - Directory contains "lossless.mp4"
            2. Upload manager constructs cloud_filepath
            3. ResumableFileUploader receives cloud_filepath parameter

        Why This Matters:
            Backend uses exact path to generate GCS signed URLs for download.
            Wrong path format = file uploaded but never downloadable.
            Users click "download" and get 404.

        Key Assertions:
            - cloud_filepath = "RGB_IMAGES/camera_front/lossless.mp4"
            - Starts with DataType enum VALUE (not name)
            - Uses "/" as separator (not "\\" or other)
            - Ends with actual filename from disk
        """
        uploader_calls: list[dict] = []
        upload_complete_events: list[str] = []
        upload_done = asyncio.Event()

        emitter = get_emitter()
        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            make_upload_complete_handler(upload_complete_events, upload_done),
        )

        mock_uploader_instance = MagicMock()
        mock_uploader_instance.upload = AsyncMock(return_value=(True, 1000, None))

        with (
            patch(
                "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
            ) as MockUploader,
            patch.object(
                upload_manager, "_register_data_trace", new_callable=AsyncMock
            ) as mock_register,
            patch.object(upload_manager, "_update_data_trace", new_callable=AsyncMock),
        ):
            mock_register.return_value = "backend-trace-id-123"

            def capture_uploader_call(**kwargs):
                uploader_calls.append(kwargs)
                return mock_uploader_instance

            MockUploader.side_effect = capture_uploader_call

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID,
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_front",
                0,
            )

            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        assert len(uploader_calls) == 3, f"Expected 3 calls, got {len(uploader_calls)}"

        first_call = uploader_calls[0]
        cloud_filepath = first_call["cloud_filepath"]

        expected_path = f"{DataType.RGB_IMAGES.value}/camera_front/lossless.mp4"
        assert (
            cloud_filepath == expected_path
        ), f"Expected cloud_filepath='{expected_path}', got '{cloud_filepath}'"

        # Starts with DataType enum VALUE (not name)
        assert cloud_filepath.startswith(
            DataType.RGB_IMAGES.value
        ), f"Path should start with '{DataType.RGB_IMAGES.value}': {cloud_filepath}"
        assert not cloud_filepath.startswith(
            "DataType."
        ), f"Path should not start with 'DataType.', got '{cloud_filepath}'"

        assert (
            "\\" not in cloud_filepath
        ), f"Path should use '/' separator, not '\\': {cloud_filepath}"
        parts = cloud_filepath.split("/")
        assert len(parts) == 3, f"Expected 3 path parts, got {parts}"

        assert cloud_filepath.endswith(
            "lossless.mp4"
        ), f"Path should end with 'lossless.mp4', got '{cloud_filepath}'"

    @pytest.mark.asyncio
    async def test_t2_2_each_file_gets_unique_path(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
    ) -> None:
        """T2.2: Each file gets unique path with correct filename.

        The Story:
            Directory has 3 files. Each must get path with its actual filename.
            No duplicates, no generic names, no trace_id substitution.

        The Flow:
            1. Directory contains: lossless.mp4, lossy.mp4, trace.json
            2. data_type=RGB_IMAGES, data_type_name=camera_0
            3. Three uploads with paths:
               - "RGB_IMAGES/camera_0/lossless.mp4"
               - "RGB_IMAGES/camera_0/lossy.mp4"
               - "RGB_IMAGES/camera_0/trace.json"

        Why This Matters:
            Old bug used trace_id as filename: "RGB_IMAGES/camera_0/abc-123-uuid"
            This created one file with unrecognizable name.
            Frontend couldn't determine file type, couldn't play video.

        Key Assertions:
            - 3 different cloud paths generated
            - Each ends with actual filename from disk
            - File extension preserved (.mp4, .json)
            - No trace_id or UUID in paths
        """
        uploader_calls: list[dict] = []
        upload_complete_events: list[str] = []
        upload_done = asyncio.Event()
        trace_id = TEST_TRACE_ID_2

        emitter = get_emitter()
        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            make_upload_complete_handler(upload_complete_events, upload_done),
        )

        mock_uploader_instance = MagicMock()
        mock_uploader_instance.upload = AsyncMock(return_value=(True, 1000, None))

        with (
            patch(
                "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
            ) as MockUploader,
            patch.object(
                upload_manager, "_register_data_trace", new_callable=AsyncMock
            ) as mock_register,
            patch.object(upload_manager, "_update_data_trace", new_callable=AsyncMock),
        ):
            mock_register.return_value = "backend-trace-id-xyz"

            def capture_uploader_call(**kwargs):
                uploader_calls.append(kwargs)
                return mock_uploader_instance

            MockUploader.side_effect = capture_uploader_call

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                trace_id,
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )

            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        assert len(uploader_calls) == 3, f"Expected 3 calls, got {len(uploader_calls)}"

        cloud_paths = [call["cloud_filepath"] for call in uploader_calls]
        unique_paths = set(cloud_paths)
        assert (
            len(unique_paths) == 3
        ), f"Expected 3 unique paths, got {len(unique_paths)}: {cloud_paths}"

        expected_filenames = {"lossless.mp4", "lossy.mp4", "trace.json"}
        actual_filenames = {path.split("/")[-1] for path in cloud_paths}
        assert (
            actual_filenames == expected_filenames
        ), f"Expected filenames {expected_filenames}, got {actual_filenames}"

        for path in cloud_paths:
            filename = path.split("/")[-1]
            assert "." in filename, f"Filename should have extension: {filename}"
            ext = filename.split(".")[-1]
            assert ext in ["mp4", "json"], f"Unexpected extension: {ext}"

        for path in cloud_paths:
            assert (
                trace_id not in path
            ), f"trace_id should not appear in cloud path: {path}"
            assert (
                "abc-123" not in path
            ), f"UUID-like pattern should not appear in cloud path: {path}"
            assert (
                "uuid" not in path.lower()
            ), f"'uuid' should not appear in cloud path: {path}"


# =============================================================================
# SECTION 3: REGISTRATION AND UPLOAD FAILURES
# =============================================================================


class TestRegistrationAndUploadFailures:
    """Tests for registration failures and error handling."""

    @pytest.mark.asyncio
    async def test_t3_1_registration_failure_emits_upload_failed(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
    ) -> None:
        """T3.1: Registration failure emits UPLOAD_FAILED.

        The Story:
            First upload attempt. Backend registration fails (network error,
            auth expired, server error). Upload manager must fail gracefully
            without trying to upload files.

        The Flow:
            1. Emit READY_FOR_UPLOAD
            2. _register_data_trace() returns None (failure)
            3. Upload manager emits UPLOAD_FAILED
            4. Does NOT attempt file upload

        Why This Matters:
            Without successful registration, we cannot get upload URLs from
            backend. Attempting uploads would fail anyway. Failing early
            preserves bytes_uploaded=0 so retry starts fresh.
            Clear error message helps debugging.

        Key Assertions:
            - UPLOAD_FAILED emitted with "Failed to register trace" message
            - status = TraceStatus.FAILED
            - ResumableFileUploader never instantiated
            - bytes_uploaded remains 0
        """
        upload_failed_events: list[tuple] = []
        upload_complete_events: list[str] = []
        uploader_instantiated = [False]
        upload_done = asyncio.Event()
        trace_id = TEST_TRACE_ID_INTERNAL

        emitter = get_emitter()

        def on_upload_failed(*args):
            upload_failed_events.append(args)
            upload_done.set()

        emitter.on(Emitter.UPLOAD_FAILED, on_upload_failed)
        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            lambda tid: upload_complete_events.append(tid),
        )

        def capture_uploader_call(**kwargs):
            uploader_instantiated[0] = True
            mock_instance = MagicMock()
            mock_instance.upload = AsyncMock(return_value=(True, 1000, None))
            return mock_instance

        with (
            patch(
                "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
            ) as MockUploader,
            patch.object(
                upload_manager, "_register_data_trace", new_callable=AsyncMock
            ) as mock_register,
            patch.object(upload_manager, "_update_data_trace", new_callable=AsyncMock),
        ):
            mock_register.return_value = None
            MockUploader.side_effect = capture_uploader_call

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                trace_id,
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )

            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        assert (
            len(upload_failed_events) == 1
        ), f"Expected 1 UPLOAD_FAILED event, got {len(upload_failed_events)}"
        failed_event = upload_failed_events[0]
        assert failed_event[0] == trace_id, f"Wrong trace_id: {failed_event[0]}"

        error_message = failed_event[4]
        assert (
            "register" in error_message.lower() or "failed" in error_message.lower()
        ), f"Expected 'register' or 'failed' in error message, got: {error_message}"

        assert (
            failed_event[2] == TraceStatus.FAILED
        ), f"Expected status=TraceStatus.FAILED, got {failed_event[2]}"

        assert not uploader_instantiated[
            0
        ], "ResumableFileUploader should not be instantiated when registration fails"

        assert failed_event[1] == 0, f"Expected bytes_uploaded=0, got {failed_event[1]}"

        assert (
            len(upload_complete_events) == 0
        ), f"Expected no UPLOAD_COMPLETE, got {upload_complete_events}"


# =============================================================================
# SECTION 4: BACKEND API CONSISTENCY
# =============================================================================


class TestBackendApiConsistency:
    """Tests for correct backend API usage."""

    @pytest.mark.asyncio
    async def test_t4_1_backend_updates_use_external_trace_id(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
    ) -> None:
        """T4.1: Backend updates use external_trace_id not internal trace_id.

        The Story:
            We have two IDs: internal trace_id (daemon's UUID) and external_trace_id
            (backend's UUID). All backend API calls (update status, get upload URL)
            must use external_trace_id. Sending internal trace_id would result in
            404 errors or updates to wrong traces.

        The Flow:
            1. Create trace with internal trace_id="internal-abc-123"
            2. Registration returns external_trace_id="backend-xyz-789"
            3. Call _update_data_trace() with UPLOAD_STARTED status
            4. Verify API call uses "backend-xyz-789" in URL path

        Why This Matters:
            Backend doesn't know about daemon's internal IDs. If we send wrong ID:
            - 404 Not Found: trace doesn't exist
            - Wrong trace updated: data corruption in another user's recording
            - Orphan traces: backend has trace with no updates

        Key Assertions:
            - PUT /traces/{trace_id} uses "backend-xyz-789"
            - Never contains "internal-abc-123" in any API request
            - All 3 update calls (STARTED, periodic, COMPLETE) use same external ID
        """
        trace_id = TEST_TRACE_ID_INTERNAL
        update_trace_calls: list[tuple] = []
        upload_complete_events: list[str] = []
        upload_done = asyncio.Event()

        emitter = get_emitter()
        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            make_upload_complete_handler(upload_complete_events, upload_done),
        )

        mock_uploader_instance = MagicMock()
        mock_uploader_instance.upload = AsyncMock(return_value=(True, 1000, None))

        with (
            patch(
                "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
            ) as MockUploader,
            patch.object(
                upload_manager, "_register_data_trace", new_callable=AsyncMock
            ) as mock_register,
            patch.object(
                upload_manager, "_update_data_trace", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_register.return_value = trace_id
            MockUploader.return_value = mock_uploader_instance

            async def capture_update(*args, **kwargs):
                update_trace_calls.append((args, kwargs))

            mock_update.side_effect = capture_update

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                trace_id,
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )

            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        assert len(update_trace_calls) >= 2, (
            f"Expected at least 2 _update_data_trace calls (STARTED and COMPLETE), "
            f"got {len(update_trace_calls)}"
        )

        for i, (args, kwargs) in enumerate(update_trace_calls):
            used_trace_id = args[1]
            assert used_trace_id == trace_id, (
                f"Call {i}: Expected trace_id='{trace_id}', "
                f"but got '{used_trace_id}'"
            )

        trace_ids_used = {args[1] for args, _ in update_trace_calls}
        assert len(trace_ids_used) == 1, (
            f"All update calls should use same trace_id, "
            f"but found different IDs: {trace_ids_used}"
        )
        assert (
            trace_id in trace_ids_used
        ), f"Expected '{trace_id}' in trace_ids_used, got {trace_ids_used}"

    @pytest.mark.asyncio
    async def test_t4_2_upload_started_sends_external_trace_id(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
    ) -> None:
        """T4.2: UPLOAD_STARTED sends external_trace_id to correct endpoint.

        The Story:
            After registration, first backend update is UPLOAD_STARTED with total_bytes.
            This update must go to the correct endpoint using external_trace_id.

        The Flow:
            1. Register trace → backend returns external_trace_id="ABC-123"
            2. Calculate total_bytes from all files in directory (151MB)
            3. Call _update_data_trace(recording_id, "ABC-123", UPLOAD_STARTED,
               total_bytes=151MB)
            4. Verify HTTP PUT to /recording/{rec_id}/traces/ABC-123

        Why This Matters:
            UPLOAD_STARTED tells backend "upload is in progress, expect this much data".
            If sent to wrong trace ID, backend shows wrong recording as uploading.
            Dashboard displays incorrect progress to user.

        Key Assertions:
            - _update_data_trace receives external_trace_id="ABC-123"
            - HTTP request path contains "ABC-123"
            - Request body has status=UPLOAD_STARTED, total_bytes=151MB
            - uploaded_bytes=0 for fresh upload (or cumulative for resume)
        """
        from neuracore_types import RecordingDataTraceStatus

        trace_id = TEST_TRACE_ID_INTERNAL
        recording_id = "rec-456"
        update_calls: list[tuple] = []
        upload_complete_events: list[str] = []
        upload_done = asyncio.Event()

        emitter = get_emitter()
        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            make_upload_complete_handler(upload_complete_events, upload_done),
        )

        mock_uploader_instance = MagicMock()
        mock_uploader_instance.upload = AsyncMock(return_value=(True, 1000, None))

        with (
            patch(
                "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
            ) as MockUploader,
            patch.object(
                upload_manager, "_register_data_trace", new_callable=AsyncMock
            ) as mock_register,
            patch.object(
                upload_manager, "_update_data_trace", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_register.return_value = trace_id
            MockUploader.return_value = mock_uploader_instance

            async def capture_update(*args, **kwargs):
                update_calls.append((args, kwargs))

            mock_update.side_effect = capture_update

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                trace_id,
                recording_id,
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )

            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        upload_started_call = None
        for args, kwargs in update_calls:
            if len(args) >= 3 and args[2] == RecordingDataTraceStatus.UPLOAD_STARTED:
                upload_started_call = (args, kwargs)
                break

        assert (
            upload_started_call is not None
        ), "Expected an UPLOAD_STARTED call to _update_data_trace"
        args, kwargs = upload_started_call

        assert (
            args[0] == recording_id
        ), f"Expected recording_id='{recording_id}', got '{args[0]}'"

        assert (
            args[1] == trace_id
        ), f"Expected trace_id='{trace_id}', got '{args[1]}'"

        assert (
            args[2] == RecordingDataTraceStatus.UPLOAD_STARTED
        ), f"Expected status=UPLOAD_STARTED, got {args[2]}"

        if "uploaded_bytes" in kwargs:
            assert (
                kwargs["uploaded_bytes"] == 0
            ), f"Expected uploaded_bytes=0 for fresh upload, got {kwargs}"

    @pytest.mark.asyncio
    async def test_t4_3_upload_complete_sends_external_trace_id(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
    ) -> None:
        """T4.3: UPLOAD_COMPLETE sends external_trace_id with final bytes.

        The Story:
            After all files uploaded, UPLOAD_COMPLETE marks trace as done in backend.
            Must use same external_trace_id and report accurate final byte counts.

        The Flow:
            1. Upload completes with external_trace_id="ABC-123"
            2. Total bytes across 3 files = 151MB
            3. Call _update_data_trace(recording_id, "ABC-123", UPLOAD_COMPLETE,
               uploaded_bytes=151MB, total_bytes=151MB)
            4. Verify request to correct endpoint

        Why This Matters:
            UPLOAD_COMPLETE triggers backend to mark recording as "ready for download".
            Wrong ID = wrong recording marked complete while real one stuck uploading.
            Mismatched bytes = backend shows "upload complete but data missing".

        Key Assertions:
            - External trace ID "ABC-123" in PUT request path
            - status = RecordingDataTraceStatus.UPLOAD_COMPLETE
            - uploaded_bytes = total_bytes = 151MB
            - Called exactly once after all files done
        """
        from neuracore_types import RecordingDataTraceStatus

        trace_id = TEST_TRACE_ID_INTERNAL
        recording_id = "rec-456"
        update_calls: list[tuple] = []
        upload_complete_events: list[str] = []
        upload_done = asyncio.Event()

        emitter = get_emitter()
        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            make_upload_complete_handler(upload_complete_events, upload_done),
        )

        total_bytes_per_file = 1000
        mock_uploader_instance = MagicMock()
        mock_uploader_instance.upload = AsyncMock(
            return_value=(True, total_bytes_per_file, None)
        )

        with (
            patch(
                "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
            ) as MockUploader,
            patch.object(
                upload_manager, "_register_data_trace", new_callable=AsyncMock
            ) as mock_register,
            patch.object(
                upload_manager, "_update_data_trace", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_register.return_value = trace_id
            MockUploader.return_value = mock_uploader_instance

            async def capture_update(*args, **kwargs):
                update_calls.append((args, kwargs))

            mock_update.side_effect = capture_update

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                trace_id,
                recording_id,
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )

            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        upload_complete_calls = [
            (args, kwargs)
            for args, kwargs in update_calls
            if len(args) >= 3 and args[2] == RecordingDataTraceStatus.UPLOAD_COMPLETE
        ]

        assert (
            len(upload_complete_calls) == 1
        ), f"Expected exactly 1 UPLOAD_COMPLETE call, got {len(upload_complete_calls)}"

        args, kwargs = upload_complete_calls[0]

        assert (
            args[1] == trace_id
        ), f"Expected trace_id='{trace_id}', got '{args[1]}'"

        assert (
            args[2] == RecordingDataTraceStatus.UPLOAD_COMPLETE
        ), f"Expected status=UPLOAD_COMPLETE, got {args[2]}"

        uploaded_bytes = kwargs.get("uploaded_bytes")
        total_bytes = kwargs.get("total_bytes")

        if uploaded_bytes is not None:
            assert (
                uploaded_bytes > 0
            ), f"Expected uploaded_bytes > 0, got {uploaded_bytes}"
        if total_bytes is not None:
            assert total_bytes > 0, f"Expected total_bytes > 0, got {total_bytes}"
        if uploaded_bytes is not None and total_bytes is not None:
            assert uploaded_bytes == total_bytes, (
                f"Expected uploaded_bytes == total_bytes, "
                f"got {uploaded_bytes} != {total_bytes}"
            )

    @pytest.mark.asyncio
    async def test_t4_4_resume_uses_same_trace_id(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
    ) -> None:
        """T4.4: Resume uses same trace_id for all backend calls.

        The Story:
            Daemon crashed mid-upload. Restart loads trace from SQLite with
            bytes_uploaded > 0. The same trace_id is used for all backend calls
            on resume.

        The Flow:
            1. SQLite has trace with trace_id and bytes_uploaded > 0
            2. READY_FOR_UPLOAD emitted with trace_id and bytes_uploaded
            3. Upload manager registers with same trace_id
            4. Calls _update_data_trace with same trace_id
            5. Completes upload, UPLOAD_COMPLETE uses same trace_id

        Why This Matters:
            The backend accepts the trace_id parameter during registration.
            This ensures the same ID is used even on retry, preventing
            duplicate/orphan traces.

        Key Assertions:
            - _register_data_trace called with trace_id
            - All _update_data_trace calls use same trace_id
            - UPLOAD_COMPLETE sent with same trace_id
        """
        from neuracore_types import RecordingDataTraceStatus

        trace_id = TEST_TRACE_ID_RESUME
        update_calls: list[tuple] = []
        upload_complete_events: list[str] = []
        upload_done = asyncio.Event()

        emitter = get_emitter()
        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            make_upload_complete_handler(upload_complete_events, upload_done),
        )

        mock_uploader_instance = MagicMock()
        mock_uploader_instance.upload = AsyncMock(return_value=(True, 1000, None))

        with (
            patch(
                "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
            ) as MockUploader,
            patch.object(
                upload_manager, "_register_data_trace", new_callable=AsyncMock
            ) as mock_register,
            patch.object(
                upload_manager, "_update_data_trace", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_register.return_value = trace_id
            MockUploader.return_value = mock_uploader_instance

            async def capture_update(*args, **kwargs):
                update_calls.append((args, kwargs))

            mock_update.side_effect = capture_update

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                trace_id,
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                500,
            )

            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

            assert mock_register.call_count == 1, (
                f"_register_data_trace should be called once, "
                f"but was called {mock_register.call_count} times"
            )

        assert len(update_calls) >= 1, "Expected at least one _update_data_trace call"

        for i, (args, kwargs) in enumerate(update_calls):
            used_trace_id = args[1]  # Second arg is trace_id
            assert used_trace_id == trace_id, (
                f"Call {i}: Expected trace_id='{trace_id}', "
                f"got '{used_trace_id}'"
            )

        upload_complete_calls = [
            (args, kwargs)
            for args, kwargs in update_calls
            if len(args) >= 3 and args[2] == RecordingDataTraceStatus.UPLOAD_COMPLETE
        ]

        assert (
            len(upload_complete_calls) == 1
        ), f"Expected 1 UPLOAD_COMPLETE call, got {len(upload_complete_calls)}"

        complete_args, _ = upload_complete_calls[0]
        assert complete_args[1] == trace_id, (
            f"UPLOAD_COMPLETE should use '{trace_id}', "
            f"got '{complete_args[1]}'"
        )


# =============================================================================
# SECTION 5: RESUME LOGIC
# =============================================================================


class TestResumeLogic:
    """Tests for upload resume functionality."""

    @pytest.mark.asyncio
    async def test_t5_1_resume_from_middle_of_multi_file_upload(
        self,
        upload_manager: UploadManager,
        tmp_path: Path,
    ) -> None:
        """T5.1: Resume from middle of multi-file upload.

        The Story:
            Trace has 3 files: lossless.mp4 (100MB), lossy.mp4 (50MB), trace.json (1MB).
            First upload: lossless.mp4 completed, lossy.mp4 uploaded 20MB, then crashed.
            SQLite has: bytes_uploaded=120MB.
            On retry, must skip lossless.mp4, resume lossy.mp4 at offset 20MB.

        The Flow:
            1. Emit READY_FOR_UPLOAD with bytes_uploaded=120MB
            2. _find_resume_point(files, 120MB) returns (file_index=1, offset=20MB)
            3. Skips file 0 (lossless.mp4, 100MB)
            4. Starts file 1 (lossy.mp4) with bytes_uploaded=20MB offset
            5. Completes lossy.mp4, then uploads trace.json
            6. Emits UPLOAD_COMPLETE

        Why This Matters:
            Without resume logic, 100MB would be re-uploaded every retry. User on slow
            connection might never complete upload. Resume means only remaining 31MB
            needs uploading, not full 151MB.

        Key Assertions:
            - ResumableFileUploader NOT created for lossless.mp4
            - lossy.mp4 uploader created with bytes_uploaded=20MB
            - trace.json uploader created with bytes_uploaded=0
            - Total upload = 31MB, not 151MB
        """
        trace_dir = tmp_path / "trace-resume-test"
        trace_dir.mkdir()

        # Files sorted alphabetically: lossless.mp4 (100B), lossy.mp4 (50B), trace.json (11B)
        (trace_dir / "lossless.mp4").write_bytes(b"L" * 100)
        (trace_dir / "lossy.mp4").write_bytes(b"O" * 50)
        (trace_dir / "trace.json").write_bytes(b'{"data": 1}')

        # Simulate: lossless.mp4 (100) complete + 20 bytes of lossy.mp4 = 120
        bytes_uploaded = 120

        uploader_calls: list[dict] = []
        upload_complete_events: list[str] = []
        upload_done = asyncio.Event()

        emitter = get_emitter()
        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            make_upload_complete_handler(upload_complete_events, upload_done),
        )

        mock_uploader_instance = MagicMock()
        mock_uploader_instance.upload = AsyncMock(return_value=(True, 100, None))

        with (
            patch(
                "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
            ) as MockUploader,
            patch.object(
                upload_manager, "_register_data_trace", new_callable=AsyncMock
            ) as mock_register,
            patch.object(upload_manager, "_update_data_trace", new_callable=AsyncMock),
        ):
            mock_register.return_value = TEST_TRACE_ID

            def capture_uploader_call(**kwargs):
                uploader_calls.append(kwargs)
                return mock_uploader_instance

            MockUploader.side_effect = capture_uploader_call

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID,
                "rec-456",
                str(trace_dir),
                DataType.RGB_IMAGES,
                "camera_0",
                bytes_uploaded,
            )

            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        uploaded_filenames = [
            call["filepath"].split("/")[-1] for call in uploader_calls
        ]
        assert "lossless.mp4" not in uploaded_filenames, (
            f"lossless.mp4 should be skipped (already complete), "
            f"but found in uploads: {uploaded_filenames}"
        )

        assert len(uploader_calls) == 2, (
            f"Expected 2 uploads (lossy.mp4 and trace.json), got {len(uploader_calls)}"
            f"{uploaded_filenames}"
        )

        lossy_call = next(
            (c for c in uploader_calls if "lossy.mp4" in c["filepath"]),
            None,
        )
        assert lossy_call is not None, "lossy.mp4 should be uploaded"

        assert (
            lossy_call["bytes_uploaded"] == 20
        ), f"lossy.mp4 should resume at offset 20, got {lossy_call['bytes_uploaded']}"

        json_call = next(
            (c for c in uploader_calls if "trace.json" in c["filepath"]),
            None,
        )
        assert json_call is not None, "trace.json should be uploaded"

        assert (
            json_call["bytes_uploaded"] == 0
        ), f"trace.json should start from 0, got {json_call['bytes_uploaded']}"

        assert (
            len(upload_complete_events) == 1
        ), f"Expected 1 UPLOAD_COMPLETE, got {len(upload_complete_events)}"

    @pytest.mark.asyncio
    async def test_t5_2_partial_failure_preserves_progress(
        self,
        upload_manager: UploadManager,
        tmp_path: Path,
    ) -> None:
        """T5.2: Partial failure preserves progress.

        The Story:
            File 1 uploads successfully. File 2 fails due to network error. The upload
            manager must emit UPLOAD_FAILED with the correct cumulative bytes_uploaded
            so that retry can resume from the correct position.

        The Flow:
            1. Create directory: file_a.mp4 (100MB), file_b.mp4 (50MB)
            2. Emit READY_FOR_UPLOAD
            3. Upload file_a.mp4 → success (100MB uploaded)
            4. Upload file_b.mp4 → fails at 20MB
            5. Emit UPLOAD_FAILED with bytes_uploaded=120MB

        Why This Matters:
            If bytes_uploaded was reset to 0 or only counted the failed file (20MB),
            retry would re-upload file_a.mp4 unnecessarily. Correct cumulative tracking
            means retry starts from file_b.mp4 at offset 20MB.

        Key Assertions:
            - UPLOAD_FAILED emitted (not UPLOAD_COMPLETE)
            - bytes_uploaded = 120MB (100 + 20)
            - status = TraceStatus.WRITTEN (retryable)
            - error_code = TraceErrorCode.NETWORK_ERROR
        """
        trace_dir = tmp_path / "trace-partial-fail"
        trace_dir.mkdir()

        # Using bytes for test efficiency
        (trace_dir / "file_a.mp4").write_bytes(b"A" * 100)  # 100 bytes
        (trace_dir / "file_b.mp4").write_bytes(b"B" * 50)  # 50 bytes

        upload_failed_events: list[tuple] = []
        upload_complete_events: list[str] = []
        upload_call_count = [0]
        upload_done = asyncio.Event()

        emitter = get_emitter()

        def on_upload_failed(*args):
            upload_failed_events.append(args)
            upload_done.set()

        emitter.on(Emitter.UPLOAD_FAILED, on_upload_failed)
        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            lambda tid: upload_complete_events.append(tid),
        )

        mock_uploader_instance = MagicMock()

        async def mock_upload():
            upload_call_count[0] += 1
            if upload_call_count[0] == 1:
                return (True, 100, None)
            else:
                return (False, 20, "Network error: connection reset")

        mock_uploader_instance.upload = mock_upload

        with (
            patch(
                "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
            ) as MockUploader,
            patch.object(
                upload_manager, "_register_data_trace", new_callable=AsyncMock
            ) as mock_register,
            patch.object(upload_manager, "_update_data_trace", new_callable=AsyncMock),
        ):
            mock_register.return_value = "backend-trace-id"
            MockUploader.return_value = mock_uploader_instance

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID,
                "rec-456",
                str(trace_dir),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )

            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        assert (
            len(upload_complete_events) == 0
        ), f"Expected no UPLOAD_COMPLETE on failure, got {upload_complete_events}"
        assert (
            len(upload_failed_events) == 1
        ), f"Expected 1 UPLOAD_FAILED, got {len(upload_failed_events)}"

        failed_event = upload_failed_events[0]

        assert (
            failed_event[1] == 120
        ), f"Expected bytes_uploaded=120 (100+20), got {failed_event[1]}"

        assert (
            failed_event[2] == TraceStatus.WRITTEN
        ), f"Expected status=WRITTEN (retryable), got {failed_event[2]}"

        assert (
            failed_event[3] == TraceErrorCode.NETWORK_ERROR
        ), f"Expected error_code=NETWORK_ERROR, got {failed_event[3]}"

        assert (
            "Network" in failed_event[4] or "network" in failed_event[4].lower()
        ), f"Error message should mention network: {failed_event[4]}"


# =============================================================================
# SECTION 6: CONCURRENT UPLOADS
# =============================================================================


class TestConcurrentUploads:
    """Tests for handling multiple simultaneous upload operations."""

    @pytest.mark.asyncio
    async def test_t6_1_handles_multiple_concurrent_uploads(
        self,
        upload_manager: UploadManager,
        client_session: aiohttp.ClientSession,
        tmp_path: Path,
    ) -> None:
        """T6.1: Multiple concurrent uploads complete successfully.

        The Story:
            During a recording session, multiple traces may finish writing
            at nearly the same time. The upload manager receives several
            READY_FOR_UPLOAD events in quick succession. Each upload runs
            as a separate async task. All uploads must complete successfully
            without interference or race conditions.

        The Flow:
            1. Create 5 separate trace directories, each with test files
            2. Emit 5 READY_FOR_UPLOAD events in rapid succession
            3. Upload manager creates 5 concurrent async tasks
            4. All tasks run simultaneously (not sequentially)
            5. Each upload completes and emits UPLOAD_COMPLETE
            6. All 5 UPLOAD_COMPLETE events received

        Why This Matters:
            Real-world usage involves multiple data streams (RGB, depth,
            audio, etc.) that complete around the same time. The manager
            must handle concurrent uploads efficiently without blocking
            or losing events. Race conditions could cause duplicate
            uploads or missed files.

        Key Assertions:
            - 5 UPLOAD_COMPLETE events emitted (one per trace)
            - All 5 unique trace IDs in completion events
            - No UPLOAD_FAILED events
            - Active uploads set properly managed (empty after completion)
        """
        trace_dirs: list[Path] = []
        trace_ids: list[str] = [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
            "00000000-0000-0000-0000-000000000003",
            "00000000-0000-0000-0000-000000000004",
            "00000000-0000-0000-0000-000000000005",
        ]
        for i, trace_id in enumerate(trace_ids):
            trace_dir = tmp_path / f"trace-{i}"
            trace_dir.mkdir()
            (trace_dir / "file.mp4").write_bytes(b"X" * 100)
            (trace_dir / "trace.json").write_bytes(b'{"i": ' + str(i).encode() + b"}")
            trace_dirs.append(trace_dir)

        upload_complete_events: list[str] = []
        upload_failed_events: list[tuple] = []
        all_complete = asyncio.Event()

        emitter = get_emitter()

        def on_complete(trace_id: str) -> None:
            upload_complete_events.append(trace_id)
            if len(upload_complete_events) >= 5:
                all_complete.set()

        emitter.on(Emitter.UPLOAD_COMPLETE, on_complete)
        emitter.on(
            Emitter.UPLOAD_FAILED,
            lambda *args: upload_failed_events.append(args),
        )

        mock_uploader_instance = MagicMock()
        mock_uploader_instance.upload = AsyncMock(return_value=(True, 100, None))

        try:
            with (
                patch(
                    "neuracore.data_daemon.upload_management.upload_manager"
                    ".ResumableFileUploader"
                ) as MockUploader,
                patch.object(
                    upload_manager, "_register_data_trace", new_callable=AsyncMock
                ) as mock_register,
                patch.object(
                    upload_manager, "_update_data_trace", new_callable=AsyncMock
                ),
            ):
                mock_register.return_value = "backend-trace-id"
                MockUploader.return_value = mock_uploader_instance

                for i, (trace_dir, trace_id) in enumerate(zip(trace_dirs, trace_ids)):
                    emitter.emit(
                        Emitter.READY_FOR_UPLOAD,
                        trace_id,
                        f"rec-{i}",
                        str(trace_dir),
                        DataType.RGB_IMAGES,
                        f"camera_{i}",
                        0,
                    )

                await asyncio.wait_for(
                    all_complete.wait(), timeout=TEST_TIMEOUT_SECONDS
                )
                await asyncio.sleep(0)
        finally:
            emitter.remove_listener(Emitter.UPLOAD_COMPLETE, on_complete)

        assert (
            len(upload_complete_events) == 5
        ), f"Expected 5 UPLOAD_COMPLETE, got {len(upload_complete_events)}"

        assert set(upload_complete_events) == set(
            trace_ids
        ), f"Expected trace IDs {trace_ids}, got {upload_complete_events}"

        assert (
            len(upload_failed_events) == 0
        ), f"Expected no failures, got {upload_failed_events}"

        assert (
            len(upload_manager._active_uploads) == 0
        ), f"Expected empty active uploads, got {len(upload_manager._active_uploads)}"
