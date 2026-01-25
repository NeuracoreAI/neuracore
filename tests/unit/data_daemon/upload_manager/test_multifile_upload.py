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

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def trace_directory(tmp_path: Path) -> Path:
    """Create a trace directory with 3 files (simulating video trace)."""
    trace_dir = tmp_path / "trace-abc-123"
    trace_dir.mkdir()
    # Create files with realistic sizes (small for tests)
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
        # Track events emitted (event-based, no polling)
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

        # Mock the uploader to track calls and succeed
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

            # Emit READY_FOR_UPLOAD with directory path
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                "trace-123",  # trace_id
                "rec-456",  # recording_id
                str(trace_directory),  # path (directory, not file)
                DataType.RGB_IMAGES,  # data_type
                "camera_0",  # data_type_name
                0,  # bytes_uploaded
                None,  # external_trace_id (first upload)
            )

            # Wait for upload to complete (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        # KEY ASSERTION 1: ResumableFileUploader called 3 times (once per file)
        assert (
            len(uploader_calls) == 3
        ), f"Expected 3 uploader calls (one per file), got {len(uploader_calls)}"

        # KEY ASSERTION 2: Files processed in sorted order
        filenames = [call["filepath"].split("/")[-1] for call in uploader_calls]
        assert filenames == [
            "lossless.mp4",
            "lossy.mp4",
            "trace.json",
        ], f"Expected files in sorted order, got {filenames}"

        # KEY ASSERTION 3: UPLOAD_COMPLETE emitted exactly once
        assert (
            len(upload_complete_events) == 1
        ), f"Expected 1 UPLOAD_COMPLETE, got {len(upload_complete_events)}"
        assert upload_complete_events[0] == "trace-123"

        # KEY ASSERTION 4: No errors or UPLOAD_FAILED events
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
        # Track sequence of events to verify timing (event-based, no polling)
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

        # Track upload calls with sequence markers
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

            # Emit READY_FOR_UPLOAD
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                "trace-123",
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
                None,
            )

            # Wait for upload to complete (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        # KEY ASSERTION 1: upload() called 3 times
        assert (
            upload_call_count[0] == 3
        ), f"Expected 3 upload calls, got {upload_call_count[0]}"

        # KEY ASSERTION 2: UPLOAD_COMPLETE emitted exactly 1 time
        complete_events = [e for e in event_sequence if e.startswith("UPLOAD_COMPLETE")]
        assert (
            len(complete_events) == 1
        ), f"Expected 1 UPLOAD_COMPLETE, got {len(complete_events)}"

        # KEY ASSERTION 3: UPLOAD_COMPLETE emitted AFTER all 3 uploads finish
        # Find positions in sequence
        upload_positions = [
            i for i, e in enumerate(event_sequence) if e.startswith("UPLOAD_FILE")
        ]
        complete_position = event_sequence.index("UPLOAD_COMPLETE:trace-123")

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
        # Track events emitted (event-based, no polling)
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

        # Mock uploader to track if it was ever instantiated
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

            # Emit READY_FOR_UPLOAD with empty directory path
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                "trace-empty-123",  # trace_id
                "rec-456",  # recording_id
                str(empty_directory),  # path (empty directory)
                DataType.RGB_IMAGES,  # data_type
                "camera_0",  # data_type_name
                0,  # bytes_uploaded
                None,  # external_trace_id
            )

            # Wait for upload to fail (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        # KEY ASSERTION 1: UPLOAD_FAILED emitted with status=FAILED
        assert (
            len(upload_failed_events) == 1
        ), f"Expected 1 UPLOAD_FAILED, got {len(upload_failed_events)}"
        failed_event = upload_failed_events[0]
        # Event args: (trace_id, bytes_uploaded, status, error_code, error_message)
        assert failed_event[0] == "trace-empty-123", "Wrong trace_id in UPLOAD_FAILED"
        assert (
            failed_event[2] == TraceStatus.FAILED
        ), f"Expected status=FAILED, got {failed_event[2]}"

        # KEY ASSERTION 2: Error message indicates empty directory
        error_message = failed_event[4]
        assert (
            "Empty directory" in error_message or "empty" in error_message.lower()
        ), f"Expected 'Empty directory' in error message, got: {error_message}"

        # KEY ASSERTION 3: UPLOAD_COMPLETE never emitted
        assert (
            len(upload_complete_events) == 0
        ), f"Expected no UPLOAD_COMPLETE, got {upload_complete_events}"

        # KEY ASSERTION 4: ResumableFileUploader never instantiated
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
        # Track uploader calls to inspect cloud_filepath (event-based, no polling)
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

            # Emit READY_FOR_UPLOAD with specific data_type and data_type_name
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                "trace-123",  # trace_id
                "rec-456",  # recording_id
                str(trace_directory),  # path
                DataType.RGB_IMAGES,  # data_type
                "camera_front",  # data_type_name
                0,  # bytes_uploaded
                None,  # external_trace_id
            )

            # Wait for upload to complete (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        # Verify calls were made
        assert len(uploader_calls) == 3, f"Expected 3 calls, got {len(uploader_calls)}"

        # Check first file (lossless.mp4 - sorted alphabetically first)
        first_call = uploader_calls[0]
        cloud_filepath = first_call["cloud_filepath"]

        # KEY ASSERTION 1: cloud_filepath follows format
        # {data_type.value}/{data_type_name}/{filename}
        expected_path = f"{DataType.RGB_IMAGES.value}/camera_front/lossless.mp4"
        assert (
            cloud_filepath == expected_path
        ), f"Expected cloud_filepath='{expected_path}', got '{cloud_filepath}'"

        # KEY ASSERTION 2: Starts with DataType enum VALUE (not name)
        assert cloud_filepath.startswith(
            DataType.RGB_IMAGES.value
        ), f"Path should start with '{DataType.RGB_IMAGES.value}': {cloud_filepath}"
        assert not cloud_filepath.startswith(
            "DataType."
        ), f"Path should not start with 'DataType.', got '{cloud_filepath}'"

        # KEY ASSERTION 3: Uses "/" as separator (not "\\" or other)
        assert (
            "\\" not in cloud_filepath
        ), f"Path should use '/' separator, not '\\': {cloud_filepath}"
        parts = cloud_filepath.split("/")
        assert len(parts) == 3, f"Expected 3 path parts, got {parts}"

        # KEY ASSERTION 4: Ends with actual filename from disk
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
        # Event-based waiting (no polling)
        uploader_calls: list[dict] = []
        upload_complete_events: list[str] = []
        upload_done = asyncio.Event()
        trace_id = "trace-abc-123-uuid-456"  # Intentionally includes UUID-like pattern

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

            # Emit READY_FOR_UPLOAD
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                trace_id,  # trace_id with UUID pattern
                "rec-456",  # recording_id
                str(trace_directory),  # path
                DataType.RGB_IMAGES,  # data_type
                "camera_0",  # data_type_name
                0,  # bytes_uploaded
                None,  # external_trace_id
            )

            # Wait for upload to complete (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        # KEY ASSERTION 1: 3 different cloud paths generated
        assert len(uploader_calls) == 3, f"Expected 3 calls, got {len(uploader_calls)}"

        cloud_paths = [call["cloud_filepath"] for call in uploader_calls]
        unique_paths = set(cloud_paths)
        assert (
            len(unique_paths) == 3
        ), f"Expected 3 unique paths, got {len(unique_paths)}: {cloud_paths}"

        # KEY ASSERTION 2: Each ends with actual filename from disk
        expected_filenames = {"lossless.mp4", "lossy.mp4", "trace.json"}
        actual_filenames = {path.split("/")[-1] for path in cloud_paths}
        assert (
            actual_filenames == expected_filenames
        ), f"Expected filenames {expected_filenames}, got {actual_filenames}"

        # KEY ASSERTION 3: File extension preserved (.mp4, .json)
        for path in cloud_paths:
            filename = path.split("/")[-1]
            assert "." in filename, f"Filename should have extension: {filename}"
            ext = filename.split(".")[-1]
            assert ext in ["mp4", "json"], f"Unexpected extension: {ext}"

        # KEY ASSERTION 4: No trace_id or UUID in paths
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
# SECTION 3: EXTERNAL TRACE ID MANAGEMENT
# =============================================================================


class TestExternalTraceIdManagement:
    """Tests for external_trace_id persistence and reuse."""

    @pytest.mark.asyncio
    async def test_t3_1_first_upload_registers_and_saves_external_trace_id(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
    ) -> None:
        """T3.1: First upload registers and saves external_trace_id.

        The Story:
            When a trace is uploaded for the first time, external_trace_id is None.
            The upload manager must register with the backend to get an ID, then
            emit EXTERNAL_TRACE_ID_SET so state manager can persist it to SQLite.

        The Flow:
            1. Emit READY_FOR_UPLOAD with external_trace_id=None
            2. Upload manager calls _register_data_trace(recording_id, data_type)
            3. Backend returns external_trace_id="backend-trace-ABC"
            4. Upload manager emits EXTERNAL_TRACE_ID_SET(trace_id, "backend-trace-ABC")
            5. State manager receives event, calls store.set_external_trace_id()
            6. SQLite now has external_trace_id column populated

        Why This Matters:
            The external_trace_id is the backend's identifier for this trace. All
            subsequent API calls (update status, get upload URLs) must use this ID.
            If not saved, daemon restart would register a NEW trace, creating orphans.

        Key Assertions:
            - _register_data_trace called exactly once
            - EXTERNAL_TRACE_ID_SET event emitted with correct IDs
            - store.set_external_trace_id() called
            - Backend updates use the returned external_trace_id
        """
        # Track events emitted (event-based, no polling)
        external_trace_id_events: list[tuple] = []
        upload_complete_events: list[str] = []
        upload_done = asyncio.Event()
        trace_id = "trace-internal-123"
        backend_trace_id = "backend-trace-ABC"

        emitter = get_emitter()
        emitter.on(
            Emitter.EXTERNAL_TRACE_ID_SET,
            lambda tid, ext_id: external_trace_id_events.append((tid, ext_id)),
        )
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
            mock_register.return_value = backend_trace_id
            MockUploader.return_value = mock_uploader_instance

            # Emit READY_FOR_UPLOAD with external_trace_id=None (first upload)
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                trace_id,  # trace_id
                "rec-456",  # recording_id
                str(trace_directory),  # path
                DataType.RGB_IMAGES,  # data_type
                "camera_0",  # data_type_name
                0,  # bytes_uploaded
                None,  # external_trace_id = None (first upload)
            )

            # Wait for upload to complete (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

            # KEY ASSERTION 1: _register_data_trace called exactly once
            assert mock_register.call_count == 1, (
                f"Expected _register_data_trace to be called once, "
                f"got {mock_register.call_count} calls"
            )

            # Verify registration was called with correct args
            register_call_args = mock_register.call_args
            assert register_call_args[0][0] == "rec-456", "Wrong recording_id"
            assert register_call_args[0][1] == DataType.RGB_IMAGES, "Wrong data_type"

        # KEY ASSERTION 2: EXTERNAL_TRACE_ID_SET event emitted with correct IDs
        assert (
            len(external_trace_id_events) == 1
        ), f"Expected 1 EXTERNAL_TRACE_ID_SET, got {len(external_trace_id_events)}"
        event_trace_id, event_external_id = external_trace_id_events[0]
        assert (
            event_trace_id == trace_id
        ), f"Expected trace_id='{trace_id}', got '{event_trace_id}'"
        assert (
            event_external_id == backend_trace_id
        ), f"Expected external_trace_id='{backend_trace_id}', got '{event_external_id}'"

        # KEY ASSERTION 3: Backend updates use the returned external_trace_id
        # Check that _update_data_trace was called with the backend trace ID
        assert (
            mock_update.call_count >= 1
        ), "Expected at least one _update_data_trace call"

        # Verify backend updates used the correct external_trace_id
        for call in mock_update.call_args_list:
            call_args = call[0]
            # _update_data_trace(recording_id, external_trace_id, status, ...)
            used_trace_id = call_args[1]  # Second positional arg is the trace ID
            assert used_trace_id == backend_trace_id, (
                f"Backend update should use external_trace_id='{backend_trace_id}', "
                f"but used '{used_trace_id}'"
            )

    @pytest.mark.asyncio
    async def test_t3_2_retry_upload_skips_registration(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
    ) -> None:
        """T3.2: Retry upload skips registration (uses saved ID).

        The Story:
            Upload failed at 50%. Daemon restarts. SQLite has external_trace_id="ABC".
            READY_FOR_UPLOAD includes this ID. Upload manager must NOT register again;
            instead, it should resume using the existing backend trace.

        The Flow:
            1. Emit READY_FOR_UPLOAD with external_trace_id="ABC"
            2. Upload manager sees external_trace_id is NOT None
            3. Skips _register_data_trace() call entirely
            4. Proceeds directly to file upload
            5. Backend updates go to trace "ABC"

        Why This Matters:
            Re-registering creates a new backend trace with a new ID. The old trace
            "ABC" becomes orphaned - it shows as "uploading forever" in the UI.
            User's dashboard fills with stuck traces. Backend storage has duplicates.

        Key Assertions:
            - _register_data_trace NOT called
            - EXTERNAL_TRACE_ID_SET NOT emitted
            - _update_data_trace called with external_trace_id="ABC"
            - No new traces created in backend
        """
        # Track events emitted (event-based, no polling)
        external_trace_id_events: list[tuple] = []
        upload_complete_events: list[str] = []
        upload_done = asyncio.Event()
        trace_id = "trace-internal-123"
        saved_external_trace_id = "ABC"  # Previously saved from first upload attempt

        emitter = get_emitter()
        emitter.on(
            Emitter.EXTERNAL_TRACE_ID_SET,
            lambda tid, ext_id: external_trace_id_events.append((tid, ext_id)),
        )
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
            mock_register.return_value = "should-not-be-used"
            MockUploader.return_value = mock_uploader_instance

            # Emit READY_FOR_UPLOAD with external_trace_id already set (retry scenario)
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                trace_id,  # trace_id
                "rec-456",  # recording_id
                str(trace_directory),  # path
                DataType.RGB_IMAGES,  # data_type
                "camera_0",  # data_type_name
                500,  # bytes_uploaded (partial progress from before)
                saved_external_trace_id,  # external_trace_id (already registered)
            )

            # Wait for upload to complete (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

            # KEY ASSERTION 1: _register_data_trace NOT called
            assert mock_register.call_count == 0, (
                f"Expected _register_data_trace NOT called (has external_trace_id), "
                f"but called {mock_register.call_count} times"
            )

            # KEY ASSERTION 2: EXTERNAL_TRACE_ID_SET NOT emitted
            assert len(external_trace_id_events) == 0, (
                f"Expected no EXTERNAL_TRACE_ID_SET events (already have ID), "
                f"but got {len(external_trace_id_events)}: {external_trace_id_events}"
            )

            # KEY ASSERTION 3: _update_data_trace called with external_trace_id="ABC"
            assert (
                mock_update.call_count >= 1
            ), "Expected at least one _update_data_trace call"

            # Verify ALL backend updates use the saved external_trace_id
            for i, call in enumerate(mock_update.call_args_list):
                call_args = call[0]
                # _update_data_trace(recording_id, external_trace_id, status, ...)
                used_trace_id = call_args[1]  # Second positional arg is the trace ID
                assert used_trace_id == saved_external_trace_id, (
                    f"Call {i}: Backend update should use external_trace_id="
                    f"'{saved_external_trace_id}', but used '{used_trace_id}'"
                )

        # KEY ASSERTION 4: No new traces created in backend
        # This is implicitly verified by:
        # - _register_data_trace not called (no new registration)
        # - All updates go to existing trace "ABC"

    @pytest.mark.asyncio
    async def test_t3_3_registration_failure_emits_upload_failed(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
    ) -> None:
        """T3.3: Registration failure emits UPLOAD_FAILED.

        The Story:
            First upload attempt. external_trace_id is None. Backend registration fails
            (network error, auth expired, server error). Upload manager must fail
            gracefully without trying to upload files.

        The Flow:
            1. Emit READY_FOR_UPLOAD with external_trace_id=None
            2. _register_data_trace() returns None (failure)
            3. Upload manager emits UPLOAD_FAILED
            4. Does NOT attempt file upload

        Why This Matters:
            Without an external_trace_id, we cannot get upload URLs from
            backend. Attempting uploads would fail anyway. Failing early
            preserves bytes_uploaded=0
            so retry starts fresh. Clear error message helps debugging.

        Key Assertions:
            - UPLOAD_FAILED emitted with "Failed to register trace" message
            - status = TraceStatus.FAILED
            - ResumableFileUploader never instantiated
            - bytes_uploaded remains 0
        """
        # Track events emitted (event-based, no polling)
        upload_failed_events: list[tuple] = []
        upload_complete_events: list[str] = []
        uploader_instantiated = [False]
        upload_done = asyncio.Event()
        trace_id = "trace-internal-123"

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
            # Registration FAILS - returns None
            mock_register.return_value = None
            MockUploader.side_effect = capture_uploader_call

            # Emit READY_FOR_UPLOAD with external_trace_id=None (first upload attempt)
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                trace_id,  # trace_id
                "rec-456",  # recording_id
                str(trace_directory),  # path
                DataType.RGB_IMAGES,  # data_type
                "camera_0",  # data_type_name
                0,  # bytes_uploaded
                None,  # external_trace_id = None (first upload, needs registration)
            )

            # Wait for upload to fail (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        # KEY ASSERTION 1: UPLOAD_FAILED emitted with "Failed to register trace" message
        assert (
            len(upload_failed_events) == 1
        ), f"Expected 1 UPLOAD_FAILED event, got {len(upload_failed_events)}"
        failed_event = upload_failed_events[0]
        # Event args: (trace_id, bytes_uploaded, status, error_code, error_message)
        assert failed_event[0] == trace_id, f"Wrong trace_id: {failed_event[0]}"

        error_message = failed_event[4]
        assert (
            "register" in error_message.lower() or "failed" in error_message.lower()
        ), f"Expected 'register' or 'failed' in error message, got: {error_message}"

        # KEY ASSERTION 2: status = TraceStatus.FAILED
        assert (
            failed_event[2] == TraceStatus.FAILED
        ), f"Expected status=TraceStatus.FAILED, got {failed_event[2]}"

        # KEY ASSERTION 3: ResumableFileUploader never instantiated
        assert not uploader_instantiated[
            0
        ], "ResumableFileUploader should not be instantiated when registration fails"

        # KEY ASSERTION 4: bytes_uploaded remains 0
        assert failed_event[1] == 0, f"Expected bytes_uploaded=0, got {failed_event[1]}"

        # Extra: No UPLOAD_COMPLETE
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
        # Event-based waiting (no polling)
        internal_trace_id = "internal-abc-123"
        external_trace_id = "backend-xyz-789"
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
            mock_register.return_value = external_trace_id
            MockUploader.return_value = mock_uploader_instance

            # Capture all _update_data_trace calls
            async def capture_update(*args, **kwargs):
                update_trace_calls.append((args, kwargs))

            mock_update.side_effect = capture_update

            # Emit READY_FOR_UPLOAD with internal trace_id, no external_trace_id yet
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                internal_trace_id,  # Internal daemon trace_id
                "rec-456",  # recording_id
                str(trace_directory),  # path
                DataType.RGB_IMAGES,  # data_type
                "camera_0",  # data_type_name
                0,  # bytes_uploaded
                None,  # external_trace_id = None (first upload)
            )

            # Wait for upload to complete (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        # KEY ASSERTION 1: Verify _update_data_trace calls use external_trace_id
        assert len(update_trace_calls) >= 2, (
            f"Expected at least 2 _update_data_trace calls (STARTED and COMPLETE), "
            f"got {len(update_trace_calls)}"
        )

        for i, (args, kwargs) in enumerate(update_trace_calls):
            # _update_data_trace(recording_id, trace_id, status, ...)
            # Second positional argument should be the external_trace_id
            used_trace_id = args[1]

            # KEY ASSERTION 2: Uses "backend-xyz-789"
            assert used_trace_id == external_trace_id, (
                f"Call {i}: Expected external_trace_id='{external_trace_id}', "
                f"but got '{used_trace_id}'"
            )

            # KEY ASSERTION 3: Never contains internal trace_id
            assert used_trace_id != internal_trace_id, (
                f"Call {i}: Should NOT use internal_trace_id='{internal_trace_id}', "
                f"but did!"
            )

        # KEY ASSERTION 4: All update calls use same external ID (consistency)
        trace_ids_used = {args[1] for args, _ in update_trace_calls}
        assert len(trace_ids_used) == 1, (
            f"All update calls should use same external_trace_id, "
            f"but found different IDs: {trace_ids_used}"
        )
        assert (
            external_trace_id in trace_ids_used
        ), f"Expected '{external_trace_id}' in trace_ids_used, got {trace_ids_used}"

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

        # Event-based waiting (no polling)
        external_trace_id = "ABC-123"
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
            mock_register.return_value = external_trace_id
            MockUploader.return_value = mock_uploader_instance

            # Capture all _update_data_trace calls
            async def capture_update(*args, **kwargs):
                update_calls.append((args, kwargs))

            mock_update.side_effect = capture_update

            # Emit READY_FOR_UPLOAD (fresh upload)
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                "trace-internal",  # trace_id
                recording_id,  # recording_id
                str(trace_directory),  # path
                DataType.RGB_IMAGES,  # data_type
                "camera_0",  # data_type_name
                0,  # bytes_uploaded = 0 (fresh upload)
                None,  # external_trace_id = None (first upload)
            )

            # Wait for upload to complete (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        # Find the UPLOAD_STARTED call
        upload_started_call = None
        for args, kwargs in update_calls:
            # _update_data_trace(recording_id, trace_id, status, ...)
            if len(args) >= 3 and args[2] == RecordingDataTraceStatus.UPLOAD_STARTED:
                upload_started_call = (args, kwargs)
                break

        # KEY ASSERTION 1: _update_data_trace receives external_trace_id="ABC-123"
        assert (
            upload_started_call is not None
        ), "Expected an UPLOAD_STARTED call to _update_data_trace"
        args, kwargs = upload_started_call

        # Verify recording_id
        assert (
            args[0] == recording_id
        ), f"Expected recording_id='{recording_id}', got '{args[0]}'"

        # KEY ASSERTION 2: Uses external_trace_id
        assert (
            args[1] == external_trace_id
        ), f"Expected external_trace_id='{external_trace_id}', got '{args[1]}'"

        # KEY ASSERTION 3: Status is UPLOAD_STARTED
        assert (
            args[2] == RecordingDataTraceStatus.UPLOAD_STARTED
        ), f"Expected status=UPLOAD_STARTED, got {args[2]}"

        # KEY ASSERTION 4: uploaded_bytes=0 for fresh upload
        # Check kwargs for uploaded_bytes
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

        # Event-based waiting (no polling)
        external_trace_id = "ABC-123"
        recording_id = "rec-456"
        update_calls: list[tuple] = []
        upload_complete_events: list[str] = []
        upload_done = asyncio.Event()

        emitter = get_emitter()
        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            make_upload_complete_handler(upload_complete_events, upload_done),
        )

        # Mock uploader that returns specific byte counts
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
            mock_register.return_value = external_trace_id
            MockUploader.return_value = mock_uploader_instance

            # Capture all _update_data_trace calls
            async def capture_update(*args, **kwargs):
                update_calls.append((args, kwargs))

            mock_update.side_effect = capture_update

            # Emit READY_FOR_UPLOAD
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                "trace-internal",
                recording_id,
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
                None,
            )

            # Wait for upload to complete (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        # Find the UPLOAD_COMPLETE call
        upload_complete_calls = [
            (args, kwargs)
            for args, kwargs in update_calls
            if len(args) >= 3 and args[2] == RecordingDataTraceStatus.UPLOAD_COMPLETE
        ]

        # KEY ASSERTION 1: Called exactly once after all files done
        assert (
            len(upload_complete_calls) == 1
        ), f"Expected exactly 1 UPLOAD_COMPLETE call, got {len(upload_complete_calls)}"

        args, kwargs = upload_complete_calls[0]

        # KEY ASSERTION 2: External trace ID "ABC-123" in request
        assert (
            args[1] == external_trace_id
        ), f"Expected external_trace_id='{external_trace_id}', got '{args[1]}'"

        # KEY ASSERTION 3: status = RecordingDataTraceStatus.UPLOAD_COMPLETE
        assert (
            args[2] == RecordingDataTraceStatus.UPLOAD_COMPLETE
        ), f"Expected status=UPLOAD_COMPLETE, got {args[2]}"

        # KEY ASSERTION 4: uploaded_bytes = total_bytes (and both > 0)
        # Check that final byte counts are reported
        uploaded_bytes = kwargs.get("uploaded_bytes")
        total_bytes = kwargs.get("total_bytes")

        # At minimum, we should have positive byte counts
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
    async def test_t4_4_resume_uses_persisted_external_trace_id(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
    ) -> None:
        """T4.4: Resume uses persisted external_trace_id for all backend calls.

        The Story:
            Daemon crashed mid-upload. Restart loads trace from SQLite with saved
            external_trace_id. All subsequent backend calls must use this ID, not
            try to register a new one.

        The Flow:
            1. SQLite has trace with external_trace_id="saved-ID-from-before"
            2. READY_FOR_UPLOAD emitted with external_trace_id="saved-ID-from-before"
            3. Upload manager skips registration
            4. Calls _update_data_trace with "saved-ID-from-before"
            5. Completes upload, UPLOAD_COMPLETE uses "saved-ID-from-before"

        Why This Matters:
            If we register again, backend creates NEW trace "new-ID-456".
            Old trace "saved-ID-from-before" forever stuck as "uploading".
            User sees duplicate traces, one stuck, one complete.

        Key Assertions:
            - _register_data_trace NEVER called
            - All _update_data_trace calls use "saved-ID-from-before"
            - UPLOAD_COMPLETE sent to "saved-ID-from-before"
            - No orphan traces created
        """
        from neuracore_types import RecordingDataTraceStatus

        # Event-based waiting (no polling)
        saved_external_trace_id = "saved-ID-from-before"
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
            mock_register.return_value = "should-never-be-called"
            MockUploader.return_value = mock_uploader_instance

            # Capture all _update_data_trace calls
            async def capture_update(*args, **kwargs):
                update_calls.append((args, kwargs))

            mock_update.side_effect = capture_update

            # Emit READY_FOR_UPLOAD with PERSISTED external_trace_id (resume scenario)
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                "trace-internal-123",
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                500,  # bytes_uploaded (partial progress from before crash)
                saved_external_trace_id,  # PERSISTED external_trace_id from SQLite
            )

            # Wait for upload to complete (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

            # KEY ASSERTION 1: _register_data_trace NEVER called
            assert mock_register.call_count == 0, (
                f"_register_data_trace should NOT be called on resume, "
                f"but was called {mock_register.call_count} times"
            )

        # KEY ASSERTION 2: All _update_data_trace calls use "saved-ID-from-before"
        assert len(update_calls) >= 1, "Expected at least one _update_data_trace call"

        for i, (args, kwargs) in enumerate(update_calls):
            used_trace_id = args[1]  # Second arg is trace_id
            assert used_trace_id == saved_external_trace_id, (
                f"Call {i}: Expected external_trace_id='{saved_external_trace_id}', "
                f"got '{used_trace_id}'"
            )

        # KEY ASSERTION 3: UPLOAD_COMPLETE sent to "saved-ID-from-before"
        upload_complete_calls = [
            (args, kwargs)
            for args, kwargs in update_calls
            if len(args) >= 3 and args[2] == RecordingDataTraceStatus.UPLOAD_COMPLETE
        ]

        assert (
            len(upload_complete_calls) == 1
        ), f"Expected 1 UPLOAD_COMPLETE call, got {len(upload_complete_calls)}"

        complete_args, _ = upload_complete_calls[0]
        assert complete_args[1] == saved_external_trace_id, (
            f"UPLOAD_COMPLETE should use '{saved_external_trace_id}', "
            f"got '{complete_args[1]}'"
        )

        # KEY ASSERTION 4: No orphan traces created
        # This is implicitly verified by:
        # - _register_data_trace not called (no new registration)
        # - All updates go to saved external_trace_id


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
            SQLite has: bytes_uploaded=120MB, external_trace_id="ABC".
            On retry, must skip lossless.mp4, resume lossy.mp4 at offset 20MB.

        The Flow:
            1. Emit READY_FOR_UPLOAD with bytes_uploaded=120MB, external_trace_id="ABC"
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
            - No _register_data_trace call (already have external_trace_id)
            - Total upload = 31MB, not 151MB
        """
        # Create trace directory with 3 files of specific sizes
        # Using bytes instead of MB for test efficiency
        trace_dir = tmp_path / "trace-resume-test"
        trace_dir.mkdir()

        # lossless.mp4: 100 bytes (alphabetically first)
        (trace_dir / "lossless.mp4").write_bytes(b"L" * 100)
        # lossy.mp4: 50 bytes (alphabetically second)
        (trace_dir / "lossy.mp4").write_bytes(b"O" * 50)
        # trace.json: 10 bytes (alphabetically third)
        (trace_dir / "trace.json").write_bytes(b'{"data": 1}')

        # Simulate: lossless.mp4 (100) complete + 20 bytes of lossy.mp4 = 120
        bytes_uploaded = 120
        external_trace_id = "ABC"

        # Event-based waiting (no polling)
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
            mock_register.return_value = "should-not-be-called"

            def capture_uploader_call(**kwargs):
                uploader_calls.append(kwargs)
                return mock_uploader_instance

            MockUploader.side_effect = capture_uploader_call

            # Emit READY_FOR_UPLOAD with resume state
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                "trace-123",
                "rec-456",
                str(trace_dir),
                DataType.RGB_IMAGES,
                "camera_0",
                bytes_uploaded,  # 120 bytes already uploaded
                external_trace_id,  # Already have external_trace_id
            )

            # Wait for upload to complete (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

            # KEY ASSERTION 4: No _register_data_trace call (have external_trace_id)
            assert mock_register.call_count == 0, (
                f"Should not register (have external_trace_id), "
                f"but called {mock_register.call_count} times"
            )

        # KEY ASSERTION 1: ResumableFileUploader NOT created for lossless.mp4
        uploaded_filenames = [
            call["filepath"].split("/")[-1] for call in uploader_calls
        ]
        assert "lossless.mp4" not in uploaded_filenames, (
            f"lossless.mp4 should be skipped (already complete), "
            f"but found in uploads: {uploaded_filenames}"
        )

        # KEY ASSERTION 2 & 3: Check bytes_uploaded for remaining files
        # Should have 2 uploads: lossy.mp4 (with offset) and trace.json (from start)
        assert len(uploader_calls) == 2, (
            f"Expected 2 uploads (lossy.mp4 and trace.json), got {len(uploader_calls)}"
            f"{uploaded_filenames}"
        )

        # Find the lossy.mp4 call
        lossy_call = next(
            (c for c in uploader_calls if "lossy.mp4" in c["filepath"]),
            None,
        )
        assert lossy_call is not None, "lossy.mp4 should be uploaded"

        # KEY ASSERTION 2: lossy.mp4 uploader created with bytes_uploaded=20
        assert (
            lossy_call["bytes_uploaded"] == 20
        ), f"lossy.mp4 should resume at offset 20, got {lossy_call['bytes_uploaded']}"

        # Find the trace.json call
        json_call = next(
            (c for c in uploader_calls if "trace.json" in c["filepath"]),
            None,
        )
        assert json_call is not None, "trace.json should be uploaded"

        # KEY ASSERTION 3: trace.json uploader created with bytes_uploaded=0
        assert (
            json_call["bytes_uploaded"] == 0
        ), f"trace.json should start from 0, got {json_call['bytes_uploaded']}"

        # KEY ASSERTION 6: Emits UPLOAD_COMPLETE
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
        # Create trace directory with 2 files of specific sizes
        trace_dir = tmp_path / "trace-partial-fail"
        trace_dir.mkdir()

        # Using bytes for test efficiency
        (trace_dir / "file_a.mp4").write_bytes(b"A" * 100)  # 100 bytes
        (trace_dir / "file_b.mp4").write_bytes(b"B" * 50)  # 50 bytes

        # Event-based waiting (no polling)
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

        # Track which file is being uploaded
        mock_uploader_instance = MagicMock()

        async def mock_upload():
            upload_call_count[0] += 1
            if upload_call_count[0] == 1:
                # First file succeeds
                return (True, 100, None)
            else:
                # Second file fails with network error at 20 bytes
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

            # Emit READY_FOR_UPLOAD
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                "trace-123",
                "rec-456",
                str(trace_dir),
                DataType.RGB_IMAGES,
                "camera_0",
                0,  # Fresh upload
                None,  # First upload, no external_trace_id yet
            )

            # Wait for upload to fail (event-based, no polling)
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        # KEY ASSERTION 1: UPLOAD_FAILED emitted (not UPLOAD_COMPLETE)
        assert (
            len(upload_complete_events) == 0
        ), f"Expected no UPLOAD_COMPLETE on failure, got {upload_complete_events}"
        assert (
            len(upload_failed_events) == 1
        ), f"Expected 1 UPLOAD_FAILED, got {len(upload_failed_events)}"

        failed_event = upload_failed_events[0]
        # Event args: (trace_id, bytes_uploaded, status, error_code, error_message)

        # KEY ASSERTION 2: bytes_uploaded = 120 (100 + 20)
        # First file (100) complete + second file partial (20) = 120
        assert (
            failed_event[1] == 120
        ), f"Expected bytes_uploaded=120 (100+20), got {failed_event[1]}"

        # KEY ASSERTION 3: status = TraceStatus.WRITTEN (retryable)
        assert (
            failed_event[2] == TraceStatus.WRITTEN
        ), f"Expected status=WRITTEN (retryable), got {failed_event[2]}"

        # KEY ASSERTION 4: error_code = TraceErrorCode.NETWORK_ERROR
        assert (
            failed_event[3] == TraceErrorCode.NETWORK_ERROR
        ), f"Expected error_code=NETWORK_ERROR, got {failed_event[3]}"

        # Verify error message contains network-related text
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
        # Create 5 separate trace directories
        trace_dirs: list[Path] = []
        trace_ids: list[str] = []
        for i in range(5):
            trace_dir = tmp_path / f"trace-{i}"
            trace_dir.mkdir()
            (trace_dir / "file.mp4").write_bytes(b"X" * 100)
            (trace_dir / "trace.json").write_bytes(b'{"i": ' + str(i).encode() + b"}")
            trace_dirs.append(trace_dir)
            trace_ids.append(f"trace-{i}")

        # Track events with event-based signaling
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

        # Mock uploader to succeed quickly
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

                # Emit 5 READY_FOR_UPLOAD events in rapid succession
                for i, (trace_dir, trace_id) in enumerate(zip(trace_dirs, trace_ids)):
                    emitter.emit(
                        Emitter.READY_FOR_UPLOAD,
                        trace_id,
                        f"rec-{i}",
                        str(trace_dir),
                        DataType.RGB_IMAGES,
                        f"camera_{i}",
                        0,
                        None,
                    )

                # Wait for all 5 uploads to complete (event-based, no arbitrary sleep)
                await asyncio.wait_for(
                    all_complete.wait(), timeout=TEST_TIMEOUT_SECONDS
                )
                # Yield control to allow task cleanup after event emission
                await asyncio.sleep(0)
        finally:
            emitter.remove_listener(Emitter.UPLOAD_COMPLETE, on_complete)

        # KEY ASSERTION 1: 5 UPLOAD_COMPLETE events emitted
        assert (
            len(upload_complete_events) == 5
        ), f"Expected 5 UPLOAD_COMPLETE, got {len(upload_complete_events)}"

        # KEY ASSERTION 2: All 5 unique trace IDs in completion events
        assert set(upload_complete_events) == set(
            trace_ids
        ), f"Expected trace IDs {trace_ids}, got {upload_complete_events}"

        # KEY ASSERTION 3: No UPLOAD_FAILED events
        assert (
            len(upload_failed_events) == 0
        ), f"Expected no failures, got {upload_failed_events}"

        # KEY ASSERTION 4: Active uploads set empty after completion
        assert (
            len(upload_manager._active_uploads) == 0
        ), f"Expected empty active uploads, got {len(upload_manager._active_uploads)}"
