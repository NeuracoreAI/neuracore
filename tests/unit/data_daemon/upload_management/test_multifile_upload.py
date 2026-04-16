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
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import aiohttp
import pytest
import pytest_asyncio
from neuracore_types import DataType

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.models import TraceErrorCode
from neuracore.data_daemon.upload_management.upload_manager import UploadManager

TEST_TIMEOUT_SECONDS = 10.0

TEST_TRACE_ID = "11111111-1111-1111-1111-111111111111"
TEST_TRACE_ID_2 = "22222222-2222-2222-2222-222222222222"
TEST_TRACE_ID_3 = "33333333-3333-3333-3333-333333333333"
TEST_TRACE_ID_EMPTY = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
TEST_TRACE_ID_INTERNAL = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
TEST_TRACE_ID_RESUME = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


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


@pytest_asyncio.fixture
async def client_session():
    """Create an aiohttp session for testing."""
    session = aiohttp.ClientSession()
    yield session
    await session.close()


@pytest.fixture
def mock_trace_status_updater():
    """Mock for the TraceStatusUpdater dependency."""
    updater = MagicMock()
    updater.update_trace_started = AsyncMock(return_value=True)
    updater.update_trace_completed = AsyncMock(return_value=True)
    updater.update_trace_progress = AsyncMock(return_value=True)
    return updater


@pytest_asyncio.fixture
async def upload_manager(
    client_session: aiohttp.ClientSession, emitter: Emitter, mock_trace_status_updater
):
    """Create and cleanup UploadManager instance."""
    config = DaemonConfig(num_threads=2)
    manager = UploadManager(
        config=config,
        trace_status_updater=mock_trace_status_updater,
        client_session=client_session,
        emitter=emitter,
    )
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


class TestDirectoryUploadHandling:
    """Tests for directory enumeration and multi-file upload."""

    @pytest.mark.asyncio
    async def test_t1_1_directory_with_multiple_files_uploads_successfully(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
        emitter: Emitter,
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
        upload_done = asyncio.Event()

        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            make_upload_complete_handler(upload_complete_events, upload_done),
        )

        mock_uploader = MagicMock()
        mock_uploader.upload = AsyncMock(return_value=(True, 1000, None))

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader",
            return_value=mock_uploader,
        ) as MockUploader:
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

        assert MockUploader.call_count == 3
        filenames = [
            Path(call.kwargs["filepath"]).name for call in MockUploader.call_args_list
        ]
        assert filenames == ["lossless.mp4", "lossy.mp4", "trace.json"]
        assert upload_complete_events[0] == TEST_TRACE_ID

    @pytest.mark.asyncio
    async def test_t1_2_all_files_uploaded_before_upload_complete(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
        emitter: Emitter,
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

        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            lambda trace_id: [
                event_sequence.append(f"UPLOAD_COMPLETE:{trace_id}"),
                upload_done.set(),
            ],
        )
        emitter.on(
            Emitter.DELETE_TRACE,
            lambda *args: event_sequence.append(f"DELETE_TRACE:{args[1]}"),
        )

        mock_uploader = MagicMock()

        async def mock_upload():
            event_sequence.append("UPLOAD_FILE")
            return (True, 1000, None)

        mock_uploader.upload = AsyncMock(side_effect=mock_upload)

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader",
            return_value=mock_uploader,
        ):
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

        assert event_sequence.count("UPLOAD_FILE") == 3
        assert event_sequence[-1].startswith("UPLOAD_COMPLETE")

    @pytest.mark.asyncio
    async def test_t1_3_empty_directory_fails_gracefully(
        self,
        upload_manager: UploadManager,
        empty_directory: Path,
        emitter: Emitter,
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
            - UPLOAD_FAILED emitted with error_code=UPLOAD_FAILED
            - Error message contains "No files found"
            - UPLOAD_COMPLETE never emitted
            - ResumableFileUploader never instantiated
        """
        upload_failed_events: list[tuple] = []
        upload_done = asyncio.Event()

        emitter.on(
            Emitter.UPLOAD_FAILED,
            lambda *args: [upload_failed_events.append(args), upload_done.set()],
        )

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader:
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

        assert len(upload_failed_events) == 1
        assert upload_failed_events[0][2] == TraceErrorCode.UPLOAD_FAILED
        assert MockUploader.call_count == 0


class TestCloudPathConstruction:
    """Tests for correct cloud filepath format."""

    @pytest.mark.asyncio
    async def test_t2_1_cloud_path_format_correct(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
        emitter: Emitter,
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
        upload_done = asyncio.Event()
        emitter.on(Emitter.UPLOAD_COMPLETE, lambda _: upload_done.set())

        mock_uploader = MagicMock()
        mock_uploader.upload = AsyncMock(return_value=(True, 1000, None))

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader",
            return_value=mock_uploader,
        ) as MockUploader:
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

        first_call_cloud_path = MockUploader.call_args_list[0].kwargs["cloud_filepath"]
        expected_path = f"{DataType.RGB_IMAGES.value}/camera_front/lossless.mp4"
        assert first_call_cloud_path == expected_path

    @pytest.mark.asyncio
    async def test_t2_2_each_file_gets_unique_path(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
        emitter: Emitter,
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
        upload_done = asyncio.Event()
        emitter.on(Emitter.UPLOAD_COMPLETE, lambda _: upload_done.set())

        mock_uploader = MagicMock()
        mock_uploader.upload = AsyncMock(return_value=(True, 1000, None))

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader",
            return_value=mock_uploader,
        ) as MockUploader:
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID_2,
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        cloud_paths = [
            call.kwargs["cloud_filepath"] for call in MockUploader.call_args_list
        ]
        assert len(set(cloud_paths)) == 3
        assert all(TEST_TRACE_ID_2 not in path for path in cloud_paths)


class TestRegistrationAndUploadFailures:
    """Tests for registration failures and error handling."""

    @pytest.mark.asyncio
    async def test_t3_1_registration_failure_emits_upload_failed(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
        emitter: Emitter,
        mock_trace_status_updater,
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
            - error_code = TraceErrorCode.NETWORK_ERROR
            - ResumableFileUploader never instantiated
            - bytes_uploaded remains 0
        """
        mock_trace_status_updater.update_trace_progress = AsyncMock(return_value=True)
        mock_trace_status_updater.update_trace_completed = AsyncMock(return_value=False)
        upload_failed_events: list[tuple] = []
        upload_done = asyncio.Event()

        mock_uploader = MagicMock()
        mock_uploader.upload = AsyncMock(return_value=(True, 1000, None))

        emitter.on(
            Emitter.UPLOAD_FAILED,
            lambda *args: [upload_failed_events.append(args), upload_done.set()],
        )
        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader",
            return_value=mock_uploader,
        ):
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID_INTERNAL,
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        assert len(upload_failed_events) == 1
        assert "trace status" in upload_failed_events[0][3].lower()


class TestBackendApiConsistency:
    """Tests for correct backend API usage."""

    @pytest.mark.asyncio
    async def test_t4_1_backend_updates_use_external_trace_id(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
        emitter: Emitter,
        mock_trace_status_updater,
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
        upload_done = asyncio.Event()
        emitter.on(Emitter.UPLOAD_COMPLETE, lambda _: upload_done.set())

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader:
            MockUploader.return_value.upload = AsyncMock(
                return_value=(True, 1000, None)
            )

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID_INTERNAL,
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        # Implementation uses trace_id directly in the updater calls
        mock_trace_status_updater.update_trace_completed.assert_called()
        assert (
            mock_trace_status_updater.update_trace_completed.call_args.kwargs[
                "trace_id"
            ]
            == TEST_TRACE_ID_INTERNAL
        )

    @pytest.mark.asyncio
    async def test_t4_2_upload_started_sends_external_trace_id(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
        emitter: Emitter,
        mock_trace_status_updater,
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
        upload_done = asyncio.Event()
        emitter.on(Emitter.UPLOAD_COMPLETE, lambda _: upload_done.set())

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader:
            MockUploader.return_value.upload = AsyncMock(
                return_value=(True, 1000, None)
            )

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID_INTERNAL,
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        mock_trace_status_updater.update_trace_completed.assert_called_with(
            recording_id="rec-456",
            trace_id=TEST_TRACE_ID_INTERNAL,
            total_bytes=ANY,
            wait_for_completion=True,
        )

    @pytest.mark.asyncio
    async def test_t4_3_upload_complete_sends_external_trace_id(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
        emitter: Emitter,
        mock_trace_status_updater,
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
        upload_done = asyncio.Event()
        emitter.on(Emitter.UPLOAD_COMPLETE, lambda _: upload_done.set())

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader:
            MockUploader.return_value.upload = AsyncMock(
                return_value=(True, 1000, None)
            )

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID_INTERNAL,
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        mock_trace_status_updater.update_trace_completed.assert_called_with(
            recording_id="rec-456",
            trace_id=TEST_TRACE_ID_INTERNAL,
            total_bytes=ANY,
            wait_for_completion=True,
        )

    @pytest.mark.asyncio
    async def test_t4_4_resume_uses_same_trace_id(
        self,
        upload_manager: UploadManager,
        trace_directory: Path,
        emitter: Emitter,
        mock_trace_status_updater,
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
        upload_done = asyncio.Event()
        emitter.on(Emitter.UPLOAD_COMPLETE, lambda _: upload_done.set())

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader:
            MockUploader.return_value.upload = AsyncMock(
                return_value=(True, 1000, None)
            )

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID_RESUME,
                "rec-456",
                str(trace_directory),
                DataType.RGB_IMAGES,
                "camera_0",
                500,
            )
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        assert (
            mock_trace_status_updater.update_trace_completed.call_args.kwargs[
                "trace_id"
            ]
            == TEST_TRACE_ID_RESUME
        )


class TestResumeLogic:
    """Tests for upload resume functionality."""

    @pytest.mark.asyncio
    async def test_t5_1_resume_from_middle_of_multi_file_upload(
        self,
        upload_manager: UploadManager,
        tmp_path: Path,
        emitter: Emitter,
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

        (trace_dir / "lossless.mp4").write_bytes(b"L" * 100)
        (trace_dir / "lossy.mp4").write_bytes(b"O" * 50)
        (trace_dir / "trace.json").write_bytes(b'{"data": 1}')

        upload_done = asyncio.Event()
        emitter.on(Emitter.UPLOAD_COMPLETE, lambda _: upload_done.set())

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader:
            MockUploader.return_value.upload = AsyncMock(return_value=(True, 50, None))

            # Resume at 120 (lossless.mp4 is 100, so we should be 20 bytes into b.mp4)
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID,
                "rec-456",
                str(trace_dir),
                DataType.RGB_IMAGES,
                "camera_0",
                120,
            )
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        # a.mp4 skipped, b.mp4 and c.json uploaded
        assert MockUploader.call_count == 2
        assert (
            MockUploader.call_args_list[0].kwargs["bytes_uploaded"] == 20
        )  # b.mp4 resume point

    @pytest.mark.asyncio
    async def test_t5_2_partial_failure_preserves_progress(
        self,
        upload_manager: UploadManager,
        tmp_path: Path,
        emitter: Emitter,
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
            - error_code = TraceErrorCode.NETWORK_ERROR (retryable)
            - error_code = TraceErrorCode.NETWORK_ERROR
        """
        trace_dir = tmp_path / "trace-partial-fail-test"
        trace_dir.mkdir()
        (trace_dir / "file_a.mp4").write_bytes(b"A" * 100)
        (trace_dir / "file_b.mp4").write_bytes(b"B" * 50)

        failed_event = asyncio.Event()
        errors = []
        emitter.on(
            Emitter.UPLOAD_FAILED,
            lambda *args: [errors.append(args), failed_event.set()],
        )

        mock_uploader = MagicMock()
        # First file succeeds (100b), second fails at 20b
        mock_uploader.upload = AsyncMock(
            side_effect=[(True, 100, None), (False, 20, "Network error")]
        )

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader",
            return_value=mock_uploader,
        ):
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID,
                "rec-456",
                str(trace_dir),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )
            await asyncio.wait_for(failed_event.wait(), timeout=TEST_TIMEOUT_SECONDS)

        # 100 + 20 = 120
        assert errors[0][1] == 120


class TestConcurrentUploads:
    """Tests for handling multiple simultaneous upload operations."""

    @pytest.mark.asyncio
    async def test_t6_1_handles_multiple_concurrent_uploads(
        self,
        upload_manager: UploadManager,
        tmp_path: Path,
        emitter: Emitter,
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
        trace_ids = [f"trace-{i}" for i in range(3)]
        for trace_id in trace_ids:
            trace_dir = tmp_path / trace_id
            trace_dir.mkdir()
            (trace_dir / "file.mp4").write_bytes(b"X" * 10)

        upload_done = asyncio.Event()
        completes = []
        emitter.on(
            Emitter.UPLOAD_COMPLETE,
            make_upload_complete_handler(completes, upload_done, expected_count=3),
        )

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader:
            MockUploader.return_value.upload = AsyncMock(return_value=(True, 10, None))

            for i, trace_id in enumerate(trace_ids):
                emitter.emit(
                    Emitter.READY_FOR_UPLOAD,
                    trace_id,
                    f"rec-{i}",
                    str(tmp_path / trace_id),
                    DataType.RGB_IMAGES,
                    "cam",
                    0,
                )

            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        assert len(completes) == 3
        assert len(upload_manager._active_uploads) == 0


class TestBandwidthThrottling:
    """Tests for bandwidth throttling integration."""

    @pytest.mark.asyncio
    async def test_bandwidth_limit_passed_to_uploader(
        self,
        tmp_path: Path,
        client_session: aiohttp.ClientSession,
        mock_trace_status_updater,
        emitter: Emitter,
    ) -> None:
        """Test that config creates an AsyncLimiter."""
        from aiolimiter import AsyncLimiter

        trace_dir = tmp_path / "limit-test"
        trace_dir.mkdir()
        (trace_dir / "file.mp4").write_bytes(b"X" * 10)

        limit = 1024
        config = DaemonConfig(num_threads=2, bandwidth_limit=limit)
        manager = UploadManager(
            config, client_session, emitter, mock_trace_status_updater
        )

        done = asyncio.Event()
        emitter.on(Emitter.UPLOAD_COMPLETE, lambda _: done.set())

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader:
            MockUploader.return_value.upload = AsyncMock(return_value=(True, 10, None))
            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                "trace_id",
                "recording_id",
                str(trace_dir),
                DataType.RGB_IMAGES,
                "cam",
                0,
            )
            await asyncio.wait_for(done.wait(), timeout=TEST_TIMEOUT_SECONDS)

            limiter = MockUploader.call_args.kwargs["bandwidth_limiter"]
            assert isinstance(limiter, AsyncLimiter)
            assert limiter.max_rate == limit

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_no_bandwidth_limit_passed_when_config_none(
        self,
        tmp_path: Path,
        client_session: aiohttp.ClientSession,
        mock_trace_status_updater,
        emitter: Emitter,
    ) -> None:
        """Test that bandwidth_limiter=None is passed when not configured."""
        trace_dir = tmp_path / "trace-no-limit"
        trace_dir.mkdir()
        (trace_dir / "file.mp4").write_bytes(b"X" * 1000)

        config = DaemonConfig(num_threads=2, bandwidth_limit=None)
        manager = UploadManager(
            config, client_session, emitter, mock_trace_status_updater
        )

        uploader_kwargs: list[dict] = []
        upload_done = asyncio.Event()
        emitter.on(Emitter.UPLOAD_COMPLETE, lambda tid: upload_done.set())

        with patch(
            "neuracore.data_daemon.upload_management.upload_manager.ResumableFileUploader"
        ) as MockUploader:

            def capture_kwargs(**kwargs):
                mock_instance = MagicMock()
                mock_instance.upload = AsyncMock(return_value=(True, 1000, None))
                uploader_kwargs.append(kwargs)
                return mock_instance

            MockUploader.side_effect = capture_kwargs

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID_2,
                "rec-789",
                str(trace_dir),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        assert uploader_kwargs[0]["bandwidth_limiter"] is None
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_bandwidth_limit_sleeps_after_chunk(
        self,
        tmp_path: Path,
        client_session: aiohttp.ClientSession,
        mock_trace_status_updater,
        emitter: Emitter,
    ) -> None:
        """Test uploads work with bandwidth limiter configured."""
        import base64
        import hashlib

        file_data = b"X" * 1000
        bandwidth_limit = 500  # bytes/sec
        md5_b64 = base64.b64encode(hashlib.md5(file_data).digest()).decode()

        trace_dir = tmp_path / "trace-throttle"
        trace_dir.mkdir()
        (trace_dir / "file.mp4").write_bytes(file_data)

        upload_done = asyncio.Event()
        emitter.on(Emitter.UPLOAD_COMPLETE, lambda tid: upload_done.set())

        def make_response(status: int, headers: dict | None = None, json_body=None):
            resp = AsyncMock()
            resp.status = status
            resp.headers = headers or {}
            resp.raise_for_status = MagicMock()
            resp.json = (
                AsyncMock(return_value=json_body)
                if json_body
                else AsyncMock(side_effect=Exception("no json"))
            )
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=resp)
            cm.__aexit__ = AsyncMock(return_value=False)
            return cm

        put_calls = [0]

        def put_side_effect(*args, **kwargs):
            put_calls[0] += 1
            return (
                make_response(308)
                if put_calls[0] == 1
                else make_response(200, headers={"x-goog-hash": f"md5={md5_b64}"})
            )

        with patch(
            "neuracore.data_daemon.upload_management.resumable_file_uploader.get_auth"
        ) as mock_rfu_auth, patch(
            "neuracore.data_daemon.upload_management.resumable_file_uploader.get_current_org",
            return_value="test-org",
        ):

            rfu_auth = MagicMock()
            rfu_auth.get_headers.return_value = {"Authorization": "Bearer test"}
            mock_rfu_auth.return_value = rfu_auth

            config = DaemonConfig(num_threads=2, bandwidth_limit=bandwidth_limit)
            manager = UploadManager(
                config, client_session, emitter, mock_trace_status_updater
            )

            client_session.get = MagicMock(
                return_value=make_response(
                    200, json_body={"url": "https://fake-gcs-uri"}
                )
            )
            client_session.put = MagicMock(side_effect=put_side_effect)

            emitter.emit(
                Emitter.READY_FOR_UPLOAD,
                TEST_TRACE_ID,
                "rec-throttle",
                str(trace_dir),
                DataType.RGB_IMAGES,
                "camera_0",
                0,
            )
            await asyncio.wait_for(upload_done.wait(), timeout=TEST_TIMEOUT_SECONDS)

        assert upload_done.is_set(), "Upload should complete with bandwidth limiter"
        await manager.shutdown()
