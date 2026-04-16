"""Upload manager for orchestrating file uploads.

This module provides the UploadManager class that manages a thread pool
of upload workers and handles upload lifecycle via events.
"""

import asyncio
import logging
from pathlib import Path

import aiohttp
from aiolimiter import AsyncLimiter
from neuracore_types import DataType

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.event_emitter import Emitter
from neuracore.data_daemon.models import TraceErrorCode
from neuracore.data_daemon.upload_management.trace_status_updater import (
    TraceStatusUpdater,
)

from .resumable_file_uploader import ResumableFileUploader

logger = logging.getLogger(__name__)

CONTENT_TYPE_MAPPING = {
    "RGB": "video/mp4",
    "JSON": "application/json",
}


class UploadManager:
    """Manages upload operations for the data daemon.

    Uploads traces to cloud storage using a thread pool of workers.
    Uploads are triggered via READY_FOR_UPLOAD events from state manager.
    """

    def __init__(
        self,
        config: DaemonConfig,
        client_session: aiohttp.ClientSession,
        emitter: Emitter,
        trace_status_updater: TraceStatusUpdater,
    ):
        """Initialize the upload manager."""
        self._config = config
        self._active_uploads: dict[str, asyncio.Task] = {}
        self._client_session = client_session
        self._trace_status_updater = trace_status_updater
        self._bandwidth_limiter = (
            AsyncLimiter(config.bandwidth_limit, time_period=1)
            if config.bandwidth_limit
            else None
        )

        self._emitter = emitter
        self._emitter.on(Emitter.READY_FOR_UPLOAD, self._on_ready_for_upload)

        logger.debug("UploadManager initialized")

    async def shutdown(self, wait: bool = True) -> None:
        """Shutdown the upload manager gracefully.

        Args:
            wait: If True, wait for in-flight uploads to complete
        """
        self._emitter.remove_listener(
            Emitter.READY_FOR_UPLOAD, self._on_ready_for_upload
        )
        logger.debug("Shutting down UploadManager...")

        active_tasks = list(self._active_uploads.values())
        if wait and active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        else:
            for task in active_tasks:
                task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)

        logger.debug("UploadManager shutdown complete")

    async def _on_ready_for_upload(
        self,
        trace_id: str,
        recording_id: str,
        filepath: str,
        data_type: DataType,
        data_type_name: str,
        bytes_uploaded: int,
        session_uris: dict[str, str] | None = None,
    ) -> None:
        """Handle READY_FOR_UPLOAD event from state manager.

        Args:
            trace_id: Trace identifier
            recording_id: Recording identifier
            filepath: local file path
            data_type: Data type
            data_type_name: Data type name
            bytes_uploaded: Starting offset for resume
            session_uris: Pre-fetched upload session URIs keyed by cloud filepath.
        """
        active_uploads = self._active_uploads.get(trace_id)
        if active_uploads is not None:
            if active_uploads.done():
                # Defensive cleanup in case callback ordering lags.
                self._active_uploads.pop(trace_id, None)
            else:
                logger.debug(
                    (
                        "Skipping READY_FOR_UPLOAD for trace %s: "
                        "upload already in progress"
                    ),
                    trace_id,
                )
                return

        loop = asyncio.get_running_loop()
        task = loop.create_task(
            self._upload_single_trace(
                filepath,
                trace_id,
                data_type,
                data_type_name,
                recording_id,
                bytes_uploaded,
                session_uris=session_uris,
            )
        )

        self._active_uploads[trace_id] = task

        def _cleanup_active_uploads(
            done_task: asyncio.Task, *, tid: str = trace_id
        ) -> None:
            tracked = self._active_uploads.get(tid)
            if tracked is done_task:
                self._active_uploads.pop(tid, None)

        task.add_done_callback(_cleanup_active_uploads)

    def _find_resume_point(
        self, files: list[Path], bytes_uploaded: int
    ) -> tuple[int, int]:
        """Find which file and offset to resume from.

        Args:
            files: Sorted list of files in the trace directory.
            bytes_uploaded: Cumulative bytes already uploaded.

        Returns:
            Tuple of (file_index, file_offset) to resume from.
        """
        cumulative = 0
        for i, file in enumerate(files):
            file_size = file.stat().st_size
            if cumulative + file_size > bytes_uploaded:
                # This file is partially uploaded
                file_offset = bytes_uploaded - cumulative
                return (i, file_offset)
            cumulative += file_size
        return (len(files), 0)  # All complete

    def _get_content_type_for_file(self, file: Path) -> str:
        """Determine content type from file extension.

        Args:
            file: Path to the file.

        Returns:
            Content type string for the file.
        """
        content_type_map = {
            ".mp4": "video/mp4",
            ".json": "application/json",
        }
        return content_type_map.get(file.suffix.lower(), "application/octet-stream")

    def _emit_upload_failure(
        self,
        trace_id: str,
        bytes_uploaded: int,
        error_message: str,
        error_code: TraceErrorCode = TraceErrorCode.UPLOAD_FAILED,
    ) -> None:
        """Emit an upload failure event.

        Args:
            trace_id: Trace identifier.
            bytes_uploaded: Bytes uploaded before failure.
            error_message: Description of the failure.
            error_code: Error code for the failure.
        """
        self._emitter.emit(
            Emitter.UPLOAD_FAILED,
            trace_id,
            bytes_uploaded,
            error_code,
            error_message,
        )

    def _validate_trace_directory(
        self, trace_dir_path: str
    ) -> tuple[list[Path] | None, str | None]:
        """Validate trace directory and return files to upload.

        Args:
            trace_dir_path: Path to the trace directory.

        Returns:
            Tuple of (files, error_message). If validation fails, files is None
            and error_message describes the issue.
        """
        trace_dir = Path(trace_dir_path)

        if not trace_dir.exists():
            return None, f"Directory not found: {trace_dir_path}"

        if not trace_dir.is_dir():
            return None, f"Path is not a directory: {trace_dir_path}"

        files = sorted([file for file in trace_dir.iterdir() if file.is_file()])

        if not files:
            return None, f"Empty directory: {trace_dir_path}"

        return files, None

    async def _upload_file(
        self,
        file: Path,
        cloud_filepath: str,
        recording_id: str,
        trace_id: str,
        file_bytes_uploaded: int,
        session_uri: str | None = None,
    ) -> tuple[bool, int, str | None]:
        """Upload a single file using ResumableFileUploader.

        Args:
            file: Path to the file to upload.
            cloud_filepath: Destination path in cloud storage.
            recording_id: Recording identifier.
            trace_id: Trace identifier.
            file_bytes_uploaded: Bytes already uploaded for this file (for resume).
            session_uri: Pre-fetched resumable upload session URI.

        Returns:
            Tuple of (success, total_bytes, error_message).
        """
        content_type = self._get_content_type_for_file(file)
        uploader = ResumableFileUploader(
            recording_id=recording_id,
            trace_id=trace_id,
            filepath=str(file),
            cloud_filepath=cloud_filepath,
            content_type=content_type,
            client_session=self._client_session,
            trace_status_updater=self._trace_status_updater,
            emitter=self._emitter,
            bytes_uploaded=file_bytes_uploaded,
            bandwidth_limiter=self._bandwidth_limiter,
            session_uri=session_uri,
        )
        return await uploader.upload()

    async def _upload_single_trace(
        self,
        trace_dir_path: str,
        trace_id: str,
        data_type: DataType,
        data_type_name: str,
        recording_id: str,
        bytes_uploaded: int,
        session_uris: dict[str, str] | None = None,
    ) -> bool:
        """Upload all files in a trace directory.

        Args:
            trace_dir_path: Local filesystem path to trace directory.
            trace_id: Trace identifier.
            data_type: Data type.
            data_type_name: Data type name.
            recording_id: Recording identifier.
            bytes_uploaded: Cumulative bytes already uploaded (for resume).
            session_uris: Pre-fetched upload session URIs keyed by cloud filepath.

        Returns:
            True if all files uploaded successfully, False otherwise.
        """
        # Validate trace data exists at path
        files, validation_error = self._validate_trace_directory(trace_dir_path)
        if validation_error or files is None:
            error_msg = validation_error or "No files found in trace directory"
            self._emit_upload_failure(
                trace_id=trace_id,
                bytes_uploaded=bytes_uploaded,
                error_message=error_msg,
            )
            return False

        async def upload_files() -> bool:
            try:

                self._emitter.emit(Emitter.UPLOAD_STARTED, trace_id)

                start_file_idx, file_offset = self._find_resume_point(
                    files, bytes_uploaded
                )
                cumulative_bytes = sum(
                    file.stat().st_size for file in files[:start_file_idx]
                )

                for file_idx, file in enumerate(
                    files[start_file_idx:], start=start_file_idx
                ):
                    cloud_filepath = f"{data_type.value}/{data_type_name}/{file.name}"
                    file_bytes_uploaded = (
                        file_offset if file_idx == start_file_idx else 0
                    )

                    file_session_uri = (
                        session_uris.get(cloud_filepath) if session_uris else None
                    )
                    success, file_total_bytes, error_message = await self._upload_file(
                        file=file,
                        cloud_filepath=cloud_filepath,
                        recording_id=recording_id,
                        trace_id=trace_id,
                        file_bytes_uploaded=file_bytes_uploaded,
                        session_uri=file_session_uri,
                    )

                    if not success:
                        failed_bytes = cumulative_bytes + file_total_bytes
                        error_code = (
                            TraceErrorCode.NETWORK_ERROR
                            if "Network" in (error_message or "")
                            else TraceErrorCode.UPLOAD_FAILED
                        )
                        self._emit_upload_failure(
                            trace_id=trace_id,
                            bytes_uploaded=failed_bytes,
                            error_message=error_message or "Upload failed",
                            error_code=error_code,
                        )
                        logger.warning(
                            f"Upload failed for trace {trace_id} file {file.name}: "
                            f"{error_message}"
                        )
                        return False

                    cumulative_bytes += file.stat().st_size

                updated_trace = await self._trace_status_updater.update_trace_completed(
                    recording_id=recording_id,
                    trace_id=trace_id,
                    total_bytes=cumulative_bytes,
                    wait_for_completion=True,
                )
                if not updated_trace:
                    logger.warning(
                        f"Failed to mark trace {trace_id} as complete on backend, "
                        "will retry"
                    )
                    self._emit_upload_failure(
                        trace_id=trace_id,
                        bytes_uploaded=cumulative_bytes,
                        error_message="Failed to update trace status to complete",
                        error_code=TraceErrorCode.NETWORK_ERROR,
                    )
                    return False

                self._emitter.emit(Emitter.UPLOAD_COMPLETE, trace_id)
                return True

            except FileNotFoundError as e:
                logger.error(f"File not found for trace {trace_id}: {e}")
                self._emit_upload_failure(
                    trace_id=trace_id,
                    bytes_uploaded=bytes_uploaded,
                    error_message=f"File not found: {e}",
                )
                return False

            except ValueError as e:
                logger.error(f"Invalid path for trace {trace_id}: {e}")
                self._emit_upload_failure(
                    trace_id=trace_id,
                    bytes_uploaded=bytes_uploaded,
                    error_message=f"Invalid path: {e}",
                )
                return False

            except Exception as e:
                error_detail = f"{type(e).__name__}: {e}"
                logger.error(
                    f"Unexpected error uploading trace {trace_id}: {error_detail}",
                    exc_info=True,
                )
                self._emit_upload_failure(
                    trace_id=trace_id,
                    bytes_uploaded=bytes_uploaded,
                    error_message=f"Upload error: {error_detail}",
                )
                return False

        return await upload_files()
