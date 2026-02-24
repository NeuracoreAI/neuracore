"""Upload manager for orchestrating file uploads.

This module provides the UploadManager class that manages a thread pool
of upload workers and handles upload lifecycle via events.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID

import aiohttp
from neuracore_types import DataType, RecordingDataTraceStatus

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import TraceErrorCode
from neuracore.data_daemon.upload_management.trace_manager import TraceManager

from .bandwidth_limiter import BandwidthLimiter
from .resumable_file_uploader import ResumableFileUploader

logger = logging.getLogger(__name__)

CONTENT_TYPE_MAPPING = {
    "RGB": "video/mp4",
    "JSON": "application/json",
}


class UploadManager(TraceManager):
    """Manages upload operations for the data daemon.

    Uploads traces to cloud storage using a thread pool of workers.
    Uploads are triggered via READY_FOR_UPLOAD events from state manager.
    """

    def __init__(self, config: DaemonConfig, client_session: aiohttp.ClientSession):
        """Initialize the upload manager."""
        self._config = config
        self._active_uploads: set[asyncio.Task] = set()
        self._client_session = client_session
        self._bandwidth_limiter = None
        super().__init__(client_session)

        self._emitter = get_emitter()
        self._emitter.on(Emitter.READY_FOR_UPLOAD, self._on_ready_for_upload)

        logger.info("UploadManager initialized")

    async def shutdown(self, wait: bool = True) -> None:
        """Shutdown the upload manager gracefully.

        Args:
            wait: If True, wait for in-flight uploads to complete
        """
        self._emitter.remove_listener(
            Emitter.READY_FOR_UPLOAD, self._on_ready_for_upload
        )
        logger.info("Shutting down UploadManager...")

        if wait and self._active_uploads:
            logger.info(
                f"Waiting for {len(self._active_uploads)} uploads to complete..."
            )
            await asyncio.gather(*self._active_uploads, return_exceptions=True)
        else:
            for task in self._active_uploads:
                task.cancel()
            if self._active_uploads:
                await asyncio.gather(*self._active_uploads, return_exceptions=True)

        logger.info("UploadManager shutdown complete")

    async def _on_ready_for_upload(
        self,
        trace_id: str,
        recording_id: str,
        filepath: str,
        data_type: DataType,
        data_type_name: str,
        bytes_uploaded: int,
    ) -> None:
        """Handle READY_FOR_UPLOAD event from state manager.

        Args:
            trace_id: Trace identifier
            recording_id: Recording identifier
            filepath: local file path
            data_type: Data type
            data_type_name: Data type name
            bytes_uploaded: Starting offset for resume
        """
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            self._upload_single_trace(
                filepath,
                trace_id,
                data_type,
                data_type_name,
                recording_id,
                bytes_uploaded,
            )
        )

        self._active_uploads.add(task)
        task.add_done_callback(self._active_uploads.discard)

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

    def _make_progress_callback(
        self,
        trace_id: str,
        recording_id: str,
        base_bytes: int,
        total_bytes: int,
        last_progress_update: list[float],
    ) -> Callable[[int], Awaitable[None]]:
        """Create a progress callback for tracking upload progress.

        Args:
            trace_id: Trace identifier.
            recording_id: Recording identifier.
            base_bytes: Cumulative bytes uploaded before this file.
            last_progress_update: Mutable list containing last update timestamp.

        Returns:
            Async callback function that handles progress updates.
        """
        cumulative_delta = [0]

        async def progress_callback(bytes_delta: int) -> None:
            cumulative_delta[0] += bytes_delta
            total_bytes_uploaded = base_bytes + cumulative_delta[0]
            self._emitter.emit(Emitter.UPLOADED_BYTES, trace_id, total_bytes_uploaded)
            now = time.time()
            if now - last_progress_update[0] >= 30.0:
                await self._update_data_trace(
                    recording_id,
                    trace_id,
                    RecordingDataTraceStatus.UPLOAD_STARTED,
                    uploaded_bytes=total_bytes_uploaded,
                    total_bytes=total_bytes,
                )
                last_progress_update[0] = now

        return progress_callback

    async def _upload_file(
        self,
        file: Path,
        cloud_filepath: str,
        recording_id: str,
        file_bytes_uploaded: int,
        progress_callback: Callable[[int], Awaitable[None]],
    ) -> tuple[bool, int, str | None]:
        """Upload a single file using ResumableFileUploader.

        Args:
            file: Path to the file to upload.
            cloud_filepath: Destination path in cloud storage.
            recording_id: Recording identifier.
            file_bytes_uploaded: Bytes already uploaded for this file (for resume).
            progress_callback: Callback for progress updates.

        Returns:
            Tuple of (success, total_bytes, error_message).
        """
        content_type = self._get_content_type_for_file(file)
        uploader = ResumableFileUploader(
            recording_id=recording_id,
            filepath=str(file),
            cloud_filepath=cloud_filepath,
            content_type=content_type,
            client_session=self._client_session,
            bytes_uploaded=file_bytes_uploaded,
            progress_callback=progress_callback,
            bandwidth_limiter=self._bandwidth_limiter,
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
    ) -> bool:
        """Upload all files in a trace directory.

        Args:
            trace_dir_path: Local filesystem path to trace directory.
            trace_id: Trace identifier.
            data_type: Data type.
            data_type_name: Data type name.
            recording_id: Recording identifier.
            bytes_uploaded: Cumulative bytes already uploaded (for resume).

        Returns:
            True if all files uploaded successfully, False otherwise.
        """
        files, validation_error = self._validate_trace_directory(trace_dir_path)
        if validation_error or files is None:
            error_msg = validation_error or "No files found in trace directory"
            self._emit_upload_failure(
                trace_id=trace_id,
                bytes_uploaded=bytes_uploaded,
                error_message=error_msg,
            )
            return False

        total_bytes = sum(file.stat().st_size for file in files)

        registered = await self._register_data_trace(
            recording_id, data_type, UUID(trace_id)
        )
        if not registered:
            logger.error(f"Failed to register trace {trace_id} with backend")
            self._emit_upload_failure(
                trace_id=trace_id,
                bytes_uploaded=bytes_uploaded,
                error_message="Failed to register trace with backend",
                error_code=TraceErrorCode.NETWORK_ERROR,
            )
            return False

        async def upload_files() -> bool:
            try:
                await self._update_data_trace(
                    recording_id,
                    trace_id,
                    RecordingDataTraceStatus.UPLOAD_STARTED,
                    uploaded_bytes=bytes_uploaded,
                    total_bytes=total_bytes,
                )
                self._emitter.emit(Emitter.UPLOAD_STARTED, trace_id)

                start_file_idx, file_offset = self._find_resume_point(
                    files, bytes_uploaded
                )
                cumulative_bytes = sum(
                    file.stat().st_size for file in files[:start_file_idx]
                )
                last_progress_update = [time.time()]

                for file_idx, file in enumerate(
                    files[start_file_idx:], start=start_file_idx
                ):
                    cloud_filepath = f"{data_type.value}/{data_type_name}/{file.name}"
                    file_bytes_uploaded = (
                        file_offset if file_idx == start_file_idx else 0
                    )

                    progress_callback = self._make_progress_callback(
                        trace_id,
                        recording_id,
                        cumulative_bytes,
                        total_bytes,
                        last_progress_update,
                    )

                    logger.info(
                        f"Uploading file {file_idx + 1}/{len(files)}: {file.name} "
                        f"(offset={file_bytes_uploaded})"
                    )

                    success, file_total_bytes, error_message = await self._upload_file(
                        file,
                        cloud_filepath,
                        recording_id,
                        file_bytes_uploaded,
                        progress_callback,
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

                updated_trace = await self._update_data_trace(
                    recording_id,
                    trace_id,
                    RecordingDataTraceStatus.UPLOAD_COMPLETE,
                    uploaded_bytes=cumulative_bytes,
                    total_bytes=cumulative_bytes,
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
                logger.info(f"Upload successful for trace {trace_id}")
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
                logger.error(
                    f"Unexpected error uploading trace {trace_id}: {e}", exc_info=True
                )
                self._emit_upload_failure(
                    trace_id=trace_id,
                    bytes_uploaded=bytes_uploaded,
                    error_message=f"Upload error: {e}",
                )
                return False

        return await upload_files()