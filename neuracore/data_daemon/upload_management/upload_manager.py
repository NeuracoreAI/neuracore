"""Upload manager for orchestrating file uploads.

This module provides the UploadManager class that manages a thread pool
of upload workers and handles upload lifecycle via events.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from aiolimiter import AsyncLimiter
from neuracore_types import DataType, RecordingDataTraceStatus

from neuracore.core.auth import get_auth
from neuracore.core.config.get_current_org import get_current_org
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.const import API_URL
from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import TraceErrorCode

from .resumable_file_uploader import ResumableFileUploader

logger = logging.getLogger(__name__)

CONTENT_TYPE_MAPPING = {
    "RGB": "video/mp4",
    "JSON": "application/json",
}


@dataclass
class _ProgressUpdateState:
    """Mutable progress timing state shared across file callbacks."""

    last_progress_update: float


class UploadManager:
    """Manages upload operations for the data daemon.

    Uploads traces to cloud storage using a thread pool of workers.
    Uploads are triggered via READY_FOR_UPLOAD events from state manager.
    """

    def __init__(self, config: DaemonConfig, client_session: aiohttp.ClientSession):
        """Initialize the upload manager."""
        self._config = config
        self._active_uploads: dict[str, asyncio.Task] = {}
        self._client_session = client_session
        self._bandwidth_limiter = (
            AsyncLimiter(config.bandwidth_limit, time_period=1)
            if config.bandwidth_limit
            else None
        )
        max_concurrent_uploads = config.max_concurrent_uploads
        if max_concurrent_uploads is not None and max_concurrent_uploads > 0:
            self._upload_semaphore: asyncio.Semaphore | None = asyncio.Semaphore(
                max_concurrent_uploads
            )
            logger.info(
                "UploadManager concurrency limit enabled (max_concurrent_uploads=%d)",
                max_concurrent_uploads,
            )
        else:
            self._upload_semaphore = None

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

        active_tasks = list(self._active_uploads.values())
        if wait and active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        else:
            for task in active_tasks:
                task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)

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
        existing = self._active_uploads.get(trace_id)
        if existing is not None:
            if existing.done():
                # Defensive cleanup in case callback ordering lags.
                self._active_uploads.pop(trace_id, None)
            else:
                logger.debug(
                    "Skipping READY_FOR_UPLOAD for trace %s: upload already in progress",
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
            )
        )

        self._active_uploads[trace_id] = task

        def _on_done(done_task: asyncio.Task, *, tid: str = trace_id) -> None:
            tracked = self._active_uploads.get(tid)
            if tracked is done_task:
                self._active_uploads.pop(tid, None)

        task.add_done_callback(_on_done)

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
        progress_state: _ProgressUpdateState,
    ) -> Callable[[int], Awaitable[None]]:
        """Create a progress callback for tracking upload progress.

        Args:
            trace_id: Trace identifier.
            recording_id: Recording identifier.
            base_bytes: Cumulative bytes uploaded before this file.
            progress_state: Shared upload progress timing state.

        Returns:
            Async callback function that handles progress updates.
        """
        cumulative_delta = 0

        async def progress_callback(bytes_delta: int) -> None:
            nonlocal cumulative_delta
            cumulative_delta += bytes_delta
            total_bytes_uploaded = base_bytes + cumulative_delta
            self._emitter.emit(Emitter.UPLOADED_BYTES, trace_id, total_bytes_uploaded)
            now = time.time()
            if now - progress_state.last_progress_update >= 30.0:
                await self._update_backend_trace_progress(
                    recording_id,
                    trace_id,
                    uploaded_bytes=total_bytes_uploaded,
                )
                progress_state.last_progress_update = now

        return progress_callback

    async def _mark_backend_trace_status_as_upload_started(
        self,
        recording_id: str,
        trace_id: str,
        *,
        uploaded_bytes: int,
        total_bytes: int,
    ) -> bool:
        """Mark backend trace status as UPLOAD_STARTED with progress counters."""
        return await self._update_backend_trace_record(
            recording_id,
            trace_id,
            updates={
                "status": RecordingDataTraceStatus.UPLOAD_STARTED,
                "uploaded_bytes": uploaded_bytes,
                "total_bytes": total_bytes,
            },
        )

    async def _mark_backend_trace_status_as_upload_complete(
        self,
        recording_id: str,
        trace_id: str,
        *,
        uploaded_bytes: int,
        total_bytes: int,
    ) -> bool:
        """Mark backend trace status as UPLOAD_COMPLETE with final counters."""
        return await self._update_backend_trace_record(
            recording_id,
            trace_id,
            updates={
                "status": RecordingDataTraceStatus.UPLOAD_COMPLETE,
                "uploaded_bytes": uploaded_bytes,
                "total_bytes": total_bytes,
            },
        )

    async def _update_backend_trace_progress(
        self,
        recording_id: str,
        trace_id: str,
        *,
        uploaded_bytes: int,
    ) -> bool:
        """Update backend upload byte counters without changing status."""
        return await self._update_backend_trace_record(
            recording_id,
            trace_id,
            updates={
                "uploaded_bytes": uploaded_bytes,
            },
        )

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

        total_bytes = sum(file.stat().st_size for file in files)

        async def upload_files() -> bool:
            try:
                await self._mark_backend_trace_status_as_upload_started(
                    recording_id,
                    trace_id,
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
                progress_state = _ProgressUpdateState(last_progress_update=time.time())

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
                        progress_state,
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

                updated_trace = await self._mark_backend_trace_status_as_upload_complete(
                    recording_id,
                    trace_id,
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

        if self._upload_semaphore is None:
            return await upload_files()
        async with self._upload_semaphore:
            return await upload_files()


    async def _update_backend_trace_record(
        self,
        recording_id: str,
        trace_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update fields of a backend DataTrace.

        Args:
            recording_id: The recording ID
            trace_id: The trace ID
            updates: JSON payload fields to update on backend DataTrace

        Returns:
            True if update succeeded, False otherwise
        """
        if not trace_id:
            logger.warning("No trace ID provided for update")
            return False
        if not updates:
            logger.warning("No data trace updates provided for trace_id=%s", trace_id)
            return False

        try:
            loop = asyncio.get_running_loop()
            auth = get_auth()
            org_id, headers = await asyncio.gather(
                loop.run_in_executor(None, get_current_org),
                loop.run_in_executor(None, auth.get_headers),
            )

            for attempt in range(2):
                async with self._client_session.put(
                    f"{API_URL}/org/{org_id}/recording/{recording_id}/traces/{trace_id}",
                    json=updates,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    assert isinstance(response, aiohttp.ClientResponse)
                    if response.status == 401 and attempt == 0:
                        logger.info("Access token expired, refreshing token")
                        await loop.run_in_executor(None, auth.login)
                        headers = await loop.run_in_executor(None, auth.get_headers)
                        continue

                    if response.status >= 400:
                        error = await response.text()
                        logger.warning(
                            f"Failed to update data trace: "
                            f"HTTP {response.status}: {error}"
                        )
                        return False

                    logger.debug(
                        "Updated trace %s with backend fields: %s",
                        trace_id,
                        list(updates.keys()),
                    )
                    return True

            return False

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("Failed to update data trace: %s: %s", type(e).__name__, e)
            return False
