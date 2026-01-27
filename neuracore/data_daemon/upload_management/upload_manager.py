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
from neuracore.data_daemon.models import TraceErrorCode, TraceStatus
from neuracore.data_daemon.upload_management.trace_manager import TraceManager

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
        logger.info(f"Received READY_FOR_UPLOAD for trace {trace_id}")

        task = asyncio.create_task(
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
        ext = file.suffix.lower()
        if ext == ".mp4":
            return "video/mp4"
        elif ext == ".json":
            return "application/json"
        else:
            return "application/octet-stream"

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
        logger.info(f"Starting upload for trace {trace_id}")

        trace_dir = Path(trace_dir_path)

        if not trace_dir.exists():
            logger.error(f"Trace directory not found: {trace_dir_path}")
            self._emitter.emit(
                Emitter.UPLOAD_FAILED,
                trace_id,
                bytes_uploaded,
                TraceStatus.FAILED,
                TraceErrorCode.UPLOAD_FAILED,
                f"Directory not found: {trace_dir_path}",
            )
            return False

        if not trace_dir.is_dir():
            logger.error(f"Path is not a directory: {trace_dir_path}")
            self._emitter.emit(
                Emitter.UPLOAD_FAILED,
                trace_id,
                bytes_uploaded,
                TraceStatus.FAILED,
                TraceErrorCode.UPLOAD_FAILED,
                f"Path is not a directory: {trace_dir_path}",
            )
            return False

        files = sorted([f for f in trace_dir.iterdir() if f.is_file()])

        if not files:
            logger.error(f"Empty directory: {trace_dir_path}")
            self._emitter.emit(
                Emitter.UPLOAD_FAILED,
                trace_id,
                bytes_uploaded,
                TraceStatus.FAILED,
                TraceErrorCode.UPLOAD_FAILED,
                f"Empty directory: {trace_dir_path}",
            )
            return False

        logger.info(f"Found {len(files)} files to upload for trace {trace_id}")

        registered = await self._register_data_trace(
            recording_id, data_type, UUID(trace_id)
        )
        if not registered:
            logger.error(f"Failed to register trace {trace_id} with backend")
            self._emitter.emit(
                Emitter.UPLOAD_FAILED,
                trace_id,
                bytes_uploaded,
                TraceStatus.FAILED,
                TraceErrorCode.UPLOAD_FAILED,
                "Failed to register trace with backend",
            )
            return False

        try:
            await self._update_data_trace(
                recording_id,
                trace_id,
                RecordingDataTraceStatus.UPLOAD_STARTED,
                uploaded_bytes=bytes_uploaded,
            )

            start_file_idx, file_offset = self._find_resume_point(files, bytes_uploaded)

            cumulative_bytes = sum(f.stat().st_size for f in files[:start_file_idx])
            last_progress_update = [time.time()]

            for i, file in enumerate(files[start_file_idx:], start=start_file_idx):
                cloud_filepath = f"{data_type.value}/{data_type_name}/{file.name}"
                content_type = self._get_content_type_for_file(file)

                file_bytes_uploaded = file_offset if i == start_file_idx else 0

                file_cumulative_bytes = cumulative_bytes

                def make_progress_callback(
                    base_bytes: int,
                ) -> "Callable[[int], Awaitable[None]]":
                    """Create a progress callback with captured base_bytes."""
                    cumulative_delta = [0]

                    async def progress_callback(bytes_delta: int) -> None:
                        """Called after each chunk with bytes uploaded in that chunk."""
                        nonlocal last_progress_update
                        cumulative_delta[0] += bytes_delta
                        total_bytes_uploaded = base_bytes + cumulative_delta[0]
                        self._emitter.emit(
                            Emitter.UPLOADED_BYTES, trace_id, total_bytes_uploaded
                        )
                        now = time.time()
                        if now - last_progress_update[0] >= 30.0:
                            await self._update_data_trace(
                                recording_id,
                                trace_id,
                                RecordingDataTraceStatus.UPLOAD_STARTED,
                                uploaded_bytes=total_bytes_uploaded,
                            )
                            last_progress_update[0] = now

                    return progress_callback

                progress_callback = make_progress_callback(file_cumulative_bytes)

                logger.info(
                    f"Uploading file {i + 1}/{len(files)}: {file.name} "
                    f"(offset={file_bytes_uploaded})"
                )

                uploader = ResumableFileUploader(
                    recording_id=recording_id,
                    filepath=str(file),
                    cloud_filepath=cloud_filepath,
                    content_type=content_type,
                    client_session=self._client_session,
                    bytes_uploaded=file_bytes_uploaded,
                    progress_callback=progress_callback,
                )

                success, file_total_bytes, error_message = await uploader.upload()

                if not success:
                    failed_bytes = cumulative_bytes + file_total_bytes
                    status = TraceStatus.WRITTEN
                    error_code = (
                        TraceErrorCode.NETWORK_ERROR
                        if "Network" in (error_message or "")
                        else TraceErrorCode.UPLOAD_FAILED
                    )
                    self._emitter.emit(
                        Emitter.UPLOAD_FAILED,
                        trace_id,
                        failed_bytes,
                        status,
                        error_code,
                        error_message,
                    )
                    logger.warning(
                        f"Upload failed for trace {trace_id} file {file.name}: "
                        f"{error_message}"
                    )
                    return False

                cumulative_bytes += file.stat().st_size

            await self._update_data_trace(
                recording_id,
                trace_id,
                RecordingDataTraceStatus.UPLOAD_COMPLETE,
                uploaded_bytes=cumulative_bytes,
                total_bytes=cumulative_bytes,
            )
            self._emitter.emit(Emitter.UPLOAD_COMPLETE, trace_id)
            logger.info(f"Upload successful for trace {trace_id}")
            return True

        except FileNotFoundError as e:
            logger.error(f"File not found for trace {trace_id}: {e}")
            self._emitter.emit(
                Emitter.UPLOAD_FAILED,
                trace_id,
                bytes_uploaded,
                TraceStatus.FAILED,
                TraceErrorCode.UPLOAD_FAILED,
                f"File not found: {e}",
            )
            return False

        except ValueError as e:
            logger.error(f"Invalid path for trace {trace_id}: {e}")
            self._emitter.emit(
                Emitter.UPLOAD_FAILED,
                trace_id,
                bytes_uploaded,
                TraceStatus.FAILED,
                TraceErrorCode.UPLOAD_FAILED,
                f"Invalid path: {e}",
            )
            return False

        except Exception as e:
            logger.error(
                f"Unexpected error uploading trace {trace_id}: {e}", exc_info=True
            )
            self._emitter.emit(
                Emitter.UPLOAD_FAILED,
                trace_id,
                bytes_uploaded,
                TraceStatus.FAILED,
                TraceErrorCode.UPLOAD_FAILED,
                f"Upload error: {e}",
            )
            return False
