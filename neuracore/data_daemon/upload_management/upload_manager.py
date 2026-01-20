"""Upload manager for orchestrating file uploads.

This module provides the UploadManager class that manages a thread pool
of upload workers and handles upload lifecycle via events.
"""

import asyncio
import logging
import time

import aiohttp
from neuracore_types import DataType, RecordingDataTraceStatus

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.event_emitter import Emitter, get_emitter
from neuracore.data_daemon.models import TraceErrorCode, TraceStatus, get_content_type
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

        # Subscribe to events
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
            # Cancel all uploads
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

        # Create upload task
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

    async def _upload_single_trace(
        self,
        filepath: str,
        trace_id: str,
        data_type: DataType,
        data_type_name: str,
        recording_id: str,
        bytes_uploaded: int,
    ) -> bool:
        """Upload a single trace file.

        Args:
            filepath: Local filesystem path to file
            trace_id: Trace identifier
            data_type: Data type
            data_type_name: Data type name
            recording_id: Recording identifier
            bytes_uploaded: Starting offset for resume

        Returns:
            True if upload succeeded, False otherwise
        """
        logger.info(f"Starting upload for trace {trace_id}")

        backend_trace_id = await self._register_data_trace(recording_id, data_type)
        if not backend_trace_id:
            logger.error(f"Failed to register backend trace for {trace_id}")
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
                backend_trace_id,
                RecordingDataTraceStatus.UPLOAD_STARTED,
                uploaded_bytes=bytes_uploaded,
            )

            content_type_category = get_content_type(data_type)
            content_type = CONTENT_TYPE_MAPPING[content_type_category]
            cloud_filepath = (
                data_type.value + "/" + data_type_name + "/" + filepath.split("/")[-1]
            )

            cumulative_delta = [0]
            last_progress_update = [time.time()]

            loop = asyncio.get_event_loop()

            def progress_callback(bytes_delta: int) -> None:
                """Called after each chunk with bytes uploaded in that chunk."""
                cumulative_delta[0] += bytes_delta
                total_bytes_uploaded = bytes_uploaded + cumulative_delta[0]
                self._emitter.emit(
                    Emitter.UPLOADED_BYTES, trace_id, total_bytes_uploaded
                )
                # Update backend every 30 seconds
                now = time.time()
                if now - last_progress_update[0] >= 30.0:
                    asyncio.run_coroutine_threadsafe(
                        self._update_data_trace(
                            recording_id,
                            backend_trace_id,
                            RecordingDataTraceStatus.UPLOAD_STARTED,
                            uploaded_bytes=total_bytes_uploaded,
                        ),
                        loop,
                    )
                    last_progress_update[0] = now

            # Create uploader
            uploader = ResumableFileUploader(
                recording_id=recording_id,
                filepath=filepath,
                cloud_filepath=cloud_filepath,
                content_type=content_type,
                client_session=self._client_session,
                bytes_uploaded=bytes_uploaded,
                progress_callback=progress_callback,
            )

            success, total_bytes_uploaded, error_message = await uploader.upload()

            if success:
                await self._update_data_trace(
                    recording_id,
                    backend_trace_id,
                    RecordingDataTraceStatus.UPLOAD_COMPLETE,
                    uploaded_bytes=total_bytes_uploaded,
                    total_bytes=total_bytes_uploaded,
                )
                self._emitter.emit(Emitter.UPLOAD_COMPLETE, trace_id, recording_id)
                logger.info(f"Upload successful for trace {trace_id}")
                return True
            else:
                status = TraceStatus.WRITTEN
                error_code = (
                    TraceErrorCode.NETWORK_ERROR
                    if "Network" in (error_message or "")
                    else TraceErrorCode.UPLOAD_FAILED
                )
                self._emitter.emit(
                    Emitter.UPLOAD_FAILED,
                    trace_id,
                    total_bytes_uploaded,
                    status,
                    error_code,
                    error_message,
                )
                logger.warning(f"Upload failed for trace {trace_id}: {error_message}")
                return False

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
