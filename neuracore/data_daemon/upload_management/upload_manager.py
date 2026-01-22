"""Upload manager for orchestrating file uploads.

This module provides the UploadManager class that manages a thread pool
of upload workers and handles upload lifecycle via events.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from neuracore_types import DataType, RecordingDataTraceStatus

from neuracore.data_daemon.event_emitter import Emitter, emitter
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

    def __init__(self, num_threads: int = 4):
        """Initialize the upload manager.

        Args:
            num_threads: Number of concurrent upload threads
        """
        # Threading
        self._num_threads = num_threads
        self._executor = ThreadPoolExecutor(
            max_workers=self._num_threads,
            thread_name_prefix="uploader",
        )

        # Subscribe to events
        emitter.on(Emitter.READY_FOR_UPLOAD, self._on_ready_for_upload)

        logger.info(f"UploadManager initialized with {self._num_threads} workers")

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the upload manager gracefully.

        Args:
            wait: If True, wait for in-flight uploads to complete
        """
        emitter.remove_listener(Emitter.READY_FOR_UPLOAD, self._on_ready_for_upload)
        logger.info("Shutting down UploadManager...")
        self._executor.shutdown(wait=wait, cancel_futures=False)
        logger.info("UploadManager shutdown complete")

    def _on_ready_for_upload(
        self,
        filepath: str,
        trace_id: str,
        data_type: DataType,
        data_type_name: str,
        recording_id: str,
        bytes_uploaded: int,
    ) -> None:
        """Handle READY_FOR_UPLOAD event from state manager.

        Queues the trace for upload in the thread pool.

        Args:
            filepath: local file path
            trace_id: Trace identifier
            data_type: Data type
            data_type_name: Data type name
            recording_id: Recording identifier
            bytes_uploaded: Starting offset for resume
        """
        logger.info(f"Received READY_FOR_UPLOAD for trace {trace_id}")

        # Submit to thread pool
        self._executor.submit(
            self._upload_single_trace,
            filepath,
            trace_id,
            data_type,
            data_type_name,
            recording_id,
            bytes_uploaded,
        )

    def _upload_single_trace(
        self,
        filepath: str,
        trace_id: str,
        data_type: DataType,
        data_type_name: str,
        recording_id: str,
        bytes_uploaded: int,
    ) -> bool:
        """Upload a single trace file.

        Creates a ResumableFileUploader and performs the upload. Emits
        events based on success or failure.

        Args:
            filepath: Local filesystem path to file
            trace_id: Trace identifier
            data_type: Data type
            recording_id: Recording identifier
            bytes_uploaded: Starting offset for resume

        Returns:
            True if upload succeeded, False otherwise
        """
        logger.info(f"Starting upload for trace {trace_id}")
        backend_trace_id = self._register_data_trace(recording_id, data_type)
        if not backend_trace_id:
            logger.error(f"Failed to register backend trace for {trace_id}")
            emitter.emit(
                Emitter.UPLOAD_FAILED,
                trace_id,
                bytes_uploaded,
                TraceStatus.FAILED,
                TraceErrorCode.UPLOAD_FAILED,
                "Failed to register trace with backend",
            )
            return False

        try:
            self._update_data_trace(
                recording_id,
                backend_trace_id,
                RecordingDataTraceStatus.UPLOAD_STARTED,
                uploaded_bytes=bytes_uploaded,
            )
            # content type from data type
            content_type_category = get_content_type(data_type)
            content_type = CONTENT_TYPE_MAPPING[content_type_category]
            cloud_filepath = (
                data_type.value + "/" + data_type_name + "/" + filepath.split("/")[-1]
            )
            # Progress callback to emit uploaded bytes
            cumulative_delta = [0]
            last_progress_update = [time.time()]

            def progress_callback(bytes_delta: int) -> None:
                """Called after each chunk with bytes uploaded in that chunk.

                Emits total bytes uploaded for the trace.
                """
                cumulative_delta[0] += bytes_delta
                total_bytes_uploaded = bytes_uploaded + cumulative_delta[0]
                emitter.emit(Emitter.UPLOADED_BYTES, trace_id, total_bytes_uploaded)

                # Update backend every 30 seconds
                now = time.time()
                if now - last_progress_update[0] >= 30.0:
                    self._update_data_trace(
                        recording_id,
                        backend_trace_id,
                        RecordingDataTraceStatus.UPLOAD_STARTED,
                        uploaded_bytes=total_bytes_uploaded,
                    )
                    last_progress_update[0] = now

            # Create uploader
            uploader = ResumableFileUploader(
                recording_id=recording_id,
                filepath=filepath,
                cloud_filepath=cloud_filepath,
                content_type=content_type,
                bytes_uploaded=bytes_uploaded,
                progress_callback=progress_callback,
            )

            # Perform upload
            success, total_bytes_uploaded, error_message = uploader.upload()

            if success:
                self._update_data_trace(
                    recording_id,
                    backend_trace_id,
                    RecordingDataTraceStatus.UPLOAD_COMPLETE,
                    uploaded_bytes=total_bytes_uploaded,
                    total_bytes=total_bytes_uploaded,
                )
                emitter.emit(Emitter.UPLOAD_COMPLETE, trace_id, recording_id)
                logger.info(f"Upload successful for trace {trace_id}")
                return True
            else:
                # Upload failed - emit failure event
                status = TraceStatus.WRITTEN
                error_code = (
                    TraceErrorCode.NETWORK_ERROR
                    if "Network" in (error_message or "")
                    else TraceErrorCode.UPLOAD_FAILED
                )
                emitter.emit(
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
            emitter.emit(
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
            emitter.emit(
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
            emitter.emit(
                Emitter.UPLOAD_FAILED,
                trace_id,
                bytes_uploaded,
                TraceStatus.FAILED,
                TraceErrorCode.UPLOAD_FAILED,
                f"Upload error: {e}",
            )
            return False
