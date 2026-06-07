"""Data stream classes for recording and uploading robot sensor data.

This module provides abstract and concrete data stream implementations for
recording various types of robot sensor data including JSON events, RGB video,
and depth data. Streams are lightweight local objects: instead of owning their
own sockets and threads, they register a :class:`StreamSession` with the robot's
:class:`RobotProducerCoordinator` and enqueue recording payloads into it.
"""

import json
import logging
import struct
from abc import ABC
from dataclasses import dataclass

import numpy as np
from neuracore_types import CameraData, DataType, NCData

from neuracore.data_daemon.communications_management.producer import (
    RobotProducerCoordinator,
    StreamPayload,
    StreamSession,
)

logger = logging.getLogger(__name__)


@dataclass
class DataRecordingContext:
    """Context information needed for recording data to the daemon.

    Contains all identifiers needed to associate recorded data with the
    correct robot, dataset, and recording session.
    """

    recording_id: str
    robot_id: str | None
    robot_name: str
    robot_instance: int
    dataset_id: str | None
    dataset_name: str | None


class DataStream(ABC):
    """Base class for data streams.

    Provides common functionality for managing recording state and the latest
    logged data across different types of sensor data streams. Transport is owned
    by the robot-scoped :class:`RobotProducerCoordinator`; each stream registers a
    :class:`StreamSession` while recording and enqueues payloads into it.
    """

    def __init__(self, data_type: DataType, stream_name: str) -> None:
        """Initialize the data stream.

        Args:
            data_type: The type of data this stream handles.
            stream_name: Unique name for this stream within its data type.

        Note:
            This must be kept lightweight and not perform any blocking operations.
        """
        self._recording = False
        self._context: DataRecordingContext | None = None
        self._latest_data: NCData | None = None
        self._data_type = data_type
        self._stream_name = stream_name
        self._coordinator: RobotProducerCoordinator | None = None
        self._session: StreamSession | None = None

    @property
    def data_type(self) -> DataType:
        """Get the data type of this stream."""
        return self._data_type

    def start_recording(
        self,
        context: DataRecordingContext,
        coordinator: RobotProducerCoordinator,
    ) -> None:
        """Start recording by registering a session with the robot coordinator.

        Args:
            context: Recording context identifiers for the recording session,
                robot, and dataset.
            coordinator: The robot-scoped producer coordinator that owns the
                socket, sender, heartbeat, and video transports.
        """
        if self.is_recording():
            if (
                self._context is not None
                and self._context.recording_id == context.recording_id
            ):
                return
            self.mark_recording_stopped()
        self._recording = True
        self._context = context
        self._coordinator = coordinator
        self._session = coordinator.register_stream_session(
            stream_name=self._stream_name,
            data_type=self._data_type,
            recording_id=context.recording_id,
            robot_instance=context.robot_instance,
            robot_id=context.robot_id,
            robot_name=context.robot_name,
            dataset_id=context.dataset_id,
            dataset_name=context.dataset_name,
        )

    def mark_recording_stopped(self) -> None:
        """Mark this stream as no longer recording.

        Transport teardown (TRACE_END, drains, stop cutoff) is owned by the
        coordinator; this only clears the stream's local recording state.
        """
        self._recording = False
        self._context = None
        self._coordinator = None
        self._session = None

    def is_recording(self) -> bool:
        """Check if recording is active.

        Returns:
            bool: True if currently recording, False otherwise
        """
        return self._recording

    def get_latest_data(self) -> NCData | None:
        """Get the latest data from the stream.

        Returns:
            Optional[NCData]: The most recently logged data item
        """
        return self._latest_data

    def get_stream_session(self) -> StreamSession | None:
        """Return the active stream session for this stream, if present."""
        return self._session

    def get_recording_context(self) -> DataRecordingContext | None:
        """Return the active recording context for this stream, if present."""
        return self._context

    def _send_to_daemon(self, data: bytes) -> None:
        """Enqueue a single-payload recording chunk into the coordinator.

        Args:
            data: Serialized data bytes to send.
        """
        if self._coordinator is None or self._session is None or not data:
            return
        self._coordinator.enqueue_stream_payload(
            StreamPayload(
                session=self._session,
                parts=(memoryview(data),),
                total_bytes=len(data),
            )
        )

    def _send_to_daemon_parts(
        self,
        parts: tuple[bytes | memoryview, ...],
        *,
        total_bytes: int,
    ) -> None:
        """Enqueue a logical payload assembled from multiple byte-like parts."""
        if self._coordinator is None or self._session is None:
            return
        self._coordinator.enqueue_stream_payload(
            StreamPayload(
                session=self._session,
                parts=parts,
                total_bytes=total_bytes,
            )
        )


class JsonDataStream(DataStream):
    """Stream that logs and sends structured JSON data to the daemon.

    Records arbitrary structured data as JSON and sends it to the daemon
    for persistence during recording sessions.
    """

    def __init__(self, data_type: DataType, data_type_name: str):
        """Initialize the JSON data stream.

        Args:
            data_type: Type of data being recorded (e.g., JSON events)
            data_type_name: Name of the JSON data stream
        """
        super().__init__(data_type=data_type, stream_name=data_type_name)

    def log(self, data: NCData, *, send_to_daemon: bool = True) -> None:
        """Log structured data as JSON.

        Args:
            data: Data object implementing NCData interface
            send_to_daemon: Whether to forward the serialized payload to the daemon
        """
        self._latest_data = data
        if not self.is_recording() or not send_to_daemon:
            return

        # Serialize to JSON bytes and send to daemon
        json_bytes = json.dumps(data.model_dump(mode="json")).encode("utf-8")
        self._send_to_daemon(json_bytes)


class VideoDataStream(DataStream):
    """Stream that sends video frame data to the daemon.

    Base class for video streams. Frame data is sent raw to the daemon
    which handles storage. Video encoding is done by the loader when
    uploading to the backend.
    """

    def __init__(
        self, data_type: DataType, camera_id: str, width: int = 640, height: int = 480
    ):
        """Initialize the video data stream.

        Args:
            data_type: Type of video data (RGB_IMAGES or DEPTH_IMAGES)
            camera_id: Unique identifier for the camera
            width: Video frame width in pixels
            height: Video frame height in pixels
        """
        super().__init__(data_type=data_type, stream_name=camera_id)
        self.camera_id = camera_id
        self.width = width
        self.height = height

    def log(self, metadata: CameraData, frame: np.ndarray) -> None:
        """Log video frame data.

        Args:
            metadata: Camera metadata including timestamp and calibration
            frame: Video frame as numpy array
        """
        metadata.frame = frame
        self._latest_data = metadata
        if not self.is_recording():
            return

        # Serialize metadata and frame to bytes
        # Frame is sent as raw numpy bytes with metadata as JSON header
        metadata_dict = metadata.model_dump(mode="json", exclude={"frame"})
        metadata_dict["width"] = self.width
        metadata_dict["height"] = self.height
        metadata_dict["frame_nbytes"] = int(frame.size * frame.itemsize)
        metadata_json = json.dumps(metadata_dict).encode("utf-8")

        # Pack: [metadata_len (4 bytes)] [metadata_json] [frame_bytes]
        header = struct.pack("<I", len(metadata_json))
        frame_source = (
            frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)
        )
        frame_view = memoryview(frame_source).cast("B")
        total_bytes = len(header) + len(metadata_json) + len(frame_view)
        self._send_to_daemon_parts(
            (header, metadata_json, frame_view),
            total_bytes=total_bytes,
        )


class DepthDataStream(VideoDataStream):
    """Stream that sends depth data to the daemon.

    Handles depth camera data. The raw depth data is sent to the daemon
    for storage and later processing by the loader.
    """

    def __init__(self, camera_id: str, width: int = 640, height: int = 480):
        """Initialize the depth data stream.

        Args:
            camera_id: Unique identifier for the camera
            width: Video frame width in pixels
            height: Video frame height in pixels
        """
        super().__init__(
            data_type=DataType.DEPTH_IMAGES,
            camera_id=camera_id,
            width=width,
            height=height,
        )


class RGBDataStream(VideoDataStream):
    """Stream that sends RGB video data to the daemon.

    Handles RGB camera data. The raw frame data is sent to the daemon
    for storage and later processing by the loader.
    """

    def __init__(self, camera_id: str, width: int = 640, height: int = 480):
        """Initialize the RGB data stream.

        Args:
            camera_id: Unique identifier for the camera
            width: Video frame width in pixels
            height: Video frame height in pixels
        """
        super().__init__(
            data_type=DataType.RGB_IMAGES,
            camera_id=camera_id,
            width=width,
            height=height,
        )
