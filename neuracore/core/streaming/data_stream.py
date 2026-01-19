"""Data stream classes for recording and uploading robot sensor data.

This module provides abstract and concrete data stream implementations for
recording various types of robot sensor data including JSON events, RGB video,
and depth data. All streams support recording lifecycle management and
daemon-based data persistence.
"""

import json
import logging
import struct
import threading
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from neuracore.data_daemon.communications_management.producer import Producer
from neuracore_types import CameraData, DataType, NCData

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

    Provides common functionality for managing recording state and data
    storage across different types of sensor data streams. Each stream
    has its own Producer for sending data to the daemon.
    """

    def __init__(self, data_type: DataType, stream_name: str) -> None:
        """Initialize the data stream.

        Args:
            data_type: The type of data this stream handles.
            stream_name: Unique name for this stream (used as producer ID).

        Note:
            This must be kept lightweight and not perform any blocking operations.
        """
        self._recording = False
        self._context: DataRecordingContext | None = None
        self._latest_data: NCData | None = None
        self._data_type = data_type
        self._stream_name = stream_name
        self._producer: Producer | None = None
        self.lock = threading.Lock()

    def start_recording(self, context: DataRecordingContext) -> None:
        """Start recording data.

        Args:
            context: Recording context containing identifiers for
                the recording session, robot, and dataset.

        Note:
            This must be kept lightweight and not perform any blocking operations.
        """
        if self.is_recording():
            self.stop_recording()
        self._recording = True
        self._context = context

        # Initialize producer with stream-specific ID and recording_id
        producer_id = f"{self._data_type.value}:{self._stream_name}"
        self._producer = Producer(
            id=producer_id,
            recording_id=context.recording_id,
        )
        self._producer.start_new_trace(recording_id=context.recording_id)

    def stop_recording(self) -> list[threading.Thread]:
        """Stop recording data.

        Returns:
            List[threading.Thread]: Empty list (no upload threads needed with daemon).
        """
        self._recording = False
        return []

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

    def _send_to_daemon(self, data: bytes) -> None:
        """Send data to the daemon via the producer.

        Args:
            data: Serialized data bytes to send.
        """
        if self._producer is None or self._context is None:
            return

        self._producer.send_data(
            data=data,
            data_type=self._data_type,
            recording_id=self._context.recording_id,
            robot_id=self._context.robot_id,
            robot_name=self._context.robot_name,
            dataset_id=self._context.dataset_id,
            dataset_name=self._context.dataset_name,
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

    def log(self, data: NCData) -> None:
        """Log structured data as JSON.

        Args:
            data: Data object implementing NCData interface
        """
        self._latest_data = data
        if not self.is_recording():
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
        metadata_json = json.dumps(metadata_dict).encode("utf-8")

        # Pack: [metadata_len (4 bytes)] [metadata_json] [frame_bytes]
        frame_bytes = frame.tobytes()
        header = struct.pack("<I", len(metadata_json))
        data = header + metadata_json + frame_bytes
        self._send_to_daemon(data)


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
