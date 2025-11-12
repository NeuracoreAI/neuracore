"""Data stream classes for recording and uploading robot sensor data.

This module provides abstract and concrete data stream implementations for
recording various types of robot sensor data including JSON events, RGB video,
and depth data. All streams support recording lifecycle management and
cloud upload functionality.
"""

import logging
import threading
from abc import ABC
from typing import Any, List, Optional

import numpy as np
from neuracore_types import CameraData, NCData

from neuracore.core.streaming.bucket_uploaders.streaming_file_uploader import (
    StreamingJsonUploader,
)
from neuracore.core.streaming.bucket_uploaders.streaming_video_uploader import (
    StreamingVideoUploader,
)

from ..utils.depth_utils import depth_to_rgb

logger = logging.getLogger(__name__)


class DataStream(ABC):
    """Base class for data streams.

    Provides common functionality for managing recording state and data
    storage across different types of sensor data streams.
    """

    def __init__(self) -> None:
        """Initialize the data stream.

        This must be kept lightweight and not perform any blocking operations.
        """
        self._recording = False
        self._recording_id: Optional[str] = None
        self._latest_data: Optional[NCData] = None
        self.lock = threading.Lock()

    def start_recording(self, recording_id: str) -> None:
        """Start recording data.

        Args:
            recording_id: Unique identifier for the recording session

        Note:
            This must be kept lightweight and not perform any blocking operations.
        """
        if self.is_recording():
            self.stop_recording()
        self._recording = True
        self._recording_id = recording_id

    def stop_recording(self) -> List[threading.Thread]:
        """Stop recording data.

        Returns:
            List[threading.Thread]: List of upload threads for cleanup

        Raises:
            ValueError: If not currently recording
        """
        if not self.is_recording():
            raise ValueError("Not recording")
        self._recording = False
        self._recording_id = None
        return []

    def is_recording(self) -> bool:
        """Check if recording is active.

        Returns:
            bool: True if currently recording, False otherwise
        """
        return self._recording

    def get_latest_data(self) -> Any:
        """Get the latest data from the stream.

        Returns:
            Any: The most recently logged data item
        """
        return self._latest_data


class JsonDataStream(DataStream):
    """Stream that logs and uploads structured JSON data.

    Records arbitrary structured data as JSON files and uploads them
    to cloud storage during recording sessions.
    """

    def __init__(self, filepath: str):
        """Initialize the JSON data stream.

        Args:
            filepath: Path for the JSON file. .json extension added if missing
        """
        super().__init__()
        # add .json if missing
        if not filepath.endswith(".json"):
            filepath += ".json"
        self.filepath = filepath
        self._streamer: Optional[StreamingJsonUploader] = None

    def start_recording(self, recording_id: str) -> None:
        """Start JSON data recording.

        Args:
            recording_id: Unique identifier for the recording session
        """
        super().start_recording(recording_id)
        self._streamer = StreamingJsonUploader(recording_id, self.filepath)

    def stop_recording(self) -> List[threading.Thread]:
        """Stop JSON recording and finalize upload.

        Returns:
            List[threading.Thread]: Upload thread for cleanup
        """
        super().stop_recording()
        if self._streamer is None:
            raise TypeError("Streamer is None")
        upload_thread = self._streamer.finish()
        self._streamer = None
        return [upload_thread]

    def log(self, data: NCData) -> None:
        """Log structured data as JSON.

        Args:
            data: Data object implementing NCData interface
        """
        self._latest_data = data
        if not self.is_recording() or self._streamer is None:
            return
        self._streamer.add_frame(data.model_dump(mode="json"))


class VideoDataStream(DataStream):
    """Stream that encodes and uploads video data.

    Base class for video streams that provides dual encoding (lossless and lossy)
    for optimal storage and streaming performance.
    """

    def __init__(self, camera_id: str, width: int = 640, height: int = 480):
        """Initialize the video data stream.

        Args:
            camera_id: Unique identifier for the camera
            width: Video frame width in pixels
            height: Video frame height in pixels
        """
        super().__init__()
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self._lossless_encoder: Optional[StreamingVideoUploader] = None
        self._lossy_encoder: Optional[StreamingVideoUploader] = None

    def start_recording(self, recording_id: str) -> None:
        """Start video recording.

        Args:
            recording_id: Unique identifier for the recording session
        """
        super().start_recording(recording_id)

    def stop_recording(self) -> List[threading.Thread]:
        """Stop video recording and finalize encoding.

        Returns:
            List[threading.Thread]: Upload threads for both lossless and lossy encoders
        """
        super().stop_recording()
        if self._lossless_encoder is None:
            raise TypeError("_lossless_encoder is None")
        lossless_upload_thread = self._lossless_encoder.finish()
        if self._lossy_encoder is None:
            raise TypeError("_lossy_encoder is None")
        lossy_upload_thread = self._lossy_encoder.finish()
        self._lossless_encoder = None
        self._lossy_encoder = None
        return [lossless_upload_thread, lossy_upload_thread]

    def log(self, data: np.ndarray, metadata: CameraData) -> None:
        """Log video frame data.

        Args:
            data: Video frame as numpy array
            metadata: Camera metadata including timestamp and calibration
        """
        self._latest_data = data
        if (
            not self.is_recording()
            or self._lossless_encoder is None
            or self._lossy_encoder is None
        ):
            return
        self._lossless_encoder.add_frame(data, metadata)
        self._lossy_encoder.add_frame(data, metadata)


class DepthDataStream(VideoDataStream):
    """Stream that encodes and uploads depth data as video.

    Converts depth data to RGB representation for video encoding while
    maintaining both lossless and lossy variants for different use cases.
    """

    def start_recording(self, recording_id: str) -> None:
        """Start depth video recording.

        Initializes both lossless (for accuracy) and lossy (for bandwidth)
        encoders with appropriate codec settings for depth data.

        Args:
            recording_id: Unique identifier for the recording session
        """
        super().start_recording(recording_id)
        self._lossless_encoder = StreamingVideoUploader(
            recording_id,
            f"depths/{self.camera_id}",
            self.width,
            self.height,
            depth_to_rgb,
            codec_context_options={"qp": "0", "preset": "ultrafast"},
        )
        self._lossy_encoder = StreamingVideoUploader(
            recording_id,
            f"depths/{self.camera_id}/lossy",
            self.width,
            self.height,
            depth_to_rgb,
            video_name="lossy.mp4",
            pixel_format="yuv420p",
            codec="libx264",
        )


class RGBDataStream(VideoDataStream):
    """Stream that encodes and uploads RGB video data.

    Handles RGB camera data with dual encoding for both archival quality
    (lossless) and streaming efficiency (lossy) use cases.
    """

    def start_recording(self, recording_id: str) -> None:
        """Start RGB video recording.

        Initializes both lossless and lossy encoders with appropriate
        settings for RGB video data.

        Args:
            recording_id: Unique identifier for the recording session
        """
        super().start_recording(recording_id)
        self._lossless_encoder = StreamingVideoUploader(
            recording_id=recording_id,
            path=f"rgbs/{self.camera_id}",
            width=self.width,
            height=self.height,
            codec_context_options={"qp": "0", "preset": "ultrafast"},
        )
        self._lossy_encoder = StreamingVideoUploader(
            recording_id=recording_id,
            path=f"rgbs/{self.camera_id}",
            width=self.width,
            height=self.height,
            video_name="lossy.mp4",
            pixel_format="yuv420p",
            codec="libx264",
        )
