import logging
import threading
from abc import ABC
from typing import Any

import numpy as np

from neuracore.core.streaming.bucket_uploaders.streaming_file_uploader import (
    StreamingJsonUploader,
)
from neuracore.core.streaming.bucket_uploaders.streaming_video_uploader import (
    StreamingVideoUploader,
)

from ..nc_types import CameraData, NCData
from ..utils.depth_utils import depth_to_rgb

logger = logging.getLogger(__name__)


class DataStream(ABC):
    """Base class for data streams."""

    def __init__(self):
        """Initialize the data stream.

        This must be kept lightweight and not perform any blocking operations.
        """
        self._recording = False
        self._recording_id = None
        self._latest_data = None
        self.lock = threading.Lock()

    def start_recording(self, recording_id: str):
        """Start recording data.

        This must be kept lightweight and not perform any blocking operations.
        """
        if self.is_recording():
            self.stop_recording()
        self._recording = True
        self._recording_id = recording_id

    def stop_recording(self) -> threading.Thread:
        """Stop recording data."""
        if not self.is_recording():
            raise ValueError("Not recording")
        self._recording = False
        self._recording_id = None

    def is_recording(self) -> bool:
        """Check if recording is active."""
        return self._recording

    def get_latest_data(self) -> Any:
        """Get the latest data from the stream."""
        return self._latest_data


class JsonDataStream(DataStream):
    """Stream that logs custom data."""

    def __init__(self, filepath: str):
        super().__init__()
        # add .json if missing
        if not filepath.endswith(".json"):
            filepath += ".json"
        self.filepath = filepath
        self._streamer = None

    def start_recording(self, recording_id):
        super().start_recording(recording_id)
        self._streamer = StreamingJsonUploader(recording_id, self.filepath)

    def stop_recording(self) -> threading.Thread:
        """Stop video recording and finalize encoding."""
        super().stop_recording()
        upload_thread = self._streamer.finish()
        self._streamer = None
        return upload_thread

    def log(self, data: NCData):
        """Convert depth to RGB and log as a video frame."""
        self._latest_data = data
        if not self.is_recording() or self._streamer is None:
            return
        self._streamer.add_frame(data.model_dump())


class VideoDataStream(DataStream):
    """Stream that encodes and uploads video data."""

    def __init__(self, camera_id: str, width: int = 640, height: int = 480):
        super().__init__()
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self._encoder: VideoDataStream | None = None

    def start_recording(self, recording_id: str):
        """Start video recording."""
        super().start_recording(recording_id)

    def stop_recording(self) -> threading.Thread:
        """Stop video recording and finalize encoding."""
        super().stop_recording()
        upload_thread = self._encoder.finish()
        self._encoder = None
        return upload_thread

    def log(self, data: np.ndarray, metadata: CameraData):
        """Convert depth to RGB and log as a video frame."""
        self._latest_data = data
        if not self.is_recording() or self._encoder is None:
            return
        self._encoder.add_frame(data, metadata)


class DepthDataStream(VideoDataStream):
    """Stream that encodes and uploads depth data as video."""

    def start_recording(self, recording_id: str):
        """Start video recording."""
        super().start_recording(recording_id)
        self._encoder = StreamingVideoUploader(
            recording_id,
            f"depths/{self.camera_id}",
            self.width,
            self.height,
            depth_to_rgb,
        )


class RGBDataStream(VideoDataStream):
    """Stream that encodes and uploads RGB data as video."""

    def start_recording(self, recording_id: str):
        """Start video recording."""
        super().start_recording(recording_id)
        self._encoder = StreamingVideoUploader(
            recording_id, f"rgbs/{self.camera_id}", self.width, self.height
        )
