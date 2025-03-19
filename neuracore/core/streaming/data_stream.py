import json
import logging
import time
from abc import ABC, abstractmethod

import numpy as np
import requests
from ...core.streaming.client_stream import ClientStreamingManager
from ...core.streaming.client_stream import get_robot_streaming_manager
from ...core.auth import get_auth
from ...core.streaming.resumable_upload import ResumableUpload, SensorType

from ..const import API_URL
from .streaming_video_encoder import StreamingVideoEncoder

MAX_DEPTH = 10.0  # Maximum depth value in meters

logger = logging.getLogger(__name__)


class DataStream(ABC):
    """Base class for data streams."""

    def __init__(self, robot_id: str):
        self._recording = False
        self._recording_id = None
        self.robot_id = robot_id

    def start_recording(self, recording_id: str):
        """Start recording data."""
        self._recording = True
        self._recording_id = recording_id

    def stop_recording(self):
        """Stop recording data."""
        self._recording = False
        self._recording_id = None

    def is_recording(self) -> bool:
        """Check if recording is active."""
        return self._recording


class BufferedDataStream(DataStream, ABC):
    """Stream that buffers data locally for later upload."""

    def __init__(self, robot_id: str):
        super().__init__(robot_id=robot_id)
        self._buffer = []

    def log(self, dict_data: dict[str, float]):
        """Log data to the buffer if recording is active."""
        if not self.is_recording():
            return
        self._buffer.append(dict_data)

    def start_recording(self, recording_id: str):
        """Upload buffered data to storage."""
        super().start_recording(recording_id)
        self._buffer = []

    def stop_recording(self):
        """Upload buffered data to storage."""
        recoding_id = self._recording_id
        super().stop_recording()
        if not self._buffer:
            return
        # Generate an upload URL
        upload_url_response = requests.get(
            f"{API_URL}/recording/{recoding_id}/json_upload_url/{self.get_datatype()}",
            headers=get_auth().get_headers(),
        )
        upload_url_response.raise_for_status()
        upload_url = upload_url_response.json()["url"]
        data = json.dumps(self._buffer)
        logger.info(f"Uploading {len(data)} bytes to {upload_url}")
        response = requests.put(
            upload_url, headers={"Content-Length": str(len(data))}, data=data
        )
        response.raise_for_status()
        self._buffer = []

    @abstractmethod
    def get_datatype(self) -> str:
        """Get the endpoint name for this stream."""
        raise NotImplementedError()


class ActionDataStream(BufferedDataStream):
    """Stream that logs robot actions."""

    def get_datatype(self) -> str:
        """Get the endpoint name for this stream."""
        return "actions"


class JointDataStream(BufferedDataStream):
    """Stream that logs robot actions."""

    def get_datatype(self) -> str:
        """Get the endpoint name for this stream."""
        return "joints"


class VideoDataStream(DataStream):
    """Stream that encodes and uploads video data."""

    def __init__(
        self, robot_id: str, camera_id: str, width: int = 640, height: int = 480
    ):
        super().__init__(robot_id=robot_id)
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self._encoder = None

    def start_recording(self, recording_id: str):
        """Start video recording."""
        super().start_recording(recording_id)
        resumable_upload = self.get_resumable_upload(recording_id)
        self._encoder = StreamingVideoEncoder(resumable_upload, self.width, self.height)

    def stop_recording(self):
        """Stop video recording and finalize encoding."""
        if self.is_recording() and self._encoder is not None:
            self._encoder.finish()
        self._encoder = None
        super().stop_recording()

    @abstractmethod
    def get_resumable_upload(self, recording_id: str) -> ResumableUpload:
        """Get a resumable upload object for the current recording."""
        raise NotImplementedError()

    @abstractmethod
    def log(self, data: np.ndarray):
        raise NotImplementedError()


class DepthDataStream(VideoDataStream):
    """Stream that encodes and uploads depth data as video."""

    def get_resumable_upload(self, recording_id):
        return ResumableUpload(recording_id, SensorType.DEPTH, self.camera_id)

    def log(self, data: np.ndarray):
        """Convert depth to RGB and log as a video frame."""
        if not self.is_recording() or self._encoder is None:
            return

        # Convert depth to RGB representation for video encoding
        # Scale to 0-255 range for visualization
        normalized_depth = np.clip(data / MAX_DEPTH, 0, 1)

        # Create a heat map representation (red = close, blue = far)
        rgb_depth = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
        rgb_depth[..., 0] = (1.0 - normalized_depth) * 255  # Red channel (close)
        rgb_depth[..., 2] = normalized_depth * 255  # Blue channel (far)

        # Add frame to encoder
        self._encoder.add_frame(rgb_depth, time.time())


class RGBDataStream(VideoDataStream):
    """Stream that encodes and uploads RGB data as video."""

    def get_resumable_upload(self, recording_id):
        return ResumableUpload(recording_id, SensorType.RGB, self.camera_id)

    def log(self, data: np.ndarray):
        """Log an RGB frame."""
        if not self.is_recording() or self._encoder is None:
            return

        get_robot_streaming_manager(robot_id=self.robot_id).get_recording_video_stream(
            self._recording_id, SensorType.RGB, self.camera_id
        ).add_frame(data)
        self._encoder.add_frame(data, time.time())