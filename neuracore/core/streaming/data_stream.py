import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import requests

from ..auth import get_auth
from ..const import API_URL
from ..utils.depth_utils import depth_to_rgb
from .streaming_video_encoder import StreamingVideoEncoder

logger = logging.getLogger(__name__)


class DataStream(ABC):
    """Base class for data streams."""

    def __init__(self):
        """Initialize the data stream.

        This must be kept lightweight and not perform any blocking operations.
        """
        self._recording = False
        self._recording_id = None

    def start_recording(self, recording_id: str):
        """Start recording data.

        This must be kept lightweight and not perform any blocking operations.
        """
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


class BufferedDataStream(DataStream):
    """Stream that buffers data locally for later upload."""

    def __init__(self, filename: str):
        super().__init__()
        self._filename = filename
        self._buffer = []

    def log(self, dict_data: dict[str, float], timestamp: Optional[float] = None):
        """Log data to the buffer if recording is active."""
        timestamp = timestamp or time.time()
        if not self.is_recording():
            return
        self._buffer.append({
            "timestamp": timestamp,
            "data": dict_data,
        })

    def start_recording(self, recording_id: str):
        """Upload buffered data to storage."""
        super().start_recording(recording_id)
        self._buffer = []

    def _upload_loop(self, recoding_id: str, json_data: str):
        """Upload buffered data to storage."""
        # Generate an upload URL
        upload_url_response = requests.get(
            f"{API_URL}/recording/{recoding_id}/json_upload_url?filename={self._filename}",
            headers=get_auth().get_headers(),
        )
        upload_url_response.raise_for_status()
        upload_url = upload_url_response.json()["url"]
        logger.info(f"Uploading {len(json_data)} bytes to {upload_url}")
        response = requests.put(
            upload_url, headers={"Content-Length": str(len(json_data))}, data=json_data
        )
        response.raise_for_status()
        self._buffer = []

    def stop_recording(self) -> threading.Thread:
        """Upload buffered data to storage."""
        recoding_id = self._recording_id
        super().stop_recording()
        if not self._buffer:
            return
        upload_thread = threading.Thread(
            target=self._upload_loop,
            args=(recoding_id, json.dumps(self._buffer)),
            daemon=False,
        )
        upload_thread.start()
        self._buffer = []
        return upload_thread


class ActionDataStream(BufferedDataStream):
    """Stream that logs robot actions."""

    def __init__(self, group_name: str):
        super().__init__(f"actions/{group_name}.json")


class JointDataStream(BufferedDataStream):
    """Stream that logs robot actions."""

    def __init__(self, group_name: str):
        super().__init__(f"joint_states/{group_name}.json")

    def log(
        self,
        dict_data: dict[str, float],
        additional_urdf_positions: dict[str, float],
        timestamp: Optional[float] = None,
    ):
        """Log data to the buffer if recording is active."""
        timestamp = timestamp or time.time()
        if not self.is_recording():
            return
        self._buffer.append({
            "timestamp": timestamp,
            "data": dict_data,
            "additional_urdf_positions": additional_urdf_positions,
        })


class GripperOpenAmountsDataStream(BufferedDataStream):
    """Stream that logs robot actions."""

    def __init__(self, group_name: str):
        super().__init__(f"gripper_open_amounts/{group_name}.json")


class LanguageDataStream(BufferedDataStream):
    """Stream that logs robot actions."""

    def __init__(self):
        super().__init__("language_annotation.json")


class VideoDataStream(DataStream):
    """Stream that encodes and uploads video data."""

    def __init__(self, camera_id: str, width: int = 640, height: int = 480):
        super().__init__()
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self._encoder = None

    def start_recording(self, recording_id: str):
        """Start video recording."""
        super().start_recording(recording_id)
        self._encoder = StreamingVideoEncoder(
            recording_id, self.camera_id, self.width, self.height
        )

    def stop_recording(self) -> threading.Thread:
        """Stop video recording and finalize encoding."""
        super().stop_recording()
        upload_thread = self._encoder.finish()
        self._encoder = None
        return upload_thread

    @abstractmethod
    def log(self, data: np.ndarray, timestamp: Optional[float] = None):
        raise NotImplementedError()


class DepthDataStream(VideoDataStream):
    """Stream that encodes and uploads depth data as video."""

    def start_recording(self, recording_id: str):
        """Start video recording."""
        super(DataStream, self).start_recording(recording_id)
        self._encoder = StreamingVideoEncoder(
            recording_id, self.camera_id, self.width, self.height, depth_to_rgb
        )

    def log(self, data: np.ndarray, timestamp: Optional[float] = None):
        """Convert depth to RGB and log as a video frame."""
        if not self.is_recording() or self._encoder is None:
            return
        timestamp = timestamp or time.time()
        self._encoder.add_frame(data, timestamp)


class RGBDataStream(VideoDataStream):
    """Stream that encodes and uploads RGB data as video."""

    def log(self, data: np.ndarray, timestamp: Optional[float] = None):
        """Log an RGB frame."""
        if not self.is_recording() or self._encoder is None:
            return
        timestamp = timestamp or time.time()
        self._encoder.add_frame(data, timestamp)
