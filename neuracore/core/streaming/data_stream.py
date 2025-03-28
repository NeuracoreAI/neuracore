import logging
import threading
from abc import ABC

import numpy as np

from ..nc_types import CameraMetaData, NCData
from ..streaming.streaming_file_uploader import StreamingJsonUploader
from ..utils.depth_utils import depth_to_rgb
from .streaming_video_uploader import StreamingVideoUploader

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


class JsonDataStream(DataStream):
    """Stream that logs custom data."""

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

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
        if not self.is_recording() or self._streamer is None:
            return
        self._streamer.add_frame(data.model_dump())


class JointDataStream(JsonDataStream):
    """Stream that logs joint data."""

    def __init__(self, sensor_name: str, group_name: str):
        super().__init__(f"{sensor_name}/{group_name}.json")


class ActionDataStream(JsonDataStream):
    """Stream that logs robot actions."""

    def __init__(self, group_name: str):
        super().__init__(f"actions/{group_name}.json")


class GripperDataStream(JsonDataStream):
    """Stream that logs gripper open amounts."""

    def __init__(self, group_name: str):
        super().__init__(f"gripper_open_amounts/{group_name}.json")


class LanguageDataStream(JsonDataStream):
    """Stream that logs language annotations."""

    def __init__(self):
        super().__init__("language_annotation.json")


class PointCloudDataStream(JsonDataStream):
    """Stream that logs point cloud data."""

    def __init__(self, sensor_name: str):
        super().__init__(f"point_clouds/{sensor_name}.json")


class CustomDataStream(JsonDataStream):
    """Stream that logs custom data."""

    def __init__(self, stream_name: str):
        super().__init__(f"custom/{stream_name}.json")


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

    def stop_recording(self) -> threading.Thread:
        """Stop video recording and finalize encoding."""
        super().stop_recording()
        upload_thread = self._encoder.finish()
        self._encoder = None
        return upload_thread

    def log(self, data: np.ndarray, metadata: CameraMetaData):
        """Convert depth to RGB and log as a video frame."""
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
