"""Data stream classes for recording robot sensor data.

This module provides abstract and concrete data stream implementations for
recording various types of robot sensor data including JSON events, RGB video,
and depth data.

Streams own no transport of their own: the daemon interface lives at the
logging-function layer (:class:`~neuracore.data_daemon.bridge.RecordingContext`),
so a stream only tracks recording state and the latest sample for live-data
consumers.
"""

import logging
from abc import ABC

import numpy as np
from neuracore_types import CameraData, DataType, JointData, NCData, PointCloudData

logger = logging.getLogger(__name__)


class DataStream(ABC):
    """Base class for data streams.

    Provides common functionality for managing recording state and the latest
    sample across different types of sensor data streams.
    """

    def __init__(self, data_type: DataType, stream_name: str) -> None:
        """Initialize the data stream.

        Args:
            data_type: The type of data this stream handles.
            stream_name: Unique name for this stream (used as channel ID).

        Note:
            This must be kept lightweight and not perform any blocking operations.
        """
        self._recording = False
        self._latest_data: NCData | None = None
        self._data_type = data_type
        self._stream_name = stream_name
        self._last_logged_timestamp: float | None = None

    @property
    def data_type(self) -> DataType:
        """Get the data type of this stream."""
        return self._data_type

    def start_recording(self) -> None:
        """Arm the stream for a new recording.

        The stream carries no recording identity — the daemon owns that, and
        decides which recording a sample belongs to from when it was published.
        All this marks is that a fresh timeline has begun, so
        :meth:`_enforce_monotonic_timestamp` measures against this recording
        rather than the last.
        """
        self._recording = True
        self._last_logged_timestamp = None

    def stop_recording(self) -> None:
        """Stop recording data for this stream."""
        self._recording = False

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

    def _enforce_monotonic_timestamp(self, timestamp: float) -> None:
        """Reject a timestamp that does not strictly increase within a recording.

        Tracked per stream and only while recording; the previous value is
        cleared when a recording starts (see :meth:`start_recording`) so each
        recording is an independent, strictly increasing timeline. A no-op when
        the stream is not recording.

        Args:
            timestamp: Capture timestamp, in seconds, of the sample being logged.

        Raises:
            ValueError: If ``timestamp`` is not strictly greater than the last
                timestamp logged to this stream during the current recording.
        """
        if not self._recording:
            return
        last_logged_timestamp = self._last_logged_timestamp
        if last_logged_timestamp is not None and timestamp <= last_logged_timestamp:
            raise ValueError(
                f"Non-monotonic timestamp for '{self._stream_name}' "
                f"({self._data_type.value}): {timestamp} is not greater than the "
                f"previous timestamp {last_logged_timestamp}. Logged timestamps "
                "must be strictly increasing within a recording."
            )
        self._last_logged_timestamp = timestamp


class JsonDataStream(DataStream):
    """Stream that tracks structured JSON data.

    The sample itself is persisted by the logging layer via
    ``RecordingContext.log_json``; this keeps the latest value for live-data
    consumers.
    """

    def __init__(self, data_type: DataType, data_type_name: str):
        """Initialize the JSON data stream.

        Args:
            data_type: Type of data being recorded (e.g., JSON events)
            data_type_name: Name of the JSON data stream
        """
        super().__init__(data_type=data_type, stream_name=data_type_name)

    def log(self, data: NCData) -> None:
        """Log structured data.

        Args:
            data: Data object implementing NCData interface
        """
        self._enforce_monotonic_timestamp(data.timestamp)
        self._latest_data = data


class JointDataStream(JsonDataStream):
    """JSON stream for scalar joint samples with deferred latest-data builds.

    Joint logging is the hottest path in the SDK: during a recording one
    :class:`JointData` was materialised per joint per frame purely to keep
    ``_latest_data`` current for live-data / endpoint consumers, which read it
    at serving rate — far below the logging rate. At high joint counts that
    per-sample Pydantic construction, and the GC churn it drove, dominated the
    ``log_joint_*`` calls.

    This stream lets the logging layer hand over the raw ``(timestamp, value)``
    cheaply via :meth:`record_scalar` and defers building the ``JointData``
    until :meth:`get_latest_data` is actually called, so the hot path performs
    two attribute writes instead of a model construction.
    """

    def __init__(self, data_type: DataType, data_type_name: str) -> None:
        """Initialize the joint data stream."""
        super().__init__(data_type=data_type, data_type_name=data_type_name)
        self._pending_timestamp: float = 0.0
        self._pending_value: float = 0.0
        self._has_pending_latest = False

    def record_scalar(self, timestamp: float, value: float) -> None:
        """Stash the latest scalar sample without building a ``JointData``.

        The model is materialised lazily in :meth:`get_latest_data`. The
        latest-data read is best-effort: under concurrency a reader may observe
        a timestamp/value drawn from adjacent samples (the pair is not written
        atomically), but it never raises and never returns a partially
        constructed ``JointData``.
        """
        self._enforce_monotonic_timestamp(timestamp)
        self._pending_timestamp = timestamp
        self._pending_value = value
        self._has_pending_latest = True

    def log(self, data: NCData) -> None:
        """Log a materialised sample, superseding any deferred scalar."""
        self._has_pending_latest = False
        super().log(data=data)

    def get_latest_data(self) -> NCData | None:
        """Return the latest sample, materialising a deferred scalar on demand."""
        if self._has_pending_latest:
            self._latest_data = JointData(
                timestamp=self._pending_timestamp,
                value=self._pending_value,
            )
            self._has_pending_latest = False
        return self._latest_data


class PointCloudDataStream(DataStream):
    """Stream that tracks point cloud data."""

    def __init__(self, data_type_name: str):
        """Initialize the point cloud data stream.

        Args:
            data_type_name: Name of the point cloud stream
        """
        super().__init__(data_type=DataType.POINT_CLOUDS, stream_name=data_type_name)

    def log(self, data: PointCloudData) -> None:
        """Log point cloud data.

        Args:
            data: Point cloud data to log
        """
        self._enforce_monotonic_timestamp(data.timestamp)
        self._latest_data = data


class VideoDataStream(DataStream):
    """Stream that tracks video frame data.

    Base class for video streams. Frames are delivered to the daemon by the
    logging layer (``RecordingContext.log_frame``); this keeps the latest frame
    for live-data consumers.
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
        self._enforce_monotonic_timestamp(metadata.timestamp)
        metadata.frame = frame
        self._latest_data = metadata


class DepthDataStream(VideoDataStream):
    """Stream that tracks depth camera data."""

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
    """Stream that tracks RGB camera data."""

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
