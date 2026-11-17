"""Fill Neuracore MCAP export messages with the frames stored in attachments.

A Neuracore MCAP export writes each camera and point cloud sample as its raw
trace entry and stores the frames once per sensor as an attachment. The channel
metadata names the sensor data type and the attachment. The readers here decode
the frame of each trace entry and add it to the message under the field name of
the matching neuracore_types model: frame for cameras, points and rgb_points for
point clouds.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

import av
import numpy as np
from neuracore_types import DataType
from neuracore_types.nc_data.point_cloud_data import (
    PointCloudData,
    decode_point_cloud_frame,
)

from neuracore.core.utils.depth_utils import decode_depth_frame
from neuracore.importer.core.exceptions import ImportError

CAMERA_FRAME_FIELD = "frame"
POINT_CLOUD_POINTS_FIELD = "points"
POINT_CLOUD_RGB_POINTS_FIELD = "rgb_points"

DATA_TYPE_CHANNEL_METADATA_KEY = "data_type"
SENSOR_NAME_CHANNEL_METADATA_KEY = "sensor_name"
ATTACHMENT_CHANNEL_METADATA_KEY = "attachment"


def _trace_entry(message: Any, source_name: str) -> dict[str, Any]:
    """Return the message as a trace entry dictionary.

    Raises:
        ImportError: If the message is not a JSON object.
    """
    if not isinstance(message, dict):
        raise ImportError(
            f"Expected a trace entry object for {source_name}, "
            f"got {type(message).__name__}."
        )
    return message


def _frame_index(trace_entry: dict[str, Any], source_name: str) -> int:
    """Return the frame index of a camera trace entry.

    Raises:
        ImportError: If the entry has no non-negative integer frame_idx.
    """
    frame_index = trace_entry.get("frame_idx")
    if not isinstance(frame_index, int) or frame_index < 0:
        raise ImportError(f"Invalid frame_idx in {source_name}: {trace_entry}.")
    return frame_index


def _attachment_byte_range(
    trace_entry: dict[str, Any], attachment_name: str, attachment_bytes: bytes
) -> bytes:
    """Return the bytes of one frame named by the offset and length of an entry.

    Raises:
        ImportError: If the range is missing or falls outside the attachment.
    """
    offset = trace_entry.get("offset")
    length = trace_entry.get("length")
    if (
        not isinstance(offset, int)
        or not isinstance(length, int)
        or offset < 0
        or length <= 0
        or offset + length > len(attachment_bytes)
    ):
        raise ImportError(
            f"Invalid frame range in {attachment_name}: "
            f"offset={offset}, length={length}."
        )
    return attachment_bytes[offset : offset + length]


class VideoAttachmentFrameReader:
    """Decode an RGB video attachment frame by frame in display order."""

    def __init__(self, attachment_name: str, video_bytes: bytes) -> None:
        """Open the video attachment for sequential decoding.

        Args:
            attachment_name: Attachment name, used in error messages.
            video_bytes: The MP4 file stored in the attachment.
        """
        self.attachment_name = attachment_name
        self._container = av.open(io.BytesIO(video_bytes))
        self._frames_in_display_order: Iterator[av.VideoFrame] = self._container.decode(
            video=0
        )
        self._next_frame_index = 0
        self._last_frame_index: int | None = None
        self._last_frame_pixels: np.ndarray | None = None

    def fill_message(self, message: Any) -> dict[str, Any]:
        """Return the trace entry with its RGB frame added under frame.

        Raises:
            ImportError: If the entry asks for an earlier frame than the last one,
                or for a frame past the end of the video.
        """
        trace_entry = _trace_entry(message, self.attachment_name)
        frame_index = _frame_index(trace_entry, self.attachment_name)
        if frame_index == self._last_frame_index:
            return {**trace_entry, CAMERA_FRAME_FIELD: self._last_frame_pixels}
        if frame_index < self._next_frame_index:
            raise ImportError(
                f"Frame {frame_index} of {self.attachment_name} comes after frame "
                f"{self._next_frame_index - 1}; frames must arrive in order."
            )
        decoded_frame = None
        while self._next_frame_index <= frame_index:
            decoded_frame = next(self._frames_in_display_order, None)
            if decoded_frame is None:
                raise ImportError(
                    f"Video {self.attachment_name} has no frame {frame_index}."
                )
            self._next_frame_index += 1
        assert decoded_frame is not None
        pixels = decoded_frame.to_ndarray(format="rgb24")
        self._last_frame_index = frame_index
        self._last_frame_pixels = pixels
        return {**trace_entry, CAMERA_FRAME_FIELD: pixels}

    def close(self) -> None:
        """Close the video container."""
        self._container.close()


class DepthFramesAttachmentReader:
    """Decode depth frames from a depth frames attachment (lossless.bin)."""

    def __init__(self, attachment_name: str, frames_file_bytes: bytes) -> None:
        """Keep the depth frames file for byte range reads.

        Args:
            attachment_name: Attachment name, used in error messages.
            frames_file_bytes: The depth frames file stored in the attachment.
        """
        self.attachment_name = attachment_name
        self._frames_file_bytes = frames_file_bytes

    def fill_message(self, message: Any) -> dict[str, Any]:
        """Return the trace entry with its uint16 depth frame added under frame."""
        trace_entry = _trace_entry(message, self.attachment_name)
        frame_bytes = _attachment_byte_range(
            trace_entry, self.attachment_name, self._frames_file_bytes
        )
        try:
            depth_units = decode_depth_frame(frame_bytes)
        except (RuntimeError, ValueError) as exc:
            raise ImportError(
                f"Failed decoding a depth frame of {self.attachment_name}: {exc}"
            ) from exc
        return {**trace_entry, CAMERA_FRAME_FIELD: depth_units}

    def close(self) -> None:
        """Close the reader."""


class PointCloudFrameReader:
    """Decode point clouds from a trace.bin attachment or the trace entry itself."""

    def __init__(self, source_name: str, trace_bin_bytes: bytes | None) -> None:
        """Keep the point cloud frames file, when the sensor has one.

        Args:
            source_name: Attachment name or topic, used in error messages.
            trace_bin_bytes: The trace.bin file stored in the attachment, or None
                when each trace entry holds its points inline.
        """
        self.source_name = source_name
        self._trace_bin_bytes = trace_bin_bytes

    def fill_message(self, message: Any) -> dict[str, Any]:
        """Return the trace entry with points and rgb_points as numpy arrays.

        Raises:
            ImportError: If the frame does not decode or holds no points.
        """
        trace_entry = _trace_entry(message, self.source_name)
        try:
            if self._trace_bin_bytes is None:
                point_cloud = PointCloudData.model_validate(trace_entry)
            else:
                point_cloud = decode_point_cloud_frame(
                    _attachment_byte_range(
                        trace_entry, self.source_name, self._trace_bin_bytes
                    )
                )
        except ValueError as exc:
            raise ImportError(
                f"Failed decoding a point cloud of {self.source_name}: {exc}"
            ) from exc
        if point_cloud.points is None:
            raise ImportError(f"Point cloud of {self.source_name} has no points.")
        return {
            **trace_entry,
            POINT_CLOUD_POINTS_FIELD: point_cloud.points,
            POINT_CLOUD_RGB_POINTS_FIELD: point_cloud.rgb_points,
        }

    def close(self) -> None:
        """Close the reader."""


NeuracoreAttachmentFrameReader = (
    VideoAttachmentFrameReader | DepthFramesAttachmentReader | PointCloudFrameReader
)


def build_neuracore_attachment_frame_readers(
    reader: Any, summary: Any | None, topics: list[str]
) -> dict[str, NeuracoreAttachmentFrameReader]:
    """Build one frame reader per requested Neuracore camera or point cloud topic.

    A topic qualifies when its channel metadata carries the data_type and
    sensor_name keys that the Neuracore MCAP exporter writes. Other topics keep
    their messages unchanged.

    Args:
        reader: MCAP reader of the episode file.
        summary: Summary section of the episode file, or None when it has none.
        topics: Topics the import configuration reads.

    Returns:
        Frame readers keyed by topic.

    Raises:
        ImportError: If a channel names an attachment that the file lacks.
    """
    channels = getattr(summary, "channels", None) or {}
    requested_topics = set(topics)
    attachment_name_by_topic: dict[str, str | None] = {}
    data_type_by_topic: dict[str, DataType] = {}
    for channel in channels.values():
        metadata = getattr(channel, "metadata", None) or {}
        if channel.topic not in requested_topics:
            continue
        if SENSOR_NAME_CHANNEL_METADATA_KEY not in metadata:
            continue
        data_type_value = metadata.get(DATA_TYPE_CHANNEL_METADATA_KEY)
        if data_type_value not in {
            DataType.RGB_IMAGES.value,
            DataType.DEPTH_IMAGES.value,
            DataType.POINT_CLOUDS.value,
        }:
            continue
        data_type_by_topic[channel.topic] = DataType(data_type_value)
        attachment_name_by_topic[channel.topic] = metadata.get(
            ATTACHMENT_CHANNEL_METADATA_KEY
        )

    needed_attachment_names = {
        name for name in attachment_name_by_topic.values() if name is not None
    }
    attachment_bytes_by_name: dict[str, bytes] = {}
    if needed_attachment_names:
        for attachment in reader.iter_attachments():
            if attachment.name in needed_attachment_names:
                attachment_bytes_by_name[attachment.name] = bytes(attachment.data)

    frame_readers: dict[str, NeuracoreAttachmentFrameReader] = {}
    try:
        for topic, data_type in data_type_by_topic.items():
            attachment_name = attachment_name_by_topic[topic]
            if attachment_name is None:
                if data_type == DataType.POINT_CLOUDS:
                    frame_readers[topic] = PointCloudFrameReader(topic, None)
                continue
            attachment_bytes = attachment_bytes_by_name.get(attachment_name)
            if attachment_bytes is None:
                raise ImportError(
                    f"Topic {topic} names attachment {attachment_name}, "
                    "which the MCAP file does not contain."
                )
            if data_type == DataType.RGB_IMAGES:
                frame_readers[topic] = VideoAttachmentFrameReader(
                    attachment_name, attachment_bytes
                )
            elif data_type == DataType.DEPTH_IMAGES:
                frame_readers[topic] = DepthFramesAttachmentReader(
                    attachment_name, attachment_bytes
                )
            else:
                frame_readers[topic] = PointCloudFrameReader(
                    attachment_name, attachment_bytes
                )
    except BaseException:
        close_neuracore_attachment_frame_readers(frame_readers)
        raise
    return frame_readers


def close_neuracore_attachment_frame_readers(
    frame_readers: dict[str, NeuracoreAttachmentFrameReader],
) -> None:
    """Close every frame reader of an episode."""
    for frame_reader in frame_readers.values():
        frame_reader.close()
