"""Read a Neuracore MCAP export with the mcap package and summarise every sensor.

The script needs only mcap, av, imagecodecs and numpy. It decodes RGB frames
from the video attachments, depth frames from the depth frames attachments and
point clouds from their trace entries or trace.bin attachment, then prints one
line per sensor. docs/mcap_export_format.md describes the file layout.

Usage:
    python example_read_mcap_export.py nc_recording.mcap
"""

import argparse
import base64
import io
import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import imagecodecs
import numpy as np
from mcap.reader import make_reader

RECORDING_METADATA_NAME = "neuracore.recording"
POINT_CLOUD_METADATA_LENGTH_FORMAT = "<I"


@dataclass
class SensorSummary:
    """Message count and first decoded value of one sensor channel."""

    data_type: str
    message_count: int = 0
    first_value_description: str = ""


def describe_value(value: Any) -> str:
    """Return a short description of a decoded value."""
    if isinstance(value, np.ndarray):
        return f"array shape={value.shape} dtype={value.dtype}"
    return repr(value)[:80]


def decode_inline_point_cloud(
    trace_entry: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the points and colours stored inline in a point cloud entry.

    Args:
        trace_entry: Point cloud trace entry with base64 points and rgb_points.

    Returns:
        Float16 points of shape (N, 3) and uint8 colours of shape (N, 3), or
        None when the entry has no colours.
    """
    points = np.frombuffer(
        base64.b64decode(trace_entry["points"]), dtype="<f2"
    ).reshape(-1, 3)
    rgb_points = None
    if trace_entry.get("rgb_points"):
        rgb_points = np.frombuffer(
            base64.b64decode(trace_entry["rgb_points"]), dtype=np.uint8
        ).reshape(-1, 3)
    return points, rgb_points


def decode_point_cloud_file_frame(
    frame_bytes: bytes,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the points and colours of one frame of a trace.bin attachment.

    A frame holds a little-endian uint32 metadata length, the JSON metadata,
    the float16 points and then the uint8 colours.

    Args:
        frame_bytes: Bytes of one frame.

    Returns:
        Float16 points of shape (N, 3) and uint8 colours of shape (N, 3), or
        None when the frame has no colours.
    """
    header_length = struct.calcsize(POINT_CLOUD_METADATA_LENGTH_FORMAT)
    (metadata_length,) = struct.unpack_from(
        POINT_CLOUD_METADATA_LENGTH_FORMAT, frame_bytes
    )
    metadata_end = header_length + metadata_length
    metadata = json.loads(frame_bytes[header_length:metadata_end])
    points_end = metadata_end + metadata["points_nbytes"]
    points = np.frombuffer(frame_bytes[metadata_end:points_end], dtype="<f2").reshape(
        -1, 3
    )
    rgb_points = None
    if metadata.get("rgb_nbytes", 0) > 0:
        colours_end = points_end + metadata["rgb_nbytes"]
        rgb_points = np.frombuffer(
            frame_bytes[points_end:colours_end], dtype=np.uint8
        ).reshape(-1, 3)
    return points, rgb_points


def frame_bytes_of_entry(trace_entry: dict[str, Any], attachment_bytes: bytes) -> bytes:
    """Return the bytes of one frame named by the offset and length of an entry."""
    start = trace_entry["offset"]
    return attachment_bytes[start : start + trace_entry["length"]]


def read_export(mcap_path: Path) -> None:
    """Decode every message of one exported recording and print a summary.

    Args:
        mcap_path: Path of the exported MCAP file.
    """
    summaries: dict[str, SensorSummary] = {}
    video_frames_by_topic: dict[str, Iterator[av.VideoFrame]] = {}
    video_containers: list[Any] = []
    with mcap_path.open("rb") as stream:
        reader = make_reader(stream)
        recording = next(
            json.loads(metadata.metadata["json"])
            for metadata in reader.iter_metadata()
            if metadata.name == RECORDING_METADATA_NAME
        )
        attachment_bytes_by_name = {
            attachment.name: bytes(attachment.data)
            for attachment in reader.iter_attachments()
        }
        try:
            for _, channel, message in reader.iter_messages(log_time_order=True):
                trace_entry = json.loads(message.data)
                data_type = channel.metadata["data_type"]
                attachment_name = channel.metadata.get("attachment")
                summary = summaries.setdefault(
                    channel.topic, SensorSummary(data_type=data_type)
                )

                decoded_value: Any
                if data_type == "RGB_IMAGES":
                    if channel.topic not in video_frames_by_topic:
                        container = av.open(
                            io.BytesIO(attachment_bytes_by_name[attachment_name])
                        )
                        video_containers.append(container)
                        video_frames_by_topic[channel.topic] = container.decode(video=0)
                    if trace_entry["frame_idx"] != summary.message_count:
                        raise ValueError(f"Frames out of order on {channel.topic}.")
                    decoded_value = next(video_frames_by_topic[channel.topic])
                    decoded_value = decoded_value.to_ndarray(format="rgb24")
                elif data_type == "DEPTH_IMAGES":
                    depth_units = imagecodecs.jpegxl_decode(
                        frame_bytes_of_entry(
                            trace_entry, attachment_bytes_by_name[attachment_name]
                        )
                    )
                    decoded_value = depth_units.astype(np.float32) * np.float32(
                        trace_entry["depth_scale_m"]
                    )
                elif data_type == "POINT_CLOUDS" and attachment_name is None:
                    decoded_value, _ = decode_inline_point_cloud(trace_entry)
                elif data_type == "POINT_CLOUDS":
                    decoded_value, _ = decode_point_cloud_file_frame(
                        frame_bytes_of_entry(
                            trace_entry, attachment_bytes_by_name[attachment_name]
                        )
                    )
                else:
                    decoded_value = {
                        key: value
                        for key, value in trace_entry.items()
                        if key not in ("timestamp", "timestamp_us", "type")
                    }

                if summary.message_count == 0:
                    summary.first_value_description = describe_value(decoded_value)
                summary.message_count += 1
        finally:
            for container in video_containers:
                container.close()

    print(f"Recording {recording['id']} with {len(summaries)} sensors")
    for topic, summary in sorted(summaries.items()):
        print(
            f"{topic}: {summary.message_count} messages, "
            f"first {summary.first_value_description}"
        )


def main() -> None:
    """Parse the command line and read the export."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mcap_path", type=Path, help="Exported MCAP file.")
    read_export(parser.parse_args().mcap_path)


if __name__ == "__main__":
    main()
