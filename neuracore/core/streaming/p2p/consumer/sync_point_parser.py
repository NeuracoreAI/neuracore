"""This module provides utilities for parsing and merging SynchronizedPoint data."""

from typing import cast

from neuracore_types import (
    DATA_TYPE_TO_NC_DATA_CLASS,
    DataType,
    NCData,
    NCDataUnion,
    RobotStreamTrack,
    SynchronizedPoint,
)
from pydantic import ValidationError

from neuracore.core.utils.image_string_encoder import ImageStringEncoder


def parse_sync_point(
    message_data: str, track_details: RobotStreamTrack
) -> SynchronizedPoint:
    """Parse a JSON message into a SynchronizedPoint based on track details.

    Args:
        message_data: The JSON string containing the data.
        track_details: RobotStreamTrack object describing the data.

    Returns:
        SynchronizedPoint: A SynchronizedPoint object containing the parsed data.

    Raises:
        ValueError: If the track data_type is unsupported or data validation fails.
    """
    try:
        data_type: DataType = track_details.data_type
        label: str = track_details.label

        # Get the appropriate data class from the mapping
        data_class: type[NCData] | None = DATA_TYPE_TO_NC_DATA_CLASS.get(data_type)
        if data_class is None:
            raise ValueError(f"Unsupported track data_type: {data_type}")

        # Parse the JSON data using the appropriate class
        data: NCData = data_class.model_validate_json(message_data)
        data = cast(NCDataUnion, data)

        # Decode image data
        if data_type in (DataType.RGB_IMAGES, DataType.DEPTH_IMAGES):
            data.frame = ImageStringEncoder.decode_image(data.frame)

        return SynchronizedPoint(
            timestamp=data.timestamp,
            data={data_type: {label: data}},
        )

    except ValidationError:
        raise ValueError("Invalid or unsupported data")


def merge_sync_points(*args: SynchronizedPoint) -> SynchronizedPoint:
    """Merge multiple SynchronizedPoint objects into a single SynchronizedPoint.

    Properties with later timestamps  will override earlier data.
    The timestamp of the combined sync point will be that of the latest sync point.

    If no sync points are provided, an empty SynchronizedPoint is returned.

    Args:
        *args: Variable number of SynchronizedPoint objects to merge.

    Returns:
        SynchronizedPoint: A new SynchronizedPoint object containing the merged data.
    """
    if len(args) == 0:
        return SynchronizedPoint()

    # Sort by timestamp so that later points override earlier ones.
    sorted_points = sorted(args, key=lambda x: x.timestamp)

    merged_synced_data: dict[DataType, dict[str, NCDataUnion]] = {}

    for sync_point in sorted_points:
        for data_type, values in sync_point.data.items():
            if data_type not in merged_synced_data:
                merged_synced_data[data_type] = {}
            merged_synced_data[data_type].update(values)

    return SynchronizedPoint(
        timestamp=sorted_points[-1].timestamp,
        data=merged_synced_data,
    )
