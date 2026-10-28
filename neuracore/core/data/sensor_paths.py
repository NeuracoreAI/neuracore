"""Where a recording stores each sensor's files."""

from neuracore_types import DataType
from neuracore_types.utils.name_utils import to_safe_name


def sensor_directory(data_type: DataType, sensor_name: str) -> str:
    """Return a sensor's directory, relative to its recording.

    Args:
        data_type: The sensor's data type.
        sensor_name: The sensor's name, URL-encoded so it stays one path segment.

    Returns:
        The directory, in the bucket and in the local cache alike.
    """
    return f"{data_type.value}/{to_safe_name(sensor_name)}"
