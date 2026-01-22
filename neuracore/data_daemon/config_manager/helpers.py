"""Helpers for parsing byte-sized CLI arguments."""

import shutil
from pathlib import Path

from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.const import (
    BYTES_PER_MIB,
    DEFAULT_MAX_BANDWIDTH_MIB_S,
    DEFAULT_MIN_BANDWIDTH_MIB_S,
    DEFAULT_RECORDING_ROOT_PATH,
    DEFAULT_STORAGE_FREE_FRACTION,
    DEFAULT_TARGET_DRAIN_HOURS,
    SECONDS_PER_HOUR,
)


def parse_bytes(value: int | str) -> int:
    """Parse a byte quantity from an integer or unit-suffixed string.

    Supported string units (case-insensitive):
        b, k, kb, m, mb, g, gb

    Args:
        value: Raw byte value as an ``int`` or string with an optional unit
            suffix.

    Returns:
        The parsed value in bytes.

    Raises:
        ValueError: If the input cannot be parsed or contains an unknown unit.
    """
    if isinstance(value, int):
        return value

    normalized_value = str(value).strip().lower()

    if normalized_value.isdigit():
        return int(normalized_value)

    numeric_part = ""
    unit_suffix = ""
    for character in normalized_value:
        if character.isdigit():
            numeric_part += character
        else:
            unit_suffix += character

    if not numeric_part or not unit_suffix:
        raise ValueError(f"Invalid byte value: {value!r}")

    base_value = int(numeric_part)
    if unit_suffix == "b":
        multiplier = 1
    elif unit_suffix in {"k", "kb"}:
        multiplier = 1024
    elif unit_suffix in {"m", "mb"}:
        multiplier = 1024**2
    elif unit_suffix in {"g", "gb"}:
        multiplier = 1024**3
    else:
        raise ValueError(f"Unknown byte unit in value: {value!r}")

    return base_value * multiplier


def calculate_storage_limit(record_dir: Path, storage_free_fraction: float) -> int:
    """Calculate the maximum number of bytes to allocate for local storage.

    Args:
        record_dir: Directory on the target filesystem used to determine free space.
        storage_free_fraction: Fraction of free bytes to allocate (e.g. 0.5 for 50%).

    Returns:
        Storage limit in bytes.
    """
    free_bytes = shutil.disk_usage(record_dir).free
    return int(storage_free_fraction * free_bytes)


def build_default_daemon_config(
    storage_free_fraction: float = DEFAULT_STORAGE_FREE_FRACTION,
    target_drain_hours: float = DEFAULT_TARGET_DRAIN_HOURS,
    min_bandwidth_mib_s: float = DEFAULT_MIN_BANDWIDTH_MIB_S,
    max_bandwidth_mib_s: float = DEFAULT_MAX_BANDWIDTH_MIB_S,
    num_threads: int = 1,
) -> DaemonConfig:
    """Build a default daemon configuration based on local disk availability.

    Args:
        storage_free_fraction: Fraction of free disk space to allocate for storage.
        target_drain_hours: Target hours to drain stored data at the computed bandwidth.
        min_bandwidth_mib_s: Minimum upload bandwidth in MiB/s.
        max_bandwidth_mib_s: Maximum upload bandwidth in MiB/s.
        num_threads: Number of worker threads for the daemon.

    Returns:
        A DaemonConfig populated with computed limits and the default recordings path.
    """
    record_dir = DEFAULT_RECORDING_ROOT_PATH
    record_dir.mkdir(parents=True, exist_ok=True)
    storage_limit = calculate_storage_limit(record_dir, storage_free_fraction)

    bandwidth_limit = int(storage_limit / (target_drain_hours * SECONDS_PER_HOUR))
    min_bw = int(min_bandwidth_mib_s * BYTES_PER_MIB)
    max_bw = int(max_bandwidth_mib_s * BYTES_PER_MIB)
    bandwidth_limit = max(min_bw, min(bandwidth_limit, max_bw))

    return DaemonConfig(
        storage_limit=storage_limit,
        bandwidth_limit=bandwidth_limit,
        path_to_store_record=str(record_dir),
        num_threads=num_threads,
        keep_wakelock_while_upload=False,
        offline=False,
        api_key=None,
        current_org_id=None,
    )
