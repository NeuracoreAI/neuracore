"""Small helper functions shared across consumer bridge components."""

from __future__ import annotations

from neuracore.data_daemon.models import TraceTransportMetadata


def str_or_none(value: str | int | None) -> str | None:
    """Convert a metadata value to a string when present."""
    return None if value is None else str(value)


def int_or_none(value: str | int | None) -> int | None:
    """Convert a metadata value to an integer when present."""
    return None if value is None else int(value)


def trace_metadata_dict(
    metadata: TraceTransportMetadata | None,
) -> dict[str, str | int | None]:
    """Return trace metadata as a plain dict for downstream consumers."""
    return {} if metadata is None else metadata.to_dict()
