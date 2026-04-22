"""Path helpers for MCAP topic and payload traversal.

Per MCAP, channel topics are opaque strings. They are not hierarchical paths in
the file format itself. We split on `.` only because importer config uses
`<topic>.<field>` syntax to address nested fields in decoded payload objects.
"""

from __future__ import annotations

from typing import Any

from neuracore.importer.core.exceptions import ImportError


def split_topic_path(source: str) -> tuple[str, list[str]]:
    """Split a source path into topic and nested field components.

    Examples:
    - `/camera/image.data` -> (`/camera/image`, [`data`])
    - `/joint_states` -> (`/joint_states`, [])
    - `topic.field` -> (`topic`, [`field`])
    """
    value = source.strip()
    if not value:
        raise ImportError("Source must include a topic.")

    topic, sep, subpath = value.partition(".")
    if not topic:
        raise ImportError(f"Invalid source '{source}': topic segment is empty.")

    path = [part for part in subpath.split(".") if part] if sep else []
    return topic, path


def resolve_path(data: Any, path: list[str]) -> Any:
    """Resolve a nested path against dict/object/list payloads."""
    current = data
    for part in path:
        current = _resolve_path_part(current, part)
    return current


def _resolve_path_part(data: Any, part: str) -> Any:
    """Resolve one path segment from dicts, objects, or indexable containers."""
    if isinstance(data, dict):
        if part in data:
            return data[part]
        if part.isdigit():
            numeric_key = int(part)
            if numeric_key in data:
                return data[numeric_key]
        raise ImportError(f"Key '{part}' not found while resolving message path.")

    if hasattr(data, part):
        return getattr(data, part)

    if part.isdigit():
        index = int(part)
        try:
            return data[index]
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                f"Index {index} is unavailable while resolving message path: {exc}"
            ) from exc

    try:
        return data[part]
    except Exception as exc:  # noqa: BLE001
        raise ImportError(f"Failed to resolve '{part}' from payload: {exc}") from exc
