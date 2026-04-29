"""Per-trace raw RGB frame spool files."""

from __future__ import annotations

import threading
from pathlib import Path

from .types import _RGBFrameRef, _TraceKey


class RGBTraceSpool:
    """Append raw RGB frame bytes into per-trace ordered backing files."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def append_frame(
        self,
        *,
        trace_key: _TraceKey,
        trace_dir: Path,
        frame_bytes: bytes,
    ) -> _RGBFrameRef:
        """Append one raw frame to the trace's spool file and return its ref."""
        del trace_key
        trace_dir.mkdir(parents=True, exist_ok=True)
        spool_path = trace_dir / "frames.rgb"

        with self._lock:
            with spool_path.open("ab") as handle:
                offset = handle.tell()
                handle.write(frame_bytes)

        return _RGBFrameRef(
            spool_path=spool_path,
            offset=offset,
            length=len(frame_bytes),
        )
