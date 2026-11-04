"""Guard that the daemon's time unit constants match neuracore_types.

The daemon writes the video PTS as the capture time in microseconds and
converts microseconds to seconds itself, so its constants and the ones
readers use from `neuracore_types.timestamps` must be the same values.

Parsed by hand, like `test_version_sync.py`, so the check needs no Rust build.
"""

from __future__ import annotations

import re
from pathlib import Path

from neuracore_types.timestamps import (
    MICROSECONDS_PER_SECOND,
    NANOSECONDS_PER_MICROSECOND,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_LIB = REPO_ROOT / "rust" / "data_daemon_shared" / "src" / "lib.rs"


def _shared_constant(name: str) -> float:
    match = re.search(
        rf"pub const {name}: \w+ = ([0-9_.]+);",
        SHARED_LIB.read_text(encoding="utf-8"),
    )
    assert match, f"No {name} found in {SHARED_LIB}"
    return float(match.group(1).replace("_", ""))


def test_daemon_video_time_base_is_one_microsecond() -> None:
    assert _shared_constant("VIDEO_SPOOL_TICKS_PER_SECOND") == MICROSECONDS_PER_SECOND


def test_daemon_unit_constants_match_neuracore_types() -> None:
    assert _shared_constant("MICROSECONDS_PER_SECOND") == MICROSECONDS_PER_SECOND
    assert (
        _shared_constant("NANOSECONDS_PER_MICROSECOND") == NANOSECONDS_PER_MICROSECOND
    )
