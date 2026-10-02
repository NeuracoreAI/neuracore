"""Guard that the daemon's video time base is one microsecond.

The daemon writes the video PTS as the capture time in microseconds, and
readers convert microseconds with `MICROSECONDS_PER_SECOND`, so the two must
be one value.

Parsed by hand, like `test_version_sync.py`, so the check needs no Rust build.
"""

from __future__ import annotations

import re
from pathlib import Path

from neuracore_types.timestamps import MICROSECONDS_PER_SECOND

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_LIB = REPO_ROOT / "rust" / "data_daemon_shared" / "src" / "lib.rs"


def _video_spool_time_base() -> int:
    match = re.search(
        r"pub const VIDEO_SPOOL_TICKS_PER_SECOND: u32 = ([0-9_]+);",
        SHARED_LIB.read_text(encoding="utf-8"),
    )
    assert match, f"No VIDEO_SPOOL_TICKS_PER_SECOND found in {SHARED_LIB}"
    return int(match.group(1).replace("_", ""))


def test_daemon_video_time_base_is_one_microsecond() -> None:
    assert _video_spool_time_base() == MICROSECONDS_PER_SECOND
