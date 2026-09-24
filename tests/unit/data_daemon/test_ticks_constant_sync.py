"""Guard that the daemon's tick rate matches the shared tick rate.

The daemon writes capture ticks to `trace.json` and uses the same number as
the video time base, and readers divide by `TICKS_PER_SECOND`, so the two must
be one value.

Parsed by hand, like `test_version_sync.py`, so the check needs no Rust build.
"""

from __future__ import annotations

import re
from pathlib import Path

from neuracore_types.timestamps import TICKS_PER_SECOND

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_LIB = REPO_ROOT / "rust" / "data_daemon_shared" / "src" / "lib.rs"


def _video_spool_ticks_per_second() -> int:
    match = re.search(
        r"pub const VIDEO_SPOOL_TICKS_PER_SECOND: u32 = ([0-9_]+);",
        SHARED_LIB.read_text(encoding="utf-8"),
    )
    assert match, f"No VIDEO_SPOOL_TICKS_PER_SECOND found in {SHARED_LIB}"
    return int(match.group(1).replace("_", ""))


def test_daemon_tick_rate_matches_shared_tick_rate() -> None:
    assert _video_spool_ticks_per_second() == TICKS_PER_SECOND
