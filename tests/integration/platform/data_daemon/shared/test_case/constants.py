"""Shared constants for data-daemon test configuration."""

import time
from pathlib import Path
from typing import Literal

# cspell:ignore PACINGS
# ---------------------------------------------------------------------------
# Test-state directories and path constants
# ---------------------------------------------------------------------------

DATA_DAEMON_TEST_STATE_ROOT = Path(".data_daemon_test_state")
"""Root directory for all test-local daemon state (DB, recordings, artifacts)."""

DATA_DAEMON_TEST_ARTIFACTS_DIR = (
    DATA_DAEMON_TEST_STATE_ROOT / "artifacts" / time.strftime("%Y%m%d_%H%M%S")
)
"""Timestamped directory where per-test artifact copies are stored."""

OFFLINE_RECORDINGS_ROOT = DATA_DAEMON_TEST_STATE_ROOT / "recordings"
"""Directory used as the offline daemon's recordings root in tests."""

OFFLINE_DB_PATH = DATA_DAEMON_TEST_STATE_ROOT / "state.db"
"""Path used for the offline daemon's SQLite state DB in tests."""

# ---------------------------------------------------------------------------
# Environment variable values
# ---------------------------------------------------------------------------

# stop_method
STOP_METHOD_CLI = "cli"
STOP_METHOD_SIGTERM = "sigterm"
STOP_METHOD_SIGINT = "sigint"
STOP_METHOD_SIGKILL = "sigkill"

# storage_state_action (governs both the SQLite DB and the recordings folder)
STORAGE_STATE_PRESERVE = "preserve"
STORAGE_STATE_EMPTY = "empty"
STORAGE_STATE_DELETE = "delete"

# mode
MODE_SEQUENTIAL = "sequential"
MODE_STAGGERED = "staggered"

# producer_channels
PRODUCER_SYNCHRONOUS = "synchronous"
PRODUCER_PER_THREAD = "per_thread"

# context_duration_mode
DURATION_MODE_FIXED = "fixed"
DURATION_MODE_VARIABLE = "variable"
DURATION_VARIABLE_MIN_FACTOR = 0.75
DURATION_VARIABLE_MAX_FACTOR = 1.25

# run_on_os (values match ``sys.platform``)
OS_LINUX = "linux"
OS_DARWIN = "darwin"
OS_WINDOWS = "win32"

# video_detail
DETAIL_REALISTIC = "realistic"
DETAIL_FLAT = "flat"

# video_pacing
PACING_DEADLINE = "deadline"
PACING_BURST = "burst"

# timestamp_mode
TIMESTAMP_MODE_MANUAL = "manual"
TIMESTAMP_MODE_REAL = "real"
TIMESTAMP_MODE_STOCHASTIC = "stochastic"
# Jitter amplitude as a proportion of half the inter-frame interval, so the
# window scales with the case's fps instead of being pinned to one frame rate.
STOCHASTIC_JITTER_FACTOR = 0.5
# OS-scheduler slack budget for the deadline-lateness assertion in stochastic mode.
SCHEDULER_TOLERANCE_S = 0.05


def stochastic_jitter_window(fps: int) -> float:
    """Max jitter amplitude (seconds) for a stream running at ``fps``.

    A fraction (:data:`STOCHASTIC_JITTER_FACTOR`) of half the inter-frame
    interval, keeping jitter comfortably below the gap between frames.
    """
    return 1 / fps / 2 * STOCHASTIC_JITTER_FACTOR


# ---------------------------------------------------------------------------
# Value sets (tuples for static validation)
# ---------------------------------------------------------------------------

STOP_METHODS = (STOP_METHOD_CLI, STOP_METHOD_SIGTERM, STOP_METHOD_SIGKILL)
STORAGE_STATE_ACTIONS = (
    STORAGE_STATE_DELETE,
    STORAGE_STATE_PRESERVE,
    STORAGE_STATE_EMPTY,
)
MODES = (MODE_SEQUENTIAL, MODE_STAGGERED)
PRODUCER_CHANNELS = (PRODUCER_SYNCHRONOUS, PRODUCER_PER_THREAD)
DURATION_MODES = (DURATION_MODE_FIXED, DURATION_MODE_VARIABLE)
VIDEO_DETAILS = (DETAIL_REALISTIC, DETAIL_FLAT)
VIDEO_PACINGS = (PACING_DEADLINE, PACING_BURST)
TIMESTAMP_MODES = (
    TIMESTAMP_MODE_MANUAL,
    TIMESTAMP_MODE_REAL,
    TIMESTAMP_MODE_STOCHASTIC,
)
OS_ALL = (OS_LINUX, OS_DARWIN, OS_WINDOWS)
# The macOS scheduler cannot hold the per-frame deadlines stochastic cases assert.
OS_EXCEPT_DARWIN = (OS_LINUX, OS_WINDOWS)

# ---------------------------------------------------------------------------
# Type aliases (for type hints)
# ---------------------------------------------------------------------------

StopMethod = Literal["cli", "sigterm", "sigkill"]
StorageStateAction = Literal["delete", "preserve", "empty"]
TimestampMode = Literal["manual", "real", "stochastic"]
VideoDetail = Literal["realistic", "flat"]
VideoPacing = Literal["deadline", "burst"]
TestOs = Literal["linux", "darwin", "win32"]

MAX_TIME_TO_START_S = 20.0
STOP_RECORDING_OVERHEAD_PER_SEC = 0.5
STOP_RECORDING_NO_WAIT_SLA_S = 1.0
STOP_RECORDING_UPLOAD_SLA_PER_JOINT_SAMPLE_S = 1.3e-4
STOP_RECORDING_UPLOAD_SLA_PER_VIDEO_PIXEL_S = 3.0e-7

BASE_DATASET_READY_TIMEOUT_S = 180.0
MAX_DATASET_READY_TIMEOUT_S = 3600.0
DATASET_POLL_INTERVAL_S = 0.25

FRAME_BYTE_LENGTH = 16
FRAME_GRID_SIZE = 4
FRAME_DEFAULT_FILL_VALUE = 100
FRAME_MAX_COLOR_VALUE = 255
FRAME_HALF_DIVISOR = 2
FRAME_COLOR_CHANNELS = 3

# Floor for encoded ``lossless.mp4`` bytes per pixel, asserted on realistic-detail
# video cases. Measured across this matrix through the daemon's own ffmpeg
# arguments: flat frames yield 0.005 (1920x1080) to 0.032 (64x64) bytes/pixel,
# realistic frames 1.18 (1920x1080) to 2.07 (64x64). Small frames sit nearer the
# floor from both directions because a keyframe amortises over fewer pixels, so
# they set the margin: at least 3x above the flat ceiling and 11x below the
# realistic floor.
MIN_ENCODED_BYTES_PER_PIXEL = 0.1
