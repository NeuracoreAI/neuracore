"""Shared constants for data-daemon test configuration."""

from typing import Literal

# ---------------------------------------------------------------------------
# Environment variable values
# ---------------------------------------------------------------------------

# stop_method
STOP_METHOD_CLI = "cli"
STOP_METHOD_SIGTERM = "sigterm"
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

# timestamp_mode
TIMESTAMP_MODE_MANUAL = "manual"
TIMESTAMP_MODE_REAL = "real"

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
TIMESTAMP_MODES = (TIMESTAMP_MODE_MANUAL, TIMESTAMP_MODE_REAL)

# ---------------------------------------------------------------------------
# Type aliases (for type hints)
# ---------------------------------------------------------------------------

StopMethod = Literal["cli", "sigterm", "sigkill"]
StorageStateAction = Literal["delete", "preserve", "empty"]
TimestampMode = Literal["manual", "real"]

MAX_TIME_TO_START_S = 20.0
STOP_RECORDING_OVERHEAD_PER_SEC = 0.5
BASE_DATASET_READY_TIMEOUT_S = 180.0
MAX_DATASET_READY_TIMEOUT_S = 3600.0
DATASET_POLL_INTERVAL_S = 0.25

FRAME_BYTE_LENGTH = 16
FRAME_GRID_SIZE = 4
FRAME_DEFAULT_FILL_VALUE = 100
FRAME_MAX_COLOR_VALUE = 255
FRAME_HALF_DIVISOR = 2
FRAME_COLOR_CHANNELS = 3
