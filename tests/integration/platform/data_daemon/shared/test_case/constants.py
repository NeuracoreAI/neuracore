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
TIMESTAMP_MODE_STOCHASTIC = "stochastic"
STOCHASTIC_JITTER_S = 0.004  # 120fps is 8.3 ms per frame jitter must be less than half
# OS-scheduler slack budget for the deadline-lateness assertion in stochastic mode.
SCHEDULER_TOLERANCE_S = 0.05

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
TIMESTAMP_MODES = (
    TIMESTAMP_MODE_MANUAL,
    TIMESTAMP_MODE_REAL,
    TIMESTAMP_MODE_STOCHASTIC,
)

# ---------------------------------------------------------------------------
# Type aliases (for type hints)
# ---------------------------------------------------------------------------

StopMethod = Literal["cli", "sigterm", "sigkill"]
StorageStateAction = Literal["delete", "preserve", "empty"]
TimestampMode = Literal["manual", "real", "stochastic"]

MAX_TIME_TO_START_S = 20.0
# Floor for the stop_recording budget: scales with recording duration so
# short recordings still get a sane minimum. Also the whole budget for the
# small-joint-count cases, whose upload is trivial.
STOP_RECORDING_OVERHEAD_PER_SEC = 0.5
# stop_recording(wait=False) is fire-and-forget — it never blocks on the
# upload pipeline — so its budget is a flat constant rather than scaling with
# duration or data volume.
STOP_RECORDING_NO_WAIT_SLA_S = 1.0
# stop_recording(wait=True) blocks until every trace has uploaded, so its
# budget scales with joint-data volume: total joint samples
# (duration_sec * joint_count * joint_fps) times an observed per-sample upload
# cost. Calibrated from the 1000-joint case — ~158s observed for 1.8M joint
# samples (~8.8e-5 s/sample) — with ~1.5x headroom for network variance. The
# factor absorbs the three joint data types (positions/velocities/torques)
# logged per joint, since it is fit to the real multi-type observation.
STOP_RECORDING_UPLOAD_SLA_PER_JOINT_SAMPLE_S = 1.3e-4
# Video traces add their own upload cost on top of the joint budget. Encoded
# video volume scales with total pixels logged
# (duration_sec * video_fps * video_count * image_width * image_height), so the
# video budget is that pixel count times an observed per-pixel upload cost.
# Calibrated from the 1-camera 120x120@120Hz video case (Rust daemon) —
# ~4.83s observed stop_recording, of which ~2.2s is the raw joint-upload time
# (25k samples at the 8.8e-5 s/sample raw rate), leaving ~2.6s of
# video-attributable cost over 17.28M video pixels (~1.5e-7 s/pixel) — with
# ~1.5x headroom for network variance.
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
