"""Shared constants for data-daemon test configuration."""

import time
from pathlib import Path
from typing import Literal

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

# daemon_log_action (the shared daemon.log, independent of storage_state_action)
LOG_PRESERVE = "preserve"
LOG_DELETE = "delete"

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

# video_detail
DETAIL_REALISTIC = "realistic"
DETAIL_FLAT = "flat"

# Phase-offset amplitude as a proportion of half the inter-frame interval, so the
# window scales with the case's fps instead of being pinned to one frame rate.
RANDOM_PHASE_JITTER_FACTOR = 0.5


def random_phase_jitter_window(fps: int) -> float:
    """Max phase offset (seconds) for a stream running at ``fps``.

    A fraction (:data:`RANDOM_PHASE_JITTER_FACTOR`) of half the inter-frame
    interval, keeping the offset comfortably below the gap between frames so it
    can never reorder a stream.
    """
    return 1 / fps / 2 * RANDOM_PHASE_JITTER_FACTOR


# ---------------------------------------------------------------------------
# Value sets (tuples for static validation)
# ---------------------------------------------------------------------------

STOP_METHODS = (STOP_METHOD_CLI, STOP_METHOD_SIGTERM, STOP_METHOD_SIGKILL)
STORAGE_STATE_ACTIONS = (
    STORAGE_STATE_DELETE,
    STORAGE_STATE_PRESERVE,
    STORAGE_STATE_EMPTY,
)
LOG_ACTIONS = (LOG_DELETE, LOG_PRESERVE)
MODES = (MODE_SEQUENTIAL, MODE_STAGGERED)
PRODUCER_CHANNELS = (PRODUCER_SYNCHRONOUS, PRODUCER_PER_THREAD)
DURATION_MODES = (DURATION_MODE_FIXED, DURATION_MODE_VARIABLE)
VIDEO_DETAILS = (DETAIL_REALISTIC, DETAIL_FLAT)

# ---------------------------------------------------------------------------
# Type aliases (for type hints)
# ---------------------------------------------------------------------------

StopMethod = Literal["cli", "sigterm", "sigkill"]
StorageStateAction = Literal["delete", "preserve", "empty"]
DepthMode = Literal["float16", "float32"]
"""Depth camera sample dtype, matching the wire labels `nc.log_depth()`
derives from the array's own dtype (`image.dtype.name`)."""
LogAction = Literal["preserve", "delete"]
VideoDetail = Literal["realistic", "flat"]

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

# Separates realistic from flat content by encoded archive bytes per pixel.
LOSSLESS_CONTENT_BYTES_PER_PIXEL = 0.1

# ---------------------------------------------------------------------------
# Depth round-trip generation and verification
# ---------------------------------------------------------------------------

# `encode_depth_frame` builds each frame as
# floor + per-frame-base + row-gradient + col-gradient, all expressed as
# fractions of MAX_DEPTH (see neuracore.core.utils.depth_utils.MAX_DEPTH) so
# the pattern automatically stays proportional if that constant ever changes.
# The floor keeps every pixel strictly non-zero; the row/col fractions are
# deliberately unequal so a width/height transpose bug shifts the decoded
# pattern rather than leaving it looking correct.
DEPTH_FRAME_FLOOR_FRACTION = 0.025
DEPTH_FRAME_BASE_FRACTION = 0.375
DEPTH_FRAME_ROW_FRACTION = 0.3
DEPTH_FRAME_COL_FRACTION = 0.15
DEPTH_FRAME_BASE_MODULUS = 997  # prime; spreads per-frame bases pseudo-randomly

# Numerical tolerance for comparing a value retrieved from a synchronized
# recording against the value that was actually passed to `nc.log_depth()`
# (after its source-dtype cast). The only lossy step between those two points
# is the 24-bit depth-to-RGB24 storage quantization
# (MAX_DEPTH / (2**24 - 1) ~= 5.96e-7 m) plus ordinary float32 rounding across
# the encode/decode arithmetic — both far smaller than this. Genuine
# corruption (wrong channel order, wrong dtype, transposed axes) produces
# errors many orders of magnitude larger, so this stays tight enough to catch
# it.
DEPTH_ROUND_TRIP_ATOL_M = 1e-4
