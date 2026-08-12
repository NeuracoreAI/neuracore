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

# daemon_log_action (the shared daemon.log, independent of storage_state_action)
LOG_PRESERVE = "preserve"
LOG_DELETE = "delete"

# mode
MODE_SEQUENTIAL = "sequential"
MODE_STAGGERED = "staggered"

# producer_channels — thread allocation and producer lifetime.
PRODUCER_SYNCHRONOUS = "synchronous"  # one thread, scoped to one recording
PRODUCER_OLD_PER_THREAD = "old_per_thread"  # thread per stream, one recording
PRODUCER_PER_THREAD = "per_thread"  # thread per stream, whole context lifetime
# per_thread, but some streams run in their own OS process
PRODUCER_MULTI_PROCESS = "multi_process"

# context_duration_mode
DURATION_MODE_FIXED = "fixed"
DURATION_MODE_VARIABLE = "variable"
DURATION_VARIABLE_MIN_FACTOR = 0.75
DURATION_VARIABLE_MAX_FACTOR = 1.25

# video_detail
DETAIL_REALISTIC = "realistic"
DETAIL_FLAT = "flat"

# producer_pacing — how hard the producer may drive the SDK. Only the lifetime
# producers may carry a rate; the rest are refused one when their case is built.
PACING_DEADLINE = "deadline"  # every stream paces
PACING_BURST_VIDEO = "burst-video"  # video only; joints stay paced
PACING_SATURATE = "saturate"  # no stream paces; a spool wedge fails the run
# As PACING_SATURATE, but a wedged spool is retried rather than fatal.
PACING_SATURATE_WITH_BACKOFF = "saturate-with-backoff"

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
PRODUCER_CHANNELS = (
    PRODUCER_SYNCHRONOUS,
    PRODUCER_OLD_PER_THREAD,
    PRODUCER_PER_THREAD,
    PRODUCER_MULTI_PROCESS,
)
DURATION_MODES = (DURATION_MODE_FIXED, DURATION_MODE_VARIABLE)
VIDEO_DETAILS = (DETAIL_REALISTIC, DETAIL_FLAT)
PRODUCER_PACINGS = (
    PACING_DEADLINE,
    PACING_BURST_VIDEO,
    PACING_SATURATE,
    PACING_SATURATE_WITH_BACKOFF,
)

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
ProducerPacing = Literal["deadline", "burst-video", "saturate", "saturate-with-backoff"]
ProducerChannels = Literal[
    "synchronous", "old_per_thread", "per_thread", "multi_process"
]

# The ``nc.log_joint_*`` calls a joint stream makes per frame.
JOINT_KINDS = ("joint_positions", "joint_velocities", "joint_torques")

MAX_TIME_TO_START_S = 20.0
STOP_RECORDING_OVERHEAD_PER_SEC = 0.5
STOP_RECORDING_NO_WAIT_SLA_S = 1.0
STOP_RECORDING_UPLOAD_SLA_PER_JOINT_SAMPLE_S = 1.3e-4
STOP_RECORDING_UPLOAD_SLA_PER_VIDEO_PIXEL_S = 3.0e-7

# Pause after the last stop_recording, so post-stop frames are logged.
PER_THREAD_LOGGING_TAIL_S = 2.0

# PACING_SATURATE_WITH_BACKOFF: retry bounds for a frame the daemon refused.
BACKLOG_BACKOFF_BASE_S = 0.05
BACKLOG_BACKOFF_MAX_S = 1.0
# Backoff one frame may absorb before the stall reads as a wedged daemon.
BACKLOG_STALL_BUDGET_S = 30.0

# RGB tail may lag stop by up to N frame intervals (prevents silent orphaning).
TRAILING_RGB_GAP_FRAME_TOLERANCE = 2

# How long a producer child gets to exit, and to deliver its report.
PRODUCER_PROCESS_JOIN_TIMEOUT_S = 30.0
PRODUCER_PROCESS_REPORT_TIMEOUT_S = 30.0

# Re-check interval, so a child that dies while connecting reports its own
# traceback rather than a timeout.
PRODUCER_PROCESS_READY_POLL_S = 0.1

# How long a producer child gets after terminate() before it reads as leaked.
PRODUCER_PROCESS_TERMINATE_TIMEOUT_S = 5.0

# watch_local_gate_close: poll interval — the width of the bracket it measures
# (see RecordingControlBounds.stop_settled_at) — and its thread join timeout.
GATE_CLOSE_POLL_INTERVAL_S = 0.001
GATE_CLOSE_WATCHER_JOIN_TIMEOUT_S = 5.0

# How far either side of the control calls a condemned frame keeps its reason.
CONDEMNED_PROVENANCE_MARGIN_S = 2.0

# recording_notifications_available: one request is all it has to cover.
NOTIFICATION_PROBE_TIMEOUT_S = 10.0

# Leading sync points a non-owning process may be missing, while it waits for
# the SSE notification (see `late_starting_trace_keys`).
#
# UNCALIBRATED placeholder: only an observed shortfall can size it, and no case
# has produced one yet. Set it from the first run that gets through.
LATE_START_SYNC_POINT_TOLERANCE = 12

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
