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

# recording_control — who opens and closes the recording window.
CONTROL_LOCAL = "local"  # nc.start_recording/nc.stop_recording, in the test process
CONTROL_REMOTE = "remote"  # the backend's own endpoints, as the web frontend calls them
CONTROL_SPLIT_PROCESS = (
    "split"  # this process starts; a peer process makes the SDK stop call
)

# producer_pacing — when a stream offers its next frame.
PACING_DEADLINE = "deadline"  # one frame per interval, what a real robot does
PACING_BURST_VIDEO = "burst-video"  # video clumps; joints keep their deadlines
PACING_SATURATE = "saturate"  # no stream waits; a spool wedge fails the run

# Frames a burst-video stream withholds and then releases back-to-back.
BURST_VIDEO_FRAMES = 8

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
PRODUCER_CHANNELS = (PRODUCER_SYNCHRONOUS, PRODUCER_PER_THREAD, PRODUCER_MULTI_PROCESS)
DURATION_MODES = (DURATION_MODE_FIXED, DURATION_MODE_VARIABLE)
VIDEO_DETAILS = (DETAIL_REALISTIC, DETAIL_FLAT)
PRODUCER_PACINGS = (PACING_DEADLINE, PACING_BURST_VIDEO, PACING_SATURATE)
RECORDING_CONTROLS = (CONTROL_LOCAL, CONTROL_REMOTE, CONTROL_SPLIT_PROCESS)

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
ProducerPacing = Literal["deadline", "burst-video", "saturate"]
RecordingControl = Literal["local", "remote", "split"]
ProducerChannels = Literal["synchronous", "per_thread", "multi_process"]

# ---------------------------------------------------------------------------
# Names: stream kinds, channels, markers and trace keys
# ---------------------------------------------------------------------------
#
# Every name a test can see is present here, so a rename lands in one place and
# the producers, the assertions and the on-disk lookups cannot drift apart.
#
# - Stream kind: what a plan logs, and the token that moves it to a process.
# - Data type: the DB's name for the trace a kind writes.
# - Channel: one camera or joint a stream logs under.
# - Marker: the CUSTOM_1D series a stream writes beside its payload.
# - Trace key: ``data_type/channel``, the identity a recording is read back by.

# Stream kinds, which are also `StreamPlan.name` values. As a producer
# placement token a kind names every channel it covers; a camera channel name
# (below) names that camera alone.
STREAM_RGB = "rgb"
STREAM_DEPTH = "depth"
STREAM_JOINTS = "joints"  # the synchronous producer's bundled joint stream
STREAM_JOINT_POSITIONS = "joint_positions"
STREAM_JOINT_VELOCITIES = "joint_velocities"
STREAM_JOINT_TORQUES = "joint_torques"

# The ``nc.log_joint_*`` calls a joint stream makes per frame.
JOINT_KINDS = (STREAM_JOINT_POSITIONS, STREAM_JOINT_VELOCITIES, STREAM_JOINT_TORQUES)

# Trace data types, and the stream kind each one is logged by. ``STREAM_JOINTS``
# is absent: it bundles the joint kinds rather than logging a type of its own.
DATA_TYPE_RGB_IMAGES = "RGB_IMAGES"
DATA_TYPE_DEPTH_IMAGES = "DEPTH_IMAGES"
DATA_TYPE_CUSTOM_1D = "CUSTOM_1D"
DATA_TYPE_BY_STREAM = {
    STREAM_RGB: DATA_TYPE_RGB_IMAGES,
    STREAM_DEPTH: DATA_TYPE_DEPTH_IMAGES,
    STREAM_JOINT_POSITIONS: "JOINT_POSITIONS",
    STREAM_JOINT_VELOCITIES: "JOINT_VELOCITIES",
    STREAM_JOINT_TORQUES: "JOINT_TORQUES",
}

CAMERA_NAME_PREFIX = "camera_"
DEPTH_CAMERA_NAME_PREFIX = "depth_camera_"
JOINT_GROUP_NAME_PREFIX = "joints_"
MARKER_NAME_PREFIX = "marker_"

BASE_JOINT_NAMES = [
    "vx300s_left/waist",
    "vx300s_left/shoulder",
    "vx300s_left/elbow",
    "vx300s_left/forearm_roll",
    "vx300s_left/wrist_angle",
    "vx300s_left/wrist_rotate",
    "vx300s_left/left_finger",
    "vx300s_left/right_finger",
    "vx300s_right/waist",
    "vx300s_right/shoulder",
    "vx300s_right/elbow",
    "vx300s_right/forearm_roll",
    "vx300s_right/wrist_angle",
    "vx300s_right/wrist_rotate",
    "vx300s_right/left_finger",
    "vx300s_right/right_finger",
]


def camera_name(index: int) -> str:
    """Return the RGB camera channel name at *index*."""
    return f"{CAMERA_NAME_PREFIX}{index}"


def depth_camera_name(index: int) -> str:
    """Return the depth camera channel name at *index*."""
    return f"{DEPTH_CAMERA_NAME_PREFIX}{index}"


def camera_names(video_count: int) -> list[str]:
    """Return a list of RGB camera names for the given count."""
    return [camera_name(index) for index in range(video_count)]


def depth_camera_names(depth_count: int) -> list[str]:
    """Return a list of depth camera names for the given count.

    Distinct from :func:`camera_names` — depth cameras are independent
    stream identities (``DEPTH_IMAGES/depth_camera_N`` traces) even though a
    depth-enabled case reuses the RGB spec's resolution and frame rate.
    """
    return [depth_camera_name(index) for index in range(depth_count)]


def joint_names_for_count(joint_count: int) -> list[str]:
    """Return a list of joint names of the requested length."""
    if joint_count <= len(BASE_JOINT_NAMES):
        return BASE_JOINT_NAMES[:joint_count]
    generated_names = list(BASE_JOINT_NAMES)
    for index in range(len(BASE_JOINT_NAMES), joint_count):
        generated_names.append(f"synthetic_joint_{index:02d}")
    return generated_names


def joint_group_name(index: int) -> str:
    """Return the placement name of the joint group at *index*."""
    return f"{JOINT_GROUP_NAME_PREFIX}{index}"


def joint_name_groups(
    joint_names: list[str], group_count: int
) -> list[tuple[str, list[str]]]:
    """Split *joint_names* into *group_count* contiguous groups.

    Each group pairs the name the joints are addressed by — a device, as a
    camera is — with the joints it owns; the remainder goes to the first groups.
    Unsplit joints are one group, not a case of their own.
    """
    if not joint_names:
        return []
    base, remainder = divmod(len(joint_names), max(group_count, 1))
    groups: list[tuple[str, list[str]]] = []
    start = 0
    for index in range(max(group_count, 1)):
        size = base + (1 if index < remainder else 0)
        groups.append((joint_group_name(index), joint_names[start : start + size]))
        start += size
    return groups


def marker_name_for(channel_or_kind: str) -> str:
    """Return the marker series a stream logs alongside *channel_or_kind*."""
    return f"{MARKER_NAME_PREFIX}{channel_or_kind}"


def trace_key_for(data_type: str, name: str) -> str:
    """Return the ``data_type/name`` key the DB resolves for one channel."""
    # Deferred: keeps this module free of package imports.
    from neuracore_types.utils import validate_safe_name

    return f"{data_type}/{validate_safe_name(name)}"


CAMERA_0 = camera_name(0)
CAMERA_1 = camera_name(1)
DEPTH_CAMERA_0 = depth_camera_name(0)

MAX_TIME_TO_START_S = 20.0
STOP_RECORDING_OVERHEAD_PER_SEC = 0.5
MAX_RECORDING_DURATION_S = 60 * 5

STOP_RECORDING_NO_WAIT_SLA_S = 1.0
STOP_RECORDING_UPLOAD_SLA_PER_JOINT_SAMPLE_S = 1.3e-4
STOP_RECORDING_UPLOAD_SLA_PER_VIDEO_PIXEL_S = 3.0e-7

# Pause after the last stop_recording, so post-stop frames are logged.
PER_THREAD_LOGGING_TAIL_S = 2.0

# RGB tail may lag stop by up to N frame intervals (prevents silent orphaning).
TRAILING_RGB_GAP_FRAME_TOLERANCE = 2

CHILD_PROCESS_JOIN_TIMEOUT_S = 30.0
CHILD_PROCESS_REPORT_TIMEOUT_S = 30.0

# Re-check interval, so a child that dies while connecting reports its own
# traceback rather than a timeout.
CHILD_PROCESS_READY_POLL_S = 0.1

CHILD_PROCESS_TERMINATE_TIMEOUT_S = 5.0

# How long after the stop call the window's publish-clock bound can still fall:
# the stop envelope is published before the flush, so only a robot lookup,
# disarming the streams and one IPC publish sit inside this.
STOP_PUBLISH_SKEW_S = 0.25

# Remote control: the HTTP timeout on the backend's own start/stop endpoints,
# how long the stop may take to come back round to this process, and how often
# the gate is checked for either.
REMOTE_CONTROL_REQUEST_TIMEOUT_S = 20.0
REMOTE_STOP_PROPAGATION_SLA_S = 2.0
REMOTE_GATE_POLL_INTERVAL_S = 0.025

REMOTE_START_ANNOUNCEMENT_SLA_S = 3.0

SPLIT_CONTROL_ACK_POLL_S = 0.05

# How far either side of the control calls a condemned frame keeps its reason.
CONDEMNED_PROVENANCE_MARGIN_S = 2.0

# Leading sync points a non-owning process may be missing, over the interval
# between the window opening and the daemon announcing it (see
# `late_starting_trace_keys`).
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
