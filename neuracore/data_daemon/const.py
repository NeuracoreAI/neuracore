"""Constants for the data daemon."""

import os
import pathlib
import struct
from pathlib import Path

HEARTBEAT_TIMEOUT_SECS = 10
NEVER_OPENED_TIMEOUT_SECS = 20
DEFAULT_DAEMON_STARTUP_TIMEOUT_SECONDS = 20
API_URL = os.getenv("NEURACORE_API_URL", "https://api.neuracore.app/api")

TRACE_ID_FIELD_SIZE = 36  # bytes allocated for the trace_id string in chunk headers
DATA_TYPE_FIELD_SIZE = 64  # bytes allocated for the data_type string in chunk headers
CHUNK_HEADER_FORMAT = f"!{TRACE_ID_FIELD_SIZE}s{DATA_TYPE_FIELD_SIZE}sIII"
# trace_id as fixed-length UTF-8 bytes, data_type as fixed-length UTF-8 bytes,
# uint32 chunk_index,
# uint32 total_chunks, uint32 chunk_len
CHUNK_HEADER_SIZE = struct.calcsize(CHUNK_HEADER_FORMAT)

VIDEO_TRANSPORT_PACKET_MAGIC = b"NCR1"
VIDEO_TRANSPORT_PACKET_HEADER_FORMAT = "!4sII"
VIDEO_TRANSPORT_PACKET_HEADER_SIZE = struct.calcsize(
    VIDEO_TRANSPORT_PACKET_HEADER_FORMAT
)

# Transport sizing.
# Keep these aligned with frontend/PFE expectations.
DEFAULT_CHUNK_SIZE = 64 * 1024  # 64 KiB
DEFAULT_TRANSPORT_BUFFER_SIZE = 8 * 1024 * 1024  # 8 MiB

# 4K RGB frame: 3840 * 2160 * 3 = 24,883,200 bytes ~= 23.73 MiB.
# A video chunk must fit in one loaned transport sample, including header + metadata.
DEFAULT_VIDEO_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB
DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE = 0
DEFAULT_VIDEO_SLOT_SIZE = DEFAULT_VIDEO_CHUNK_SIZE + (
    64 * 1024
)  # metadata + header headroom

# iceoryx2 video transport settings.
# One zero-copy publish/subscribe service per producer channel, named
# f"{IOX2_SERVICE_PREFIX}{channel_id}".
IOX2_SERVICE_PREFIX = "neuracore/video/"
# Slots in the daemon subscriber ring buffer per channel. Under overload the
# oldest frames are overwritten (DiscardData semantics).
IOX2_SUBSCRIBER_BUFFER_SIZE = 16
# Historical samples retained so a daemon subscriber that registers slightly
# after the producer starts publishing can still catch up on recent frames.
IOX2_HISTORY_SIZE = 16
# Maximum encoded frame packet (header + metadata + chunk) per loaned slot.
IOX2_MAX_FRAME_BYTES = DEFAULT_VIDEO_SLOT_SIZE


BASE_DIR = Path("/tmp/ndd")
SOCKET_PATH = BASE_DIR / "management.sock"

# Uploads Configuration paths and files
CONFIG_DIR = Path.home() / ".neuracore"
CONFIG_FILE = "config.json"
CONFIG_ENCODING = "utf-8"

REGISTER_TRACES_API_ENDPOINT = "/register-traces"

SENTINEL = object()
DEFAULT_FLUSH_BYTES = 4 * 1024 * 1024  # 4 MiB

MIN_FREE_DISK_BYTES = 32 * 1024 * 1024  # 32 MiB safety margin
STORAGE_REFRESH_SECONDS = 5.0

SECONDS_PER_HOUR = 60 * 60
BYTES_PER_MIB = 1024 * 1024

DEFAULT_RECORDING_ROOT_PATH = (
    pathlib.Path.home() / ".neuracore" / "data_daemon" / "recordings"
)
DEFAULT_DAEMON_DB_PATH = Path.home() / ".neuracore" / "data_daemon" / "state.db"

DEFAULT_STORAGE_FREE_FRACTION = 0.5  # Use 50% of free disk space for local storage
DEFAULT_TARGET_DRAIN_HOURS = 12.0  # Aim to drain stored data within ~12 hours
DEFAULT_MIN_BANDWIDTH_MIB_S = 1.0  # Avoid too-slow uploads even on large disks
DEFAULT_MAX_BANDWIDTH_MIB_S = 20.0  # Cap upload bandwidth to avoid saturating links

# Backend API retry configuration
BACKEND_API_MAX_RETRIES = 3
BACKEND_API_MAX_BACKOFF_SECONDS = 30
BACKEND_API_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

UPLOAD_MAX_RETRIES = 5
UPLOAD_RETRY_BASE_SECONDS = 2
UPLOAD_RETRY_MAX_SECONDS = 300

COMPLETED_RECORDING_RETENTION_HOURS = 24 * 30

# default profile name
DEFAULT_PROFILE_NAME = "default_profile"

DEFAULT_UPLOAD_WAIT_TIMEOUT_SECONDS = 180
DURATION_VARIATION_TOLERANCE_SECONDS = 4
