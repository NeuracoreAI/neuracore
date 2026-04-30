"""Constants for the data daemon."""

import os
import pathlib
import struct
from pathlib import Path

HEARTBEAT_TIMEOUT_SECS = 10
NEVER_OPENED_TIMEOUT_SECS = 20
API_URL = os.getenv("NEURACORE_API_URL", "https://api.neuracore.app/api")

TRACE_ID_FIELD_SIZE = 36  # bytes allocated for the trace_id string in chunk headers
DATA_TYPE_FIELD_SIZE = 64  # bytes allocated for the data_type string in chunk headers
CHUNK_HEADER_FORMAT = f"!{TRACE_ID_FIELD_SIZE}s{DATA_TYPE_FIELD_SIZE}sIII"
# trace_id as fixed-length UTF-8 bytes, data_type as fixed-length UTF-8 bytes,
# uint32 chunk_index,
# uint32 total_chunks, uint32 chunk_len
CHUNK_HEADER_SIZE = struct.calcsize(CHUNK_HEADER_FORMAT)

SHARED_MEMORY_RECORD_MAGIC = b"NCR1"
SHARED_MEMORY_RECORD_HEADER_FORMAT = "!4sII"
SHARED_MEMORY_RECORD_HEADER_SIZE = struct.calcsize(
    SHARED_MEMORY_RECORD_HEADER_FORMAT
)

# Shared transport sizing.
# Keep these aligned with frontend/PFE expectations.
DEFAULT_CHUNK_SIZE = 64 * 1024  # 64 KiB
DEFAULT_SHARED_MEMORY_SIZE = 8 * 1024 * 1024  # 8 MiB

# 4K RGB frame: 3840 * 2160 * 3 = 24,883,200 bytes ~= 23.73 MiB.
# A video record must fit in one shared-memory slot, including header + metadata.
DEFAULT_VIDEO_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB
# DEFAULT_VIDEO_SHARED_MEMORY_SIZE = 128 * 1024 * 1024  # 128 MiB total budget
DEFAULT_VIDEO_SHARED_MEMORY_SIZE = 32 * 1024 * 1024  # 32 MiB
DEFAULT_VIDEO_SLOT_OVERHEAD_SIZE = 64 * 1024  # metadata + header headroom
DEFAULT_VIDEO_SEND_QUEUE_MAXSIZE = 0
DEFAULT_VIDEO_SLOT_SIZE = DEFAULT_VIDEO_CHUNK_SIZE + DEFAULT_VIDEO_SLOT_OVERHEAD_SIZE
DEFAULT_VIDEO_SLOT_COUNT = max(
    1, DEFAULT_VIDEO_SHARED_MEMORY_SIZE // DEFAULT_VIDEO_SLOT_SIZE
)
DEFAULT_VIDEO_ACK_TIMEOUT_SECONDS = 5.0
DEFAULT_VIDEO_SLOT_ALLOCATE_TIMEOUT_SECONDS = 5.0


BASE_DIR = Path("/tmp/ndd")
SOCKET_PATH = BASE_DIR / "management.sock"
ACK_BASE_DIR = BASE_DIR / "slot_acks"

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

DEFAULT_STORAGE_FREE_FRACTION = 0.5  # Use 50% of free disk space for local storage.
DEFAULT_TARGET_DRAIN_HOURS = 12.0  # Aim to drain stored data within ~12 hours.
DEFAULT_MIN_BANDWIDTH_MIB_S = 1.0  # Avoid too-slow uploads even on large disks.
DEFAULT_MAX_BANDWIDTH_MIB_S = 20.0  # Cap upload bandwidth to avoid saturating links.

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
