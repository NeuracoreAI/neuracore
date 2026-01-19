"""Constants for the data daemon."""

import os
import struct
from pathlib import Path

HEARTBEAT_TIMEOUT_SECS = 10
API_URL = os.getenv("NEURACORE_API_URL", "https://api.neuracore.app/api")

TRACE_ID_FIELD_SIZE = 36  # bytes allocated for the trace_id string in chunk headers
DATA_TYPE_FIELD_SIZE = 32  # bytes allocated for the data_type string in chunk headers
CHUNK_HEADER_FORMAT = f"!{TRACE_ID_FIELD_SIZE}s{DATA_TYPE_FIELD_SIZE}sIII"
# trace_id as fixed-length UTF-8 bytes, data_type as fixed-length UTF-8 bytes,
# uint32 chunk_index,
# uint32 total_chunks, uint32 chunk_len
CHUNK_HEADER_SIZE = struct.calcsize(CHUNK_HEADER_FORMAT)

# This mismatches the front nd need to agree on a size
# ...(PFE's - 67,108,864 seems very large and created to suite GCS constraints)
DEFAULT_CHUNK_SIZE = 16384  # (16kb)
DEFAULT_RING_BUFFER_SIZE = 4184304  # (4mb)


BASE_DIR = Path("/tmp/ndd")
SOCKET_PATH = BASE_DIR / "management.sock"
RECORDING_EVENTS_SOCKET_PATH = BASE_DIR / "recording_events.sock"

# Uploads Configuration paths and files
CONFIG_DIR = Path.home() / ".neuracore"
CONFIG_FILE = "config.json"
CONFIG_ENCODING = "utf-8"

REGISTER_TRACES_API_ENDPOINT = "/register-traces"
