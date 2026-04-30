from dataclasses import dataclass

from neuracore.data_daemon.communications_management.bridge_chunk_spool import BridgeChunkSpool, ChunkSpoolRef
from neuracore.data_daemon.models import TraceTransportMetadata
from neuracore_types import DataType

@dataclass
class SharedSlotTransportState:
    """Daemon-side state for the shared-slot transport."""

    control_endpoint: str | None = None
    shm_name: str | None = None

    def reset(self) -> None:
        """Clear transport-specific state."""
        self.control_endpoint = None
        self.shm_name = None


@dataclass(frozen=True)
class CompletionChunkWork:
    producer_id: str
    trace_id: str
    recording_id: str
    chunk_index: int
    total_chunks: int
    chunk_spool: BridgeChunkSpool
    chunk_spool_ref: ChunkSpoolRef
    trace_metadata: TraceTransportMetadata | None = None
    fallback_data_type: DataType | None = None