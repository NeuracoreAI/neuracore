"""Models used by the daemon."""

import base64
import json
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import msgpack
from neuracore_types import DATA_TYPE_CONTENT_MAPPING, DataType
from pydantic import BaseModel

_BATCH_RECORD_LENGTH_FORMAT = "!I"
_BATCH_RECORD_LENGTH_SIZE = 4


def get_content_type(data_type: DataType) -> str:
    """Return the content type for a given DataType."""
    try:
        return DATA_TYPE_CONTENT_MAPPING[data_type]
    except KeyError as exc:
        raise ValueError(f"Unhandled data type: {data_type}") from exc


class CommandType(Enum):
    """Commands sent from the producer to the daemon."""

    OPEN_FIXED_SHARED_SLOTS = "open_fixed_shared_slots"
    SHARED_SLOT_DESCRIPTOR = "shared_slot_descriptor"
    SHARED_SLOT_READY = "shared_slot_ready"
    SHARED_SLOT_OPEN_FAILED = "shared_slot_open_failed"
    SHARED_SLOT_CREDIT_RETURN = "shared_slot_credit_return"
    HEARTBEAT = "heartbeat"
    DATA_CHUNK = "data_chunk"
    BATCHED_JOINT_DATA = "batched_joint_data"
    TRACE_END = "trace_end"
    RECORDING_STOPPED = "recording_stopped"


class TraceStatus(str, Enum):
    """Lifecycle states for a trace.

    State transitions:
    - (none) + START_TRACE    -> INITIALIZING
    - INITIALIZING + TRACE_WRITE_PROGRESS -> WRITING
    - (none) + TRACE_WRITTEN  -> PENDING_METADATA
    - INITIALIZING/WRITING + TRACE_WRITTEN -> WRITTEN
    - PENDING_METADATA + START_TRACE  -> WRITTEN
    - WRITTEN -> UPLOADING -> UPLOADED
    - UPLOADING -> PAUSED -> UPLOADING (resume)
    - UPLOADING -> RETRYING -> WRITTEN (retry on failure)
    - Any -> FAILED (on error)
    """

    INITIALIZING = "initializing"
    WRITING = "writing"
    PENDING_METADATA = "pending_metadata"
    WRITTEN = "written"
    UPLOADING = "uploading"
    RETRYING = "retrying"
    PAUSED = "paused"
    UPLOADED = "uploaded"
    FAILED = "failed"


class TraceWriteStatus(str, Enum):
    """Write/persistence lifecycle for a trace."""

    PENDING = "pending"
    INITIALIZING = "initializing"
    WRITING = "writing"
    PENDING_METADATA = "pending_metadata"
    WRITTEN = "written"
    FAILED = "failed"


class TraceRegistrationStatus(str, Enum):
    """Backend registration lifecycle for a trace."""

    PENDING = "pending"
    REGISTERING = "registering"
    REGISTERED = "registered"
    RETRYING = "retrying"
    FAILED = "failed"


class TraceUploadStatus(str, Enum):
    """Upload lifecycle for a trace."""

    PENDING = "pending"
    QUEUED = "queued"
    UPLOADING = "uploading"
    PAUSED = "paused"
    UPLOADED = "uploaded"
    RETRYING = "retrying"
    FAILED = "failed"


class TraceErrorCode(str, Enum):
    """Standardized error codes for trace failures."""

    UNKNOWN = "unknown"
    WRITE_FAILED = "write_failed"
    ENCODE_FAILED = "encode_failed"
    UPLOAD_FAILED = "upload_failed"
    DISK_FULL = "disk_full"
    NETWORK_ERROR = "network_error"
    PROGRESS_REPORT_ERROR = "progress_report_error"


class TraceRegistrationErrorCode(str, Enum):
    """Standardized error codes for data-trace registration failures."""

    UNKNOWN = "unknown"
    PENDING_RECORDING_NOT_FOUND = "pending_recording_not_found"
    STREAM_REGISTRATION_ERROR = "stream_registration_error"
    REGISTER_DATA_TRACE_FAILED = "register_data_trace_failed"
    NETWORK_ERROR = "network_error"


class ProgressReportStatus(str, Enum):
    """Status of progress report for a recording."""

    PENDING = "pending"
    REPORTING = "reporting"
    REPORTED = "reported"


class RecordingStopReportStatus(str, Enum):
    """Status of backend stop reporting for a recording."""

    PENDING = "pending"
    REPORTING = "reporting"
    REPORTED = "reported"


def _parse_progress_reported(value: Any) -> ProgressReportStatus:
    """Parse progress_reported from DB (int 0/1 or enum string) to enum."""
    if value is None or value == 0 or value == "pending":
        return ProgressReportStatus.PENDING
    if value == "reporting":
        return ProgressReportStatus.REPORTING
    if value == 1 or value == "reported":
        return ProgressReportStatus.REPORTED
    if isinstance(value, ProgressReportStatus):
        return value
    return ProgressReportStatus(str(value))


def _parse_recording_stop_report_status(value: Any) -> RecordingStopReportStatus:
    """Parse stop-report status from DB (int 0/1 or enum string) to enum."""
    if value is None or value == 0 or value == "pending":
        return RecordingStopReportStatus.PENDING
    if value == "reporting":
        return RecordingStopReportStatus.REPORTING
    if value == 1 or value == "reported":
        return RecordingStopReportStatus.REPORTED
    if isinstance(value, RecordingStopReportStatus):
        return value
    return RecordingStopReportStatus(str(value))


def _parse_write_status(value: Any) -> TraceWriteStatus:
    """Parse write_status from DB value to enum."""
    if value is None:
        return TraceWriteStatus.PENDING
    if isinstance(value, TraceWriteStatus):
        return value
    return TraceWriteStatus(str(value))


def _parse_registration_status(value: Any) -> TraceRegistrationStatus:
    """Parse registration_status from DB value to enum."""
    if value is None:
        return TraceRegistrationStatus.PENDING
    if isinstance(value, TraceRegistrationStatus):
        return value
    return TraceRegistrationStatus(str(value))


def _parse_upload_status(value: Any) -> TraceUploadStatus:
    """Parse upload_status from DB value to enum."""
    if value is None:
        return TraceUploadStatus.PENDING
    if isinstance(value, TraceUploadStatus):
        return value
    return TraceUploadStatus(str(value))


@dataclass(frozen=True)
class TraceRecord:
    """Typed representation of a trace row in the state store."""

    trace_id: str
    recording_id: str
    data_type: DataType | None
    data_type_name: str | None
    dataset_id: str | None
    dataset_name: str | None
    robot_name: str | None
    robot_id: str | None
    robot_instance: int | None
    path: str | None
    bytes_written: int | None
    total_bytes: int | None
    bytes_uploaded: int
    progress_reported: ProgressReportStatus
    expected_trace_count_reported: int
    error_code: TraceErrorCode | None
    error_message: str | None
    created_at: datetime
    last_updated: datetime
    num_upload_attempts: int
    next_retry_at: datetime | None
    stopped_at: datetime | None
    write_status: TraceWriteStatus = TraceWriteStatus.PENDING
    registration_status: TraceRegistrationStatus = TraceRegistrationStatus.PENDING
    upload_status: TraceUploadStatus = TraceUploadStatus.PENDING

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TraceRecord":
        """Build a TraceRecord from a SQLAlchemy mapping row."""
        data_type_raw = row.get("data_type")
        data_type = (
            None
            if data_type_raw is None
            else (
                data_type_raw
                if isinstance(data_type_raw, DataType)
                else DataType(str(data_type_raw))
            )
        )
        error_code_raw = row.get("error_code")
        error_code = (
            error_code_raw
            if error_code_raw is None or isinstance(error_code_raw, TraceErrorCode)
            else TraceErrorCode(str(error_code_raw))
        )
        path_raw = row.get("path")
        robot_instance_raw = row.get("robot_instance")
        bytes_written_raw = row.get("bytes_written")
        return cls(
            trace_id=str(row["trace_id"]),
            recording_id=str(row["recording_id"]),
            data_type=data_type,
            data_type_name=row.get("data_type_name"),
            dataset_id=row.get("dataset_id"),
            dataset_name=row.get("dataset_name"),
            robot_name=row.get("robot_name"),
            robot_id=row.get("robot_id"),
            robot_instance=(
                int(robot_instance_raw) if robot_instance_raw is not None else None
            ),
            path=str(path_raw) if path_raw is not None else None,
            bytes_written=(
                int(bytes_written_raw) if bytes_written_raw is not None else None
            ),
            total_bytes=row.get("total_bytes"),
            bytes_uploaded=int(row.get("bytes_uploaded", 0)),
            progress_reported=_parse_progress_reported(row.get("progress_reported")),
            expected_trace_count_reported=row.get("expected_trace_count_reported", 0),
            error_code=error_code,
            error_message=row.get("error_message"),
            created_at=row["created_at"],
            last_updated=row["last_updated"],
            num_upload_attempts=int(row.get("num_upload_attempts", 0)),
            next_retry_at=row.get("next_retry_at"),
            stopped_at=row.get("stopped_at"),
            write_status=_parse_write_status(row.get("write_status")),
            registration_status=_parse_registration_status(
                row.get("registration_status")
            ),
            upload_status=_parse_upload_status(row.get("upload_status")),
        )


class OpenFixedSharedSlotsModel(BaseModel):
    """Producer request to open daemon-owned fixed shared slots."""

    transport_mode: str = "FIXED_SHARED_SLOTS_DAEMON_OWNED"
    control_endpoint: str
    slot_size: int
    slot_count: int


class SharedSlotReadyModel(BaseModel):
    """Daemon response describing one opened shared-slot transport."""

    shm_name: str
    slot_size: int
    slot_count: int


class SharedSlotOpenFailedModel(BaseModel):
    """Daemon response describing why a shared-slot open request failed."""

    error_message: str


class ManagementModel(BaseModel):
    """Model for management commands from the producer to the daemon."""

    producer_id: str
    command: CommandType
    open_fixed_shared_slots: OpenFixedSharedSlotsModel | None = None


@dataclass(frozen=True)
class TraceTransportMetadata:
    """Trace-level metadata carried by transport messages."""

    recording_id: str
    data_type: DataType
    data_type_name: str
    dataset_id: str | None = None
    dataset_name: str | None = None
    robot_name: str | None = None
    robot_id: str | None = None
    robot_instance: int | None = None

    def __getitem__(self, key: str) -> str | int | None:
        """Provide dict-like access for legacy callers and tests."""
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Provide dict-like access with a default for legacy callers."""
        return self.to_dict().get(key, default)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceTransportMetadata | None":
        """Parse trace-level transport metadata from a dict when present."""
        if "recording_id" not in data and "data_type" not in data:
            return None

        recording_id_raw = data.get("recording_id")
        if recording_id_raw is None:
            raise ValueError("recording_id is required when trace metadata is present")

        data_type_raw = data.get("data_type")
        if data_type_raw is None:
            raise ValueError("data_type is required when trace metadata is present")

        robot_instance_raw = data.get("robot_instance")
        return cls(
            recording_id=str(recording_id_raw),
            data_type=(
                data_type_raw
                if isinstance(data_type_raw, DataType)
                else DataType(str(data_type_raw))
            ),
            data_type_name=str(data.get("data_type_name", "")),
            dataset_id=(
                None if data.get("dataset_id") is None else str(data.get("dataset_id"))
            ),
            dataset_name=(
                None
                if data.get("dataset_name") is None
                else str(data.get("dataset_name"))
            ),
            robot_name=(
                None if data.get("robot_name") is None else str(data.get("robot_name"))
            ),
            robot_id=(
                None if data.get("robot_id") is None else str(data.get("robot_id"))
            ),
            robot_instance=(
                None if robot_instance_raw is None else int(robot_instance_raw)
            ),
        )

    def to_dict(self) -> dict[str, str | int | None]:
        """Serialize the metadata to a JSON-friendly dict."""
        return {
            "recording_id": self.recording_id,
            "data_type": self.data_type.value,
            "data_type_name": self.data_type_name,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "robot_name": self.robot_name,
            "robot_id": self.robot_id,
            "robot_instance": self.robot_instance,
        }

    def merged_with(
        self, other: "TraceTransportMetadata"
    ) -> tuple[
        "TraceTransportMetadata", dict[str, tuple[str | int | None, str | int | None]]
    ]:
        """Merge non-empty fields from another metadata instance."""
        merged = self.to_dict()
        mismatches: dict[str, tuple[str | int | None, str | int | None]] = {}
        for key, value in other.to_dict().items():
            if value in (None, ""):
                continue
            existing = merged.get(key)
            if existing in (None, ""):
                merged[key] = value
            elif existing != value:
                mismatches[key] = (existing, value)
        merged_metadata = type(self).from_dict(merged)
        if merged_metadata is None:
            raise ValueError("Merged trace metadata unexpectedly missing")
        return merged_metadata, mismatches


@dataclass(frozen=True)
class SharedMemoryChunkMetadata:
    """Per-chunk metadata written into shared memory."""

    trace_id: str
    chunk_index: int
    total_chunks: int
    trace_metadata: TraceTransportMetadata | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedMemoryChunkMetadata":
        """Parse a shared-memory chunk metadata record from JSON."""
        return cls(
            trace_id=str(data["trace_id"]),
            chunk_index=int(data["chunk_index"]),
            total_chunks=int(data["total_chunks"]),
            trace_metadata=TraceTransportMetadata.from_dict(data),
        )

    def to_dict(self) -> dict[str, str | int | None]:
        """Serialize the shared-memory chunk metadata to a JSON-friendly dict."""
        payload: dict[str, str | int | None] = {
            "trace_id": self.trace_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }
        if self.trace_metadata is not None:
            payload.update(self.trace_metadata.to_dict())
        return payload


@dataclass(frozen=True)
class SharedSlotDescriptor:
    """Descriptor for one packet stored in shared memory."""

    shm_name: str
    slot_id: int
    offset: int
    length: int
    sequence_id: int
    slot_size: int
    ack_endpoint: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedSlotDescriptor":
        """Parse a shared-slot descriptor from a dict payload."""
        return cls(
            shm_name=str(data["shm_name"]),
            slot_id=int(data["slot_id"]),
            offset=int(data["offset"]),
            length=int(data["length"]),
            sequence_id=int(data["sequence_id"]),
            slot_size=int(data["slot_size"]),
            ack_endpoint=(
                None if data.get("ack_endpoint") is None else str(data["ack_endpoint"])
            ),
        )

    def to_dict(self) -> dict[str, str | int | None]:
        """Serialize the descriptor to a JSON-friendly dict."""
        return {
            "shm_name": self.shm_name,
            "slot_id": self.slot_id,
            "offset": self.offset,
            "length": self.length,
            "sequence_id": self.sequence_id,
            "slot_size": self.slot_size,
            "ack_endpoint": self.ack_endpoint,
        }


@dataclass(frozen=True)
class SharedSlotCreditReturn:
    """Credit return for one daemon-owned shared-memory slot."""

    shm_name: str
    slot_id: int
    sequence_id: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedSlotCreditReturn":
        """Parse a slot credit return from a dict payload."""
        return cls(
            shm_name=str(data["shm_name"]),
            slot_id=int(data["slot_id"]),
            sequence_id=int(data["sequence_id"]),
        )

    def to_dict(self) -> dict[str, str | int]:
        """Serialize the credit return to a JSON-friendly dict."""
        return {
            "shm_name": self.shm_name,
            "slot_id": self.slot_id,
            "sequence_id": self.sequence_id,
        }


@dataclass
class DataChunkPayload:
    """Payload for the DATA_CHUNK command."""

    channel_id: str
    recording_id: str
    trace_id: str
    chunk_index: int
    total_chunks: int
    data_type_name: str
    dataset_id: str | None
    dataset_name: str | None
    robot_name: str | None
    robot_id: str | None
    robot_instance: int
    data: bytes
    data_type: DataType

    @property
    def trace_metadata(self) -> TraceTransportMetadata:
        """Return the trace-level metadata for this payload."""
        return TraceTransportMetadata(
            recording_id=self.recording_id,
            data_type=self.data_type,
            data_type_name=self.data_type_name,
            dataset_id=self.dataset_id,
            dataset_name=self.dataset_name,
            robot_name=self.robot_name,
            robot_id=self.robot_id,
            robot_instance=self.robot_instance,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "DataChunkPayload":
        """Construct a DataChunkPayload from a dict.

        The dict should have the following keys with corresponding types:
        - "channel_id": str
        - "trace_id": str
        - "recording_id": str (required)
        - "chunk_index": int
        - "total_chunks": int
        - "data": bytes (base64 encoded)
        - "data_type": str
        - "data_type_name": str | None
        - "dataset_id": str | None
        - "dataset_name": str | None
        - "robot_name": str | None
        - "robot_id": str | None
        - "robot_instance": int

        :param data: dict containing the data chunk payload data
        :return: DataChunkPayload
        """
        robot_instance_raw = data.get("robot_instance")
        if robot_instance_raw is None:
            raise ValueError("robot_instance is required")
        data_type_raw = data.get("data_type")
        data_type = (
            DataType(data_type_raw) if data_type_raw is not None else DataType.CUSTOM_1D
        )
        return cls(
            channel_id=str(data.get("channel_id", "")),
            trace_id=str(data["trace_id"]),
            recording_id=str(data["recording_id"]),
            chunk_index=int(data["chunk_index"]),
            total_chunks=int(data["total_chunks"]),
            data_type_name=data.get("data_type_name", ""),
            dataset_id=data.get("dataset_id"),
            dataset_name=data.get("dataset_name"),
            robot_name=data.get("robot_name"),
            robot_id=data.get("robot_id"),
            robot_instance=int(robot_instance_raw),
            data=base64.b64decode(data["data"]),
            data_type=data_type,
        )

    def to_dict(self) -> dict:
        """Return a dict containing the data chunk payload data.

        The dict will have the following keys with corresponding types:
        - "channel_id": str
        - "trace_id": int
        - "chunk_index": int
        - "total_chunks": int
        - "data": str (base64 encoded)
        - "data_type": str
        - "data_type_name": str | None
        - "dataset_id": str | None
        - "dataset_name": str | None
        - "robot_name": str | None
        - "robot_id": str | None
        - "robot_instance": int

        :return: dict containing the data chunk payload data
        """
        return {
            "channel_id": self.channel_id,
            "trace_id": self.trace_id,
            "recording_id": self.recording_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "data_type_name": self.data_type_name,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "robot_name": self.robot_name,
            "robot_id": self.robot_id,
            "robot_instance": self.robot_instance,
            "data": base64.b64encode(self.data).decode("ascii"),
            "data_type": self.data_type.value,
        }


@dataclass
class BatchedJointDataItemPayload:
    """One joint sample carried inside a batched joint transport message."""

    trace_id: str
    data_type_name: str
    value: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchedJointDataItemPayload":
        """Construct one batched joint item from a dict."""
        return cls(
            trace_id=str(data["trace_id"]),
            data_type_name=str(data["data_type_name"]),
            value=float(data["value"]),
        )

    def to_dict(self) -> dict[str, str | float]:
        """Return a JSON-friendly dict for one batched joint item."""
        return {
            "trace_id": self.trace_id,
            "data_type_name": self.data_type_name,
            "value": self.value,
        }


@dataclass
class BatchedJointDataPayload:
    """Payload for one explicit batched joint transport message."""

    recording_id: str
    timestamp: float
    dataset_id: str | None
    dataset_name: str | None
    robot_name: str | None
    robot_id: str | None
    robot_instance: int
    data_type: DataType
    items: list[BatchedJointDataItemPayload]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchedJointDataPayload":
        """Construct batched joint transport payload from a dict."""
        robot_instance_raw = data.get("robot_instance")
        if robot_instance_raw is None:
            raise ValueError("robot_instance is required")
        data_type_raw = data.get("data_type")
        if data_type_raw is None:
            raise ValueError("data_type is required")
        return cls(
            recording_id=str(data["recording_id"]),
            timestamp=float(data["timestamp"]),
            dataset_id=data.get("dataset_id"),
            dataset_name=data.get("dataset_name"),
            robot_name=data.get("robot_name"),
            robot_id=data.get("robot_id"),
            robot_instance=int(robot_instance_raw),
            data_type=DataType(data_type_raw),
            items=[
                BatchedJointDataItemPayload.from_dict(item)
                for item in list(data.get("items") or [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict for batched joint transport."""
        return {
            "recording_id": self.recording_id,
            "timestamp": self.timestamp,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "robot_name": self.robot_name,
            "robot_id": self.robot_id,
            "robot_instance": self.robot_instance,
            "data_type": self.data_type.value,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass
class MessageEnvelope:
    """JSON-friendly representation of the daemon management message."""

    producer_id: str | None
    command: CommandType
    payload: dict = field(default_factory=dict)
    sequence_number: int | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "MessageEnvelope":
        """Construct a MessageEnvelope from a dict.

        The dict should have the following keys with corresponding types:
        - "producer_id": str
        - "command": str
        - "payload": dict (optional)

        :param data: dict containing the message envelope data
        :return: MessageEnvelope
        """
        producer_id = data.get("producer_id")
        return cls(
            producer_id=str(producer_id) if producer_id is not None else None,
            command=CommandType(data["command"]),
            payload=dict(data.get("payload") or {}),
            sequence_number=(
                int(data["sequence_number"])
                if data.get("sequence_number") is not None
                else None
            ),
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "MessageEnvelope":
        """Construct a MessageEnvelope from a JSON-serialized bytes object.

        The bytes object is expected to contain a JSON-serialized dict
        containing the message envelope data.

        :param raw: bytes object containing the serialized message envelope data
        :return: MessageEnvelope
        """
        parsed = json.loads(raw.decode("utf-8"))
        return cls.from_dict(parsed)

    def to_bytes(self) -> bytes:
        """Serialize the message envelope to a JSON-serialized bytes object.

        The bytes object will contain a JSON-serialized dict containing the
        message envelope data.

        :return: bytes object containing the serialized message envelope data
        """
        return json.dumps({
            "producer_id": self.producer_id,
            "command": self.command.value,
            "payload": self.payload,
            "sequence_number": self.sequence_number,
        }).encode("utf-8")


@dataclass
class CompleteMessage:
    """A record of a completed message."""

    producer_id: str
    trace_id: str
    recording_id: str
    dataset_id: str | None
    dataset_name: str | None
    robot_name: str | None
    robot_id: str | None
    data_type: DataType
    data_type_name: str
    robot_instance: int
    received_at: str
    data: bytes
    final_chunk: bool

    @classmethod
    def from_bytes(
        cls,
        producer_id: str,
        recording_id: str,
        final_chunk: bool,
        trace_id: str,
        data_type: DataType,
        data_type_name: str,
        robot_instance: int,
        data: bytes,
        dataset_id: str | None = None,
        dataset_name: str | None = None,
        robot_name: str | None = None,
        robot_id: str | None = None,
    ) -> "CompleteMessage":
        """Construct a TraceRecord from a producer ID, trace ID, and message data.

        The returned TraceRecord will have the "received_at" field set to the current
        UTC time.

        :param producer_id: The ID of the producer that sent the message.
        :param trace_id: The trace ID of the message.
        :param data_type: The data type of the message payload.
        :param data_type_name: The name of the data type.
        :param robot_instance: The robot instance number.
        :param data: The message data.
        :param dataset_id: The dataset ID for the message payload.
        :param dataset_name: The dataset name for the message payload.
        :param robot_name: The robot name for the message payload.
        :param robot_id: The robot ID for the message payload.
        :return: A TraceRecord containing the provided data.
        """
        return cls(
            producer_id=producer_id,
            trace_id=trace_id,
            recording_id=recording_id,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            robot_name=robot_name,
            robot_id=robot_id,
            data_type=data_type,
            data_type_name=data_type_name,
            robot_instance=robot_instance,
            final_chunk=final_chunk,
            received_at=datetime.now(timezone.utc).isoformat(),
            data=data if isinstance(data, bytes) else bytes(data),
        )

    def to_batch_record(self) -> bytes:
        """Serialize CompleteMessage into a length-prefixed msgpack record."""
        packed = msgpack.packb(
            {
                "producer_id": self.producer_id,
                "trace_id": self.trace_id,
                "recording_id": self.recording_id,
                "dataset_id": self.dataset_id,
                "dataset_name": self.dataset_name,
                "robot_name": self.robot_name,
                "robot_id": self.robot_id,
                "data_type": self.data_type.value,
                "data_type_name": self.data_type_name,
                "robot_instance": self.robot_instance,
                "received_at": self.received_at,
                "data": self.data,
                "final_chunk": self.final_chunk,
            },
            use_bin_type=True,
        )
        return struct.pack(_BATCH_RECORD_LENGTH_FORMAT, len(packed)) + packed

    @classmethod
    def iter_batch_records(cls, raw: bytes) -> list["CompleteMessage"]:
        """Parse length-prefixed msgpack CompleteMessage records from raw bytes."""
        messages: list[CompleteMessage] = []
        offset = 0
        total = len(raw)

        while offset < total:
            if total - offset < _BATCH_RECORD_LENGTH_SIZE:
                raise ValueError("Truncated batch record length header")

            record_len = struct.unpack(
                _BATCH_RECORD_LENGTH_FORMAT,
                raw[offset : offset + _BATCH_RECORD_LENGTH_SIZE],
            )[0]
            offset += _BATCH_RECORD_LENGTH_SIZE

            if total - offset < record_len:
                raise ValueError("Truncated batch record payload")

            payload = raw[offset : offset + record_len]
            offset += record_len

            record = msgpack.unpackb(payload, raw=False)

            messages.append(
                cls(
                    producer_id=str(record["producer_id"]),
                    trace_id=str(record["trace_id"]),
                    recording_id=str(record["recording_id"]),
                    dataset_id=record.get("dataset_id"),
                    dataset_name=record.get("dataset_name"),
                    robot_name=record.get("robot_name"),
                    robot_id=record.get("robot_id"),
                    data_type=DataType(record["data_type"]),
                    data_type_name=str(record["data_type_name"]),
                    robot_instance=int(record["robot_instance"]),
                    received_at=str(record["received_at"]),
                    data=bytes(record["data"]),
                    final_chunk=bool(record["final_chunk"]),
                )
            )

        return messages


def parse_data_type(value: str | DataType) -> DataType:
    """Parse a DataType from a string or return it unchanged.

    Args:
        value: DataType instance or string representation.

    Returns:
        Parsed DataType value.

    Raises:
        ValueError: If the value cannot be parsed as a DataType.
    """
    if isinstance(value, DataType):
        return value
    try:
        return DataType(value)
    except ValueError:
        try:
            return DataType[value]
        except KeyError as exc:
            raise ValueError(f"Unhandled data type: {value}") from exc
