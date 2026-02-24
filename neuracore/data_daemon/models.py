"""Models used by the daemon."""

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from neuracore_types import DATA_TYPE_CONTENT_MAPPING, DataType
from pydantic import BaseModel


def get_content_type(data_type: DataType) -> str:
    """Return the content type for a given DataType."""
    try:
        return DATA_TYPE_CONTENT_MAPPING[data_type]
    except KeyError as exc:
        raise ValueError(f"Unhandled data type: {data_type}") from exc


class CommandType(Enum):
    """Commands sent from the producer to the daemon."""

    OPEN_RING_BUFFER = "open_ring_buffer"
    HEARTBEAT = "heartbeat"
    DATA_CHUNK = "data_chunk"
    TRACE_END = "trace_end"
    RECORDING_STOPPED = "recording_stopped"


class TraceStatus(str, Enum):
    """Lifecycle states for a trace.

    State transitions:
    - (none) + START_TRACE    -> INITIALIZING
    - (none) + TRACE_WRITTEN  -> PENDING_METADATA
    - INITIALIZING + TRACE_WRITTEN -> WRITTEN
    - PENDING_METADATA + START_TRACE  -> WRITTEN
    - WRITTEN -> UPLOADING -> UPLOADED
    - UPLOADING -> PAUSED -> UPLOADING (resume)
    - UPLOADING -> RETRYING -> WRITTEN (retry on failure)
    - Any -> FAILED (on error)
    """

    INITIALIZING = "initializing"
    PENDING_METADATA = "pending_metadata"
    WRITTEN = "written"
    UPLOADING = "uploading"
    RETRYING = "retrying"
    PAUSED = "paused"
    UPLOADED = "uploaded"
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


class ProgressReportStatus(str, Enum):
    """Status of progress report for a trace."""

    PENDING = "pending"
    REPORTED = "reported"


def _parse_progress_reported(value: Any) -> ProgressReportStatus:
    """Parse progress_reported from DB (int 0/1 or enum string) to enum."""
    if value is None or value == 0 or value == "pending":
        return ProgressReportStatus.PENDING
    if value == 1 or value == "reported":
        return ProgressReportStatus.REPORTED
    if isinstance(value, ProgressReportStatus):
        return value
    return ProgressReportStatus(str(value))


@dataclass(frozen=True)
class TraceRecord:
    """Typed representation of a trace row in the state store."""

    trace_id: str
    status: TraceStatus
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
    error_code: TraceErrorCode | None
    error_message: str | None
    created_at: datetime
    last_updated: datetime
    num_upload_attempts: int
    next_retry_at: datetime | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TraceRecord":
        """Build a TraceRecord from a SQLAlchemy mapping row."""
        status_raw = row["status"]
        status = (
            status_raw
            if isinstance(status_raw, TraceStatus)
            else TraceStatus(str(status_raw))
        )
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
            status=status,
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
            error_code=error_code,
            error_message=row.get("error_message"),
            created_at=row["created_at"],
            last_updated=row["last_updated"],
            num_upload_attempts=int(row.get("num_upload_attempts", 0)),
            next_retry_at=row.get("next_retry_at"),
        )


class OpenRingBufferModel(BaseModel):
    """Model for the OPEN_RING_BUFFER command."""

    size: int = 1024


class ManagementModel(BaseModel):
    """Model for management commands from the producer to the daemon."""

    producer_id: str
    command: CommandType
    open_ring_buffer: OpenRingBufferModel | None = None


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
    data: str
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
            data=base64.b64encode(data).decode("ascii"),
        )

    def to_dict(self) -> dict:
        """Convert the TraceRecord to a dict.

        The returned dict will contain the following keys
        with corresponding types:
        - "producer_id": str
        - "trace_id": str
        - "received_at": str
        - "data": str (base64 encoded)

        :return: dict containing the trace record data
        """
        return {
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
        }

    def to_json(self) -> str:
        """Convert the TraceRecord to a JSON string.

        The returned JSON string will contain the following
        keys with corresponding types:
        - "producer_id": str
        - "trace_id": str
        - "received_at": str
        - "data": str (base64 encoded)

        :return: JSON string containing the trace record data
        """
        return json.dumps(self.to_dict())


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
