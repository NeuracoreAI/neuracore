"""SQLAlchemy table definitions for trace state."""

from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    func,
)

from neuracore.data_daemon.models import (
    DataType,
    ProgressReportStatus,
    TraceRegistrationStatus,
    TraceUploadStatus,
    TraceWriteStatus,
)

metadata = MetaData()

traces = Table(
    "traces",
    metadata,
    Column("trace_id", Text, primary_key=True),
    Column(
        "write_status",
        Enum(
            TraceWriteStatus,
            native_enum=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=TraceWriteStatus.PENDING,
    ),
    Column(
        "registration_status",
        Enum(
            TraceRegistrationStatus,
            native_enum=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=TraceRegistrationStatus.PENDING,
    ),
    Column(
        "upload_status",
        Enum(
            TraceUploadStatus,
            native_enum=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=TraceUploadStatus.PENDING,
    ),
    Column("recording_id", Text, nullable=False),
    Column(
        "data_type",
        Enum(
            DataType, native_enum=False, values_callable=lambda x: [e.value for e in x]
        ),
        nullable=True,
    ),
    Column("data_type_name", Text, nullable=True),
    Column("dataset_id", Text, nullable=True),
    Column("dataset_name", Text, nullable=True),
    Column("robot_name", Text, nullable=True),
    Column("robot_id", Text, nullable=True),
    Column("robot_instance", Integer, nullable=True),
    Column("path", Text, nullable=True),
    Column("bytes_written", Integer, nullable=True),
    Column("total_bytes", Integer, nullable=True, default=None),
    Column("bytes_uploaded", Integer, default=0),
    Column(
        "progress_reported",
        Enum(
            ProgressReportStatus,
            native_enum=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=ProgressReportStatus.PENDING,
    ),
    Column("expected_trace_count_reported", Integer, nullable=False, default=0),
    Column("error_code", Text, nullable=True, default=None),
    Column("error_message", Text, nullable=True, default=None),
    Column("stopped_at", DateTime(timezone=False), nullable=True, default=None),
    Column(
        "created_at",
        DateTime(timezone=False),
        nullable=False,
        server_default=func.now(),
    ),
    Column(
        "last_updated",
        DateTime(timezone=False),
        nullable=False,
        server_default=func.now(),
    ),
    Column("num_upload_attempts", Integer, nullable=False, default=0),
    Column("next_retry_at", DateTime(timezone=False), nullable=True, default=None),
)

Index("idx_traces_trace_id", traces.c.trace_id)
Index("idx_traces_recording_id", traces.c.recording_id)
Index("idx_traces_recording_id_stopped_at", traces.c.recording_id, traces.c.stopped_at)
Index(
    "idx_traces_recording_id_upload_progress",
    traces.c.recording_id,
    traces.c.upload_status,
    traces.c.progress_reported,
)
Index("idx_traces_write_status", traces.c.write_status)
Index("idx_traces_registration_status", traces.c.registration_status)
Index("idx_traces_upload_status", traces.c.upload_status)
Index("idx_traces_next_retry_at", traces.c.next_retry_at)
