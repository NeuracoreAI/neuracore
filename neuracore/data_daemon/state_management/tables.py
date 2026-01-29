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

from neuracore.data_daemon.models import DataType, TraceStatus

metadata = MetaData()

traces = Table(
    "traces",
    metadata,
    Column("trace_id", Text, primary_key=True),
    Column(
        "status",
        Enum(TraceStatus, native_enum=False),
        nullable=False,
        default=TraceStatus.PENDING,
    ),
    Column("recording_id", Text, nullable=False),
    Column("data_type", Enum(DataType, native_enum=False), nullable=True),
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
    Column("progress_reported", Integer, nullable=False, default=0),
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
)

Index("idx_traces_trace_id", traces.c.trace_id)
Index("idx_traces_status", traces.c.status)
