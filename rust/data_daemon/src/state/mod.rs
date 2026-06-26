//! Daemon state management: SQLite persistence and the broadcast event bus.
//!
//! Defines the [`StateStore`] trait, its [`SqliteStateStore`] implementation,
//! and the [`DaemonEvent`] broadcast bus that the dispatcher, trace actors,
//! and cloud coordinators subscribe to.

pub mod events;
pub mod schema;
pub mod store;
pub mod trace_event_database_writer;

#[allow(unused_imports)]
pub use events::{ConnectionState, DaemonEvent, EventBus};
#[allow(unused_imports)]
pub use schema::{
    ProgressReportStatus, RecordingRow, TraceErrorCode, TraceRecord, TraceRegistrationStatus,
    TraceUploadStatus, TraceWriteStatus,
};
#[allow(unused_imports)]
pub use store::{
    CoalescedTraceWrite, NewRecording, SqliteStateStore, StateStore, StateStoreError, TraceUpdate,
};
#[allow(unused_imports)]
pub use trace_event_database_writer::{TraceEventDatabaseWriter, TraceWriteHandle};
