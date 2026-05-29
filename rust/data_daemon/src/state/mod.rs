//! Daemon state management: SQLite persistence and the broadcast event bus.
//!
//! Defines the [`StateStore`] trait, its [`SqliteStateStore`] implementation,
//! and the [`DaemonEvent`] broadcast bus that the dispatcher, trace actors,
//! and cloud coordinators subscribe to.
//!
//! The trait surface and broadcast variants include items not reachable from
//! every build configuration; the `#[allow(dead_code)]` on the sub-modules
//! keeps the compile clean without hiding the typed surface from doctests.

#[allow(dead_code)]
pub mod events;
#[allow(dead_code)]
pub mod schema;
#[allow(dead_code)]
pub mod store;

#[allow(unused_imports)]
pub use events::{ConnectionState, DaemonEvent, EventBus};
#[allow(unused_imports)]
pub use schema::{
    ProgressReportStatus, RecordingRow, TraceErrorCode, TraceRecord, TraceRegistrationStatus,
    TraceUploadStatus, TraceWriteStatus,
};
#[allow(unused_imports)]
pub use store::{NewRecording, SqliteStateStore, StateStore, StateStoreError, TraceUpdate};
