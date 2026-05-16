//! Daemon state management: SQLite persistence and the broadcast event bus.
//!
//! Phase 3 of the rewrite (see `docs/data-daemon-rewrite.md`). Defines the
//! [`StateStore`] trait, its [`SqliteStateStore`] implementation, and the
//! [`DaemonEvent`] broadcast bus subscribers will use in Phase 4+.
//!
//! The trait surface and broadcast variants are populated ahead of their first
//! caller — the dispatcher, registration coordinator, and friends land in
//! Phase 4. The `#[allow(dead_code)]` on the sub-modules silences the
//! corresponding "never used" warnings until then; remove it once Phase 4
//! wires the consumers in.

#[allow(dead_code)]
pub mod events;
#[allow(dead_code)]
pub mod schema;
#[allow(dead_code)]
pub mod store;

#[allow(unused_imports)]
pub use events::{DaemonEvent, EventBus};
#[allow(unused_imports)]
pub use schema::{
    ProgressReportStatus, RecordingRow, TraceErrorCode, TraceRecord, TraceRegistrationStatus,
    TraceUploadStatus, TraceWriteStatus,
};
#[allow(unused_imports)]
pub use store::{SqliteStateStore, StateStore, StateStoreError};
