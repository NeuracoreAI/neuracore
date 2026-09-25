//! Cloud coordinators that drive each trace's lifecycle to the backend: batch
//! registration, resumable uploads, debounced status updates, and the periodic
//! progress reporter. Each exposes a single `spawn_*` entry point so the launch
//! routine can drive ordered shutdown by dropping the handle.

use std::sync::Arc;

use crate::state::{DaemonEvent, EventBus, SqliteStateStore, StateStore};

pub mod progress;
pub mod registration;
pub mod status;
mod upload_transfer;
pub mod uploader;

/// React to the backend reporting a recording absent (HTTP 404).
///
/// Cancelling soft-deletes the recording, so every endpoint touching it 404s
/// from that instant and a background task deletes its uploaded bytes about a
/// minute later. A 404 is therefore not a failure to retry but a statement
/// that the recording was discarded, and anything still writing when that
/// cleanup fires is orphaned in the bucket, unfindable and undeletable.
///
/// Burning the row settles every coordinator at once: queued upload work is
/// dropped, the progress sweep skips the recording, and the announcement
/// aborts uploads already in flight. Idempotent.
pub(crate) async fn discard_recording_locally(
    store: &Arc<SqliteStateStore>,
    bus: &EventBus,
    recording_index: i64,
) {
    let discarded_at = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
    match store.cancel_recording(recording_index, discarded_at).await {
        Ok((_, touched)) => {
            tracing::info!(
                recording_index,
                trace_rows_touched = touched,
                "backend reports this recording discarded; abandoning its work"
            );
            bus.publish(DaemonEvent::RecordingCancelled { recording_index });
        }
        Err(error) => {
            tracing::warn!(
                %error,
                recording_index,
                "failed to burn discarded recording"
            );
        }
    }
}
