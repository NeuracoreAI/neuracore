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

/// Discard a recording the backend reports cancelled (410 `RECORDING_CANCELLED`).
///
/// Burns its rows so every coordinator drops it, and announces it so uploads
/// still in flight abort. The cancel is stamped as delivered first, since the
/// backend already has it and the cancel notifier must not POST it back.
/// Idempotent.
pub(crate) async fn discard_recording_locally(
    store: &Arc<SqliteStateStore>,
    bus: &EventBus,
    recording_index: i64,
) {
    let discarded_at = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
    let touched = match store.cancel_recording(recording_index, discarded_at).await {
        Ok((_, touched)) => touched,
        Err(error) => {
            tracing::warn!(%error, recording_index, "failed to burn cancelled recording");
            return;
        }
    };
    if let Err(error) = store.mark_recording_cancel_notified(recording_index).await {
        tracing::warn!(%error, recording_index, "could not stamp backend_cancel_notified_at");
    }
    tracing::info!(
        recording_index,
        trace_rows_touched = touched,
        "backend reports this recording cancelled; discarding it"
    );
    bus.publish(DaemonEvent::RecordingCancelled { recording_index });
}

/// Stop sending anything for a recording the backend answered 404 for.
///
/// Fails its unfinished uploads and aborts those in flight, but keeps its
/// files: a 404 carries no error code, so it may be a saved or reaped
/// recording, or not about the recording at all. Idempotent.
pub(crate) async fn stop_uploading_not_found_recording(
    store: &Arc<SqliteStateStore>,
    bus: &EventBus,
    recording_index: i64,
    recording_id: &str,
) {
    match store.mark_recording_not_found(recording_index).await {
        Ok(failed) => tracing::warn!(
            recording_index,
            recording_id,
            uploads_failed = failed,
            "backend has no pending recording with this id; stopped uploading it \
             and kept its local files"
        ),
        Err(error) => {
            tracing::warn!(%error, recording_index, "failed to mark recording not found");
            return;
        }
    }
    bus.publish(DaemonEvent::RecordingNotFound { recording_index });
}
