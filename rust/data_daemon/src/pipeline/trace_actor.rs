//! Per-trace actor task.
//!
//! Owns the SQLite lifecycle for one trace. Phase 4 intentionally keeps the
//! actor "skeletal" (see `docs/data-daemon-rewrite.md` §Phase 4): it counts
//! frames, advances the write-status state machine, and emits structured log
//! lines. The JSON / NUT writers land in phase 5; the state-machine
//! transitions chosen here are exactly the ones phase 5 will continue to
//! drive, so the SQLite contract — `initializing → writing → written` — is
//! already in place once encoding wires in.

use std::sync::Arc;

use data_daemon_ipc::Envelope;
use tokio::sync::mpsc;

use crate::state::store::TraceUpdate;
use crate::state::{SqliteStateStore, StateStore, TraceWriteStatus};

/// Key identifying one per-trace actor.
///
/// Frame and EndTrace envelopes only carry `trace_id` on the wire, so the
/// dispatcher routes by `trace_id` alone. The actor learns its
/// `recording_id` from the first `StartTrace` it sees.
pub type TraceKey = String;

/// Derive the per-actor routing key from an envelope.
pub fn trace_key(envelope: &Envelope) -> Option<TraceKey> {
    match envelope {
        Envelope::StartTrace { trace_id, .. }
        | Envelope::Frame { trace_id, .. }
        | Envelope::EndTrace { trace_id, .. }
        | Envelope::OpenFrameStream { trace_id, .. } => Some(trace_id.clone()),
        Envelope::StartRecording { .. } | Envelope::StopRecording { .. } => None,
    }
}

/// Message accepted by a per-trace actor.
#[derive(Debug)]
pub enum TraceActorMessage {
    /// A producer-originated envelope routed to this trace.
    Envelope(Envelope),
}

/// Run the per-trace actor until the dispatcher closes the inbox.
pub async fn run(
    store: Arc<SqliteStateStore>,
    trace_id: TraceKey,
    mut inbox: mpsc::Receiver<TraceActorMessage>,
) {
    let mut frame_count: u64 = 0;
    let mut bytes_seen: i64 = 0;
    let mut recording_id: Option<String> = None;

    while let Some(message) = inbox.recv().await {
        let TraceActorMessage::Envelope(envelope) = message;
        match envelope {
            Envelope::StartTrace {
                recording_id: rec_id,
                data_type,
                ..
            } => {
                recording_id = Some(rec_id.clone());
                match store
                    .create_trace(&rec_id, &trace_id, Some(&data_type))
                    .await
                {
                    Ok(_) => tracing::debug!(
                        trace_id,
                        recording_id = rec_id,
                        data_type,
                        "trace initialised"
                    ),
                    Err(error) => tracing::warn!(
                        %error,
                        trace_id,
                        recording_id = rec_id,
                        "failed to create trace row"
                    ),
                }
            }
            Envelope::Frame { payload, .. } => {
                frame_count = frame_count.saturating_add(1);
                bytes_seen = bytes_seen.saturating_add(payload.len() as i64);

                // Move from `initializing` to `writing` on the first frame so
                // the registration coordinator's filter (§5 schema) eventually
                // picks the trace up. Subsequent frames just bump the bytes
                // counter; phase 5 will debounce these to avoid one UPDATE
                // per frame.
                let update = TraceUpdate {
                    write_status: if frame_count == 1 {
                        Some(TraceWriteStatus::Writing)
                    } else {
                        None
                    },
                    bytes_written: Some(bytes_seen),
                    ..TraceUpdate::default()
                };
                if let Err(error) = store.update_trace(&trace_id, update).await {
                    tracing::warn!(
                        %error,
                        trace_id,
                        "failed to update trace bytes_written"
                    );
                }
            }
            Envelope::OpenFrameStream { width, height, .. } => {
                // Phase 4 only records the intent; phase 5 will pick the NUT
                // writer's pixel format and the per-resolution loan pool size
                // from these values.
                tracing::debug!(trace_id, width, height, "video resolution announced");
            }
            Envelope::EndTrace { .. } => {
                let update = TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    total_bytes: Some(bytes_seen),
                    bytes_written: Some(bytes_seen),
                    ..TraceUpdate::default()
                };
                if let Err(error) = store.update_trace(&trace_id, update).await {
                    tracing::warn!(%error, trace_id, "failed to mark trace as written");
                }
                tracing::info!(
                    trace_id,
                    recording_id = recording_id.as_deref().unwrap_or("<unknown>"),
                    frame_count,
                    bytes_seen,
                    "trace finalised"
                );
                // After end-of-trace there is nothing more for this actor to
                // do; returning drops the receiver so the dispatcher's
                // `DashMap::clear` on shutdown is free of dangling senders.
                return;
            }
            Envelope::StartRecording { .. } | Envelope::StopRecording { .. } => {
                // Recording-scoped envelopes never carry a trace_id, so the
                // dispatcher routes them through its own branch — they can't
                // reach a per-trace actor. The match arm exists so adding a
                // new recording-scoped variant is a compile error here rather
                // than a silent ignore.
                unreachable!("recording-scoped envelopes are filtered by trace_key");
            }
        }
    }

    // Inbox closed without an explicit `EndTrace` — typically a shutdown or
    // a stop-recording cancel. Mark the trace as failed so the lifecycle is
    // observable from the DB and the registration coordinator doesn't pick
    // it up. Skip when no StartTrace has been seen — there is no row yet,
    // and writing one in `failed` state would be misleading.
    if recording_id.is_some() {
        let update = TraceUpdate {
            write_status: Some(TraceWriteStatus::Failed),
            bytes_written: Some(bytes_seen),
            ..TraceUpdate::default()
        };
        if let Err(error) = store.update_trace(&trace_id, update).await {
            tracing::warn!(
                %error,
                trace_id,
                "failed to mark trace as failed on shutdown"
            );
        }
    }
}
