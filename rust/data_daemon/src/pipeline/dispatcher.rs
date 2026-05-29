//! Routes envelopes to per-trace actors.
//!
//! The dispatcher owns a `DashMap` keyed by `trace_id`. When the first
//! envelope for a trace arrives it spawns a
//! [`trace_actor`](super::trace_actor) task, stores the sender, and forwards
//! the message. Subsequent envelopes look up the existing sender and forward
//! to it.
//!
//! The map is intentionally a `DashMap` rather than a single tokio `Mutex<…>`
//! because trace lookups happen on the hot ingest path; sharded interior
//! mutability gives lock-free reads while still allowing concurrent inserts
//! when the daemon is starting multiple traces in parallel.

use std::collections::HashSet;
use std::sync::Arc;

use dashmap::DashMap;
use data_daemon_ipc::Envelope;
use tokio::sync::{broadcast, mpsc, Notify};
use tokio::task::JoinHandle;

use crate::lifecycle::signals::ShutdownSignal;
use crate::pipeline::trace_actor::{
    self, trace_key, TraceActorContext, TraceActorMessage, TraceKey,
};
use crate::state::{SqliteStateStore, StateStore};

/// Bounded per-trace queue size.
///
/// Sized for the high-dimensionality case (1000-channel scalar trace at
/// ~3 kHz): a smaller cap acts as a forced flush throttle, where a brief
/// SQLite write stall on the trace_actor back-propagates through the listener
/// and stutters every other trace running in the same daemon. 256 absorbs the
/// burst at the cost of ~10 KiB extra of `Envelope` headers per trace, still
/// comfortably below the iceoryx2 publisher's `Block`-on-full strategy.
const TRACE_QUEUE_CAPACITY: usize = 256;

/// Bounded listener → dispatcher channel.
///
/// Sized to absorb a one-second burst of lifecycle envelopes from the
/// integration matrix's parallel-context recordings without forcing the
/// listener loop into per-envelope await stalls.
const DISPATCHER_INBOX_CAPACITY: usize = 1024;

/// Handle owned by the daemon main loop. Drop it on shutdown to close every
/// per-trace actor.
pub struct DispatcherHandle {
    /// Background join handle for the dispatcher task.
    join: JoinHandle<()>,
    /// Fires when the dispatcher loop has exited; used by `shutdown` to wait
    /// for the inbox to fully drain before returning.
    drained: Arc<Notify>,
}

impl DispatcherHandle {
    /// Wait for the dispatcher to finish processing in-flight messages and
    /// the per-trace actors to terminate.
    pub async fn shutdown(self) {
        // The dispatcher's mpsc receiver is dropped when its task returns;
        // that closes every per-trace sender it holds, which in turn ends each
        // actor task once their inbox drains.
        self.drained.notified().await;
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "dispatcher task join failed during shutdown");
        }
    }
}

/// Optional runtime context passed to the dispatcher. Carries pieces of
/// configuration that influence per-recording side effects but that the
/// dispatcher itself does not own.
#[derive(Clone, Default)]
pub struct DispatcherContext {
    /// Org identifier the daemon stamps on every new recording row. Set from
    /// the local `~/.neuracore/config.json` at launch; left as `None` when
    /// the daemon is running offline.
    pub org_id: Option<String>,
    /// Daemon event bus. Used by the dispatcher to publish `TraceWritten`
    /// once a trace actor reports an `EndTrace`. Optional so unit tests can
    /// run the dispatcher without a bus.
    pub event_bus: Option<crate::state::EventBus>,
}

/// Spawn the dispatcher task and return its inbound `mpsc::Sender`.
///
/// The returned sender is handed to the IPC listener. The dispatcher task
/// owns the matching receiver and the per-trace `DashMap`. `actor_context`
/// is shared with every per-trace actor it spawns — recordings root,
/// storage-budget tracker, and the configured video encoder.
#[allow(dead_code)]
pub fn spawn(
    store: SqliteStateStore,
    actor_context: Arc<TraceActorContext>,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> (mpsc::Sender<Envelope>, DispatcherHandle) {
    spawn_with_context(
        store,
        actor_context,
        DispatcherContext::default(),
        shutdown_rx,
    )
}

/// Spawn the dispatcher with an explicit [`DispatcherContext`].
///
/// The cloud coordinators need side-effects keyed by the local org_id and the
/// event bus; this variant accepts both, while [`spawn`] supplies defaults for
/// tests that do not need an event bus or org_id.
pub fn spawn_with_context(
    store: SqliteStateStore,
    actor_context: Arc<TraceActorContext>,
    context: DispatcherContext,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> (mpsc::Sender<Envelope>, DispatcherHandle) {
    let (tx, rx) = mpsc::channel::<Envelope>(DISPATCHER_INBOX_CAPACITY);
    let drained = Arc::new(Notify::new());
    let drained_for_task = Arc::clone(&drained);
    let join = tokio::spawn(async move {
        run(store, actor_context, context, rx, shutdown_rx).await;
        drained_for_task.notify_one();
    });
    (tx, DispatcherHandle { join, drained })
}

async fn run(
    store: SqliteStateStore,
    actor_context: Arc<TraceActorContext>,
    context: DispatcherContext,
    mut rx: mpsc::Receiver<Envelope>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) {
    let store = Arc::new(store);
    let routing: Arc<DashMap<TraceKey, mpsc::Sender<TraceActorMessage>>> = Arc::new(DashMap::new());
    // Reverse index from recording_id → trace_id so `CancelRecording` can
    // reach every actor it owns in one pass. The set entry is removed when
    // the actor reports `EndTrace` (via the routing map's natural eviction
    // path) or when cancellation tears the actor down.
    let recording_traces: Arc<DashMap<String, HashSet<TraceKey>>> = Arc::new(DashMap::new());
    // The dispatcher tracks each spawned actor's `JoinHandle` so it can await
    // their completion on shutdown — otherwise the runtime would tear them
    // down mid-write when the daemon exits.
    let mut actor_handles: Vec<JoinHandle<()>> = Vec::new();

    tracing::info!("dispatcher started");

    loop {
        tokio::select! {
            biased;
            signal = shutdown_rx.recv() => {
                tracing::debug!(?signal, "dispatcher shutting down");
                break;
            }
            envelope = rx.recv() => {
                let Some(envelope) = envelope else {
                    tracing::debug!("dispatcher inbox closed; exiting");
                    break;
                };
                let spawned = handle_envelope(
                    &store,
                    &actor_context,
                    &context,
                    &routing,
                    &recording_traces,
                    envelope,
                )
                .await;
                actor_handles.extend(spawned);
            }
        }
    }

    // Drop every per-trace sender so the actors observe EOF and exit cleanly.
    routing.clear();
    recording_traces.clear();

    for handle in actor_handles {
        if let Err(error) = handle.await {
            tracing::warn!(?error, "trace actor join failed during shutdown");
        }
    }
    tracing::info!("dispatcher stopped");
}

/// Route a single envelope to the appropriate per-trace actor.
///
/// Returns the join handles of any newly-spawned actors so the dispatcher
/// can await them on shutdown. `StopRecording` may spawn one actor per
/// `TraceEnding` whose trace never had a routed envelope; every other
/// envelope spawns at most one actor.
async fn handle_envelope(
    store: &Arc<SqliteStateStore>,
    actor_context: &Arc<TraceActorContext>,
    context: &DispatcherContext,
    routing: &Arc<DashMap<TraceKey, mpsc::Sender<TraceActorMessage>>>,
    recording_traces: &Arc<DashMap<String, HashSet<TraceKey>>>,
    envelope: Envelope,
) -> Vec<JoinHandle<()>> {
    let Some(key) = trace_key(&envelope) else {
        match envelope {
            Envelope::StartRecording { recording_id, .. } => {
                if let Err(error) = store.create_recording(&recording_id).await {
                    tracing::warn!(%error, recording_id, "failed to upsert recording row");
                }
                if let Some(org_id) = context.org_id.as_deref() {
                    if let Err(error) = store.set_recording_org(&recording_id, org_id).await {
                        tracing::warn!(%error, recording_id, "failed to stamp recording org_id");
                    }
                }
                return Vec::new();
            }
            Envelope::StopRecording {
                recording_id,
                trace_endings,
            } => {
                tracing::info!(
                    recording_id,
                    trace_count = trace_endings.len(),
                    "recording stop received"
                );
                let spawned = handle_stop_recording(
                    store,
                    actor_context,
                    routing,
                    recording_traces,
                    &recording_id,
                    trace_endings,
                )
                .await;
                match store.mark_recording_stopped(&recording_id).await {
                    Ok(row) => {
                        tracing::info!(
                            recording_id,
                            stopped_at = ?row.stopped_at,
                            "recording marked stopped"
                        );
                        if let Some(bus) = context.event_bus.as_ref() {
                            bus.publish(crate::state::DaemonEvent::RecordingStopped {
                                recording_id: recording_id.clone(),
                            });
                        }
                    }
                    Err(error) => {
                        tracing::warn!(%error, recording_id, "failed to mark recording stopped");
                    }
                }
                return spawned;
            }
            Envelope::CancelRecording { recording_id } => {
                handle_cancel_recording(store, context, routing, recording_traces, recording_id)
                    .await;
                return Vec::new();
            }
            // `trace_key` also returns `None` for `BatchedFrames`, but the
            // IPC listener expands that into per-trace `Frame`s before the
            // dispatcher ever sees it — so it cannot reach this branch.
            _ => unreachable!("only recording-scoped envelopes reach this branch"),
        }
    };

    // Maintain the recording → trace_id reverse index so `CancelRecording`
    // can reach every actor. We learn the recording from `StartTrace`; later
    // envelopes (`Frame`, `VideoChunkReady`) only carry the trace id and
    // don't update the index.
    if let Envelope::StartTrace { recording_id, .. } = &envelope {
        recording_traces
            .entry(recording_id.clone())
            .or_default()
            .insert(key.clone());
    }

    if let Some(sender) = routing.get(&key) {
        if sender
            .send(TraceActorMessage::Envelope(envelope))
            .await
            .is_err()
        {
            tracing::warn!(
                trace_id = key,
                "trace actor inbox closed; dropping envelope"
            );
        }
        return Vec::new();
    }

    // First message for this trace — spawn the actor.
    let (tx, join) = spawn_actor(store, actor_context, routing, key.clone());

    if tx
        .send(TraceActorMessage::Envelope(envelope))
        .await
        .is_err()
    {
        tracing::warn!(
            trace_id = key,
            "trace actor inbox closed before first envelope"
        );
    }
    vec![join]
}

/// Fan a `StopRecording` envelope's `trace_endings` out to per-trace actors.
///
/// For each `TraceEnding` we send a [`TraceActorMessage::StopAtSequence`] to
/// the trace's actor. If the trace has never had a routed envelope (its
/// `StartTrace` and any data have not yet been observed) we spawn the actor
/// first so it can hold the stop signal — the producer's lossless delivery
/// guarantees the missing envelopes are still in flight; the actor's
/// pre-`StartTrace` buffer absorbs them and the 5 s timer is the watchdog.
async fn handle_stop_recording(
    store: &Arc<SqliteStateStore>,
    actor_context: &Arc<TraceActorContext>,
    routing: &Arc<DashMap<TraceKey, mpsc::Sender<TraceActorMessage>>>,
    recording_traces: &Arc<DashMap<String, HashSet<TraceKey>>>,
    recording_id: &str,
    trace_endings: Vec<data_daemon_ipc::TraceEnding>,
) -> Vec<JoinHandle<()>> {
    let mut spawned = Vec::new();
    for ending in trace_endings {
        let trace_id = ending.trace_id;
        let stop = TraceActorMessage::StopAtSequence {
            final_sequence_number: ending.final_sequence_number,
        };
        let sender = match routing.get(&trace_id) {
            Some(existing) => existing.clone(),
            None => {
                // No envelope has been routed for this trace yet — spawn the
                // actor so it can buffer pending envelopes and observe the
                // stop signal.
                recording_traces
                    .entry(recording_id.to_string())
                    .or_default()
                    .insert(trace_id.clone());
                let (tx, join) = spawn_actor(store, actor_context, routing, trace_id.clone());
                spawned.push(join);
                tx
            }
        };
        if sender.send(stop).await.is_err() {
            tracing::warn!(
                trace_id,
                "trace actor inbox closed; dropping stop_at_sequence"
            );
        }
    }
    spawned
}

/// Spawn a per-trace actor task and register its sender in the routing map.
fn spawn_actor(
    store: &Arc<SqliteStateStore>,
    actor_context: &Arc<TraceActorContext>,
    routing: &Arc<DashMap<TraceKey, mpsc::Sender<TraceActorMessage>>>,
    key: TraceKey,
) -> (mpsc::Sender<TraceActorMessage>, JoinHandle<()>) {
    let (tx, actor_rx) = mpsc::channel(TRACE_QUEUE_CAPACITY);
    routing.insert(key.clone(), tx.clone());
    let actor_store = Arc::clone(store);
    let actor_context = Arc::clone(actor_context);
    let join = tokio::spawn(async move {
        trace_actor::run(actor_store, actor_context, key, actor_rx).await;
    });
    (tx, join)
}

/// Tear down every per-trace actor that belongs to `recording_id`, then mark
/// the recording cancelled in the state store and publish the
/// `RecordingCancelled` event.
async fn handle_cancel_recording(
    store: &Arc<SqliteStateStore>,
    context: &DispatcherContext,
    routing: &Arc<DashMap<TraceKey, mpsc::Sender<TraceActorMessage>>>,
    recording_traces: &Arc<DashMap<String, HashSet<TraceKey>>>,
    recording_id: String,
) {
    tracing::info!(recording_id, "recording cancel received");

    // Take ownership of the reverse-index entry so subsequent traces for the
    // (now-cancelled) recording don't get caught up in a second sweep.
    let traces = recording_traces
        .remove(&recording_id)
        .map(|(_, set)| set)
        .unwrap_or_default();

    let mut senders: Vec<mpsc::Sender<TraceActorMessage>> = Vec::with_capacity(traces.len());
    for trace_id in &traces {
        // `routing.remove` instead of `get` so the actor's sender is taken
        // off the map immediately — any in-flight envelope that races the
        // cancel sees a missing entry and is dropped on the floor (intended:
        // a cancelled recording's data should not survive).
        if let Some((_, sender)) = routing.remove(trace_id) {
            senders.push(sender);
        }
    }

    for sender in &senders {
        if sender.send(TraceActorMessage::Cancel).await.is_err() {
            tracing::debug!(
                recording_id,
                "per-trace actor already exited before cancel arrived"
            );
        }
    }
    // Drop the senders we held locally so each actor observes its inbox
    // closed even if the Cancel message couldn't be delivered (e.g. the
    // actor had already returned).
    drop(senders);

    match store.cancel_recording(&recording_id).await {
        Ok((row, touched)) => {
            tracing::info!(
                recording_id,
                cancelled_at = ?row.cancelled_at,
                trace_rows_touched = touched,
                "recording marked cancelled"
            );
            if let Some(bus) = context.event_bus.as_ref() {
                bus.publish(crate::state::DaemonEvent::RecordingCancelled {
                    recording_id: recording_id.clone(),
                });
            }
        }
        Err(error) => {
            tracing::warn!(%error, recording_id, "failed to mark recording cancelled");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::video_encoder::VideoEncoder;
    use crate::state::{SqliteStateStore, TraceWriteStatus};
    use crate::storage::budget::{StorageBudget, StoragePolicy};
    use std::path::PathBuf;
    use tempfile::TempDir;
    use tokio::sync::broadcast;
    use tokio::time::{timeout, Duration};

    async fn open_store() -> (SqliteStateStore, TempDir) {
        let dir = TempDir::new().expect("tempdir");
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .expect("open store");
        (store, dir)
    }

    fn test_context(recordings_root: PathBuf) -> Arc<TraceActorContext> {
        let policy = StoragePolicy {
            storage_limit_bytes: None,
            min_free_disk_bytes: 0,
            refresh_interval: Duration::from_secs(60),
        };
        let budget = Arc::new(StorageBudget::new(&recordings_root, policy));
        Arc::new(TraceActorContext::new(
            recordings_root,
            budget,
            VideoEncoder::new(),
        ))
    }

    #[tokio::test]
    async fn dispatcher_cancels_recording_and_deletes_on_disk_artefacts() {
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"));
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let bus = crate::state::EventBus::new();
        let mut bus_subscriber = bus.subscribe();
        let dispatcher_context = DispatcherContext {
            org_id: None,
            event_bus: Some(bus.clone()),
        };
        let (tx, handle) = spawn_with_context(
            store.clone(),
            context.clone(),
            dispatcher_context,
            shutdown_rx,
        );

        tx.send(Envelope::StartRecording {
            recording_id: "rec-cancel".into(),
            robot_id: None,
            robot_name: None,
            dataset_id: None,
            dataset_name: None,
        })
        .await
        .expect("start recording");
        tx.send(Envelope::StartTrace {
            recording_id: "rec-cancel".into(),
            trace_id: "trace-cancel".into(),
            data_type: "joints".into(),
            data_type_name: None,
        })
        .await
        .expect("start trace");
        // Ship a couple of frames so the writer is open and the trace
        // directory exists on disk before cancellation lands.
        for index in 0..3i64 {
            let payload = serde_json::to_vec(&serde_json::json!({"i": index})).unwrap();
            tx.send(Envelope::frame(
                "trace-cancel".into(),
                index as u64,
                index,
                None,
                payload,
            ))
            .await
            .expect("frame");
        }

        // Give the actor a chance to apply the first frames before we cancel
        // — otherwise the trace directory check races the writer's lazy
        // open. The dispatcher serialises StartTrace → Frame → Cancel
        // through a single tokio task so a brief yield is enough.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let trace_dir = context
            .recordings_root
            .join("rec-cancel")
            .join("joints")
            .join("trace-cancel");
        assert!(trace_dir.exists(), "writer must have opened before cancel");

        tx.send(Envelope::CancelRecording {
            recording_id: "rec-cancel".into(),
        })
        .await
        .expect("cancel recording");

        drop(tx);
        timeout(Duration::from_secs(2), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        // The on-disk trace directory must be gone.
        assert!(
            !trace_dir.exists(),
            "cancelled trace directory should be removed; found {}",
            trace_dir.display()
        );

        // The DB row reflects the cancellation.
        let recording = store
            .get_recording("rec-cancel")
            .await
            .expect("get recording")
            .expect("recording exists");
        assert!(recording.cancelled_at.is_some());
        let trace = store
            .get_trace("trace-cancel")
            .await
            .expect("get trace")
            .expect("trace exists");
        assert_eq!(trace.write_status, TraceWriteStatus::Failed);
        assert_eq!(trace.upload_status, crate::state::TraceUploadStatus::Failed);

        // The event bus saw the cancellation.
        let mut found_cancel = false;
        while let Ok(event) = bus_subscriber.try_recv() {
            if matches!(
                event,
                crate::state::DaemonEvent::RecordingCancelled { ref recording_id } if recording_id == "rec-cancel"
            ) {
                found_cancel = true;
            }
        }
        assert!(found_cancel, "RecordingCancelled event must be published");
    }

    #[tokio::test]
    async fn dispatcher_drives_trace_to_written_on_stop_recording() {
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"));
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let (tx, handle) = spawn(store.clone(), context.clone(), shutdown_rx);

        tx.send(Envelope::StartRecording {
            recording_id: "rec-1".into(),
            robot_id: None,
            robot_name: None,
            dataset_id: None,
            dataset_name: None,
        })
        .await
        .expect("start recording");
        tx.send(Envelope::StartTrace {
            recording_id: "rec-1".into(),
            trace_id: "trace-1".into(),
            data_type: "joints".into(),
            data_type_name: None,
        })
        .await
        .expect("start trace");
        for index in 0..5i64 {
            // Send valid JSON payloads so the actor writes structured entries
            // into the trace.json array (rather than the byte-count fallback).
            let payload = serde_json::to_vec(&serde_json::json!({"i": index})).unwrap();
            tx.send(Envelope::frame(
                "trace-1".into(),
                index as u64,
                index,
                None,
                payload,
            ))
            .await
            .expect("frame");
        }
        tx.send(Envelope::StopRecording {
            recording_id: "rec-1".into(),
            trace_endings: vec![data_daemon_ipc::TraceEnding {
                trace_id: "trace-1".into(),
                final_sequence_number: 5,
            }],
        })
        .await
        .expect("stop recording");

        // Close the dispatcher inbox so the actor observes EOF after draining.
        drop(tx);
        timeout(Duration::from_secs(2), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        let trace = store
            .get_trace("trace-1")
            .await
            .expect("get trace")
            .expect("trace exists");
        assert_eq!(trace.write_status, TraceWriteStatus::Written);

        // Verify the artefact on disk parses back to exactly what we sent.
        let trace_dir = context
            .recordings_root
            .join("rec-1")
            .join("joints")
            .join("trace-1");
        let bytes = std::fs::read(trace_dir.join("trace.json")).expect("trace.json exists");
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).expect("valid JSON");
        assert_eq!(
            parsed,
            serde_json::json!([
                {"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}, {"i": 4}
            ])
        );
        assert_eq!(trace.total_bytes as u64, bytes.len() as u64);
    }
}
