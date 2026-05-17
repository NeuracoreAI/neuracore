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

use std::sync::Arc;

use dashmap::DashMap;
use data_daemon_ipc::Envelope;
use tokio::sync::{broadcast, mpsc, Notify};
use tokio::task::JoinHandle;

use crate::lifecycle::signals::ShutdownSignal;
use crate::pipeline::trace_actor::{self, trace_key, TraceActorMessage, TraceKey};
use crate::state::{SqliteStateStore, StateStore};

/// Bounded per-trace queue size, matching the planning doc (§4).
///
/// Sized so the listener can briefly outrun a slow actor (e.g. SQLite write
/// stall) without losing samples, but small enough that backpressure
/// propagates back through the listener → iceoryx2 publisher within ~1 s at
/// 60 fps.
const TRACE_QUEUE_CAPACITY: usize = 64;

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

/// Spawn the dispatcher task and return its inbound `mpsc::Sender`.
///
/// The returned sender is handed to the IPC listener. The dispatcher task
/// owns the matching receiver and the per-trace `DashMap`.
pub fn spawn(
    store: SqliteStateStore,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> (mpsc::Sender<Envelope>, DispatcherHandle) {
    // Capacity sized to absorb a burst of envelopes from the listener while
    // the dispatcher fans them out to per-trace actors; small enough that
    // listener-side backpressure still kicks in promptly under sustained
    // overload.
    let (tx, rx) = mpsc::channel::<Envelope>(256);
    let drained = Arc::new(Notify::new());
    let drained_for_task = Arc::clone(&drained);
    let join = tokio::spawn(async move {
        run(store, rx, shutdown_rx).await;
        drained_for_task.notify_one();
    });
    (tx, DispatcherHandle { join, drained })
}

async fn run(
    store: SqliteStateStore,
    mut rx: mpsc::Receiver<Envelope>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) {
    let store = Arc::new(store);
    let routing: Arc<DashMap<TraceKey, mpsc::Sender<TraceActorMessage>>> = Arc::new(DashMap::new());
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
                if let Some(handle) = handle_envelope(&store, &routing, envelope).await {
                    actor_handles.push(handle);
                }
            }
        }
    }

    // Drop every per-trace sender so the actors observe EOF and exit cleanly.
    routing.clear();

    for handle in actor_handles {
        if let Err(error) = handle.await {
            tracing::warn!(?error, "trace actor join failed during shutdown");
        }
    }
    tracing::info!("dispatcher stopped");
}

/// Route a single envelope to the appropriate per-trace actor.
///
/// Returns the join handle of a newly-spawned actor when one was created so
/// the dispatcher can await it on shutdown. Returns `None` for envelopes that
/// don't trigger actor creation (or that target an actor already running).
async fn handle_envelope(
    store: &Arc<SqliteStateStore>,
    routing: &Arc<DashMap<TraceKey, mpsc::Sender<TraceActorMessage>>>,
    envelope: Envelope,
) -> Option<JoinHandle<()>> {
    let Some(key) = trace_key(&envelope) else {
        match envelope {
            Envelope::StartRecording { recording_id, .. } => {
                if let Err(error) = store.create_recording(&recording_id).await {
                    tracing::warn!(%error, recording_id, "failed to upsert recording row");
                }
            }
            Envelope::StopRecording { recording_id } => {
                tracing::info!(recording_id, "recording stop received");
                // No trace context on the wire — phase 5's recording-scoped
                // bookkeeping closes the loop here. For phase 4 we just log.
            }
            _ => unreachable!("trace_key only returns None for recording-scoped envelopes"),
        }
        return None;
    };

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
        return None;
    }

    // First message for this trace — spawn the actor.
    let (tx, actor_rx) = mpsc::channel(TRACE_QUEUE_CAPACITY);
    routing.insert(key.clone(), tx.clone());
    let actor_store = Arc::clone(store);
    let actor_key = key.clone();
    let join = tokio::spawn(async move {
        trace_actor::run(actor_store, actor_key, actor_rx).await;
    });

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
    Some(join)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{SqliteStateStore, TraceWriteStatus};
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

    #[tokio::test]
    async fn dispatcher_drives_trace_to_written_on_end_trace() {
        let (store, _dir) = open_store().await;
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let (tx, handle) = spawn(store.clone(), shutdown_rx);

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
        })
        .await
        .expect("start trace");
        for index in 0..5 {
            tx.send(Envelope::Frame {
                trace_id: "trace-1".into(),
                timestamp_ns: index,
                payload: vec![index as u8; 4],
            })
            .await
            .expect("frame");
        }
        tx.send(Envelope::EndTrace {
            trace_id: "trace-1".into(),
        })
        .await
        .expect("end trace");

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
        assert_eq!(trace.bytes_written, 20);
    }
}
