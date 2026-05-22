//! Debounced trace status updater.
//!
//! The uploader pushes [`StatusUpdate`] entries onto an unbounded mpsc; the
//! updater coalesces them into per-recording batches and flushes when one of
//! the following becomes true:
//!
//! - `MAX_BATCH_SIZE` (50) traces are queued.
//! - `IN_PROGRESS_MAX_WAIT` (4 s) elapsed since the batch opened.
//! - A completed-trace entry is in the batch and `COMPLETION_MAX_WAIT`
//!   (0.2 s) has elapsed.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::{interval, MissedTickBehavior};

use crate::api::models::{TraceStatusUpdate, TraceStatusValue};
use crate::api::ApiClient;
use crate::lifecycle::signals::ShutdownSignal;
use crate::state::{SqliteStateStore, StateStore};

/// Maximum number of traces to coalesce before flushing.
pub const MAX_BATCH_SIZE: usize = 50;
/// Maximum age of an in-progress batch before flushing.
pub const IN_PROGRESS_MAX_WAIT: Duration = Duration::from_secs(4);
/// Maximum age of a batch containing a completed trace.
pub const COMPLETION_MAX_WAIT: Duration = Duration::from_millis(200);
/// How long to wait before re-attempting a flush when the recording's
/// `org_id` isn't yet stamped on the row. Picked larger than `MAX_WAIT`
/// triggers above so a perpetually-missing org_id doesn't spin the executor
/// while waiting for the producer's `StartRecording` envelope.
const ORG_RESOLVE_RETRY_BACKOFF: Duration = Duration::from_secs(2);

/// Update emitted by the uploader for the status coordinator to forward to
/// the backend.
#[derive(Debug, Clone)]
pub struct StatusUpdate {
    /// Recording the trace belongs to.
    pub recording_id: String,
    /// Trace identifier.
    pub trace_id: String,
    /// Bytes uploaded so far.
    pub uploaded_bytes: i64,
    /// `true` when this update represents an `UPLOAD_COMPLETE` transition.
    pub completed: bool,
    /// Total bytes once finalised; required when `completed` is `true`.
    pub total_bytes: Option<i64>,
}

impl StatusUpdate {
    /// Build an in-progress (bytes-only) status update.
    pub fn in_progress(recording_id: String, trace_id: String, uploaded_bytes: i64) -> Self {
        Self {
            recording_id,
            trace_id,
            uploaded_bytes,
            completed: false,
            total_bytes: None,
        }
    }

    /// Build a completion update (status=UPLOAD_COMPLETE).
    pub fn completed(recording_id: String, trace_id: String, total_bytes: i64) -> Self {
        Self {
            recording_id,
            trace_id,
            uploaded_bytes: total_bytes,
            completed: true,
            total_bytes: Some(total_bytes),
        }
    }
}

/// Handle returned by [`spawn_status_updater`].
pub struct StatusUpdaterHandle {
    join: JoinHandle<()>,
}

impl StatusUpdaterHandle {
    /// Wait for the status updater to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "status updater join failed");
        }
    }
}

/// Spawn the status updater. Returns the mpsc sender used by the uploader.
pub fn spawn_status_updater(
    store: SqliteStateStore,
    client: Arc<ApiClient>,
    inbox: mpsc::UnboundedReceiver<StatusUpdate>,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> StatusUpdaterHandle {
    let store = Arc::new(store);
    let join = tokio::spawn(async move {
        run(store, client, inbox, shutdown_rx).await;
    });
    StatusUpdaterHandle { join }
}

async fn run(
    store: Arc<SqliteStateStore>,
    client: Arc<ApiClient>,
    mut inbox: mpsc::UnboundedReceiver<StatusUpdate>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) {
    // Per-recording pending batches keyed by recording_id; preserves the
    // last-seen update per trace (later updates supersede earlier ones).
    let mut pending: HashMap<String, RecordingBatch> = HashMap::new();
    // Flush tasks running in the background — spawned by flush_due and the
    // max-batch path so the select loop never blocks on HTTP round-trips.
    let mut background_flushes: JoinSet<Option<RecordingBatch>> = JoinSet::new();
    // Periodic flush ticker — fires every 100 ms regardless of inbox load.
    let mut flush_ticker = interval(Duration::from_millis(100));
    flush_ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            biased;
            signal = shutdown_rx.recv() => {
                tracing::debug!(?signal, "status updater shutting down");
                // Let in-flight flushes finish; re-queue any deferred batches
                // so flush_all gets a chance to send them.
                while let Some(flush_result) = background_flushes.join_next().await {
                    if let Ok(Some(deferred_batch)) = flush_result {
                        pending.insert(deferred_batch.recording_id.clone(), deferred_batch);
                    }
                }
                flush_all(&store, &client, &mut pending).await;
                break;
            }
            // Drain completed background flush tasks without blocking the loop.
            Some(flush_result) = background_flushes.join_next(),
                if !background_flushes.is_empty() =>
            {
                match flush_result {
                    Ok(Some(deferred_batch)) => {
                        pending.insert(deferred_batch.recording_id.clone(), deferred_batch);
                    }
                    Ok(None) => {}
                    Err(panic_err) => {
                        tracing::warn!(?panic_err, "flush_batch task panicked");
                    }
                }
            }
            _ = flush_ticker.tick() => {
                flush_due(&store, &client, &mut pending, &mut background_flushes);
            }
            maybe_update = inbox.recv() => {
                let Some(update) = maybe_update else { break };
                let recording_id = update.recording_id.clone();
                let batch = pending
                    .entry(update.recording_id.clone())
                    .or_insert_with(|| RecordingBatch::new(update.recording_id.clone()));
                batch.add(update);
                if batch.size() >= MAX_BATCH_SIZE {
                    if let Some(batch) = pending.remove(&recording_id) {
                        background_flushes.spawn(flush_batch(
                            Arc::clone(&store),
                            Arc::clone(&client),
                            batch,
                        ));
                    }
                }
            }
        }
    }
}

/// Spawn a background task for every batch whose deadline has passed.
/// Synchronous — never blocks the select loop on HTTP I/O.
fn flush_due(
    store: &Arc<SqliteStateStore>,
    client: &Arc<ApiClient>,
    pending: &mut HashMap<String, RecordingBatch>,
    background_flushes: &mut JoinSet<Option<RecordingBatch>>,
) {
    let now = Instant::now();
    let due_ids: Vec<String> = pending
        .iter()
        .filter(|(_, batch)| now >= batch.deadline())
        .map(|(recording_id, _)| recording_id.clone())
        .collect();
    for recording_id in &due_ids {
        if let Some(batch) = pending.remove(recording_id) {
            background_flushes.spawn(flush_batch(Arc::clone(store), Arc::clone(client), batch));
        }
    }
}

async fn flush_all(
    store: &Arc<SqliteStateStore>,
    client: &Arc<ApiClient>,
    pending: &mut HashMap<String, RecordingBatch>,
) {
    let mut tasks: JoinSet<Option<RecordingBatch>> = JoinSet::new();
    for (_, batch) in pending.drain() {
        tasks.spawn(flush_batch(Arc::clone(store), Arc::clone(client), batch));
    }
    while let Some(result) = tasks.join_next().await {
        if let Err(panic_err) = result {
            tracing::warn!(?panic_err, "flush_batch task panicked on shutdown");
        }
        // Deferred batches (missing org_id) are dropped on shutdown.
    }
}

/// Flush a single recording's batch. Returns the batch back if the recording's
/// org_id isn't available yet (caller should re-insert with deferred deadline),
/// or `None` when the flush was sent (or the batch was empty).
async fn flush_batch(
    store: Arc<SqliteStateStore>,
    client: Arc<ApiClient>,
    mut batch: RecordingBatch,
) -> Option<RecordingBatch> {
    let recording_id = batch.recording_id.clone();
    let org_id = match resolve_org(&store, &recording_id).await {
        Some(org_id) => org_id,
        None => {
            // Re-queue with a fresh `opened_at` pushed
            // `ORG_RESOLVE_RETRY_BACKOFF` into the future so the next
            // `flush_due` skips this batch until the producer has had a
            // chance to stamp the org. Without this, a missing org_id pins
            // `deadline()` permanently in the past and the select loop
            // becomes a busy-wait until the org appears.
            batch.defer(ORG_RESOLVE_RETRY_BACKOFF);
            return Some(batch);
        }
    };
    let updates = batch.into_updates();
    if updates.is_empty() {
        return None;
    }
    let updates_payload: HashMap<String, TraceStatusUpdate> = updates.into_iter().collect();
    match client
        .batch_update_traces(&org_id, &recording_id, &updates_payload)
        .await
    {
        Ok(()) => {
            tracing::debug!(
                recording_id,
                count = updates_payload.len(),
                "flushed status updates"
            );
        }
        Err(error) => {
            tracing::warn!(%error, recording_id, count = updates_payload.len(), "status batch update failed");
        }
    }
    None
}

async fn resolve_org(store: &Arc<SqliteStateStore>, recording_id: &str) -> Option<String> {
    match store.get_recording(recording_id).await {
        Ok(Some(row)) => row.org_id,
        Ok(None) => None,
        Err(error) => {
            tracing::warn!(%error, recording_id, "status updater could not read recording row");
            None
        }
    }
}

#[derive(Debug)]
struct RecordingBatch {
    #[allow(dead_code)]
    recording_id: String,
    opened_at: Instant,
    has_completion: bool,
    updates: HashMap<String, TraceStatusUpdate>,
}

impl RecordingBatch {
    fn new(recording_id: String) -> Self {
        Self {
            recording_id,
            opened_at: Instant::now(),
            has_completion: false,
            updates: HashMap::new(),
        }
    }

    fn add(&mut self, update: StatusUpdate) {
        let entry = self.updates.entry(update.trace_id).or_default();
        entry.uploaded_bytes = Some(update.uploaded_bytes);
        if update.completed {
            entry.status = Some(TraceStatusValue::UploadComplete);
            entry.total_bytes = update.total_bytes.or(entry.total_bytes);
            self.has_completion = true;
        }
    }

    fn size(&self) -> usize {
        self.updates.len()
    }

    fn deadline(&self) -> Instant {
        if self.has_completion {
            self.opened_at + COMPLETION_MAX_WAIT
        } else {
            self.opened_at + IN_PROGRESS_MAX_WAIT
        }
    }

    /// Slide `opened_at` forward by `delay` so the next deadline tick lands
    /// at least `delay` from now. Used by the org-id retry path to space
    /// out flush attempts when the recording's org isn't yet stamped.
    fn defer(&mut self, delay: Duration) {
        // Pin the new `opened_at` so that whatever the current deadline
        // policy returns is at least `delay` from now.
        let target = Instant::now() + delay;
        let policy_wait = if self.has_completion {
            COMPLETION_MAX_WAIT
        } else {
            IN_PROGRESS_MAX_WAIT
        };
        self.opened_at = target.checked_sub(policy_wait).unwrap_or(target);
    }

    fn into_updates(self) -> Vec<(String, TraceStatusUpdate)> {
        self.updates.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_records_completion_flag() {
        let mut batch = RecordingBatch::new("rec".to_string());
        batch.add(StatusUpdate::in_progress(
            "rec".to_string(),
            "t1".to_string(),
            1,
        ));
        assert!(!batch.has_completion);
        batch.add(StatusUpdate::completed(
            "rec".to_string(),
            "t1".to_string(),
            100,
        ));
        assert!(batch.has_completion);
        // The latest update for the same trace_id overrides bytes_uploaded.
        let entry = batch.updates.get("t1").unwrap();
        assert_eq!(entry.uploaded_bytes, Some(100));
        assert!(matches!(
            entry.status,
            Some(TraceStatusValue::UploadComplete)
        ));
    }

    #[test]
    fn defer_slides_deadline_forward_into_future() {
        // The defer path is invoked when the recording's org_id isn't
        // stamped yet. Without it the batch's deadline stays in the past
        // and the select loop spins; with it the next deadline is at
        // least `delay` from now.
        let mut batch = RecordingBatch::new("rec".to_string());
        batch.add(StatusUpdate::in_progress(
            "rec".to_string(),
            "t".to_string(),
            1,
        ));
        // Force the batch's apparent deadline well into the past.
        batch.opened_at = Instant::now() - Duration::from_secs(60);
        assert!(batch.deadline() < Instant::now());

        let delay = Duration::from_secs(2);
        let before = Instant::now();
        batch.defer(delay);
        let deadline = batch.deadline();
        // `deadline` should be at least `delay` from `before` (timing
        // slop ~50ms is generous for CI). The exact value is `before +
        // delay` because the batch is in-progress (IN_PROGRESS_MAX_WAIT
        // is subtracted then re-added by deadline()).
        assert!(deadline >= before + delay - Duration::from_millis(50));
    }

    #[test]
    fn completion_deadline_is_shorter() {
        let mut batch = RecordingBatch::new("rec".to_string());
        let baseline = batch.opened_at + IN_PROGRESS_MAX_WAIT;
        assert!(batch.deadline() <= baseline);
        batch.add(StatusUpdate::completed(
            "rec".to_string(),
            "t".to_string(),
            1,
        ));
        assert!(batch.deadline() < baseline);
    }
}
