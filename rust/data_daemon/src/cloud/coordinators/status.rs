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
use crate::cloud::OrgIdRx;
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::store::MAX_COMPLETION_REPORT_ATTEMPTS;
use crate::state::{RecordingRow, SqliteStateStore, StateStore};

/// Maximum number of traces to coalesce before flushing.
pub const MAX_BATCH_SIZE: usize = 50;
/// Maximum age of an in-progress batch before flushing.
pub const IN_PROGRESS_MAX_WAIT: Duration = Duration::from_secs(4);
/// Maximum age of a batch containing a completed trace.
pub const COMPLETION_MAX_WAIT: Duration = Duration::from_millis(200);
/// How long to wait before re-attempting a flush when no current `org_id` is
/// configured yet, or the recording's cloud id hasn't been assigned. Picked
/// larger than the `MAX_WAIT` triggers above so a perpetually-missing org
/// doesn't spin the executor while waiting for login / org selection.
const ORG_RESOLVE_RETRY_BACKOFF: Duration = Duration::from_secs(2);
/// Ignore unreported completions younger than this during a reconcile pass —
/// a freshly-queued completion normally stamps within the flush deadline, so
/// only traces that have sat unacknowledged for a while are re-driven.
const COMPLETION_RECONCILE_GRACE: Duration = Duration::from_secs(10);

/// Update emitted by the uploader for the status coordinator to forward to
/// the backend.
#[derive(Debug, Clone)]
pub struct StatusUpdate {
    /// Recording the trace belongs to (local `recording_index`).
    pub recording_index: i64,
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
    pub fn in_progress(recording_index: i64, trace_id: String, uploaded_bytes: i64) -> Self {
        Self {
            recording_index,
            trace_id,
            uploaded_bytes,
            completed: false,
            total_bytes: None,
        }
    }

    /// Build a completion update (status=UPLOAD_COMPLETE).
    pub fn completed(recording_index: i64, trace_id: String, total_bytes: i64) -> Self {
        Self {
            recording_index,
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
    org_rx: OrgIdRx,
    inbox: mpsc::UnboundedReceiver<StatusUpdate>,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> StatusUpdaterHandle {
    let store = Arc::new(store);
    let join = tokio::spawn(async move {
        run(store, client, org_rx, inbox, shutdown_rx).await;
    });
    StatusUpdaterHandle { join }
}

async fn run(
    store: Arc<SqliteStateStore>,
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    mut inbox: mpsc::UnboundedReceiver<StatusUpdate>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) {
    // Per-recording pending batches keyed by recording_index; preserves the
    // last-seen update per trace (later updates supersede earlier ones).
    let mut pending: HashMap<i64, RecordingBatch> = HashMap::new();
    // Flush tasks running in the background — spawned by flush_due and the
    // max-batch path so the select loop never blocks on HTTP round-trips.
    let mut background_flushes: JoinSet<Option<RecordingBatch>> = JoinSet::new();
    // Periodic flush ticker — fires on the STATUS_FLUSH cadence regardless of inbox load.
    let mut flush_ticker = interval(crate::intervals::STATUS_FLUSH);
    flush_ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    // Reconcile ticker — re-drives completions whose backend ack never landed
    // (dropped batch, crash between ack and stamp, failed stamp write). The
    // reclaim gate reads `completion_reported_at`, so without this a single
    // lost batch would retain its recording forever.
    let mut reconcile_ticker = interval(crate::intervals::COMPLETION_RECONCILE);
    reconcile_ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            biased;
            signal = shutdown_rx.recv() => {
                tracing::debug!(?signal, "status updater shutting down");
                // Let in-flight flushes finish; re-queue any deferred batches
                // so flush_all gets a chance to send them.
                while let Some(flush_result) = background_flushes.join_next().await {
                    if let Ok(Some(deferred_batch)) = flush_result {
                        pending.insert(deferred_batch.recording_index, deferred_batch);
                    }
                }
                flush_all(&store, &client, &org_rx, &mut pending).await;
                break;
            }
            // Drain completed background flush tasks without blocking the loop.
            Some(flush_result) = background_flushes.join_next(),
                if !background_flushes.is_empty() =>
            {
                match flush_result {
                    Ok(Some(deferred_batch)) => {
                        pending.insert(deferred_batch.recording_index, deferred_batch);
                    }
                    Ok(None) => {}
                    Err(panic_err) => {
                        tracing::warn!(?panic_err, "flush_batch task panicked");
                    }
                }
            }
            _ = flush_ticker.tick() => {
                flush_due(&store, &client, &org_rx, &mut pending, &mut background_flushes);
            }
            _ = reconcile_ticker.tick() => {
                reconcile_unreported_completions(
                    &store,
                    &mut pending,
                    COMPLETION_RECONCILE_GRACE,
                ).await;
            }
            maybe_update = inbox.recv() => {
                let Some(update) = maybe_update else { break };
                let recording_index = update.recording_index;
                let batch = pending
                    .entry(recording_index)
                    .or_insert_with(|| RecordingBatch::new(recording_index));
                batch.add(update);
                if batch.size() >= MAX_BATCH_SIZE {
                    if let Some(batch) = pending.remove(&recording_index) {
                        background_flushes.spawn(flush_batch(
                            Arc::clone(&store),
                            Arc::clone(&client),
                            org_rx.clone(),
                            batch,
                        ));
                    }
                }
            }
        }
    }
}

/// Re-queue completion updates for uploaded traces whose backend ack never
/// landed. The batch PUT is idempotent (same status body the uploader already
/// sent), so re-driving is safe; each pass increments the trace's attempt
/// counter and the reclaim gate opens once the attempts are exhausted, so a
/// permanently-rejected completion cannot retain its recording forever.
async fn reconcile_unreported_completions(
    store: &Arc<SqliteStateStore>,
    pending: &mut HashMap<i64, RecordingBatch>,
    grace: Duration,
) {
    let stale = match store
        .claim_unreported_completions(grace.as_secs() as i64, MAX_BATCH_SIZE as i64)
        .await
    {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(%error, "could not list unreported completions");
            return;
        }
    };
    for trace in stale {
        if trace.completion_report_attempts >= MAX_COMPLETION_REPORT_ATTEMPTS {
            tracing::warn!(
                trace_id = %trace.trace_id,
                recording_index = trace.recording_index,
                attempts = trace.completion_report_attempts,
                "final re-send of unacknowledged trace completion; reclaim unblocks after this"
            );
        } else {
            tracing::info!(
                trace_id = %trace.trace_id,
                recording_index = trace.recording_index,
                attempts = trace.completion_report_attempts,
                "re-sending unacknowledged trace completion"
            );
        }
        let recording_index = trace.recording_index;
        let batch = pending
            .entry(recording_index)
            .or_insert_with(|| RecordingBatch::new(recording_index));
        batch.add(StatusUpdate::completed(
            recording_index,
            trace.trace_id,
            trace.total_bytes,
        ));
    }
}

/// Spawn a background task for every batch whose deadline has passed.
/// Synchronous — never blocks the select loop on HTTP I/O.
fn flush_due(
    store: &Arc<SqliteStateStore>,
    client: &Arc<ApiClient>,
    org_rx: &OrgIdRx,
    pending: &mut HashMap<i64, RecordingBatch>,
    background_flushes: &mut JoinSet<Option<RecordingBatch>>,
) {
    let now = Instant::now();
    let due_ids: Vec<i64> = pending
        .iter()
        .filter(|(_, batch)| now >= batch.deadline())
        .map(|(recording_index, _)| *recording_index)
        .collect();
    for recording_index in &due_ids {
        if let Some(batch) = pending.remove(recording_index) {
            background_flushes.spawn(flush_batch(
                Arc::clone(store),
                Arc::clone(client),
                org_rx.clone(),
                batch,
            ));
        }
    }
}

async fn flush_all(
    store: &Arc<SqliteStateStore>,
    client: &Arc<ApiClient>,
    org_rx: &OrgIdRx,
    pending: &mut HashMap<i64, RecordingBatch>,
) {
    let mut tasks: JoinSet<Option<RecordingBatch>> = JoinSet::new();
    for (_, batch) in pending.drain() {
        tasks.spawn(flush_batch(
            Arc::clone(store),
            Arc::clone(client),
            org_rx.clone(),
            batch,
        ));
    }
    // Deferred batches (org_id / cloud id not yet known) can't be sent and are
    // dropped on shutdown. The persisted trace rows and the final reclaim are
    // the source of truth that recovers state; the live per-trace progress in
    // these dropped batches is forfeited on shutdown. Count them so a
    // surprising number is visible rather than silent.
    let mut dropped = 0usize;
    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Some(_deferred_batch)) => dropped += 1,
            Ok(None) => {}
            Err(panic_err) => {
                tracing::warn!(?panic_err, "flush_batch task panicked on shutdown");
            }
        }
    }
    if dropped > 0 {
        tracing::info!(
            dropped,
            "dropped deferred status batches on shutdown (no org/cloud id yet; \
             persisted rows remain source-of-truth)"
        );
    }
}

/// Flush a single recording's batch. Returns the batch back if the recording's
/// `org_id` / cloud `recording_id` isn't available yet (caller should re-insert
/// with deferred deadline), or `None` when the flush was sent (or the batch was
/// empty).
async fn flush_batch(
    store: Arc<SqliteStateStore>,
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    mut batch: RecordingBatch,
) -> Option<RecordingBatch> {
    let recording_index = batch.recording_index;
    let row = match resolve_recording(&store, recording_index).await {
        Some(row) => row,
        None => {
            // Re-queue with a fresh `opened_at` pushed
            // `ORG_RESOLVE_RETRY_BACKOFF` into the future so the next
            // `flush_due` skips this batch until the start notifier has
            // populated the cloud id. Without this, a missing field pins
            // `deadline()` permanently in the past and the select loop becomes
            // a busy-wait until the row is ready.
            batch.defer(ORG_RESOLVE_RETRY_BACKOFF);
            return Some(batch);
        }
    };
    let (Some(org_id), Some(recording_id)) = (org_rx.borrow().clone(), row.recording_id) else {
        batch.defer(ORG_RESOLVE_RETRY_BACKOFF);
        return Some(batch);
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
                recording_index,
                recording_id,
                count = updates_payload.len(),
                "flushed status updates"
            );
            // Watermark the acknowledged completions so the recording reaper
            // knows they durably reached the backend; without the stamp a
            // sweep can destroy a recording whose completion is still queued.
            let completed: Vec<String> = updates_payload
                .iter()
                .filter(|(_, update)| update.status == Some(TraceStatusValue::UploadComplete))
                .map(|(trace_id, _)| trace_id.clone())
                .collect();
            if let Err(error) = store.mark_traces_completion_reported(&completed).await {
                tracing::warn!(%error, recording_index, "could not stamp completion_reported_at");
            }
        }
        Err(error) => {
            tracing::warn!(%error, recording_index, recording_id, count = updates_payload.len(), "status batch update failed");
        }
    }
    None
}

async fn resolve_recording(
    store: &Arc<SqliteStateStore>,
    recording_index: i64,
) -> Option<RecordingRow> {
    match store.get_recording(recording_index).await {
        Ok(Some(row)) => Some(row),
        Ok(None) => None,
        Err(error) => {
            tracing::warn!(%error, recording_index, "status updater could not read recording row");
            None
        }
    }
}

#[derive(Debug)]
struct RecordingBatch {
    recording_index: i64,
    opened_at: Instant,
    has_completion: bool,
    updates: HashMap<String, TraceStatusUpdate>,
}

impl RecordingBatch {
    fn new(recording_index: i64) -> Self {
        Self {
            recording_index,
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

    use crate::api::auth::StaticAuthProvider;
    use crate::api::client::ApiClientOptions;
    use crate::state::store::NewRecording;
    use tempfile::TempDir;
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn open_store() -> (SqliteStateStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .unwrap();
        (store, dir)
    }

    fn client(server: &MockServer) -> Arc<ApiClient> {
        let auth = Arc::new(StaticAuthProvider::new("test"));
        let mut options = ApiClientOptions::new(server.uri());
        options.max_backoff = Duration::from_millis(10);
        Arc::new(ApiClient::new(options, auth).unwrap())
    }

    /// A live-org receiver fixed at `org`. The sender is leaked so the channel
    /// stays open for the test's duration (matches `progress.rs`).
    fn org_rx(org: Option<&str>) -> OrgIdRx {
        let (org_tx, org_rx) = tokio::sync::watch::channel(org.map(str::to_string));
        Box::leak(Box::new(org_tx));
        org_rx
    }

    /// Create a recording stamped with the given cloud `recording_id` so the
    /// wiremock URL expectations resolve. Returns the local `recording_index`.
    async fn seed_recording(store: &SqliteStateStore, cloud_recording_id: &str) -> i64 {
        let recording = store
            .create_recording(NewRecording::default())
            .await
            .unwrap();
        store
            .mark_recording_start_notified(recording.recording_index, cloud_recording_id)
            .await
            .unwrap();
        recording.recording_index
    }

    #[test]
    fn batch_records_completion_flag() {
        let mut batch = RecordingBatch::new(1);
        batch.add(StatusUpdate::in_progress(1, "t1".to_string(), 1));
        assert!(!batch.has_completion);
        batch.add(StatusUpdate::completed(1, "t1".to_string(), 100));
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
        // The defer path is invoked when no current org_id is configured
        // yet. Without it the batch's deadline stays in the past
        // and the select loop spins; with it the next deadline is at
        // least `delay` from now.
        let mut batch = RecordingBatch::new(1);
        batch.add(StatusUpdate::in_progress(1, "t".to_string(), 1));
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
        let mut batch = RecordingBatch::new(1);
        let baseline = batch.opened_at + IN_PROGRESS_MAX_WAIT;
        assert!(batch.deadline() <= baseline);
        batch.add(StatusUpdate::completed(1, "t".to_string(), 1));
        assert!(batch.deadline() < baseline);
    }

    #[tokio::test]
    async fn flush_batch_sends_coalesced_updates() {
        // The whole point of the coordinator: the per-trace coalesced state
        // reaches the backend in one batch-update PUT. The body asserts the
        // coalescing — t1's later byte count supersedes the earlier one, and
        // t2 carries its completion status + totals.
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/org/org-1/recording/rec-1/traces/batch-update"))
            .and(body_json(serde_json::json!({
                "updates": {
                    "t1": {"uploaded_bytes": 30},
                    "t2": {"status": "UPLOAD_COMPLETE", "uploaded_bytes": 200, "total_bytes": 200}
                }
            })))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let index = seed_recording(&store, "rec-1").await;
        let mut batch = RecordingBatch::new(index);
        batch.add(StatusUpdate::in_progress(index, "t1".to_string(), 10));
        batch.add(StatusUpdate::in_progress(index, "t1".to_string(), 30)); // supersedes 10
        batch.add(StatusUpdate::completed(index, "t2".to_string(), 200));

        let result = flush_batch(
            Arc::new(store.clone()),
            client(&server),
            org_rx(Some("org-1")),
            batch,
        )
        .await;
        assert!(result.is_none(), "a sent batch is not re-queued");
    }

    #[tokio::test]
    async fn flush_batch_stamps_acked_completions() {
        // The reaper's reclaim gate reads `completion_reported_at`; an acked
        // batch must stamp exactly the traces whose UPLOAD_COMPLETE it
        // carried, and only those.
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let index = seed_recording(&store, "rec-1").await;
        store
            .create_trace(index, "t1", Some("J"), None)
            .await
            .unwrap();
        store
            .create_trace(index, "t2", Some("J"), None)
            .await
            .unwrap();
        let mut batch = RecordingBatch::new(index);
        batch.add(StatusUpdate::in_progress(index, "t1".to_string(), 10));
        batch.add(StatusUpdate::completed(index, "t2".to_string(), 200));

        flush_batch(
            Arc::new(store.clone()),
            client(&server),
            org_rx(Some("org-1")),
            batch,
        )
        .await;

        let traces = store.list_traces_for_recording(index).await.unwrap();
        let stamped_at = |id: &str| {
            traces
                .iter()
                .find(|trace| trace.trace_id == id)
                .unwrap()
                .completion_reported_at
        };
        assert!(
            stamped_at("t2").is_some(),
            "the acked completion is watermarked"
        );
        assert!(
            stamped_at("t1").is_none(),
            "an in-progress update is not watermarked"
        );
    }

    /// The dropped-batch recovery path: an uploaded trace whose completion
    /// batch was lost (PUT failure, crash before stamp) is re-driven by the
    /// reconcile pass, stamped once the re-sent batch is acked, and the
    /// recording becomes reclaimable.
    #[tokio::test]
    async fn reconcile_re_sends_dropped_completion_until_reclaimable() {
        use crate::state::{
            ProgressReportStatus, TraceUpdate, TraceUploadStatus, TraceWriteStatus,
        };

        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let store = Arc::new(store);
        // Fully-settled recording with one uploaded trace, but its completion
        // was never acked (the batch carrying it was dropped).
        let index = seed_recording(&store, "rec-1").await;
        store
            .create_trace(index, "t1", Some("J"), None)
            .await
            .unwrap();
        store
            .update_trace(
                "t1",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    upload_status: Some(TraceUploadStatus::Uploaded),
                    total_bytes: Some(200),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();
        store.mark_recording_stopped(index, 1).await.unwrap();
        store.mark_recording_stop_notified(index).await.unwrap();
        store.set_expected_trace_count(index, 1).await.unwrap();
        store
            .set_progress_report_status(
                index,
                ProgressReportStatus::Pending,
                ProgressReportStatus::Reported,
            )
            .await
            .unwrap();
        assert!(
            store.recordings_pending_reclaim().await.unwrap().is_empty(),
            "the dropped completion blocks reclaim"
        );

        // Reconcile pass picks the trace up and queues a fresh completion.
        let mut pending: HashMap<i64, RecordingBatch> = HashMap::new();
        reconcile_unreported_completions(&store, &mut pending, Duration::ZERO).await;
        let batch = pending.remove(&index).expect("reconcile queues a batch");
        assert!(batch.has_completion, "the re-driven update is a completion");

        // Flushing the re-driven batch stamps the trace and unblocks reclaim.
        flush_batch(
            Arc::clone(&store),
            client(&server),
            org_rx(Some("org-1")),
            batch,
        )
        .await;
        let traces = store.list_traces_for_recording(index).await.unwrap();
        assert!(traces[0].completion_reported_at.is_some());
        let reclaimable = store.recordings_pending_reclaim().await.unwrap();
        assert_eq!(reclaimable.len(), 1, "the recording reclaims after re-send");
    }

    #[tokio::test]
    async fn flush_batch_defers_when_recording_row_missing() {
        // The start notifier hasn't written the recording row yet. The batch
        // must be re-queued with its deadline pushed into the future so the
        // select loop doesn't busy-wait on a permanently-past deadline.
        let server = MockServer::start().await; // no mock mounted: nothing is sent
        let (store, _dir) = open_store().await;
        let mut batch = RecordingBatch::new(999); // never created
        batch.add(StatusUpdate::in_progress(999, "t1".to_string(), 5));

        let before = Instant::now();
        let result = flush_batch(
            Arc::new(store.clone()),
            client(&server),
            org_rx(Some("org-1")),
            batch,
        )
        .await;
        let deferred = result.expect("a missing recording row re-queues the batch");
        assert!(
            deferred.deadline() > before,
            "the re-queued batch's deadline is pushed into the future"
        );
    }

    #[tokio::test]
    async fn flush_batch_defers_when_org_id_unset() {
        let server = MockServer::start().await; // nothing is sent
        let (store, _dir) = open_store().await;
        let index = seed_recording(&store, "rec-1").await;
        let mut batch = RecordingBatch::new(index);
        batch.add(StatusUpdate::in_progress(index, "t1".to_string(), 5));

        let result = flush_batch(
            Arc::new(store.clone()),
            client(&server),
            org_rx(None),
            batch,
        )
        .await;
        assert!(
            result.is_some(),
            "no org_id → the batch is deferred, not sent"
        );
    }

    #[tokio::test]
    async fn flush_batch_defers_when_cloud_recording_id_unset() {
        let server = MockServer::start().await; // nothing is sent
        let (store, _dir) = open_store().await;
        // Created but NOT start-notified, so the cloud recording_id is absent.
        let index = store
            .create_recording(NewRecording::default())
            .await
            .unwrap()
            .recording_index;
        let mut batch = RecordingBatch::new(index);
        batch.add(StatusUpdate::in_progress(index, "t1".to_string(), 5));

        let result = flush_batch(
            Arc::new(store.clone()),
            client(&server),
            org_rx(Some("org-1")),
            batch,
        )
        .await;
        assert!(
            result.is_some(),
            "no cloud recording_id → the batch is deferred, not sent"
        );
    }

    #[tokio::test]
    async fn flush_batch_skips_empty_batch() {
        let server = MockServer::start().await; // nothing is sent
        let (store, _dir) = open_store().await;
        let index = seed_recording(&store, "rec-1").await;
        let batch = RecordingBatch::new(index); // no updates added

        let result = flush_batch(
            Arc::new(store.clone()),
            client(&server),
            org_rx(Some("org-1")),
            batch,
        )
        .await;
        assert!(
            result.is_none(),
            "an empty batch is a no-op, nothing is sent"
        );
    }

    #[tokio::test]
    async fn flush_all_drains_every_pending_batch() {
        // The shutdown path: every pending recording's batch is flushed and the
        // map left empty.
        let server = MockServer::start().await;
        for recording_id in ["rec-1", "rec-2"] {
            Mock::given(method("PUT"))
                .and(path(format!(
                    "/org/org-1/recording/{recording_id}/traces/batch-update"
                )))
                .respond_with(ResponseTemplate::new(200))
                .expect(1)
                .mount(&server)
                .await;
        }

        let (store, _dir) = open_store().await;
        let first = seed_recording(&store, "rec-1").await;
        let second = seed_recording(&store, "rec-2").await;
        let mut pending: HashMap<i64, RecordingBatch> = HashMap::new();
        let mut first_batch = RecordingBatch::new(first);
        first_batch.add(StatusUpdate::in_progress(first, "t1".to_string(), 1));
        pending.insert(first, first_batch);
        let mut second_batch = RecordingBatch::new(second);
        second_batch.add(StatusUpdate::completed(second, "t2".to_string(), 9));
        pending.insert(second, second_batch);

        flush_all(
            &Arc::new(store.clone()),
            &client(&server),
            &org_rx(Some("org-1")),
            &mut pending,
        )
        .await;
        assert!(pending.is_empty(), "flush_all drains the pending map");
    }

    #[tokio::test]
    async fn flush_due_spawns_only_batches_past_their_deadline() {
        // The periodic tick must flush only batches whose deadline has elapsed,
        // leaving younger batches pending to keep coalescing.
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/org/org-1/recording/rec-1/traces/batch-update"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let due_index = seed_recording(&store, "rec-1").await;
        let fresh_index = seed_recording(&store, "rec-2").await;
        let mut pending: HashMap<i64, RecordingBatch> = HashMap::new();
        // A batch whose deadline is already well in the past.
        let mut due = RecordingBatch::new(due_index);
        due.add(StatusUpdate::in_progress(due_index, "t1".to_string(), 1));
        due.opened_at = Instant::now() - Duration::from_secs(60);
        pending.insert(due_index, due);
        // A just-opened batch whose deadline is comfortably in the future.
        let mut fresh = RecordingBatch::new(fresh_index);
        fresh.add(StatusUpdate::in_progress(fresh_index, "t2".to_string(), 1));
        pending.insert(fresh_index, fresh);

        let mut background: JoinSet<Option<RecordingBatch>> = JoinSet::new();
        flush_due(
            &Arc::new(store.clone()),
            &client(&server),
            &org_rx(Some("org-1")),
            &mut pending,
            &mut background,
        );

        assert_eq!(background.len(), 1, "only the past-deadline batch flushes");
        assert!(
            pending.contains_key(&fresh_index),
            "the fresh batch keeps coalescing"
        );
        assert!(
            !pending.contains_key(&due_index),
            "the due batch was taken for flushing"
        );

        // Drain the spawned flush so the task doesn't outlive the test.
        while background.join_next().await.is_some() {}
    }

    #[tokio::test]
    async fn run_loop_flushes_a_full_batch_via_the_inbox() {
        // Drive the public updater end to end: MAX_BATCH_SIZE distinct-trace
        // updates for one recording trip the size trigger, which spawns a
        // background flush. Exercises run()'s inbox + max-batch + join_next arms.
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/org/org-1/recording/rec-1/traces/batch-update"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let index = seed_recording(&store, "rec-1").await;

        let (tx, rx) = mpsc::unbounded_channel::<StatusUpdate>();
        let (shutdown_tx, shutdown_rx) = broadcast::channel(4);
        let handle = spawn_status_updater(
            store.clone(),
            client(&server),
            org_rx(Some("org-1")),
            rx,
            shutdown_rx,
        );

        for n in 0..MAX_BATCH_SIZE {
            tx.send(StatusUpdate::in_progress(index, format!("t-{n}"), n as i64))
                .unwrap();
        }

        // Poll the mock until the size-triggered flush lands, breaking as soon
        // as it does rather than sleeping a fixed duration. The only requests
        // this server sees are batch-update PUTs.
        let mut flushed = false;
        for _ in 0..100 {
            if let Some(requests) = server.received_requests().await {
                if !requests.is_empty() {
                    flushed = true;
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(
            flushed,
            "a full batch is flushed to the backend by the run loop"
        );

        // Closing the inbox stops the loop (recv → None → break); keep the
        // shutdown sender alive until then so the shutdown arm doesn't pre-empt.
        drop(tx);
        handle.join().await;
        drop(shutdown_tx);
    }
}
