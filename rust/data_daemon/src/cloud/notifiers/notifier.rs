//! Shared skeleton for the backend recording-lifecycle notifiers.
//!
//! The start / stop / cancel notifiers each POST a different backend endpoint,
//! but their machinery is identical: subscribe to the event bus, sweep any
//! recordings whose notification is pending from a previous (offline) session,
//! then POST whenever the relevant lifecycle event fires — retrying via a
//! startup sweep, a periodic re-sweep, and on broadcast lag. This module owns
//! that machinery once; a
//! notifier supplies only the three things that actually differ via
//! [`RecordingNotifier`]: which event(s) trigger it, which "pending" query
//! drives its recovery sweep, and the per-recording POST itself.
//!
//! Events are processed sequentially: each POST is awaited inline before the
//! next event is read, so a slow or retrying POST delays later events on the
//! same notifier and can push the broadcast channel into `Lagged`. That is
//! handled by re-running the recovery sweep (the POSTs are idempotent), which
//! is the recovery mechanism rather than a failure.
//!
//! Each `recording_*_notifier` module defines a small unit struct implementing
//! the trait plus a thin `spawn_recording_*_notifier` wrapper, so the call
//! sites (and their tests) are unchanged.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::api::ApiClient;
use crate::cloud::OrgIdRx;
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::{
    DaemonEvent, EventBus, RecordingRow, SqliteStateStore, StateStore, StateStoreError,
};

/// Handle returned by every recording notifier's `spawn_*` wrapper.
pub struct NotifierHandle {
    join: JoinHandle<()>,
    label: &'static str,
}

impl NotifierHandle {
    /// Wait for the notifier task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(
                ?error,
                notifier = self.label,
                "recording notifier join failed"
            );
        }
    }
}

/// Shared dependencies handed to a notifier's `notify`.
pub struct NotifierCtx {
    /// State store (already `Arc`-wrapped for the spawned task).
    pub store: Arc<SqliteStateStore>,
    /// Backend HTTP client.
    pub client: Arc<ApiClient>,
    /// Event bus — the start notifier publishes `RecordingCloudIdAssigned` on it.
    pub bus: EventBus,
    /// Live current-org receiver, read at POST time.
    pub org_rx: OrgIdRx,
}

/// One backend recording-lifecycle notifier (start / stop / cancel).
///
/// Everything common — the spawn loop, the offline-recovery sweep, the
/// shutdown/lag handling — lives in [`spawn_notifier`]; an implementor supplies
/// only what differs.
#[async_trait]
pub trait RecordingNotifier: Send + Sync + 'static {
    /// Short label used in this notifier's log lines.
    fn label(&self) -> &'static str;

    /// The recording index to notify for `event`, or `None` to ignore it.
    fn triggered_by(&self, event: &DaemonEvent) -> Option<i64>;

    /// Recordings whose notification is still pending — the offline-recovery
    /// sweep set, run on startup and after a broadcast lag.
    async fn pending(
        &self,
        store: &Arc<SqliteStateStore>,
    ) -> Result<Vec<RecordingRow>, StateStoreError>;

    /// The subset of [`pending`](Self::pending) that is safe to re-drive on
    /// the periodic re-sweep, which fires for the life of the daemon rather
    /// than once per start. Defaults to the whole set, which is right for any
    /// notifier whose POST is idempotent.
    async fn periodic_pending(
        &self,
        store: &Arc<SqliteStateStore>,
    ) -> Result<Vec<RecordingRow>, StateStoreError> {
        self.pending(store).await
    }

    /// Fire the backend POST for one recording. Idempotent and self-logging:
    /// the shared loop never inspects the result.
    async fn notify(&self, ctx: &NotifierCtx, recording_index: i64);
}

/// Spawn a notifier task driven by `notifier` on the current Tokio runtime.
///
/// Sweeps pending notifications first (so recordings that finished while the
/// daemon was offline recover), then serves live bus events until shutdown.
pub fn spawn_notifier<N: RecordingNotifier>(
    notifier: N,
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> NotifierHandle {
    spawn_notifier_every(
        notifier,
        store,
        bus,
        client,
        org_rx,
        shutdown_rx,
        crate::intervals::NOTIFY_RESWEEP,
    )
}

/// [`spawn_notifier`] with the re-sweep cadence spelled out, so a test can
/// drive the retry path without waiting a real [`NOTIFY_RESWEEP`].
///
/// [`NOTIFY_RESWEEP`]: crate::intervals::NOTIFY_RESWEEP
pub fn spawn_notifier_every<N: RecordingNotifier>(
    notifier: N,
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
    resweep_every: Duration,
) -> NotifierHandle {
    let label = notifier.label();
    let mut subscriber = bus.subscribe();
    let ctx = NotifierCtx {
        store: Arc::new(store),
        client,
        bus,
        org_rx,
    };
    let join = tokio::spawn(async move {
        // Recover pending notifications before serving live events. Run inside a
        // `select!` against shutdown so a long sweep cannot hold up exit.
        tokio::select! {
            biased;
            signal = shutdown_rx.recv() => {
                tracing::debug!(?signal, notifier = label, "recording notifier shutting down before sweep");
                return;
            }
            _ = sweep(&notifier, &ctx, SweepScope::Full) => {}
        }
        let mut resweep = interval(resweep_every);
        resweep.set_missed_tick_behavior(MissedTickBehavior::Delay);
        // The interval fires immediately on its first tick; the sweep above
        // has just run, so skip straight to the first real deadline.
        resweep.reset();
        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, notifier = label, "recording notifier shutting down");
                    break;
                }
                // A POST can fail for reasons no event will ever repeat —
                // offline, 5xx, a dropped connection — and the event that
                // triggered it does not come round again. Re-run the pending
                // set on a timer so delivery is eventually guaranteed within
                // one daemon lifetime rather than only across a restart.
                // Cheap when idle: `pending()` is a server-side filter that
                // returns nothing once every recording is settled.
                _ = resweep.tick() => sweep(&notifier, &ctx, SweepScope::Periodic).await,
                event = subscriber.recv() => {
                    match event {
                        Ok(event) => {
                            if let Some(recording_index) = notifier.triggered_by(&event) {
                                notifier.notify(&ctx, recording_index).await;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            tracing::warn!(
                                skipped,
                                notifier = label,
                                "recording notifier missed bus events; re-sweeping pending notifications",
                            );
                            sweep(&notifier, &ctx, SweepScope::Full).await;
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            tracing::debug!(notifier = label, "event bus closed; recording notifier exiting");
                            break;
                        }
                    }
                }
            }
        }
    });
    NotifierHandle { join, label }
}

/// Which set a sweep draws from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SweepScope {
    /// Everything pending — startup and post-lag recovery.
    Full,
    /// Only what is safe to re-drive on a timer.
    Periodic,
}

/// Notify every recording the notifier reports as pending for `scope`.
async fn sweep<N: RecordingNotifier>(notifier: &N, ctx: &NotifierCtx, scope: SweepScope) {
    let query = match scope {
        SweepScope::Full => notifier.pending(&ctx.store).await,
        SweepScope::Periodic => notifier.periodic_pending(&ctx.store).await,
    };
    let pending = match query {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(%error, notifier = notifier.label(), "failed to query recordings pending notify");
            return;
        }
    };
    if pending.is_empty() {
        return;
    }
    tracing::info!(
        count = pending.len(),
        notifier = notifier.label(),
        "sweeping recordings with pending backend notify",
    );
    for row in pending {
        notifier.notify(ctx, row.recording_index).await;
    }
}

/// Which `/recording/*` endpoint a lifecycle notify targets. The stop and
/// cancel notifiers run the *same* guard chain (row fetch → already-notified
/// guard → cloud-id guard → org guard → `stop_timestamp_ns` guard → POST →
/// 404-as-success → mark-notified); only these per-kind bits differ.
#[derive(Clone, Copy)]
pub enum LifecycleKind {
    Stop,
    Cancel,
}

impl LifecycleKind {
    /// Word used in this notifier's log lines ("stop" / "cancel").
    fn action(self) -> &'static str {
        match self {
            LifecycleKind::Stop => "stop",
            LifecycleKind::Cancel => "cancel",
        }
    }

    /// Whether this recording's notification has already been persisted.
    fn already_notified(self, row: &RecordingRow) -> bool {
        match self {
            LifecycleKind::Stop => row.backend_stop_notified_at.is_some(),
            LifecycleKind::Cancel => row.backend_cancel_notified_at.is_some(),
        }
    }
}

/// Run the shared stop/cancel backend-notify flow for one recording.
///
/// Idempotent and self-logging (the spawn loop never inspects the result): a
/// 404 is treated as success (the recording is already closed server-side, or
/// the backend reaped it as an abandoned pending recording), and a persist
/// failure after a successful POST is left for the next sweep since the POST is
/// idempotent.
pub async fn notify_recording_lifecycle(
    kind: LifecycleKind,
    store: &Arc<SqliteStateStore>,
    client: &Arc<ApiClient>,
    org_rx: &OrgIdRx,
    recording_index: i64,
) {
    let action = kind.action();
    let row = match store.get_recording(recording_index).await {
        Ok(Some(row)) => row,
        Ok(None) => {
            tracing::warn!(
                recording_index,
                "recording row missing on {action}; skipping backend notify"
            );
            return;
        }
        Err(error) => {
            tracing::warn!(%error, recording_index, "failed to look up recording for {action} notify");
            return;
        }
    };

    if kind.already_notified(&row) {
        // Another path (sweep or earlier event) already notified.
        return;
    }
    // Both kinds are also triggered by `RecordingCloudIdAssigned`, which fires
    // for every recording whose start POST lands — including ones that are
    // still running, and ones that were never stopped or cancelled at all. Hold
    // the POST until this recording has actually reached the state it names.
    match kind {
        LifecycleKind::Stop if row.stopped_at.is_none() => return,
        LifecycleKind::Cancel if row.cancelled_at.is_none() => return,
        _ => {}
    }
    let Some(recording_id) = row.recording_id.clone() else {
        // No cloud id → either the start POST was never made (nothing exists
        // server-side to act on) or it is in flight right now. Defer either
        // way: the start notifier owns the distinction. It settles a cancel
        // itself when it skips the POST, and publishes
        // `RecordingCloudIdAssigned` when the POST lands, which brings the
        // recording back here with an id.
        tracing::debug!(
            recording_index,
            "recording has no cloud id at {action} time; deferring backend notify"
        );
        return;
    };
    let Some(org_id) = org_rx.borrow().clone() else {
        tracing::warn!(
            recording_index,
            recording_id,
            "no current org_id configured at {action} time; skipping backend notify"
        );
        return;
    };
    let Some(stop_timestamp_ns) = row.stop_timestamp_ns else {
        tracing::warn!(
            recording_index,
            recording_id,
            "recording has no stop_timestamp_ns at {action} time; skipping backend notify"
        );
        return;
    };
    // The producer captured this as the recording window's real upper bound;
    // the backend requires it (seconds) and derives the reported duration from
    // it, so a late notify still reports correctly.
    let end_time = stop_timestamp_ns as f64 / 1_000_000_000.0;

    let post_result = match kind {
        LifecycleKind::Stop => {
            client
                .recording_stop(&org_id, &recording_id, end_time)
                .await
        }
        LifecycleKind::Cancel => {
            client
                .recording_cancel(&org_id, &recording_id, end_time)
                .await
        }
    };

    let mark_result = match &post_result {
        Ok(()) => mark_notified(kind, store, recording_index).await,
        // 404 means the backend does not have this recording open — already
        // closed, or reaped as an abandoned pending recording. That is the
        // post-condition we wanted, so record it rather than re-sweeping.
        Err(error) if error.is_not_found() => mark_notified(kind, store, recording_index).await,
        // Cancellation can lose a race with upload completion. The backend
        // cannot cancel an already-uploaded recording, so retrying would only
        // pin the local row and artefacts forever.
        Err(error)
            if matches!(kind, LifecycleKind::Cancel) && error.is_recording_already_uploaded() =>
        {
            tracing::info!(
                recording_index,
                recording_id,
                "recording finished uploading before it could be cancelled"
            );
            mark_notified(kind, store, recording_index).await
        }
        // 403 is permanent, unlike a transport or 5xx failure. Left unstamped
        // it would re-POST on every sweep and, since the reaper only reclaims
        // a notified recording, pin the row and its artefacts forever.
        Err(error) if error.is_forbidden() => {
            tracing::error!(
                %error,
                recording_index,
                recording_id,
                "not permitted to {action} this recording; giving up on the backend notify"
            );
            mark_notified(kind, store, recording_index).await
        }
        Err(error) => {
            tracing::warn!(%error, recording_index, recording_id, "failed to notify backend of recording {action}");
            return;
        }
    };
    if let Err(error) = mark_result {
        tracing::warn!(
            %error,
            recording_index,
            recording_id,
            "POST succeeded but persisting backend_{action}_notified_at failed; \
             the next sweep will re-post (the backend POST is idempotent)",
        );
    } else {
        tracing::info!(
            recording_index,
            recording_id,
            "backend notified of recording {action}"
        );
    }
}

/// Persist the "notified" timestamp for the given lifecycle kind.
async fn mark_notified(
    kind: LifecycleKind,
    store: &Arc<SqliteStateStore>,
    recording_index: i64,
) -> Result<(), StateStoreError> {
    match kind {
        LifecycleKind::Stop => store.mark_recording_stop_notified(recording_index).await,
        LifecycleKind::Cancel => store.mark_recording_cancel_notified(recording_index).await,
    }
    .map(|_| ())
}
