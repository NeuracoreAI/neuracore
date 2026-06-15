//! Shared skeleton for the backend recording-lifecycle notifiers.
//!
//! The start / stop / cancel notifiers each POST a different backend endpoint,
//! but their machinery is identical: subscribe to the event bus, sweep any
//! recordings whose notification is pending from a previous (offline) session,
//! then POST whenever the relevant lifecycle event fires — retrying via a
//! startup sweep and on broadcast lag. This module owns that machinery once; a
//! notifier supplies only the three things that actually differ via
//! [`RecordingNotifier`]: which event(s) trigger it, which "pending" query
//! drives its recovery sweep, and the per-recording POST itself.
//!
//! Each `recording_*_notifier` module defines a small unit struct implementing
//! the trait plus a thin `spawn_recording_*_notifier` wrapper, so the call
//! sites (and their tests) are unchanged.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;

use crate::api::ApiClient;
use crate::cloud::OrgIdRx;
use crate::lifecycle::signals::ShutdownSignal;
use crate::state::{DaemonEvent, EventBus, RecordingRow, SqliteStateStore, StateStoreError};

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
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
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
            _ = sweep(&notifier, &ctx) => {}
        }
        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, notifier = label, "recording notifier shutting down");
                    break;
                }
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
                            sweep(&notifier, &ctx).await;
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

/// Notify every recording the notifier reports as pending.
async fn sweep<N: RecordingNotifier>(notifier: &N, ctx: &NotifierCtx) {
    let pending = match notifier.pending(&ctx.store).await {
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
