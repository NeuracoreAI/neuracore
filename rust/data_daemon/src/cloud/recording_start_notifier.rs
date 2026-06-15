//! Backend recording-start notifier.
//!
//! Subscribes to [`DaemonEvent::RecordingStarted`] and POSTs
//! `/org/{org}/recording/start` to the backend, persisting the cloud
//! `recording_id` the backend mints in response. The Python SDK used to make
//! this call inline from `nc.start_recording`, but the staging POST has a fat
//! upper tail. Doing it here means the SDK call returns as soon as the
//! producer publishes the `StartRecording` envelope, and the cloud-id mint
//! rides the daemon's standard retry policy in the background.
//!
//! A direct mirror of [`recording_stop_notifier`](super::recording_stop_notifier):
//! each event spawns a single retried request and failures are logged with the
//! recording index but never surfaced to the SDK. The cloud `recording_id` is
//! always minted here — every downstream coordinator (registration, progress,
//! upload) waits for this id, so an offline recording simply stays pending
//! until the daemon is online and `/recording/start` lands.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::broadcast;

use crate::api::ApiClient;
use crate::cloud::notifier::{spawn_notifier, NotifierCtx, NotifierHandle, RecordingNotifier};
use crate::cloud::OrgIdRx;
use crate::lifecycle::signals::ShutdownSignal;
use crate::state::{
    DaemonEvent, EventBus, RecordingRow, SqliteStateStore, StateStore, StateStoreError,
};

/// Notifier that POSTs `/recording/start` and persists the cloud `recording_id`
/// the backend mints. The cloud id is always minted here — every downstream
/// coordinator waits on it — so an offline recording stays pending until the
/// daemon is online and the start POST lands. Before opening the new recording
/// it closes any earlier still-pending recording for the same source (see
/// [`resolve_prior_pending`]).
struct StartNotifier;

#[async_trait]
impl RecordingNotifier for StartNotifier {
    fn label(&self) -> &'static str {
        "recording-start"
    }

    fn triggered_by(&self, event: &DaemonEvent) -> Option<i64> {
        match event {
            DaemonEvent::RecordingStarted { recording_index } => Some(*recording_index),
            _ => None,
        }
    }

    async fn pending(
        &self,
        store: &Arc<SqliteStateStore>,
    ) -> Result<Vec<RecordingRow>, StateStoreError> {
        store.recordings_pending_start_notify().await
    }

    async fn notify(&self, ctx: &NotifierCtx, recording_index: i64) {
        notify_backend(
            &ctx.store,
            &ctx.client,
            &ctx.bus,
            &ctx.org_rx,
            recording_index,
        )
        .await;
    }
}

/// Spawn the recording-start notifier on the current Tokio runtime.
pub fn spawn_recording_start_notifier(
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> NotifierHandle {
    spawn_notifier(StartNotifier, store, bus, client, org_rx, shutdown_rx)
}

async fn notify_backend(
    store: &Arc<SqliteStateStore>,
    client: &Arc<ApiClient>,
    bus: &EventBus,
    org_rx: &OrgIdRx,
    recording_index: i64,
) {
    let row = match store.get_recording(recording_index).await {
        Ok(Some(row)) => row,
        Ok(None) => {
            tracing::warn!(
                recording_index,
                "recording row missing on start; skipping backend notify",
            );
            return;
        }
        Err(error) => {
            tracing::warn!(
                %error,
                recording_index,
                "failed to look up recording for start notify",
            );
            return;
        }
    };
    if row.recording_id.is_some() || row.backend_start_notified_at.is_some() {
        // Already notified — another path handled it.
        return;
    }

    let Some(org_id) = org_rx.borrow().clone() else {
        // No current org configured yet (not logged in / org not selected).
        // Without it we can't address the POST; the next sweep retries once
        // the config watcher picks up a current org.
        tracing::warn!(
            recording_index,
            "no current org_id configured at start time; skipping backend notify",
        );
        return;
    };
    let Some(robot_id) = row.robot_id else {
        tracing::warn!(
            recording_index,
            "recording has no robot_id at start time; skipping backend notify",
        );
        return;
    };
    let Some(dataset_id) = row.dataset_id else {
        tracing::warn!(
            recording_index,
            "recording has no dataset_id at start time; skipping backend notify",
        );
        return;
    };
    let instance = row.robot_instance.unwrap_or(0);
    let Some(start_timestamp_ns) = row.start_timestamp_ns else {
        tracing::warn!(
            recording_index,
            "recording has no start_timestamp_ns at start time; skipping backend notify",
        );
        return;
    };
    // The producer captured this as the recording window's real lower bound;
    // the backend requires it (seconds) and derives the reported duration from
    // it, so a late notify (e.g. after reconnecting) still reports correctly.
    let start_time = start_timestamp_ns as f64 / 1_000_000_000.0;

    // Before opening this recording server-side, close any earlier recording for
    // the same source that finished locally (cancel/stop) but whose backend
    // notification has not landed yet. The backend dedupes pending recordings
    // per robot instance — it returns the existing pending recording instead of
    // minting a new one — so a still-pending prior recording would otherwise
    // hand its cloud id to this one, collapsing both into one backend recording
    // (e.g. cancel-then-start with no gap). The start notifier processes
    // `RecordingStarted` events in order, so the prior recording's cloud id is
    // already on its row by the time we reach here.
    resolve_prior_pending(store, client, &org_id, &robot_id, instance, recording_index).await;

    match client
        .recording_start(&org_id, &robot_id, instance, &dataset_id, start_time)
        .await
    {
        Ok(recording_id) => {
            if let Err(error) = store
                .mark_recording_start_notified(recording_index, &recording_id)
                .await
            {
                tracing::warn!(
                    %error,
                    recording_index,
                    recording_id,
                    "POST succeeded but persisting the cloud recording_id failed; \
                     the next sweep will re-post (the start notify is idempotent)",
                );
            } else {
                tracing::info!(
                    recording_index,
                    recording_id,
                    "backend notified of recording start",
                );
                // The cloud id is now available. Wake any coordinator that was
                // waiting on it — notably the stop notifier, for a recording
                // that was stopped while offline before its start was notified.
                bus.publish(DaemonEvent::RecordingCloudIdAssigned { recording_index });
            }
        }
        Err(error) => {
            // The producer-side iceoryx2 publish has already succeeded by
            // the time we get here; logging is the only available recourse
            // until the next sweep retries.
            tracing::warn!(
                %error,
                recording_index,
                "failed to notify backend of recording start",
            );
        }
    }
}

/// Close, on the backend, any earlier recording for `(robot_id, instance)` that
/// finished locally (cancelled or stopped) but is still pending server-side, so
/// the backend does not hand its cloud id to the next `/recording/start` for
/// this instance. See
/// [`StateStore::recordings_pending_backend_resolution_for_source`].
async fn resolve_prior_pending(
    store: &Arc<SqliteStateStore>,
    client: &Arc<ApiClient>,
    org_id: &str,
    robot_id: &str,
    instance: i64,
    before_index: i64,
) {
    let prior = match store
        .recordings_pending_backend_resolution_for_source(robot_id, instance, before_index)
        .await
    {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(
                %error,
                before_index,
                "failed to query prior pending recordings for source; next start may reuse a cloud id",
            );
            return;
        }
    };
    for row in prior {
        let index = row.recording_index;
        let is_cancelled = row.cancelled_at.is_some();
        // Cancel and stop both report the recording's captured stop time as
        // `end_time` (a cancel is a stop that discards data). Compute it before
        // `recording_id` is moved out of `row`.
        let end_time = row.stop_timestamp_ns.map(|ns| ns as f64 / 1_000_000_000.0);
        let Some(recording_id) = row.recording_id else {
            continue;
        };
        let Some(end_time) = end_time else {
            continue;
        };
        if is_cancelled {
            match client
                .recording_cancel(org_id, &recording_id, end_time)
                .await
            {
                Ok(()) => {
                    let _ = store.mark_recording_cancel_notified(index).await;
                    tracing::info!(
                        recording_index = index,
                        recording_id,
                        next_recording_index = before_index,
                        "cancelled prior pending recording on the backend before opening the next",
                    );
                }
                Err(error) if error.is_not_found() => {
                    // Already closed — the cancel-notifier sweep won the race.
                    // The prior recording is not pending on the backend, so the
                    // next start cannot reuse its id; mark it notified so the
                    // sweep stops re-posting too.
                    let _ = store.mark_recording_cancel_notified(index).await;
                    tracing::debug!(
                        recording_index = index,
                        recording_id,
                        next_recording_index = before_index,
                        "prior pending recording already cancelled on backend (404)",
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        %error,
                        recording_index = index,
                        recording_id,
                        "failed to cancel prior pending recording before next start; \
                         the next start may reuse its cloud id",
                    );
                }
            }
        } else {
            match client.recording_stop(org_id, &recording_id, end_time).await {
                Ok(()) => {
                    let _ = store.mark_recording_stop_notified(index).await;
                    tracing::info!(
                        recording_index = index,
                        recording_id,
                        next_recording_index = before_index,
                        "stopped prior pending recording on the backend before opening the next",
                    );
                }
                Err(error) if error.is_not_found() => {
                    // Already closed — the stop-notifier sweep won the race. Mark
                    // it notified so the sweep stops re-posting too.
                    let _ = store.mark_recording_stop_notified(index).await;
                    tracing::debug!(
                        recording_index = index,
                        recording_id,
                        next_recording_index = before_index,
                        "prior pending recording already stopped on backend (404)",
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        %error,
                        recording_index = index,
                        recording_id,
                        "failed to stop prior pending recording before next start",
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    use tempfile::TempDir;
    use tokio::sync::broadcast;
    use tokio::time::{sleep, timeout};
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::api::auth::StaticAuthProvider;
    use crate::api::{ApiClient, ApiClientOptions};
    use crate::lifecycle::signals::ShutdownSignal;
    use crate::state::{DaemonEvent, EventBus, NewRecording, SqliteStateStore, StateStore};

    async fn open_store() -> (SqliteStateStore, TempDir) {
        let dir = TempDir::new().expect("tempdir");
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .expect("open store");
        (store, dir)
    }

    fn options(base_url: String) -> ApiClientOptions {
        ApiClientOptions {
            base_url,
            timeout: Duration::from_secs(5),
            max_retries: 1,
            max_backoff: Duration::from_secs(1),
        }
    }

    /// Insert a fresh recording (no cloud id yet) and return its local index.
    async fn seed_recording(store: &SqliteStateStore) -> i64 {
        store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(7),
                dataset_id: Some("ds-1"),
                start_timestamp_ns: 1_700_000_000_000_000_000,
            })
            .await
            .expect("create recording")
            .recording_index
    }

    /// A live-org receiver fixed at `org`. The sender is leaked so the channel
    /// stays open for the test's duration.
    fn org_rx(org: Option<&str>) -> OrgIdRx {
        let (org_tx, org_rx) = tokio::sync::watch::channel(org.map(str::to_string));
        Box::leak(Box::new(org_tx));
        org_rx
    }

    fn start_ok_mock(recording_id: &'static str) -> wiremock::Mock {
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/start"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "id": recording_id })),
            )
    }

    #[tokio::test]
    async fn posts_backend_start_on_recording_started_event() {
        let server = MockServer::start().await;
        start_ok_mock("cloud-rec-1").mount(&server).await;

        let (store, _dir) = open_store().await;
        let index = seed_recording(&store).await;

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));

        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_start_notifier(
            store.clone(),
            bus.clone(),
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
        );

        bus.publish(DaemonEvent::RecordingStarted {
            recording_index: index,
        });

        // The cloud id lands on the row once the POST round-trips.
        timeout(Duration::from_secs(3), async {
            loop {
                let row = store
                    .get_recording(index)
                    .await
                    .expect("get")
                    .expect("exists");
                if row.recording_id.is_some() {
                    assert_eq!(row.recording_id.as_deref(), Some("cloud-rec-1"));
                    assert!(row.backend_start_notified_at.is_some());
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("cloud recording_id must be persisted within 3s");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn cancels_prior_pending_recording_before_opening_the_next() {
        // Cancel-then-start (no gap) for one source: the prior recording was
        // cancelled before its cloud id was notified, so it is still pending on
        // the backend. Opening the next recording must cancel it FIRST, so the
        // backend mints a fresh id instead of handing back the cancelled one
        // (which would collapse both recordings into one cloud recording).
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/cancel"))
            .and(body_partial_json(
                serde_json::json!({ "recording_id": "cloud-cancelled-A" }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!("ok")))
            .mount(&server)
            .await;
        start_ok_mock("cloud-fresh-B").mount(&server).await;

        let (store, _dir) = open_store().await;
        // Prior recording A (same source): start-notified, then cancelled, with
        // its backend cancel still pending.
        let prior = seed_recording(&store).await;
        store
            .mark_recording_start_notified(prior, "cloud-cancelled-A")
            .await
            .expect("mark start notified");
        store
            .cancel_recording(prior, 5_000_000_000)
            .await
            .expect("cancel");
        // The next recording B for the same source.
        let next = seed_recording(&store).await;

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));
        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_start_notifier(
            store.clone(),
            bus.clone(),
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
        );

        bus.publish(DaemonEvent::RecordingStarted {
            recording_index: next,
        });

        timeout(Duration::from_secs(3), async {
            loop {
                let prior_row = store
                    .get_recording(prior)
                    .await
                    .expect("get")
                    .expect("exists");
                let next_row = store
                    .get_recording(next)
                    .await
                    .expect("get")
                    .expect("exists");
                if prior_row.backend_cancel_notified_at.is_some() && next_row.recording_id.is_some()
                {
                    // Prior cancelled server-side; next opened with a FRESH id.
                    assert_eq!(next_row.recording_id.as_deref(), Some("cloud-fresh-B"));
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("prior recording must be cancelled and next opened fresh within 3s");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn startup_sweep_notifies_recordings_opened_while_offline() {
        // A recording opened during a previous offline session: no cloud id,
        // no start-notify/failed stamps. The pre-loop sweep must POST and
        // persist the minted cloud id.
        let server = MockServer::start().await;
        start_ok_mock("cloud-rec-offline").mount(&server).await;

        let (store, _dir) = open_store().await;
        let index = seed_recording(&store).await;

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));

        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_start_notifier(
            store.clone(),
            bus,
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
        );

        timeout(Duration::from_secs(3), async {
            loop {
                let row = store
                    .get_recording(index)
                    .await
                    .expect("get")
                    .expect("exists");
                if row.recording_id.as_deref() == Some("cloud-rec-offline") {
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("sweep must persist the minted cloud id within 3s");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }
}
