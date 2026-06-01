//! Backend recording-stop notifier.
//!
//! Subscribes to [`DaemonEvent::RecordingStopped`] and POSTs
//! `/org/{org}/recording/stop?recording_id=…` to the backend. The Python SDK
//! used to make this call inline from `nc.stop_recording`, but the staging
//! POST has a fat upper tail (occasional 1-2 s spikes on otherwise
//! sub-second calls). Doing it here means the SDK call returns as soon as
//! the producer publishes the `StopRecording` envelope, and the staging
//! notification rides the daemon's standard retry policy in the background.
//!
//! The notifier never blocks the dispatcher: each event spawns its own
//! `tokio::task` that drives a single retried request. Failures are logged
//! with the recording index but never surfaced to the SDK — by the time we
//! reach this notifier the SDK is long gone and the producer's iceoryx2
//! publish already succeeded.

use std::sync::Arc;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;

use crate::api::ApiClient;
use crate::cloud::OrgIdRx;
use crate::lifecycle::signals::ShutdownSignal;
use crate::state::{DaemonEvent, EventBus, SqliteStateStore, StateStore};

/// Handle returned by [`spawn_recording_stop_notifier`].
pub struct RecordingStopNotifierHandle {
    join: JoinHandle<()>,
}

impl RecordingStopNotifierHandle {
    /// Wait for the notifier task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "recording-stop notifier join failed");
        }
    }
}

/// Spawn the recording-stop notifier on the current Tokio runtime.
///
/// The task first sweeps any recordings stopped while the daemon was offline
/// (rows with `stopped_at NOT NULL` and no `backend_stop_notified_at`), then
/// drops into the event-bus loop and POSTs whenever a fresh `RecordingStopped`
/// event fires.
pub fn spawn_recording_stop_notifier(
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> RecordingStopNotifierHandle {
    let mut subscriber = bus.subscribe();
    let store = Arc::new(store);
    let join = tokio::spawn(async move {
        // Recover any stopped-while-offline recordings before serving live
        // events. Run inside a `select!` against the shutdown signal so a
        // long sweep cannot hold the daemon shutting down.
        tokio::select! {
            biased;
            signal = shutdown_rx.recv() => {
                tracing::debug!(?signal, "recording-stop notifier shutting down before sweep");
                return;
            }
            _ = sweep_pending(&store, &client, &org_rx) => {}
        }
        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "recording-stop notifier shutting down");
                    break;
                }
                event = subscriber.recv() => {
                    match event {
                        // `RecordingStopped` is the live path. `RecordingCloudIdAssigned`
                        // covers offline recovery: a recording stopped while offline has
                        // already fired its `RecordingStopped` (which no coordinator was
                        // running to see), so the stop POST is unblocked only once the
                        // start notifier later assigns the cloud id. `notify_backend`
                        // no-ops for a recording that is not yet stopped.
                        Ok(DaemonEvent::RecordingStopped { recording_index })
                        | Ok(DaemonEvent::RecordingCloudIdAssigned { recording_index }) => {
                            notify_backend(&store, &client, &org_rx, recording_index).await;
                        }
                        Ok(_) => {}
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            tracing::warn!(
                                skipped,
                                "recording-stop notifier missed bus events; \
                                 re-sweeping pending notifications",
                            );
                            sweep_pending(&store, &client, &org_rx).await;
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            tracing::debug!("event bus closed; recording-stop notifier exiting");
                            break;
                        }
                    }
                }
            }
        }
    });
    RecordingStopNotifierHandle { join }
}

async fn sweep_pending(store: &Arc<SqliteStateStore>, client: &Arc<ApiClient>, org_rx: &OrgIdRx) {
    let pending = match store.recordings_pending_stop_notify().await {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(%error, "failed to query recordings pending stop notify");
            return;
        }
    };
    if pending.is_empty() {
        return;
    }
    tracing::info!(
        count = pending.len(),
        "sweeping recordings with pending backend stop notify",
    );
    for row in pending {
        notify_backend(store, client, org_rx, row.recording_index).await;
    }
}

async fn notify_backend(
    store: &Arc<SqliteStateStore>,
    client: &Arc<ApiClient>,
    org_rx: &OrgIdRx,
    recording_index: i64,
) {
    let row = match store.get_recording(recording_index).await {
        Ok(Some(row)) => row,
        Ok(None) => {
            tracing::warn!(
                recording_index,
                "recording row missing on stop; skipping backend notify",
            );
            return;
        }
        Err(error) => {
            tracing::warn!(
                %error,
                recording_index,
                "failed to look up recording for stop notify",
            );
            return;
        }
    };
    if row.backend_stop_notified_at.is_some() {
        // Another path (sweep or earlier event) already notified.
        return;
    }
    if row.stopped_at.is_none() {
        // Not stopped yet. Reached here via `RecordingCloudIdAssigned` for a
        // still-running recording; the stop will arrive on its own event.
        return;
    }
    let Some(recording_id) = row.recording_id else {
        // The `/start` was never notified, so there is nothing to stop
        // server-side. The stop sweep will re-fire once the start notifier
        // fills in the cloud id.
        tracing::debug!(
            recording_index,
            "recording has no cloud id at stop time; deferring backend notify",
        );
        return;
    };
    let Some(org_id) = org_rx.borrow().clone() else {
        // No current org configured yet. Without it we can't address the POST;
        // the next sweep retries once the config watcher has a current org.
        tracing::warn!(
            recording_index,
            recording_id,
            "no current org_id configured at stop time; skipping backend notify",
        );
        return;
    };
    match client.recording_stop(&org_id, &recording_id).await {
        Ok(()) => {
            if let Err(error) = store.mark_recording_stop_notified(recording_index).await {
                tracing::warn!(
                    %error,
                    recording_index,
                    recording_id,
                    "POST succeeded but persisting backend_stop_notified_at failed; \
                     the next sweep will re-post (the backend POST is idempotent)",
                );
            } else {
                tracing::info!(
                    recording_index,
                    recording_id,
                    "backend notified of recording stop",
                );
            }
        }
        Err(error) => {
            // The producer-side iceoryx2 publish has already succeeded by
            // the time we get here; logging is the only available recourse
            // until the next sweep retries.
            tracing::warn!(
                %error,
                recording_index,
                recording_id,
                "failed to notify backend of recording stop",
            );
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
    use wiremock::matchers::{method, path, query_param};
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

    /// Insert a recording, stamp its cloud id (as if `/start` was notified),
    /// and return its local index.
    async fn seed_notified_recording(store: &SqliteStateStore, recording_id: &str) -> i64 {
        let index = store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                robot_name: Some("arm"),
                dataset_id: Some("ds-1"),
                dataset_name: Some("warehouse"),
                start_timestamp_ns: 1_700_000_000_000_000_000,
            })
            .await
            .expect("create recording")
            .recording_index;
        store
            .mark_recording_start_notified(index, recording_id)
            .await
            .expect("mark start notified");
        index
    }

    /// A live-org receiver fixed at `org`. The sender is leaked so the channel
    /// stays open for the test's duration.
    fn org_rx(org: Option<&str>) -> OrgIdRx {
        let (org_tx, org_rx) = tokio::sync::watch::channel(org.map(str::to_string));
        Box::leak(Box::new(org_tx));
        org_rx
    }

    #[tokio::test]
    async fn posts_backend_stop_on_recording_stopped_event() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/stop"))
            .and(query_param("recording_id", "rec-stop-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!("ok")))
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let index = seed_notified_recording(&store, "rec-stop-1").await;
        store
            .mark_recording_stopped(index, 1)
            .await
            .expect("mark stopped");

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));

        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_stop_notifier(
            store.clone(),
            bus.clone(),
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
        );

        bus.publish(DaemonEvent::RecordingStopped {
            recording_index: index,
        });

        // Give the notifier task a moment to drain the event and call wiremock.
        timeout(Duration::from_secs(3), async {
            loop {
                let received = server.received_requests().await.unwrap_or_default();
                if !received.is_empty() {
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("expected one POST within 3s");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn startup_sweep_recovers_recordings_stopped_while_offline() {
        // Simulate a daemon coming online with a recording that was
        // stopped during a previous offline session: `stopped_at` is
        // already set, `backend_stop_notified_at` is still NULL. The
        // notifier's pre-loop sweep must POST and mark the row notified.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/stop"))
            .and(query_param("recording_id", "rec-offline-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!("ok")))
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let index = seed_notified_recording(&store, "rec-offline-1").await;
        store
            .mark_recording_stopped(index, 1)
            .await
            .expect("mark stopped");

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));

        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_stop_notifier(
            store.clone(),
            bus,
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
        );

        timeout(Duration::from_secs(3), async {
            loop {
                let received = server.received_requests().await.unwrap_or_default();
                if !received.is_empty() {
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("sweep must POST within 3s");

        // Give the notifier a beat to persist the success column.
        timeout(Duration::from_secs(3), async {
            loop {
                let row = store
                    .get_recording(index)
                    .await
                    .expect("get")
                    .expect("exists");
                if row.backend_stop_notified_at.is_some() {
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("backend_stop_notified_at must be stamped within 3s");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn skips_notify_when_recording_row_missing() {
        let server = MockServer::start().await;
        let (store, _dir) = open_store().await;
        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));

        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_stop_notifier(
            store,
            bus.clone(),
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
        );

        bus.publish(DaemonEvent::RecordingStopped {
            recording_index: 9_999,
        });

        // Yield enough for the notifier to process the event and bail. We
        // assert *absence* of an HTTP request: wiremock has no mocks armed,
        // so any incoming request would have already failed the test. A
        // short sleep is the cheapest way to observe quiescence.
        sleep(Duration::from_millis(150)).await;
        let received = server.received_requests().await.unwrap_or_default();
        assert!(
            received.is_empty(),
            "no backend POST expected when recording row is missing"
        );

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn skips_notify_when_cloud_id_absent() {
        // A stopped recording without a cloud id has nothing to stop
        // server-side; the notifier must defer (no POST) until the start
        // notifier fills the id.
        let server = MockServer::start().await;
        let (store, _dir) = open_store().await;
        let index = store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                start_timestamp_ns: 1_700_000_000_000_000_000,
                ..NewRecording::default()
            })
            .await
            .expect("create recording")
            .recording_index;
        store
            .mark_recording_stopped(index, 1)
            .await
            .expect("mark stopped");

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));

        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_stop_notifier(
            store,
            bus.clone(),
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
        );

        bus.publish(DaemonEvent::RecordingStopped {
            recording_index: index,
        });

        sleep(Duration::from_millis(150)).await;
        let received = server.received_requests().await.unwrap_or_default();
        assert!(
            received.is_empty(),
            "no backend POST expected when the cloud recording_id is absent"
        );

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn cloud_id_assigned_event_notifies_a_recording_stopped_while_offline() {
        // Offline recovery: a recording stopped while offline already fired its
        // `RecordingStopped` (which no coordinator saw). Once the start notifier
        // assigns the cloud id and publishes `RecordingCloudIdAssigned`, the
        // stop notifier must POST `/recording/stop`.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/stop"))
            .and(query_param("recording_id", "rec-recovered-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!("ok")))
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let index = seed_notified_recording(&store, "rec-recovered-1").await;
        store
            .mark_recording_stopped(index, 1)
            .await
            .expect("mark stopped");

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));

        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_stop_notifier(
            store.clone(),
            bus.clone(),
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
        );

        // The cloud-id-assigned event — not RecordingStopped — drives the POST.
        bus.publish(DaemonEvent::RecordingCloudIdAssigned {
            recording_index: index,
        });

        timeout(Duration::from_secs(3), async {
            loop {
                let received = server.received_requests().await.unwrap_or_default();
                if !received.is_empty() {
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("cloud-id-assigned event must POST within 3s");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn cloud_id_assigned_event_ignores_a_running_recording() {
        // A recording that just got its cloud id but has not stopped yet must
        // not be stop-notified — the `stopped_at` guard holds the POST until the
        // recording actually stops.
        let server = MockServer::start().await;
        let (store, _dir) = open_store().await;
        let index = seed_notified_recording(&store, "rec-running-1").await;
        // Deliberately NOT stopped.

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));

        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_stop_notifier(
            store,
            bus.clone(),
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
        );

        bus.publish(DaemonEvent::RecordingCloudIdAssigned {
            recording_index: index,
        });

        sleep(Duration::from_millis(150)).await;
        let received = server.received_requests().await.unwrap_or_default();
        assert!(
            received.is_empty(),
            "no backend POST expected for a recording that has not stopped"
        );

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }
}
