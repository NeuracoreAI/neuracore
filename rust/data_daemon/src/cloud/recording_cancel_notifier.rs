//! Backend recording-cancel notifier.
//!
//! Subscribes to [`DaemonEvent::RecordingCancelled`] and POSTs
//! `/org/{org}/recording/cancel` (JSON body `{recording_id, end_time}`) to the
//! backend. The Python
//! SDK used to make this call inline from `nc.cancel_recording`, but that
//! required the SDK to know the cloud `recording_id` — which the thin-shipper
//! model removes. The notifier picks up the responsibility: once the local
//! cancel is stamped and the cloud id is known, it fires the POST in the
//! background with the daemon's standard retry policy.
//!
//! Recordings cancelled before `/recording/start` was ever notified (i.e.
//! `recording_id IS NULL`) have no cloud representation, so there is nothing
//! to cancel server-side; the notifier silently skips them.

use std::sync::Arc;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;

use crate::api::ApiClient;
use crate::cloud::OrgIdRx;
use crate::lifecycle::signals::ShutdownSignal;
use crate::state::{DaemonEvent, EventBus, SqliteStateStore, StateStore};

/// Handle returned by [`spawn_recording_cancel_notifier`].
pub struct RecordingCancelNotifierHandle {
    join: JoinHandle<()>,
}

impl RecordingCancelNotifierHandle {
    /// Wait for the notifier task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "recording-cancel notifier join failed");
        }
    }
}

/// Spawn the recording-cancel notifier on the current Tokio runtime.
///
/// The task first sweeps any recordings cancelled while the daemon was offline
/// (rows with `cancelled_at NOT NULL`, `recording_id NOT NULL`, and no
/// `backend_cancel_notified_at`), then drops into the event-bus loop and POSTs
/// whenever a fresh `RecordingCancelled` event fires.
pub fn spawn_recording_cancel_notifier(
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> RecordingCancelNotifierHandle {
    let mut subscriber = bus.subscribe();
    let store = Arc::new(store);
    let join = tokio::spawn(async move {
        tokio::select! {
            biased;
            signal = shutdown_rx.recv() => {
                tracing::debug!(?signal, "recording-cancel notifier shutting down before sweep");
                return;
            }
            _ = sweep_pending(&store, &client, &org_rx) => {}
        }
        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "recording-cancel notifier shutting down");
                    break;
                }
                event = subscriber.recv() => {
                    match event {
                        Ok(DaemonEvent::RecordingCancelled { recording_index }) => {
                            notify_backend(&store, &client, &org_rx, recording_index).await;
                        }
                        Ok(_) => {}
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                            tracing::warn!(
                                skipped,
                                "recording-cancel notifier missed bus events; \
                                 re-sweeping pending notifications",
                            );
                            sweep_pending(&store, &client, &org_rx).await;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                            tracing::debug!("event bus closed; recording-cancel notifier exiting");
                            break;
                        }
                    }
                }
            }
        }
    });
    RecordingCancelNotifierHandle { join }
}

async fn sweep_pending(store: &Arc<SqliteStateStore>, client: &Arc<ApiClient>, org_rx: &OrgIdRx) {
    let pending = match store.recordings_pending_cancel_notify().await {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(%error, "failed to query recordings pending cancel notify");
            return;
        }
    };
    if pending.is_empty() {
        return;
    }
    tracing::info!(
        count = pending.len(),
        "sweeping recordings with pending backend cancel notify",
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
                "recording row missing on cancel; skipping backend notify",
            );
            return;
        }
        Err(error) => {
            tracing::warn!(%error, recording_index, "failed to look up recording for cancel notify");
            return;
        }
    };

    if row.backend_cancel_notified_at.is_some() {
        return;
    }

    // If the recording was never started server-side there is nothing to
    // cancel — the recording simply doesn't exist on the backend.
    let Some(recording_id) = row.recording_id else {
        tracing::debug!(
            recording_index,
            "recording has no cloud id; skipping cancel notify (was never started server-side)"
        );
        return;
    };

    let Some(org_id) = org_rx.borrow().clone() else {
        tracing::warn!(
            recording_index,
            "no current org_id configured at cancel time; skipping backend notify",
        );
        return;
    };

    // A cancel is a recording stop that discards data: send the captured cancel
    // time as `end_time`, exactly as the stop notifier does.
    let Some(stop_timestamp_ns) = row.stop_timestamp_ns else {
        tracing::warn!(
            recording_index,
            recording_id,
            "cancelled recording has no stop_timestamp_ns; skipping backend cancel notify",
        );
        return;
    };
    let end_time = stop_timestamp_ns as f64 / 1_000_000_000.0;

    match client
        .recording_cancel(&org_id, &recording_id, end_time)
        .await
    {
        Ok(()) => {
            if let Err(error) = store.mark_recording_cancel_notified(recording_index).await {
                tracing::warn!(
                    %error,
                    recording_index,
                    recording_id,
                    "POST succeeded but persisting backend_cancel_notified_at failed; \
                     the next sweep will re-post (the backend POST is idempotent)",
                );
            } else {
                tracing::info!(
                    recording_index,
                    recording_id,
                    "backend notified of recording cancel"
                );
            }
        }
        Err(error) if error.is_not_found() => {
            // 404 means the backend no longer has this recording open — the
            // start-notifier's `resolve_prior_pending` already closed it when
            // the next recording for this source opened. That is exactly the
            // post-condition we wanted, so record it as notified rather than
            // re-sweeping forever.
            if let Err(error) = store.mark_recording_cancel_notified(recording_index).await {
                tracing::warn!(
                    %error,
                    recording_index,
                    recording_id,
                    "persisting backend_cancel_notified_at after a 404 failed; will re-sweep",
                );
            } else {
                tracing::debug!(
                    recording_index,
                    recording_id,
                    "recording already closed on backend (404); treated as cancel-notified",
                );
            }
        }
        Err(error) => {
            tracing::warn!(
                %error,
                recording_index,
                recording_id,
                "failed to notify backend of recording cancel",
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
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::api::auth::StaticAuthProvider;
    use crate::api::{ApiClient, ApiClientOptions};
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

    async fn seed_cancelled_recording_with_cloud_id(
        store: &SqliteStateStore,
        cloud_id: &str,
    ) -> i64 {
        let row = store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                start_timestamp_ns: 0,
                ..NewRecording::default()
            })
            .await
            .expect("create_recording");
        let index = row.recording_index;
        store
            .mark_recording_start_notified(index, cloud_id)
            .await
            .expect("mark start notified");
        store
            .cancel_recording(index, 5_000_000_000)
            .await
            .expect("cancel");
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
    async fn posts_backend_cancel_on_recording_cancelled_event() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/cancel"))
            .and(body_partial_json(
                serde_json::json!({ "recording_id": "rec-cancel-1" }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!("ok")))
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        seed_cancelled_recording_with_cloud_id(&store, "rec-cancel-1").await;

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));
        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_cancel_notifier(
            store.clone(),
            bus.clone(),
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
        );

        bus.publish(DaemonEvent::RecordingCancelled { recording_index: 1 });

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
    async fn startup_sweep_recovers_recordings_cancelled_while_offline() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/cancel"))
            .and(body_partial_json(
                serde_json::json!({ "recording_id": "rec-offline-cancel" }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!("ok")))
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let index = seed_cancelled_recording_with_cloud_id(&store, "rec-offline-cancel").await;

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));
        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_cancel_notifier(
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

        timeout(Duration::from_secs(3), async {
            loop {
                let row = store
                    .get_recording(index)
                    .await
                    .expect("get")
                    .expect("exists");
                if row.backend_cancel_notified_at.is_some() {
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("backend_cancel_notified_at must be stamped within 3s");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn treats_backend_404_as_already_cancelled() {
        // The start-notifier's `resolve_prior_pending` may have closed this
        // recording on the backend first (cancel-then-start with no gap), so a
        // 404 here is the desired post-condition, not a failure: the row must
        // still be marked notified so the sweep stops re-posting.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/cancel"))
            .respond_with(
                ResponseTemplate::new(404)
                    .set_body_json(serde_json::json!({ "detail": "Recording not found." })),
            )
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let index = seed_cancelled_recording_with_cloud_id(&store, "rec-already-gone").await;

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));
        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_cancel_notifier(
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
                if row.backend_cancel_notified_at.is_some() {
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("a 404 must still stamp backend_cancel_notified_at within 3s");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn skips_notify_when_recording_has_no_cloud_id() {
        let server = MockServer::start().await;
        let (store, _dir) = open_store().await;

        // A recording that was cancelled before /start was ever notified.
        let row = store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                start_timestamp_ns: 0,
                ..NewRecording::default()
            })
            .await
            .unwrap();
        store
            .cancel_recording(row.recording_index, 5_000_000_000)
            .await
            .unwrap();

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));
        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_cancel_notifier(
            store,
            bus.clone(),
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
        );

        bus.publish(DaemonEvent::RecordingCancelled {
            recording_index: row.recording_index,
        });

        sleep(Duration::from_millis(150)).await;
        let received = server.received_requests().await.unwrap_or_default();
        assert!(
            received.is_empty(),
            "no backend POST expected when recording has no cloud id"
        );

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }
}
