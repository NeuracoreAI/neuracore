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

use tokio::sync::broadcast;
use tokio::task::JoinHandle;

use crate::api::ApiClient;
use crate::lifecycle::signals::ShutdownSignal;
use crate::state::{DaemonEvent, EventBus, SqliteStateStore, StateStore};

/// Handle returned by [`spawn_recording_start_notifier`].
pub struct RecordingStartNotifierHandle {
    join: JoinHandle<()>,
}

impl RecordingStartNotifierHandle {
    /// Wait for the notifier task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "recording-start notifier join failed");
        }
    }
}

/// Spawn the recording-start notifier on the current Tokio runtime.
///
/// The task first sweeps any recordings opened while the daemon was offline
/// (rows whose `/recording/start` POST has not yet succeeded), then drops into
/// the event-bus loop and POSTs whenever a fresh `RecordingStarted` event
/// fires.
pub fn spawn_recording_start_notifier(
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> RecordingStartNotifierHandle {
    let mut subscriber = bus.subscribe();
    let store = Arc::new(store);
    let join = tokio::spawn(async move {
        // Recover any opened-while-offline recordings before serving live
        // events. Run inside a `select!` against the shutdown signal so a
        // long sweep cannot hold the daemon shutting down.
        tokio::select! {
            biased;
            signal = shutdown_rx.recv() => {
                tracing::debug!(?signal, "recording-start notifier shutting down before sweep");
                return;
            }
            _ = sweep_pending(&store, &client, &bus) => {}
        }
        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "recording-start notifier shutting down");
                    break;
                }
                event = subscriber.recv() => {
                    match event {
                        Ok(DaemonEvent::RecordingStarted { recording_index }) => {
                            notify(&store, &client, &bus, recording_index).await;
                        }
                        Ok(_) => {}
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            tracing::warn!(
                                skipped,
                                "recording-start notifier missed bus events; \
                                 re-sweeping pending notifications",
                            );
                            sweep_pending(&store, &client, &bus).await;
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            tracing::debug!("event bus closed; recording-start notifier exiting");
                            break;
                        }
                    }
                }
            }
        }
    });
    RecordingStartNotifierHandle { join }
}

async fn sweep_pending(store: &Arc<SqliteStateStore>, client: &Arc<ApiClient>, bus: &EventBus) {
    let pending = match store.recordings_pending_start_notify().await {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(%error, "failed to query recordings pending start notify");
            return;
        }
    };
    if pending.is_empty() {
        return;
    }
    tracing::info!(
        count = pending.len(),
        "sweeping recordings with pending backend start notify",
    );
    for row in pending {
        notify(store, client, bus, row.recording_index).await;
    }
}

async fn notify(
    store: &Arc<SqliteStateStore>,
    client: &Arc<ApiClient>,
    bus: &EventBus,
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

    let Some(org_id) = row.org_id else {
        // No org stamped yet — the producer never reached the daemon's
        // recording row with one. Without it we can't address the POST.
        tracing::warn!(
            recording_index,
            "recording has no org_id at start time; skipping backend notify",
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

    match client
        .recording_start(&org_id, &robot_id, instance, &dataset_id)
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    use tempfile::TempDir;
    use tokio::sync::broadcast;
    use tokio::time::{sleep, timeout};
    use wiremock::matchers::{method, path};
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
    async fn seed_recording(store: &SqliteStateStore, org_id: &str) -> i64 {
        store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(7),
                robot_name: Some("arm"),
                dataset_id: Some("ds-1"),
                dataset_name: Some("warehouse"),
                org_id: Some(org_id),
                start_timestamp_ns: 1_700_000_000_000_000_000,
            })
            .await
            .expect("create recording")
            .recording_index
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
        let index = seed_recording(&store, "org-1").await;

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));

        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_recording_start_notifier(
            store.clone(),
            bus.clone(),
            client,
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
    async fn startup_sweep_notifies_recordings_opened_while_offline() {
        // A recording opened during a previous offline session: no cloud id,
        // no start-notify/failed stamps. The pre-loop sweep must POST and
        // persist the minted cloud id.
        let server = MockServer::start().await;
        start_ok_mock("cloud-rec-offline").mount(&server).await;

        let (store, _dir) = open_store().await;
        let index = seed_recording(&store, "org-1").await;

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));

        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle =
            spawn_recording_start_notifier(store.clone(), bus, client, shutdown_tx.subscribe());

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
