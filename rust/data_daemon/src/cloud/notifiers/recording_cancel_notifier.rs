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
//! to cancel server-side. A NULL id is ambiguous here, since the start POST
//! may not have been sent or may still be in flight, so this notifier defers
//! the recording. The start notifier settles the cancel itself if it never
//! POSTs, and otherwise publishes [`DaemonEvent::RecordingCloudIdAssigned`],
//! which brings the recording back here with an id to cancel.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::broadcast;

use super::notifier::{
    notify_recording_lifecycle, spawn_notifier, LifecycleKind, NotifierCtx, NotifierHandle,
    RecordingNotifier,
};
use crate::api::ApiClient;
use crate::cloud::OrgIdRx;
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::{
    DaemonEvent, EventBus, RecordingRow, SqliteStateStore, StateStore, StateStoreError,
};

/// Notifier that POSTs `/recording/cancel` once a recording is cancelled and
/// its cloud id is known. Recordings cancelled before `/recording/start` ever
/// landed have no cloud representation, so `notify_recording_lifecycle`
/// defers them until the start notifier resolves which case they are.
struct CancelNotifier;

#[async_trait]
impl RecordingNotifier for CancelNotifier {
    fn label(&self) -> &'static str {
        "recording-cancel"
    }

    fn triggered_by(&self, event: &DaemonEvent) -> Option<i64> {
        match event {
            // A cancel deferred for want of a cloud id resumes on
            // `RecordingCloudIdAssigned`, which `notify_recording_lifecycle`
            // ignores for a recording that is not cancelled.
            DaemonEvent::RecordingCancelled { recording_index }
            | DaemonEvent::RecordingCloudIdAssigned { recording_index } => Some(*recording_index),
            _ => None,
        }
    }

    async fn pending(
        &self,
        store: &Arc<SqliteStateStore>,
    ) -> Result<Vec<RecordingRow>, StateStoreError> {
        store.recordings_pending_cancel_notify().await
    }

    async fn notify(&self, ctx: &NotifierCtx, recording_index: i64) {
        notify_recording_lifecycle(
            LifecycleKind::Cancel,
            &ctx.store,
            &ctx.client,
            &ctx.org_rx,
            recording_index,
        )
        .await;
    }
}

/// Spawn the recording-cancel notifier on the current Tokio runtime.
pub fn spawn_recording_cancel_notifier(
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> NotifierHandle {
    spawn_notifier(CancelNotifier, store, bus, client, org_rx, shutdown_rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::RecordingTimestamp;

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
                start: RecordingTimestamp::observed_at(0),
                ..NewRecording::default()
            })
            .await
            .expect("create_recording");
        let index = row.recording_index;
        store
            .mark_recording_start_notified(index, cloud_id)
            .await
            .expect("mark start notified");
        let cancel = RecordingTimestamp {
            publish_timestamp_ns: 5_000_000_000,
            timestamp_us: Some(3_000_000),
        };
        store.cancel_recording(index, cancel).await.expect("cancel");
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
            // The backend answers a successful cancel with 204 and no body.
            .respond_with(ResponseTemplate::new(204))
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

        let received = timeout(Duration::from_secs(3), async {
            loop {
                let received = server.received_requests().await.unwrap_or_default();
                if !received.is_empty() {
                    break received;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("expected one POST within 3s");
        let body: serde_json::Value = received[0].body_json().expect("json body");
        assert_eq!(
            body,
            serde_json::json!({
                "recording_id": "rec-cancel-1",
                "end_time": 3.0,
            }),
            "end_time is the caller's cancel"
        );

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    /// A cancel that races the start POST finds no cloud id and defers; the id
    /// assignment that follows must fire it.
    #[tokio::test]
    async fn cloud_id_assignment_fires_a_cancel_that_raced_the_start_post() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/cancel"))
            .and(body_partial_json(
                serde_json::json!({ "recording_id": "rec-raced" }),
            ))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        // Cancelled first, with no cloud id yet — the start POST is in flight.
        let row = store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                start: RecordingTimestamp::observed_at(0),
                ..NewRecording::default()
            })
            .await
            .expect("create_recording");
        let index = row.recording_index;
        store
            .cancel_recording(index, RecordingTimestamp::observed_at(5_000_000_000))
            .await
            .expect("cancel");

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

        // The cancel event alone can do nothing: there is no id to cancel.
        bus.publish(DaemonEvent::RecordingCancelled {
            recording_index: index,
        });
        sleep(Duration::from_millis(150)).await;
        assert!(
            server
                .received_requests()
                .await
                .unwrap_or_default()
                .is_empty(),
            "no POST is possible before the cloud id exists"
        );

        // The start POST lands; its id assignment must resume the cancel.
        store
            .mark_recording_start_notified(index, "rec-raced")
            .await
            .expect("mark start notified");
        bus.publish(DaemonEvent::RecordingCloudIdAssigned {
            recording_index: index,
        });

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
        .expect("the id assignment must drive the cancel POST within 3s");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    /// `RecordingCloudIdAssigned` fires for every recording whose start lands,
    /// so it must not cancel a recording that is merely running.
    #[tokio::test]
    async fn cloud_id_assignment_does_not_cancel_a_live_recording() {
        let server = MockServer::start().await;
        let (store, _dir) = open_store().await;
        let row = store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                start: RecordingTimestamp::observed_at(0),
                ..NewRecording::default()
            })
            .await
            .expect("create_recording");
        let index = row.recording_index;
        store
            .mark_recording_start_notified(index, "rec-live")
            .await
            .expect("mark start notified");

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

        bus.publish(DaemonEvent::RecordingCloudIdAssigned {
            recording_index: index,
        });
        sleep(Duration::from_millis(150)).await;

        assert!(
            server
                .received_requests()
                .await
                .unwrap_or_default()
                .is_empty(),
            "a running recording must not be cancelled by its own id assignment"
        );
        assert!(
            store
                .get_recording(index)
                .await
                .expect("get")
                .expect("exists")
                .backend_cancel_notified_at
                .is_none(),
            "a running recording must not be stamped cancel-notified"
        );

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
            // The backend answers a successful cancel with 204 and no body.
            .respond_with(ResponseTemplate::new(204))
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

    /// A cancel POST that fails transiently is retried by the periodic
    /// re-sweep, without a restart.
    #[tokio::test]
    async fn resweep_retries_a_cancel_whose_post_failed() {
        let server = MockServer::start().await;
        // First attempt fails; wiremock serves mounts newest-first, so the
        // limited 503 shadows the 204 until it is used up.
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/cancel"))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/cancel"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let index = seed_cancelled_recording_with_cloud_id(&store, "rec-flaky").await;

        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));
        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = super::super::notifier::spawn_notifier_every(
            CancelNotifier,
            store.clone(),
            bus,
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
            Duration::from_millis(50),
        );

        timeout(Duration::from_secs(5), async {
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
        .expect("the re-sweep must recover the failed cancel without a restart");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn treats_backend_404_as_already_cancelled() {
        // The recording may already be closed on the backend — an earlier POST
        // whose response was lost, or the backend reaping it as abandoned — so a
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
    async fn posts_backend_cancel_when_the_cloud_id_arrives_after_the_cancel() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/cancel"))
            .and(body_partial_json(
                serde_json::json!({ "recording_id": "rec-late" }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!("ok")))
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let row = store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                start: RecordingTimestamp::observed_at(0),
                ..NewRecording::default()
            })
            .await
            .unwrap();
        store
            .cancel_recording(
                row.recording_index,
                RecordingTimestamp::observed_at(5_000_000_000),
            )
            .await
            .unwrap();

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

        // The startup sweep runs first and finds nothing to cancel.
        sleep(Duration::from_millis(150)).await;
        assert!(server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty());

        // The start POST that was in flight at the cancel returns now.
        store
            .mark_recording_start_notified(row.recording_index, "rec-late")
            .await
            .unwrap();
        bus.publish(DaemonEvent::RecordingCloudIdAssigned {
            recording_index: row.recording_index,
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
        .expect("expected one POST within 3s");
        let received = server.received_requests().await.unwrap_or_default();
        let body: serde_json::Value = received[0].body_json().expect("json body");
        assert_eq!(body["recording_id"], "rec-late");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn leaves_a_running_recording_alone_when_its_cloud_id_arrives() {
        let server = MockServer::start().await;
        let (store, _dir) = open_store().await;
        let row = store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                start: RecordingTimestamp::observed_at(0),
                ..NewRecording::default()
            })
            .await
            .unwrap();
        store
            .mark_recording_start_notified(row.recording_index, "rec-running")
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

        bus.publish(DaemonEvent::RecordingCloudIdAssigned {
            recording_index: row.recording_index,
        });

        sleep(Duration::from_millis(150)).await;
        let received = server.received_requests().await.unwrap_or_default();
        assert!(
            received.is_empty(),
            "no backend POST expected for a recording that is not cancelled"
        );

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
                start: RecordingTimestamp::observed_at(0),
                ..NewRecording::default()
            })
            .await
            .unwrap();
        store
            .cancel_recording(
                row.recording_index,
                RecordingTimestamp::observed_at(5_000_000_000),
            )
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

    #[tokio::test]
    async fn treats_backend_403_as_terminal() {
        // 403 is permanent, so the cancel is stamped rather than retried.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/cancel"))
            .respond_with(
                ResponseTemplate::new(403)
                    .set_body_json(serde_json::json!({ "detail": "Not permitted." })),
            )
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let index = seed_cancelled_recording_with_cloud_id(&store, "rec-forbidden").await;

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
        .expect("a 403 must still stamp backend_cancel_notified_at within 3s");

        // Nothing left for a sweep to pick up.
        assert!(
            store
                .recordings_pending_cancel_notify()
                .await
                .expect("pending")
                .is_empty(),
            "a 403 must not leave the recording pending a re-post"
        );

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn treats_already_uploaded_conflict_as_terminal() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/cancel"))
            .respond_with(ResponseTemplate::new(409).set_body_json(serde_json::json!({
                "detail": {
                    "error": "Recording has already been uploaded.",
                    "status": 409,
                    "error_code": "RECORDING_ALREADY_UPLOADED"
                }
            })))
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let index = seed_cancelled_recording_with_cloud_id(&store, "rec-uploaded").await;

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
        .expect("the terminal 409 must stamp backend_cancel_notified_at within 3s");

        assert!(
            store
                .recordings_pending_cancel_notify()
                .await
                .expect("pending")
                .is_empty(),
            "the terminal 409 must not leave the recording pending a re-post"
        );

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }
}
