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
//! The shared loop/sweep/lag semantics live in
//! [`notifier`](super::notifier); see there for how events are processed. What
//! is start-specific: the cloud `recording_id` is minted and persisted here.
//! Every downstream coordinator (registration, progress, upload) waits for this
//! id, so an offline recording simply stays pending until the daemon is online
//! and `/recording/start` lands.
//!
//! A recording cancelled before its start POST goes out is settled here
//! instead: no start is sent, and `backend_cancel_notified_at` is stamped
//! directly, since a recording the backend never heard of has nothing to
//! cancel. See the guard in [`notify_backend`] for why this notifier — not the
//! cancel notifier — owns that case.
//!
//! Every POST opens a distinct backend recording — the backend never reuses a
//! pending one for the source — so recordings that follow each other with no
//! gap stay separate no matter what order their stop and start notifications
//! land in.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::broadcast;

use super::notifier::{spawn_notifier, NotifierCtx, NotifierHandle, RecordingNotifier};
use crate::api::ApiClient;
use crate::cloud::OrgIdRx;
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::{
    DaemonEvent, EventBus, RecordingRow, SqliteStateStore, StateStore, StateStoreError,
};

/// Notifier that POSTs `/recording/start` and persists the cloud `recording_id`
/// the backend mints. The cloud id is always minted here — every downstream
/// coordinator waits on it — so an offline recording stays pending until the
/// daemon is online and the start POST lands.
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

    /// Only the cancelled rows, which this notifier settles locally without
    /// POSTing anything.
    ///
    /// `/recording/start` is deliberately not idempotent — every POST opens a
    /// distinct backend recording — so a start the backend commits but the
    /// client never sees (a timeout, a dropped connection) leaves a row that
    /// re-posting turns into a second, orphaned backend recording. Re-driving
    /// that on a 30 s timer would mint one orphan per tick for as long as the
    /// condition lasts. The full set is still swept at startup and after a
    /// lag, which is where an offline recording recovers.
    async fn periodic_pending(
        &self,
        store: &Arc<SqliteStateStore>,
    ) -> Result<Vec<RecordingRow>, StateStoreError> {
        Ok(store
            .recordings_pending_start_notify()
            .await?
            .into_iter()
            .filter(|row| row.cancelled_at.is_some())
            .collect())
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
    if row.cancelled_at.is_some() {
        // Cancelled before `/recording/start` went out, so the backend never
        // learned this recording exists and there is nothing to cancel
        // server-side. Settle the cancel here rather than POSTing a start
        // purely to cancel it a moment later.
        //
        // This notifier is the only place that can make that call. At the
        // cancel notifier's own cloud-id guard a NULL `recording_id` is
        // ambiguous — never POSTed, or POSTed and still in flight — and
        // stamping the cancel in the second case would let the reaper delete
        // the row while the response that mints its id is still coming back.
        // Here the ambiguity cannot arise: the notifier task issues its POST
        // inline, so reaching this line proves no start POST is outstanding.
        tracing::info!(
            recording_index,
            "recording cancelled before its start was notified; \
             nothing to cancel server-side"
        );
        if let Err(error) = store.mark_recording_cancel_notified(recording_index).await {
            // Left unstamped the reaper cannot reclaim it; the next sweep
            // retries, since the row stays in the pending-start set.
            tracing::warn!(
                %error,
                recording_index,
                "failed to settle the cancel of a never-started recording",
            );
        }
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
                     the next sweep re-posts and opens a second backend recording \
                     for this one — the orphan carries no traces and the backend \
                     reaps it once it passes the maximum recording duration",
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
    use crate::lifecycle::shutdown::ShutdownSignal;
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

    /// A recording cancelled before its start POST goes out is settled here:
    /// no `/recording/start` is sent, and the cancel is stamped locally so the
    /// reaper can reclaim it. Without this the row sits in no sweep at all —
    /// the cancel notifier's pending set requires a cloud id it will never
    /// get — and is pinned forever.
    #[tokio::test]
    async fn settles_a_recording_cancelled_before_its_start_was_posted() {
        let server = MockServer::start().await;
        start_ok_mock("cloud-never-used").mount(&server).await;

        let (store, _dir) = open_store().await;
        let index = seed_recording(&store).await;
        store
            .cancel_recording(index, 5_000_000_000)
            .await
            .expect("cancel");

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

        // The startup sweep alone must settle it — the recording is cancelled,
        // so no `RecordingStarted` event is coming.
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
        .expect("the sweep must settle the cancel within 3s");

        assert!(
            server
                .received_requests()
                .await
                .unwrap_or_default()
                .is_empty(),
            "a cancelled recording must not open a backend recording"
        );
        assert!(
            store
                .get_recording(index)
                .await
                .expect("get")
                .expect("exists")
                .recording_id
                .is_none(),
            "no cloud id is minted for a recording the backend never heard of"
        );

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    /// The periodic re-sweep must never re-POST a start. The POST is not
    /// idempotent, so a backend that committed a start the client never saw
    /// would collect one orphan recording per tick.
    #[tokio::test]
    async fn resweep_never_reposts_a_live_recordings_start() {
        let server = MockServer::start().await;
        start_ok_mock("cloud-orphan").mount(&server).await;

        let (store, _dir) = open_store().await;
        let live = seed_recording(&store).await;
        let cancelled = seed_recording(&store).await;
        store
            .cancel_recording(cancelled, 5_000_000_000)
            .await
            .expect("cancel");

        // Both rows are pending a start POST; only the cancelled one is safe
        // to re-drive on a timer.
        let full = StartNotifier
            .pending(&Arc::new(store.clone()))
            .await
            .unwrap();
        assert_eq!(full.len(), 2, "the full sweep still sees both");
        let periodic = StartNotifier
            .periodic_pending(&Arc::new(store.clone()))
            .await
            .unwrap();
        let indices: Vec<i64> = periodic.iter().map(|row| row.recording_index).collect();
        assert_eq!(
            indices,
            vec![cancelled],
            "the periodic re-sweep must skip the live recording"
        );
        assert_ne!(indices, vec![live]);
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
    async fn consecutive_recordings_for_one_source_each_get_their_own_cloud_id() {
        // Stop-then-start (or cancel-then-start) with no gap for one source: the
        // prior recording's stop notification may not have landed yet when the
        // next start is notified. Each start must still open its own backend
        // recording and persist its own cloud id, or the two collapse into one
        // and a recording is silently lost at every boundary.
        let server = MockServer::start().await;
        start_ok_mock("cloud-rec-A")
            .up_to_n_times(1)
            .mount(&server)
            .await;
        start_ok_mock("cloud-rec-B").mount(&server).await;

        let (store, _dir) = open_store().await;
        let first = seed_recording(&store).await;
        let second = seed_recording(&store).await;

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

        timeout(Duration::from_secs(3), async {
            loop {
                let first_row = store
                    .get_recording(first)
                    .await
                    .expect("get")
                    .expect("exists");
                let second_row = store
                    .get_recording(second)
                    .await
                    .expect("get")
                    .expect("exists");
                if let (Some(first_id), Some(second_id)) =
                    (first_row.recording_id, second_row.recording_id)
                {
                    assert_ne!(
                        first_id, second_id,
                        "consecutive recordings must not share a cloud recording id",
                    );
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("both recordings must get a distinct cloud id within 3s");

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
