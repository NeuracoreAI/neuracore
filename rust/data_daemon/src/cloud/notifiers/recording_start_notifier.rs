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
//! A recording cancelled before its start POST goes out is settled here: no
//! start is sent and the cancel is stamped as delivered, since the backend
//! never heard of it. [`notify_backend`] explains why this notifier, not the
//! cancel notifier, owns that case.
//!
//! A recording with a robot or dataset row is gated on it: it is started only
//! once the server has confirmed the row, with the server's id. This notifier
//! knows nothing about how rows are confirmed: every
//! [`START_RESWEEP`](crate::intervals::START_RESWEEP) it queues again the
//! recordings whose start was never sent and whose row is confirmed. The
//! queue sends one start at a time and pauses while offline. A recording's
//! `start_sent_at` is stamped before its start is sent, so a start that may
//! have reached the server is never sent again by the re-check.
//! Every server call about a recording goes to its robot row's organization
//! ([`crate::cloud::recording_org_id`]).
//!
//! Every POST opens a distinct backend recording — the backend never reuses a
//! pending one for the source — so recordings that follow each other with no
//! gap stay separate no matter what order their stop and start notifications
//! land in.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::broadcast;

use super::notifier::{spawn_notifier_every, NotifierCtx, NotifierHandle, RecordingNotifier};
use crate::api::ApiClient;
use crate::cloud::OrgIdRx;
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::{
    DaemonEvent, EventBus, RecordingRow, ResolutionStatus, SqliteStateStore, StateStore,
    StateStoreError,
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

    /// The cancelled rows, which are settled locally without a POST, and the
    /// rows whose start was never sent and whose dataset row is confirmed.
    ///
    /// `/recording/start` is not idempotent: a start the backend committed but
    /// the client never saw (a timeout) would become another orphaned backend
    /// recording on every re-sweep, so a start once sent is left out. The full
    /// set is still swept at startup and after a lag.
    async fn periodic_pending(
        &self,
        store: &Arc<SqliteStateStore>,
    ) -> Result<Vec<RecordingRow>, StateStoreError> {
        store.recordings_for_start_resweep().await
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
    spawn_start_notifier_every(
        store,
        bus,
        client,
        org_rx,
        shutdown_rx,
        crate::intervals::START_RESWEEP,
    )
}

/// [`spawn_recording_start_notifier`] with the re-check interval as a
/// parameter, for tests.
fn spawn_start_notifier_every(
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
    resweep_every: Duration,
) -> NotifierHandle {
    spawn_notifier_every(
        StartNotifier,
        store,
        bus,
        client,
        org_rx,
        shutdown_rx,
        resweep_every,
    )
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
        // Cancelled before `/recording/start` went out, so the backend has
        // nothing to cancel. Only this notifier can settle that: it sends its
        // POSTs inline, so here no start can be in flight, whereas to the
        // cancel notifier a missing cloud id may just mean the start response
        // has not arrived yet.
        tracing::info!(
            recording_index,
            "recording cancelled before its start was notified; \
             nothing to cancel server-side"
        );
        if let Err(error) = store.mark_recording_cancel_notified(recording_index).await {
            // The row stays in the pending-start set, so the next sweep retries.
            tracing::warn!(
                %error,
                recording_index,
                "failed to settle the cancel of a never-started recording",
            );
        }
        return;
    }

    // The gate: a recording with a robot or dataset row waits until the
    // server has confirmed it. The re-check brings it back here once it is.
    let robot_id = match row.robot_index {
        Some(robot_index) => match store.get_robot(robot_index).await {
            Ok(Some(robot)) if robot.status == ResolutionStatus::Done => robot.cloud_robot_id,
            Ok(Some(_)) => {
                tracing::debug!(
                    recording_index,
                    robot_index,
                    "robot not confirmed by the server yet; start waits for it",
                );
                return;
            }
            Ok(None) => None,
            Err(error) => {
                tracing::warn!(%error, recording_index, "failed to read the recording's robot");
                return;
            }
        },
        None => row.robot_id,
    };
    let dataset_id = match row.dataset_index {
        Some(dataset_index) => match store.get_dataset(dataset_index).await {
            Ok(Some(dataset)) if dataset.status == ResolutionStatus::Done => {
                dataset.cloud_dataset_id
            }
            Ok(Some(_)) => {
                tracing::debug!(
                    recording_index,
                    dataset_index,
                    "dataset not confirmed by the server yet; start waits for it",
                );
                return;
            }
            Ok(None) => None,
            Err(error) => {
                tracing::warn!(%error, recording_index, "failed to read the recording's dataset");
                return;
            }
        },
        None => row.dataset_id,
    };
    let org_id = match crate::cloud::recording_org_id(store, recording_index, org_rx).await {
        Ok(org_id) => org_id,
        Err(error) => {
            tracing::warn!(%error, recording_index, "failed to read the recording's organization");
            return;
        }
    };
    let Some(org_id) = org_id else {
        // No current org configured yet (not logged in / org not selected).
        // Without it we can't address the POST; the next sweep retries once
        // the config watcher picks up a current org.
        tracing::warn!(
            recording_index,
            "no current org_id configured at start time; skipping backend notify",
        );
        return;
    };
    let Some(robot_id) = robot_id else {
        tracing::warn!(
            recording_index,
            "recording has no robot_id at start time; skipping backend notify",
        );
        return;
    };
    let Some(dataset_id) = dataset_id else {
        tracing::warn!(
            recording_index,
            "recording has no dataset_id at start time; skipping backend notify",
        );
        return;
    };
    let instance = row.robot_instance.unwrap_or(0);
    let Some(start_timestamp_us) = row.start_timestamp_us else {
        tracing::warn!(
            recording_index,
            "recording has no start_timestamp_us at start time; skipping backend notify",
        );
        return;
    };
    // Stamped first: from here on the start may reach the server, so the
    // re-check must not send it again.
    if let Err(error) = store.mark_start_sent(recording_index).await {
        tracing::warn!(%error, recording_index, "failed to stamp the start as sent; retrying later");
        return;
    }
    // The producer captured this as the recording window's real lower bound;
    // the backend derives the reported duration from it, so a late notify
    // (e.g. after reconnecting) still reports correctly.
    match client
        .recording_start(
            &org_id,
            &robot_id,
            instance,
            &dataset_id,
            start_timestamp_us,
        )
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
    use crate::state::RecordingTimestamp;

    use std::time::Duration;

    use tempfile::TempDir;
    use tokio::sync::broadcast;
    use tokio::time::{sleep, timeout};
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::api::auth::StaticAuthProvider;
    use crate::api::{ApiClient, ApiClientOptions};
    use crate::lifecycle::shutdown::ShutdownSignal;
    use crate::state::{
        ConnectionState, DaemonEvent, EventBus, NewRecording, SqliteStateStore, StateStore,
    };

    /// A re-check short enough for the tests to wait for.
    const RESWEEP: Duration = Duration::from_millis(50);

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
    /// The caller's start is on its own clock, apart from the publish time.
    async fn seed_recording(store: &SqliteStateStore) -> i64 {
        store
            .create_recording(NewRecording {
                robot_index: None,
                dataset_index: None,
                robot_id: Some("robot-1"),
                robot_instance: Some(7),
                dataset_id: Some("ds-1"),
                start: RecordingTimestamp {
                    publish_timestamp_ns: 1_700_000_000_000_000_000,
                    timestamp_us: Some(1_600_000_000_123_457),
                },
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

    /// A recording into dataset row `dataset_index` with robot row
    /// `robot_index`, as the dispatcher makes one.
    async fn seed_recording_with_rows(
        store: &SqliteStateStore,
        robot_index: Option<i64>,
        dataset_index: i64,
    ) -> i64 {
        store
            .create_recording(NewRecording {
                robot_index,
                dataset_index: Some(dataset_index),
                robot_id: Some("robot-1"),
                robot_instance: Some(7),
                dataset_id: None,
                start: RecordingTimestamp {
                    publish_timestamp_ns: 1_700_000_000_000_000_000,
                    timestamp_us: Some(1_600_000_000_123_457),
                },
            })
            .await
            .expect("create recording")
            .recording_index
    }

    async fn select_warehouse(store: &SqliteStateStore) -> i64 {
        store
            .select_dataset(&data_daemon_shared::DatasetSpec {
                org_id: "org-1".into(),
                name: "warehouse".into(),
                shared: false,
                description: None,
                tags: Vec::new(),
            })
            .await
            .expect("select dataset")
    }

    #[tokio::test]
    async fn a_recording_waits_for_its_dataset_and_starts_once_it_is_confirmed() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/start"))
            .and(body_partial_json(
                serde_json::json!({ "dataset_id": "cloud-dataset-1" }),
            ))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "id": "rec-1" })),
            )
            .expect(1)
            .mount(&server)
            .await;
        let (store, _dir) = open_store().await;
        let dataset_index = select_warehouse(&store).await;
        let index = seed_recording_with_rows(&store, None, dataset_index).await;
        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));
        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_start_notifier_every(
            store.clone(),
            bus.clone(),
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
            RESWEEP,
        );

        bus.publish(DaemonEvent::RecordingStarted {
            recording_index: index,
        });
        sleep(Duration::from_millis(200)).await;
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "nothing is started before the dataset is confirmed"
        );

        // Confirmed by the resolver; the re-check picks the recording up.
        store
            .resolve_dataset(dataset_index, "cloud-dataset-1")
            .await
            .unwrap();
        timeout(Duration::from_secs(3), async {
            while store
                .get_recording(index)
                .await
                .unwrap()
                .unwrap()
                .recording_id
                .is_none()
            {
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the confirmed dataset lets the recording start");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn a_recording_waits_for_its_robot_and_starts_with_its_server_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/start"))
            .and(body_partial_json(
                serde_json::json!({ "robot_id": "cloud-robot-1", "dataset_id": "cloud-dataset-1" }),
            ))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "id": "rec-1" })),
            )
            .expect(1)
            .mount(&server)
            .await;
        let (store, _dir) = open_store().await;
        let robot_index = store
            .connect_robot(&data_daemon_shared::RobotSpec {
                org_id: "org-1".into(),
                name: "arm".into(),
                shared: false,
                instance: 7,
                overwrite_urdf: false,
                package_path: None,
            })
            .await
            .unwrap()
            .robot_index;
        let dataset_index = select_warehouse(&store).await;
        store
            .resolve_dataset(dataset_index, "cloud-dataset-1")
            .await
            .unwrap();
        let index = seed_recording_with_rows(&store, Some(robot_index), dataset_index).await;
        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));
        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_start_notifier_every(
            store.clone(),
            bus.clone(),
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
            RESWEEP,
        );
        sleep(Duration::from_millis(200)).await;
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "nothing is started before the robot is confirmed"
        );

        assert!(store
            .resolve_robot(robot_index, "cloud-robot-1", None)
            .await
            .unwrap());
        timeout(Duration::from_secs(3), async {
            while store
                .get_recording(index)
                .await
                .unwrap()
                .unwrap()
                .recording_id
                .is_none()
            {
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the confirmed robot lets the recording start");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn recordings_whose_dataset_is_confirmed_offline_start_once_online() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/start"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "id": "rec-1" })),
            )
            .mount(&server)
            .await;
        let (store, _dir) = open_store().await;
        let dataset_index = select_warehouse(&store).await;
        let mut indexes = Vec::new();
        for _ in 0..3 {
            indexes.push(seed_recording_with_rows(&store, None, dataset_index).await);
        }
        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));
        let bus = EventBus::new();
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_start_notifier_every(
            store.clone(),
            bus.clone(),
            client,
            org_rx(Some("org-1")),
            shutdown_tx.subscribe(),
            RESWEEP,
        );
        sleep(Duration::from_millis(200)).await;

        // The link drops, then the dataset is confirmed.
        bus.publish(DaemonEvent::ConnectionStateChanged(ConnectionState::Down));
        sleep(Duration::from_millis(50)).await;
        store
            .resolve_dataset(dataset_index, "cloud-dataset-1")
            .await
            .unwrap();
        sleep(Duration::from_millis(300)).await;
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "nothing is sent while offline"
        );

        // Back online: every recording is started.
        bus.publish(DaemonEvent::ConnectionStateChanged(ConnectionState::Up));
        timeout(Duration::from_secs(3), async {
            for index in &indexes {
                while store
                    .get_recording(*index)
                    .await
                    .unwrap()
                    .unwrap()
                    .recording_id
                    .is_none()
                {
                    sleep(Duration::from_millis(20)).await;
                }
            }
        })
        .await
        .expect("the recordings start once online");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn a_start_sent_once_is_never_sent_again_by_the_recheck() {
        let server = MockServer::start().await;
        // The start may have reached the server: the reply is an error.
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/start"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let (store, _dir) = open_store().await;
        let index = seed_recording(&store).await;
        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));
        let store = Arc::new(store);
        assert_eq!(
            StartNotifier.periodic_pending(&store).await.unwrap().len(),
            1,
            "never sent: the re-check may send it"
        );

        notify_backend(
            &store,
            &client,
            &EventBus::new(),
            &org_rx(Some("org-1")),
            index,
        )
        .await;

        assert!(!server.received_requests().await.unwrap().is_empty());
        assert!(
            StartNotifier
                .periodic_pending(&store)
                .await
                .unwrap()
                .is_empty(),
            "sent once: only a restart or a missed event sends it again"
        );
    }

    #[tokio::test]
    async fn a_recording_starts_in_the_org_its_robot_was_connected_in() {
        let server = MockServer::start().await;
        start_ok_mock("rec-1").expect(1).mount(&server).await;
        let (store, _dir) = open_store().await;
        let robot_index = store
            .connect_robot(&data_daemon_shared::RobotSpec {
                org_id: "org-1".into(),
                name: "arm".into(),
                shared: false,
                instance: 7,
                overwrite_urdf: false,
                package_path: None,
            })
            .await
            .unwrap()
            .robot_index;
        assert!(store
            .resolve_robot(robot_index, "cloud-robot-1", None)
            .await
            .unwrap());
        let dataset_index = select_warehouse(&store).await;
        store
            .resolve_dataset(dataset_index, "cloud-dataset-1")
            .await
            .unwrap();
        let index = seed_recording_with_rows(&store, Some(robot_index), dataset_index).await;
        let auth = Arc::new(StaticAuthProvider::new("token-1"));
        let client = Arc::new(ApiClient::new(options(server.uri()), auth).expect("client"));

        // org-2 is selected now; the recording still starts in org-1.
        notify_backend(
            &Arc::new(store.clone()),
            &client,
            &EventBus::new(),
            &org_rx(Some("org-2")),
            index,
        )
        .await;

        assert_eq!(
            store
                .get_recording(index)
                .await
                .unwrap()
                .unwrap()
                .recording_id
                .as_deref(),
            Some("rec-1")
        );
    }

    /// A recording cancelled before its start POST goes out is settled without
    /// a POST, so the reaper can reclaim it.
    #[tokio::test]
    async fn settles_a_recording_cancelled_before_its_start_was_posted() {
        let server = MockServer::start().await;
        start_ok_mock("cloud-never-used").mount(&server).await;

        let (store, _dir) = open_store().await;
        let index = seed_recording(&store).await;
        store
            .cancel_recording(index, RecordingTimestamp::observed_at(5_000_000_000))
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

    /// The periodic re-sweep must never re-send a live recording's start.
    #[tokio::test]
    async fn resweep_never_reposts_a_live_recordings_start() {
        let server = MockServer::start().await;
        start_ok_mock("cloud-orphan").mount(&server).await;

        let (store, _dir) = open_store().await;
        let live = seed_recording(&store).await;
        // Its start was sent; the reply may not have come back.
        store.mark_start_sent(live).await.expect("stamp");
        let cancelled = seed_recording(&store).await;
        store
            .cancel_recording(cancelled, RecordingTimestamp::observed_at(5_000_000_000))
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

        let received = server.received_requests().await.expect("recorded requests");
        let body: serde_json::Value = received[0].body_json().expect("json body");
        assert_eq!(
            body,
            serde_json::json!({
                "robot_id": "robot-1",
                "instance": 7,
                "dataset_id": "ds-1",
                "start_time": 1_600_000_000.123_457,
                "start_timestamp_us": 1_600_000_000_123_457_i64,
            }),
            "the start carries the caller's start in microseconds and in seconds"
        );
        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn skips_the_start_post_for_a_cancelled_recording() {
        let server = MockServer::start().await;
        start_ok_mock("cloud-rec-1").mount(&server).await;

        let (store, _dir) = open_store().await;
        let index = seed_recording(&store).await;
        store
            .cancel_recording(
                index,
                RecordingTimestamp::observed_at(1_700_000_000_000_000_001),
            )
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

        bus.publish(DaemonEvent::RecordingStarted {
            recording_index: index,
        });

        sleep(Duration::from_millis(150)).await;
        let received = server.received_requests().await.unwrap_or_default();
        assert!(
            received.is_empty(),
            "no backend POST expected for a cancelled recording"
        );

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
