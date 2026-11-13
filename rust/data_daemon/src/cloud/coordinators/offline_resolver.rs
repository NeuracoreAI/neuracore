//! Getting the backend's ids for the robots and datasets recordings were
//! started offline with.
//!
//! A recording started offline has a `recording_offline` row and is
//! `resolving`; the robot instance and dataset it used have rows in
//! `local_robots` and `local_datasets` (migration `0006`). This task is the
//! only code that talks to the backend about them. For each one a recording
//! still waits on, it makes the call the SDK makes online: the robot upsert of
//! `nc.connect_robot`, and the dataset look-up-then-create of
//! `Dataset.create`. It works per robot and per dataset, so recordings that
//! share one ask about it once.
//!
//! A robot connected offline is registered here and only here, so only when
//! a recording made with it waits; its saved URDF package
//! ([`offline_robots`](crate::cloud::offline_robots)) is uploaded before the
//! robot counts as registered, as `connect_robot` uploads it online.
//!
//! A refusal that retrying cannot fix marks the robot or dataset `failed`.
//! One the user can fix (an archived robot, not being in the org, a rejected
//! API key, a dataset name taken) leaves it `pending` with the reason, so the
//! status the user sees can say what to do, and the backend is asked again
//! after [`BLOCKED_ID_RETRY`]. An outage or a 5xx is retried after
//! [`PASSING_ID_RETRY`].
//!
//! After each pass the recordings are settled: one whose robot and dataset
//! are all `done` becomes `resolved`, with the backend's ids on its
//! `recording_offline` row, and the start notifier is woken
//! ([`DaemonEvent::RecordingOfflineResolved`]); one that used a refused robot
//! or dataset becomes `failed` and never uploads.
//!
//! It runs at startup, when a recording starts, on reconnect and on a timer
//! while online. Settling needs no network, so a recording whose robot and
//! dataset an earlier recording already resolved is ready at once.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chrono::{NaiveDateTime, Utc};
use reqwest::StatusCode;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::api::auth::AuthError;
use crate::api::models::DatasetSummary;
use crate::api::{ApiClient, ApiClientError};
use crate::cloud::offline_robots::{self, SendError};
use crate::cloud::OrgIdRx;
use crate::intervals::{BLOCKED_ID_RETRY, PASSING_ID_RETRY};
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::{
    resolution_detail, ConnectionState, DaemonEvent, EventBus, LocalDatasetRow, LocalRobotRow,
    SqliteStateStore, StateStore,
};

/// Handle to the offline resolver task.
pub struct OfflineResolverHandle {
    join: JoinHandle<()>,
}

impl OfflineResolverHandle {
    /// Wait for the task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "offline resolver task join failed");
        }
    }
}

/// Spawn the offline resolver on the current Tokio runtime.
pub fn spawn_offline_resolver(
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    offline_robots: PathBuf,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> OfflineResolverHandle {
    spawn_with_interval(
        store,
        bus,
        client,
        org_rx,
        offline_robots,
        shutdown_rx,
        crate::intervals::OFFLINE_RESOLVE_SWEEP,
    )
}

fn spawn_with_interval(
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    offline_robots: PathBuf,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
    every: Duration,
) -> OfflineResolverHandle {
    let resolver = Resolver {
        store: Arc::new(store),
        client,
        bus: bus.clone(),
        org_rx,
        offline_robots,
    };
    let mut subscriber = bus.subscribe();
    let join = tokio::spawn(async move {
        // Assume online until the connection monitor says otherwise, as the
        // recording notifiers do: its first `Down` may predate this task.
        let mut connected = true;
        let mut ticker = interval(every);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "offline resolver shutting down");
                    break;
                }
                event = subscriber.recv() => match event {
                    Ok(DaemonEvent::ConnectionStateChanged(state)) => {
                        let reconnected = !connected && state == ConnectionState::Up;
                        connected = state == ConnectionState::Up;
                        if reconnected {
                            resolver.pass(connected).await;
                        }
                    }
                    Ok(DaemonEvent::RecordingStarted { .. }) => resolver.pass(connected).await,
                    Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => break,
                },
                // The first tick fires at once: the startup pass.
                _ = ticker.tick() => resolver.pass(connected).await,
            }
        }
    });
    OfflineResolverHandle { join }
}

/// What a resolver pass works with.
struct Resolver {
    store: Arc<SqliteStateStore>,
    client: Arc<ApiClient>,
    bus: EventBus,
    org_rx: OrgIdRx,
    /// Where robots connected offline save their packages (`.offline_robots`).
    offline_robots: PathBuf,
}

impl Resolver {
    /// Ask the backend about every robot and dataset a recording waits on and
    /// that may be asked about now, when `online`, then settle the
    /// recordings.
    async fn pass(&self, online: bool) {
        // The org is read live, as every other request does; with none
        // selected yet the recordings wait.
        let org_id = self.org_rx.borrow().clone();
        if let (true, Some(org_id)) = (online, org_id) {
            let now = Utc::now().naive_utc();
            match self.store.local_robots_to_resolve(now).await {
                Ok(robots) => {
                    for robot in &robots {
                        self.resolve_robot(&org_id, robot).await;
                    }
                }
                Err(error) => tracing::warn!(%error, "failed to list robots to register"),
            }
            match self.store.local_datasets_to_resolve(now).await {
                Ok(datasets) => {
                    for dataset in &datasets {
                        self.resolve_dataset(&org_id, dataset).await;
                    }
                }
                Err(error) => tracing::warn!(%error, "failed to list datasets to find"),
            }
        }
        self.settle().await;
    }

    /// Register one instance of a robot connected offline as
    /// `nc.connect_robot` does online: the robot upsert, then its saved
    /// package if the backend needs it.
    async fn resolve_robot(&self, org_id: &str, robot: &LocalRobotRow) {
        let LocalRobotRow {
            robot_id,
            instance,
            name,
            shared,
            overwrite_urdf,
            ..
        } = robot;
        let robot_dir = self.offline_robots.join(robot_id);
        let (error, step) = match self
            .client
            .robot_upsert(org_id, name, *instance, *shared)
            .await
        {
            // The package goes up before the robot counts as registered, so
            // a recording never starts on the backend without its robot's
            // description.
            Ok(registered) => match offline_robots::send_package(
                &self.client,
                org_id,
                &robot_dir,
                &registered,
                *shared,
                *overwrite_urdf,
            )
            .await
            {
                Ok(sent) => {
                    self.registered(robot, &registered.robot_id, sent).await;
                    return;
                }
                Err(SendError::Upload(error)) => (error, "upload the robot's description"),
                Err(SendError::Read(error)) => {
                    tracing::warn!(
                        %error,
                        robot = name,
                        "failed to read the robot's saved description; retrying later",
                    );
                    let until = retry_at(&Failure::Retry);
                    if let Err(error) = self
                        .store
                        .hold_local_robot(robot_id, *instance, None, until)
                        .await
                    {
                        tracing::warn!(%error, robot_id, "failed to hold the robot back");
                    }
                    return;
                }
            },
            Err(error) => (error, "register the robot"),
        };
        // Neither call has a 403 of its own for a shared robot: any 403 is the
        // role or org check, which the user can fix. The package upload never
        // gets here refused: `send_package` drops a package the backend
        // rejects.
        let failure = classify(&error, false);
        let stored = match failure {
            Failure::Refused(detail) => {
                tracing::warn!(
                    %error,
                    robot = name,
                    detail,
                    "backend refused to register the robot; its recordings will not upload",
                );
                offline_robots::discard(&robot_dir).await;
                self.store.fail_local_robot(robot_id, detail).await
            }
            Failure::Blocked(detail) => {
                tracing::warn!(%error, robot = name, detail, "{}", blocked_message(detail));
                self.store
                    .hold_local_robot(robot_id, *instance, Some(detail), retry_at(&failure))
                    .await
            }
            Failure::Retry => {
                tracing::warn!(%error, robot = name, "failed to {step}; retrying later");
                self.store
                    .hold_local_robot(robot_id, *instance, None, retry_at(&failure))
                    .await
            }
        };
        if let Err(error) = stored {
            tracing::warn!(%error, robot_id, "failed to record how registering the robot went");
        }
    }

    /// Record that the backend registered a robot instance, and delete its
    /// package once no other instance still waits to register with it.
    async fn registered(
        &self,
        robot: &LocalRobotRow,
        cloud_robot_id: &str,
        sent: offline_robots::Sent,
    ) {
        let LocalRobotRow {
            robot_id, instance, ..
        } = robot;
        if let Err(error) = self
            .store
            .resolve_local_robot(robot_id, *instance, cloud_robot_id)
            .await
        {
            // The upsert is idempotent: the next pass gets the same id.
            tracing::warn!(%error, robot_id, "failed to store the cloud robot id");
            return;
        }
        tracing::info!(
            robot_id,
            instance,
            cloud_robot_id,
            "robot connected offline is registered"
        );
        match self.store.local_robot_has_pending_instances(robot_id).await {
            // Another instance may need the package, if it was connected with
            // `overwrite_urdf`; the startup sweep deletes it if none does.
            Ok(true) | Err(_) => {}
            Ok(false) => {
                offline_robots::remove_package(&self.offline_robots.join(robot_id), sent).await
            }
        }
    }

    /// Find or create a dataset selected offline by its name, as
    /// `Dataset.create` does online.
    async fn resolve_dataset(&self, org_id: &str, dataset: &LocalDatasetRow) {
        let LocalDatasetRow {
            dataset_id,
            name,
            shared,
            description,
            tags,
            ..
        } = dataset;
        let found = find_or_create_dataset(
            &self.client,
            org_id,
            name,
            description.as_deref(),
            tags,
            *shared,
        )
        .await;
        let (failure, reason) = match found {
            Ok(DatasetByName::Found(cloud_dataset_id)) => {
                match self
                    .store
                    .resolve_local_dataset(dataset_id, &cloud_dataset_id)
                    .await
                {
                    Ok(()) => tracing::info!(
                        dataset_id,
                        cloud_dataset_id,
                        "dataset selected offline is found or created"
                    ),
                    // Found by name again on the next pass.
                    Err(error) => {
                        tracing::warn!(%error, dataset_id, "failed to store the cloud dataset id")
                    }
                }
                return;
            }
            Ok(DatasetByName::OtherKind) => (
                Failure::Blocked(resolution_detail::NAME_TAKEN),
                "a dataset of the other kind, shared or private, has this name".to_string(),
            ),
            Ok(DatasetByName::Gone) => (
                Failure::Retry,
                "the backend said the dataset exists but it could not be read back".to_string(),
            ),
            Err(error) => (classify(&error, *shared), error.to_string()),
        };
        let stored = match failure {
            Failure::Refused(detail) => {
                tracing::warn!(
                    reason,
                    dataset = name,
                    detail,
                    "backend refused to create the dataset; its recordings will not upload",
                );
                self.store.fail_local_dataset(dataset_id, detail).await
            }
            Failure::Blocked(detail) => {
                tracing::warn!(
                    reason,
                    dataset = name,
                    detail,
                    "{}",
                    blocked_message(detail)
                );
                self.store
                    .hold_local_dataset(dataset_id, Some(detail), retry_at(&failure))
                    .await
            }
            Failure::Retry => {
                tracing::warn!(
                    reason,
                    dataset = name,
                    "failed to find or create the dataset; retrying later",
                );
                self.store
                    .hold_local_dataset(dataset_id, None, retry_at(&failure))
                    .await
            }
        };
        if let Err(error) = stored {
            tracing::warn!(%error, dataset_id, "failed to record how finding the dataset went");
        }
    }

    /// Move the recordings whose robots and datasets are all answered for,
    /// and wake the start notifier for those that may now start.
    async fn settle(&self) {
        let settled = match self.store.settle_offline_recordings().await {
            Ok(settled) => settled,
            Err(error) => {
                tracing::warn!(%error, "failed to settle offline recordings");
                return;
            }
        };
        for recording_index in settled.failed {
            tracing::warn!(
                recording_index,
                "recording started offline will not upload: the backend refused its robot \
                 or dataset"
            );
        }
        for recording_index in settled.resolved {
            self.bus
                .publish(DaemonEvent::RecordingOfflineResolved { recording_index });
        }
    }
}

/// When the backend may next be asked about a robot or dataset after
/// `failure`.
fn retry_at(failure: &Failure) -> NaiveDateTime {
    let wait = match failure {
        Failure::Blocked(_) => BLOCKED_ID_RETRY,
        Failure::Retry | Failure::Refused(_) => PASSING_ID_RETRY,
    };
    Utc::now().naive_utc() + wait
}

/// What looking a dataset up by name, then creating it, found.
#[derive(Debug, PartialEq, Eq)]
enum DatasetByName {
    /// The backend's id for a dataset of the kind asked for.
    Found(String),
    /// Only a dataset of the other kind (shared or private) has the name, so
    /// one of the kind asked for cannot be created with it.
    OtherKind,
    /// The create said a dataset has the name, but none could be read back.
    Gone,
}

/// Find the dataset called `name`, or create it, as the SDK's
/// `Dataset.create` does: one of the other kind (shared or private) is not
/// reused, and one created by someone else meanwhile is read back.
async fn find_or_create_dataset(
    client: &ApiClient,
    org_id: &str,
    name: &str,
    description: Option<&str>,
    tags: &[String],
    shared: bool,
) -> Result<DatasetByName, ApiClientError> {
    let by_name = |found: Option<DatasetSummary>| match found {
        Some(dataset) if dataset.is_shared == shared => DatasetByName::Found(dataset.id),
        Some(_) => DatasetByName::OtherKind,
        None => DatasetByName::Gone,
    };
    if let found @ DatasetByName::Found(_) = by_name(client.dataset_by_name(org_id, name).await?) {
        return Ok(found);
    }
    if let Some(created) = client
        .dataset_create(org_id, name, description, tags, shared)
        .await?
    {
        return Ok(DatasetByName::Found(created.id));
    }
    Ok(by_name(client.dataset_by_name(org_id, name).await?))
}

/// What a failed find-or-create call means for the recordings waiting on it.
#[derive(Debug, PartialEq, Eq)]
enum Failure {
    /// Final: retrying the request cannot help. Carries the reason.
    Refused(&'static str),
    /// Stuck until the user fixes something; retried meanwhile. Carries the
    /// reason, which tells them what.
    Blocked(&'static str),
    /// Passing (an outage, a 5xx): retried soon.
    Retry,
}

/// Sort a backend error into [`Failure`].
///
/// A 403 is final only when creating a shared dataset, which the user is not
/// allowed to do; otherwise it means the user is not in the org or lacks the
/// role there.
/// An archived robot (409 `ROBOT_ARCHIVED`) can be unarchived. A rejected or
/// missing API key can be fixed by logging in again; any other failure to
/// exchange the key is the server's and passes.
fn classify(error: &ApiClientError, shared_dataset: bool) -> Failure {
    if error.is_robot_archived() {
        return Failure::Blocked(resolution_detail::ROBOT_ARCHIVED);
    }
    match error {
        ApiClientError::Status { status, .. } => match *status {
            StatusCode::UNAUTHORIZED => Failure::Blocked(resolution_detail::CREDENTIALS),
            StatusCode::FORBIDDEN if shared_dataset => {
                Failure::Refused(resolution_detail::SHARED_NOT_ALLOWED)
            }
            StatusCode::FORBIDDEN => Failure::Blocked(resolution_detail::NOT_IN_ORG),
            StatusCode::BAD_REQUEST | StatusCode::CONFLICT | StatusCode::UNPROCESSABLE_ENTITY => {
                Failure::Refused(resolution_detail::REJECTED)
            }
            _ => Failure::Retry,
        },
        ApiClientError::Auth(AuthError::Missing { .. }) => {
            Failure::Blocked(resolution_detail::CREDENTIALS)
        }
        ApiClientError::Auth(AuthError::ExchangeStatus { status, .. })
            if matches!(*status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) =>
        {
            Failure::Blocked(resolution_detail::CREDENTIALS)
        }
        _ => Failure::Retry,
    }
}

/// What the user can do about a [`Failure::Blocked`] reason, for the log.
fn blocked_message(detail: &str) -> &'static str {
    match detail {
        resolution_detail::ROBOT_ARCHIVED => {
            "the robot is archived; unarchive it on the Archived Robots page of the \
             Neuracore dashboard so its recordings can upload"
        }
        resolution_detail::NOT_IN_ORG => {
            "the account is not in the current org; select another org with \
             `neuracore select-org` so the recordings can upload"
        }
        resolution_detail::NAME_TAKEN => {
            "a dataset of the other kind, shared or private, already has this name; \
             rename or delete it on the Neuracore dashboard so the recordings can upload"
        }
        resolution_detail::CREDENTIALS => {
            "the API key was rejected or is missing; run `neuracore login` so the \
             recordings can upload"
        }
        _ => "recordings are waiting; retrying later",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use data_daemon_shared::{LocalDataset, LocalRobot};
    use tempfile::TempDir;
    use tokio::time::{sleep, timeout};
    use wiremock::matchers::{body_json, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::api::auth::StaticAuthProvider;
    use crate::api::ApiClientOptions;
    use crate::state::{LocalIdStatus, NewRecording, OfflineState};

    struct Fixture {
        store: Arc<SqliteStateStore>,
        bus: EventBus,
        offline_robots: PathBuf,
        _dir: TempDir,
    }

    impl Fixture {
        async fn new() -> Self {
            let dir = TempDir::new().unwrap();
            let store = SqliteStateStore::open(&dir.path().join("state.db"))
                .await
                .unwrap();
            Fixture {
                store: Arc::new(store),
                bus: EventBus::new(),
                offline_robots: dir.path().join(".offline_robots"),
                _dir: dir,
            }
        }

        /// A recording started offline: instance 7 of local robot
        /// `local-robot-1` ("arm") into local dataset `local-dataset-1`
        /// ("warehouse").
        async fn recording(&self, dataset_shared: bool) -> i64 {
            self.store
                .create_recording(NewRecording {
                    robot_id: Some("local-robot-1"),
                    robot_instance: Some(7),
                    dataset_id: Some("local-dataset-1"),
                    local_robot: Some(&LocalRobot {
                        name: "arm".into(),
                        shared: false,
                        overwrite_urdf: false,
                    }),
                    local_dataset: Some(&LocalDataset {
                        name: "warehouse".into(),
                        shared: dataset_shared,
                        description: Some("pick and place".into()),
                        tags: vec!["shelf-a".into()],
                    }),
                    ..NewRecording::default()
                })
                .await
                .unwrap()
                .recording_index
        }

        fn resolver(&self, server: &MockServer) -> Resolver {
            Resolver {
                store: Arc::clone(&self.store),
                client: Arc::new(client(server)),
                bus: self.bus.clone(),
                org_rx: org_rx(Some("org-1")),
                offline_robots: self.offline_robots.clone(),
            }
        }

        async fn offline(&self, recording_index: i64) -> Option<OfflineState> {
            self.store
                .get_recording(recording_index)
                .await
                .unwrap()
                .unwrap()
                .offline
        }

        /// What the recording got: its `recording_offline` row.
        async fn settled(&self, recording_index: i64) -> crate::state::RecordingOfflineRow {
            self.store
                .get_recording_offline(recording_index)
                .await
                .unwrap()
                .unwrap()
        }

        async fn robot(&self) -> LocalRobotRow {
            self.store
                .get_local_robot("local-robot-1", 7)
                .await
                .unwrap()
                .unwrap()
        }

        async fn dataset(&self) -> LocalDatasetRow {
            self.store
                .get_local_dataset("local-dataset-1")
                .await
                .unwrap()
                .unwrap()
        }

        /// Let `id`'s wait run out, as time passing would.
        async fn end_waits(&self) {
            for table in ["local_robots", "local_datasets"] {
                sqlx::query(&format!("UPDATE {table} SET next_attempt_at = NULL"))
                    .execute(self.store.write_pool())
                    .await
                    .unwrap();
            }
        }
    }

    fn client(server: &MockServer) -> ApiClient {
        let options = ApiClientOptions {
            base_url: server.uri(),
            timeout: Duration::from_secs(5),
            max_retries: 1,
            max_backoff: Duration::from_secs(1),
        };
        ApiClient::new(options, Arc::new(StaticAuthProvider::new("token-1"))).unwrap()
    }

    fn org_rx(org: Option<&str>) -> OrgIdRx {
        let (org_tx, org_rx) = tokio::sync::watch::channel(org.map(str::to_string));
        Box::leak(Box::new(org_tx));
        org_rx
    }

    fn robot_upsert(status: u16) -> Mock {
        Mock::given(method("POST"))
            .and(path("/org/org-1/robots"))
            .respond_with(
                ResponseTemplate::new(status).set_body_json(serde_json::json!({
                    "robot_id": "cloud-robot-9",
                    "has_urdf": false,
                })),
            )
    }

    fn archived() -> ResponseTemplate {
        ResponseTemplate::new(409).set_body_json(serde_json::json!({
            "detail": {
                "error": "Robot is archived",
                "status": 409,
                "error_type": "ROBOT_ARCHIVED",
            },
        }))
    }

    fn dataset_by_name(response: ResponseTemplate) -> Mock {
        Mock::given(method("GET"))
            .and(path("/org/org-1/datasets/search/by-name"))
            .and(query_param("name", "warehouse"))
            .respond_with(response)
    }

    fn dataset_json(id: &str, is_shared: bool) -> serde_json::Value {
        serde_json::json!({ "id": id, "name": "warehouse", "is_shared": is_shared })
    }

    fn dataset_create(response: ResponseTemplate) -> Mock {
        Mock::given(method("POST"))
            .and(path("/org/org-1/datasets"))
            .respond_with(response)
    }

    fn found_dataset() -> Mock {
        dataset_by_name(
            ResponseTemplate::new(200).set_body_json(dataset_json("cloud-dataset-3", false)),
        )
    }

    #[tokio::test]
    async fn registers_the_robot_and_creates_the_dataset_then_wakes_the_start() {
        let fixture = Fixture::new().await;
        let index = fixture.recording(false).await;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/robots"))
            .and(query_param("is_shared", "false"))
            .and(body_json(serde_json::json!({"name": "arm", "instance": 7})))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "robot_id": "cloud-robot-9",
                "has_urdf": false,
            })))
            .expect(1)
            .mount(&server)
            .await;
        dataset_by_name(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/datasets"))
            .and(body_json(serde_json::json!({
                "name": "warehouse",
                "description": "pick and place",
                "tags": ["shelf-a"],
                "is_shared": false,
            })))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(dataset_json("cloud-dataset-4", false)),
            )
            .expect(1)
            .mount(&server)
            .await;
        let mut events = fixture.bus.subscribe();

        fixture.resolver(&server).pass(true).await;

        assert_eq!(fixture.offline(index).await, Some(OfflineState::Resolved));
        let offline = fixture
            .store
            .get_recording_offline(index)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(offline.cloud_robot_id.as_deref(), Some("cloud-robot-9"));
        assert_eq!(offline.cloud_dataset_id.as_deref(), Some("cloud-dataset-4"));
        let row = fixture.store.get_recording(index).await.unwrap().unwrap();
        assert_eq!(row.robot_id.as_deref(), Some("local-robot-1"));
        assert_eq!(row.dataset_id.as_deref(), Some("local-dataset-1"));
        assert!(matches!(
            events.try_recv(),
            Ok(DaemonEvent::RecordingOfflineResolved { recording_index }) if recording_index == index
        ));
    }

    #[tokio::test]
    async fn recordings_of_one_robot_and_dataset_ask_the_backend_once() {
        let fixture = Fixture::new().await;
        let first = fixture.recording(false).await;
        let second = fixture.recording(false).await;
        let server = MockServer::start().await;
        robot_upsert(200).expect(1).mount(&server).await;
        found_dataset().expect(1).mount(&server).await;

        fixture.resolver(&server).pass(true).await;

        assert_eq!(fixture.offline(first).await, Some(OfflineState::Resolved));
        assert_eq!(fixture.offline(second).await, Some(OfflineState::Resolved));
    }

    #[tokio::test]
    async fn a_refusal_fixed_since_lets_a_later_recording_through() {
        let fixture = Fixture::new().await;
        let refused = fixture.recording(true).await;
        let server = MockServer::start().await;
        robot_upsert(200).mount(&server).await;
        dataset_by_name(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        dataset_create(ResponseTemplate::new(403))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        fixture.resolver(&server).pass(true).await;
        assert_eq!(fixture.offline(refused).await, Some(OfflineState::Failed));

        // The user is now allowed to create shared datasets.
        dataset_create(
            ResponseTemplate::new(200).set_body_json(dataset_json("cloud-dataset-4", true)),
        )
        .mount(&server)
        .await;
        let later = fixture.recording(true).await;
        fixture.resolver(&server).pass(true).await;

        assert_eq!(fixture.offline(later).await, Some(OfflineState::Resolved));
        assert_eq!(fixture.offline(refused).await, Some(OfflineState::Failed));
    }

    #[tokio::test]
    async fn nothing_is_asked_while_offline_or_with_no_org() {
        let fixture = Fixture::new().await;
        let index = fixture.recording(false).await;
        let server = MockServer::start().await;

        fixture.resolver(&server).pass(false).await;
        let mut no_org = fixture.resolver(&server);
        no_org.org_rx = org_rx(None);
        no_org.pass(true).await;

        assert!(server.received_requests().await.unwrap().is_empty());
        assert_eq!(fixture.offline(index).await, Some(OfflineState::Resolving));
    }

    #[tokio::test]
    async fn an_existing_dataset_of_the_same_kind_is_reused() {
        let fixture = Fixture::new().await;
        let index = fixture.recording(false).await;
        let server = MockServer::start().await;
        robot_upsert(200).mount(&server).await;
        found_dataset().mount(&server).await;
        dataset_create(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;

        fixture.resolver(&server).pass(true).await;

        assert_eq!(fixture.offline(index).await, Some(OfflineState::Resolved));
        assert_eq!(
            fixture.settled(index).await.cloud_dataset_id.as_deref(),
            Some("cloud-dataset-3")
        );
    }

    #[tokio::test]
    async fn a_dataset_of_the_other_kind_waits_with_its_reason() {
        let fixture = Fixture::new().await;
        let index = fixture.recording(true).await;
        let server = MockServer::start().await;
        robot_upsert(200).mount(&server).await;
        // Only a private dataset has the name, and creating a shared one
        // with it is refused as taken.
        dataset_by_name(
            ResponseTemplate::new(200).set_body_json(dataset_json("cloud-private", false)),
        )
        .mount(&server)
        .await;
        dataset_create(ResponseTemplate::new(409))
            .mount(&server)
            .await;

        fixture.resolver(&server).pass(true).await;

        let dataset = fixture.dataset().await;
        assert_eq!(dataset.status, LocalIdStatus::Pending);
        assert_eq!(
            dataset.detail.as_deref(),
            Some(resolution_detail::NAME_TAKEN)
        );
        assert!(dataset.next_attempt_at.is_some());
        assert_eq!(fixture.offline(index).await, Some(OfflineState::Resolving));
    }

    #[tokio::test]
    async fn a_dataset_created_by_someone_else_meanwhile_is_read_back() {
        let fixture = Fixture::new().await;
        let index = fixture.recording(false).await;
        let server = MockServer::start().await;
        robot_upsert(200).mount(&server).await;
        dataset_by_name(ResponseTemplate::new(404))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        dataset_create(ResponseTemplate::new(409))
            .mount(&server)
            .await;
        found_dataset().mount(&server).await;

        fixture.resolver(&server).pass(true).await;

        assert_eq!(
            fixture.settled(index).await.cloud_dataset_id.as_deref(),
            Some("cloud-dataset-3")
        );
    }

    #[tokio::test]
    async fn a_shared_dataset_the_backend_refuses_fails_its_recordings() {
        let fixture = Fixture::new().await;
        let index = fixture.recording(true).await;
        let server = MockServer::start().await;
        robot_upsert(200).mount(&server).await;
        dataset_by_name(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        dataset_create(ResponseTemplate::new(403))
            .mount(&server)
            .await;

        fixture.resolver(&server).pass(true).await;

        assert_eq!(fixture.offline(index).await, Some(OfflineState::Failed));
        assert_eq!(
            fixture.settled(index).await.failure.as_deref(),
            Some(resolution_detail::SHARED_NOT_ALLOWED)
        );
    }

    #[tokio::test]
    async fn a_refused_robot_fails_its_recordings() {
        let fixture = Fixture::new().await;
        let index = fixture.recording(false).await;
        let server = MockServer::start().await;
        robot_upsert(422).mount(&server).await;
        found_dataset().mount(&server).await;

        fixture.resolver(&server).pass(true).await;

        assert_eq!(fixture.offline(index).await, Some(OfflineState::Failed));
        assert_eq!(
            fixture.settled(index).await.failure.as_deref(),
            Some(resolution_detail::REJECTED)
        );
    }

    #[tokio::test]
    async fn an_archived_robot_waits_with_its_reason_and_goes_through_once_unarchived() {
        let fixture = Fixture::new().await;
        let index = fixture.recording(false).await;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/robots"))
            .respond_with(archived())
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
        found_dataset().mount(&server).await;

        fixture.resolver(&server).pass(true).await;
        // Within the wait the backend is not asked again.
        fixture.resolver(&server).pass(true).await;

        let robot = fixture.robot().await;
        assert_eq!(robot.status, LocalIdStatus::Pending);
        assert_eq!(
            robot.detail.as_deref(),
            Some(resolution_detail::ROBOT_ARCHIVED)
        );
        let wait = robot.next_attempt_at.unwrap() - Utc::now().naive_utc();
        assert!(wait > chrono::Duration::minutes(4), "{wait}");
        assert_eq!(fixture.offline(index).await, Some(OfflineState::Resolving));

        // Unarchived, and the wait is over.
        robot_upsert(200).mount(&server).await;
        fixture.end_waits().await;
        fixture.resolver(&server).pass(true).await;

        assert_eq!(fixture.offline(index).await, Some(OfflineState::Resolved));
    }

    #[tokio::test]
    async fn an_outage_retries_soon_and_keeps_an_earlier_reason() {
        let fixture = Fixture::new().await;
        fixture.recording(false).await;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/robots"))
            .respond_with(archived())
            .up_to_n_times(1)
            .mount(&server)
            .await;
        robot_upsert(503).mount(&server).await;
        found_dataset().mount(&server).await;

        fixture.resolver(&server).pass(true).await;
        fixture.end_waits().await;
        fixture.resolver(&server).pass(true).await;

        let robot = fixture.robot().await;
        assert_eq!(robot.status, LocalIdStatus::Pending);
        assert_eq!(
            robot.detail.as_deref(),
            Some(resolution_detail::ROBOT_ARCHIVED)
        );
        let wait = robot.next_attempt_at.unwrap() - Utc::now().naive_utc();
        assert!(wait <= chrono::Duration::seconds(20), "{wait}");
    }

    #[tokio::test]
    async fn a_cancelled_recording_registers_nothing() {
        let fixture = Fixture::new().await;
        let index = fixture.recording(false).await;
        fixture
            .store
            .cancel_recording(
                index,
                crate::state::RecordingTimestamp::observed_at(5_000_000_000),
            )
            .await
            .unwrap();
        let server = MockServer::start().await;

        fixture.resolver(&server).pass(true).await;

        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn the_task_resolves_at_startup() {
        let fixture = Fixture::new().await;
        let index = fixture.recording(false).await;
        let server = MockServer::start().await;
        robot_upsert(200).mount(&server).await;
        found_dataset().mount(&server).await;
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);

        let handle = spawn_with_interval(
            (*fixture.store).clone(),
            fixture.bus.clone(),
            Arc::new(client(&server)),
            org_rx(Some("org-1")),
            fixture.offline_robots.clone(),
            shutdown_tx.subscribe(),
            Duration::from_secs(3600),
        );
        timeout(Duration::from_secs(3), async {
            while fixture.offline(index).await != Some(OfflineState::Resolved) {
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the startup pass resolves the recording");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn the_task_resolves_on_reconnect() {
        let fixture = Fixture::new().await;
        let server = MockServer::start().await;
        robot_upsert(200).mount(&server).await;
        found_dataset().mount(&server).await;
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_with_interval(
            (*fixture.store).clone(),
            fixture.bus.clone(),
            Arc::new(client(&server)),
            org_rx(Some("org-1")),
            fixture.offline_robots.clone(),
            shutdown_tx.subscribe(),
            Duration::from_secs(3600),
        );
        sleep(Duration::from_millis(100)).await;
        fixture
            .bus
            .publish(DaemonEvent::ConnectionStateChanged(ConnectionState::Down));
        sleep(Duration::from_millis(100)).await;
        let index = fixture.recording(false).await;
        fixture.bus.publish(DaemonEvent::RecordingStarted {
            recording_index: index,
        });
        sleep(Duration::from_millis(100)).await;
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "nothing is asked while offline"
        );

        fixture
            .bus
            .publish(DaemonEvent::ConnectionStateChanged(ConnectionState::Up));
        timeout(Duration::from_secs(3), async {
            while fixture.offline(index).await != Some(OfflineState::Resolved) {
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("reconnecting resolves the recording");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    impl Fixture {
        /// Save `local-robot-1`'s package, as an offline `connect_robot`
        /// does. Returns the robot's directory.
        fn save_package(&self) -> PathBuf {
            let dir = self.offline_robots.join("local-robot-1");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("package.zip"), b"zip bytes").unwrap();
            dir
        }
    }

    fn package_upload(status: u16) -> Mock {
        Mock::given(method("PUT"))
            .and(path("/org/org-1/robots/cloud-robot-9/package"))
            .respond_with(ResponseTemplate::new(status))
    }

    #[tokio::test]
    async fn the_saved_package_is_uploaded_before_the_robot_counts_as_registered() {
        let fixture = Fixture::new().await;
        let index = fixture.recording(false).await;
        let saved = fixture.save_package();
        let server = MockServer::start().await;
        robot_upsert(200).mount(&server).await;
        package_upload(200).expect(1).mount(&server).await;
        found_dataset().mount(&server).await;

        fixture.resolver(&server).pass(true).await;

        assert_eq!(fixture.offline(index).await, Some(OfflineState::Resolved));
        assert!(!saved.exists(), "the sent package is deleted");
    }

    #[tokio::test]
    async fn a_failed_package_upload_holds_the_robot() {
        let fixture = Fixture::new().await;
        let index = fixture.recording(false).await;
        let saved = fixture.save_package();
        let server = MockServer::start().await;
        robot_upsert(200).mount(&server).await;
        package_upload(503).mount(&server).await;
        found_dataset().mount(&server).await;

        fixture.resolver(&server).pass(true).await;

        let robot = fixture.robot().await;
        assert_eq!(robot.status, LocalIdStatus::Pending);
        assert!(robot.next_attempt_at.is_some());
        assert_eq!(fixture.offline(index).await, Some(OfflineState::Resolving));
        assert!(saved.join("package.zip").exists());
    }

    #[tokio::test]
    async fn the_package_stays_while_another_instance_waits() {
        let fixture = Fixture::new().await;
        let index = fixture.recording(false).await;
        // Instance 8 waits too, but the backend is only asked about 7 now.
        fixture
            .store
            .create_recording(NewRecording {
                robot_id: Some("local-robot-1"),
                robot_instance: Some(8),
                dataset_id: Some("cloud-dataset-1"),
                local_robot: Some(&LocalRobot {
                    name: "arm".into(),
                    shared: false,
                    overwrite_urdf: false,
                }),
                ..NewRecording::default()
            })
            .await
            .unwrap();
        fixture
            .store
            .hold_local_robot(
                "local-robot-1",
                8,
                None,
                Utc::now().naive_utc() + chrono::Duration::minutes(5),
            )
            .await
            .unwrap();
        let saved = fixture.save_package();
        let server = MockServer::start().await;
        robot_upsert(200).mount(&server).await;
        package_upload(200).mount(&server).await;
        found_dataset().mount(&server).await;

        fixture.resolver(&server).pass(true).await;

        assert_eq!(fixture.offline(index).await, Some(OfflineState::Resolved));
        assert!(saved.join("package.zip").exists());
    }

    #[tokio::test]
    async fn a_refused_robot_discards_its_package() {
        let fixture = Fixture::new().await;
        fixture.recording(false).await;
        let saved = fixture.save_package();
        let server = MockServer::start().await;
        robot_upsert(422).mount(&server).await;
        found_dataset().mount(&server).await;

        fixture.resolver(&server).pass(true).await;

        assert!(!saved.exists());
    }

    #[test]
    fn errors_sort_into_refused_blocked_and_passing() {
        let status = |code: u16| ApiClientError::Status {
            status: StatusCode::from_u16(code).unwrap(),
            body: String::new(),
        };
        assert_eq!(
            classify(&status(401), false),
            Failure::Blocked(resolution_detail::CREDENTIALS)
        );
        assert_eq!(
            classify(&status(403), false),
            Failure::Blocked(resolution_detail::NOT_IN_ORG)
        );
        assert_eq!(
            classify(&status(403), true),
            Failure::Refused(resolution_detail::SHARED_NOT_ALLOWED)
        );
        for code in [400, 409, 422] {
            assert_eq!(
                classify(&status(code), false),
                Failure::Refused(resolution_detail::REJECTED)
            );
        }
        for code in [404, 429, 500, 503] {
            assert_eq!(classify(&status(code), false), Failure::Retry);
        }
    }
}
