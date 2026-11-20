//! Getting the server's ids for the robot instances processes connect.
//!
//! Every robot instance a process connects (`nc.connect_robot`) has a row in
//! `robots`, made by the daemon when the process asks, online or offline. This
//! task has the server confirm each row that is `pending`: it makes the robot
//! upsert the SDK makes online, which registers the instance, in the
//! organization stored on the row, and stores the server's id. It knows
//! nothing about recordings; a recording only waits for its robot row to be
//! confirmed before it is registered.
//!
//! An archived robot, an account that cannot record in the organization, or a
//! rejected API key keeps the row `pending` with the reason, which is logged
//! with what to do, and the server is asked again after [`BLOCKED_ID_RETRY`].
//! A request the server rejects marks the row `failed`. An outage or a 5xx is
//! retried after [`PASSING_ID_RETRY`].
//!
//! When it runs is shared with the dataset resolver ([`super::resolution`]):
//! at startup, when a robot is connected, on reconnect, after it missed
//! events, and while online when the earliest wait runs out.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{NaiveDateTime, Utc};
use tokio::sync::broadcast;

use super::resolution::{
    blocked_message, classify, retry_at, spawn_resolver, Failure, ResolverHandle, RowResolver,
};
use crate::api::ApiClient;
use crate::cloud::robot_packages::{self, SendError};
use crate::intervals::{BLOCKED_ID_RETRY, PASSING_ID_RETRY};
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::{
    DaemonEvent, EventBus, RobotRow, SqliteStateStore, StateStore, StateStoreError,
};
use crate::storage::paths::is_in_robot_packages;

/// Spawn the robot resolver on the current Tokio runtime.
pub fn spawn_robot_resolver(
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    packages_root: PathBuf,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> ResolverHandle {
    let resolver = RobotResolver {
        store: Arc::new(store),
        client,
        bus: bus.clone(),
        packages_root,
    };
    spawn_resolver(resolver, &bus, shutdown_rx)
}

/// What the robot resolver works with.
struct RobotResolver {
    store: Arc<SqliteStateStore>,
    client: Arc<ApiClient>,
    bus: EventBus,
    /// Where the SDK saves robot packages
    /// ([`crate::storage::paths::robot_packages_root`]).
    packages_root: PathBuf,
}

#[async_trait]
impl RowResolver for RobotResolver {
    fn label(&self) -> &'static str {
        "robot-resolver"
    }

    fn wakes_on(&self, event: &DaemonEvent) -> bool {
        matches!(event, DaemonEvent::RobotConnected { .. })
    }

    /// Delete old packages no robot row names.
    async fn startup(&self) {
        match self.store.robot_package_paths().await {
            Ok(paths) => {
                let referenced: HashSet<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
                robot_packages::remove_unreferenced(&self.packages_root, &referenced).await;
            }
            Err(error) => tracing::warn!(%error, "failed to list the saved robot packages"),
        }
    }

    async fn resolve_due(&self) {
        match self.store.robots_to_resolve(Utc::now().naive_utc()).await {
            Ok(robots) => {
                for robot in &robots {
                    self.resolve(robot).await;
                }
            }
            Err(error) => tracing::warn!(%error, "failed to list robots to register"),
        }
    }

    async fn next_attempt_at(
        &self,
        now: NaiveDateTime,
    ) -> Result<Option<NaiveDateTime>, StateStoreError> {
        self.store.next_robot_attempt_at(now).await
    }
}

impl RobotResolver {
    /// Register one robot row's instance, as `nc.connect_robot` does online,
    /// in the organization stored on the row, then send its saved package if
    /// the server needs it.
    async fn resolve(&self, robot: &RobotRow) {
        let RobotRow {
            robot_index,
            org_id,
            name,
            shared,
            instance,
            overwrite_urdf,
            package_path,
            ..
        } = robot;
        // Only rows made for earlier recordings have no organization or name,
        // and those are confirmed already.
        let (Some(org_id), Some(name)) = (org_id, name) else {
            self.hold(
                *robot_index,
                "",
                Failure::Retry,
                "the row has no organization or name to ask the server with",
            )
            .await;
            return;
        };
        let error = match self
            .client
            .robot_upsert(org_id, name, *instance, *shared)
            .await
        {
            Ok(registered) => {
                // The daemon only reads or deletes a package in its own
                // folder; a path from anywhere else is left alone.
                let package = package_path.as_deref().filter(|package| {
                    is_in_robot_packages(&self.packages_root, Path::new(package))
                });
                if package.is_none() && package_path.is_some() {
                    tracing::warn!(
                        robot_index,
                        "the robot's package is outside the package folder; it is not sent"
                    );
                }
                if let Some(package) = package {
                    let sent = robot_packages::send_package(
                        &self.client,
                        org_id,
                        Path::new(package),
                        &registered,
                        *shared,
                        *overwrite_urdf,
                    )
                    .await;
                    if let Err(error) = sent {
                        // The upload has no 403 of its own either.
                        let failure = match &error {
                            SendError::Read(_) => Failure::Retry,
                            SendError::Upload(error) => classify(error),
                        };
                        self.hold(*robot_index, name, failure, &error.to_string())
                            .await;
                        return;
                    }
                }
                match self
                    .store
                    .resolve_robot(*robot_index, &registered.robot_id, package_path.as_deref())
                    .await
                {
                    Ok(true) => {
                        if let Some(package) = package {
                            robot_packages::remove_package(Path::new(package)).await;
                        }
                        tracing::info!(
                            robot_index,
                            robot = name,
                            instance,
                            cloud_robot_id = registered.robot_id,
                            "robot registered"
                        );
                        self.bus.publish(DaemonEvent::RobotResolved {
                            robot_index: *robot_index,
                        });
                    }
                    // Connected again with a new package meanwhile: the next
                    // pass sends it.
                    Ok(false) => {}
                    // The upsert is idempotent: the next pass gets the same id.
                    Err(error) => {
                        tracing::warn!(%error, robot_index, "failed to store the cloud robot id")
                    }
                }
                return;
            }
            Err(error) => error,
        };
        // The upsert has no 403 of its own for a shared robot: any 403 is the
        // role or organization check, which the user can fix.
        self.hold(*robot_index, name, classify(&error), &error.to_string())
            .await;
    }

    /// Log a failure to register a robot and store what it means.
    async fn hold(&self, robot_index: i64, name: &str, failure: Failure, reason: &str) {
        let stored = match failure {
            Failure::Refused(code) => {
                tracing::warn!(
                    reason,
                    robot = name,
                    detail = code.as_str(),
                    "the server refused the robot; recordings made with it will not upload",
                );
                self.store.fail_robot(robot_index, code).await
            }
            Failure::Blocked(code) => {
                tracing::warn!(
                    reason,
                    robot = name,
                    detail = code.as_str(),
                    "{}",
                    blocked_message(code)
                );
                self.store
                    .hold_robot(robot_index, Some(code), retry_at(BLOCKED_ID_RETRY))
                    .await
            }
            Failure::Retry => {
                tracing::warn!(
                    reason,
                    robot = name,
                    "failed to register the robot; retrying later"
                );
                self.store
                    .hold_robot(robot_index, None, retry_at(PASSING_ID_RETRY))
                    .await
            }
        };
        if let Err(error) = stored {
            tracing::warn!(%error, robot_index, "failed to record how registering the robot went");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    use data_daemon_shared::RobotSpec;
    use tempfile::TempDir;
    use tokio::time::{sleep, timeout};
    use wiremock::matchers::{body_json, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::super::resolution::pass;
    use crate::api::auth::StaticAuthProvider;
    use crate::api::ApiClientOptions;
    use crate::state::{ResolutionReason, ResolutionStatus};

    struct Fixture {
        store: Arc<SqliteStateStore>,
        bus: EventBus,
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
                _dir: dir,
            }
        }

        /// Connect instance `instance` of robot "arm" in `org`.
        async fn connect(&self, org: &str, instance: i64) -> i64 {
            self.store
                .connect_robot(&RobotSpec {
                    org_id: org.into(),
                    name: "arm".into(),
                    shared: false,
                    instance,
                    overwrite_urdf: false,
                    package_path: None,
                })
                .await
                .unwrap()
                .robot_index
        }

        fn resolver(&self, server: &MockServer) -> RobotResolver {
            RobotResolver {
                store: Arc::clone(&self.store),
                client: Arc::new(client(server)),
                bus: self.bus.clone(),
                packages_root: self.packages_root(),
            }
        }

        fn packages_root(&self) -> PathBuf {
            self._dir.path().join("packages")
        }

        /// Connect instance 0 of "arm" in org-1 with a saved package.
        async fn connect_with_package(&self) -> (i64, PathBuf) {
            std::fs::create_dir_all(self.packages_root()).unwrap();
            let package = self.packages_root().join("arm.zip");
            std::fs::write(&package, b"zip bytes").unwrap();
            let robot_index = self
                .store
                .connect_robot(&RobotSpec {
                    org_id: "org-1".into(),
                    name: "arm".into(),
                    shared: false,
                    instance: 0,
                    overwrite_urdf: false,
                    package_path: Some(package.display().to_string()),
                })
                .await
                .unwrap()
                .robot_index;
            (robot_index, package)
        }

        async fn robot(&self, robot_index: i64) -> RobotRow {
            self.store.get_robot(robot_index).await.unwrap().unwrap()
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

    fn robot_upsert(org: &str, status: u16) -> Mock {
        Mock::given(method("POST"))
            .and(path(format!("/org/{org}/robots")))
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

    #[tokio::test]
    async fn registers_the_instance_and_announces_it() {
        let fixture = Fixture::new().await;
        let robot_index = fixture.connect("org-1", 7).await;
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
        let mut events = fixture.bus.subscribe();

        assert_eq!(pass(&fixture.resolver(&server), true).await, None);

        let robot = fixture.robot(robot_index).await;
        assert_eq!(robot.status, ResolutionStatus::Done);
        assert_eq!(robot.cloud_robot_id.as_deref(), Some("cloud-robot-9"));
        assert!(matches!(
            events.try_recv(),
            Ok(DaemonEvent::RobotResolved { robot_index: index }) if index == robot_index
        ));
    }

    #[tokio::test]
    async fn each_instance_is_registered_in_the_org_stored_on_its_row() {
        let fixture = Fixture::new().await;
        let first = fixture.connect("org-2", 0).await;
        let second = fixture.connect("org-2", 1).await;
        let server = MockServer::start().await;
        robot_upsert("org-2", 200).expect(2).mount(&server).await;

        pass(&fixture.resolver(&server), true).await;

        for robot_index in [first, second] {
            assert_eq!(
                fixture.robot(robot_index).await.status,
                ResolutionStatus::Done
            );
        }
    }

    #[tokio::test]
    async fn an_archived_robot_holds_its_row_until_it_is_unarchived() {
        let fixture = Fixture::new().await;
        let robot_index = fixture.connect("org-1", 0).await;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/robots"))
            .respond_with(archived())
            .up_to_n_times(1)
            .mount(&server)
            .await;
        robot_upsert("org-1", 200).mount(&server).await;

        pass(&fixture.resolver(&server), true).await;
        let robot = fixture.robot(robot_index).await;
        assert_eq!(robot.status, ResolutionStatus::Pending);
        assert_eq!(robot.reason, Some(ResolutionReason::RobotArchived));

        // Unarchived on the dashboard; the next attempt goes through.
        sqlx::query("UPDATE robots SET next_attempt_at = NULL")
            .execute(fixture.store.write_pool())
            .await
            .unwrap();
        pass(&fixture.resolver(&server), true).await;

        let robot = fixture.robot(robot_index).await;
        assert_eq!(robot.status, ResolutionStatus::Done);
        assert_eq!(robot.reason, None);
    }

    #[tokio::test]
    async fn a_rejected_robot_fails_its_row() {
        let fixture = Fixture::new().await;
        let robot_index = fixture.connect("org-1", 0).await;
        let server = MockServer::start().await;
        robot_upsert("org-1", 422).mount(&server).await;

        pass(&fixture.resolver(&server), true).await;

        let robot = fixture.robot(robot_index).await;
        assert_eq!(robot.status, ResolutionStatus::Failed);
        assert_eq!(robot.reason, Some(ResolutionReason::Rejected));
    }

    fn package_upload(status: u16) -> Mock {
        Mock::given(method("PUT"))
            .and(path("/org/org-1/robots/cloud-robot-9/package"))
            .respond_with(ResponseTemplate::new(status))
    }

    #[tokio::test]
    async fn the_saved_package_is_sent_before_the_robot_counts_as_registered() {
        let fixture = Fixture::new().await;
        let (robot_index, package) = fixture.connect_with_package().await;
        let server = MockServer::start().await;
        robot_upsert("org-1", 200).mount(&server).await;
        package_upload(200).expect(1).mount(&server).await;

        pass(&fixture.resolver(&server), true).await;

        let robot = fixture.robot(robot_index).await;
        assert_eq!(robot.status, ResolutionStatus::Done);
        assert_eq!(robot.package_path, None);
        assert!(!package.exists(), "the sent package is deleted");
    }

    #[tokio::test]
    async fn a_package_outside_the_package_folder_is_neither_sent_nor_deleted() {
        let fixture = Fixture::new().await;
        let outside = fixture._dir.path().join("arm.urdf");
        std::fs::write(&outside, b"urdf").unwrap();
        let robot_index = fixture
            .store
            .connect_robot(&RobotSpec {
                org_id: "org-1".into(),
                name: "arm".into(),
                shared: false,
                instance: 0,
                overwrite_urdf: false,
                package_path: Some(outside.display().to_string()),
            })
            .await
            .unwrap()
            .robot_index;
        let server = MockServer::start().await;
        robot_upsert("org-1", 200).mount(&server).await;
        package_upload(200).expect(0).mount(&server).await;

        pass(&fixture.resolver(&server), true).await;

        assert_eq!(
            fixture.robot(robot_index).await.status,
            ResolutionStatus::Done
        );
        assert!(outside.exists(), "a file outside the package folder stays");
    }

    #[tokio::test]
    async fn a_failed_package_upload_holds_the_robot_and_keeps_the_package() {
        let fixture = Fixture::new().await;
        let (robot_index, package) = fixture.connect_with_package().await;
        let server = MockServer::start().await;
        robot_upsert("org-1", 200).mount(&server).await;
        package_upload(503).mount(&server).await;

        pass(&fixture.resolver(&server), true).await;

        let robot = fixture.robot(robot_index).await;
        assert_eq!(robot.status, ResolutionStatus::Pending);
        assert!(robot.next_attempt_at.is_some());
        assert!(package.exists());
    }

    #[tokio::test]
    async fn a_rejected_package_does_not_hold_the_robot_back() {
        let fixture = Fixture::new().await;
        let (robot_index, package) = fixture.connect_with_package().await;
        let server = MockServer::start().await;
        robot_upsert("org-1", 200).mount(&server).await;
        package_upload(422).mount(&server).await;

        pass(&fixture.resolver(&server), true).await;

        assert_eq!(
            fixture.robot(robot_index).await.status,
            ResolutionStatus::Done
        );
        assert!(!package.exists());
    }

    #[tokio::test]
    async fn the_task_registers_a_connected_robot() {
        let fixture = Fixture::new().await;
        let server = MockServer::start().await;
        robot_upsert("org-1", 200).mount(&server).await;
        let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
        let handle = spawn_robot_resolver(
            (*fixture.store).clone(),
            fixture.bus.clone(),
            Arc::new(client(&server)),
            fixture.packages_root(),
            shutdown_tx.subscribe(),
        );
        sleep(Duration::from_millis(100)).await;

        let robot_index = fixture.connect("org-1", 0).await;
        fixture
            .bus
            .publish(DaemonEvent::RobotConnected { robot_index });
        timeout(Duration::from_secs(3), async {
            while fixture.robot(robot_index).await.status != ResolutionStatus::Done {
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the connected robot is registered");

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }
}
