//! Getting the server's ids for the datasets processes select.
//!
//! Every dataset a process selects (`nc.create_dataset`, and `nc.get_dataset`
//! by name after an offline login) has a row in `datasets`, made by the
//! daemon when the process asks, online or offline. This task has the server
//! confirm each row that is `pending`: it makes `Dataset.create`'s
//! look-up-then-create, in the organization stored on the row, and stores the
//! server's id. It knows nothing about recordings; a recording only
//! waits for its dataset row to be confirmed before it is registered.
//!
//! A refusal that retrying cannot fix marks the row `failed`. One the user can
//! fix (an account that cannot record in the organization, a rejected API
//! key, a dataset name taken) keeps it `pending` with the reason, which is
//! logged with what to do, and the server is asked again after
//! [`BLOCKED_ID_RETRY`]. An outage or a 5xx is retried after
//! [`PASSING_ID_RETRY`].
//!
//! When it runs is shared with the robot resolver
//! ([`super::resolution`]): at startup, when a dataset is selected, on
//! reconnect, after it missed events, and while online when the earliest
//! wait runs out.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{NaiveDateTime, Utc};
use tokio::sync::broadcast;

use super::resolution::{
    blocked_message, classify, retry_at, spawn_resolver, Failure, ResolverHandle, RowResolver,
};
use crate::api::models::DatasetSummary;
use crate::api::{ApiClient, ApiClientError};
use crate::intervals::{BLOCKED_ID_RETRY, PASSING_ID_RETRY};
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::{
    DaemonEvent, DatasetRow, EventBus, ResolutionReason, SqliteStateStore, StateStore,
    StateStoreError,
};

/// Spawn the dataset resolver on the current Tokio runtime.
pub fn spawn_dataset_resolver(
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> ResolverHandle {
    let resolver = DatasetResolver {
        store: Arc::new(store),
        client,
    };
    spawn_resolver(resolver, &bus, shutdown_rx)
}

/// What the dataset resolver works with.
struct DatasetResolver {
    store: Arc<SqliteStateStore>,
    client: Arc<ApiClient>,
}

#[async_trait]
impl RowResolver for DatasetResolver {
    fn label(&self) -> &'static str {
        "dataset-resolver"
    }

    fn wakes_on(&self, event: &DaemonEvent) -> bool {
        matches!(event, DaemonEvent::DatasetSelected { .. })
    }

    async fn resolve_due(&self) {
        match self.store.datasets_to_resolve(Utc::now().naive_utc()).await {
            Ok(datasets) => {
                for dataset in &datasets {
                    self.resolve(dataset).await;
                }
            }
            Err(error) => tracing::warn!(%error, "failed to list datasets to find"),
        }
    }

    async fn next_attempt_at(
        &self,
        now: NaiveDateTime,
    ) -> Result<Option<NaiveDateTime>, StateStoreError> {
        self.store.next_dataset_attempt_at(now).await
    }
}

impl DatasetResolver {
    /// Find or create one dataset row's dataset, as `Dataset.create` does
    /// online, in the organization stored on the row.
    async fn resolve(&self, dataset: &DatasetRow) {
        let DatasetRow {
            dataset_index,
            org_id,
            name,
            shared,
            description,
            tags,
            ..
        } = dataset;
        // Only rows made for earlier recordings or by the web app have no
        // organization or name, and those are confirmed already.
        let (Some(org_id), Some(name)) = (org_id, name) else {
            let reason = "the row has no organization or name to ask the server with";
            self.store_outcome(*dataset_index, "", Failure::Retry, reason.into())
                .await;
            return;
        };
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
                    .resolve_dataset(*dataset_index, &cloud_dataset_id)
                    .await
                {
                    Ok(()) => {
                        tracing::info!(
                            dataset_index,
                            cloud_dataset_id,
                            dataset = name,
                            "dataset found or created"
                        );
                    }
                    // Found by name again on the next pass.
                    Err(error) => {
                        tracing::warn!(%error, dataset_index, "failed to store the cloud dataset id")
                    }
                }
                return;
            }
            Ok(DatasetByName::OtherKind) => (
                Failure::Blocked(ResolutionReason::NameTaken),
                "a dataset of the other kind, shared or private, has this name".to_string(),
            ),
            Ok(DatasetByName::Gone) => (
                Failure::Retry,
                "the server said the dataset exists but it could not be read back".to_string(),
            ),
            Err(error) => (classify(&error), error.to_string()),
        };
        self.store_outcome(*dataset_index, name, failure, reason)
            .await;
    }

    /// Log a failure to find or create a dataset and store what it means.
    async fn store_outcome(
        &self,
        dataset_index: i64,
        name: &str,
        failure: Failure,
        reason: String,
    ) {
        let stored = match failure {
            Failure::Refused(code) => {
                tracing::warn!(
                    reason,
                    dataset = name,
                    detail = code.as_str(),
                    "the server refused the dataset; recordings into it will not upload",
                );
                self.store.fail_dataset(dataset_index, code).await
            }
            Failure::Blocked(code) => {
                tracing::warn!(
                    reason,
                    dataset = name,
                    detail = code.as_str(),
                    "{}",
                    blocked_message(code)
                );
                self.store
                    .hold_dataset(dataset_index, Some(code), retry_at(BLOCKED_ID_RETRY))
                    .await
            }
            Failure::Retry => {
                tracing::warn!(
                    reason,
                    dataset = name,
                    "failed to find or create the dataset; retrying later",
                );
                self.store
                    .hold_dataset(dataset_index, None, retry_at(PASSING_ID_RETRY))
                    .await
            }
        };
        if let Err(error) = stored {
            tracing::warn!(%error, dataset_index, "failed to record how finding the dataset went");
        }
    }
}

/// What looking a dataset up by name, then creating it, found.
#[derive(Debug, PartialEq, Eq)]
enum DatasetByName {
    /// The server's id for a dataset of the kind asked for.
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    use tokio::time::Instant;

    use super::super::resolution::pass;
    use crate::intervals::PASSING_ID_RETRY;
    use crate::state::{ConnectionState, StateStore};

    use data_daemon_shared::DatasetSpec;
    use tempfile::TempDir;
    use tokio::time::{sleep, timeout};
    use wiremock::matchers::{body_json, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::api::auth::StaticAuthProvider;
    use crate::api::ApiClientOptions;
    use crate::state::ResolutionStatus;

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

        /// Select dataset `name` in `org`, as a process does.
        async fn select(&self, org: &str, name: &str, shared: bool) -> i64 {
            self.store
                .select_dataset(&DatasetSpec {
                    org_id: org.into(),
                    name: name.into(),
                    shared,
                    description: Some("pick and place".into()),
                    tags: vec!["shelf-a".into()],
                })
                .await
                .unwrap()
        }

        fn resolver(&self, server: &MockServer) -> DatasetResolver {
            DatasetResolver {
                store: Arc::clone(&self.store),
                client: Arc::new(client(server)),
            }
        }

        async fn dataset(&self, dataset_index: i64) -> DatasetRow {
            self.store
                .get_dataset(dataset_index)
                .await
                .unwrap()
                .unwrap()
        }

        /// Let every wait run out, as time passing would.
        async fn end_waits(&self) {
            sqlx::query("UPDATE datasets SET next_attempt_at = NULL")
                .execute(self.store.write_pool())
                .await
                .unwrap();
        }

        fn spawn(
            &self,
            server: &MockServer,
        ) -> (ResolverHandle, broadcast::Sender<ShutdownSignal>) {
            let (shutdown_tx, _) = broadcast::channel::<ShutdownSignal>(8);
            let handle = spawn_dataset_resolver(
                (*self.store).clone(),
                self.bus.clone(),
                Arc::new(client(server)),
                shutdown_tx.subscribe(),
            );
            (handle, shutdown_tx)
        }

        async fn wait_until_done(&self, dataset_index: i64, within: Duration) {
            timeout(within, async {
                while self.dataset(dataset_index).await.status != ResolutionStatus::Done {
                    sleep(Duration::from_millis(20)).await;
                }
            })
            .await
            .expect("the dataset row is confirmed");
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

    fn dataset_by_name(name: &str, response: ResponseTemplate) -> Mock {
        Mock::given(method("GET"))
            .and(path("/org/org-1/datasets/search/by-name"))
            .and(query_param("name", name))
            .respond_with(response)
    }

    fn dataset_json(id: &str, name: &str, is_shared: bool) -> serde_json::Value {
        serde_json::json!({ "id": id, "name": name, "is_shared": is_shared })
    }

    fn dataset_create(response: ResponseTemplate) -> Mock {
        Mock::given(method("POST"))
            .and(path("/org/org-1/datasets"))
            .respond_with(response)
    }

    fn found(name: &str, id: &str) -> Mock {
        dataset_by_name(
            name,
            ResponseTemplate::new(200).set_body_json(dataset_json(id, name, false)),
        )
    }

    /// The server's 403 for an account that may not create shared datasets.
    fn shared_upload_refused() -> ResponseTemplate {
        ResponseTemplate::new(403).set_body_json(serde_json::json!({
            "detail": {
                "error": "User is not authorized to upload shared datasets.",
                "status": 403,
                "error_type": "SHARED_UPLOAD_NOT_ALLOWED",
            },
        }))
    }

    #[tokio::test]
    async fn creates_a_missing_dataset_and_announces_it() {
        let fixture = Fixture::new().await;
        let dataset_index = fixture.select("org-1", "warehouse", false).await;
        let server = MockServer::start().await;
        dataset_by_name("warehouse", ResponseTemplate::new(404))
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
            .respond_with(ResponseTemplate::new(200).set_body_json(dataset_json(
                "cloud-dataset-3",
                "warehouse",
                false,
            )))
            .expect(1)
            .mount(&server)
            .await;

        assert_eq!(pass(&fixture.resolver(&server), true).await, None);

        let dataset = fixture.dataset(dataset_index).await;
        assert_eq!(dataset.status, ResolutionStatus::Done);
        assert_eq!(dataset.cloud_dataset_id.as_deref(), Some("cloud-dataset-3"));
    }

    #[tokio::test]
    async fn an_existing_dataset_of_the_same_kind_is_reused() {
        let fixture = Fixture::new().await;
        let dataset_index = fixture.select("org-1", "warehouse", false).await;
        let server = MockServer::start().await;
        found("warehouse", "cloud-dataset-1").mount(&server).await;
        dataset_create(ResponseTemplate::new(500))
            .expect(0)
            .mount(&server)
            .await;

        pass(&fixture.resolver(&server), true).await;

        assert_eq!(
            fixture
                .dataset(dataset_index)
                .await
                .cloud_dataset_id
                .as_deref(),
            Some("cloud-dataset-1")
        );
    }

    #[tokio::test]
    async fn each_dataset_is_found_in_the_org_stored_on_its_row() {
        let fixture = Fixture::new().await;
        let dataset_index = fixture.select("org-2", "warehouse", false).await;
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/org/org-2/datasets/search/by-name"))
            .respond_with(ResponseTemplate::new(200).set_body_json(dataset_json(
                "cloud-dataset-9",
                "warehouse",
                false,
            )))
            .expect(1)
            .mount(&server)
            .await;

        pass(&fixture.resolver(&server), true).await;

        assert_eq!(
            fixture
                .dataset(dataset_index)
                .await
                .cloud_dataset_id
                .as_deref(),
            Some("cloud-dataset-9")
        );
    }

    #[tokio::test]
    async fn a_dataset_of_the_other_kind_holds_the_row_with_its_reason() {
        let fixture = Fixture::new().await;
        let dataset_index = fixture.select("org-1", "warehouse", true).await;
        let server = MockServer::start().await;
        found("warehouse", "cloud-dataset-1").mount(&server).await;
        // Creating a shared one with the name is refused as taken.
        dataset_create(ResponseTemplate::new(409))
            .mount(&server)
            .await;

        pass(&fixture.resolver(&server), true).await;

        let dataset = fixture.dataset(dataset_index).await;
        assert_eq!(dataset.status, ResolutionStatus::Pending);
        assert_eq!(dataset.reason, Some(ResolutionReason::NameTaken));
        let wait = dataset.next_attempt_at.unwrap() - Utc::now().naive_utc();
        assert!(wait > chrono::Duration::minutes(4), "{wait}");
    }

    #[tokio::test]
    async fn a_dataset_created_by_someone_else_meanwhile_is_read_back() {
        let fixture = Fixture::new().await;
        let dataset_index = fixture.select("org-1", "warehouse", false).await;
        let server = MockServer::start().await;
        dataset_by_name("warehouse", ResponseTemplate::new(404))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        dataset_create(ResponseTemplate::new(409))
            .mount(&server)
            .await;
        found("warehouse", "cloud-dataset-5").mount(&server).await;

        pass(&fixture.resolver(&server), true).await;

        assert_eq!(
            fixture
                .dataset(dataset_index)
                .await
                .cloud_dataset_id
                .as_deref(),
            Some("cloud-dataset-5")
        );
    }

    #[tokio::test]
    async fn a_shared_dataset_the_server_refuses_fails_its_row() {
        let fixture = Fixture::new().await;
        let dataset_index = fixture.select("org-1", "warehouse", true).await;
        let server = MockServer::start().await;
        dataset_by_name("warehouse", ResponseTemplate::new(404))
            .mount(&server)
            .await;
        dataset_create(shared_upload_refused()).mount(&server).await;

        pass(&fixture.resolver(&server), true).await;

        let dataset = fixture.dataset(dataset_index).await;
        assert_eq!(dataset.status, ResolutionStatus::Failed);
        assert_eq!(dataset.reason, Some(ResolutionReason::SharedNotAllowed));
    }

    #[tokio::test]
    async fn a_403_without_the_shared_code_holds_the_row() {
        // Not a member of the organization, or only a viewer there: logging in
        // with an account that can record fixes it, so nothing fails.
        let fixture = Fixture::new().await;
        let dataset_index = fixture.select("org-1", "warehouse", true).await;
        let server = MockServer::start().await;
        dataset_by_name(
            "warehouse",
            ResponseTemplate::new(403)
                .set_body_json(serde_json::json!({"detail": "Insufficient permissions"})),
        )
        .mount(&server)
        .await;

        pass(&fixture.resolver(&server), true).await;

        let dataset = fixture.dataset(dataset_index).await;
        assert_eq!(dataset.status, ResolutionStatus::Pending);
        assert_eq!(dataset.reason, Some(ResolutionReason::NotInOrg));
    }

    #[tokio::test]
    async fn an_outage_retries_soon_and_keeps_an_earlier_reason() {
        let fixture = Fixture::new().await;
        let dataset_index = fixture.select("org-1", "warehouse", false).await;
        let server = MockServer::start().await;
        dataset_by_name(
            "warehouse",
            ResponseTemplate::new(403)
                .set_body_json(serde_json::json!({"detail": "Insufficient permissions"})),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
        dataset_by_name("warehouse", ResponseTemplate::new(503))
            .mount(&server)
            .await;

        pass(&fixture.resolver(&server), true).await;
        fixture.end_waits().await;
        let before = Instant::now();
        let due = pass(&fixture.resolver(&server), true)
            .await
            .expect("a pass is due");

        let dataset = fixture.dataset(dataset_index).await;
        assert_eq!(dataset.reason, Some(ResolutionReason::NotInOrg));
        assert!(
            due >= before + Duration::from_secs(15),
            "{:?}",
            due - before
        );
        assert!(due <= Instant::now() + PASSING_ID_RETRY);
    }

    #[tokio::test]
    async fn nothing_is_asked_while_offline() {
        let fixture = Fixture::new().await;
        let dataset_index = fixture.select("org-1", "warehouse", false).await;
        let server = MockServer::start().await;

        assert_eq!(pass(&fixture.resolver(&server), false).await, None);

        assert!(server.received_requests().await.unwrap().is_empty());
        assert_eq!(
            fixture.dataset(dataset_index).await.status,
            ResolutionStatus::Pending
        );
    }

    #[tokio::test]
    async fn the_task_resolves_at_startup() {
        let fixture = Fixture::new().await;
        let dataset_index = fixture.select("org-1", "warehouse", false).await;
        let server = MockServer::start().await;
        found("warehouse", "cloud-dataset-3").mount(&server).await;

        let (handle, shutdown_tx) = fixture.spawn(&server);
        fixture
            .wait_until_done(dataset_index, Duration::from_secs(3))
            .await;

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn the_task_resolves_a_selected_dataset_and_on_reconnect() {
        let fixture = Fixture::new().await;
        let server = MockServer::start().await;
        found("warehouse", "cloud-dataset-3").mount(&server).await;
        found("yard", "cloud-dataset-4").mount(&server).await;
        let (handle, shutdown_tx) = fixture.spawn(&server);
        sleep(Duration::from_millis(100)).await;

        let selected = fixture.select("org-1", "warehouse", false).await;
        fixture.bus.publish(DaemonEvent::DatasetSelected {
            dataset_index: selected,
        });
        fixture
            .wait_until_done(selected, Duration::from_secs(3))
            .await;

        fixture
            .bus
            .publish(DaemonEvent::ConnectionStateChanged(ConnectionState::Down));
        sleep(Duration::from_millis(100)).await;
        let offline = fixture.select("org-1", "yard", false).await;
        fixture.bus.publish(DaemonEvent::DatasetSelected {
            dataset_index: offline,
        });
        sleep(Duration::from_millis(200)).await;
        assert_eq!(
            fixture.dataset(offline).await.status,
            ResolutionStatus::Pending,
            "nothing is asked while offline"
        );
        fixture
            .bus
            .publish(DaemonEvent::ConnectionStateChanged(ConnectionState::Up));
        fixture
            .wait_until_done(offline, Duration::from_secs(3))
            .await;

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn the_task_asks_again_once_a_wait_runs_out() {
        let fixture = Fixture::new().await;
        let dataset_index = fixture.select("org-1", "warehouse", false).await;
        let due = Utc::now().naive_utc() + chrono::Duration::milliseconds(1500);
        fixture
            .store
            .hold_dataset(dataset_index, None, due)
            .await
            .unwrap();
        let server = MockServer::start().await;
        found("warehouse", "cloud-dataset-3").mount(&server).await;

        let (handle, shutdown_tx) = fixture.spawn(&server);
        sleep(Duration::from_millis(300)).await;
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "nothing is asked before the wait runs out"
        );
        // No event: only the wait running out wakes the resolver.
        fixture
            .wait_until_done(dataset_index, Duration::from_secs(5))
            .await;

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }

    #[tokio::test]
    async fn a_missed_event_makes_the_task_look_again() {
        let fixture = Fixture::new().await;
        fixture.select("org-1", "warehouse", false).await;
        let server = MockServer::start().await;
        // The startup pass is still asking about this one while events pile up.
        dataset_by_name(
            "warehouse",
            ResponseTemplate::new(200)
                .set_body_json(dataset_json("cloud-dataset-3", "warehouse", false))
                .set_delay(Duration::from_millis(800)),
        )
        .mount(&server)
        .await;
        found("yard", "cloud-dataset-5").mount(&server).await;
        let (handle, shutdown_tx) = fixture.spawn(&server);
        sleep(Duration::from_millis(200)).await;

        // A dataset is selected during that pass, and its event is lost among
        // more than the bus holds.
        let later = fixture.select("org-1", "yard", false).await;
        for _ in 0..crate::state::events::EVENT_BUS_CAPACITY + 10 {
            fixture
                .bus
                .publish(DaemonEvent::ConnectionStateChanged(ConnectionState::Up));
        }

        // Well before the 20 s a held row would wait.
        fixture.wait_until_done(later, Duration::from_secs(5)).await;

        let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
        handle.join().await;
    }
}
