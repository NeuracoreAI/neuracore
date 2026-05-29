//! Batch registration coordinator.
//!
//! Subscribes to [`DaemonEvent::TraceWritten`], buffers up to
//! `BATCH_SIZE` traces (or up to `MAX_WAIT`) and POSTs them to
//! `/org/{org}/recording/traces/batch-register`. Successful registrations
//! persist the returned resumable session URIs on the trace row and emit
//! `ReadyForUpload`; failures roll the registration status back to `Pending`
//! so the next tick re-claims them.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use uuid::Uuid;

use crate::api::models::RegisterTraceRequest;
use crate::api::ApiClient;
use crate::cloud::cloud_files::cloud_file_list;
use crate::lifecycle::signals::ShutdownSignal;
use crate::state::store::TraceUpdate;
use crate::state::{
    DaemonEvent, EventBus, RecordingRow, SqliteStateStore, StateStore, TraceRecord,
    TraceRegistrationStatus, TraceUploadStatus,
};

/// Maximum traces to register in a single call. Matches the
/// `claim_traces_for_registration` size trigger.
pub const BATCH_SIZE: usize = 50;
/// Maximum age before flushing a partial batch.
pub const MAX_WAIT: Duration = Duration::from_millis(200);
/// Poll interval the coordinator falls back to when the bus is quiet.
pub const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Handle returned by [`spawn_registration`].
pub struct RegistrationCoordinatorHandle {
    join: JoinHandle<()>,
}

impl RegistrationCoordinatorHandle {
    /// Wait for the coordinator task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "registration coordinator join failed");
        }
    }
}

/// Spawn the registration coordinator on the current Tokio runtime.
pub fn spawn_registration(
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> RegistrationCoordinatorHandle {
    let mut subscriber = bus.subscribe();
    let store = Arc::new(store);
    let join = tokio::spawn(async move {
        let mut ticker = interval(POLL_INTERVAL);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "registration coordinator shutting down");
                    break;
                }
                event = subscriber.recv() => {
                    match event {
                        Ok(DaemonEvent::TraceWritten { .. }) => {
                            drain_once(&store, &bus, &client, MAX_WAIT).await;
                        }
                        Ok(_) => {}
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            tracing::warn!(
                                skipped,
                                "registration coordinator missed bus events; \
                                falling back to a drain"
                            );
                            drain_once(&store, &bus, &client, MAX_WAIT).await;
                        }
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
                _ = ticker.tick() => {
                    drain_once(&store, &bus, &client, MAX_WAIT).await;
                }
            }
        }
    });
    RegistrationCoordinatorHandle { join }
}

async fn drain_once(
    store: &Arc<SqliteStateStore>,
    bus: &EventBus,
    client: &Arc<ApiClient>,
    max_wait: Duration,
) {
    let claimed = match store
        .claim_traces_for_registration(BATCH_SIZE, max_wait.as_secs_f64())
        .await
    {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(%error, "claim_traces_for_registration failed");
            return;
        }
    };
    if claimed.is_empty() {
        return;
    }
    tracing::debug!(count = claimed.len(), "claimed traces for registration");
    submit_batch(store, bus, client, claimed).await;
}

async fn submit_batch(
    store: &Arc<SqliteStateStore>,
    bus: &EventBus,
    client: &Arc<ApiClient>,
    traces: Vec<TraceRecord>,
) {
    // Group by recording so we can look up the recording row once per
    // recording rather than once per trace; in practice every claim ships
    // traces from a single recording but the protocol does not require that.
    let mut by_recording: HashMap<i64, Vec<TraceRecord>> = HashMap::new();
    for trace in traces {
        by_recording
            .entry(trace.recording_index)
            .or_default()
            .push(trace);
    }

    for (recording_index, traces) in by_recording {
        let row = match store.get_recording(recording_index).await {
            Ok(Some(row)) => row,
            Ok(None) => {
                tracing::warn!(
                    recording_index,
                    "recording row missing; rolling traces back to pending"
                );
                rollback_to_pending(store, &traces).await;
                continue;
            }
            Err(error) => {
                tracing::warn!(%error, recording_index, "failed to read recording row");
                rollback_to_pending(store, &traces).await;
                continue;
            }
        };

        let Some(org_id) = row.org_id.clone() else {
            tracing::warn!(
                recording_index,
                "recording has no org_id yet; rolling traces back to pending"
            );
            rollback_to_pending(store, &traces).await;
            continue;
        };

        // Resolve the cloud recording_id required by the backend. For an
        // offline recording (or one whose `/recording/start` POST has not yet
        // landed) the row carries no cloud id, so mint one on demand and
        // persist it via COALESCE — preferring the persisted value so a
        // concurrent start-notify cannot leave us double-minting.
        let cloud_id = match resolve_cloud_id(store, &row).await {
            Some(cloud_id) => cloud_id,
            None => {
                rollback_to_pending(store, &traces).await;
                continue;
            }
        };

        let payload: Vec<RegisterTraceRequest> = traces
            .iter()
            .map(|trace| RegisterTraceRequest {
                recording_id: cloud_id.clone(),
                data_type: trace.data_type.clone().unwrap_or_default(),
                trace_id: trace.trace_id.clone(),
                cloud_files: cloud_file_list(
                    trace.data_type.as_deref().unwrap_or(""),
                    trace.data_type_name.as_deref(),
                ),
            })
            .collect();

        match client.batch_register(&org_id, &payload).await {
            Ok(response) => {
                let registered_ids: HashMap<String, _> = response
                    .registered_traces
                    .into_iter()
                    .map(|entry| (entry.trace_id.clone(), entry.upload_session_uris))
                    .collect();
                let failed_ids: HashMap<String, Option<String>> = response
                    .failed_traces
                    .into_iter()
                    .map(|entry| (entry.trace_id, entry.error))
                    .collect();

                for trace in &traces {
                    if let Some(uris) = registered_ids.get(&trace.trace_id) {
                        let serialised = serde_json::to_string(uris).unwrap_or_else(|error| {
                            tracing::warn!(%error, trace_id = trace.trace_id, "failed to serialise session URIs");
                            "{}".to_string()
                        });
                        let update = TraceUpdate {
                            registration_status: Some(TraceRegistrationStatus::Registered),
                            upload_session_uris: Some(serialised),
                            upload_status: Some(TraceUploadStatus::Queued),
                            ..TraceUpdate::default()
                        };
                        if let Err(error) = store.update_trace(&trace.trace_id, update).await {
                            tracing::warn!(%error, trace_id = trace.trace_id, "failed to persist registration outcome");
                            continue;
                        }
                        bus.publish(DaemonEvent::TraceRegistered {
                            trace_id: trace.trace_id.clone(),
                            recording_index,
                        });
                        bus.publish(DaemonEvent::ReadyForUpload {
                            trace_id: trace.trace_id.clone(),
                            recording_index,
                        });
                    } else if let Some(error) = failed_ids.get(&trace.trace_id) {
                        tracing::warn!(
                            trace_id = trace.trace_id,
                            error = error.as_deref().unwrap_or(""),
                            "trace registration failed"
                        );
                        let update = TraceUpdate {
                            registration_status: Some(TraceRegistrationStatus::Failed),
                            error_message: Some(error.clone()),
                            ..TraceUpdate::default()
                        };
                        let _ = store.update_trace(&trace.trace_id, update).await;
                    } else {
                        // Backend silently dropped the trace — treat as a
                        // transient failure so the next tick retries it.
                        rollback_single_to_pending(store, &trace.trace_id).await;
                    }
                }
            }
            Err(error) => {
                tracing::warn!(%error, recording_index, "batch register request failed");
                rollback_to_pending(store, &traces).await;
            }
        }
    }
}

/// Resolve the cloud `recording_id` for a recording row, minting one on demand
/// when the row has none yet.
///
/// Offline recordings — and recordings whose `/recording/start` POST has not
/// landed — carry a NULL cloud id. Registration cannot wait for the start
/// notifier, so it mints a UUID and persists it via
/// [`StateStore::set_recording_cloud_id`], which COALESCEs to avoid clobbering
/// a concurrently-notified id. The persisted row's id is preferred so a race
/// with the start notifier resolves to a single shared value. Returns `None`
/// only when persistence fails, signalling the caller to roll back.
async fn resolve_cloud_id(store: &Arc<SqliteStateStore>, row: &RecordingRow) -> Option<String> {
    if let Some(cloud_id) = row.recording_id.clone() {
        return Some(cloud_id);
    }

    let minted = Uuid::new_v4().to_string();
    match store
        .set_recording_cloud_id(row.recording_index, &minted)
        .await
    {
        Ok(Some(persisted)) => persisted.recording_id.or(Some(minted)),
        Ok(None) => {
            tracing::warn!(
                recording_index = row.recording_index,
                "recording row vanished while minting cloud id"
            );
            None
        }
        Err(error) => {
            tracing::warn!(
                %error,
                recording_index = row.recording_index,
                "failed to persist minted cloud recording id"
            );
            None
        }
    }
}

async fn rollback_to_pending(store: &Arc<SqliteStateStore>, traces: &[TraceRecord]) {
    for trace in traces {
        rollback_single_to_pending(store, &trace.trace_id).await;
    }
}

async fn rollback_single_to_pending(store: &Arc<SqliteStateStore>, trace_id: &str) {
    let update = TraceUpdate {
        registration_status: Some(TraceRegistrationStatus::Pending),
        ..TraceUpdate::default()
    };
    if let Err(error) = store.update_trace(trace_id, update).await {
        tracing::warn!(%error, trace_id, "failed to roll registration status back");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::auth::StaticAuthProvider;
    use crate::api::client::ApiClientOptions;
    use crate::state::store::TraceUpdate;
    use crate::state::{NewRecording, TraceWriteStatus};
    use std::time::Duration;
    use tempfile::TempDir;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn open_store() -> (SqliteStateStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .unwrap();
        (store, dir)
    }

    /// Seed a recording (with `org_id` set) plus a single written trace under
    /// it, returning the local `recording_index`. When `cloud_id` is `Some`,
    /// the recording's cloud `recording_id` is persisted so registration finds
    /// one without minting; when `None`, registration mints on demand.
    async fn seed_written_trace(
        store: &SqliteStateStore,
        trace_id: &str,
        org_id: &str,
        cloud_id: Option<&str>,
    ) -> i64 {
        let recording_index = store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                robot_name: Some("arm"),
                dataset_id: Some("ds-1"),
                dataset_name: Some("warehouse"),
                org_id: Some(org_id),
                start_timestamp_ns: 1_700_000_000_000_000_000,
            })
            .await
            .unwrap()
            .recording_index;
        if let Some(cloud_id) = cloud_id {
            store
                .set_recording_cloud_id(recording_index, cloud_id)
                .await
                .unwrap();
        }
        store
            .create_trace(
                recording_index,
                trace_id,
                Some("JOINT_POSITIONS"),
                Some("arm0"),
            )
            .await
            .unwrap();
        store
            .update_trace(
                trace_id,
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();
        recording_index
    }

    fn client(server: &MockServer) -> Arc<ApiClient> {
        let auth = Arc::new(StaticAuthProvider::new("test-token"));
        let mut options = ApiClientOptions::new(server.uri());
        options.max_backoff = Duration::from_millis(10);
        Arc::new(ApiClient::new(options, auth).unwrap())
    }

    #[tokio::test]
    async fn successful_registration_persists_session_uri_and_emits_event() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/traces/batch-register"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "registered_traces": [{
                    "trace_id": "trace-1",
                    "upload_session_uris": {"JOINT_POSITIONS/arm0/trace.json": "https://upload/abc"}
                }],
                "failed_traces": []
            })))
            .expect(1)
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let recording_index =
            seed_written_trace(&store, "trace-1", "org-1", Some("cloud-rec-1")).await;
        let bus = EventBus::new();
        let mut subscriber = bus.subscribe();
        let api = client(&server);

        // Drive a single drain directly so the test does not depend on the
        // ticker firing.
        let claimed = store
            .claim_traces_for_registration(BATCH_SIZE, 0.0)
            .await
            .unwrap();
        submit_batch(&Arc::new(store.clone()), &bus, &api, claimed).await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(
            trace.registration_status,
            TraceRegistrationStatus::Registered
        );
        assert_eq!(trace.upload_status, TraceUploadStatus::Queued);
        assert!(trace
            .upload_session_uris
            .as_ref()
            .unwrap()
            .contains("https://upload/abc"));

        // First two events on the bus are TraceRegistered + ReadyForUpload.
        let mut saw_registered = false;
        let mut saw_ready = false;
        for _ in 0..2 {
            match subscriber.recv().await.unwrap() {
                DaemonEvent::TraceRegistered {
                    trace_id,
                    recording_index: event_index,
                } => {
                    assert_eq!(trace_id, "trace-1");
                    assert_eq!(event_index, recording_index);
                    saw_registered = true;
                }
                DaemonEvent::ReadyForUpload {
                    trace_id,
                    recording_index: event_index,
                } => {
                    assert_eq!(trace_id, "trace-1");
                    assert_eq!(event_index, recording_index);
                    saw_ready = true;
                }
                other => panic!("unexpected event: {other:?}"),
            }
        }
        assert!(saw_registered);
        assert!(saw_ready);
    }

    #[tokio::test]
    async fn failed_request_rolls_back_to_pending() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/traces/batch-register"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        seed_written_trace(&store, "trace-1", "org-1", Some("cloud-rec-1")).await;
        let bus = EventBus::new();
        let api = client(&server);

        let claimed = store
            .claim_traces_for_registration(BATCH_SIZE, 0.0)
            .await
            .unwrap();
        submit_batch(&Arc::new(store.clone()), &bus, &api, claimed).await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.registration_status, TraceRegistrationStatus::Pending);
    }

    #[tokio::test]
    async fn missing_org_id_rolls_back_to_pending() {
        let server = MockServer::start().await;
        let (store, _dir) = open_store().await;
        let recording_index = store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                robot_name: Some("arm"),
                dataset_id: Some("ds-1"),
                dataset_name: Some("warehouse"),
                org_id: None,
                start_timestamp_ns: 1_700_000_000_000_000_000,
            })
            .await
            .unwrap()
            .recording_index;
        store
            .create_trace(
                recording_index,
                "trace-1",
                Some("JOINT_POSITIONS"),
                Some("arm"),
            )
            .await
            .unwrap();
        store
            .update_trace(
                "trace-1",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();
        let bus = EventBus::new();
        let api = client(&server);

        let claimed = store
            .claim_traces_for_registration(BATCH_SIZE, 0.0)
            .await
            .unwrap();
        submit_batch(&Arc::new(store.clone()), &bus, &api, claimed).await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.registration_status, TraceRegistrationStatus::Pending);
    }

    #[tokio::test]
    async fn mints_cloud_id_on_demand_when_recording_has_none() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/traces/batch-register"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "registered_traces": [{
                    "trace_id": "trace-1",
                    "upload_session_uris": {"JOINT_POSITIONS/arm0/trace.json": "https://upload/abc"}
                }],
                "failed_traces": []
            })))
            .expect(1)
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        // No cloud id seeded: registration must mint one on demand.
        let recording_index = seed_written_trace(&store, "trace-1", "org-1", None).await;
        assert_eq!(
            store
                .get_recording(recording_index)
                .await
                .unwrap()
                .unwrap()
                .recording_id,
            None,
            "recording starts with no cloud id"
        );
        let bus = EventBus::new();
        let api = client(&server);

        let claimed = store
            .claim_traces_for_registration(BATCH_SIZE, 0.0)
            .await
            .unwrap();
        submit_batch(&Arc::new(store.clone()), &bus, &api, claimed).await;

        // A cloud id was minted and persisted on the recording row.
        let row = store.get_recording(recording_index).await.unwrap().unwrap();
        assert!(
            row.recording_id.is_some(),
            "cloud id should be minted on demand"
        );
        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(
            trace.registration_status,
            TraceRegistrationStatus::Registered
        );
    }
}
