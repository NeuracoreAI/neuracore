//! Batch registration coordinator.
//!
//! Claims traces whose row exists (any write_status except `failed`) — not just
//! fully-written ones — buffers up to `BATCH_SIZE` (or `MAX_WAIT`) and POSTs
//! them to `/org/{org}/recording/traces/batch-register`. Registration only
//! needs the trace's *identity* (recording id, trace id, data type, cloud
//! files), all known at `/recording/start`, so it runs **while the recording is
//! still writing** — overlapping the round trip with the recording instead of
//! adding it to the post-stop tail ("pre-registration").
//!
//! Because registration and the on-disk write can now finish in either order,
//! `ReadyForUpload` is gated on BOTH: every drain runs [`publish_ready_traces`],
//! which atomically promotes traces that are now registered *and* written to
//! `queued` and emits the event. Running it on the periodic tick is the safety
//! net for the lag between the `TraceWritten` event and the write-behind commit
//! of `write_status`. Registration failures roll the status back to `Pending`
//! so the next tick re-claims them.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::api::models::RegisterTraceRequest;
use crate::api::ApiClient;
use crate::cloud::cloud_files::cloud_file_list;
use crate::cloud::OrgIdRx;
use crate::lifecycle::signals::ShutdownSignal;
use crate::state::store::TraceUpdate;
use crate::state::{
    DaemonEvent, EventBus, SqliteStateStore, StateStore, TraceRecord, TraceRegistrationStatus,
};

/// Maximum traces to register in a single call. Matches the
/// `claim_traces_for_registration` size trigger.
pub const BATCH_SIZE: usize = 50;
/// Maximum age before flushing a partial batch.
pub const MAX_WAIT: Duration = Duration::from_millis(200);
/// Poll interval the coordinator falls back to when the bus is quiet.
pub const POLL_INTERVAL: Duration = Duration::from_millis(500);
/// How many times a trace the backend explicitly rejects (returns in
/// `failed_traces`) is rolled back to `pending` and retried before being marked
/// terminally `failed`. Backend registration errors are frequently transient
/// (e.g. a staging "Unexpected error during registration" under a large
/// registration burst); terminally failing on the first one permanently wedges
/// the whole recording (its traces never upload, so it never reaches "all
/// uploaded" and is never reaped). A small bounded retry rides out the hiccup
/// while still terminating a genuinely-permanent failure.
const MAX_REGISTRATION_ATTEMPTS: u32 = 5;

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
    org_rx: OrgIdRx,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> RegistrationCoordinatorHandle {
    let mut subscriber = bus.subscribe();
    let store = Arc::new(store);
    let join = tokio::spawn(async move {
        let mut ticker = interval(POLL_INTERVAL);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

        // Per-trace count of backend-rejected registration attempts, kept for
        // the coordinator's lifetime so the retry budget spans drains. Entries
        // are removed once a trace registers or is terminally failed, so the map
        // only ever holds currently-retrying traces.
        let mut registration_attempts: HashMap<String, u32> = HashMap::new();

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
                            drain_once(&store, &bus, &client, &org_rx, MAX_WAIT, &mut registration_attempts).await;
                        }
                        Ok(_) => {}
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            tracing::warn!(
                                skipped,
                                "registration coordinator missed bus events; \
                                falling back to a drain"
                            );
                            drain_once(&store, &bus, &client, &org_rx, MAX_WAIT, &mut registration_attempts).await;
                        }
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
                _ = ticker.tick() => {
                    drain_once(&store, &bus, &client, &org_rx, MAX_WAIT, &mut registration_attempts).await;
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
    org_rx: &OrgIdRx,
    max_wait: Duration,
    registration_attempts: &mut HashMap<String, u32>,
) {
    // Safety net: promote any traces that became (registered + written) since
    // the last drain. This runs even when there is nothing new to register, so
    // the periodic tick eventually promotes a pre-registered trace once its
    // write-behind `write_status = written` commit lands.
    publish_ready_traces(store, bus).await;

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
    submit_batch(store, bus, client, org_rx, claimed, registration_attempts).await;
    publish_ready_traces(store, bus).await;
}

async fn submit_batch(
    store: &Arc<SqliteStateStore>,
    bus: &EventBus,
    client: &Arc<ApiClient>,
    org_rx: &OrgIdRx,
    traces: Vec<TraceRecord>,
    registration_attempts: &mut HashMap<String, u32>,
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

        let Some(org_id) = org_rx.borrow().clone() else {
            tracing::warn!(
                recording_index,
                "no current org_id configured yet; rolling traces back to pending"
            );
            rollback_to_pending(store, &traces).await;
            continue;
        };

        // The backend recording_id always comes from `/recording/start`. An
        // offline recording (or one whose `/recording/start` POST has not yet
        // landed) carries no cloud id, so there is nothing to register against
        // yet — roll the traces back to pending and retry once the start
        // notifier has populated the id.
        let Some(cloud_id) = row.recording_id.clone() else {
            rollback_to_pending(store, &traces).await;
            continue;
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
                        // A serialise failure must NOT mark the trace registered
                        // with a "{}" placeholder — that records an empty URI map
                        // and the uploader later finalises it as 0 bytes uploaded
                        // (silent data loss). Roll back to pending so the next
                        // tick re-registers it instead.
                        let serialised = match serde_json::to_string(uris) {
                            Ok(serialised) => serialised,
                            Err(error) => {
                                tracing::warn!(%error, trace_id = trace.trace_id, "failed to serialise session URIs; rolling back to pending");
                                rollback_single_to_pending(store, &trace.trace_id).await;
                                continue;
                            }
                        };
                        let update = TraceUpdate {
                            registration_status: Some(TraceRegistrationStatus::Registered),
                            upload_session_uris: Some(serialised),
                            ..TraceUpdate::default()
                        };
                        if let Err(error) = store.update_trace(&trace.trace_id, update).await {
                            tracing::warn!(%error, trace_id = trace.trace_id, "failed to persist registration outcome");
                            continue;
                        }
                        // Registered — clear any accumulated retry budget.
                        registration_attempts.remove(&trace.trace_id);
                        bus.publish(DaemonEvent::TraceRegistered {
                            trace_id: trace.trace_id.clone(),
                            recording_index,
                        });
                    } else if let Some(error) = failed_ids.get(&trace.trace_id) {
                        // Backend rejections are usually transient (e.g. a
                        // staging burst error). Roll the trace back to `pending`
                        // and retry up to MAX_REGISTRATION_ATTEMPTS before giving
                        // up — terminally failing on the first rejection would
                        // permanently wedge the whole recording.
                        let attempts = registration_attempts
                            .entry(trace.trace_id.clone())
                            .or_insert(0);
                        *attempts += 1;
                        if *attempts < MAX_REGISTRATION_ATTEMPTS {
                            tracing::warn!(
                                trace_id = trace.trace_id,
                                error = error.as_deref().unwrap_or(""),
                                attempt = *attempts,
                                "trace registration rejected by backend; rolling back to pending for retry"
                            );
                            rollback_single_to_pending(store, &trace.trace_id).await;
                        } else {
                            tracing::warn!(
                                trace_id = trace.trace_id,
                                error = error.as_deref().unwrap_or(""),
                                attempts = *attempts,
                                "trace registration rejected by backend after retry budget exhausted; marking failed"
                            );
                            registration_attempts.remove(&trace.trace_id);
                            let update = TraceUpdate {
                                registration_status: Some(TraceRegistrationStatus::Failed),
                                error_message: Some(error.clone()),
                                ..TraceUpdate::default()
                            };
                            // If persisting the `failed` status itself fails the
                            // trace would otherwise sit in `registering` forever
                            // (no coordinator re-claims that state mid-session).
                            // Roll it back to `pending` so the next tick retries.
                            if let Err(persist_error) =
                                store.update_trace(&trace.trace_id, update).await
                            {
                                tracing::warn!(%persist_error, trace_id = trace.trace_id, "failed to persist registration failure; rolling back to pending");
                                rollback_single_to_pending(store, &trace.trace_id).await;
                            }
                        }
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

/// Promote any traces that are now both registered and written to `queued` and
/// emit `ReadyForUpload` for each.
///
/// Run on every drain (including the periodic tick) so it doubles as the safety
/// net for the lag between the `TraceWritten` event and the write-behind commit
/// of `write_status`: a pre-registered trace is promoted on whichever drain
/// first sees both states committed, rather than depending on a single event.
async fn publish_ready_traces(store: &Arc<SqliteStateStore>, bus: &EventBus) {
    match store.promote_ready_traces_to_queued().await {
        Ok(ready) => {
            for (trace_id, recording_index) in ready {
                bus.publish(DaemonEvent::ReadyForUpload {
                    trace_id,
                    recording_index,
                });
            }
        }
        Err(error) => {
            tracing::warn!(%error, "failed to promote ready traces for upload");
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
    use crate::state::{NewRecording, TraceUploadStatus, TraceWriteStatus};
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

    /// A live-org receiver fixed at `org` for the duration of a test. The
    /// sender is leaked so the channel stays open and `borrow()` keeps
    /// returning the seeded value.
    fn org_rx(org: Option<&str>) -> OrgIdRx {
        let (org_tx, org_rx) = tokio::sync::watch::channel(org.map(str::to_string));
        Box::leak(Box::new(org_tx));
        org_rx
    }

    /// Seed a recording plus a single written trace under it, returning the
    /// local `recording_index`. When `cloud_id` is `Some`, the recording's
    /// cloud `recording_id` is persisted (as the start notifier would) so
    /// registration finds one; when `None`, the recording has no cloud id yet
    /// and registration must defer.
    async fn seed_written_trace(
        store: &SqliteStateStore,
        trace_id: &str,
        cloud_id: Option<&str>,
    ) -> i64 {
        let recording_index = store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                dataset_id: Some("ds-1"),
                start_timestamp_ns: 1_700_000_000_000_000_000,
            })
            .await
            .unwrap()
            .recording_index;
        if let Some(cloud_id) = cloud_id {
            store
                .mark_recording_start_notified(recording_index, cloud_id)
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
        let recording_index = seed_written_trace(&store, "trace-1", Some("cloud-rec-1")).await;
        let bus = EventBus::new();
        let mut subscriber = bus.subscribe();
        let api = client(&server);

        // Drive a single drain directly so the test does not depend on the
        // ticker firing: register the batch, then run the promotion sweep that
        // emits ReadyForUpload once a trace is both registered and written.
        let store_arc = Arc::new(store.clone());
        let claimed = store
            .claim_traces_for_registration(BATCH_SIZE, 0.0)
            .await
            .unwrap();
        submit_batch(
            &store_arc,
            &bus,
            &api,
            &org_rx(Some("org-1")),
            claimed,
            &mut HashMap::new(),
        )
        .await;
        publish_ready_traces(&store_arc, &bus).await;

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
        seed_written_trace(&store, "trace-1", Some("cloud-rec-1")).await;
        let bus = EventBus::new();
        let api = client(&server);

        let claimed = store
            .claim_traces_for_registration(BATCH_SIZE, 0.0)
            .await
            .unwrap();
        submit_batch(
            &Arc::new(store.clone()),
            &bus,
            &api,
            &org_rx(Some("org-1")),
            claimed,
            &mut HashMap::new(),
        )
        .await;

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
                dataset_id: Some("ds-1"),
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
        submit_batch(
            &Arc::new(store.clone()),
            &bus,
            &api,
            &org_rx(None),
            claimed,
            &mut HashMap::new(),
        )
        .await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.registration_status, TraceRegistrationStatus::Pending);
    }

    #[tokio::test]
    async fn defers_registration_when_recording_has_no_cloud_id() {
        let server = MockServer::start().await;
        // The recording has no cloud id yet, so registration must not POST.
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/traces/batch-register"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        // No cloud id seeded: the start notifier hasn't populated one yet.
        let recording_index = seed_written_trace(&store, "trace-1", None).await;
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
        submit_batch(
            &Arc::new(store.clone()),
            &bus,
            &api,
            &org_rx(Some("org-1")),
            claimed,
            &mut HashMap::new(),
        )
        .await;

        // The recording still has no cloud id — none is minted locally.
        let row = store.get_recording(recording_index).await.unwrap().unwrap();
        assert_eq!(
            row.recording_id, None,
            "registration must not mint a cloud id"
        );
        // The trace is rolled back to pending for a later retry.
        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.registration_status, TraceRegistrationStatus::Pending);
    }

    #[tokio::test]
    async fn backend_rejection_retries_then_fails_after_budget() {
        // A backend that rejects a trace (returns it in `failed_traces`) is
        // treated as transient: the trace is rolled back to `pending` and
        // retried up to MAX_REGISTRATION_ATTEMPTS, then marked terminally
        // `failed`. Terminally failing on the first rejection would permanently
        // wedge the recording (the regression a staging burst-error exposed).
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/traces/batch-register"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "registered_traces": [],
                "failed_traces": [{
                    "trace_id": "trace-1",
                    "error": "Unexpected error during registration"
                }]
            })))
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        seed_written_trace(&store, "trace-1", Some("cloud-rec-1")).await;
        let bus = EventBus::new();
        let api = client(&server);
        let store_arc = Arc::new(store.clone());
        let mut attempts = HashMap::new();

        // Each of the first MAX-1 rejections rolls the trace back to pending so
        // the next tick re-claims and retries it.
        for attempt in 1..MAX_REGISTRATION_ATTEMPTS {
            let claimed = store
                .claim_traces_for_registration(BATCH_SIZE, 0.0)
                .await
                .unwrap();
            assert_eq!(claimed.len(), 1, "the pending trace is re-claimable");
            submit_batch(
                &store_arc,
                &bus,
                &api,
                &org_rx(Some("org-1")),
                claimed,
                &mut attempts,
            )
            .await;
            let trace = store.get_trace("trace-1").await.unwrap().unwrap();
            assert_eq!(
                trace.registration_status,
                TraceRegistrationStatus::Pending,
                "attempt {attempt} (< budget) must retry, not terminate"
            );
        }

        // The final rejection exhausts the budget → terminal failure.
        let claimed = store
            .claim_traces_for_registration(BATCH_SIZE, 0.0)
            .await
            .unwrap();
        submit_batch(
            &store_arc,
            &bus,
            &api,
            &org_rx(Some("org-1")),
            claimed,
            &mut attempts,
        )
        .await;
        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(
            trace.registration_status,
            TraceRegistrationStatus::Failed,
            "an exhausted retry budget terminates the trace"
        );
        assert_eq!(
            trace.error_message.as_deref(),
            Some("Unexpected error during registration")
        );
    }
}
