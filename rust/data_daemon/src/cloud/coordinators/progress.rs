//! Periodic progress reporter.
//!
//! Every [`crate::intervals::PROGRESS_TICK`] the reporter sweeps the recordings still
//! pending a report ([`StateStore::recordings_pending_progress`] — a
//! server-side filter, so fully-settled recordings drop out of the scan) and,
//! for every stopped recording whose traces have all finished *writing* (and
//! whose `progress_reported` is still `Pending`),
//! POSTs `/org/{org}/recording/{rec}/traces-metadata` with the per-trace
//! `total_bytes` snapshot. This establishes the recording's upload
//! denominators on the backend up front — before uploads finish — so the
//! live per-trace `uploaded_bytes` stream renders as a partial-upload
//! percentage rather than a single jump to 100%. On success the recording
//! row flips to `progress_reported = 'reported'`.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::api::ApiClient;
use crate::cloud::OrgIdRx;
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::{
    ProgressReportStatus, RecordingRow, SqliteStateStore, StateStore, TraceRecord, TraceWriteStatus,
};

/// Handle returned by [`spawn_progress_reporter`].
pub struct ProgressReporterHandle {
    join: JoinHandle<()>,
}

impl ProgressReporterHandle {
    /// Wait for the reporter task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "progress reporter join failed");
        }
    }
}

/// Spawn the progress reporter task on the current Tokio runtime.
pub fn spawn_progress_reporter(
    store: SqliteStateStore,
    client: Arc<ApiClient>,
    org_rx: OrgIdRx,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> ProgressReporterHandle {
    let store = Arc::new(store);
    let join = tokio::spawn(async move {
        let mut ticker = interval(crate::intervals::PROGRESS_TICK);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "progress reporter shutting down");
                    break;
                }
                _ = ticker.tick() => {
                    sweep_once(&store, &client, &org_rx).await;
                }
            }
        }
    });
    ProgressReporterHandle { join }
}

async fn sweep_once(store: &Arc<SqliteStateStore>, client: &Arc<ApiClient>, org_rx: &OrgIdRx) {
    // Server-side filter to stopped, non-cancelled, cloud-id-assigned
    // recordings that still have reporting work outstanding, so fully-settled
    // recordings drop out of the sweep instead of being re-scanned (and their
    // traces re-fetched) on every tick. The cancelled/stopped/cloud-id guards
    // below are kept as belt-and-braces against a row racing the query.
    let recordings = match store.recordings_pending_progress().await {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(%error, "progress reporter could not query pending recordings");
            return;
        }
    };
    for recording in recordings {
        if recording.stopped_at.is_none() || recording.cancelled_at.is_some() {
            continue;
        }
        let Some(org_id) = org_rx.borrow().clone() else {
            continue;
        };
        // Every cloud URL needs the backend `recording_id`. A None here means
        // the start notifier hasn't populated the cloud id yet — skip until it
        // has (e.g. a recording made while the daemon was offline).
        let Some(recording_id) = recording.recording_id.clone() else {
            tracing::warn!(
                recording_index = recording.recording_index,
                "progress reporter skipping recording with no cloud recording_id yet"
            );
            continue;
        };
        let traces = match store
            .list_traces_for_recording(recording.recording_index)
            .await
        {
            Ok(rows) => rows,
            Err(error) => {
                tracing::warn!(%error, recording_index = recording.recording_index, "progress reporter could not list traces");
                continue;
            }
        };
        if traces.is_empty() {
            continue;
        }
        report_expected_trace_count(store, client, &recording, &org_id, &recording_id, &traces)
            .await;
        if matches!(recording.progress_reported, ProgressReportStatus::Reported) {
            continue;
        }
        report_progress(store, client, &recording, &org_id, &recording_id, &traces).await;
    }
}

/// Tell the backend how many traces this recording will have. Until this PUT
/// lands, the backend keeps the recording hidden from its parent dataset
/// regardless of how many trace blobs are already uploaded. Idempotent:
/// short-circuits once `expected_trace_count_reported` is non-zero.
async fn report_expected_trace_count(
    store: &Arc<SqliteStateStore>,
    client: &Arc<ApiClient>,
    recording: &RecordingRow,
    org_id: &str,
    recording_id: &str,
    traces: &[TraceRecord],
) {
    if recording.expected_trace_count_reported > 0 {
        return;
    }
    // Wait until every trace has reached a terminal write state. Reporting
    // the count too early would race the per-trace actors and risk telling
    // the backend a number that excludes traces still being flushed.
    if !traces.iter().all(write_status_is_terminal) {
        return;
    }
    let count = i64::try_from(traces.len()).unwrap_or(i64::MAX);

    // Persist locally first so a transient PUT failure does not lose the
    // count, and so a re-claim by the next tick sees the same value.
    if let Err(error) = store
        .set_expected_trace_count(recording.recording_index, count)
        .await
    {
        tracing::warn!(
            %error,
            recording_index = recording.recording_index,
            "failed to persist expected trace count"
        );
        return;
    }

    match client
        .put_expected_trace_count(org_id, recording_id, count)
        .await
    {
        Ok(()) => {
            if let Err(error) = store
                .mark_expected_trace_count_reported(recording.recording_index, count)
                .await
            {
                tracing::warn!(
                    %error,
                    recording_index = recording.recording_index,
                    "failed to mark expected trace count as reported"
                );
                return;
            }
            tracing::info!(
                recording_index = recording.recording_index,
                recording_id,
                count,
                "expected trace count reported"
            );
        }
        Err(error) => {
            tracing::warn!(
                %error,
                recording_index = recording.recording_index,
                "expected trace count PUT failed"
            );
        }
    }
}

async fn report_progress(
    store: &Arc<SqliteStateStore>,
    client: &Arc<ApiClient>,
    recording: &RecordingRow,
    org_id: &str,
    recording_id: &str,
    traces: &[TraceRecord],
) {
    // Send the snapshot of per-trace sizes (`total_bytes`) as soon as every
    // trace has finished *writing* — not once it has finished *uploading*.
    // This establishes the recording's denominators on the backend early, so
    // the live per-trace `uploaded_bytes` stream (sent via the batch-update
    // endpoint) can render a partial-upload percentage. Gating on upload
    // completion instead would withhold the denominators until the whole
    // recording is already uploaded, collapsing progress to a single 0→100%
    // jump. Failed writes are terminal too, so one bad trace can't pin the
    // recording in `progress_reported = pending` forever.
    if !traces.iter().all(write_status_is_terminal) {
        return;
    }
    let trace_map: HashMap<String, i64> = traces
        .iter()
        .map(|trace| (trace.trace_id.clone(), trace.total_bytes))
        .collect();
    // Move into a Reporting state so a slow request can't be re-issued
    // by the next tick.
    match store
        .set_progress_report_status(
            recording.recording_index,
            ProgressReportStatus::Pending,
            ProgressReportStatus::Reporting,
        )
        .await
    {
        Ok(Some(row)) if matches!(row.progress_reported, ProgressReportStatus::Reporting) => {}
        _ => return,
    }

    match client
        .report_progress(org_id, recording_id, &trace_map)
        .await
    {
        Ok(()) => {
            let _ = store
                .set_progress_report_status(
                    recording.recording_index,
                    ProgressReportStatus::Reporting,
                    ProgressReportStatus::Reported,
                )
                .await;
            tracing::info!(
                recording_index = recording.recording_index,
                recording_id,
                "progress report sent"
            );
        }
        Err(error) => {
            tracing::warn!(%error, recording_index = recording.recording_index, "progress report failed");
            let _ = store
                .set_progress_report_status(
                    recording.recording_index,
                    ProgressReportStatus::Reporting,
                    ProgressReportStatus::Pending,
                )
                .await;
        }
    }
}

fn write_status_is_terminal(trace: &TraceRecord) -> bool {
    matches!(
        trace.write_status,
        TraceWriteStatus::Written | TraceWriteStatus::Failed
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use crate::api::auth::StaticAuthProvider;
    use crate::api::client::ApiClientOptions;
    use crate::state::store::{NewRecording, TraceUpdate};
    use crate::state::{TraceUploadStatus, TraceWriteStatus};
    use tempfile::TempDir;
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn open_store() -> (SqliteStateStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .unwrap();
        (store, dir)
    }

    /// Create a recording stamped with `org-1` and the given cloud
    /// `recording_id` so the wiremock URL expectations resolve. Returns the
    /// local `recording_index`.
    async fn seed_recording(store: &SqliteStateStore, cloud_recording_id: &str) -> i64 {
        let recording = store
            .create_recording(NewRecording::default())
            .await
            .unwrap();
        store
            .mark_recording_start_notified(recording.recording_index, cloud_recording_id)
            .await
            .unwrap();
        recording.recording_index
    }

    /// A live-org receiver fixed at `org`. The sender is leaked so the channel
    /// stays open for the test's duration.
    fn org_rx(org: Option<&str>) -> OrgIdRx {
        let (org_tx, org_rx) = tokio::sync::watch::channel(org.map(str::to_string));
        Box::leak(Box::new(org_tx));
        org_rx
    }

    fn client(server: &MockServer) -> Arc<ApiClient> {
        let auth = Arc::new(StaticAuthProvider::new("test"));
        let mut options = ApiClientOptions::new(server.uri());
        options.max_backoff = Duration::from_millis(10);
        Arc::new(ApiClient::new(options, auth).unwrap())
    }

    #[tokio::test]
    async fn sweep_reports_count_and_progress_once_writes_settle() {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/org/org-1/recording/rec-1/expected-trace-count"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;
        // The progress snapshot must carry each trace's `total_bytes` (the
        // upload denominator), not its `uploaded_bytes` — and it must fire as
        // soon as writes settle, before uploads finish, so the backend can
        // render a live percentage from the streamed byte counts.
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/rec-1/traces-metadata"))
            .and(body_json(serde_json::json!({
                "traces": { "t-1": 100, "t-2": 200 }
            })))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let recording_index = seed_recording(&store, "rec-1").await;
        // Two traces finished writing (with known sizes) but neither has
        // uploaded yet — both the expected-count PUT and the progress POST
        // must fire on write completion, not upload completion.
        for (trace_id, total_bytes) in [("t-1", 100), ("t-2", 200)] {
            store
                .create_trace(recording_index, trace_id, Some("JOINT_POSITIONS"), None)
                .await
                .unwrap();
            store
                .update_trace(
                    trace_id,
                    TraceUpdate {
                        write_status: Some(TraceWriteStatus::Written),
                        total_bytes: Some(total_bytes),
                        ..TraceUpdate::default()
                    },
                )
                .await
                .unwrap();
        }
        store
            .mark_recording_stopped(recording_index, 0)
            .await
            .unwrap();

        let api = client(&server);
        sweep_once(&Arc::new(store.clone()), &api, &org_rx(Some("org-1"))).await;

        let recording = store.get_recording(recording_index).await.unwrap().unwrap();
        assert_eq!(recording.expected_trace_count, Some(2));
        assert_eq!(recording.expected_trace_count_reported, 2);
        // Progress reports once writes settle — uploads need not be done.
        assert!(matches!(
            recording.progress_reported,
            ProgressReportStatus::Reported
        ));
    }

    #[tokio::test]
    async fn sweep_skips_expected_count_while_writes_in_flight() {
        let server = MockServer::start().await;
        // No mock for the PUT — if the sweep fires it would 404 and we'd
        // catch a state-change side effect via the assertion below.
        let (store, _dir) = open_store().await;
        let recording_index = seed_recording(&store, "rec-1").await;
        store
            .create_trace(recording_index, "t-1", Some("JOINT_POSITIONS"), None)
            .await
            .unwrap();
        store
            .update_trace(
                "t-1",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Writing),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();
        store
            .mark_recording_stopped(recording_index, 0)
            .await
            .unwrap();

        let api = client(&server);
        sweep_once(&Arc::new(store.clone()), &api, &org_rx(Some("org-1"))).await;

        let recording = store.get_recording(recording_index).await.unwrap().unwrap();
        assert_eq!(recording.expected_trace_count, None);
        assert_eq!(recording.expected_trace_count_reported, 0);
    }

    #[tokio::test]
    async fn sweep_reports_when_one_trace_failed_and_rest_uploaded() {
        // Mixed terminal state: one trace Uploaded, one trace Failed.
        // The progress reporter should still POST and flip the
        // recording's status — a single failure must not deadlock the
        // whole recording.
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/org/org-1/recording/rec-1/expected-trace-count"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/rec-1/traces-metadata"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let recording_index = seed_recording(&store, "rec-1").await;
        store
            .create_trace(recording_index, "ok", Some("JOINT_POSITIONS"), None)
            .await
            .unwrap();
        store
            .update_trace(
                "ok",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    upload_status: Some(TraceUploadStatus::Uploaded),
                    bytes_uploaded: Some(7),
                    total_bytes: Some(7),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();
        store
            .create_trace(recording_index, "bad", Some("JOINT_POSITIONS"), None)
            .await
            .unwrap();
        store
            .update_trace(
                "bad",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    upload_status: Some(TraceUploadStatus::Failed),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();
        store
            .mark_recording_stopped(recording_index, 0)
            .await
            .unwrap();

        let api = client(&server);
        sweep_once(&Arc::new(store.clone()), &api, &org_rx(Some("org-1"))).await;

        let recording = store.get_recording(recording_index).await.unwrap().unwrap();
        assert!(
            matches!(recording.progress_reported, ProgressReportStatus::Reported),
            "progress should be reported even when one trace failed; \
             got {:?}",
            recording.progress_reported
        );
    }

    #[tokio::test]
    async fn sweep_marks_recording_reported_after_post() {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/org/org-1/recording/rec-1/expected-trace-count"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/rec-1/traces-metadata"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        let recording_index = seed_recording(&store, "rec-1").await;
        store
            .create_trace(recording_index, "trace-1", Some("JOINT_POSITIONS"), None)
            .await
            .unwrap();
        store
            .update_trace(
                "trace-1",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    upload_status: Some(TraceUploadStatus::Uploaded),
                    bytes_uploaded: Some(42),
                    total_bytes: Some(42),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();
        store
            .mark_recording_stopped(recording_index, 0)
            .await
            .unwrap();

        let api = client(&server);
        sweep_once(&Arc::new(store.clone()), &api, &org_rx(Some("org-1"))).await;

        let recording = store.get_recording(recording_index).await.unwrap().unwrap();
        assert!(matches!(
            recording.progress_reported,
            ProgressReportStatus::Reported
        ));
    }
}
