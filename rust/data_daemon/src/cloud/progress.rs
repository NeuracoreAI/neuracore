//! Periodic progress reporter.
//!
//! Phase 6f. Every [`PROGRESS_REPORT_INTERVAL`] ticks the reporter walks the
//! recordings table and, for every stopped recording whose traces have all
//! reached `Uploaded` (and whose `progress_reported` is still `Pending`),
//! POSTs `/org/{org}/recording/{rec}/traces-metadata` with the bytes-
//! uploaded snapshot. On success the recording row flips to
//! `progress_reported = 'reported'`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::api::ApiClient;
use crate::lifecycle::signals::ShutdownSignal;
use crate::state::{
    ProgressReportStatus, RecordingRow, SqliteStateStore, StateStore, TraceRecord,
    TraceUploadStatus, TraceWriteStatus,
};

/// Interval between progress-report sweeps.
pub const PROGRESS_REPORT_INTERVAL: Duration = Duration::from_secs(30);

/// Faster heartbeat used by tests / first-run so that a newly-uploaded
/// recording doesn't sit for 30 s before reporting. The tick advances the
/// scheduler but every actual flush is guarded by the upload-complete check.
const FAST_PROGRESS_TICK: Duration = Duration::from_secs(2);

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
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> ProgressReporterHandle {
    let store = Arc::new(store);
    let join = tokio::spawn(async move {
        let mut ticker = interval(FAST_PROGRESS_TICK);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "progress reporter shutting down");
                    break;
                }
                _ = ticker.tick() => {
                    sweep_once(&store, &client).await;
                }
            }
        }
    });
    ProgressReporterHandle { join }
}

async fn sweep_once(store: &Arc<SqliteStateStore>, client: &Arc<ApiClient>) {
    let recordings = match store.list_recordings().await {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(%error, "progress reporter could not list recordings");
            return;
        }
    };
    for recording in recordings {
        if recording.stopped_at.is_none() {
            continue;
        }
        let Some(org_id) = recording.org_id.clone() else {
            continue;
        };
        let traces = match store
            .list_traces_for_recording(&recording.recording_id)
            .await
        {
            Ok(rows) => rows,
            Err(error) => {
                tracing::warn!(%error, recording_id = recording.recording_id, "progress reporter could not list traces");
                continue;
            }
        };
        if traces.is_empty() {
            continue;
        }
        report_expected_trace_count(store, client, &recording, &org_id, &traces).await;
        if matches!(recording.progress_reported, ProgressReportStatus::Reported) {
            continue;
        }
        report_progress(store, client, &recording, &org_id, &traces).await;
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
        .set_expected_trace_count(&recording.recording_id, count)
        .await
    {
        tracing::warn!(
            %error,
            recording_id = recording.recording_id,
            "failed to persist expected trace count"
        );
        return;
    }

    match client
        .put_expected_trace_count(org_id, &recording.recording_id, count)
        .await
    {
        Ok(()) => {
            if let Err(error) = store
                .mark_expected_trace_count_reported(&recording.recording_id, count)
                .await
            {
                tracing::warn!(
                    %error,
                    recording_id = recording.recording_id,
                    "failed to mark expected trace count as reported"
                );
                return;
            }
            tracing::info!(
                recording_id = recording.recording_id,
                count,
                "expected trace count reported"
            );
        }
        Err(error) => {
            tracing::warn!(
                %error,
                recording_id = recording.recording_id,
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
    traces: &[TraceRecord],
) {
    // Treat Failed as terminal alongside Uploaded so a single bad trace
    // doesn't pin the recording in `progress_reported = pending` forever.
    // The backend receives `bytes_uploaded = 0` for failed entries, which
    // matches the Python progress-report semantics (the snapshot includes
    // every trace's `total_bytes`, even if the actual upload bytes are 0).
    let all_settled = traces.iter().all(|trace| {
        matches!(
            trace.upload_status,
            TraceUploadStatus::Uploaded | TraceUploadStatus::Failed
        )
    });
    if !all_settled {
        return;
    }
    let trace_map: HashMap<String, i64> = traces
        .iter()
        .map(|trace| (trace.trace_id.clone(), trace.bytes_uploaded))
        .collect();
    // Move into a Reporting state so a slow request can't be re-issued
    // by the next tick.
    match store
        .set_progress_report_status(
            &recording.recording_id,
            ProgressReportStatus::Pending,
            ProgressReportStatus::Reporting,
        )
        .await
    {
        Ok(Some(row)) if matches!(row.progress_reported, ProgressReportStatus::Reporting) => {}
        _ => return,
    }

    match client
        .report_progress(org_id, &recording.recording_id, &trace_map)
        .await
    {
        Ok(()) => {
            let _ = store
                .set_progress_report_status(
                    &recording.recording_id,
                    ProgressReportStatus::Reporting,
                    ProgressReportStatus::Reported,
                )
                .await;
            tracing::info!(
                recording_id = recording.recording_id,
                "progress report sent"
            );
        }
        Err(error) => {
            tracing::warn!(%error, recording_id = recording.recording_id, "progress report failed");
            let _ = store
                .set_progress_report_status(
                    &recording.recording_id,
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
    use crate::api::auth::StaticAuthProvider;
    use crate::api::client::ApiClientOptions;
    use crate::state::store::TraceUpdate;
    use crate::state::TraceWriteStatus;
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

    fn client(server: &MockServer) -> Arc<ApiClient> {
        let auth = Arc::new(StaticAuthProvider::new("test"));
        let mut options = ApiClientOptions::new(server.uri());
        options.max_backoff = Duration::from_millis(10);
        Arc::new(ApiClient::new(options, auth).unwrap())
    }

    #[tokio::test]
    async fn sweep_reports_expected_trace_count_once_writes_settle() {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/org/org-1/recording/rec-1/expected-trace-count"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let (store, _dir) = open_store().await;
        store.create_recording("rec-1").await.unwrap();
        store.set_recording_org("rec-1", "org-1").await.unwrap();
        // Two traces both finished writing but neither uploaded yet — the
        // expected-count PUT must fire before upload completion.
        for trace_id in ["t-1", "t-2"] {
            store
                .create_trace("rec-1", trace_id, Some("JOINT_POSITIONS"), None)
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
        }
        store.mark_recording_stopped("rec-1").await.unwrap();

        let api = client(&server);
        sweep_once(&Arc::new(store.clone()), &api).await;

        let recording = store.get_recording("rec-1").await.unwrap().unwrap();
        assert_eq!(recording.expected_trace_count, Some(2));
        assert_eq!(recording.expected_trace_count_reported, 2);
        // Progress report should not have fired — uploads aren't done.
        assert!(matches!(
            recording.progress_reported,
            ProgressReportStatus::Pending
        ));
    }

    #[tokio::test]
    async fn sweep_skips_expected_count_while_writes_in_flight() {
        let server = MockServer::start().await;
        // No mock for the PUT — if the sweep fires it would 404 and we'd
        // catch a state-change side effect via the assertion below.
        let (store, _dir) = open_store().await;
        store.create_recording("rec-1").await.unwrap();
        store.set_recording_org("rec-1", "org-1").await.unwrap();
        store
            .create_trace("rec-1", "t-1", Some("JOINT_POSITIONS"), None)
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
        store.mark_recording_stopped("rec-1").await.unwrap();

        let api = client(&server);
        sweep_once(&Arc::new(store.clone()), &api).await;

        let recording = store.get_recording("rec-1").await.unwrap().unwrap();
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
        store.create_recording("rec-1").await.unwrap();
        store.set_recording_org("rec-1", "org-1").await.unwrap();
        store
            .create_trace("rec-1", "ok", Some("JOINT_POSITIONS"), None)
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
            .create_trace("rec-1", "bad", Some("JOINT_POSITIONS"), None)
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
        store.mark_recording_stopped("rec-1").await.unwrap();

        let api = client(&server);
        sweep_once(&Arc::new(store.clone()), &api).await;

        let recording = store.get_recording("rec-1").await.unwrap().unwrap();
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
        store.create_recording("rec-1").await.unwrap();
        store.set_recording_org("rec-1", "org-1").await.unwrap();
        store
            .create_trace("rec-1", "trace-1", Some("JOINT_POSITIONS"), None)
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
        store.mark_recording_stopped("rec-1").await.unwrap();

        let api = client(&server);
        sweep_once(&Arc::new(store.clone()), &api).await;

        let recording = store.get_recording("rec-1").await.unwrap().unwrap();
        assert!(matches!(
            recording.progress_reported,
            ProgressReportStatus::Reported
        ));
    }
}
