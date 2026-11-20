//! Multipart file uploader coordinator.
//!
//! Subscribes to [`DaemonEvent::ReadyForUpload`] (and re-scans the
//! state store on startup for any traces already in the registered/queued
//! state). For each on-disk artefact the coordinator PUTs the parts of the
//! multipart upload persisted by the registration coordinator to presigned
//! URLs, one part at a time, and persists the file's CRC32C once every part is
//! up. On completion the trace is marked `Uploaded` and the status updater
//! sends the uploads to the backend to complete. A file whose upload the
//! backend no longer holds restarts the trace's uploads.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use tokio::sync::{broadcast, watch, Semaphore};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::{interval, MissedTickBehavior};

use super::status::{uploaded_files, StatusUpdate};
use super::upload_transfer::{upload_one_file, UploadFileError};
use crate::api::ApiClient;
use crate::cloud::OrgIdRx;
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::store::TraceUpdate;
use crate::state::{
    ConnectionState, DaemonEvent, EventBus, MultipartUploads, SqliteStateStore, StateStore,
    TraceRecord, TraceUploadStatus, TraceWriteHandle,
};
use crate::storage::paths::TracePath;

/// Maximum number of traces uploading concurrently. A trade-off: more slots
/// drain a large backlog in fewer rounds, but oversubscribing the link inflates
/// per-request latency and its p95/p99 tail, which drives timeouts and retries.
pub const MAX_CONCURRENT_UPLOADS: usize = 16;

/// Handle returned by [`spawn_uploader`].
pub struct UploaderHandle {
    join: JoinHandle<()>,
}

impl UploaderHandle {
    /// Wait for the uploader task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "uploader join failed");
        }
    }
}

/// Spawn the uploader task on the current Tokio runtime.
#[allow(clippy::too_many_arguments)]
pub fn spawn_uploader(
    store: SqliteStateStore,
    trace_writer: TraceWriteHandle,
    bus: EventBus,
    client: Arc<ApiClient>,
    recordings_root: Arc<PathBuf>,
    org_rx: OrgIdRx,
    status_tx: tokio::sync::mpsc::UnboundedSender<StatusUpdate>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> UploaderHandle {
    let mut subscriber = bus.subscribe();
    let store = Arc::new(store);
    let join = tokio::spawn(async move {
        let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_UPLOADS));
        let mut in_flight: JoinSet<String> = JoinSet::new();
        // Tracks dispatched trace IDs to prevent a drain triggered by
        // join_next from re-queuing a trace whose task hasn't yet run the
        // DB update to mark itself Uploading.
        let mut in_flight_ids: HashSet<String> = HashSet::new();
        let aborts = UploadAborts::default();
        // Safety-net rescan: catch any traces that were skipped when the
        // semaphore was full during a drain, without relying on bus events.
        let mut rescan_tick = interval(crate::intervals::UPLOAD_RESCAN);
        rescan_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut connected = false;
        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "uploader shutting down");
                    break;
                }
                // Reap a completed task immediately and chain the next drain
                // so a finishing upload starts the next one without waiting
                // for a bus event or the rescan tick.
                Some(join_result) = in_flight.join_next(), if !in_flight.is_empty() => {
                    match join_result {
                        Ok(completed_trace_id) => { in_flight_ids.remove(&completed_trace_id); }
                        Err(panic_err) => { tracing::warn!(?panic_err, "upload task panicked"); }
                    }
                    if connected {
                        drain_ready_traces(
                            &store,
                            &trace_writer,
                            &bus,
                            &client,
                            &recordings_root,
                            &org_rx,
                            &status_tx,
                            &semaphore,
                            &mut in_flight,
                            &mut in_flight_ids,
                            &aborts,
                        )
                        .await;
                    }
                }
                event = subscriber.recv() => {
                    match event {
                        Ok(DaemonEvent::ConnectionStateChanged(state)) => {
                            connected = matches!(state, ConnectionState::Up);
                            if connected {
                                drain_ready_traces(
                                    &store,
                                    &trace_writer,
                                    &bus,
                                    &client,
                                    &recordings_root,
                                    &org_rx,
                                    &status_tx,
                                    &semaphore,
                                    &mut in_flight,
                                    &mut in_flight_ids,
                                    &aborts,
                                )
                                .await;
                            }
                        }
                        Ok(DaemonEvent::ReadyForUpload { trace_id, .. }) => {
                            if !connected {
                                tracing::debug!(trace_id, "deferring upload until connection up");
                                continue;
                            }
                            spawn_upload_task(
                                &store,
                                &trace_writer,
                                &bus,
                                &client,
                                &recordings_root,
                                &org_rx,
                                &status_tx,
                                &semaphore,
                                &mut in_flight,
                                &mut in_flight_ids,
                                &aborts,
                                trace_id,
                            );
                        }
                        Ok(
                            DaemonEvent::RecordingCancelled { recording_index }
                            | DaemonEvent::RecordingNotFound { recording_index },
                        ) => aborts.abort(recording_index),
                        Ok(_) => {}
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            tracing::warn!(skipped, "uploader missed bus events; rescanning");
                            aborts.abort_settled(&store).await;
                            if connected {
                                drain_ready_traces(
                                    &store,
                                    &trace_writer,
                                    &bus,
                                    &client,
                                    &recordings_root,
                                    &org_rx,
                                    &status_tx,
                                    &semaphore,
                                    &mut in_flight,
                                    &mut in_flight_ids,
                                    &aborts,
                                )
                                .await;
                            }
                        }
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
                _ = rescan_tick.tick() => {
                    if connected {
                        drain_ready_traces(
                            &store,
                            &trace_writer,
                            &bus,
                            &client,
                            &recordings_root,
                            &org_rx,
                            &status_tx,
                            &semaphore,
                            &mut in_flight,
                            &mut in_flight_ids,
                            &aborts,
                        )
                        .await;
                    }
                }
            }
        }
        in_flight.shutdown().await;
    });
    UploaderHandle { join }
}

#[allow(clippy::too_many_arguments)]
async fn drain_ready_traces(
    store: &Arc<SqliteStateStore>,
    trace_writer: &TraceWriteHandle,
    bus: &EventBus,
    client: &Arc<ApiClient>,
    recordings_root: &Arc<PathBuf>,
    org_rx: &OrgIdRx,
    status_tx: &tokio::sync::mpsc::UnboundedSender<StatusUpdate>,
    semaphore: &Arc<Semaphore>,
    in_flight: &mut JoinSet<String>,
    in_flight_ids: &mut HashSet<String>,
    aborts: &UploadAborts,
) {
    // Server-side filter for `queued`/`retrying` traces (uses
    // `idx_traces_upload_status`) instead of walking every recording's full
    // trace set on each completed upload — the old N+1 scan was quadratic under
    // the burst this loop runs after every `join_next`.
    let trace_ids = match store.traces_ready_for_upload().await {
        Ok(ids) => ids,
        Err(error) => {
            tracing::warn!(%error, "uploader could not query traces ready for upload");
            return;
        }
    };
    for trace_id in trace_ids {
        spawn_upload_task(
            store,
            trace_writer,
            bus,
            client,
            recordings_root,
            org_rx,
            status_tx,
            semaphore,
            in_flight,
            in_flight_ids,
            aborts,
            trace_id,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_upload_task(
    store: &Arc<SqliteStateStore>,
    trace_writer: &TraceWriteHandle,
    bus: &EventBus,
    client: &Arc<ApiClient>,
    recordings_root: &Arc<PathBuf>,
    org_rx: &OrgIdRx,
    status_tx: &tokio::sync::mpsc::UnboundedSender<StatusUpdate>,
    semaphore: &Arc<Semaphore>,
    in_flight: &mut JoinSet<String>,
    in_flight_ids: &mut HashSet<String>,
    aborts: &UploadAborts,
    trace_id: String,
) {
    if in_flight_ids.contains(&trace_id) {
        tracing::debug!(trace_id, "trace already dispatched; skipping duplicate");
        return;
    }
    let Ok(permit) = Arc::clone(semaphore).try_acquire_owned() else {
        tracing::debug!(trace_id, "upload semaphore full; will retry on next drain");
        return;
    };
    in_flight_ids.insert(trace_id.clone());
    let store = Arc::clone(store);
    let trace_writer = trace_writer.clone();
    let bus = bus.clone();
    let client = Arc::clone(client);
    let recordings_root = Arc::clone(recordings_root);
    let org_rx = org_rx.clone();
    let status_tx = status_tx.clone();
    let aborts = aborts.clone();
    in_flight.spawn(async move {
        if let Some(abort) = aborts.watch_trace(&store, &trace_id).await {
            tokio::select! {
                () = upload_single(
                    &store,
                    &trace_writer,
                    &bus,
                    &client,
                    &recordings_root,
                    &org_rx,
                    &status_tx,
                    &trace_id,
                ) => {}
                // Dropping the upload future cuts the PUT off mid-chunk.
                () = aborted(abort) => {
                    tracing::info!(trace_id, "recording cancelled or not found; aborted its upload");
                }
            }
        }
        drop(permit);
        trace_id
    });
}

/// One abort signal per recording, raised by the uploader loop when the
/// recording is cancelled or not found, so upload tasks need no bus
/// subscription of their own.
#[derive(Clone, Default)]
struct UploadAborts(Arc<Mutex<HashMap<i64, watch::Sender<bool>>>>);

impl UploadAborts {
    /// The abort signal for `trace_id`'s recording, or `None` when the
    /// recording is already cancelled or not found and must not upload.
    ///
    /// Registers before reading the recording, so an abort raised between the
    /// two is not missed. An unreadable store never aborts an upload.
    async fn watch_trace(
        &self,
        store: &Arc<SqliteStateStore>,
        trace_id: &str,
    ) -> Option<watch::Receiver<bool>> {
        let Ok(Some(trace)) = store.get_trace(trace_id).await else {
            return Some(watch::channel(false).1);
        };
        let abort = self.watch(trace.recording_index);
        if is_settled(store, trace.recording_index).await {
            return None;
        }
        Some(abort)
    }

    fn watch(&self, recording_index: i64) -> watch::Receiver<bool> {
        let mut senders = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        senders.retain(|_, sender| sender.receiver_count() > 0);
        senders
            .entry(recording_index)
            .or_insert_with(|| watch::channel(false).0)
            .subscribe()
    }

    fn abort(&self, recording_index: i64) {
        let sender = self
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&recording_index);
        if let Some(sender) = sender {
            let _ = sender.send(true);
        }
    }

    /// Abort every watched recording the store shows as settled, for when the
    /// loop may have missed the event that would have aborted it.
    async fn abort_settled(&self, store: &Arc<SqliteStateStore>) {
        let watched: Vec<i64> = self
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .keys()
            .copied()
            .collect();
        for recording_index in watched {
            if is_settled(store, recording_index).await {
                self.abort(recording_index);
            }
        }
    }
}

/// Resolve once the recording is aborted. Never resolves otherwise.
async fn aborted(mut abort: watch::Receiver<bool>) {
    if abort.wait_for(|aborted| *aborted).await.is_err() {
        std::future::pending::<()>().await;
    }
}

/// True when the recording is cancelled or not found. A failed read answers
/// `false`.
async fn is_settled(store: &Arc<SqliteStateStore>, recording_index: i64) -> bool {
    matches!(
        store.get_recording(recording_index).await,
        Ok(Some(row)) if row.cancelled_at.is_some() || row.backend_not_found_at.is_some()
    )
}

#[allow(clippy::too_many_arguments)]
async fn upload_single(
    store: &Arc<SqliteStateStore>,
    trace_writer: &TraceWriteHandle,
    bus: &EventBus,
    client: &Arc<ApiClient>,
    recordings_root: &Arc<PathBuf>,
    org_rx: &OrgIdRx,
    status_tx: &tokio::sync::mpsc::UnboundedSender<StatusUpdate>,
    trace_id: &str,
) {
    let trace = match store.get_trace(trace_id).await {
        Ok(Some(trace)) => trace,
        Ok(None) => {
            tracing::warn!(trace_id, "uploader could not find trace row");
            return;
        }
        Err(error) => {
            tracing::warn!(%error, trace_id, "uploader failed to load trace row");
            return;
        }
    };
    let mut uploads = match trace.decode_uploads() {
        Ok(Some(uploads)) => uploads,
        Ok(None) => {
            tracing::warn!(
                trace_id,
                "trace ready-for-upload but no multipart uploads stored"
            );
            return;
        }
        Err(error) => {
            tracing::warn!(%error, trace_id, "failed to decode stored multipart uploads");
            return;
        }
    };
    if uploads.is_empty() {
        // Every trace registers at least `trace.json`, so an empty map means
        // the files would never be uploaded.
        mark_failed_and_emit(store, bus, &trace, "no multipart uploads").await;
        return;
    }
    // Leave the trace queued until an org is selected; a later drain retries.
    let Some(org_id) = org_rx.borrow().clone() else {
        tracing::warn!(trace_id, "no current org_id configured; deferring upload");
        return;
    };

    // Mark the trace as uploading so the next bus tick doesn't repeat the
    // attempt (the registration path is one-shot, but the periodic rescan
    // could re-enter on a long-running upload).
    let _ = store
        .update_trace(
            trace_id,
            TraceUpdate {
                upload_status: Some(TraceUploadStatus::Uploading),
                ..TraceUpdate::default()
            },
        )
        .await;

    tracing::info!(trace_id, data_type = ?trace.data_type, "starting trace upload");
    let Some(data_type) = trace.data_type.as_deref() else {
        // No data_type means we never saw a `StartTrace` for this row, so we
        // can't locate the on-disk artefact. Surface the failure on the event
        // bus so the recording's progress reporter (which gates on every trace
        // having settled) doesn't wait for an upload that can never happen.
        tracing::warn!(trace_id, "trace row missing data_type; marking failed");
        mark_failed_and_emit(store, bus, &trace, "trace missing data_type").await;
        return;
    };
    // On-disk artefacts are keyed by the local `recording_index`, matching the
    // directory the dispatcher / trace actors wrote to.
    let trace_dir = TracePath::new(
        trace.recording_index.to_string(),
        data_type,
        trace_id.to_string(),
    )
    .directory(recordings_root.as_path());
    let upload_started = Instant::now();
    crate::perf_events::emit(
        "upload",
        "started",
        Some(trace.recording_index),
        Some(trace_id),
        None,
        serde_json::json!({
            "data_type": data_type,
            "file_count": uploads.len(),
        }),
    );

    // Upload each on-disk artefact under its multipart upload, persisting its
    // CRC32C as it finishes so a restart skips the files already up.
    let mut total_uploaded: i64 = 0;
    let filepaths: Vec<String> = uploads.keys().cloned().collect();
    for filename in filepaths {
        let local_path = trace_dir.join(file_basename(&filename));
        let local_len = local_path.metadata().map(|metadata| metadata.len()).ok();
        if local_len.unwrap_or(0) == 0 {
            tracing::warn!(
                trace_id,
                path = %local_path.display(),
                bytes = local_len,
                "expected upload artefact missing or empty; marking trace failed"
            );
            mark_failed_and_emit(
                store,
                bus,
                &trace,
                &format!("missing or empty artefact {filename}"),
            )
            .await;
            emit_upload_failed(
                &trace,
                &filename,
                upload_started,
                total_uploaded,
                if local_len.is_none() {
                    "missing_artefact"
                } else {
                    "empty_artefact"
                },
            );
            return;
        }
        let local_len = local_len.unwrap_or(0) as i64;
        let upload = &uploads[&filename];
        if upload.crc32c.is_some() {
            total_uploaded = total_uploaded.saturating_add(local_len);
            continue;
        }

        let outcome = upload_one_file(
            client,
            trace_writer,
            bus,
            status_tx,
            &trace,
            &org_id,
            &local_path,
            &filename,
            upload,
        )
        .await;
        match outcome {
            Ok(crc) => {
                total_uploaded = total_uploaded.saturating_add(local_len);
                if let Some(upload) = uploads.get_mut(&filename) {
                    upload.crc32c = Some(BASE64.encode(crc.to_be_bytes()));
                }
                persist_uploads(store, trace_id, &uploads).await;
            }
            Err(UploadFileError::UploadMissing) => {
                emit_upload_failed(
                    &trace,
                    &filename,
                    upload_started,
                    total_uploaded,
                    "upload_missing",
                );
                super::restart_trace_uploads(
                    store,
                    trace_id,
                    TraceUploadStatus::Uploading,
                    "the backend no longer holds an upload",
                )
                .await;
                return;
            }
            Err(UploadFileError::Failed(error)) => {
                tracing::warn!(%error, trace_id, "file cannot upload; marking trace failed");
                emit_upload_failed(
                    &trace,
                    &filename,
                    upload_started,
                    total_uploaded,
                    "upload_failed",
                );
                mark_failed_and_emit(store, bus, &trace, &error).await;
                return;
            }
            Err(UploadFileError::Retry(error)) => {
                tracing::warn!(%error, trace_id, "upload failed; rolling back to retrying");
                emit_upload_failed(
                    &trace,
                    &filename,
                    upload_started,
                    total_uploaded,
                    "transfer_failed",
                );
                let update = TraceUpdate {
                    upload_status: Some(TraceUploadStatus::Retrying),
                    error_message: Some(Some(error)),
                    ..TraceUpdate::default()
                };
                if let Err(error) = store.update_trace(trace_id, update).await {
                    tracing::warn!(%error, trace_id, "failed to mark trace as retrying");
                }
                return;
            }
        }
    }

    crate::perf_events::emit(
        "upload",
        "completed",
        Some(trace.recording_index),
        Some(trace_id),
        Some(upload_started.elapsed()),
        serde_json::json!({
            "outcome": "ok",
            "file_count": uploads.len(),
            "bytes_uploaded": total_uploaded,
        }),
    );
    finalise_upload(store, bus, status_tx, &trace, total_uploaded, &uploads).await;
}

fn emit_upload_failed(
    trace: &TraceRecord,
    filename: &str,
    upload_started: Instant,
    bytes_uploaded: i64,
    reason: &str,
) {
    crate::perf_events::emit(
        "upload",
        "failed",
        Some(trace.recording_index),
        Some(&trace.trace_id),
        Some(upload_started.elapsed()),
        serde_json::json!({
            "outcome": "error",
            "reason": reason,
            "filename": file_basename(filename),
            "bytes_uploaded": bytes_uploaded,
        }),
    );
}

async fn finalise_upload(
    store: &Arc<SqliteStateStore>,
    bus: &EventBus,
    status_tx: &tokio::sync::mpsc::UnboundedSender<StatusUpdate>,
    trace: &TraceRecord,
    total_uploaded: i64,
    uploads: &MultipartUploads,
) {
    let update = TraceUpdate {
        upload_status: Some(TraceUploadStatus::Uploaded),
        bytes_uploaded: Some(total_uploaded),
        total_bytes: Some(total_uploaded.max(trace.total_bytes)),
        ..TraceUpdate::default()
    };
    if let Err(error) = store.update_trace(&trace.trace_id, update).await {
        tracing::warn!(%error, trace_id = trace.trace_id, "failed to mark trace uploaded");
    }
    bus.publish(DaemonEvent::UploadComplete {
        trace_id: trace.trace_id.clone(),
        recording_index: trace.recording_index,
    });
    let _ = status_tx.send(StatusUpdate::completed(
        trace.recording_index,
        trace.trace_id.clone(),
        total_uploaded.max(trace.total_bytes),
        uploaded_files(uploads),
    ));
}

async fn mark_failed_and_emit(
    store: &Arc<SqliteStateStore>,
    bus: &EventBus,
    trace: &TraceRecord,
    message: &str,
) {
    let update = TraceUpdate {
        upload_status: Some(TraceUploadStatus::Failed),
        error_message: Some(Some(message.to_string())),
        ..TraceUpdate::default()
    };
    if let Err(error) = store.update_trace(&trace.trace_id, update).await {
        tracing::warn!(%error, trace_id = trace.trace_id, "failed to mark trace as failed");
    }
    // Publishing on the upload-complete topic lets the progress reporter and
    // status updater treat the trace as terminal — without this signal a
    // single bad trace would block the recording's progress report forever.
    bus.publish(DaemonEvent::UploadComplete {
        trace_id: trace.trace_id.clone(),
        recording_index: trace.recording_index,
    });
    // No `completed` status: the backend must not save a recording with
    // files missing.
}

async fn persist_uploads(
    store: &Arc<SqliteStateStore>,
    trace_id: &str,
    uploads: &MultipartUploads,
) {
    let serialised = match serde_json::to_string(uploads) {
        Ok(serialised) => serialised,
        Err(error) => {
            tracing::warn!(%error, trace_id, "failed to serialise multipart uploads");
            return;
        }
    };
    let update = TraceUpdate {
        multipart_uploads: Some(serialised),
        ..TraceUpdate::default()
    };
    if let Err(error) = store.update_trace(trace_id, update).await {
        tracing::warn!(%error, trace_id, "failed to persist multipart uploads");
    }
}

fn file_basename(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some((_, tail)) => tail,
        None => path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::auth::StaticAuthProvider;
    use crate::api::client::ApiClientOptions;
    use crate::api::CompletedFileUpload;
    use crate::state::store::{NewRecording, TraceUpdate};
    use crate::state::{
        MultipartUpload, RecordingTimestamp, TraceRegistrationStatus, TraceUploadStatus,
        TraceWriteStatus,
    };
    use crate::storage::paths::{LOSSY_VIDEO_FILENAME, TRACE_JSON_FILENAME};
    use std::time::Duration;
    use tempfile::TempDir;
    use tokio::sync::mpsc;
    use tokio::time::{sleep, timeout};
    use wiremock::matchers::{method, path, path_regex};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    const JSON_FILEPATH: &str = "JOINT_POSITIONS/arm/trace.json";
    const PARTS_ROUTE: &str = "/v2/org/org-1/uploads/parts";

    /// A live-org receiver fixed at `org`. The sender is leaked so the channel
    /// stays open for the test's duration.
    fn org_rx(org: Option<&str>) -> OrgIdRx {
        let (org_tx, org_rx) = tokio::sync::watch::channel(org.map(str::to_string));
        Box::leak(Box::new(org_tx));
        org_rx
    }

    async fn open_store() -> (SqliteStateStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .unwrap();
        (store, dir)
    }

    fn client(server: &MockServer) -> Arc<ApiClient> {
        let auth = Arc::new(StaticAuthProvider::new("t"));
        let mut options = ApiClientOptions::new(server.uri());
        options.max_backoff = Duration::from_millis(10);
        Arc::new(ApiClient::new(options, auth).unwrap())
    }

    fn upload(upload_id: &str, part_size_bytes: u64) -> MultipartUpload {
        MultipartUpload {
            upload_id: upload_id.to_string(),
            part_size_bytes,
            crc32c: None,
        }
    }

    fn crc_base64(contents: &[u8]) -> String {
        BASE64.encode(crc32c::crc32c(contents).to_be_bytes())
    }

    /// Seed `trace-1` of a start-notified recording `rec-1`, written and queued
    /// with `uploads`, and write `on_disk` into its trace directory. Returns the
    /// recording index.
    async fn seed_trace(
        store: &SqliteStateStore,
        recordings_root: &std::path::Path,
        (data_type, data_type_name): (&str, &str),
        uploads: &MultipartUploads,
        on_disk: &[(&str, &[u8])],
    ) -> i64 {
        let recording_index = store
            .create_recording(NewRecording::default())
            .await
            .unwrap()
            .recording_index;
        store
            .mark_recording_start_notified(recording_index, "rec-1")
            .await
            .unwrap();
        store
            .create_trace(
                recording_index,
                "trace-1",
                Some(data_type),
                Some(data_type_name),
            )
            .await
            .unwrap();
        // On-disk artefacts are keyed by the local `recording_index`.
        let dir = TracePath::new(
            recording_index.to_string(),
            data_type,
            "trace-1".to_string(),
        )
        .directory(recordings_root);
        std::fs::create_dir_all(&dir).unwrap();
        for (basename, contents) in on_disk {
            std::fs::write(dir.join(basename), contents).unwrap();
        }
        store
            .update_trace(
                "trace-1",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    upload_status: Some(TraceUploadStatus::Queued),
                    multipart_uploads: Some(serde_json::to_string(uploads).unwrap()),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();
        recording_index
    }

    /// Seed a joint trace whose `trace.json` holds `contents` and uploads as
    /// `upload-1` in parts of `part_size_bytes`.
    async fn seed_json_trace(
        store: &SqliteStateStore,
        recordings_root: &std::path::Path,
        contents: &[u8],
        part_size_bytes: u64,
    ) -> i64 {
        let uploads = MultipartUploads::from([(
            JSON_FILEPATH.to_string(),
            upload("upload-1", part_size_bytes),
        )]);
        seed_trace(
            store,
            recordings_root,
            ("JOINT_POSITIONS", "arm"),
            &uploads,
            &[(TRACE_JSON_FILENAME, contents)],
        )
        .await
    }

    /// Answer the parts route with a URL on `server` for every requested part,
    /// `/part/{upload_id}/{part_number}`, reporting `uploaded_parts` as held.
    async fn mount_parts_route(server: &MockServer, uploaded_parts: &[u32]) {
        let base_url = server.uri();
        let uploaded_parts = uploaded_parts.to_vec();
        Mock::given(method("POST"))
            .and(path(PARTS_ROUTE))
            .respond_with(move |request: &Request| {
                let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
                let uploads: Vec<serde_json::Value> = body["uploads"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|upload| {
                        let upload_id = upload["upload_id"].as_str().unwrap();
                        let part_urls: serde_json::Map<String, serde_json::Value> = upload
                            ["part_numbers"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|number| {
                                let url = format!("{base_url}/part/{upload_id}/{number}");
                                (number.to_string(), url.into())
                            })
                            .collect();
                        serde_json::json!({
                            "upload_id": upload_id,
                            "part_urls": part_urls,
                            "uploaded_parts": uploaded_parts,
                        })
                    })
                    .collect();
                ResponseTemplate::new(200).set_body_json(
                    serde_json::json!({"uploads": uploads, "missing_upload_ids": []}),
                )
            })
            .mount(server)
            .await;
    }

    /// Accept every part PUT to `/part/…`.
    async fn mount_part_store(server: &MockServer) {
        Mock::given(method("PUT"))
            .and(path_regex("^/part/"))
            .respond_with(ResponseTemplate::new(200))
            .mount(server)
            .await;
    }

    /// Paths of the part PUTs `server` received, in order.
    async fn part_puts(server: &MockServer) -> Vec<String> {
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .filter(|request| request.method.as_str() == "PUT")
            .map(|request| request.url.path().to_string())
            .collect()
    }

    /// Drive `upload_single` for `trace-1` with the standard single-trace
    /// harness: an owned store handle, a fresh trace-event writer (held alive
    /// for the await), the `recordings_root`, and org `org-1`. The caller owns
    /// `bus`/`status_tx` so it can observe the emitted events and status updates.
    async fn run_upload_single(
        store: &SqliteStateStore,
        recordings_root: std::path::PathBuf,
        api: &Arc<ApiClient>,
        bus: &EventBus,
        status_tx: &mpsc::UnboundedSender<StatusUpdate>,
    ) {
        let store_arc = Arc::new(store.clone());
        let (trace_writer, _trace_writer_owner) =
            crate::state::trace_event_database_writer::spawn(store_arc.clone());
        let recordings_root = Arc::new(recordings_root);
        upload_single(
            &store_arc,
            &trace_writer,
            bus,
            api,
            &recordings_root,
            &org_rx(Some("org-1")),
            status_tx,
            "trace-1",
        )
        .await;
    }

    /// Upload `trace-1` against `server` and return its row and the completion
    /// status update, if one was sent.
    async fn upload_trace(
        server: &MockServer,
        store: &SqliteStateStore,
        recordings_root: std::path::PathBuf,
    ) -> (TraceRecord, Option<StatusUpdate>) {
        let (status_tx, mut status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(
            store,
            recordings_root,
            &client(server),
            &EventBus::new(),
            &status_tx,
        )
        .await;
        let mut completion = None;
        while let Ok(update) = status_rx.try_recv() {
            if update.completed {
                completion = Some(update);
            }
        }
        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        (trace, completion)
    }

    #[tokio::test]
    async fn every_part_is_sent_and_the_completion_carries_the_file_crc() {
        let server = MockServer::start().await;
        mount_parts_route(&server, &[]).await;
        mount_part_store(&server).await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let payload = b"ten bytes!";
        seed_json_trace(&store, &recordings_root, payload, 4).await;

        let (trace, completion) = upload_trace(&server, &store, recordings_root).await;

        assert_eq!(trace.upload_status, TraceUploadStatus::Uploaded);
        assert_eq!(trace.bytes_uploaded, payload.len() as i64);
        assert_eq!(
            part_puts(&server).await,
            ["/part/upload-1/1", "/part/upload-1/2", "/part/upload-1/3"]
        );
        let requests = server.received_requests().await.unwrap();
        assert!(
            requests
                .iter()
                .filter(|request| request.method.as_str() == "PUT")
                .all(|request| !request.headers.contains_key("authorization")),
            "a presigned PUT carries no Authorization header"
        );
        let crc32c = crc_base64(payload);
        assert_eq!(
            trace.decode_uploads().unwrap().unwrap()[JSON_FILEPATH].crc32c,
            Some(crc32c.clone()),
            "the CRC is persisted for a restart"
        );
        assert_eq!(
            completion.expect("a completion is sent").uploaded_files,
            vec![CompletedFileUpload {
                upload_id: "upload-1".to_string(),
                crc32c,
            }]
        );
    }

    /// After a restart the store holds `lossy.mp4` whole and part 1 of
    /// `trace.json`: only the rest of `trace.json` is sent, and the CRCs still
    /// cover every byte.
    #[tokio::test]
    async fn a_restart_sends_only_the_parts_the_store_lacks() {
        let server = MockServer::start().await;
        mount_parts_route(&server, &[1]).await;
        mount_part_store(&server).await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let video = b"fake-mp4-bytes";
        let sidecar = b"[1, 2, 3]";
        let mut lossy = upload("upload-lossy", 4);
        lossy.crc32c = Some(crc_base64(video));
        let uploads = MultipartUploads::from([
            (format!("RGB_IMAGES/cam0/{LOSSY_VIDEO_FILENAME}"), lossy),
            (
                format!("RGB_IMAGES/cam0/{TRACE_JSON_FILENAME}"),
                upload("upload-json", 4),
            ),
        ]);
        seed_trace(
            &store,
            &recordings_root,
            ("RGB_IMAGES", "cam0"),
            &uploads,
            &[
                (LOSSY_VIDEO_FILENAME, video.as_slice()),
                (TRACE_JSON_FILENAME, sidecar.as_slice()),
            ],
        )
        .await;

        let (trace, completion) = upload_trace(&server, &store, recordings_root).await;

        assert_eq!(trace.upload_status, TraceUploadStatus::Uploaded);
        assert_eq!(
            part_puts(&server).await,
            ["/part/upload-json/2", "/part/upload-json/3"]
        );
        let mut uploaded_files = completion.expect("a completion is sent").uploaded_files;
        uploaded_files.sort_by(|left, right| left.upload_id.cmp(&right.upload_id));
        assert_eq!(
            uploaded_files,
            vec![
                CompletedFileUpload {
                    upload_id: "upload-json".to_string(),
                    crc32c: crc_base64(sidecar),
                },
                CompletedFileUpload {
                    upload_id: "upload-lossy".to_string(),
                    crc32c: crc_base64(video),
                },
            ]
        );
    }

    #[tokio::test]
    async fn an_expired_part_url_is_fetched_again() {
        let gcs_expired_token = ResponseTemplate::new(400).set_body_string(
            "<Error><Code>ExpiredToken</Code><Message>The provided token has expired.</Message></Error>",
        );
        for expired in [ResponseTemplate::new(403), gcs_expired_token] {
            let server = MockServer::start().await;
            mount_parts_route(&server, &[]).await;
            Mock::given(method("PUT"))
                .and(path("/part/upload-1/1"))
                .respond_with(expired)
                .up_to_n_times(1)
                .mount(&server)
                .await;
            mount_part_store(&server).await;
            let (store, tempdir) = open_store().await;
            let recordings_root = tempdir.path().join("recordings");
            seed_json_trace(&store, &recordings_root, b"payload", 16 * 1024 * 1024).await;

            let (trace, _) = upload_trace(&server, &store, recordings_root).await;

            assert_eq!(trace.upload_status, TraceUploadStatus::Uploaded);
            assert_eq!(
                part_puts(&server).await,
                ["/part/upload-1/1", "/part/upload-1/1"]
            );
            let parts_requests = server
                .received_requests()
                .await
                .unwrap()
                .iter()
                .filter(|request| request.url.path() == PARTS_ROUTE)
                .count();
            assert_eq!(parts_requests, 2, "the expired URL fetches fresh URLs");
        }
    }

    #[tokio::test]
    async fn a_missing_upload_sends_the_trace_back_to_registration() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(PARTS_ROUTE))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "uploads": [],
                "missing_upload_ids": ["upload-1"]
            })))
            .expect(1)
            .mount(&server)
            .await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        seed_json_trace(&store, &recordings_root, b"payload", 16 * 1024 * 1024).await;
        store
            .update_trace(
                "trace-1",
                TraceUpdate {
                    registration_status: Some(TraceRegistrationStatus::Registered),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();

        let (trace, completion) = upload_trace(&server, &store, recordings_root).await;

        assert_eq!(trace.registration_status, TraceRegistrationStatus::Pending);
        assert_eq!(trace.upload_status, TraceUploadStatus::Pending);
        assert_eq!(trace.multipart_uploads, None, "the old uploads are cleared");
        assert!(completion.is_none());
        assert!(part_puts(&server).await.is_empty());
    }

    #[tokio::test]
    async fn a_file_needing_more_than_10000_parts_fails_the_trace() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(PARTS_ROUTE))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        seed_json_trace(&store, &recordings_root, &[b'x'; 10_001], 1).await;

        let (trace, completion) = upload_trace(&server, &store, recordings_root).await;

        assert_eq!(trace.upload_status, TraceUploadStatus::Failed);
        assert!(
            trace
                .error_message
                .as_deref()
                .is_some_and(|message| message.contains("10001 parts")),
            "the error names the part count: {:?}",
            trace.error_message
        );
        assert!(completion.is_none());
    }

    #[tokio::test]
    async fn transient_5xx_part_is_retried_then_uploaded() {
        let server = MockServer::start().await;
        mount_parts_route(&server, &[]).await;
        Mock::given(method("PUT"))
            .and(path_regex("^/part/"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        mount_part_store(&server).await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        seed_json_trace(&store, &recordings_root, b"transient-ok", 16 * 1024 * 1024).await;

        let (trace, _) = upload_trace(&server, &store, recordings_root).await;

        assert_eq!(trace.upload_status, TraceUploadStatus::Uploaded);
    }

    #[tokio::test]
    async fn hard_4xx_part_error_rolls_trace_to_retrying() {
        // A non-retryable 4xx is a hard error: the uploader rolls the trace to
        // Retrying and the next drain resumes it.
        let server = MockServer::start().await;
        mount_parts_route(&server, &[]).await;
        Mock::given(method("PUT"))
            .and(path_regex("^/part/"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .expect(1)
            .mount(&server)
            .await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        seed_json_trace(&store, &recordings_root, b"hard-error", 16 * 1024 * 1024).await;

        let (trace, completion) = upload_trace(&server, &store, recordings_root).await;

        assert_eq!(trace.upload_status, TraceUploadStatus::Retrying);
        assert!(completion.is_none());
    }

    #[tokio::test]
    async fn missing_data_type_emits_terminal_failure_and_unblocks_progress() {
        // A trace registered without a data_type cannot be located on
        // disk. The uploader must mark it Failed *and* emit an
        // UploadComplete so the progress reporter's "all settled" gate
        // moves on — otherwise the recording sits as `pending` forever.
        let server = MockServer::start().await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");

        let recording = store
            .create_recording(NewRecording::default())
            .await
            .unwrap();
        // Insert directly with NULL data_type so the uploader hits the
        // missing-data-type branch.
        store
            .create_trace(recording.recording_index, "trace-1", None, None)
            .await
            .unwrap();
        let uploads = MultipartUploads::from([("dummy".to_string(), upload("upload-1", 1))]);
        store
            .update_trace(
                "trace-1",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    upload_status: Some(TraceUploadStatus::Queued),
                    multipart_uploads: Some(serde_json::to_string(&uploads).unwrap()),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();

        let bus = EventBus::new();
        let mut subscriber = bus.subscribe();
        let (status_tx, mut status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(&store, recordings_root, &client(&server), &bus, &status_tx).await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Failed);
        // UploadComplete fires so the recording's progress report can
        // proceed — otherwise a stray no-data-type trace would deadlock
        // the whole recording.
        match subscriber.try_recv() {
            Ok(DaemonEvent::UploadComplete { trace_id, .. }) => {
                assert_eq!(trace_id, "trace-1");
            }
            other => panic!("expected UploadComplete event, got {other:?}"),
        }
        assert!(!completed_status_sent(&mut status_rx));
    }

    #[tokio::test]
    async fn empty_uploads_fail_the_trace() {
        // An empty map would upload no files, so the trace fails without a
        // completed status, and UploadComplete still unblocks the progress gate.
        let server = MockServer::start().await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        seed_trace(
            &store,
            &recordings_root,
            ("JOINT_POSITIONS", "arm"),
            &MultipartUploads::new(),
            &[],
        )
        .await;

        let bus = EventBus::new();
        let mut subscriber = bus.subscribe();
        let (status_tx, mut status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(&store, recordings_root, &client(&server), &bus, &status_tx).await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Failed);
        assert!(matches!(
            subscriber.try_recv(),
            Ok(DaemonEvent::UploadComplete { .. })
        ));
        assert!(!completed_status_sent(&mut status_rx));
    }

    #[tokio::test]
    async fn no_uploads_leaves_trace_queued() {
        // Ready-for-upload but no stored uploads (registration hasn't
        // persisted them yet): the uploader returns without touching the trace
        // so a later drain retries once they land.
        let server = MockServer::start().await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let recording = store
            .create_recording(NewRecording::default())
            .await
            .unwrap();
        store
            .create_trace(
                recording.recording_index,
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
                    upload_status: Some(TraceUploadStatus::Queued),
                    ..TraceUpdate::default() // no multipart_uploads
                },
            )
            .await
            .unwrap();

        let (trace, _) = upload_trace(&server, &store, recordings_root).await;
        assert_eq!(
            trace.upload_status,
            TraceUploadStatus::Queued,
            "no uploads → trace untouched for retry"
        );
    }

    #[tokio::test]
    async fn missing_artefact_marks_trace_failed() {
        // Uploads + data_type present, but the on-disk file is gone. The
        // uploader marks the trace Failed and emits UploadComplete so the
        // recording's progress gate isn't wedged on a vanished file, but never
        // reports it completed to the backend.
        let server = MockServer::start().await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let uploads = MultipartUploads::from([(
            JSON_FILEPATH.to_string(),
            upload("upload-1", 16 * 1024 * 1024),
        )]);
        // Deliberately do NOT create the on-disk artefact.
        seed_trace(
            &store,
            &recordings_root,
            ("JOINT_POSITIONS", "arm"),
            &uploads,
            &[],
        )
        .await;

        let bus = EventBus::new();
        let mut subscriber = bus.subscribe();
        let (status_tx, mut status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(&store, recordings_root, &client(&server), &bus, &status_tx).await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Failed);
        assert!(matches!(
            subscriber.try_recv(),
            Ok(DaemonEvent::UploadComplete { .. })
        ));
        assert!(!completed_status_sent(&mut status_rx));
    }

    /// Dispatch `trace-1` through `spawn_upload_task` with `aborts`.
    async fn dispatch_upload(
        store: &SqliteStateStore,
        recordings_root: std::path::PathBuf,
        api: &Arc<ApiClient>,
        aborts: &UploadAborts,
    ) -> JoinSet<String> {
        let store_arc = Arc::new(store.clone());
        let (trace_writer, owner) =
            crate::state::trace_event_database_writer::spawn(store_arc.clone());
        // The writer must outlive the upload task.
        Box::leak(Box::new(owner));
        let (status_tx, _status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        let mut in_flight: JoinSet<String> = JoinSet::new();
        spawn_upload_task(
            &store_arc,
            &trace_writer,
            &EventBus::new(),
            api,
            &Arc::new(recordings_root),
            &org_rx(Some("org-1")),
            &status_tx,
            &Arc::new(Semaphore::new(MAX_CONCURRENT_UPLOADS)),
            &mut in_flight,
            &mut HashSet::new(),
            aborts,
            "trace-1".to_string(),
        );
        in_flight
    }

    async fn wait_for_put(server: &MockServer, upload_path: &str) {
        timeout(Duration::from_secs(10), async {
            loop {
                let received = server.received_requests().await.unwrap_or_default();
                if received
                    .iter()
                    .any(|request| request.url.path() == upload_path)
                {
                    break;
                }
                sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the upload PUT must be in flight within 10s");
    }

    #[tokio::test]
    async fn an_abort_cuts_off_an_in_flight_upload() {
        let server = MockServer::start().await;
        mount_parts_route(&server, &[]).await;
        // Never realistically returns, so an upload that is not aborted times
        // out rather than passing by luck.
        Mock::given(method("PUT"))
            .and(path_regex("^/part/"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(120)))
            .mount(&server)
            .await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let recording_index =
            seed_json_trace(&store, &recordings_root, b"payload", 16 * 1024 * 1024).await;

        let aborts = UploadAborts::default();
        let mut in_flight =
            dispatch_upload(&store, recordings_root, &client(&server), &aborts).await;
        wait_for_put(&server, "/part/upload-1/1").await;
        aborts.abort(recording_index);

        let joined = timeout(Duration::from_secs(10), in_flight.join_next())
            .await
            .expect("the abort must end the upload well inside the 120s PUT");
        assert!(matches!(joined, Some(Ok(trace_id)) if trace_id == "trace-1"));
        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_ne!(trace.upload_status, TraceUploadStatus::Uploaded);
    }

    #[tokio::test]
    async fn a_trace_of_a_not_found_recording_is_never_uploaded() {
        let server = MockServer::start().await;
        mount_parts_route(&server, &[]).await;
        Mock::given(method("PUT"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let recording_index =
            seed_json_trace(&store, &recordings_root, b"payload", 16 * 1024 * 1024).await;
        // Not found after this trace was dispatched, before its task ran.
        store
            .mark_recording_not_found(recording_index)
            .await
            .unwrap();

        let mut in_flight = dispatch_upload(
            &store,
            recordings_root,
            &client(&server),
            &UploadAborts::default(),
        )
        .await;
        timeout(Duration::from_secs(5), in_flight.join_next())
            .await
            .expect("the task returns without uploading");
        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Failed);
        // The `expect(0)` mock asserts no PUT was sent.
    }

    /// The uploader loop, not each task, turns the bus events into aborts.
    #[tokio::test]
    async fn the_uploader_aborts_uploads_on_cancel_and_not_found_events() {
        for not_found in [false, true] {
            let server = MockServer::start().await;
            mount_parts_route(&server, &[]).await;
            // Completes after a short delay, so an upload that is not aborted
            // ends up `Uploaded`.
            Mock::given(method("PUT"))
                .and(path_regex("^/part/"))
                .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(1500)))
                .mount(&server)
                .await;
            let (store, tempdir) = open_store().await;
            let recordings_root = tempdir.path().join("recordings");
            let recording_index =
                seed_json_trace(&store, &recordings_root, b"hello world", 16 * 1024 * 1024).await;

            let bus = EventBus::new();
            let (status_tx, _status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
            let (shutdown_tx, shutdown_rx) = broadcast::channel::<ShutdownSignal>(1);
            let store_arc = Arc::new(store.clone());
            let (trace_writer, _owner) =
                crate::state::trace_event_database_writer::spawn(store_arc.clone());
            let handle = spawn_uploader(
                store.clone(),
                trace_writer,
                bus.clone(),
                client(&server),
                Arc::new(recordings_root),
                org_rx(Some("org-1")),
                status_tx,
                shutdown_rx,
            );
            sleep(Duration::from_millis(50)).await;
            bus.publish(DaemonEvent::ConnectionStateChanged(ConnectionState::Up));
            wait_for_put(&server, "/part/upload-1/1").await;

            if not_found {
                store
                    .mark_recording_not_found(recording_index)
                    .await
                    .unwrap();
                bus.publish(DaemonEvent::RecordingNotFound { recording_index });
            } else {
                store
                    .cancel_recording(recording_index, RecordingTimestamp::observed_at(0))
                    .await
                    .unwrap();
                bus.publish(DaemonEvent::RecordingCancelled { recording_index });
            }
            sleep(Duration::from_millis(2500)).await;

            let trace = store.get_trace("trace-1").await.unwrap().unwrap();
            assert_eq!(
                trace.upload_status,
                TraceUploadStatus::Failed,
                "the upload must be aborted, not completed (not_found = {not_found})"
            );
            let _ = shutdown_tx.send(ShutdownSignal::Sigterm);
            handle.join().await;
        }
    }

    #[tokio::test]
    async fn spawn_upload_task_skips_already_dispatched_trace() {
        // The dedup guard: a trace already in `in_flight_ids` must not be
        // dispatched twice (a drain triggered before the task marks itself
        // Uploading would otherwise double-upload).
        let server = MockServer::start().await;
        let (store, tempdir) = open_store().await;
        let recordings_root = Arc::new(tempdir.path().join("recordings"));
        let api = client(&server);
        let bus = EventBus::new();
        let (status_tx, _status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        let store_arc = Arc::new(store.clone());
        let (trace_writer, _owner) =
            crate::state::trace_event_database_writer::spawn(store_arc.clone());
        let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_UPLOADS));
        let mut in_flight: JoinSet<String> = JoinSet::new();
        let mut in_flight_ids: HashSet<String> = HashSet::new();
        in_flight_ids.insert("trace-1".to_string());

        spawn_upload_task(
            &store_arc,
            &trace_writer,
            &bus,
            &api,
            &recordings_root,
            &org_rx(Some("org-1")),
            &status_tx,
            &semaphore,
            &mut in_flight,
            &mut in_flight_ids,
            &UploadAborts::default(),
            "trace-1".to_string(),
        );

        assert!(in_flight.is_empty(), "a duplicate dispatch is skipped");
        assert_eq!(in_flight_ids.len(), 1);
    }

    #[tokio::test]
    async fn spawn_upload_task_skips_when_semaphore_full() {
        // No permit available → the task is not dispatched and the trace is not
        // marked in-flight, so a later drain (which frees permits) retries it.
        let server = MockServer::start().await;
        let (store, tempdir) = open_store().await;
        let recordings_root = Arc::new(tempdir.path().join("recordings"));
        let api = client(&server);
        let bus = EventBus::new();
        let (status_tx, _status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        let store_arc = Arc::new(store.clone());
        let (trace_writer, _owner) =
            crate::state::trace_event_database_writer::spawn(store_arc.clone());
        let semaphore = Arc::new(Semaphore::new(0)); // no permits
        let mut in_flight: JoinSet<String> = JoinSet::new();
        let mut in_flight_ids: HashSet<String> = HashSet::new();

        spawn_upload_task(
            &store_arc,
            &trace_writer,
            &bus,
            &api,
            &recordings_root,
            &org_rx(Some("org-1")),
            &status_tx,
            &semaphore,
            &mut in_flight,
            &mut in_flight_ids,
            &UploadAborts::default(),
            "trace-1".to_string(),
        );

        assert!(in_flight.is_empty(), "no permit → nothing dispatched");
        assert!(
            in_flight_ids.is_empty(),
            "no permit → id not marked in-flight"
        );
    }

    /// Upload a written RGB trace for `cam0` with an upload for `lossy.mp4` and
    /// `trace.json`, against a store that accepts every part. Only the files in
    /// `on_disk` are written to the trace directory.
    async fn upload_rgb_trace(
        server: &MockServer,
        on_disk: &[(&str, &[u8])],
    ) -> (SqliteStateStore, TempDir, Option<StatusUpdate>) {
        mount_parts_route(server, &[]).await;
        mount_part_store(server).await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let uploads = MultipartUploads::from([
            (
                format!("RGB_IMAGES/cam0/{LOSSY_VIDEO_FILENAME}"),
                upload("upload-lossy", 16 * 1024 * 1024),
            ),
            (
                format!("RGB_IMAGES/cam0/{TRACE_JSON_FILENAME}"),
                upload("upload-json", 16 * 1024 * 1024),
            ),
        ]);
        seed_trace(
            &store,
            &recordings_root,
            ("RGB_IMAGES", "cam0"),
            &uploads,
            on_disk,
        )
        .await;
        let (_, completion) = upload_trace(server, &store, recordings_root).await;
        (store, tempdir, completion)
    }

    fn completed_status_sent(status_rx: &mut mpsc::UnboundedReceiver<StatusUpdate>) -> bool {
        let mut completed = false;
        while let Ok(update) = status_rx.try_recv() {
            completed |= update.completed;
        }
        completed
    }

    #[tokio::test]
    async fn every_registered_file_is_uploaded_before_uploaded() {
        let server = MockServer::start().await;
        let (store, _tempdir, completion) = upload_rgb_trace(
            &server,
            &[
                (LOSSY_VIDEO_FILENAME, b"fake-mp4-bytes"),
                (TRACE_JSON_FILENAME, b"[]"),
            ],
        )
        .await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Uploaded);
        assert_eq!(completion.expect("completed").uploaded_files.len(), 2);
        let mut puts = part_puts(&server).await;
        puts.sort();
        assert_eq!(puts, ["/part/upload-json/1", "/part/upload-lossy/1"]);
    }

    #[tokio::test]
    async fn zero_byte_sidecar_fails_trace_without_completion() {
        let server = MockServer::start().await;
        let (store, _tempdir, completion) = upload_rgb_trace(
            &server,
            &[
                (LOSSY_VIDEO_FILENAME, b"fake-mp4-bytes"),
                (TRACE_JSON_FILENAME, b""),
            ],
        )
        .await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Failed);
        assert!(completion.is_none());
        assert!(!part_puts(&server)
            .await
            .contains(&"/part/upload-json/1".to_string()));
    }

    #[tokio::test]
    async fn missing_sidecar_fails_trace_without_completion() {
        let server = MockServer::start().await;
        let (store, _tempdir, completion) =
            upload_rgb_trace(&server, &[(LOSSY_VIDEO_FILENAME, b"fake-mp4-bytes")]).await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Failed);
        assert!(completion.is_none());
    }
}
