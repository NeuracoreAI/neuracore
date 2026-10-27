//! Resumable file uploader coordinator.
//!
//! Subscribes to [`DaemonEvent::ReadyForUpload`] (and re-scans the
//! state store on startup for any traces already in the registered/queued
//! state). For each on-disk artefact the coordinator PUTs `CHUNK_SIZE` (16 MiB)
//! chunks to the GCS resumable session URI persisted by the registration coordinator,
//! handling 308-continue, 410-session-expired, and 401-auth-refresh
//! transitions. On completion the trace is marked `Uploaded` and the upload
//! sub-system publishes `UploadComplete` for the status updater.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use tokio::sync::{broadcast, watch, Semaphore};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::{interval, MissedTickBehavior};

use super::status::StatusUpdate;
use super::upload_transfer::{upload_one_file, UploadFileError, UploadFileOutcome};
use crate::api::ApiClient;
use crate::cloud::cloud_files::content_type_for_filename;
use crate::cloud::OrgIdRx;
use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::store::TraceUpdate;
use crate::state::{
    ConnectionState, DaemonEvent, EventBus, SqliteStateStore, StateStore, TraceRecord,
    TraceUploadStatus, TraceWriteHandle,
};
use crate::storage::paths::TracePath;

/// Default number of traces uploading concurrently. A trade-off: more slots
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
    upload_workers: usize,
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
        let semaphore = Arc::new(Semaphore::new(upload_workers));
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
    let session_uris = match parse_session_uris(&trace) {
        Some(uris) => uris,
        None => return,
    };
    if session_uris.is_empty() {
        // Every trace registers at least `trace.json`, so an empty map means
        // the files would never be uploaded.
        mark_failed_and_emit(store, bus, &trace, "no upload session URIs").await;
        return;
    }

    // Resolve the cloud `recording_id` (needed for the resumable-upload-url
    // refresh) before we touch the trace's upload state. A None here means
    // registration hasn't minted the cloud id yet — leave the trace in its
    // queued/retrying state and skip; a later drain re-enters once it lands.
    let Some(recording_id) = recording_cloud_id(store, trace.recording_index).await else {
        tracing::warn!(
            trace_id,
            recording_index = trace.recording_index,
            "recording has no cloud recording_id yet; deferring upload"
        );
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
            "file_count": session_uris.len(),
        }),
    );

    // Upload each on-disk artefact under its session URI and persist the
    // refreshed URI back into the same slot (by index) for resume on retry.
    let mut total_uploaded: i64 = 0;
    let mut session_uris = session_uris;
    for index in 0..session_uris.len() {
        let (filename, session_uri) = session_uris[index].clone();
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
            crate::perf_events::emit(
                "upload",
                "failed",
                Some(trace.recording_index),
                Some(trace_id),
                Some(upload_started.elapsed()),
                serde_json::json!({
                    "outcome": "error",
                    "reason": if local_len.is_none() { "missing_artefact" } else { "empty_artefact" },
                    "filename": file_basename(&filename),
                    "bytes_uploaded": total_uploaded,
                }),
            );
            return;
        }

        // `content_type` here drives the GCS-side metadata when we re-acquire a
        // session URI on 410. Use the same filename→type mapping registration
        // used (`cloud_files::content_type_for_filename`) so the refresh can't
        // disagree with what was originally registered.
        let content_type = content_type_for_filename(&filename);
        let outcome = upload_one_file(
            client,
            trace_writer,
            bus,
            org_rx,
            status_tx,
            &trace,
            &recording_id,
            &local_path,
            &filename,
            content_type,
            session_uri,
        )
        .await;
        match outcome {
            Ok(UploadFileOutcome {
                bytes_uploaded,
                final_session_uri,
            }) => {
                total_uploaded = total_uploaded.saturating_add(bytes_uploaded);
                // Persist the (possibly refreshed) URI so a subsequent
                // restart resumes from the right session, even if the
                // refresh path fired mid-stream.
                if let Some(new_uri) = final_session_uri {
                    session_uris[index].1 = new_uri;
                    persist_session_uris(store, trace_id, &session_uris).await;
                }
            }
            Err(UploadFileError::RecordingCancelled) => {
                emit_upload_failed(
                    &trace,
                    &filename,
                    upload_started,
                    total_uploaded,
                    "recording_cancelled",
                );
                // Before the discard publishes the event that aborts this task.
                mark_failed_and_emit(store, bus, &trace, "recording cancelled").await;
                super::discard_recording_locally(store, bus, trace.recording_index).await;
                return;
            }
            Err(UploadFileError::RecordingNotFound) => {
                emit_upload_failed(
                    &trace,
                    &filename,
                    upload_started,
                    total_uploaded,
                    "recording_not_found",
                );
                super::stop_uploading_not_found_recording(
                    store,
                    bus,
                    trace.recording_index,
                    &recording_id,
                )
                .await;
                return;
            }
            Err(UploadFileError::Retry(error)) => {
                tracing::warn!(%error, trace_id, "upload failed; rolling back to retrying");
                crate::perf_events::emit(
                    "upload",
                    "failed",
                    Some(trace.recording_index),
                    Some(trace_id),
                    Some(upload_started.elapsed()),
                    serde_json::json!({
                        "outcome": "error",
                        "reason": "transfer_failed",
                        "filename": file_basename(&filename),
                        "bytes_uploaded": total_uploaded,
                    }),
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
            "file_count": session_uris.len(),
            "bytes_uploaded": total_uploaded,
        }),
    );
    finalise_upload(store, bus, status_tx, &trace, total_uploaded).await;
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

async fn persist_session_uris(
    store: &Arc<SqliteStateStore>,
    trace_id: &str,
    uris: &[(String, String)],
) {
    let map: HashMap<&str, &str> = uris
        .iter()
        .map(|(filename, uri)| (filename.as_str(), uri.as_str()))
        .collect();
    let serialised = match serde_json::to_string(&map) {
        Ok(serialised) => serialised,
        Err(error) => {
            tracing::warn!(%error, trace_id, "failed to serialise refreshed session URIs");
            return;
        }
    };
    let update = TraceUpdate {
        upload_session_uris: Some(serialised),
        ..TraceUpdate::default()
    };
    if let Err(error) = store.update_trace(trace_id, update).await {
        tracing::warn!(%error, trace_id, "failed to persist refreshed session URIs");
    }
}

fn parse_session_uris(trace: &TraceRecord) -> Option<Vec<(String, String)>> {
    let Some(serialised) = &trace.upload_session_uris else {
        tracing::warn!(
            trace_id = trace.trace_id,
            "trace ready-for-upload but no session URIs stored"
        );
        return None;
    };
    match serde_json::from_str::<HashMap<String, String>>(serialised) {
        Ok(map) => Some(map.into_iter().collect()),
        Err(error) => {
            tracing::warn!(%error, trace_id = trace.trace_id, "failed to decode stored session URIs");
            None
        }
    }
}

fn file_basename(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some((_, tail)) => tail,
        None => path,
    }
}

/// Resolve the cloud `recording_id` (the backend handle every cloud URL needs)
/// from its local `recording_index`. `None` when registration hasn't minted
/// the cloud id yet, or the row is missing.
async fn recording_cloud_id(store: &Arc<SqliteStateStore>, recording_index: i64) -> Option<String> {
    match store.get_recording(recording_index).await {
        Ok(Some(row)) => row.recording_id,
        Ok(None) => None,
        Err(error) => {
            tracing::warn!(%error, recording_index, "uploader could not read recording cloud id");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::auth::StaticAuthProvider;
    use crate::api::client::ApiClientOptions;
    use crate::state::store::{NewRecording, TraceUpdate};
    use crate::state::{TraceUploadStatus, TraceWriteStatus};
    use crate::storage::paths::{LOSSY_VIDEO_FILENAME, TRACE_JSON_FILENAME};
    use base64::engine::general_purpose::STANDARD as BASE64;
    use base64::Engine;
    use std::collections::HashMap;
    use std::time::Duration;
    use tempfile::TempDir;
    use tokio::sync::mpsc;
    use tokio::time::{sleep, timeout};
    use wiremock::matchers::{header, method, path, path_regex};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

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

    #[allow(clippy::too_many_arguments)]
    async fn seed_ready_trace(
        store: &SqliteStateStore,
        recordings_root: &std::path::Path,
        cloud_recording_id: &str,
        trace_id: &str,
        data_type: &str,
        data_type_name: &str,
        session_uri: &str,
        contents: &[u8],
    ) -> (i64, std::path::PathBuf) {
        let recording = store
            .create_recording(NewRecording::default())
            .await
            .unwrap();
        let recording_index = recording.recording_index;
        // Stamp the cloud `recording_id` so the uploader's cloud-id resolution
        // and the resumable-upload-url refresh see the same id the wiremock
        // expectations assert on.
        store
            .mark_recording_start_notified(recording_index, cloud_recording_id)
            .await
            .unwrap();
        store
            .create_trace(
                recording_index,
                trace_id,
                Some(data_type),
                Some(data_type_name),
            )
            .await
            .unwrap();
        // On-disk artefacts are keyed by the local `recording_index`.
        let dir = TracePath::new(recording_index.to_string(), data_type, trace_id.to_string())
            .directory(recordings_root);
        std::fs::create_dir_all(&dir).unwrap();
        let local = dir.join(TRACE_JSON_FILENAME);
        std::fs::write(&local, contents).unwrap();
        let mut uris = HashMap::new();
        uris.insert(
            format!("{data_type}/{data_type_name}/{TRACE_JSON_FILENAME}"),
            session_uri.to_string(),
        );
        store
            .update_trace(
                trace_id,
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    upload_status: Some(TraceUploadStatus::Queued),
                    upload_session_uris: Some(serde_json::to_string(&uris).unwrap()),
                    total_bytes: Some(contents.len() as i64),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();
        (recording_index, local)
    }

    /// Drive `upload_single` for `trace_id` with the standard single-trace
    /// harness: an owned store handle, a fresh trace-event writer (held alive
    /// for the await), the `recordings_root`, and org `org-1`. The caller owns
    /// `bus`/`status_tx` so it can observe the emitted events and status updates.
    async fn run_upload_single(
        store: &SqliteStateStore,
        recordings_root: std::path::PathBuf,
        api: &Arc<ApiClient>,
        bus: &EventBus,
        status_tx: &mpsc::UnboundedSender<StatusUpdate>,
        trace_id: &str,
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
            trace_id,
        )
        .await;
    }

    #[tokio::test]
    async fn bad_server_checksum_marks_trace_retrying() {
        // Server returns a deliberately wrong CRC32C (0) — the real payload's
        // checksum is non-zero, so the guard must reject the upload and roll
        // the trace back to `retrying`
        // (the registration coordinator's recovery sweep takes it from
        // there). The doc-claimed happy path is covered by
        // `uploader_marks_uploaded_when_checksum_matches` below.
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/upload/abc"))
            .respond_with(|_req: &Request| {
                ResponseTemplate::new(200).insert_header("X-Goog-Hash", "crc32c=AAAAAA==")
            })
            .expect(1)
            .mount(&server)
            .await;

        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let payload = b"some-bytes";
        let (_recording_index, _) = seed_ready_trace(
            &store,
            &recordings_root,
            "rec-1",
            "trace-1",
            "JOINT_POSITIONS",
            "arm",
            &format!("{}/upload/abc", server.uri()),
            payload,
        )
        .await;

        let api = client(&server);
        let bus = EventBus::new();
        let (status_tx, mut status_rx) = mpsc::unbounded_channel::<StatusUpdate>();

        run_upload_single(&store, recordings_root, &api, &bus, &status_tx, "trace-1").await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Retrying);
        // Status updates are sent regardless.
        let _ = status_rx.try_recv();
    }

    #[tokio::test]
    async fn uploader_marks_uploaded_when_checksum_matches() {
        let server = MockServer::start().await;
        // Use the CRC32C of the payload below.
        let payload = b"hello world";
        let b64 = BASE64.encode(crc32c::crc32c(payload).to_be_bytes());
        let header_value = format!("crc32c={b64}");
        let header_value_clone = header_value.clone();
        Mock::given(method("PUT"))
            .and(path("/upload/abc"))
            .respond_with(move |_req: &Request| {
                ResponseTemplate::new(200).insert_header("X-Goog-Hash", header_value_clone.as_str())
            })
            .expect(1)
            .mount(&server)
            .await;

        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let (_recording_index, _) = seed_ready_trace(
            &store,
            &recordings_root,
            "rec-1",
            "trace-1",
            "JOINT_POSITIONS",
            "arm",
            &format!("{}/upload/abc", server.uri()),
            payload,
        )
        .await;

        let api = client(&server);
        let bus = EventBus::new();
        let (status_tx, mut status_rx) = mpsc::unbounded_channel::<StatusUpdate>();

        run_upload_single(&store, recordings_root, &api, &bus, &status_tx, "trace-1").await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Uploaded);
        assert_eq!(trace.bytes_uploaded, payload.len() as i64);
        // At least one in-progress + one final status update should have
        // been queued.
        let mut count = 0;
        while status_rx.try_recv().is_ok() {
            count += 1;
        }
        assert!(count >= 1);
    }

    /// Upload `trace-1` against an expired session whose URI refetch answers
    /// `refetch`. Returns the store, its tempdir, the recording index, the
    /// trace's local file and every event published.
    async fn upload_with_refetch_answering(
        refetch: ResponseTemplate,
    ) -> (
        SqliteStateStore,
        TempDir,
        i64,
        std::path::PathBuf,
        Vec<DaemonEvent>,
    ) {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/upload/dead"))
            .respond_with(ResponseTemplate::new(410))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/org/org-1/recording/rec-1/resumable_upload_url"))
            .respond_with(refetch)
            .expect(1)
            .mount(&server)
            .await;

        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let (recording_index, local) = seed_ready_trace(
            &store,
            &recordings_root,
            "rec-1",
            "trace-1",
            "JOINT_POSITIONS",
            "arm",
            &format!("{}/upload/dead", server.uri()),
            b"payload",
        )
        .await;

        let bus = EventBus::new();
        let mut subscriber = bus.subscribe();
        let (status_tx, _status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(
            &store,
            recordings_root,
            &client(&server),
            &bus,
            &status_tx,
            "trace-1",
        )
        .await;

        let mut events = Vec::new();
        while let Ok(event) = subscriber.try_recv() {
            events.push(event);
        }
        (store, tempdir, recording_index, local, events)
    }

    #[tokio::test]
    async fn session_uri_refetch_cancelled_410_discards_the_recording() {
        let (store, _tempdir, recording_index, _local, events) =
            upload_with_refetch_answering(ResponseTemplate::new(410).set_body_json(
                serde_json::json!({
                    "detail": { "error": "Recording has been cancelled.", "error_code": "RECORDING_CANCELLED" }
                }),
            ))
            .await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Failed);
        let row = store.get_recording(recording_index).await.unwrap().unwrap();
        assert!(
            row.cancelled_at.is_some(),
            "a cancelled 410 discards the recording"
        );
        assert!(
            row.backend_cancel_notified_at.is_some(),
            "the backend already has the cancel; it must not be POSTed back"
        );
        assert!(events.contains(&DaemonEvent::RecordingCancelled { recording_index }));
    }

    /// A 404 can mean the recording was saved or reaped, or nothing about the
    /// recording at all, so uploading stops but the files stay.
    #[tokio::test]
    async fn session_uri_refetch_404_stops_uploading_but_keeps_the_recording() {
        let (store, _tempdir, recording_index, local, events) = upload_with_refetch_answering(
            ResponseTemplate::new(404)
                .set_body_json(serde_json::json!({ "detail": "Recording not found." })),
        )
        .await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Failed);
        assert_eq!(
            trace.error_code,
            Some(crate::state::TraceErrorCode::RecordingNotPending)
        );
        let row = store.get_recording(recording_index).await.unwrap().unwrap();
        assert!(
            row.cancelled_at.is_none(),
            "a 404 must not cancel the recording"
        );
        assert!(row.backend_not_found_at.is_some());
        assert!(local.exists(), "a 404 must not delete the local file");
        assert!(events.contains(&DaemonEvent::RecordingNotFound { recording_index }));
        assert!(!events
            .iter()
            .any(|event| matches!(event, DaemonEvent::RecordingCancelled { .. })));
        assert!(
            store.traces_ready_for_upload().await.unwrap().is_empty(),
            "nothing is left queued to retry"
        );
        assert!(
            store
                .recordings_pending_reclaim()
                .await
                .unwrap()
                .iter()
                .all(|row| row.recording_index != recording_index),
            "the reaper must not reclaim it"
        );
    }

    #[tokio::test]
    async fn session_expired_410_fetches_fresh_uri_and_restarts() {
        // First PUT to /upload/dead returns 410 (expired session).
        // GET resumable_upload_url returns a fresh /upload/live URI.
        // Subsequent PUT to /upload/live succeeds with a valid checksum.
        let server = MockServer::start().await;
        let payload = b"resumable-payload";
        let b64 = BASE64.encode(crc32c::crc32c(payload).to_be_bytes());
        let header_value = format!("crc32c={b64}");

        Mock::given(method("PUT"))
            .and(path("/upload/dead"))
            .respond_with(ResponseTemplate::new(410))
            .expect(1)
            .mount(&server)
            .await;
        let live_uri = format!("{}/upload/live", server.uri());
        let fresh_response = serde_json::json!({"url": live_uri});
        Mock::given(method("GET"))
            .and(path("/org/org-1/recording/rec-1/resumable_upload_url"))
            .respond_with(ResponseTemplate::new(200).set_body_json(fresh_response))
            .expect(1)
            .mount(&server)
            .await;
        let header_value_clone = header_value.clone();
        Mock::given(method("PUT"))
            .and(path("/upload/live"))
            .respond_with(move |_req: &Request| {
                ResponseTemplate::new(200).insert_header("X-Goog-Hash", header_value_clone.as_str())
            })
            .expect(1)
            .mount(&server)
            .await;

        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let dead_uri = format!("{}/upload/dead", server.uri());
        let (_recording_index, _) = seed_ready_trace(
            &store,
            &recordings_root,
            "rec-1",
            "trace-1",
            "JOINT_POSITIONS",
            "arm",
            &dead_uri,
            payload,
        )
        .await;

        let api = client(&server);
        let bus = EventBus::new();
        let (status_tx, _status_rx) = mpsc::unbounded_channel::<StatusUpdate>();

        run_upload_single(&store, recordings_root, &api, &bus, &status_tx, "trace-1").await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Uploaded);
        // Persisted URI must be the refreshed one so a restart resumes
        // against the live session, not the dead one.
        let serialised = trace.upload_session_uris.as_ref().expect("uris stored");
        assert!(
            serialised.contains("/upload/live"),
            "refreshed URI not persisted: {serialised}"
        );
        assert!(
            !serialised.contains("/upload/dead"),
            "dead URI still present: {serialised}"
        );
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
        let recording_index = recording.recording_index;
        // Stamp a cloud id so the uploader's cloud-id resolution passes and it
        // reaches the missing-data-type branch (not the deferral path).
        store
            .mark_recording_start_notified(recording_index, "rec-1")
            .await
            .unwrap();
        // Insert directly with NULL data_type so the uploader hits the
        // missing-data-type branch.
        store
            .create_trace(recording_index, "trace-1", None, None)
            .await
            .unwrap();
        let mut uris = HashMap::new();
        uris.insert("dummy".to_string(), "https://upload/abc".to_string());
        store
            .update_trace(
                "trace-1",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    upload_status: Some(TraceUploadStatus::Queued),
                    upload_session_uris: Some(serde_json::to_string(&uris).unwrap()),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();

        let api = client(&server);
        let bus = EventBus::new();
        let mut subscriber = bus.subscribe();
        let (status_tx, mut status_rx) = mpsc::unbounded_channel::<StatusUpdate>();

        run_upload_single(&store, recordings_root, &api, &bus, &status_tx, "trace-1").await;

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
    async fn empty_session_uris_fail_the_trace() {
        // An empty map would upload no files, so the trace fails without a
        // completed status, and UploadComplete still unblocks the progress gate.
        let server = MockServer::start().await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let recording = store
            .create_recording(NewRecording::default())
            .await
            .unwrap();
        store
            .mark_recording_start_notified(recording.recording_index, "rec-1")
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
                    write_status: Some(TraceWriteStatus::Written),
                    upload_status: Some(TraceUploadStatus::Queued),
                    upload_session_uris: Some("{}".to_string()),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();

        let api = client(&server);
        let bus = EventBus::new();
        let mut subscriber = bus.subscribe();
        let (status_tx, mut status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(&store, recordings_root, &api, &bus, &status_tx, "trace-1").await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Failed);
        assert!(matches!(
            subscriber.try_recv(),
            Ok(DaemonEvent::UploadComplete { .. })
        ));
        assert!(!completed_status_sent(&mut status_rx));
    }

    #[tokio::test]
    async fn no_session_uris_leaves_trace_queued() {
        // Ready-for-upload but no stored session URIs (registration hasn't
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
            .mark_recording_start_notified(recording.recording_index, "rec-1")
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
                    ..TraceUpdate::default() // no upload_session_uris
                },
            )
            .await
            .unwrap();

        let api = client(&server);
        let bus = EventBus::new();
        let (status_tx, _status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(&store, recordings_root, &api, &bus, &status_tx, "trace-1").await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(
            trace.upload_status,
            TraceUploadStatus::Queued,
            "no URIs → trace untouched for retry"
        );
    }

    #[tokio::test]
    async fn missing_cloud_recording_id_defers_upload() {
        // The recording row has no cloud id yet (start not notified). The
        // uploader must defer (leave the trace queued), never marking it
        // Uploading, so the periodic rescan re-enters once the id lands.
        let server = MockServer::start().await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let recording = store
            .create_recording(NewRecording::default())
            .await
            .unwrap();
        // Deliberately NOT start-notified, so recording_id is None.
        store
            .create_trace(
                recording.recording_index,
                "trace-1",
                Some("JOINT_POSITIONS"),
                Some("arm"),
            )
            .await
            .unwrap();
        let mut uris = HashMap::new();
        uris.insert(
            "JOINT_POSITIONS/arm/trace.json".to_string(),
            "https://upload/abc".to_string(),
        );
        store
            .update_trace(
                "trace-1",
                TraceUpdate {
                    upload_status: Some(TraceUploadStatus::Queued),
                    upload_session_uris: Some(serde_json::to_string(&uris).unwrap()),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();

        let api = client(&server);
        let bus = EventBus::new();
        let (status_tx, _status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(&store, recordings_root, &api, &bus, &status_tx, "trace-1").await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(
            trace.upload_status,
            TraceUploadStatus::Queued,
            "no cloud id → upload deferred, not started"
        );
    }

    #[tokio::test]
    async fn missing_artefact_marks_trace_failed() {
        // Session URI + cloud id + data_type all present, but the on-disk file
        // is gone. The uploader marks the trace Failed and emits UploadComplete
        // so the recording's progress gate isn't wedged on a vanished file, but
        // never reports it completed to the backend.
        let server = MockServer::start().await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let recording = store
            .create_recording(NewRecording::default())
            .await
            .unwrap();
        store
            .mark_recording_start_notified(recording.recording_index, "rec-1")
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
        let mut uris = HashMap::new();
        uris.insert(
            "JOINT_POSITIONS/arm/trace.json".to_string(),
            format!("{}/upload/abc", server.uri()),
        );
        store
            .update_trace(
                "trace-1",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    upload_status: Some(TraceUploadStatus::Queued),
                    upload_session_uris: Some(serde_json::to_string(&uris).unwrap()),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();
        // Deliberately do NOT create the on-disk artefact.

        let api = client(&server);
        let bus = EventBus::new();
        let mut subscriber = bus.subscribe();
        let (status_tx, mut status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(&store, recordings_root, &api, &bus, &status_tx, "trace-1").await;

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
        // Never realistically returns, so an upload that is not aborted times
        // out rather than passing by luck.
        Mock::given(method("PUT"))
            .and(path("/upload/slow"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(120)))
            .mount(&server)
            .await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let (recording_index, _) = seed_ready_trace(
            &store,
            &recordings_root,
            "rec-1",
            "trace-1",
            "JOINT_POSITIONS",
            "arm",
            &format!("{}/upload/slow", server.uri()),
            b"payload",
        )
        .await;

        let aborts = UploadAborts::default();
        let mut in_flight =
            dispatch_upload(&store, recordings_root, &client(&server), &aborts).await;
        wait_for_put(&server, "/upload/slow").await;
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
        Mock::given(method("PUT"))
            .and(path("/upload/abc"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        let (recording_index, _) = seed_ready_trace(
            &store,
            &recordings_root,
            "rec-1",
            "trace-1",
            "JOINT_POSITIONS",
            "arm",
            &format!("{}/upload/abc", server.uri()),
            b"payload",
        )
        .await;
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
            let payload = b"hello world";
            let checksum = format!(
                "crc32c={}",
                BASE64.encode(crc32c::crc32c(payload).to_be_bytes())
            );
            // Completes after a short delay, so an upload that is not aborted
            // ends up `Uploaded`.
            Mock::given(method("PUT"))
                .and(path("/upload/abc"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .insert_header("X-Goog-Hash", checksum.as_str())
                        .set_delay(Duration::from_millis(1500)),
                )
                .mount(&server)
                .await;
            let (store, tempdir) = open_store().await;
            let recordings_root = tempdir.path().join("recordings");
            let (recording_index, _) = seed_ready_trace(
                &store,
                &recordings_root,
                "rec-1",
                "trace-1",
                "JOINT_POSITIONS",
                "arm",
                &format!("{}/upload/abc", server.uri()),
                payload,
            )
            .await;

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
            wait_for_put(&server, "/upload/abc").await;

            if not_found {
                store
                    .mark_recording_not_found(recording_index)
                    .await
                    .unwrap();
                bus.publish(DaemonEvent::RecordingNotFound { recording_index });
            } else {
                store.cancel_recording(recording_index, 0).await.unwrap();
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

    // The following drive the wire-level transfer layer (`upload_transfer.rs`)
    // through the uploader entry point, exercising branches that only a live
    // server reaches: transient-retry, auth refresh, a hard error, and the
    // no-progress stall guard.

    #[tokio::test]
    async fn transient_5xx_chunk_is_retried_then_uploaded() {
        // A 503 on the chunk PUT is transient: put_chunk retries within budget
        // and the trace still settles as Uploaded.
        let server = MockServer::start().await;
        let payload = b"transient-ok";
        let header = format!(
            "crc32c={}",
            BASE64.encode(crc32c::crc32c(payload).to_be_bytes())
        );
        Mock::given(method("PUT"))
            .and(path("/upload/abc"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .and(path("/upload/abc"))
            .respond_with(move |_req: &Request| {
                ResponseTemplate::new(200).insert_header("X-Goog-Hash", header.as_str())
            })
            .expect(1)
            .mount(&server)
            .await;

        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        seed_ready_trace(
            &store,
            &recordings_root,
            "rec-1",
            "trace-1",
            "JOINT_POSITIONS",
            "arm",
            &format!("{}/upload/abc", server.uri()),
            payload,
        )
        .await;

        let api = client(&server);
        let bus = EventBus::new();
        let (status_tx, _status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(&store, recordings_root, &api, &bus, &status_tx, "trace-1").await;

        assert_eq!(
            store
                .get_trace("trace-1")
                .await
                .unwrap()
                .unwrap()
                .upload_status,
            TraceUploadStatus::Uploaded
        );
    }

    #[tokio::test]
    async fn auth_401_on_chunk_is_reloaded_then_uploaded() {
        // A 401 mid-upload triggers one token reload and then the retried PUT
        // succeeds.
        let server = MockServer::start().await;
        let payload = b"auth-ok";
        let header = format!(
            "crc32c={}",
            BASE64.encode(crc32c::crc32c(payload).to_be_bytes())
        );
        Mock::given(method("PUT"))
            .and(path("/upload/abc"))
            .respond_with(ResponseTemplate::new(401))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .and(path("/upload/abc"))
            .respond_with(move |_req: &Request| {
                ResponseTemplate::new(200).insert_header("X-Goog-Hash", header.as_str())
            })
            .expect(1)
            .mount(&server)
            .await;

        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        seed_ready_trace(
            &store,
            &recordings_root,
            "rec-1",
            "trace-1",
            "JOINT_POSITIONS",
            "arm",
            &format!("{}/upload/abc", server.uri()),
            payload,
        )
        .await;

        let api = client(&server);
        let bus = EventBus::new();
        let (status_tx, _status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(&store, recordings_root, &api, &bus, &status_tx, "trace-1").await;

        assert_eq!(
            store
                .get_trace("trace-1")
                .await
                .unwrap()
                .unwrap()
                .upload_status,
            TraceUploadStatus::Uploaded
        );
    }

    #[tokio::test]
    async fn hard_4xx_chunk_error_rolls_trace_to_retrying() {
        // A non-retryable 4xx is a hard error: put_chunk returns Failed,
        // upload_one_file errors, and the uploader rolls the trace to Retrying
        // for the registration recovery sweep to re-arm.
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/upload/abc"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .expect(1)
            .mount(&server)
            .await;

        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        seed_ready_trace(
            &store,
            &recordings_root,
            "rec-1",
            "trace-1",
            "JOINT_POSITIONS",
            "arm",
            &format!("{}/upload/abc", server.uri()),
            b"hard-error",
        )
        .await;

        let api = client(&server);
        let bus = EventBus::new();
        let (status_tx, _status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(&store, recordings_root, &api, &bus, &status_tx, "trace-1").await;

        assert_eq!(
            store
                .get_trace("trace-1")
                .await
                .unwrap()
                .unwrap()
                .upload_status,
            TraceUploadStatus::Retrying
        );
    }

    #[tokio::test]
    async fn no_progress_308_stall_marks_trace_retrying() {
        // A peer that keeps replying 308 without committing any bytes must not
        // wedge the upload task (and its concurrency permit) forever: the
        // stall guard bails after MAX_RETRIES and the trace rolls to Retrying.
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/upload/abc"))
            .respond_with(ResponseTemplate::new(308)) // no Range → no forward progress
            .mount(&server)
            .await;

        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        seed_ready_trace(
            &store,
            &recordings_root,
            "rec-1",
            "trace-1",
            "JOINT_POSITIONS",
            "arm",
            &format!("{}/upload/abc", server.uri()),
            b"stall-me",
        )
        .await;

        let api = client(&server);
        let bus = EventBus::new();
        let (status_tx, _status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(&store, recordings_root, &api, &bus, &status_tx, "trace-1").await;

        assert_eq!(
            store
                .get_trace("trace-1")
                .await
                .unwrap()
                .unwrap()
                .upload_status,
            TraceUploadStatus::Retrying
        );
    }

    /// Upload a written RGB trace for `cam0` whose map links `lossy.mp4` and
    /// `trace.json` to `/upload/lossy` and `/upload/trace-json`, against a fake
    /// GCS that accepts every PUT as the final chunk. Only the files in
    /// `on_disk` are written to the trace directory.
    async fn upload_rgb_trace(
        server: &MockServer,
        on_disk: &[(&str, &[u8])],
    ) -> (
        SqliteStateStore,
        TempDir,
        mpsc::UnboundedReceiver<StatusUpdate>,
    ) {
        Mock::given(method("PUT"))
            .and(path_regex("^/upload/"))
            .respond_with(|request: &Request| {
                let crc = BASE64.encode(crc32c::crc32c(&request.body).to_be_bytes());
                ResponseTemplate::new(200)
                    .insert_header("X-Goog-Hash", format!("crc32c={crc}").as_str())
            })
            .mount(server)
            .await;
        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
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
            .create_trace(recording_index, "trace-1", Some("RGB_IMAGES"), Some("cam0"))
            .await
            .unwrap();
        let dir = TracePath::new(
            recording_index.to_string(),
            "RGB_IMAGES",
            "trace-1".to_string(),
        )
        .directory(&recordings_root);
        std::fs::create_dir_all(&dir).unwrap();
        for (basename, contents) in on_disk {
            std::fs::write(dir.join(basename), contents).unwrap();
        }
        let uris = HashMap::from([
            (
                format!("RGB_IMAGES/cam0/{LOSSY_VIDEO_FILENAME}"),
                format!("{}/upload/lossy", server.uri()),
            ),
            (
                format!("RGB_IMAGES/cam0/{TRACE_JSON_FILENAME}"),
                format!("{}/upload/trace-json", server.uri()),
            ),
        ]);
        store
            .update_trace(
                "trace-1",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    upload_status: Some(TraceUploadStatus::Queued),
                    upload_session_uris: Some(serde_json::to_string(&uris).unwrap()),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();

        let (status_tx, status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(
            &store,
            recordings_root,
            &client(server),
            &EventBus::new(),
            &status_tx,
            "trace-1",
        )
        .await;
        (store, tempdir, status_rx)
    }

    /// PUTs to `upload_path` that declare the object's total size, which is
    /// what finalises a GCS resumable session.
    async fn finalising_puts(server: &MockServer, upload_path: &str) -> usize {
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .filter(|request| request.method.as_str() == "PUT" && request.url.path() == upload_path)
            .filter(|request| {
                request
                    .headers
                    .get("content-range")
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.rsplit_once('/'))
                    .is_some_and(|(_, total)| total != "*")
            })
            .count()
    }

    fn completed_status_sent(status_rx: &mut mpsc::UnboundedReceiver<StatusUpdate>) -> bool {
        let mut completed = false;
        while let Ok(update) = status_rx.try_recv() {
            completed |= update.completed;
        }
        completed
    }

    #[tokio::test]
    async fn every_registered_file_is_finalised_before_uploaded() {
        let server = MockServer::start().await;
        let (store, _tempdir, mut status_rx) = upload_rgb_trace(
            &server,
            &[
                (LOSSY_VIDEO_FILENAME, b"fake-mp4-bytes"),
                (TRACE_JSON_FILENAME, b"[]"),
            ],
        )
        .await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Uploaded);
        assert!(completed_status_sent(&mut status_rx));
        assert_eq!(finalising_puts(&server, "/upload/lossy").await, 1);
        assert_eq!(finalising_puts(&server, "/upload/trace-json").await, 1);
    }

    #[tokio::test]
    async fn zero_byte_sidecar_fails_trace_without_completion() {
        let server = MockServer::start().await;
        let (store, _tempdir, mut status_rx) = upload_rgb_trace(
            &server,
            &[
                (LOSSY_VIDEO_FILENAME, b"fake-mp4-bytes"),
                (TRACE_JSON_FILENAME, b""),
            ],
        )
        .await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Failed);
        assert!(!completed_status_sent(&mut status_rx));
        assert_eq!(finalising_puts(&server, "/upload/trace-json").await, 0);
    }

    #[tokio::test]
    async fn missing_sidecar_fails_trace_without_completion() {
        // Upload order is random, so `lossy.mp4` may already be up.
        let server = MockServer::start().await;
        let (store, _tempdir, mut status_rx) =
            upload_rgb_trace(&server, &[(LOSSY_VIDEO_FILENAME, b"fake-mp4-bytes")]).await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Failed);
        assert!(!completed_status_sent(&mut status_rx));
    }

    /// Upload a 13-byte trace whose data PUT gets a 308 with a full `Range`,
    /// answering the `bytes */13` status query with `status_query_status`.
    async fn upload_after_full_range_308(status_query_status: u16) -> TraceUploadStatus {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/upload/abc"))
            .and(header("content-range", "bytes */13"))
            .respond_with(ResponseTemplate::new(status_query_status))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .and(path("/upload/abc"))
            .respond_with(ResponseTemplate::new(308).insert_header("Range", "bytes=0-12"))
            .mount(&server)
            .await;

        let (store, tempdir) = open_store().await;
        let recordings_root = tempdir.path().join("recordings");
        seed_ready_trace(
            &store,
            &recordings_root,
            "rec-1",
            "trace-1",
            "JOINT_POSITIONS",
            "arm",
            &format!("{}/upload/abc", server.uri()),
            b"thirteen-byte",
        )
        .await;
        let (status_tx, _status_rx) = mpsc::unbounded_channel::<StatusUpdate>();
        run_upload_single(
            &store,
            recordings_root,
            &client(&server),
            &EventBus::new(),
            &status_tx,
            "trace-1",
        )
        .await;
        store
            .get_trace("trace-1")
            .await
            .unwrap()
            .unwrap()
            .upload_status
    }

    #[tokio::test]
    async fn full_range_308_without_a_2xx_is_retried() {
        assert_eq!(
            upload_after_full_range_308(308).await,
            TraceUploadStatus::Retrying
        );
    }

    #[tokio::test]
    async fn full_range_308_is_uploaded_once_the_status_query_returns_2xx() {
        assert_eq!(
            upload_after_full_range_308(200).await,
            TraceUploadStatus::Uploaded
        );
    }
}
