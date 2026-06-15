//! Resumable file uploader coordinator.
//!
//! Subscribes to [`DaemonEvent::ReadyForUpload`] (and re-scans the
//! state store on startup for any traces already in the registered/queued
//! state). For each on-disk artefact the coordinator PUTs `CHUNK_SIZE` (4 MiB)
//! chunks to the GCS resumable session URI persisted by the registration coordinator,
//! handling 308-continue, 410-session-expired, and 401-auth-refresh
//! transitions. On completion the trace is marked `Uploaded` and the upload
//! sub-system publishes `UploadComplete` for the status updater.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, Semaphore};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::{interval, MissedTickBehavior};

use crate::api::ApiClient;
use crate::cloud::cloud_files::content_type_for_filename;
use crate::cloud::status::StatusUpdate;
use crate::cloud::upload_transfer::{upload_one_file, UploadFileOutcome};
use crate::cloud::OrgIdRx;
use crate::lifecycle::signals::ShutdownSignal;
use crate::state::store::TraceUpdate;
use crate::state::{
    ConnectionState, DaemonEvent, EventBus, SqliteStateStore, StateStore, TraceRecord,
    TraceUploadStatus, TraceWriteHandle,
};
use crate::storage::paths::TracePath;

/// Maximum number of traces uploading concurrently. With 8 parallel contexts
/// each queuing ~128 traces simultaneously (1024 total), 32 slots serialise
/// into ~32 rounds × 300 ms ≈ 9.6 s. 128 slots cuts that to ~8 rounds ×
/// 300 ms ≈ 2.4 s, giving ~6 s headroom against the 9 s stop-recording SLA.
pub const MAX_CONCURRENT_UPLOADS: usize = 128;

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
        // Safety-net rescan: catch any traces that were skipped when the
        // semaphore was full during a drain, without relying on bus events.
        let mut rescan_tick = interval(Duration::from_secs(5));
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
                                trace_id,
                            );
                        }
                        Ok(_) => {}
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            tracing::warn!(skipped, "uploader missed bus events; rescanning");
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
    in_flight.spawn(async move {
        upload_single(
            &store,
            &trace_writer,
            &bus,
            &client,
            &recordings_root,
            &org_rx,
            &status_tx,
            &trace_id,
        )
        .await;
        drop(permit);
        trace_id
    });
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
        // Nothing to upload — mark uploaded immediately so downstream
        // accounting matches a registered-but-empty trace.
        finalise_upload(store, bus, status_tx, &trace, 0).await;
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
        // can't locate the on-disk artefact. Surface the failure both to the
        // status updater and on the event bus so the recording's progress
        // reporter (which gates on every trace having settled) doesn't wait
        // for an upload that can never happen.
        tracing::warn!(trace_id, "trace row missing data_type; marking failed");
        mark_failed_and_emit(store, bus, status_tx, &trace, "trace missing data_type").await;
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

    // Group on-disk artefacts and their session URIs in a stable order. We
    // walk by-reference rather than by-value so the recovery message stays
    // tied to the source filename if anything fails downstream.
    let mut total_uploaded: i64 = 0;
    let mut session_uris = session_uris;
    for index in 0..session_uris.len() {
        let (filename, session_uri) = session_uris[index].clone();
        let local_path = trace_dir.join(file_basename(&filename));
        if !local_path.exists() {
            tracing::warn!(
                trace_id,
                path = %local_path.display(),
                "expected upload artefact missing; marking trace failed"
            );
            mark_failed_and_emit(
                store,
                bus,
                status_tx,
                &trace,
                &format!("missing artefact {filename}"),
            )
            .await;
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
            Err(error) => {
                tracing::warn!(%error, trace_id, "upload failed; rolling back to retrying");
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

    finalise_upload(store, bus, status_tx, &trace, total_uploaded).await;
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
    status_tx: &tokio::sync::mpsc::UnboundedSender<StatusUpdate>,
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
    let _ = status_tx.send(StatusUpdate::completed(
        trace.recording_index,
        trace.trace_id.clone(),
        trace.total_bytes.max(0),
    ));
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
    use crate::storage::paths::TRACE_JSON_FILENAME;
    use base64::engine::general_purpose::STANDARD as BASE64;
    use base64::Engine;
    use md5::{Digest, Md5};
    use std::collections::HashMap;
    use std::time::Duration;
    use tempfile::TempDir;
    use tokio::sync::mpsc;
    use wiremock::matchers::{method, path};
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

    #[tokio::test]
    async fn bad_server_checksum_marks_trace_retrying() {
        // Server returns a deliberately wrong MD5 — the checksum guard
        // must reject the upload and roll the trace back to `retrying`
        // (the registration coordinator's recovery sweep takes it from
        // there). The doc-claimed happy path is covered by
        // `uploader_marks_uploaded_when_checksum_matches` below.
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/upload/abc"))
            .respond_with(|_req: &Request| {
                ResponseTemplate::new(200)
                    .insert_header("X-Goog-Hash", "md5=oZ1Y1O3hI+9w3VPvbQRcVw==")
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

        let store_arc = Arc::new(store.clone());
        let (trace_writer, _trace_writer_owner) =
            crate::state::trace_writer::spawn(store_arc.clone());
        let recordings_root = Arc::new(recordings_root);
        upload_single(
            &store_arc,
            &trace_writer,
            &bus,
            &api,
            &recordings_root,
            &org_rx(Some("org-1")),
            &status_tx,
            "trace-1",
        )
        .await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Retrying);
        // Status updates are sent regardless.
        let _ = status_rx.try_recv();
    }

    #[tokio::test]
    async fn uploader_marks_uploaded_when_checksum_matches() {
        let server = MockServer::start().await;
        // Use the MD5 of the payload below.
        let payload = b"hello world";
        let mut hasher = Md5::new();
        hasher.update(payload);
        let b64 = BASE64.encode(hasher.finalize());
        let header_value = format!("md5={b64}");
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

        let store_arc = Arc::new(store.clone());
        let (trace_writer, _trace_writer_owner) =
            crate::state::trace_writer::spawn(store_arc.clone());
        let recordings_root = Arc::new(recordings_root);
        upload_single(
            &store_arc,
            &trace_writer,
            &bus,
            &api,
            &recordings_root,
            &org_rx(Some("org-1")),
            &status_tx,
            "trace-1",
        )
        .await;

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

    #[tokio::test]
    async fn session_expired_410_fetches_fresh_uri_and_restarts() {
        // First PUT to /upload/dead returns 410 (expired session).
        // GET resumable_upload_url returns a fresh /upload/live URI.
        // Subsequent PUT to /upload/live succeeds with a valid checksum.
        let server = MockServer::start().await;
        let payload = b"resumable-payload";
        let mut hasher = Md5::new();
        hasher.update(payload);
        let b64 = BASE64.encode(hasher.finalize());
        let header_value = format!("md5={b64}");

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

        let store_arc = Arc::new(store.clone());
        let (trace_writer, _trace_writer_owner) =
            crate::state::trace_writer::spawn(store_arc.clone());
        let recordings_root = Arc::new(recordings_root);
        upload_single(
            &store_arc,
            &trace_writer,
            &bus,
            &api,
            &recordings_root,
            &org_rx(Some("org-1")),
            &status_tx,
            "trace-1",
        )
        .await;

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

        let store_arc = Arc::new(store.clone());
        let (trace_writer, _trace_writer_owner) =
            crate::state::trace_writer::spawn(store_arc.clone());
        let recordings_root = Arc::new(recordings_root);
        upload_single(
            &store_arc,
            &trace_writer,
            &bus,
            &api,
            &recordings_root,
            &org_rx(Some("org-1")),
            &status_tx,
            "trace-1",
        )
        .await;

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
        // Status updater also gets a terminal entry.
        let update = status_rx.try_recv().expect("status update enqueued");
        assert!(update.completed);
    }
}
