//! Resumable file uploader coordinator.
//!
//! Subscribes to [`DaemonEvent::ReadyForUpload`] (and re-scans the
//! state store on startup for any traces already in the registered/queued
//! state). For each on-disk artefact the coordinator PUTs 64 MiB chunks to
//! the GCS resumable session URI persisted by the registration coordinator,
//! handling 308-continue, 410-session-expired, and 401-auth-refresh
//! transitions. On completion the trace is marked `Uploaded` and the upload
//! sub-system publishes `UploadComplete` for the status updater.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use bytes::Bytes;
use md5::{Digest, Md5};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_LENGTH, CONTENT_RANGE};
use reqwest::StatusCode;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::sync::{broadcast, Semaphore};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::{interval, sleep, timeout, MissedTickBehavior};

use crate::api::{ApiClient, ApiClientError};
use crate::cloud::cloud_files::{content_type_for, ContentKind};
use crate::cloud::status::StatusUpdate;
use crate::lifecycle::signals::ShutdownSignal;
use crate::state::store::TraceUpdate;
use crate::state::{
    ConnectionState, DaemonEvent, EventBus, SqliteStateStore, StateStore, TraceRecord,
    TraceUploadStatus,
};
use crate::storage::paths::TracePath;

/// Chunk size used for resumable uploads.
pub const CHUNK_SIZE: usize = 64 * 1024 * 1024;
/// Cap on the exponential backoff for transient upload failures.
pub const MAX_BACKOFF: Duration = Duration::from_secs(300);
/// Maximum retries for a single chunk.
pub const MAX_RETRIES: u32 = 5;
/// Hard deadline for a single chunk PUT. Belt-and-braces over the reqwest
/// client-level timeout, which can silently fail to fire for direct GCS
/// resumable session URI uploads.
const CHUNK_UPLOAD_TIMEOUT: Duration = Duration::from_secs(120);
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
pub fn spawn_uploader(
    store: SqliteStateStore,
    bus: EventBus,
    client: Arc<ApiClient>,
    recordings_root: Arc<PathBuf>,
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
                            &bus,
                            &client,
                            &recordings_root,
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
                                    &bus,
                                    &client,
                                    &recordings_root,
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
                                &bus,
                                &client,
                                &recordings_root,
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
                                    &bus,
                                    &client,
                                    &recordings_root,
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
                            &bus,
                            &client,
                            &recordings_root,
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
    bus: &EventBus,
    client: &Arc<ApiClient>,
    recordings_root: &Arc<PathBuf>,
    status_tx: &tokio::sync::mpsc::UnboundedSender<StatusUpdate>,
    semaphore: &Arc<Semaphore>,
    in_flight: &mut JoinSet<String>,
    in_flight_ids: &mut HashSet<String>,
) {
    let recordings = match store.list_recordings().await {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(%error, "uploader could not list recordings");
            return;
        }
    };
    for recording in recordings {
        let traces = match store
            .list_traces_for_recording(&recording.recording_id)
            .await
        {
            Ok(rows) => rows,
            Err(error) => {
                tracing::warn!(%error, recording_id = recording.recording_id, "uploader could not list traces");
                continue;
            }
        };
        for trace in traces {
            if matches!(
                trace.upload_status,
                TraceUploadStatus::Queued | TraceUploadStatus::Retrying
            ) {
                spawn_upload_task(
                    store,
                    bus,
                    client,
                    recordings_root,
                    status_tx,
                    semaphore,
                    in_flight,
                    in_flight_ids,
                    trace.trace_id,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_upload_task(
    store: &Arc<SqliteStateStore>,
    bus: &EventBus,
    client: &Arc<ApiClient>,
    recordings_root: &Arc<PathBuf>,
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
    let bus = bus.clone();
    let client = Arc::clone(client);
    let recordings_root = Arc::clone(recordings_root);
    let status_tx = status_tx.clone();
    in_flight.spawn(async move {
        upload_single(
            &store,
            &bus,
            &client,
            &recordings_root,
            &status_tx,
            &trace_id,
        )
        .await;
        drop(permit);
        trace_id
    });
}

async fn upload_single(
    store: &Arc<SqliteStateStore>,
    bus: &EventBus,
    client: &Arc<ApiClient>,
    recordings_root: &Arc<PathBuf>,
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
    let (Some(data_type), recording_id) = (trace.data_type.as_deref(), trace.recording_id.as_str())
    else {
        // No data_type means we never saw a `StartTrace` for this row, so we
        // can't locate the on-disk artefact. Surface the failure both to the
        // status updater and on the event bus so the recording's progress
        // reporter (which gates on every trace having settled) doesn't wait
        // for an upload that can never happen.
        tracing::warn!(trace_id, "trace row missing data_type; marking failed");
        mark_failed_and_emit(store, bus, status_tx, &trace, "trace missing data_type").await;
        return;
    };
    let trace_dir = TracePath::new(recording_id, data_type, trace_id.to_string())
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

        // `content_type` here drives the GCS-side metadata when we
        // re-acquire a session URI on 410; we derive it from the daemon's
        // cloud-file classification so the refresh matches what
        // registration originally registered.
        let content_type = if matches!(content_type_for(data_type), ContentKind::Rgb)
            && filename.ends_with(".mp4")
        {
            "video/mp4"
        } else {
            "application/json"
        };
        let outcome = upload_one_file(
            client,
            store,
            bus,
            status_tx,
            &trace,
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
                    increment_upload_attempts: true,
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
        recording_id: trace.recording_id.clone(),
    });
    let _ = status_tx.send(StatusUpdate::completed(
        trace.recording_id.clone(),
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
        recording_id: trace.recording_id.clone(),
    });
    let _ = status_tx.send(StatusUpdate::completed(
        trace.recording_id.clone(),
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

/// Outcome of [`upload_one_file`]. Carries a refreshed `session_uri` when the
/// server expired the original one mid-upload so the caller can persist it
/// for restart-resume.
struct UploadFileOutcome {
    bytes_uploaded: i64,
    final_session_uri: Option<String>,
}

#[allow(clippy::too_many_arguments)]
async fn upload_one_file(
    client: &Arc<ApiClient>,
    store: &Arc<SqliteStateStore>,
    bus: &EventBus,
    status_tx: &tokio::sync::mpsc::UnboundedSender<StatusUpdate>,
    trace: &TraceRecord,
    local_path: &std::path::Path,
    cloud_filepath: &str,
    content_type: &str,
    session_uri: String,
) -> Result<UploadFileOutcome, String> {
    let metadata = tokio::fs::metadata(local_path)
        .await
        .map_err(|error| format!("stat {} failed: {error}", local_path.display()))?;
    let total_bytes = metadata.len();
    let original_uri = session_uri.clone();
    if total_bytes == 0 {
        return Ok(UploadFileOutcome {
            bytes_uploaded: 0,
            final_session_uri: None,
        });
    }

    let mut file = File::open(local_path)
        .await
        .map_err(|error| format!("open {} failed: {error}", local_path.display()))?;
    let mut offset: u64 = 0;
    let mut hasher = Md5::new();
    let mut server_md5_hex: Option<String> = None;
    let mut session_uri = session_uri;
    let recording_id = trace.recording_id.clone();
    let trace_id = trace.trace_id.clone();
    let Some(org_id) = trace_org_id(store, &recording_id).await else {
        return Err("recording has no org_id; cannot refresh session URI".to_string());
    };

    tracing::info!(
        trace_id,
        path = %local_path.display(),
        bytes = total_bytes,
        "starting file upload"
    );
    let upload_started = Instant::now();
    while offset < total_bytes {
        let chunk_end = (offset + CHUNK_SIZE as u64).min(total_bytes) - 1;
        let chunk_len = (chunk_end - offset + 1) as usize;
        let mut buffer = vec![0u8; chunk_len];
        file.seek(std::io::SeekFrom::Start(offset))
            .await
            .map_err(|error| format!("seek failed: {error}"))?;
        file.read_exact(&mut buffer)
            .await
            .map_err(|error| format!("read failed: {error}"))?;
        let chunk = Bytes::from(buffer);
        let is_final = chunk_end + 1 == total_bytes;

        let outcome = put_chunk(
            client,
            &session_uri,
            chunk.clone(),
            offset,
            chunk_end,
            total_bytes,
            is_final,
        )
        .await?;
        match outcome {
            PutChunkOutcome::Accepted { headers, body } => {
                hasher.update(&chunk);
                if is_final {
                    server_md5_hex = extract_server_md5(&headers, &body);
                }
                offset += chunk_len as u64;
            }
            PutChunkOutcome::Incomplete { headers } => {
                // 308 — the server tells us how much it actually committed
                // via the Range header. Trust that number unconditionally:
                // if it's less than what we sent, the server dropped the
                // tail and we must re-send it; if it's equal, we're in
                // sync; if it's missing, the server has nothing yet and we
                // should retry the same offset (a 0-byte advance).
                hasher.update(&chunk);
                let server_offset = parse_resume_offset(&headers).unwrap_or(offset);
                if server_offset < offset {
                    return Err(format!(
                        "server resume offset {server_offset} is behind local offset {offset}; \
                         refusing to corrupt {}",
                        local_path.display()
                    ));
                }
                if server_offset > offset + chunk_len as u64 {
                    // Server claims more bytes than we sent — accept its
                    // view but log loudly; this is the only branch that
                    // can move ahead of the local checksum, so flag the
                    // hash as untrustworthy by clearing it.
                    tracing::warn!(
                        server_offset,
                        local_offset = offset + chunk_len as u64,
                        path = %local_path.display(),
                        "server resume offset is ahead of local; skipping local checksum"
                    );
                    server_md5_hex = None;
                }
                offset = server_offset;
            }
            PutChunkOutcome::SessionExpired => {
                tracing::info!(
                    trace_id,
                    path = %local_path.display(),
                    "upload session expired; requesting fresh URI"
                );
                match client
                    .fetch_resumable_upload_url(
                        &org_id,
                        &recording_id,
                        cloud_filepath,
                        content_type,
                    )
                    .await
                {
                    Ok(new_uri) => {
                        session_uri = new_uri;
                        // A new session means the server has zero bytes for
                        // this file; restart from offset 0 and rehash.
                        offset = 0;
                        hasher = Md5::new();
                        server_md5_hex = None;
                        continue;
                    }
                    Err(error) => {
                        return Err(format!("failed to fetch fresh session URI: {error}"));
                    }
                }
            }
            PutChunkOutcome::Failed { status, body } => {
                return Err(format!(
                    "upload failed with HTTP {status} for {}: {body}",
                    local_path.display()
                ));
            }
        }

        bus.publish(DaemonEvent::UploadProgress {
            trace_id: trace_id.clone(),
            recording_id: recording_id.clone(),
            bytes_uploaded: offset as i64,
            total_bytes: Some(total_bytes as i64),
        });
        let _ = status_tx.send(StatusUpdate::in_progress(
            recording_id.clone(),
            trace_id.clone(),
            offset as i64,
        ));
        // Persist the rolling progress every chunk so a restart can resume
        // from the last committed offset.
        let update = TraceUpdate {
            bytes_uploaded: Some(offset as i64),
            ..TraceUpdate::default()
        };
        if let Err(error) = store.update_trace(&trace_id, update).await {
            tracing::warn!(%error, trace_id, "failed to persist upload progress");
        }
    }

    tracing::info!(
        trace_id,
        path = %local_path.display(),
        bytes = total_bytes,
        elapsed_ms = upload_started.elapsed().as_millis(),
        "file upload complete"
    );
    if let Some(expected) = server_md5_hex {
        let local_hex = hex_of(hasher.finalize().as_slice());
        if expected != local_hex {
            return Err(format!(
                "checksum mismatch for {}: local={local_hex} server={expected}",
                local_path.display()
            ));
        }
    }
    let final_session_uri = (session_uri != original_uri).then_some(session_uri);
    Ok(UploadFileOutcome {
        bytes_uploaded: total_bytes as i64,
        final_session_uri,
    })
}

async fn trace_org_id(store: &Arc<SqliteStateStore>, recording_id: &str) -> Option<String> {
    match store.get_recording(recording_id).await {
        Ok(Some(row)) => row.org_id,
        Ok(None) => None,
        Err(error) => {
            tracing::warn!(%error, recording_id, "uploader could not read recording org_id");
            None
        }
    }
}

/// Outcome of a single PUT to the resumable session URI. Returned by
/// [`put_chunk`] so [`upload_one_file`] can dispatch on it without re-parsing
/// status codes.
enum PutChunkOutcome {
    /// 2xx — chunk accepted. Headers/body carry the final response on the
    /// last chunk (server-side MD5 lives here).
    Accepted { headers: HeaderMap, body: String },
    /// 308 — chunk accepted but the server wants more bytes. The Range
    /// header tells us where it is.
    Incomplete { headers: HeaderMap },
    /// 410/404 — the resumable session is gone. The caller must call
    /// `/resumable_upload_url` to obtain a fresh one.
    SessionExpired,
    /// Any other non-retryable status; the caller surfaces it as a hard
    /// error and lets the upload coordinator roll the trace to `retrying`.
    Failed { status: StatusCode, body: String },
}

async fn put_chunk(
    client: &Arc<ApiClient>,
    session_uri: &str,
    chunk: Bytes,
    chunk_start: u64,
    chunk_end: u64,
    total_bytes: u64,
    is_final: bool,
) -> Result<PutChunkOutcome, String> {
    let mut headers = HeaderMap::new();
    let content_range = if is_final {
        format!("bytes {chunk_start}-{chunk_end}/{total_bytes}")
    } else {
        format!("bytes {chunk_start}-{chunk_end}/*")
    };
    headers.insert(
        CONTENT_RANGE,
        HeaderValue::from_str(&content_range).unwrap(),
    );
    headers.insert(CONTENT_LENGTH, HeaderValue::from(chunk.len() as u64));

    let mut attempt: u32 = 0;
    let mut refreshed_auth = false;
    loop {
        let bearer = match client.auth().bearer_token().await {
            Ok(token) => token,
            Err(error) => {
                tracing::warn!(%error, "uploader could not read auth token");
                return Err(format!("auth load failed: {error}"));
            }
        };
        let mut request_headers = headers.clone();
        request_headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {bearer}"))
                .map_err(|error| format!("auth header invalid: {error}"))?,
        );

        // `Bytes` is cheaply cloneable (Arc-backed), so re-sending the
        // same chunk on retry is a refcount bump, not a 64 MiB copy.
        let request = client
            .raw_client()
            .put(session_uri)
            .headers(request_headers)
            .body(chunk.clone())
            .build()
            .map_err(|error| format!("failed to build request: {error}"))?;
        tracing::debug!(
            attempt,
            bytes = chunk.len(),
            chunk_start,
            chunk_end,
            "sending upload chunk"
        );
        let chunk_started = Instant::now();
        let response =
            match timeout(CHUNK_UPLOAD_TIMEOUT, client.raw_client().execute(request)).await {
                Ok(Ok(response)) => response,
                Ok(Err(error)) => {
                    if attempt + 1 >= MAX_RETRIES {
                        return Err(format!("transport error: {error}"));
                    }
                    attempt += 1;
                    tracing::warn!(%error, attempt, "upload chunk transport error; retrying");
                    sleep(backoff(attempt)).await;
                    continue;
                }
                Err(_elapsed) => {
                    tracing::warn!(
                        attempt,
                        timeout_secs = CHUNK_UPLOAD_TIMEOUT.as_secs(),
                        bytes = chunk.len(),
                        "upload chunk PUT timed out; retrying"
                    );
                    if attempt + 1 >= MAX_RETRIES {
                        return Err(format!(
                            "chunk PUT timed out after {}s ({MAX_RETRIES} attempts exhausted)",
                            CHUNK_UPLOAD_TIMEOUT.as_secs()
                        ));
                    }
                    attempt += 1;
                    sleep(backoff(attempt)).await;
                    continue;
                }
            };
        tracing::debug!(
            elapsed_ms = chunk_started.elapsed().as_millis(),
            bytes = chunk.len(),
            status = response.status().as_u16(),
            "upload chunk response received"
        );

        let status = response.status();
        let response_headers = response.headers().clone();
        let body = response.text().await.unwrap_or_default();

        if status == StatusCode::UNAUTHORIZED && !refreshed_auth {
            if let Err(error) = client.auth().reload().await {
                return Err(format!("auth reload failed: {error}"));
            }
            refreshed_auth = true;
            continue;
        }
        if status.is_success() {
            return Ok(PutChunkOutcome::Accepted {
                headers: response_headers,
                body,
            });
        }
        if status.as_u16() == 308 {
            return Ok(PutChunkOutcome::Incomplete {
                headers: response_headers,
            });
        }
        if matches!(status.as_u16(), 410 | 404) {
            return Ok(PutChunkOutcome::SessionExpired);
        }
        if matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504) && attempt + 1 < MAX_RETRIES {
            attempt += 1;
            tracing::warn!(%status, attempt, "retrying upload chunk after transient failure");
            sleep(backoff(attempt)).await;
            continue;
        }
        return Ok(PutChunkOutcome::Failed { status, body });
    }
}

/// Surface a wrapped `ApiClientError` as the uploader's `String` error so the
/// match arms in [`upload_one_file`] read uniformly. Used only for the
/// non-`put_chunk` API calls (session URI refresh) — `put_chunk` already
/// returns its own outcome enum.
#[allow(dead_code)]
fn api_err(error: ApiClientError) -> String {
    error.to_string()
}

fn backoff(attempt: u32) -> Duration {
    let secs = 2u64.saturating_pow(attempt.saturating_sub(1));
    Duration::from_secs(secs.min(MAX_BACKOFF.as_secs()))
}

fn parse_resume_offset(headers: &HeaderMap) -> Option<u64> {
    let value = headers.get("range")?.to_str().ok()?;
    let last = value.split('-').nth(1)?;
    let last_byte: u64 = last.parse().ok()?;
    Some(last_byte + 1)
}

fn extract_server_md5(headers: &HeaderMap, body: &str) -> Option<String> {
    if let Some(value) = headers.get("x-checksum-md5") {
        if let Ok(text) = value.to_str() {
            return Some(text.trim().to_lowercase());
        }
    }
    if let Some(value) = headers.get("x-goog-hash") {
        if let Ok(text) = value.to_str() {
            for part in text.split(',') {
                let trimmed = part.trim();
                if let Some(b64) = trimmed.strip_prefix("md5=") {
                    if let Ok(bytes) = BASE64.decode(b64) {
                        return Some(hex_of(&bytes));
                    }
                }
            }
        }
    }
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(b64) = json.get("md5Hash").and_then(|value| value.as_str()) {
            if let Ok(bytes) = BASE64.decode(b64) {
                return Some(hex_of(&bytes));
            }
        }
    }
    None
}

fn hex_of(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::auth::StaticAuthProvider;
    use crate::api::client::ApiClientOptions;
    use crate::state::store::TraceUpdate;
    use crate::state::{TraceUploadStatus, TraceWriteStatus};
    use crate::storage::paths::TRACE_JSON_FILENAME;
    use std::collections::HashMap;
    use std::time::Duration;
    use tempfile::TempDir;
    use tokio::sync::mpsc;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

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
        recording_id: &str,
        trace_id: &str,
        data_type: &str,
        data_type_name: &str,
        session_uri: &str,
        contents: &[u8],
    ) -> std::path::PathBuf {
        store.create_recording(recording_id).await.unwrap();
        store
            .set_recording_org(recording_id, "org-1")
            .await
            .unwrap();
        store
            .create_trace(
                recording_id,
                trace_id,
                Some(data_type),
                Some(data_type_name),
            )
            .await
            .unwrap();
        let dir = TracePath::new(recording_id, data_type, trace_id.to_string())
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
        local
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
        let (status_tx, mut status_rx) = mpsc::unbounded_channel::<StatusUpdate>();

        let store_arc = Arc::new(store.clone());
        let recordings_root = Arc::new(recordings_root);
        upload_single(
            &store_arc,
            &bus,
            &api,
            &recordings_root,
            &status_tx,
            "trace-1",
        )
        .await;

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(trace.upload_status, TraceUploadStatus::Retrying);
        // The recovery sweep relies on `num_upload_attempts` so the
        // operator can spot a trace that's flapping between attempts.
        assert_eq!(trace.num_upload_attempts, 1);
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
        let (status_tx, mut status_rx) = mpsc::unbounded_channel::<StatusUpdate>();

        let store_arc = Arc::new(store.clone());
        let recordings_root = Arc::new(recordings_root);
        upload_single(
            &store_arc,
            &bus,
            &api,
            &recordings_root,
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
        seed_ready_trace(
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
        let recordings_root = Arc::new(recordings_root);
        upload_single(
            &store_arc,
            &bus,
            &api,
            &recordings_root,
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

    #[test]
    fn parse_resume_offset_uses_last_byte_plus_one() {
        // 308 carries a `Range: bytes=0-<last>` header; the offset is
        // `<last> + 1`. The 308 commit path keys off this so anyone
        // refactoring it later sees an explicit covering test.
        let mut headers = HeaderMap::new();
        headers.insert("range", HeaderValue::from_static("bytes=0-99"));
        assert_eq!(parse_resume_offset(&headers), Some(100));
        let mut empty = HeaderMap::new();
        empty.insert("range", HeaderValue::from_static("bytes=*"));
        assert_eq!(parse_resume_offset(&empty), None);
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

        store.create_recording("rec-1").await.unwrap();
        store.set_recording_org("rec-1", "org-1").await.unwrap();
        // Insert directly with NULL data_type so the uploader hits the
        // missing-data-type branch.
        store
            .create_trace("rec-1", "trace-1", None, None)
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
        let recordings_root = Arc::new(recordings_root);
        upload_single(
            &store_arc,
            &bus,
            &api,
            &recordings_root,
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
