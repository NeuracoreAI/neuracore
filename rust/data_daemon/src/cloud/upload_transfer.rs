//! Wire-level resumable-upload transfer mechanics.
//!
//! The per-file and per-chunk PUT machinery the upload coordinator
//! ([`super::uploader`]) drives: [`upload_one_file`] streams a single on-disk
//! artefact as 16 MiB chunks to the GCS resumable session URI, handling the
//! 308-continue, 410-session-expired, and 401-auth-refresh transitions, and
//! verifies the server-side MD5 on completion.

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
use tokio::time::{sleep, timeout};

use crate::api::ApiClient;
use crate::cloud::status::StatusUpdate;
use crate::cloud::OrgIdRx;
use crate::state::{DaemonEvent, EventBus, TraceRecord, TraceWriteHandle};

/// Chunk size used for resumable uploads.
///
/// Must be a multiple of 256 KiB (the GCS resumable-upload requirement for
/// every non-final chunk); 16 MiB = 64 × 256 KiB. Larger chunks raise peak
/// throughput on fast links (fewer sequential PUTs) at the cost of coarser
/// upload-progress granularity, higher per-upload memory, and a higher minimum
/// sustained speed: a chunk must transfer within `CHUNK_UPLOAD_TIMEOUT` (200 s),
/// so the minimum sustained speed is `CHUNK_SIZE / CHUNK_UPLOAD_TIMEOUT`
/// (16 MiB / 200 s ≈ 0.67 Mbit/s).
pub const CHUNK_SIZE: usize = 16 * 1024 * 1024;
/// Persist `bytes_uploaded` to SQLite only every Nth chunk (plus once when the
/// file finishes), instead of every chunk. The per-chunk write took the store's
/// single `write_guard` once per 16 MiB and, at `MAX_CONCURRENT_UPLOADS`
/// in-flight files, serialised all uploads against each other and against the
/// notifiers/progress reporter — eroding the stop-recording SLA. Resume
/// correctness does not depend on this value: the 308-continue path
/// (`parse_resume_offset`) re-derives the committed offset from the server on
/// restart, so a stale DB offset only costs re-sending at most this many chunks.
const PROGRESS_PERSIST_EVERY_CHUNKS: u32 = 4;
/// Cap on the exponential backoff for transient upload failures.
pub const MAX_BACKOFF: Duration = Duration::from_secs(300);
/// Maximum retries for a single chunk.
pub const MAX_RETRIES: u32 = 5;
/// Hard deadline for a single chunk PUT. Belt-and-braces over the reqwest
/// client-level timeout, which can silently fail to fire for direct GCS
/// resumable session URI uploads.
const CHUNK_UPLOAD_TIMEOUT: Duration = Duration::from_secs(200);

/// Outcome of [`upload_one_file`]. Carries a refreshed `session_uri` when the
/// server expired the original one mid-upload so the caller can persist it
/// for restart-resume.
pub(crate) struct UploadFileOutcome {
    pub(crate) bytes_uploaded: i64,
    pub(crate) final_session_uri: Option<String>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn upload_one_file(
    client: &Arc<ApiClient>,
    trace_writer: &TraceWriteHandle,
    bus: &EventBus,
    org_rx: &OrgIdRx,
    status_tx: &tokio::sync::mpsc::UnboundedSender<StatusUpdate>,
    trace: &TraceRecord,
    recording_id: &str,
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
    let recording_index = trace.recording_index;
    let trace_id = trace.trace_id.clone();
    let Some(org_id) = org_rx.borrow().clone() else {
        return Err("no current org_id configured; cannot refresh session URI".to_string());
    };

    tracing::info!(
        trace_id,
        path = %local_path.display(),
        bytes = total_bytes,
        "starting file upload"
    );
    let upload_started = Instant::now();
    let mut chunks_since_persist: u32 = 0;
    let mut last_persisted_offset: u64 = 0;
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
                // 308 — the server reports how much it actually committed via
                // the Range header. GCS commits in 256 KiB units, so it can
                // accept only a *prefix* of a 4 MiB chunk. We must hash exactly
                // the committed prefix: hashing the whole chunk then resuming at
                // `server_offset` re-reads — and re-hashes — the uncommitted
                // tail on the next iteration, double-counting it into the local
                // MD5 and failing the final compare with a spurious mismatch.
                let server_offset = parse_resume_offset(&headers).unwrap_or(offset);
                match resume_decision(offset, chunk_len, server_offset) {
                    ResumeDecision::Behind => {
                        return Err(format!(
                            "server resume offset {server_offset} is behind local offset \
                             {offset}; refusing to corrupt {}",
                            local_path.display()
                        ));
                    }
                    ResumeDecision::Ahead { new_offset } => {
                        // Server has bytes we didn't send this session (e.g. a
                        // prior session) and can't re-hash — accept its view but
                        // flag the local checksum untrustworthy.
                        tracing::warn!(
                            server_offset,
                            local_offset = offset + chunk_len as u64,
                            path = %local_path.display(),
                            "server resume offset is ahead of local; skipping local checksum"
                        );
                        hasher.update(&chunk);
                        server_md5_hex = None;
                        offset = new_offset;
                    }
                    ResumeDecision::Committed {
                        hash_len,
                        new_offset,
                    } => {
                        // Fold in only the committed prefix; the tail is re-sent
                        // (and hashed) on the next read, so every byte is hashed
                        // exactly once.
                        hasher.update(&chunk[..hash_len]);
                        offset = new_offset;
                    }
                }
            }
            PutChunkOutcome::SessionExpired => {
                tracing::info!(
                    trace_id,
                    path = %local_path.display(),
                    "upload session expired; requesting fresh URI"
                );
                match client
                    .fetch_resumable_upload_url(&org_id, recording_id, cloud_filepath, content_type)
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
            recording_index,
            bytes_uploaded: offset as i64,
            total_bytes: Some(total_bytes as i64),
        });
        let _ = status_tx.send(StatusUpdate::in_progress(
            recording_index,
            trace_id.clone(),
            offset as i64,
        ));
        // Persist the rolling progress on a coarse cadence (not every chunk):
        // the in-memory bus/status updates above are debounced downstream, and
        // only the SQLite write contends on the shared write_guard. Resume
        // correctness comes from the server's 308 offset, not this row.
        chunks_since_persist += 1;
        if chunks_since_persist >= PROGRESS_PERSIST_EVERY_CHUNKS {
            persist_upload_offset(trace_writer, &trace_id, offset);
            chunks_since_persist = 0;
            last_persisted_offset = offset;
        }
    }

    // Persist the final offset once so the DB row reflects the completed bytes
    // even if the last persisted checkpoint was several chunks back.
    if offset != last_persisted_offset {
        persist_upload_offset(trace_writer, &trace_id, offset);
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

/// Persist the rolling `bytes_uploaded` checkpoint for a trace via the
/// coalescing write-behind — fire-and-forget, so a burst of concurrent uploads
/// collapses to one batched row write per flush tick instead of a synchronous
/// transaction each. A missed checkpoint only costs re-sending a few chunks on
/// restart, never correctness (resume uses the server's 308 offset).
fn persist_upload_offset(trace_writer: &TraceWriteHandle, trace_id: &str, offset: u64) {
    trace_writer.upload_progress(trace_id, offset as i64);
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
        // same chunk on retry is a refcount bump, not a 4 MiB copy.
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

/// How a 308's committed `server_offset` reconciles against the just-sent chunk
/// `[offset, offset + chunk_len)`.
#[derive(Debug, PartialEq, Eq)]
enum ResumeDecision {
    /// Server is behind our local offset — would corrupt the object; abort.
    Behind,
    /// Server is ahead of anything we sent this session (bytes we can't
    /// re-hash); accept its offset but treat the local checksum as unusable.
    Ahead { new_offset: u64 },
    /// Server committed `hash_len` bytes of this chunk; fold exactly that prefix
    /// into the running MD5 and resume from `new_offset`.
    Committed { hash_len: usize, new_offset: u64 },
}

/// Decide how many bytes of the just-sent chunk the running MD5 should absorb
/// after a 308, given the server's committed `server_offset`. Hashing only the
/// committed prefix is what keeps every byte hashed exactly once across a
/// partial (sub-chunk) commit and the resend of its tail.
fn resume_decision(offset: u64, chunk_len: usize, server_offset: u64) -> ResumeDecision {
    if server_offset < offset {
        ResumeDecision::Behind
    } else if server_offset > offset + chunk_len as u64 {
        ResumeDecision::Ahead {
            new_offset: server_offset,
        }
    } else {
        ResumeDecision::Committed {
            hash_len: (server_offset - offset) as usize,
            new_offset: server_offset,
        }
    }
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
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        // Write straight into the buffer — `push_str(&format!(..))` would heap-
        // allocate a throwaway `String` per byte.
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn resume_full_chunk_commit_hashes_whole_chunk() {
        // Server committed the entire chunk → hash all of it, advance fully.
        assert_eq!(
            resume_decision(0, CHUNK_SIZE, CHUNK_SIZE as u64),
            ResumeDecision::Committed {
                hash_len: CHUNK_SIZE,
                new_offset: CHUNK_SIZE as u64,
            }
        );
    }

    #[test]
    fn resume_partial_commit_hashes_only_committed_prefix() {
        // M7 regression: GCS commits in 256 KiB units, so a 4 MiB chunk can be
        // committed only up to, say, 4 MiB − 256 KiB. We must hash exactly that
        // committed prefix — NOT the whole chunk — or the re-sent tail is hashed
        // twice and the final MD5 spuriously mismatches.
        let committed = (CHUNK_SIZE - 256 * 1024) as u64;
        assert_eq!(
            resume_decision(0, CHUNK_SIZE, committed),
            ResumeDecision::Committed {
                hash_len: committed as usize,
                new_offset: committed,
            }
        );
    }

    #[test]
    fn resume_zero_advance_hashes_nothing() {
        // Server has nothing yet (missing/zero Range) → hash nothing, retry the
        // same offset; otherwise the whole chunk would be double-hashed.
        assert_eq!(
            resume_decision(100, CHUNK_SIZE, 100),
            ResumeDecision::Committed {
                hash_len: 0,
                new_offset: 100,
            }
        );
    }

    #[test]
    fn resume_ahead_marks_checksum_untrustworthy() {
        assert_eq!(
            resume_decision(0, CHUNK_SIZE, CHUNK_SIZE as u64 + 1),
            ResumeDecision::Ahead {
                new_offset: CHUNK_SIZE as u64 + 1,
            }
        );
    }

    #[test]
    fn resume_behind_is_a_corruption_guard() {
        assert_eq!(resume_decision(100, CHUNK_SIZE, 50), ResumeDecision::Behind);
    }
}
