//! Wire-level multipart upload transfer mechanics.
//!
//! The per-file and per-part PUT machinery the upload coordinator
//! ([`super::uploader`]) drives: [`upload_one_file`] reads a single on-disk
//! artefact part by part, folds every part into the file's CRC32C, and PUTs
//! the parts the store does not hold yet to presigned URLs from the backend's
//! parts route, refreshing the URLs when they expire.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use reqwest::StatusCode;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::time::{sleep, timeout};

use super::status::StatusUpdate;
use crate::api::{ApiClient, UploadPartsRequest};
use crate::state::{DaemonEvent, EventBus, MultipartUpload, TraceRecord, TraceWriteHandle};

/// Most parts one multipart upload can have.
const MAX_PARTS: u64 = 10_000;
/// Parts signed by one parts request. At the 16 MiB minimum part size this is
/// 128 MiB, which one part at a time sends well within the URLs' 30-minute
/// lifetime.
const PART_URL_WINDOW: usize = 8;
/// Persist `bytes_uploaded` to SQLite only every Nth part (plus once when the
/// file finishes), instead of every part. The per-part write took the store's
/// single `write_guard` once per 16 MiB and, at `MAX_CONCURRENT_UPLOADS`
/// in-flight files, serialised all uploads against each other and against the
/// notifiers/progress reporter — eroding the stop-recording SLA. Resume
/// correctness does not depend on this value: the parts route reports the
/// parts the store already holds, so a stale DB offset costs nothing.
const PROGRESS_PERSIST_EVERY_PARTS: u32 = 4;
/// What a 400 body holds when the store rejects an expired presigned URL: GCS
/// answers `ExpiredToken`, S3 `Request has expired`.
const EXPIRED_URL_ERRORS: [&str; 2] = ["ExpiredToken", "Request has expired"];
/// Maximum attempts for a single part, and for refreshing its expired URL.
const MAX_RETRIES: u32 = 5;
/// Hard deadline for a single part PUT. Belt-and-braces over the reqwest
/// client-level timeout. A retry of a full part gets all of this time.
pub(crate) const PART_UPLOAD_TIMEOUT: Duration = Duration::from_secs(200);
/// Latency time for each attempt of a part.
const PART_LATENCY_FLOOR: Duration = Duration::from_secs(2);
/// Minimum transfer speed for the first attempt, in bits per second.
const FIRST_ATTEMPT_MIN_SPEED_BPS: f64 = 1_000_000.0;
/// Lowest minimum transfer speed for a retry, in bits per second.
const RETRY_MIN_SPEED_BPS: f64 = 500_000.0;

/// Returns the time limit for one part PUT attempt.
///
/// The limit is the part transfer time at the minimum speed plus the latency
/// time. The minimum speed starts at 1 Mbit/s and halves on each retry, but it
/// is not less than 0.5 Mbit/s. The limit is not more than
/// [`PART_UPLOAD_TIMEOUT`].
fn part_timeout(len: usize, attempt: u32) -> Duration {
    let speed_bps =
        (FIRST_ATTEMPT_MIN_SPEED_BPS / 2f64.powi(attempt as i32)).max(RETRY_MIN_SPEED_BPS);
    let transfer = Duration::from_secs_f64(len as f64 * 8.0 / speed_bps);
    (transfer + PART_LATENCY_FLOOR).min(PART_UPLOAD_TIMEOUT)
}

/// Why [`upload_one_file`] gave up.
pub(crate) enum UploadFileError {
    /// Transient or unclassified: the caller rolls the trace back to `retrying`.
    Retry(String),
    /// The backend no longer holds the upload: the caller restarts the trace's
    /// uploads.
    UploadMissing,
    /// The file can never upload under this upload: the caller fails the trace.
    Failed(String),
}

impl From<String> for UploadFileError {
    fn from(message: String) -> Self {
        UploadFileError::Retry(message)
    }
}

/// Upload one file under its multipart upload and return the CRC32C of the
/// whole file.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn upload_one_file(
    client: &Arc<ApiClient>,
    trace_writer: &TraceWriteHandle,
    bus: &EventBus,
    status_tx: &tokio::sync::mpsc::UnboundedSender<StatusUpdate>,
    trace: &TraceRecord,
    org_id: &str,
    local_path: &std::path::Path,
    cloud_filepath: &str,
    upload: &MultipartUpload,
) -> Result<u32, UploadFileError> {
    let total_bytes = tokio::fs::metadata(local_path)
        .await
        .map_err(|error| format!("stat {} failed: {error}", local_path.display()))?
        .len();
    let part_size = upload.part_size_bytes;
    let part_count = total_bytes.div_ceil(part_size).max(1);
    if part_count > MAX_PARTS {
        return Err(UploadFileError::Failed(format!(
            "{} is {total_bytes} bytes, which needs {part_count} parts of {part_size} bytes; \
             an upload holds at most {MAX_PARTS} parts",
            local_path.display()
        )));
    }
    let mut file = File::open(local_path)
        .await
        .map_err(|error| format!("open {} failed: {error}", local_path.display()))?;
    let trace_id = &trace.trace_id;
    let mut parts = PartTracker {
        client,
        org_id,
        upload_id: &upload.upload_id,
        part_count: part_count as u32,
        file: FileLabel {
            recording_index: trace.recording_index,
            trace_id,
            cloud_filepath,
        },
        urls: HashMap::new(),
        uploaded: HashSet::new(),
    };

    tracing::info!(
        trace_id,
        path = %local_path.display(),
        bytes = total_bytes,
        parts = part_count,
        "starting file upload"
    );
    let upload_started = Instant::now();
    let mut crc: u32 = 0;
    let mut offset: u64 = 0;
    let mut parts_since_persist: u32 = 0;
    let mut last_persisted_offset: u64 = 0;
    for part_number in 1..=parts.part_count {
        let part_len = part_size.min(total_bytes - offset) as usize;
        let mut buffer = vec![0u8; part_len];
        file.read_exact(&mut buffer)
            .await
            .map_err(|error| format!("read {} failed: {error}", local_path.display()))?;
        let part = Bytes::from(buffer);
        // Every part feeds the CRC, including those the store already holds.
        crc = crc32c::crc32c_append(crc, &part);
        parts.send(part_number, part).await?;
        offset += part_len as u64;

        bus.publish(DaemonEvent::UploadProgress {
            trace_id: trace_id.clone(),
            recording_index: trace.recording_index,
            bytes_uploaded: offset as i64,
            total_bytes: Some(total_bytes as i64),
        });
        let _ = status_tx.send(StatusUpdate::in_progress(
            trace.recording_index,
            trace_id.clone(),
            offset as i64,
        ));
        // Persist the rolling progress on a coarse cadence (not every part):
        // the in-memory bus/status updates above are debounced downstream, and
        // only the SQLite write contends on the shared write_guard.
        parts_since_persist += 1;
        if parts_since_persist >= PROGRESS_PERSIST_EVERY_PARTS {
            trace_writer.upload_progress(trace_id, offset as i64);
            parts_since_persist = 0;
            last_persisted_offset = offset;
        }
    }
    // Persist the final offset once so the DB row reflects the completed bytes
    // even if the last persisted checkpoint was several parts back.
    if offset != last_persisted_offset {
        trace_writer.upload_progress(trace_id, offset as i64);
    }

    tracing::info!(
        trace_id,
        path = %local_path.display(),
        bytes = total_bytes,
        elapsed_ms = upload_started.elapsed().as_millis(),
        "file upload complete"
    );
    Ok(crc)
}

/// The file a part belongs to, for logs and performance events.
struct FileLabel<'a> {
    recording_index: i64,
    trace_id: &'a str,
    cloud_filepath: &'a str,
}

/// One file's upload as the transfer sees it: the part URLs fetched so far
/// and the parts the store holds.
struct PartTracker<'a> {
    client: &'a ApiClient,
    org_id: &'a str,
    upload_id: &'a str,
    part_count: u32,
    file: FileLabel<'a>,
    urls: HashMap<u32, String>,
    uploaded: HashSet<u32>,
}

impl PartTracker<'_> {
    /// PUT `part` unless the store already holds it, fetching fresh URLs when
    /// none is known or the known one has expired.
    async fn send(&mut self, part_number: u32, part: Bytes) -> Result<(), UploadFileError> {
        for _ in 0..MAX_RETRIES {
            if !self.uploaded.contains(&part_number) && !self.urls.contains_key(&part_number) {
                self.fetch(part_number).await?;
            }
            if self.uploaded.contains(&part_number) {
                return Ok(());
            }
            let url = self
                .urls
                .get(&part_number)
                .ok_or_else(|| format!("backend returned no URL for part {part_number}"))?;
            if put_part(self.client, url, &part, &self.file).await? {
                return Ok(());
            }
            tracing::info!(
                trace_id = self.file.trace_id,
                part_number,
                "part URL expired; requesting fresh URLs"
            );
            emit_upload_retry(&self.file, 0, "url_expired", Duration::ZERO);
            self.urls.clear();
        }
        Err(UploadFileError::Retry(format!(
            "part {part_number} of {} got an expired URL {MAX_RETRIES} times",
            self.file.cloud_filepath
        )))
    }

    /// Fetch URLs for the next [`PART_URL_WINDOW`] parts from `first` that the
    /// store does not hold, and learn every part it holds.
    async fn fetch(&mut self, first: u32) -> Result<(), UploadFileError> {
        let part_numbers = (first..=self.part_count)
            .filter(|number| !self.uploaded.contains(number))
            .take(PART_URL_WINDOW)
            .collect();
        let request = UploadPartsRequest {
            upload_id: self.upload_id.to_string(),
            part_numbers,
        };
        let response = self
            .client
            .upload_parts(self.org_id, &[request])
            .await
            .map_err(|error| format!("failed to fetch part URLs: {error}"))?;
        if response
            .missing_upload_ids
            .iter()
            .any(|id| id == self.upload_id)
        {
            return Err(UploadFileError::UploadMissing);
        }
        let upload = response
            .uploads
            .into_iter()
            .find(|upload| upload.upload_id == self.upload_id)
            .ok_or_else(|| format!("parts response has no entry for upload {}", self.upload_id))?;
        self.urls = upload.part_urls;
        self.uploaded.extend(upload.uploaded_parts);
        Ok(())
    }
}

/// What one PUT attempt of a part came to.
enum PartAttempt {
    Uploaded,
    /// 403, or a 400 naming an expired token: the presigned URL has expired.
    Expired,
    /// Worth sending again; `backoff` says whether to wait first.
    Transient {
        reason: String,
        backoff: bool,
    },
    Rejected(String),
}

/// PUT one part, retrying transport errors, timeouts and transient statuses.
/// Returns `false` when the URL has expired.
async fn put_part(
    client: &ApiClient,
    url: &str,
    part: &Bytes,
    file: &FileLabel<'_>,
) -> Result<bool, String> {
    let mut attempt: u32 = 0;
    loop {
        let (reason, wait) = match attempt_part(client, url, part, attempt).await {
            PartAttempt::Uploaded => return Ok(true),
            PartAttempt::Expired => return Ok(false),
            PartAttempt::Rejected(message) => return Err(message),
            PartAttempt::Transient { reason, backoff } => (reason, backoff),
        };
        attempt += 1;
        if attempt >= MAX_RETRIES {
            return Err(format!(
                "part PUT failed after {MAX_RETRIES} attempts: {reason}"
            ));
        }
        let delay = if wait {
            backoff(attempt, client.options().max_backoff)
        } else {
            Duration::ZERO
        };
        tracing::warn!(attempt, reason, "retrying part PUT");
        emit_upload_retry(file, attempt, &reason, delay);
        sleep(delay).await;
    }
}

async fn attempt_part(client: &ApiClient, url: &str, part: &Bytes, attempt: u32) -> PartAttempt {
    let deadline = part_timeout(part.len(), attempt);
    // No Authorization header: the presigned URL carries its own signature.
    // `Bytes` is cheaply cloneable (Arc-backed), so a retry is a refcount bump.
    let send = client.raw_client().put(url).body(part.clone()).send();
    let response = match timeout(deadline, send).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            let error = error.without_url();
            tracing::warn!(error = ?error, attempt, "part PUT transport error");
            return PartAttempt::Transient {
                reason: "transport_error".to_string(),
                backoff: true,
            };
        }
        // Retry immediately. The timeout was the wait.
        Err(_elapsed) => {
            tracing::warn!(
                attempt,
                timeout_ms = deadline.as_millis(),
                bytes = part.len(),
                "part PUT timed out"
            );
            return PartAttempt::Transient {
                reason: "timeout".to_string(),
                backoff: false,
            };
        }
    };
    let status = response.status();
    if status.is_success() {
        return PartAttempt::Uploaded;
    }
    if status == StatusCode::FORBIDDEN {
        return PartAttempt::Expired;
    }
    if matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504) {
        return PartAttempt::Transient {
            reason: format!("http_{}", status.as_u16()),
            backoff: true,
        };
    }
    let body = response.text().await.unwrap_or_default();
    if status == StatusCode::BAD_REQUEST
        && EXPIRED_URL_ERRORS.iter().any(|error| body.contains(error))
    {
        return PartAttempt::Expired;
    }
    PartAttempt::Rejected(format!("part PUT failed with HTTP {status}: {body}"))
}

fn emit_upload_retry(file: &FileLabel<'_>, attempt: u32, reason: &str, delay: Duration) {
    crate::perf_events::emit(
        "retry_backoff",
        "scheduled",
        Some(file.recording_index),
        Some(file.trace_id),
        None,
        serde_json::json!({
            "retry_domain": "upload",
            "attempt": attempt,
            "reason": reason,
            "filename": file.cloud_filepath,
            "backoff_ms": delay.as_secs_f64() * 1_000.0,
        }),
    );
}

/// Exponential backoff (1s, 2s, 4s, …) for a part retry, capped at the API
/// client's configured `max_backoff`. At [`MAX_RETRIES`] the cap never binds in
/// production (the largest delay reached is 8s), but exposing it lets tests
/// drive the retry path without real-time sleeps.
fn backoff(attempt: u32, max: Duration) -> Duration {
    let secs = 2u64.saturating_pow(attempt.saturating_sub(1));
    Duration::from_secs(secs).min(max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::auth::StaticAuthProvider;
    use crate::api::ApiClientOptions;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const PART_SIZE: usize = 16 * 1024 * 1024;

    fn client(base_url: String) -> ApiClient {
        let mut options = ApiClientOptions::new(base_url);
        options.max_backoff = Duration::from_millis(10);
        ApiClient::new(options, Arc::new(StaticAuthProvider::new("test-token"))).expect("client")
    }

    fn file() -> FileLabel<'static> {
        FileLabel {
            recording_index: 1,
            trace_id: "trace",
            cloud_filepath: "RGB_IMAGES/cam0/lossy.mp4",
        }
    }

    #[test]
    fn part_timeout_small_part_halves_speed_once() {
        // 19,291 bytes at 1 Mbit/s, then 0.5 Mbit/s, plus 2s latency.
        assert_eq!(part_timeout(19_291, 0).as_millis(), 2_154);
        for attempt in 1..MAX_RETRIES {
            assert_eq!(part_timeout(19_291, attempt).as_millis(), 2_308);
        }
    }

    #[test]
    fn part_timeout_full_part_is_capped_on_retry() {
        // 16 MiB at 1 Mbit/s is 134.2s; at 0.5 Mbit/s it is more than the cap.
        assert_eq!(part_timeout(PART_SIZE, 0).as_millis(), 136_217);
        for attempt in 1..MAX_RETRIES {
            assert_eq!(part_timeout(PART_SIZE, attempt), PART_UPLOAD_TIMEOUT);
        }
    }

    #[tokio::test]
    async fn put_part_retries_a_stalled_small_part_without_backoff() {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/upload"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(30)))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .and(path("/upload"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;
        let client = client(server.uri());
        let url = format!("{}/upload", server.uri());
        let part = Bytes::from(vec![0u8; 19_291]);

        let started = Instant::now();
        let uploaded = put_part(&client, &url, &part, &file())
            .await
            .expect("put_part");

        assert!(uploaded);
        // The first deadline is about 2.2s. A 1s backoff makes this more than 3s.
        assert!(started.elapsed() < Duration::from_secs(3));
        assert_eq!(server.received_requests().await.expect("requests").len(), 2);
    }

    /// Read one HTTP request's head and return its `Content-Length`.
    async fn read_request_head(stream: &mut tokio::net::TcpStream) -> usize {
        let mut head = Vec::new();
        while !head.ends_with(b"\r\n\r\n") {
            head.push(stream.read_u8().await.expect("request head"));
        }
        let head = String::from_utf8(head).expect("utf-8 head");
        assert!(
            !head.to_ascii_lowercase().contains("authorization:"),
            "a presigned PUT carries no Authorization header"
        );
        let head = head.to_ascii_lowercase();
        let length = head
            .lines()
            .find_map(|line| line.strip_prefix("content-length:"))
            .expect("content-length");
        length.trim().parse().expect("numeric content-length")
    }

    #[tokio::test]
    async fn put_part_resends_a_part_whose_connection_dropped_mid_body() {
        // A presigned-PUT stub that drops the first connection halfway through
        // the body, then accepts the part whole on the next one.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let stub = tokio::spawn(async move {
            let (mut dropped, _) = listener.accept().await.unwrap();
            let length = read_request_head(&mut dropped).await;
            let mut half = vec![0u8; length / 2];
            dropped.read_exact(&mut half).await.unwrap();
            drop(dropped);

            let (mut accepted, _) = listener.accept().await.unwrap();
            let length = read_request_head(&mut accepted).await;
            let mut body = vec![0u8; length];
            accepted.read_exact(&mut body).await.unwrap();
            accepted
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
                .await
                .unwrap();
            body
        });
        let part = Bytes::from(vec![7u8; 256 * 1024]);

        let uploaded = put_part(
            &client(format!("http://{address}")),
            &format!("http://{address}/part/1"),
            &part,
            &file(),
        )
        .await
        .expect("the dropped part is sent again");

        assert!(uploaded);
        assert_eq!(stub.await.unwrap(), part.to_vec());
    }

    #[test]
    fn crc32c_matches_known_vector() {
        // Castagnoli CRC32C of the standard check string; guards against a
        // future swap to the wrong (zlib/ISO) polynomial, which would compile
        // fine but never match the store's checksum.
        assert_eq!(crc32c::crc32c(b"123456789"), 0xE306_9283);
    }
}
