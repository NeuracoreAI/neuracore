//! Wire-level multipart upload transfer mechanics.
//!
//! [`super::uploader`] calls [`upload_one_file`] for each file. Parts that are
//! not saved yet are PUT in order to presigned URLs, and the ETag of each part
//! is saved. The upload is then completed with the saved parts and the CRC32C
//! of the store is compared with the CRC32C of the file.

use std::collections::btree_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use bytes::Bytes;
use reqwest::header::ETAG;
use reqwest::StatusCode;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::time::{sleep, timeout};

use super::status::StatusUpdate;
use crate::api::{ApiClient, UploadPartsRequest};
use crate::state::{
    DaemonEvent, EventBus, MultipartUpload, SqliteStateStore, StateStore, TraceRecord,
    TraceWriteHandle, UploadedParts,
};

/// Most parts one multipart upload can have.
const MAX_PARTS: u64 = 10_000;
/// Number of parts signed by one parts request. At the 32 MiB minimum part
/// size this is 256 MiB, which uploads well within the 7-day URL lifetime.
const PART_URL_WINDOW: usize = 8;
/// `bytes_uploaded` is saved every Nth part and when the file finishes. Resume
/// uses the saved parts, not this value.
const PROGRESS_PERSIST_EVERY_PARTS: u32 = 4;
/// Text in a 400 body when the store rejects an expired presigned URL. GCS
/// sends `ExpiredToken` and S3 sends `Request has expired`.
const EXPIRED_URL_ERRORS: [&str; 2] = ["ExpiredToken", "Request has expired"];
/// Error codes of a Complete request whose upload is gone or whose listed
/// parts the store does not hold.
const MISSING_UPLOAD_ERRORS: [&str; 3] = ["NoSuchUpload", "NoSuchPart", "InvalidPart"];
/// Response header with the object's CRC32C. A store that does not send it
/// puts `<ChecksumCRC32C>` in the response body.
const CRC32C_HEADER: &str = "x-amz-checksum-crc32c";
/// Maximum attempts for a single part or Complete request, and for refreshing
/// an expired part URL.
const MAX_RETRIES: u32 = 5;
/// Hard deadline for a single part PUT. The upload client has no timeout of
/// its own. A retry of a full part gets all of this time. A 32 MiB part needs
/// 0.67 Mbit/s to finish within it.
pub(crate) const PART_UPLOAD_TIMEOUT: Duration = Duration::from_secs(400);
/// Hard deadline for a single Complete request.
const COMPLETE_TIMEOUT: Duration = Duration::from_secs(200);
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
    /// The upload cannot complete with the file's bytes: the store no longer
    /// holds it, does not accept the saved parts, or completed it with another
    /// or no CRC32C. The caller restarts the trace's uploads.
    UploadMissing(String),
    /// The file needs more than [`MAX_PARTS`] parts: the caller fails the trace.
    TooManyParts(String),
}

impl From<String> for UploadFileError {
    fn from(message: String) -> Self {
        UploadFileError::Retry(message)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn upload_one_file(
    store: &Arc<SqliteStateStore>,
    client: &Arc<ApiClient>,
    trace_writer: &TraceWriteHandle,
    bus: &EventBus,
    status_tx: &tokio::sync::mpsc::UnboundedSender<StatusUpdate>,
    trace: &TraceRecord,
    org_id: &str,
    local_path: &std::path::Path,
    cloud_filepath: &str,
    upload: &MultipartUpload,
) -> Result<(), UploadFileError> {
    let total_bytes = tokio::fs::metadata(local_path)
        .await
        .map_err(|error| format!("stat {} failed: {error}", local_path.display()))?
        .len();
    let part_size = upload.part_size_bytes;
    let part_count = total_bytes.div_ceil(part_size).max(1);
    if part_count > MAX_PARTS {
        return Err(UploadFileError::TooManyParts(format!(
            "{} is {total_bytes} bytes, which needs {part_count} parts of {part_size} bytes; \
             an upload holds at most {MAX_PARTS} parts",
            local_path.display()
        )));
    }
    let trace_id = &trace.trace_id;
    let mut saved = store
        .get_uploaded_parts(trace_id, cloud_filepath)
        .await
        .map_err(|error| format!("failed to load the uploaded parts: {error}"))?;
    let mut file = File::open(local_path)
        .await
        .map_err(|error| format!("open {} failed: {error}", local_path.display()))?;
    let next_part = saved.keys().next_back().map_or(1, |last| last + 1);
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
        urls: upload
            .active_part_url
            .iter()
            .map(|url| (next_part, url.clone()))
            .collect(),
    };

    tracing::info!(
        trace_id,
        path = %local_path.display(),
        bytes = total_bytes,
        parts = part_count,
        saved_parts = saved.len(),
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
        // The CRC includes the saved parts.
        crc = crc32c::crc32c_append(crc, &part);
        if let Entry::Vacant(unsaved) = saved.entry(part_number) {
            let etag = parts.send(part_number, part).await?;
            let next_part_url = parts.urls.get(&(part_number + 1)).map(String::as_str);
            store
                .save_uploaded_part(trace_id, cloud_filepath, part_number, &etag, next_part_url)
                .await
                .map_err(|error| format!("failed to save part {part_number}: {error}"))?;
            unsaved.insert(etag);
        }
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

    let server_crc = complete_upload(client, &upload.complete_url, &saved, &parts.file).await?;
    if server_crc != crc {
        return Err(UploadFileError::UploadMissing(format!(
            "crc32c mismatch for {}: local={crc:#010x} server={server_crc:#010x}",
            local_path.display()
        )));
    }
    tracing::info!(
        trace_id,
        path = %local_path.display(),
        bytes = total_bytes,
        elapsed_ms = upload_started.elapsed().as_millis(),
        "file upload complete"
    );
    Ok(())
}

/// The file a part belongs to, for logs and performance events.
struct FileLabel<'a> {
    recording_index: i64,
    trace_id: &'a str,
    cloud_filepath: &'a str,
}

struct PartTracker<'a> {
    client: &'a ApiClient,
    org_id: &'a str,
    upload_id: &'a str,
    part_count: u32,
    file: FileLabel<'a>,
    urls: HashMap<u32, String>,
}

impl PartTracker<'_> {
    /// PUT `part` and return its ETag, fetching fresh URLs when none is known
    /// or the known one no longer works.
    async fn send(&mut self, part_number: u32, part: Bytes) -> Result<String, UploadFileError> {
        for _ in 0..MAX_RETRIES {
            if !self.urls.contains_key(&part_number) {
                self.fetch(part_number).await?;
            }
            let url = self
                .urls
                .get(&part_number)
                .ok_or_else(|| format!("backend returned no URL for part {part_number}"))?;
            if let Some(etag) = put_part(self.client, url, &part, &self.file).await? {
                return Ok(etag);
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

    /// Fetch URLs for the next [`PART_URL_WINDOW`] parts from `first`.
    async fn fetch(&mut self, first: u32) -> Result<(), UploadFileError> {
        let part_numbers = (first..=self.part_count).take(PART_URL_WINDOW).collect();
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
            return Err(UploadFileError::UploadMissing(format!(
                "the backend no longer holds upload {}",
                self.upload_id
            )));
        }
        let upload = response
            .uploads
            .into_iter()
            .find(|upload| upload.upload_id == self.upload_id)
            .ok_or_else(|| format!("parts response has no entry for upload {}", self.upload_id))?;
        self.urls = upload.part_urls;
        Ok(())
    }
}

/// Result of one PUT attempt of a part.
enum PartAttempt {
    Uploaded(String),
    /// 403, 404, or a 400 that names an expired token. The presigned URL has
    /// expired or its upload is gone. The parts route tells which.
    Expired,
    Transient(Transient),
    Rejected(String),
}

#[derive(Clone, Copy)]
enum Transient {
    Transport,
    Timeout,
    Status(StatusCode),
}

impl std::fmt::Display for Transient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Transient::Transport => formatter.write_str("transport_error"),
            Transient::Timeout => formatter.write_str("timeout"),
            Transient::Status(status) => write!(formatter, "http_{}", status.as_u16()),
        }
    }
}

/// True for the statuses a part PUT or Complete request is sent again for.
fn is_transient_status(status: StatusCode) -> bool {
    matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504)
}

/// Count a failed attempt of `request` and wait before the next one. Fails
/// once [`MAX_RETRIES`] attempts have failed.
async fn wait_to_retry(
    client: &ApiClient,
    file: &FileLabel<'_>,
    request: &str,
    attempt: &mut u32,
    reason: Transient,
) -> Result<(), String> {
    *attempt += 1;
    if *attempt >= MAX_RETRIES {
        return Err(format!(
            "{request} failed after {MAX_RETRIES} attempts: {reason}"
        ));
    }
    // Retry a timeout immediately. The timeout was the wait.
    let delay = match reason {
        Transient::Timeout => Duration::ZERO,
        Transient::Transport | Transient::Status(_) => {
            backoff(*attempt, client.options().max_backoff)
        }
    };
    tracing::warn!(attempt = *attempt, %reason, request, "retrying upload request");
    emit_upload_retry(file, *attempt, &reason.to_string(), delay);
    sleep(delay).await;
    Ok(())
}

/// PUT one part, retrying transport errors, timeouts and transient statuses.
/// Returns the part's ETag, or `None` when the URL has expired.
async fn put_part(
    client: &ApiClient,
    url: &str,
    part: &Bytes,
    file: &FileLabel<'_>,
) -> Result<Option<String>, String> {
    let mut attempt: u32 = 0;
    loop {
        let reason = match attempt_part(client, url, part, attempt).await {
            PartAttempt::Uploaded(etag) => return Ok(Some(etag)),
            PartAttempt::Expired => return Ok(None),
            PartAttempt::Rejected(message) => return Err(message),
            PartAttempt::Transient(reason) => reason,
        };
        wait_to_retry(client, file, "part PUT", &mut attempt, reason).await?;
    }
}

async fn attempt_part(client: &ApiClient, url: &str, part: &Bytes, attempt: u32) -> PartAttempt {
    let deadline = part_timeout(part.len(), attempt);
    // No Authorization header. The presigned URL carries its own signature.
    let send = client.raw_client().put(url).body(part.clone()).send();
    let response = match timeout(deadline, send).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            let error = error.without_url();
            tracing::warn!(error = ?error, attempt, "part PUT transport error");
            return PartAttempt::Transient(Transient::Transport);
        }
        Err(_elapsed) => {
            tracing::warn!(
                attempt,
                timeout_ms = deadline.as_millis(),
                bytes = part.len(),
                "part PUT timed out"
            );
            return PartAttempt::Transient(Transient::Timeout);
        }
    };
    let status = response.status();
    if status.is_success() {
        return match response
            .headers()
            .get(ETAG)
            .and_then(|etag| etag.to_str().ok())
        {
            Some(etag) => PartAttempt::Uploaded(etag.to_string()),
            None => {
                PartAttempt::Rejected(format!("part PUT returned HTTP {status} without an ETag"))
            }
        };
    }
    if matches!(status, StatusCode::FORBIDDEN | StatusCode::NOT_FOUND) {
        return PartAttempt::Expired;
    }
    if is_transient_status(status) {
        return PartAttempt::Transient(Transient::Status(status));
    }
    let body = response.text().await.unwrap_or_default();
    if status == StatusCode::BAD_REQUEST
        && EXPIRED_URL_ERRORS.iter().any(|error| body.contains(error))
    {
        return PartAttempt::Expired;
    }
    PartAttempt::Rejected(format!("part PUT failed with HTTP {status}: {body}"))
}

enum CompleteAttempt {
    /// The store completed the upload. Holds the CRC32C it returned, if any.
    Completed(Option<String>),
    /// The upload is gone, a listed part does not match, or the URL expired.
    Missing(String),
    Transient(Transient),
    Rejected(String),
}

/// Complete the upload with `parts`, retrying transport errors, timeouts and
/// transient statuses. Returns the CRC32C of the object.
async fn complete_upload(
    client: &ApiClient,
    url: &str,
    parts: &UploadedParts,
    file: &FileLabel<'_>,
) -> Result<u32, UploadFileError> {
    let body = complete_body(parts);
    let mut attempt: u32 = 0;
    loop {
        let reason = match attempt_complete(client, url, &body).await {
            CompleteAttempt::Completed(Some(crc32c)) => {
                return decode_crc32c(&crc32c).ok_or_else(|| {
                    UploadFileError::UploadMissing(format!(
                        "Complete of {} returned a malformed CRC32C {crc32c}",
                        file.cloud_filepath
                    ))
                });
            }
            CompleteAttempt::Completed(None) => {
                return Err(UploadFileError::UploadMissing(format!(
                    "Complete of {} returned no CRC32C",
                    file.cloud_filepath
                )));
            }
            CompleteAttempt::Missing(message) => {
                return Err(UploadFileError::UploadMissing(message))
            }
            CompleteAttempt::Rejected(message) => return Err(UploadFileError::Retry(message)),
            CompleteAttempt::Transient(reason) => reason,
        };
        wait_to_retry(client, file, "Complete", &mut attempt, reason).await?;
    }
}

async fn attempt_complete(client: &ApiClient, url: &str, body: &str) -> CompleteAttempt {
    // No Authorization header. The presigned URL carries its own signature.
    let send = client.raw_client().post(url).body(body.to_string()).send();
    let response = match timeout(COMPLETE_TIMEOUT, send).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            let error = error.without_url();
            tracing::warn!(error = ?error, "Complete transport error");
            return CompleteAttempt::Transient(Transient::Transport);
        }
        Err(_elapsed) => {
            tracing::warn!("Complete timed out");
            return CompleteAttempt::Transient(Transient::Timeout);
        }
    };
    let status = response.status();
    let header_crc32c = response
        .headers()
        .get(CRC32C_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let Ok(response_body) = response.text().await else {
        return CompleteAttempt::Transient(Transient::Transport);
    };
    // A store can answer 200 with an error in the body.
    let error_code = response_body
        .contains("<Error>")
        .then(|| xml_element(&response_body, "Code").unwrap_or_default());
    if status.is_success() && error_code.is_none() {
        return CompleteAttempt::Completed(
            header_crc32c
                .or_else(|| xml_element(&response_body, "ChecksumCRC32C").map(str::to_string)),
        );
    }
    if matches!(status, StatusCode::FORBIDDEN | StatusCode::NOT_FOUND)
        || error_code.is_some_and(|code| MISSING_UPLOAD_ERRORS.contains(&code))
        || EXPIRED_URL_ERRORS
            .iter()
            .any(|error| response_body.contains(error))
    {
        return CompleteAttempt::Missing(format!(
            "Complete failed with HTTP {status}: {response_body}"
        ));
    }
    if is_transient_status(status) {
        return CompleteAttempt::Transient(Transient::Status(status));
    }
    CompleteAttempt::Rejected(format!(
        "Complete failed with HTTP {status}: {response_body}"
    ))
}

fn complete_body(parts: &UploadedParts) -> String {
    let listed: String = parts
        .iter()
        .map(|(part_number, etag)| {
            format!(
                "<Part><PartNumber>{part_number}</PartNumber><ETag>{}</ETag></Part>",
                xml_escape(etag)
            )
        })
        .collect();
    format!("<CompleteMultipartUpload>{listed}</CompleteMultipartUpload>")
}

fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn xml_element<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = start + xml[start..].find(&format!("</{tag}>"))?;
    Some(xml[start..end].trim())
}

fn decode_crc32c(encoded: &str) -> Option<u32> {
    let bytes: [u8; 4] = BASE64.decode(encoded.trim()).ok()?.try_into().ok()?;
    Some(u32::from_be_bytes(bytes))
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

/// Retry delay: 1s, 2s, 4s and so on, at most `max`.
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

    const PART_SIZE: usize = 32 * 1024 * 1024;
    const ETAG_VALUE: &str = "\"etag-1\"";

    fn stored() -> ResponseTemplate {
        ResponseTemplate::new(200).insert_header("etag", ETAG_VALUE)
    }

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
        // 32 MiB at 1 Mbit/s is 268.4s; at 0.5 Mbit/s it is more than the cap.
        assert_eq!(part_timeout(PART_SIZE, 0).as_millis(), 270_435);
        for attempt in 1..MAX_RETRIES {
            assert_eq!(part_timeout(PART_SIZE, attempt), PART_UPLOAD_TIMEOUT);
        }
    }

    #[tokio::test]
    async fn put_part_retries_a_stalled_small_part_without_backoff() {
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/upload"))
            .respond_with(stored().set_delay(Duration::from_secs(30)))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .and(path("/upload"))
            .respond_with(stored())
            .mount(&server)
            .await;
        let client = client(server.uri());
        let url = format!("{}/upload", server.uri());
        let part = Bytes::from(vec![0u8; 19_291]);

        let started = Instant::now();
        let etag = put_part(&client, &url, &part, &file())
            .await
            .expect("put_part");

        assert_eq!(etag.as_deref(), Some(ETAG_VALUE));
        // The first deadline is about 2.2s. A 1s backoff makes this more than 3s.
        assert!(started.elapsed() < Duration::from_secs(3));
        assert_eq!(server.received_requests().await.expect("requests").len(), 2);
    }

    #[tokio::test]
    async fn attempt_part_classifies_every_status() {
        let cases = [
            ("200", stored(), "uploaded"),
            (
                "204",
                ResponseTemplate::new(204).insert_header("etag", ETAG_VALUE),
                "uploaded",
            ),
            ("200 without ETag", ResponseTemplate::new(200), "rejected"),
            ("403", ResponseTemplate::new(403), "expired"),
            ("404", ResponseTemplate::new(404), "expired"),
            (
                "400 ExpiredToken",
                ResponseTemplate::new(400).set_body_string("ExpiredToken"),
                "expired",
            ),
            (
                "400 Request has expired",
                ResponseTemplate::new(400).set_body_string("Request has expired"),
                "expired",
            ),
            ("429", ResponseTemplate::new(429), "transient"),
            ("500", ResponseTemplate::new(500), "transient"),
            ("502", ResponseTemplate::new(502), "transient"),
            ("503", ResponseTemplate::new(503), "transient"),
            ("504", ResponseTemplate::new(504), "transient"),
            (
                "400 unrelated body",
                ResponseTemplate::new(400).set_body_string("bad"),
                "rejected",
            ),
            ("401", ResponseTemplate::new(401), "rejected"),
            ("409", ResponseTemplate::new(409), "rejected"),
            ("422", ResponseTemplate::new(422), "rejected"),
        ];
        for (label, response, expected) in cases {
            let server = MockServer::start().await;
            Mock::given(method("PUT"))
                .and(path("/upload"))
                .respond_with(response)
                .mount(&server)
                .await;
            let url = format!("{}/upload", server.uri());

            let attempt =
                attempt_part(&client(server.uri()), &url, &Bytes::from_static(b"part"), 0).await;

            let actual = match attempt {
                PartAttempt::Uploaded(etag) => {
                    assert_eq!(etag, ETAG_VALUE, "status {label}");
                    "uploaded"
                }
                PartAttempt::Expired => "expired",
                PartAttempt::Transient(_) => "transient",
                PartAttempt::Rejected(_) => "rejected",
            };
            assert_eq!(actual, expected, "status {label}");
        }
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
                .write_all(b"HTTP/1.1 200 OK\r\netag: \"etag-1\"\r\ncontent-length: 0\r\n\r\n")
                .await
                .unwrap();
            body
        });
        let part = Bytes::from(vec![7u8; 256 * 1024]);

        let etag = put_part(
            &client(format!("http://{address}")),
            &format!("http://{address}/part/1"),
            &part,
            &file(),
        )
        .await
        .expect("the dropped part is sent again");

        assert_eq!(etag.as_deref(), Some(ETAG_VALUE));
        assert_eq!(stub.await.unwrap(), part.to_vec());
    }

    #[test]
    fn the_complete_body_lists_every_part_in_order_with_escaped_etags() {
        let parts =
            UploadedParts::from([(2, "\"etag-2\"".to_string()), (1, "\"a&b<c>\"".to_string())]);

        assert_eq!(
            complete_body(&parts),
            "<CompleteMultipartUpload>\
             <Part><PartNumber>1</PartNumber><ETag>&quot;a&amp;b&lt;c&gt;&quot;</ETag></Part>\
             <Part><PartNumber>2</PartNumber><ETag>&quot;etag-2&quot;</ETag></Part>\
             </CompleteMultipartUpload>"
        );
    }

    #[tokio::test]
    async fn attempt_complete_classifies_every_response() {
        let error = |status: u16, code: &str| {
            ResponseTemplate::new(status)
                .set_body_string(format!("<Error><Code>{code}</Code></Error>"))
        };
        let cases = [
            (
                "200 with the CRC in the header",
                ResponseTemplate::new(200)
                    .insert_header(CRC32C_HEADER, "4waSgw==")
                    .set_body_string("<CompleteMultipartUploadResult/>"),
                "completed 4waSgw==",
            ),
            (
                "200 with the CRC in the body",
                ResponseTemplate::new(200).set_body_string(
                    "<CompleteMultipartUploadResult>\
                     <ChecksumCRC32C>4waSgw==</ChecksumCRC32C>\
                     </CompleteMultipartUploadResult>",
                ),
                "completed 4waSgw==",
            ),
            (
                "200 with the CRC in the header and the body",
                ResponseTemplate::new(200)
                    .insert_header(CRC32C_HEADER, "4waSgw==")
                    .set_body_string("<ChecksumCRC32C>AAAAAA==</ChecksumCRC32C>"),
                "completed 4waSgw==",
            ),
            (
                "200 without a CRC",
                ResponseTemplate::new(200).set_body_string("<CompleteMultipartUploadResult/>"),
                "completed none",
            ),
            (
                "200 with InternalError",
                error(200, "InternalError"),
                "rejected",
            ),
            (
                "200 with NoSuchUpload",
                error(200, "NoSuchUpload"),
                "missing",
            ),
            ("404 NoSuchUpload", error(404, "NoSuchUpload"), "missing"),
            ("404 NoSuchPart", error(404, "NoSuchPart"), "missing"),
            ("400 InvalidPart", error(400, "InvalidPart"), "missing"),
            ("400 ExpiredToken", error(400, "ExpiredToken"), "missing"),
            ("403", ResponseTemplate::new(403), "missing"),
            ("429", ResponseTemplate::new(429), "transient"),
            ("500", ResponseTemplate::new(500), "transient"),
            ("502", ResponseTemplate::new(502), "transient"),
            ("503", ResponseTemplate::new(503), "transient"),
            ("504", ResponseTemplate::new(504), "transient"),
            ("400 MalformedXML", error(400, "MalformedXML"), "rejected"),
        ];
        for (label, response, expected) in cases {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/complete"))
                .respond_with(response)
                .mount(&server)
                .await;
            let url = format!("{}/complete", server.uri());

            let attempt = attempt_complete(&client(server.uri()), &url, "<body/>").await;

            let actual = match attempt {
                CompleteAttempt::Completed(crc32c) => {
                    format!("completed {}", crc32c.as_deref().unwrap_or("none"))
                }
                CompleteAttempt::Missing(_) => "missing".to_string(),
                CompleteAttempt::Transient(_) => "transient".to_string(),
                CompleteAttempt::Rejected(_) => "rejected".to_string(),
            };
            assert_eq!(actual, expected, "{label}");
        }
    }

    #[test]
    fn decode_crc32c_reads_base64_of_four_big_endian_bytes() {
        assert_eq!(decode_crc32c("4waSgw=="), Some(0xE306_9283));
        assert_eq!(decode_crc32c("AAAA"), None);
        assert_eq!(decode_crc32c("not base64"), None);
    }

    #[test]
    fn crc32c_matches_known_vector() {
        // Castagnoli CRC32C of the standard check string; guards against a
        // future swap to the wrong (zlib/ISO) polynomial, which would compile
        // fine but never match the store's checksum.
        assert_eq!(crc32c::crc32c(b"123456789"), 0xE306_9283);
    }
}
