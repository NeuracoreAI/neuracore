//! Shared HTTP client used by every cloud coordinator.
//!
//! Centralises the auth header, retry policy, timeouts, and base URL so each
//! coordinator (registration, uploader, status updater, progress reporter)
//! talks to the backend through a single configured instance. The retry
//! policy matches the Python `BACKEND_API_*` constants in `const.py`: max 3
//! attempts on `{408, 425, 429, 500..504}`, exponential backoff capped at
//! 30 s.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::{Client, Method, Request, Response, StatusCode};
use serde::Serialize;
use thiserror::Error;
use tokio::time::sleep;

use crate::api::auth::{AuthError, AuthProvider};
use crate::api::models::{
    BatchRegisterResponse, RegisterTraceRequest, ResumableUploadUrlResponse, TraceStatusUpdate,
};

/// Retry policy constants — match `const.py::BACKEND_API_*`.
pub const BACKEND_API_MAX_RETRIES: u32 = 3;
/// Cap for exponential backoff between retries (seconds).
pub const BACKEND_API_MAX_BACKOFF_SECONDS: u64 = 30;
/// Status codes the client retries on automatically.
pub const RETRYABLE_STATUS_CODES: &[u16] = &[408, 425, 429, 500, 502, 503, 504];

/// Construction-time configuration for [`ApiClient`].
#[derive(Debug, Clone)]
pub struct ApiClientOptions {
    /// Base URL, e.g. `https://api.neuracore.app/api`.
    pub base_url: String,
    /// Per-request timeout. Defaults to 30 seconds.
    pub timeout: Duration,
    /// Retry budget on retryable status codes.
    pub max_retries: u32,
    /// Cap on the exponential backoff between retries.
    pub max_backoff: Duration,
}

impl ApiClientOptions {
    /// Build options for the given backend base URL with the policy defaults.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            timeout: Duration::from_secs(30),
            max_retries: BACKEND_API_MAX_RETRIES,
            max_backoff: Duration::from_secs(BACKEND_API_MAX_BACKOFF_SECONDS),
        }
    }
}

/// Errors raised by the API client.
#[derive(Debug, Error)]
pub enum ApiClientError {
    /// Underlying transport failure (DNS, timeout, TLS, etc.).
    #[error(transparent)]
    Transport(#[from] reqwest::Error),
    /// Auth provider failed to supply a token (file missing, malformed, etc.).
    #[error(transparent)]
    Auth(#[from] AuthError),
    /// Non-retryable response status.
    #[error("backend responded with HTTP {status}: {body}")]
    Status {
        /// HTTP status code returned by the backend.
        status: StatusCode,
        /// Response body (truncated to a few KiB for the log line).
        body: String,
    },
    /// Response body did not deserialise.
    #[error("failed to decode backend response: {0}")]
    Decode(#[source] serde_json::Error),
    /// Response was missing a header the client expected.
    #[error("response missing required header {0}")]
    MissingHeader(&'static str),
}

/// Generic HTTP client wrapping a [`reqwest::Client`] with auth + retry.
pub struct ApiClient {
    inner: Client,
    options: ApiClientOptions,
    auth: Arc<dyn AuthProvider>,
}

impl ApiClient {
    /// Build a client with the given options and auth provider.
    pub fn new(
        options: ApiClientOptions,
        auth: Arc<dyn AuthProvider>,
    ) -> Result<Self, ApiClientError> {
        let inner = Client::builder().timeout(options.timeout).build()?;
        Ok(Self {
            inner,
            options,
            auth,
        })
    }

    /// Construct a client wrapping an externally-configured `reqwest::Client`.
    /// Used by tests that need to attach a `wiremock` URL.
    pub fn with_client(
        inner: Client,
        options: ApiClientOptions,
        auth: Arc<dyn AuthProvider>,
    ) -> Self {
        Self {
            inner,
            options,
            auth,
        }
    }

    /// Borrow the underlying reqwest client — exposed for the uploader, which
    /// PUTs chunks straight to GCS-issued URLs that are not relative to the
    /// configured `base_url`.
    pub fn raw_client(&self) -> &Client {
        &self.inner
    }

    /// Borrow the configured auth provider.
    pub fn auth(&self) -> &Arc<dyn AuthProvider> {
        &self.auth
    }

    /// Borrow the configured options.
    pub fn options(&self) -> &ApiClientOptions {
        &self.options
    }

    /// Build a URL beneath the configured `base_url`. The `path` is appended
    /// verbatim (and may start with `/`).
    pub fn url(&self, path: &str) -> String {
        if path.starts_with("http://") || path.starts_with("https://") {
            return path.to_string();
        }
        let base = self.options.base_url.trim_end_matches('/');
        if let Some(stripped) = path.strip_prefix('/') {
            format!("{base}/{stripped}")
        } else {
            format!("{base}/{path}")
        }
    }

    /// `HEAD /status/health` — used by the connection monitor.
    ///
    /// Returns `true` when the backend reports any non-5xx status, matching
    /// the Python `ConnectionManager._check_connectivity` semantics.
    pub async fn health_check(&self) -> Result<bool, ApiClientError> {
        let request = self.inner.head(self.url("/status/health")).build()?;
        let response = self.inner.execute(request).await?;
        Ok(response.status().as_u16() < 500)
    }

    /// `POST /org/{org}/recording/traces/batch-register`.
    pub async fn batch_register(
        &self,
        org_id: &str,
        traces: &[RegisterTraceRequest],
    ) -> Result<BatchRegisterResponse, ApiClientError> {
        let path = format!("/org/{org_id}/recording/traces/batch-register");
        #[derive(Serialize)]
        struct Body<'a> {
            traces: &'a [RegisterTraceRequest],
        }
        let body = Body { traces };
        let response = self
            .send_with_retry(Method::POST, &path, |builder| builder.json(&body))
            .await?;
        let bytes = response.bytes().await?;
        serde_json::from_slice::<BatchRegisterResponse>(&bytes).map_err(ApiClientError::Decode)
    }

    /// `GET /org/{org}/recording/{rec}/resumable_upload_url`.
    pub async fn fetch_resumable_upload_url(
        &self,
        org_id: &str,
        recording_id: &str,
        filepath: &str,
        content_type: &str,
    ) -> Result<String, ApiClientError> {
        let path = format!("/org/{org_id}/recording/{recording_id}/resumable_upload_url");
        let query = [("filepath", filepath), ("content_type", content_type)];
        let response = self
            .send_with_retry(Method::GET, &path, |builder| builder.query(&query))
            .await?;
        let bytes = response.bytes().await?;
        let parsed: ResumableUploadUrlResponse =
            serde_json::from_slice(&bytes).map_err(ApiClientError::Decode)?;
        Ok(parsed.url)
    }

    /// `PUT /org/{org}/recording/{rec}/traces/batch-update`.
    pub async fn batch_update_traces(
        &self,
        org_id: &str,
        recording_id: &str,
        updates: &HashMap<String, TraceStatusUpdate>,
    ) -> Result<(), ApiClientError> {
        let path = format!("/org/{org_id}/recording/{recording_id}/traces/batch-update");
        #[derive(Serialize)]
        struct Body<'a> {
            updates: &'a HashMap<String, TraceStatusUpdate>,
        }
        let body = Body { updates };
        let _ = self
            .send_with_retry(Method::PUT, &path, |builder| builder.json(&body))
            .await?;
        Ok(())
    }

    /// `POST /org/{org}/recording/{rec}/traces-metadata`.
    pub async fn report_progress(
        &self,
        org_id: &str,
        recording_id: &str,
        traces: &HashMap<String, i64>,
    ) -> Result<(), ApiClientError> {
        let path = format!("/org/{org_id}/recording/{recording_id}/traces-metadata");
        #[derive(Serialize)]
        struct Body<'a> {
            traces: &'a HashMap<String, i64>,
        }
        let body = Body { traces };
        let _ = self
            .send_with_retry(Method::POST, &path, |builder| builder.json(&body))
            .await?;
        Ok(())
    }

    /// `PUT /org/{org}/recording/{rec}/expected-trace-count`.
    ///
    /// Tells the backend how many traces to expect for this recording so it
    /// can promote the recording into its parent dataset once all traces are
    /// uploaded. Mirrors the Python `state_manager._set_expected_trace_count`
    /// flow — without this the recording stays hidden from
    /// `nc.get_dataset(...)` indefinitely even after every trace is uploaded.
    pub async fn put_expected_trace_count(
        &self,
        org_id: &str,
        recording_id: &str,
        expected_trace_count: i64,
    ) -> Result<(), ApiClientError> {
        let path = format!("/org/{org_id}/recording/{recording_id}/expected-trace-count");
        #[derive(Serialize)]
        struct Body {
            expected_trace_count: i64,
        }
        let body = Body {
            expected_trace_count,
        };
        let _ = self
            .send_with_retry(Method::PUT, &path, |builder| builder.json(&body))
            .await?;
        Ok(())
    }

    /// Send a request with the daemon's standard retry policy.
    ///
    /// `build` is invoked on a fresh `RequestBuilder` so the body / query
    /// closure is *re-evaluated* on every retry (`reqwest::RequestBuilder`
    /// captures the body up front; sharing one across retries would
    /// re-transmit the same buffer, which is what we want here).
    async fn send_with_retry<F>(
        &self,
        method: Method,
        path: &str,
        build: F,
    ) -> Result<Response, ApiClientError>
    where
        F: Fn(reqwest::RequestBuilder) -> reqwest::RequestBuilder,
    {
        let url = self.url(path);
        let mut refreshed_auth = false;
        let mut attempt: u32 = 0;
        loop {
            let headers = self.authorised_headers().await?;
            let builder = self
                .inner
                .request(method.clone(), &url)
                .headers(headers.clone());
            let builder = build(builder);
            let request: Request = builder.build()?;
            let response = self.inner.execute(request).await?;

            let status = response.status();
            if status == StatusCode::UNAUTHORIZED && !refreshed_auth {
                tracing::debug!(%url, "received 401, reloading auth token");
                self.auth.reload().await?;
                refreshed_auth = true;
                continue;
            }

            if status.is_success() {
                return Ok(response);
            }

            if RETRYABLE_STATUS_CODES.contains(&status.as_u16())
                && attempt + 1 < self.options.max_retries
            {
                attempt += 1;
                let backoff = self.backoff(attempt);
                tracing::warn!(
                    %url,
                    %status,
                    attempt,
                    "retrying after retryable status"
                );
                sleep(backoff).await;
                continue;
            }

            let body = response.text().await.unwrap_or_default();
            return Err(ApiClientError::Status { status, body });
        }
    }

    async fn authorised_headers(&self) -> Result<HeaderMap, ApiClientError> {
        let token = self.auth.bearer_token().await?;
        let mut headers = HeaderMap::new();
        let value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|_| {
            // The token came from user-controlled JSON, but a value that
            // cannot fit in a header byte string would mean the file is
            // corrupt — surface it as a Decode error so it shows up in the
            // tracing log alongside other parse failures.
            ApiClientError::Decode(serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bearer token contains invalid header characters",
            )))
        })?;
        headers.insert(AUTHORIZATION, value);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(headers)
    }

    fn backoff(&self, attempt: u32) -> Duration {
        let secs = 2u64.saturating_pow(attempt.saturating_sub(1));
        let capped = secs.min(self.options.max_backoff.as_secs().max(1));
        Duration::from_secs(capped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::auth::StaticAuthProvider;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn options(base_url: String) -> ApiClientOptions {
        ApiClientOptions {
            base_url,
            timeout: Duration::from_secs(5),
            max_retries: 3,
            // Tighten the backoff cap so retry-tests run inside their own
            // tokio time advance window without waiting real seconds.
            max_backoff: Duration::from_secs(1),
        }
    }

    fn client(server: &MockServer) -> ApiClient {
        let auth = Arc::new(StaticAuthProvider::new("test-token"));
        ApiClient::new(options(server.uri()), auth).expect("client")
    }

    #[tokio::test]
    async fn health_check_returns_true_on_2xx() {
        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .and(path("/status/health"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let client = client(&server);
        assert!(client.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn health_check_returns_false_on_5xx() {
        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .and(path("/status/health"))
            .respond_with(ResponseTemplate::new(503))
            .expect(1)
            .mount(&server)
            .await;

        let client = client(&server);
        assert!(!client.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn batch_register_round_trips_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/traces/batch-register"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "registered_traces": [{
                    "trace_id": "trace-1",
                    "upload_session_uris": {"rgb/cam_0/lossy.mp4": "https://upload.example/1"}
                }],
                "failed_traces": []
            })))
            .expect(1)
            .mount(&server)
            .await;
        let client = client(&server);

        let traces = vec![RegisterTraceRequest {
            recording_id: "rec-1".to_string(),
            data_type: "RGB_IMAGES".to_string(),
            trace_id: "trace-1".to_string(),
            cloud_files: vec![],
        }];
        let outcome = client.batch_register("org-1", &traces).await.unwrap();
        assert_eq!(outcome.registered_traces.len(), 1);
        assert_eq!(outcome.registered_traces[0].trace_id, "trace-1");
    }

    #[tokio::test]
    async fn retry_on_5xx_until_success() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/traces/batch-register"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/traces/batch-register"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "registered_traces": [], "failed_traces": []
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = client(&server);
        let result = client.batch_register("org-1", &[]).await.unwrap();
        assert!(result.registered_traces.is_empty());
    }

    #[tokio::test]
    async fn reloads_auth_on_401() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/org/org-1/recording/rec-1/resumable_upload_url"))
            .respond_with(ResponseTemplate::new(401))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/org/org-1/recording/rec-1/resumable_upload_url"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "url": "https://upload.example/abc"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let calls = Arc::new(AtomicUsize::new(0));
        struct CountingProvider {
            calls: Arc<AtomicUsize>,
        }
        #[async_trait::async_trait]
        impl AuthProvider for CountingProvider {
            async fn bearer_token(&self) -> Result<String, AuthError> {
                Ok("token".to_string())
            }
            async fn reload(&self) -> Result<(), AuthError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }
        let auth = Arc::new(CountingProvider {
            calls: Arc::clone(&calls),
        });
        let client = ApiClient::new(options(server.uri()), auth).unwrap();
        let url = client
            .fetch_resumable_upload_url("org-1", "rec-1", "path", "application/json")
            .await
            .unwrap();
        assert_eq!(url, "https://upload.example/abc");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn non_retryable_status_surfaces_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/org/org-1/recording/traces/batch-register"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .expect(1)
            .mount(&server)
            .await;

        let client = client(&server);
        let error = client.batch_register("org-1", &[]).await.unwrap_err();
        match error {
            ApiClientError::Status { status, body } => {
                assert_eq!(status, StatusCode::BAD_REQUEST);
                assert!(body.contains("bad request"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
