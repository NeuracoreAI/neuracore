//! Auth provider for the Neuracore API client.
//!
//! Reads the API key from `~/.neuracore/config.json` — the same file the
//! Python SDK writes after a successful `nc.login()` — and exchanges it for a
//! short-lived JWT via `POST {api_url}/auth/verify-api-key`. The JWT is the
//! actual bearer token sent to the backend; `nrc_…` API keys are not accepted
//! directly. If the file already contains an `access_token` (set by tooling
//! that does the exchange itself), the provider uses it verbatim and skips
//! the exchange. On a 401 response the API client calls
//! [`AuthProvider::reload`], which drops the cached JWT and forces a fresh
//! exchange on the next call.
//!
//! Tests rely on a custom provider via [`AuthProvider`] (the trait) so we can
//! inject a fixed token without touching the user's home directory.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use thiserror::Error;
use tokio::fs;
use tokio::sync::Mutex;

/// Errors surfaced by the auth provider.
#[derive(Debug, Error)]
pub enum AuthError {
    /// Underlying I/O error reading the config file.
    #[error("failed to read auth config {path}: {source}")]
    Io {
        /// File the provider tried to open.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Config file present but did not deserialise.
    #[error("failed to parse auth config {path}: {source}")]
    Parse {
        /// File the provider tried to parse.
        path: PathBuf,
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
    },
    /// Config loaded but contained no API key / access token.
    #[error("auth config {path} is missing an access token")]
    Missing {
        /// File the provider read from.
        path: PathBuf,
    },
    /// `verify-api-key` request failed at the transport level.
    #[error("verify-api-key request to {url} failed: {source}")]
    ExchangeTransport {
        /// URL the provider tried to reach.
        url: String,
        /// Underlying reqwest error.
        #[source]
        source: reqwest::Error,
    },
    /// `verify-api-key` returned a non-2xx response.
    #[error("verify-api-key at {url} returned HTTP {status}: {body}")]
    ExchangeStatus {
        /// URL the provider tried to reach.
        url: String,
        /// HTTP status the backend returned.
        status: StatusCode,
        /// Response body (capped to a few KiB by the caller).
        body: String,
    },
    /// `verify-api-key` returned a 2xx response without an `access_token`.
    #[error("verify-api-key at {url} returned no access_token")]
    ExchangeMissingToken {
        /// URL the provider tried to reach.
        url: String,
    },
}

/// Trait implemented by every auth source — the file-backed implementation in
/// production, and the in-memory stub used by tests.
#[async_trait]
pub trait AuthProvider: Send + Sync {
    /// Return the current bearer token. Cached internally; cheap to call.
    async fn bearer_token(&self) -> Result<String, AuthError>;

    /// Drop the cached token and re-load on the next call. Invoked by the
    /// HTTP client after a 401 response.
    async fn reload(&self) -> Result<(), AuthError>;
}

/// On-disk config shape — matches `neuracore.core.config.config_manager.Config`.
///
/// The Python SDK writes the `api_key` field after a successful `nc.login()`
/// and trades it for an in-memory `access_token` via `auth/verify-api-key`.
/// The daemon does the same exchange at the boundary so it never relies on
/// the SDK persisting a JWT to disk (it doesn't). A pre-populated
/// `access_token` is still honoured for tests and tooling that wants to
/// bypass the exchange.
#[derive(Debug, Default, Deserialize)]
struct AuthConfig {
    api_key: Option<String>,
    #[serde(default)]
    access_token: Option<String>,
}

/// Response body for `POST /auth/verify-api-key`. Matches the
/// `neuracore.core.auth.AccessTokenResponse` shape on the Python side.
#[derive(Debug, Deserialize)]
struct VerifyApiKeyResponse {
    #[serde(default)]
    access_token: Option<String>,
}

/// HTTP timeout for the verify-api-key exchange. Matches the default
/// per-request budget of the main API client so a stalled identity service
/// can't pin the registration coordinator indefinitely.
const VERIFY_API_KEY_TIMEOUT: Duration = Duration::from_secs(30);
/// Cap on the response body the provider reads back when surfacing a non-2xx
/// error; the JSON payload is ~100 bytes and a runaway HTML page from a
/// misconfigured proxy could otherwise blow up the trace log line.
const VERIFY_API_KEY_ERROR_BODY_LIMIT: usize = 4096;

/// Default auth source: reads `~/.neuracore/config.json` lazily, exchanges
/// the API key for a JWT, and caches the JWT until [`AuthProvider::reload`]
/// is called.
pub struct FileAuthProvider {
    path: PathBuf,
    api_url: String,
    http: Client,
    cached: Mutex<Option<String>>,
}

impl FileAuthProvider {
    /// Build a provider that reads from `path` and exchanges the API key via
    /// `{api_url}/auth/verify-api-key`.
    pub fn new(path: impl Into<PathBuf>, api_url: impl Into<String>) -> Result<Self, AuthError> {
        // A fresh `reqwest::Client` per provider instance is fine — the
        // provider itself is a long-lived `Arc` shared by every coordinator,
        // so the underlying connection pool is reused across the daemon's
        // lifetime.
        let http = Client::builder()
            .timeout(VERIFY_API_KEY_TIMEOUT)
            .build()
            .map_err(|source| AuthError::ExchangeTransport {
                url: String::new(),
                source,
            })?;
        Ok(Self {
            path: path.into(),
            api_url: api_url.into(),
            http,
            cached: Mutex::new(None),
        })
    }

    /// Build a provider that reads from `~/.neuracore/config.json`.
    #[allow(dead_code)]
    pub fn default_path(api_url: impl Into<String>) -> Result<Self, AuthError> {
        let path = dirs::home_dir()
            .map(|home| home.join(".neuracore").join("config.json"))
            .unwrap_or_else(|| PathBuf::from(".neuracore/config.json"));
        Self::new(path, api_url)
    }

    /// Borrow the file path the provider reads.
    #[allow(dead_code)]
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    async fn read_config(&self) -> Result<AuthConfig, AuthError> {
        let bytes = fs::read(&self.path).await.map_err(|source| AuthError::Io {
            path: self.path.clone(),
            source,
        })?;
        serde_json::from_slice(&bytes).map_err(|source| AuthError::Parse {
            path: self.path.clone(),
            source,
        })
    }

    async fn load(&self) -> Result<String, AuthError> {
        let config = self.read_config().await?;
        // A pre-populated access_token wins so tests / tooling can pin a
        // specific JWT without standing up a verify-api-key endpoint.
        if let Some(token) = config.access_token {
            return Ok(token);
        }
        let api_key = config.api_key.ok_or_else(|| AuthError::Missing {
            path: self.path.clone(),
        })?;
        self.exchange_api_key(&api_key).await
    }

    async fn exchange_api_key(&self, api_key: &str) -> Result<String, AuthError> {
        let url = verify_api_key_url(&self.api_url);
        let response = self
            .http
            .post(&url)
            .json(&serde_json::json!({ "api_key": api_key }))
            .send()
            .await
            .map_err(|source| AuthError::ExchangeTransport {
                url: url.clone(),
                source,
            })?;
        let status = response.status();
        if !status.is_success() {
            let mut body = response.text().await.unwrap_or_default();
            if body.len() > VERIFY_API_KEY_ERROR_BODY_LIMIT {
                body.truncate(VERIFY_API_KEY_ERROR_BODY_LIMIT);
            }
            return Err(AuthError::ExchangeStatus { url, status, body });
        }
        let parsed: VerifyApiKeyResponse =
            response
                .json()
                .await
                .map_err(|source| AuthError::ExchangeTransport {
                    url: url.clone(),
                    source,
                })?;
        parsed
            .access_token
            .ok_or(AuthError::ExchangeMissingToken { url })
    }
}

/// Compose `{api_url}/auth/verify-api-key` without doubling up the separator
/// when `api_url` already ends in a slash.
fn verify_api_key_url(api_url: &str) -> String {
    let base = api_url.trim_end_matches('/');
    format!("{base}/auth/verify-api-key")
}

#[async_trait]
impl AuthProvider for FileAuthProvider {
    async fn bearer_token(&self) -> Result<String, AuthError> {
        // The cache lock is intentionally held across `load().await` (the
        // verify-api-key exchange, up to ~30 s on a slow link). This makes the
        // method single-flight: a cold-cache thundering herd triggers exactly
        // ONE backend exchange while the rest wait, then all observe the cached
        // token — instead of every caller firing its own verify. The cost is
        // that those concurrent callers serialise for the duration of that one
        // exchange; acceptable because it is bounded by the client request
        // timeout and only happens on a cold/just-reloaded cache.
        let mut cached = self.cached.lock().await;
        if let Some(token) = cached.as_ref() {
            return Ok(token.clone());
        }
        let token = self.load().await?;
        *cached = Some(token.clone());
        Ok(token)
    }

    async fn reload(&self) -> Result<(), AuthError> {
        let mut cached = self.cached.lock().await;
        *cached = None;
        Ok(())
    }
}

/// In-memory provider for tests: returns a fixed token every call.
pub struct StaticAuthProvider {
    token: Arc<Mutex<String>>,
}

impl StaticAuthProvider {
    /// Create a provider that returns `token` on every call.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: Arc::new(Mutex::new(token.into())),
        }
    }

    /// Replace the cached token; useful for asserting reload behaviour.
    #[allow(dead_code)]
    pub async fn set_token(&self, token: impl Into<String>) {
        let mut guard = self.token.lock().await;
        *guard = token.into();
    }
}

#[async_trait]
impl AuthProvider for StaticAuthProvider {
    async fn bearer_token(&self) -> Result<String, AuthError> {
        Ok(self.token.lock().await.clone())
    }

    async fn reload(&self) -> Result<(), AuthError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn file_provider_exchanges_api_key_for_jwt() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/auth/verify-api-key"))
            .and(body_json(serde_json::json!({"api_key": "nrc_abc"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "jwt-1"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let dir = TempDir::new().unwrap();
        let config = dir.path().join("config.json");
        tokio::fs::write(&config, r#"{"api_key":"nrc_abc"}"#)
            .await
            .unwrap();
        let provider = FileAuthProvider::new(&config, server.uri()).unwrap();
        assert_eq!(provider.bearer_token().await.unwrap(), "jwt-1");
        // Cached — second call must not re-hit the exchange endpoint.
        assert_eq!(provider.bearer_token().await.unwrap(), "jwt-1");
    }

    #[tokio::test]
    async fn file_provider_reload_re_exchanges() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/auth/verify-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "jwt-1"
            })))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/auth/verify-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "jwt-2"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let dir = TempDir::new().unwrap();
        let config = dir.path().join("config.json");
        tokio::fs::write(&config, r#"{"api_key":"nrc_abc"}"#)
            .await
            .unwrap();
        let provider = FileAuthProvider::new(&config, server.uri()).unwrap();
        assert_eq!(provider.bearer_token().await.unwrap(), "jwt-1");
        provider.reload().await.unwrap();
        assert_eq!(provider.bearer_token().await.unwrap(), "jwt-2");
    }

    #[tokio::test]
    async fn file_provider_prefers_pre_populated_access_token() {
        // No mock server needed — a pre-populated access_token bypasses the
        // exchange entirely so we should not hit the network at all.
        let dir = TempDir::new().unwrap();
        let config = dir.path().join("config.json");
        tokio::fs::write(
            &config,
            r#"{"api_key":"nrc_abc","access_token":"jwt-pinned"}"#,
        )
        .await
        .unwrap();
        let provider = FileAuthProvider::new(&config, "http://127.0.0.1:1/unused").unwrap();
        assert_eq!(provider.bearer_token().await.unwrap(), "jwt-pinned");
    }

    #[tokio::test]
    async fn file_provider_missing_token_errors() {
        let dir = TempDir::new().unwrap();
        let config = dir.path().join("config.json");
        tokio::fs::write(&config, r#"{}"#).await.unwrap();
        let provider = FileAuthProvider::new(&config, "http://127.0.0.1:1/unused").unwrap();
        let err = provider.bearer_token().await.unwrap_err();
        assert!(matches!(err, AuthError::Missing { .. }));
    }

    #[tokio::test]
    async fn file_provider_exchange_non_2xx_surfaces_status() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/auth/verify-api-key"))
            .respond_with(ResponseTemplate::new(401).set_body_string("nope"))
            .mount(&server)
            .await;

        let dir = TempDir::new().unwrap();
        let config = dir.path().join("config.json");
        tokio::fs::write(&config, r#"{"api_key":"nrc_abc"}"#)
            .await
            .unwrap();
        let provider = FileAuthProvider::new(&config, server.uri()).unwrap();
        let err = provider.bearer_token().await.unwrap_err();
        match err {
            AuthError::ExchangeStatus { status, body, .. } => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
                assert!(body.contains("nope"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn file_provider_exchange_missing_access_token_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/auth/verify-api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&server)
            .await;

        let dir = TempDir::new().unwrap();
        let config = dir.path().join("config.json");
        tokio::fs::write(&config, r#"{"api_key":"nrc_abc"}"#)
            .await
            .unwrap();
        let provider = FileAuthProvider::new(&config, server.uri()).unwrap();
        let err = provider.bearer_token().await.unwrap_err();
        assert!(matches!(err, AuthError::ExchangeMissingToken { .. }));
    }

    #[test]
    fn verify_api_key_url_dedupes_trailing_slash() {
        assert_eq!(
            verify_api_key_url("https://api/api"),
            "https://api/api/auth/verify-api-key"
        );
        assert_eq!(
            verify_api_key_url("https://api/api/"),
            "https://api/api/auth/verify-api-key"
        );
    }
}
