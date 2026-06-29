//! HTTP client, auth, and request/response types for the Neuracore backend.
//!
//! Centralises the construction of a single [`ApiClient`] used by every
//! upload coordinator so the bearer header, retry policy, and timeouts are
//! configured in exactly one place. The backend endpoints are exposed as
//! methods on the client.

pub mod auth;
pub mod client;
pub mod models;

#[allow(unused_imports)]
pub use auth::{AuthError, AuthProvider, FileAuthProvider};
#[allow(unused_imports)]
pub use client::{ApiClient, ApiClientError, ApiClientOptions};
#[allow(unused_imports)]
pub use models::{
    BatchRegisterResponse, CloudFile, RegisterTraceRequest, ResumableUploadUrlResponse,
    TraceStatusUpdate, TraceStatusValue,
};
