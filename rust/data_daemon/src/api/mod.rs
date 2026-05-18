//! HTTP client, auth, and request/response types for the Neuracore backend.
//!
//! Phase 6a of the rewrite. Centralises the construction of a single
//! [`ApiClient`] used by every upload coordinator so the bearer header, retry
//! policy, and timeouts are configured in exactly one place. The seven
//! endpoints from the rewrite plan §8 are exposed as methods on the client.

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
