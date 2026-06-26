//! Cloud coordinators that drive each trace's lifecycle to the backend: batch
//! registration, resumable uploads, debounced status updates, and the periodic
//! progress reporter. Each exposes a single `spawn_*` entry point so the launch
//! routine can drive ordered shutdown by dropping the handle.

pub mod progress;
pub mod registration;
pub mod status;
mod upload_transfer;
pub mod uploader;
