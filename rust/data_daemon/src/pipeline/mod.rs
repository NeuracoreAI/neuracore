//! Per-trace dispatcher and trace-actor pipeline.
//!
//! - [`dispatcher`] owns a `DashMap` keyed by `(recording_id, trace_id)` and
//!   spawns a per-trace actor on first message.
//! - [`trace_actor`] is the per-trace task: it serialises envelopes for one
//!   trace, updates the SQLite state store, and drives the JSON / NUT writers.

pub mod dispatcher;
pub mod json_writer;
pub mod trace_actor;
