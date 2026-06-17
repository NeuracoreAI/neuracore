//! Per-trace dispatcher and trace-actor pipeline.
//!
//! - [`dispatcher`] is a single tokio task that owns a lock-free HashMap of
//!   per-source recording windows, routes held data into windows by publish
//!   timestamp, and spawns a per-trace actor (keyed by recording_index,
//!   data_type, sensor_name) on first datum.
//! - [`trace_actor`] is the per-trace task: it serialises envelopes for one
//!   trace, updates the SQLite state store, and drives the JSON / NUT writers.

pub mod dispatcher;
pub mod json_writer;
pub mod trace_actor;
