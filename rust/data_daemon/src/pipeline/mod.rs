//! Per-trace dispatcher and trace-actor pipeline.
//!
//! Phase 4 of the rewrite (see `docs/data-daemon-rewrite.md` §4).
//!
//! - [`dispatcher`] owns a `DashMap` keyed by `(recording_id, trace_id)` and
//!   spawns a per-trace actor on first message.
//! - [`trace_actor`] is the per-trace task: it serialises envelopes for one
//!   trace, updates the SQLite state store, and (in later phases) drives the
//!   JSON / NUT writers.

pub mod dispatcher;
pub mod trace_actor;
