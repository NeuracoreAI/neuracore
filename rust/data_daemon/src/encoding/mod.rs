//! Per-trace on-disk encoders.
//!
//! Phase 5 of the rewrite plan (see `docs/data-daemon-rewrite.md`). Each
//! submodule corresponds to a sub-phase deliverable:
//!
//! - [`json_trace`] (5b) — incremental JSON-array writer used by scalar /
//!   sensor traces and the video sidecar.
//! - [`nut_writer`] (5c) — minimal NUT muxer that spools raw RGB frames to a
//!   growing on-disk file for the ffmpeg transcoder spawned in 5d.
//! - [`video_encoder`] (5d) — supervised `ffmpeg` subprocess that turns a
//!   spooled NUT into both `lossy.mp4` and `lossless.mp4`.
//! - [`metadata`] (5e) — accumulator that flushes the video-trace sidecar
//!   `trace.json` alongside the mp4 outputs.
//!
//! The trace actor in [`crate::pipeline::trace_actor`] wires the writers in
//! at sub-phase 5f. A few writer methods (`add_entries`, `record_value`,
//! `with_binary`, the path getters) are exercised by unit tests and slated
//! for Phase 6/7 callers — the module-wide `dead_code` allow keeps the
//! current compile clean without hiding the typed surface from doctests.

#[allow(dead_code)]
pub mod json_trace;
#[allow(dead_code)]
pub mod metadata;
#[allow(dead_code)]
pub mod nut_writer;
#[allow(dead_code)]
pub mod video_encoder;
