//! Per-trace on-disk encoders.
//!
//! - [`json_trace`] — incremental JSON-array writer used by scalar / sensor
//!   traces and the video sidecar.
//! - [`nut_writer`] — minimal NUT muxer that spools raw RGB frames to a
//!   growing on-disk file for the ffmpeg transcoder.
//! - [`video_encoder`] — supervised `ffmpeg` subprocess that turns a spooled
//!   NUT into both `lossy.mp4` and `lossless.mp4`.
//! - [`metadata`] — accumulator that flushes the video-trace sidecar
//!   `trace.json` alongside the mp4 outputs.
//!
//! A few writer methods (`add_entries`, `record_value`, `with_binary`, the
//! path getters) are exercised only by unit tests; the module-wide `dead_code`
//! allow keeps the compile clean without hiding the typed surface from
//! doctests.

#[allow(dead_code)]
pub mod json_trace;
#[allow(dead_code)]
pub mod metadata;
#[allow(dead_code)]
pub mod nut_writer;
#[allow(dead_code)]
pub mod video_encoder;
