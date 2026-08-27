//! Per-trace on-disk encoders.
//!
//! - [`json_trace`], an incremental JSON-array writer used by scalar / sensor
//!   traces and the video sidecar.
//! - [`video_encoder`], a supervised `ffmpeg` subprocess that turns one or
//!   more NUT chunks into one MP4 pair per batch, and stitches the segments
//!   into the final `lossy.mp4` / `lossless.mp4` on `EndTrace`.
//! - [`metadata`], an accumulator that flushes the video-trace sidecar
//!   `trace.json` alongside the mp4 outputs.
//!
//! A few writer methods are exercised only by unit tests; those carry a
//! targeted `#[allow(dead_code)]` at their own definition rather than a
//! module-wide allow, so genuinely-dead code elsewhere still surfaces.

pub mod json_trace;
pub mod metadata;
pub mod video_encoder;
