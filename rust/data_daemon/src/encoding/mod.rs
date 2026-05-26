//! Per-trace on-disk encoders.
//!
//! - [`json_trace`] — incremental JSON-array writer used by scalar / sensor
//!   traces and the video sidecar.
//! - [`video_encoder`] — supervised `ffmpeg` subprocess that turns a NUT
//!   chunk into a per-chunk MP4 pair, and stitches the chunk segments into
//!   the final `lossy.mp4` / `lossless.mp4` on `EndTrace`.
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
pub mod video_encoder;
