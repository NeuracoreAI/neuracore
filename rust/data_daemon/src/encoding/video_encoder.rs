//! Batched `ffmpeg` transcoder and segment concatenator.
//!
//! The producer spools video frames into a sequence of NUT chunk files
//! beneath each trace's `chunks/` directory. The per-trace actor's encode
//! worker hands a batch of one or more contiguous chunks to
//! [`VideoEncoder::encode_chunk_batch`], which shells out to ffmpeg to
//! produce two MP4 segments for the whole batch:
//!
//! - `chunk_NNNN_lossy.mp4` — `libx264` `-pix_fmt yuv420p -preset ultrafast
//!   -qp 23` for fast playback, downscaled to a preview resolution (see
//!   [`LOSSY_PREVIEW_MAX_HEIGHT`]) since it is only a derivable proxy and the
//!   full-resolution encode is the transcoder's dominant cost.
//! - `chunk_NNNN_lossless.mp4` — `libx264rgb` `-pix_fmt rgb24 -preset
//!   ultrafast -qp 0` for mathematically-lossless archival. Encoding the
//!   captured rgb24 frames directly (rather than converting to a YUV format)
//!   keeps the output bit-exact to the captured pixels and encodes ~2.5×
//!   faster than a `yuv444p10le` pass.
//!   `ffv1` would also be lossless but is incompatible with the `.mp4`
//!   container the on-disk layout contract requires.
//!
//! A batch of one delegates to [`VideoEncoder::encode_chunk`], which skips
//! both the concat demuxer and `verify_nut_header`. A batch of two or more
//! goes through the ffmpeg concat demuxer with a list file that carries one
//! `duration` line per non-last entry. `declared_batch_span_us` sets that
//! duration to the capture span to the next chunk's first frame, floored at
//! the chunk's replayed PTS extent plus 1 us and capped at that extent plus
//! `MAX_BOUNDARY_DELTA_US`. `verify_nut_header` checks every entry of such a
//! batch, because the concat demuxer treats an entry it cannot open as end of
//! stream. The frame cap `-frames:v` is the sum over the
//! batch, which is correct because a chunk the dispatcher cut is always the
//! trace's last. The caller names the outputs by the batch's first chunk
//! index.
//!
//! On `EndTrace` the per-trace actor calls [`VideoEncoder::concat_segments`]
//! which stream-copies the per-chunk segments into the final `lossy.mp4` /
//! `lossless.mp4`. Stream-copy avoids a second decode/encode pass, so the
//! tail of a recording finishes in seconds regardless of total length.
//!
//! Both outputs are verified non-empty before the caller is told the
//! invocation succeeded; ffmpeg occasionally exits 0 but produces a
//! zero-byte file when the requested codec is unavailable in the local
//! build.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use data_daemon_shared::ffmpeg::passthrough_frame_sync_arg;
use data_daemon_shared::service_name::VIDEO_SPOOL_TICKS_PER_SECOND;
use serde::{Serialize, Serializer};

use tokio::process::Command;

/// Default ffmpeg binary name. Tests override via [`VideoEncoder::with_binary`]
/// when they need to point at a specific build.
pub const DEFAULT_FFMPEG_BINARY: &str = "ffmpeg";

/// `nice` value applied to each transcode child via `setpriority` before exec.
///
/// Per-chunk transcoding is throughput-oriented background work; a robot's
/// `nc.log_*` calls are latency-critical. On a small (2-vCPU) host an unniced
/// ffmpeg child preempts the producer's logging threads at recording
/// boundaries, so a joint publish that does ~3 ms of work spends ~20 ms
/// descheduled. Renicing the encoder lets the kernel scheduler favour the
/// foreground logging threads while ffmpeg still consumes otherwise-idle CPU.
const ENCODER_NICENESS: libc::c_int = 10;

/// Floor for the libx264 frame-thread count applied to *each* encode output —
/// the value used when the transcode fleet is fully loaded.
///
/// libx264 defaults to roughly one frame-thread per core. With the transcode
/// concurrency permit pool also scaling with the core count, the two multiply:
/// a 14-core host ran ~14 ffmpeg children each spawning ~14 threads, ~200
/// encode threads fighting over 14 cores. That thrashes the scheduler and
/// steals cycles from the latency-critical `nc.log_*` threads — the exact
/// path the renice above tries to protect. Capping each output's thread pool
/// keeps the total encode-thread count near the core count instead. Measured
/// sweet spot on a 14-core host: ~`cores / 2` concurrent children at 2 threads
/// per output beat the uncapped default on both aggregate throughput and
/// logging-thread jitter, so [`default_ffmpeg_concurrency`] divides by this.
///
/// A *floor*, not a hard cap: [`adaptive_encode_threads`] gives each encode more
/// threads (`cores / active`) when fewer are running, filling the idle cores
/// while keeping the full-load thread total unchanged.
///
/// [`default_ffmpeg_concurrency`]: crate::pipeline::trace_actor::default_ffmpeg_concurrency
/// [`adaptive_encode_threads`]: crate::pipeline::trace_actor::adaptive_encode_threads
pub const ENCODE_THREADS_PER_OUTPUT: usize = 2;

/// Height (in lines) the lossy *preview* proxy is downscaled to.
///
/// At 8-context 1080p60 the transcoder is CPU-bound, and the full-resolution
/// lossy pass is the long pole (~38% of the per-chunk encode work) — yet the
/// lossy output is only a fast-playback proxy, derivable from the lossless
/// archival copy. Encoding it at preview resolution instead cuts that pass'
/// cost roughly with the pixel-count reduction: measured ~+21% aggregate
/// transcode throughput at 8×1080p60, which is what buys the spool real-time
/// headroom without touching the bit-exact lossless output (which stays at
/// native resolution).
///
/// The downscale (see [`preview_scale_filter`]) caps *height* at this many
/// lines while preserving aspect ratio, never upscales a smaller source, and
/// rounds both dimensions to even (an H.264 `yuv420p` requirement) — so it is
/// correct for any input resolution or aspect ratio. 480 lines is ample for a
/// scrub/preview proxy.
const LOSSY_PREVIEW_MAX_HEIGHT: u32 = 480;

/// Lossy RGB video codec selection for a trace, resolved once at the trace's
/// first chunk (and by the registration coordinator, from the same source).
///
/// The default produces the lossless archive plus a downscaled lossy preview.
/// `H264MediumLossyOnly` (the SDK's `nc.Codec.H264_MEDIUM`) instead produces a
/// single full-resolution `libx264 -crf 23 -preset medium` video and skips the
/// lossless archive — smaller uploads, with that one video used for training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LossyVideoCodec {
    /// Default: lossless archive (`libx264rgb -qp 0`) plus a preview-resolution
    /// lossy proxy (`libx264 -qp 23`). Both outputs are produced.
    #[default]
    LosslessPlusPreview,
    /// `nc.Codec.H264_MEDIUM`: one full-resolution `libx264 -crf 23 -preset
    /// medium` video; no lossless archive, no preview downscale.
    H264MediumLossyOnly,
}

impl LossyVideoCodec {
    /// Resolve a codec from a config/env string. Only `"h264_medium"` selects
    /// lossy-only; `"h264_lossless"` and unset/empty keep the default silently.
    /// An unrecognised value also keeps the default but logs a warning (parity
    /// with the SDK's `resolve_codec`), so a typo can't silently change codecs.
    /// Callers gate this to RGB traces — depth always keeps lossless storage.
    pub fn from_config_str(value: Option<&str>) -> Self {
        match value {
            Some("h264_medium") => Self::H264MediumLossyOnly,
            None | Some("") | Some("h264_lossless") => Self::LosslessPlusPreview,
            Some(other) => {
                tracing::warn!(
                    codec = other,
                    "Ignoring unknown video codec; expected one of: \
                     h264_lossless, h264_medium"
                );
                Self::LosslessPlusPreview
            }
        }
    }

    /// Resolve the codec for a trace of `data_type` given the configured global
    /// codec string (the resolved `NCD_VIDEO_CODEC` / active-profile
    /// `video_codec`).
    ///
    /// Only RGB cameras honour the selection — a depth trace's lossy proxy is a
    /// visualisation, not precise depth, so depth (and every non-RGB stream)
    /// always keeps the default lossless archive. This RGB-only gate is
    /// deliberately narrower than the video-family predicate in
    /// [`crate::cloud::cloud_files`] (which includes depth). Kept pure (the
    /// config string is passed in, not read here) so the encoder path and the
    /// registration coordinator resolve from the same source, and the gate is
    /// unit-testable without touching the environment.
    pub fn for_trace(data_type: &str, codec_value: Option<&str>) -> Self {
        if data_type != "RGB_IMAGES" {
            return Self::LosslessPlusPreview;
        }
        Self::from_config_str(codec_value)
    }

    /// Whether this codec produces only the lossy output (no lossless archive).
    pub fn is_lossy_only(self) -> bool {
        matches!(self, Self::H264MediumLossyOnly)
    }

    /// The wire identifier for this codec — the inverse of [`Self::from_config_str`].
    ///
    /// These are the same identifiers the profile and `NCD_VIDEO_CODEC` accept,
    /// and they are persisted as-is against the recording, so the round trip is
    /// pinned by a unit test.
    pub fn as_wire_str(self) -> &'static str {
        match self {
            Self::LosslessPlusPreview => "h264_lossless",
            Self::H264MediumLossyOnly => "h264_medium",
        }
    }
}

/// Serialise as the wire identifier, so callers hold the enum right up to the
/// point a request is encoded and no codec string exists anywhere else.
impl Serialize for LossyVideoCodec {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_wire_str())
    }
}

/// Inputs to one single-entry transcode invocation.
#[derive(Debug, Clone)]
pub struct ChunkEncodeRequest {
    /// Source NUT chunk file produced by the producer.
    pub raw_nut: PathBuf,
    /// Destination for the lossy mp4 segment of this entry.
    pub lossy_out: PathBuf,
    /// Destination for the lossless mp4 segment of this entry. Unused in
    /// lossy-only mode (no lossless output is produced).
    pub lossless_out: PathBuf,
    /// Lossy codec selection for this trace. Controls whether a lossless
    /// archive is produced and how the lossy output is encoded.
    pub codec: LossyVideoCodec,
    /// Frames to encode once `skip_frames` are discarded: the rest of the file
    /// unless the dispatcher cut the chunk, and always what the sidecar is
    /// built from.
    pub frame_count: u32,
    /// Leading frames of `raw_nut` to discard before encoding: the ones the
    /// dispatcher resolved as published before this recording's window opened.
    /// Zero for a chunk whose first frame the window already owns.
    pub skip_frames: u32,
}

/// One NUT entry of a batched chunk encode.
#[derive(Debug, Clone)]
pub struct BatchNutInput {
    /// Source NUT chunk file, already relinked into the trace's chunks dir.
    pub raw_nut: PathBuf,
    /// Declared capture span to the next chunk's first frame, in
    /// microseconds. `None` on the batch's last entry, which gets no
    /// `duration` line.
    pub span_to_next_us: Option<i64>,
    /// Frames this entry contributes to the batch: the whole NUT unless the
    /// dispatcher cut the chunk (see [`ChunkEncodeRequest::frame_count`]).
    pub frame_count: u32,
    /// Leading frames to discard (see [`ChunkEncodeRequest::skip_frames`]).
    /// Always zero unless this entry is its batch's only one — the head cut
    /// is a whole-stream filter, so a batch never mixes it with a second
    /// entry (see `drain_encode_batch`).
    pub skip_frames: u32,
}

/// Inputs to one batched transcode invocation covering a contiguous run of
/// NUT chunks. The outputs carry the batch's first chunk index in their
/// names (the caller builds the paths), so downstream segment handling is
/// unchanged from the single-chunk case.
#[derive(Debug, Clone)]
pub struct BatchEncodeRequest {
    /// Batch entries in chunk-index order.
    pub inputs: Vec<BatchNutInput>,
    /// Destination for the batch's lossy mp4 segment.
    pub lossy_out: PathBuf,
    /// Destination for the batch's lossless mp4 segment. Unused in
    /// lossy-only mode (no lossless output is produced).
    pub lossless_out: PathBuf,
    /// Lossy codec selection for this trace.
    pub codec: LossyVideoCodec,
}

/// Outcome of a successful transcode: the sizes of the batch's lossy and
/// lossless output files.
#[derive(Debug, Clone, Copy)]
pub struct ChunkEncodeOutcome {
    /// Bytes written to the lossy segment.
    pub lossy_bytes: u64,
    /// Bytes written to the lossless segment.
    pub lossless_bytes: u64,
}

/// Outcome of a successful concat invocation.
#[derive(Debug, Clone, Copy)]
pub struct ConcatOutcome {
    /// Bytes written to the concatenated output.
    pub bytes: u64,
}

/// Errors raised by [`VideoEncoder`] operations.
#[derive(Debug, thiserror::Error)]
pub enum VideoEncodeError {
    /// `ffmpeg` could not be located or spawned (typically `ENOENT`).
    #[error("failed to spawn `{}`: {source}", binary.to_string_lossy())]
    Spawn {
        /// Binary that failed to spawn.
        binary: OsString,
        /// Underlying OS error.
        #[source]
        source: std::io::Error,
    },
    /// `ffmpeg` exited with a non-zero status. `stderr_tail` captures the last
    /// few KiB of ffmpeg's stderr so the caller can surface a diagnostic
    /// without trawling the daemon log.
    #[error("`ffmpeg` exited with status {status}: {stderr_tail}")]
    NonZeroExit {
        /// Exit status reported by the child.
        status: String,
        /// Tail of the child's stderr (UTF-8 with replacements).
        stderr_tail: String,
    },
    /// One of the expected mp4 outputs was missing or empty after the encoder
    /// claimed success — usually means the codec is not built into the local
    /// ffmpeg binary.
    #[error("expected output {path} is missing or empty after ffmpeg exit")]
    OutputMissing {
        /// Path that should have been written.
        path: PathBuf,
    },
    /// An I/O operation around the encode (file metadata, unlink, concat list
    /// write) failed.
    #[error("I/O failure during transcode for {path}: {source}")]
    Io {
        /// Path being inspected when the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// `concat_segments` was called with no input segments — caller bug.
    #[error("concat_segments called with empty segment list")]
    EmptySegments,
    /// A batched NUT input failed the header check before the invocation
    /// (see [`verify_nut_header`]).
    #[error("batch input {path} is not a NUT container")]
    InvalidNutInput {
        /// The input that failed the header check.
        path: PathBuf,
    },
}

/// Failure modes of [`VideoEncoder::preflight`], surfaced at daemon startup so
/// an unusable ffmpeg is reported once, clearly, instead of failing every
/// video encode at recording time.
#[derive(Debug, thiserror::Error)]
pub enum FfmpegPreflightError {
    /// The ffmpeg binary could not be executed at all — typically not
    /// installed or not on `PATH`.
    #[error(
        "ffmpeg not found: could not run `{}` ({source}). \
         Install ffmpeg (>= 4.0, built with libx264) and ensure it is on PATH.",
        binary.to_string_lossy()
    )]
    NotFound {
        /// Binary that could not be executed.
        binary: OsString,
        /// Underlying spawn error (e.g. `ENOENT`).
        #[source]
        source: std::io::Error,
    },
    /// ffmpeg ran but rejected a capability the encoder depends on: the
    /// passthrough frame-timing mode or the libx264 encoder.
    #[error(
        "ffmpeg at `{}` (version {version}) is incompatible: a required capability was \
         rejected. The daemon needs passthrough frame timing (drop-free, frame-accurate \
         encoding; spelled `{frame_sync_arg}` on this build) and the libx264 / libx264rgb \
         encoders. Install a compatible ffmpeg (>= 4.0 with libx264). ffmpeg reported:\
         \n{stderr_tail}",
        binary.to_string_lossy()
    )]
    Incompatible {
        /// Binary that was probed.
        binary: OsString,
        /// Detected ffmpeg version, or `"unknown"`.
        version: String,
        /// Passthrough frame-timing option the probe used on this build.
        frame_sync_arg: &'static str,
        /// Tail of ffmpeg's stderr from the failed probe.
        stderr_tail: String,
    },
}

/// Builder for ffmpeg invocations. Keeps the ffmpeg binary path configurable
/// so unit tests can shim in a wrapper script if needed.
#[derive(Debug, Clone)]
pub struct VideoEncoder {
    binary: OsString,
}

impl Default for VideoEncoder {
    fn default() -> Self {
        Self {
            binary: OsString::from(DEFAULT_FFMPEG_BINARY),
        }
    }
}

impl VideoEncoder {
    /// Construct an encoder that resolves `ffmpeg` from `PATH`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the ffmpeg binary location (test/diagnostic seam).
    #[allow(dead_code)]
    pub fn with_binary(mut self, binary: impl Into<OsString>) -> Self {
        self.binary = binary.into();
        self
    }

    /// Option name this ffmpeg accepts for passthrough frame timing; the two
    /// spellings do not overlap across the supported version range.
    fn frame_sync_arg(&self) -> &'static str {
        passthrough_frame_sync_arg(&self.binary)
    }

    /// Verify the configured ffmpeg is present and supports the capabilities
    /// [`encode_chunk`](Self::encode_chunk) depends on, returning the detected
    /// version string on success.
    ///
    /// Run once at daemon startup so an incompatible install fails fast with a
    /// clear message instead of silently marking every video trace `failed` at
    /// recording time. Two steps: `ffmpeg -version` confirms the binary runs
    /// (and yields a version for diagnostics), then a one-frame synthetic
    /// encode to the null muxer exercises the passthrough frame-timing mode
    /// together with the libx264 encoder.
    pub fn preflight(&self) -> Result<String, FfmpegPreflightError> {
        let version = self.detect_ffmpeg_version()?;
        self.probe_passthrough_encode(&version)?;
        Ok(version)
    }

    /// Run `ffmpeg -version`, mapping a spawn failure to
    /// [`FfmpegPreflightError::NotFound`] and parsing the reported version.
    fn detect_ffmpeg_version(&self) -> Result<String, FfmpegPreflightError> {
        let output = std::process::Command::new(&self.binary)
            .arg("-hide_banner")
            .arg("-version")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|source| FfmpegPreflightError::NotFound {
                binary: self.binary.clone(),
                source,
            })?;
        Ok(parse_ffmpeg_version(&output.stdout))
    }

    /// Encode one synthetic frame to the null muxer through **both** output
    /// configurations the real [`encode_chunk`](Self::encode_chunk) uses — the
    /// `yuv420p libx264` lossy pass *and* the `rgb24 libx264rgb -qp 0` lossless
    /// pass, with the passthrough spelling this build accepts. A non-zero exit
    /// means the local ffmpeg lacks a capability the encoder needs. The
    /// lossless `libx264rgb` path is the one that actually varies between
    /// builds, so probing only the lossy pass (as before) let the "fail fast at
    /// startup" check pass while every real lossless encode failed at recording
    /// time.
    fn probe_passthrough_encode(&self, version: &str) -> Result<(), FfmpegPreflightError> {
        // One 16x16 yuv420p frame (a 16x16 plane plus two 8x8 planes = 384
        // bytes) fed via the rawvideo demuxer on stdin — no lavfi/input-file
        // dependency, so the probe works even on a minimal build. ffmpeg parses
        // (and would reject) the options before reading stdin, so an unsupported
        // passthrough mode, `-enc_time_base` or `libx264rgb` encode fails
        // immediately rather than on a healthy input. The two `-map 0:v -c:v …`
        // blocks exercise the same codec, pixel formats and timestamp pinning
        // as `encode_chunk` (the build-dependent parts); the real lossy encode
        // adds options the probe omits (e.g. `-qp 23` / `+genpts`), so the full
        // option set is not identical.
        //
        // `-video_track_timescale` is a mov-muxer private option and the null
        // muxer silently ignores unknown muxer options, so probing it demands a
        // real mp4 output: the first block writes a one-frame mp4 to a temp
        // path (removed afterwards) while the second keeps the null muxer.
        const PROBE_FRAME_LEN: usize = 16 * 16 * 3 / 2;
        let frame = vec![128u8; PROBE_FRAME_LEN];
        let enc_time_base = format!("1:{VIDEO_SPOOL_TICKS_PER_SECOND}");
        let track_timescale = VIDEO_SPOOL_TICKS_PER_SECOND.to_string();
        let mp4_probe_out =
            std::env::temp_dir().join(format!("ncd_ffmpeg_preflight_{}.mp4", std::process::id()));
        let frame_sync_arg = self.frame_sync_arg();

        let child = std::process::Command::new(&self.binary)
            .arg("-y")
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-f")
            .arg("rawvideo")
            .arg("-pix_fmt")
            .arg("yuv420p")
            .arg("-video_size")
            .arg("16x16")
            .arg("-i")
            .arg("-")
            // Lossy pass (matches encode_chunk's first output), written
            // through the real mp4 muxer so the timescale pin is genuinely
            // validated.
            .arg("-map")
            .arg("0:v")
            .arg(frame_sync_arg)
            .arg("passthrough")
            .arg("-enc_time_base")
            .arg(&enc_time_base)
            .arg("-c:v")
            .arg("libx264")
            .arg("-pix_fmt")
            .arg("yuv420p")
            .arg("-preset")
            .arg("ultrafast")
            .arg("-video_track_timescale")
            .arg(&track_timescale)
            .arg(&mp4_probe_out)
            // Lossless pass (matches encode_chunk's second output) — the
            // build-dependent `libx264rgb` rgb24 capability the encoder relies on.
            .arg("-map")
            .arg("0:v")
            .arg(frame_sync_arg)
            .arg("passthrough")
            .arg("-enc_time_base")
            .arg(&enc_time_base)
            .arg("-c:v")
            .arg("libx264rgb")
            .arg("-pix_fmt")
            .arg("rgb24")
            .arg("-qp")
            .arg("0")
            .arg("-preset")
            .arg("ultrafast")
            .arg("-f")
            .arg("null")
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| FfmpegPreflightError::NotFound {
                binary: self.binary.clone(),
                source,
            });
        let mut child = match child {
            Ok(child) => child,
            Err(error) => {
                let _ = std::fs::remove_file(&mp4_probe_out);
                return Err(error);
            }
        };

        // The frame is far smaller than a pipe buffer, so writing then dropping
        // stdin cannot deadlock against ffmpeg's reads.
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(&frame);
        }

        let output = child.wait_with_output();
        let _ = std::fs::remove_file(&mp4_probe_out);
        let output = output.map_err(|source| FfmpegPreflightError::NotFound {
            binary: self.binary.clone(),
            source,
        })?;

        if output.status.success() {
            Ok(())
        } else {
            Err(FfmpegPreflightError::Incompatible {
                binary: self.binary.clone(),
                version: version.to_string(),
                frame_sync_arg,
                stderr_tail: tail_stderr(&output.stderr),
            })
        }
    }

    /// Transcode one NUT chunk into the configured per-chunk mp4 outputs.
    ///
    /// `encode_threads` bounds each output's libx264 frame-thread pool; the
    /// caller sizes it to the live transcode concurrency (see
    /// [`adaptive_encode_threads`]) so a few-camera workload uses the otherwise
    /// idle cores.
    ///
    /// The source `raw.nut` is left in place — the caller is responsible for
    /// unlinking it after verifying both outputs landed (the encode worker
    /// unlinks every NUT of the batch once the outputs verify; the NUTs of a
    /// failed batch stay on disk to be collected by the recovery sweep).
    ///
    /// [`adaptive_encode_threads`]: crate::pipeline::trace_actor::adaptive_encode_threads
    pub async fn encode_chunk(
        &self,
        request: &ChunkEncodeRequest,
        encode_threads: usize,
    ) -> Result<ChunkEncodeOutcome, VideoEncodeError> {
        ensure_parent_dirs(&request.lossy_out)?;
        // No lossless output is produced in lossy-only mode, so don't prepare a
        // directory for a file that will never be written.
        if !request.codec.is_lossy_only() {
            ensure_parent_dirs(&request.lossless_out)?;
        }

        // `-y` overwrites existing outputs (resume safety: a previous failed
        // run may have left a partial mp4). `-fflags +genpts` rebuilds the
        // presentation timestamps from the NUT timing when the spool was
        // truncated mid-frame. Passthrough frame timing (applied per output) is
        // the critical knob here: the NUT chunk uses `time_base = 1/1_000_000`
        // so ffmpeg's demuxer reports `r_frame_rate = 1_000_000/1` (one
        // million fps). With the default `cfr` policy the encoder would then
        // synthesise an output frame at every microsecond slot between
        // consecutive input PTS values — for a 10 s clip that is ~10 million
        // duplicate output frames, and the encode effectively never completes.
        //
        // We must NOT use `vfr` here: vfr drops any frame whose PTS rounds to
        // the same tick as its predecessor at the output stream timescale.
        // Real-time capture has jitter, so closely-spaced frames (a few hundred
        // µs apart under threaded logging) collide and are silently dropped —
        // the encoded video then has fewer frames than the per-frame timestamp
        // sidecar (`trace.json`), and the downstream synced-recording reader
        // dereferences a frame index the video never contained. `passthrough`
        // emits every input frame exactly once at its original PTS and never
        // drops, which is what real-time camera capture actually is.
        //
        // `frame_sync_arg` resolves the spelling this build accepts; the two
        // do not overlap across the supported ffmpeg versions. The default
        // branch emits two `-map 0:v -c:v ...` output blocks from a single demux
        // pass; lossy-only emits a single block (no preview/archive split).
        //
        // Every output also pins its timing to the NUT's microsecond clock
        // ([`VIDEO_SPOOL_TICKS_PER_SECOND`], shared with the producer's NUT
        // writer): `-enc_time_base` fixes the encoder time base and
        // `-video_track_timescale` fixes the mp4 track timescale.
        // Without both, ffmpeg derives them from a per-chunk *guessed* frame
        // rate, and the guess is unstable across chunks of one recording
        // (near-constant capture deltas keep the microsecond base while
        // jittery ones normalise to e.g. 59.94 fps → a 1/60000 track). The
        // final stream-copy concat mishandles mixed-timescale segments and
        // emits whole chunks crammed onto consecutive single ticks with
        // backwards decoded PTS — the "Video missing logged frames" rejection
        // — from perfectly clean input. Pinning both bases keeps every
        // segment's PTS equal to its capture timestamps (microsecond-exact,
        // matching the trace sidecar) and makes the concat timescale-uniform.
        // Bound each output's libx264 thread pool to the caller-sized value
        // (see `adaptive_encode_threads`) so the transcode fleet fills idle
        // cores at low concurrency without oversubscribing at high concurrency.
        let mut command = Command::new(&self.binary);
        command
            .arg("-y")
            .arg("-hide_banner")
            .arg("-nostdin")
            .arg("-loglevel")
            .arg("error")
            .arg("-fflags")
            .arg("+genpts")
            .arg("-i")
            .arg(&request.raw_nut);
        append_encode_output_args(
            &mut command,
            request.codec,
            encode_threads,
            self.frame_sync_arg(),
            request.frame_count,
            request.skip_frames,
            &request.lossy_out,
            &request.lossless_out,
        );
        let lossless_out =
            (!request.codec.is_lossy_only()).then_some(request.lossless_out.as_path());
        self.run_encode_command(command, &request.lossy_out, lossless_out)
            .await
    }

    /// Transcode a contiguous batch of NUT chunks with one ffmpeg
    /// invocation, fed through the concat demuxer with the per-entry
    /// `duration` directives that place every frame on its batch-relative
    /// capture timestamp (see [`write_batch_concat_list`]). A batch of one
    /// delegates to [`Self::encode_chunk`] for an identical invocation.
    pub async fn encode_chunk_batch(
        &self,
        request: &BatchEncodeRequest,
        encode_threads: usize,
    ) -> Result<ChunkEncodeOutcome, VideoEncodeError> {
        if let [single] = request.inputs.as_slice() {
            return self
                .encode_chunk(
                    &ChunkEncodeRequest {
                        raw_nut: single.raw_nut.clone(),
                        lossy_out: request.lossy_out.clone(),
                        lossless_out: request.lossless_out.clone(),
                        codec: request.codec,
                        frame_count: single.frame_count,
                        skip_frames: single.skip_frames,
                    },
                    encode_threads,
                )
                .await;
        }

        // The concat demuxer treats a list entry it cannot open as end of
        // stream: ffmpeg exits 0 and the frames of that entry and every later
        // entry are silently lost. Verify every input is a NUT container up
        // front so a corrupt chunk fails the batch instead of truncating it.
        for input in &request.inputs {
            verify_nut_header(&input.raw_nut)?;
        }

        ensure_parent_dirs(&request.lossy_out)?;
        if !request.codec.is_lossy_only() {
            ensure_parent_dirs(&request.lossless_out)?;
        }

        let list_path = list_file_for(&request.lossy_out);
        write_batch_concat_list(&list_path, &request.inputs)?;

        let mut command = Command::new(&self.binary);
        command
            .arg("-y")
            .arg("-hide_banner")
            .arg("-nostdin")
            .arg("-loglevel")
            .arg("error")
            .arg("-fflags")
            .arg("+genpts")
            .arg("-f")
            .arg("concat")
            // `-safe 0` permits the absolute NUT paths in the list file.
            .arg("-safe")
            .arg("0")
            .arg("-i")
            .arg(&list_path);
        append_encode_output_args(
            &mut command,
            request.codec,
            encode_threads,
            self.frame_sync_arg(),
            batch_frame_count(&request.inputs),
            // No head cut on a multi-entry batch: the filter would trim the
            // concatenated stream, and the trimmed frames still occupy the
            // first entry's declared span, so the caller keeps a cut entry in
            // a batch of its own (see `drain_encode_batch`).
            0,
            &request.lossy_out,
            &request.lossless_out,
        );
        let lossless_out =
            (!request.codec.is_lossy_only()).then_some(request.lossless_out.as_path());
        let result = self
            .run_encode_command(command, &request.lossy_out, lossless_out)
            .await;
        let _ = std::fs::remove_file(&list_path);
        result
    }

    /// Configure the encode child's stdio and niceness, run it, and verify
    /// the expected outputs are non-empty. `lossless_out` is `None` in
    /// lossy-only mode, where no lossless archive is produced.
    async fn run_encode_command(
        &self,
        mut command: Command,
        lossy_out: &Path,
        lossless_out: Option<&Path>,
    ) -> Result<ChunkEncodeOutcome, VideoEncodeError> {
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            // ffmpeg keeps file descriptors open across `fork`/`exec`; the
            // daemon's iceoryx2 sockets must NOT leak into the encoder, so we
            // rely on Tokio's default `cloexec` behaviour and additionally
            // request `kill_on_drop` to clean up if the supervising future is
            // cancelled mid-flight.
            .kill_on_drop(true);
        // SAFETY: the closure runs in the forked child between `fork` and
        // `exec`; `setpriority` is a single raw syscall that touches no
        // userspace lock or allocator state, so it is safe to call here between
        // fork and exec. A failed renice is non-fatal (ignored), so the encode
        // still runs at default priority.
        unsafe {
            command.pre_exec(|| {
                libc::setpriority(libc::PRIO_PROCESS, 0, ENCODER_NICENESS);
                Ok(())
            });
        }

        let output = command
            .output()
            .await
            .map_err(|source| VideoEncodeError::Spawn {
                binary: self.binary.clone(),
                source,
            })?;

        if !output.status.success() {
            let stderr_tail = tail_stderr(&output.stderr);
            return Err(VideoEncodeError::NonZeroExit {
                status: format!("{:?}", output.status),
                stderr_tail,
            });
        }

        let lossy_bytes = non_empty_file_size(lossy_out)?;
        // With no lossless archive there is no file to size; report zero
        // rather than erroring on a missing output.
        let lossless_bytes = match lossless_out {
            Some(path) => non_empty_file_size(path)?,
            None => 0,
        };

        Ok(ChunkEncodeOutcome {
            lossy_bytes,
            lossless_bytes,
        })
    }
    /// Stream-copy concatenate `segments` into `out`.
    ///
    /// Uses ffmpeg's `concat` demuxer with `-c copy`, so no transcode
    /// happens — total cost is bounded by the read+write of the segment
    /// bytes. `spans_to_next_us` carries one declared capture span per
    /// segment except the last (see [`write_concat_list`]), so each segment
    /// stacks at its trace-relative capture offset instead of its probed
    /// duration. Caller unlinks the source segments after the concat.
    pub async fn concat_segments(
        &self,
        segments: &[PathBuf],
        spans_to_next_us: &[i64],
        out: &Path,
    ) -> Result<ConcatOutcome, VideoEncodeError> {
        if segments.is_empty() {
            return Err(VideoEncodeError::EmptySegments);
        }
        ensure_parent_dirs(out)?;

        // The concat demuxer reads a list-file describing absolute segment
        // paths. We write it next to the output so a future debugging pass
        // can see exactly which segments were concatenated; the file is
        // unlinked on the success path so it doesn't accumulate.
        let list_path = list_file_for(out);
        write_concat_list(&list_path, segments, spans_to_next_us)?;

        let result = Command::new(&self.binary)
            .arg("-y")
            .arg("-hide_banner")
            .arg("-nostdin")
            .arg("-loglevel")
            .arg("error")
            .arg("-f")
            .arg("concat")
            // `-safe 0` permits absolute paths (and any non-portable chars)
            // in the list file. Without it ffmpeg rejects paths that aren't
            // simple relative names.
            .arg("-safe")
            .arg("0")
            .arg("-i")
            .arg(&list_path)
            .arg("-c")
            .arg("copy")
            .arg(out)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .output()
            .await;

        // Always try to clean up the list file, even on failure — leaving it
        // around just clutters the trace directory.
        let _ = std::fs::remove_file(&list_path);

        let output = result.map_err(|source| VideoEncodeError::Spawn {
            binary: self.binary.clone(),
            source,
        })?;

        if !output.status.success() {
            let stderr_tail = tail_stderr(&output.stderr);
            return Err(VideoEncodeError::NonZeroExit {
                status: format!("{:?}", output.status),
                stderr_tail,
            });
        }

        let bytes = non_empty_file_size(out)?;
        Ok(ConcatOutcome { bytes })
    }
}

/// Append the per-output encoder arguments, shared by
/// [`VideoEncoder::encode_chunk`] and [`VideoEncoder::encode_chunk_batch`] so
/// both invocations keep the same output shape (see `encode_chunk` for the
/// rationale behind each knob).
#[allow(clippy::too_many_arguments)]
fn append_encode_output_args(
    command: &mut Command,
    codec: LossyVideoCodec,
    encode_threads: usize,
    frame_sync_arg: &'static str,
    frame_count: u32,
    skip_frames: u32,
    lossy_out: &Path,
    lossless_out: &Path,
) {
    let encode_threads = encode_threads.to_string();
    let enc_time_base = format!("1:{VIDEO_SPOOL_TICKS_PER_SECOND}");
    let track_timescale = VIDEO_SPOOL_TICKS_PER_SECOND.to_string();
    // Per-output frame cap (see [`ChunkEncodeRequest::frame_count`]), a no-op
    // unless the dispatcher cut a chunk of this encode at a recording boundary.
    let frame_limit = (frame_count > 0).then(|| frame_count.to_string());
    // Head cut (see [`ChunkEncodeRequest::skip_frames`]). `-frames:v` counts
    // frames *after* the graph, so the cap above still bounds the tail.
    let head_skip = (skip_frames > 0).then(|| head_skip_filter(skip_frames));
    command
        .arg("-map")
        .arg("0:v")
        .arg(frame_sync_arg)
        .arg("passthrough");
    if codec.is_lossy_only() {
        // Single full-resolution training-quality video: libx264 CRF 23 at
        // `-preset medium`. No preview downscale and no lossless pass — this
        // is the canonical (and only) upload for the trace.
        command
            .arg("-enc_time_base")
            .arg(&enc_time_base)
            .arg("-c:v")
            .arg("libx264")
            .arg("-threads")
            .arg(&encode_threads)
            .arg("-pix_fmt")
            .arg("yuv420p")
            .arg("-preset")
            .arg("medium")
            .arg("-crf")
            .arg("23")
            .arg("-video_track_timescale")
            .arg(&track_timescale);
        if let Some(skip) = head_skip.as_deref() {
            command.arg("-vf").arg(skip);
        }
        if let Some(limit) = frame_limit.as_deref() {
            command.arg("-frames:v").arg(limit);
        }
        command.arg(lossy_out);
    } else {
        // Downscale the lossy preview proxy (only) to keep this dominant
        // pass cheap at high resolution; the lossless output stays native.
        let preview_filter = match head_skip.as_deref() {
            // Trim first, then scale: scaling frames that are about to be
            // discarded is the expensive half of this pass.
            Some(skip) => format!("{skip},{}", preview_scale_filter(LOSSY_PREVIEW_MAX_HEIGHT)),
            None => preview_scale_filter(LOSSY_PREVIEW_MAX_HEIGHT),
        };
        command
            // Lossy preview proxy only: cap to preview resolution (see
            // `preview_scale_filter`). Passthrough frame timing still emits
            // every input frame, so the lossy frame count matches the
            // lossless output and the per-frame timestamp sidecar.
            .arg("-vf")
            .arg(&preview_filter)
            .arg("-enc_time_base")
            .arg(&enc_time_base)
            .arg("-c:v")
            .arg("libx264")
            .arg("-threads")
            .arg(&encode_threads)
            .arg("-pix_fmt")
            .arg("yuv420p")
            .arg("-preset")
            .arg("ultrafast")
            .arg("-qp")
            .arg("23")
            .arg("-video_track_timescale")
            .arg(&track_timescale);
        if let Some(limit) = frame_limit.as_deref() {
            command.arg("-frames:v").arg(limit);
        }
        command
            .arg(lossy_out)
            .arg("-map")
            .arg("0:v")
            .arg(frame_sync_arg)
            .arg("passthrough")
            .arg("-enc_time_base")
            .arg(&enc_time_base)
            // libx264rgb encodes the rgb24 frames directly: bit-exact to
            // the captured pixels and ~2.5× faster than a yuv444p10le pass.
            .arg("-c:v")
            .arg("libx264rgb")
            .arg("-threads")
            .arg(&encode_threads)
            .arg("-pix_fmt")
            .arg("rgb24")
            .arg("-preset")
            .arg("ultrafast")
            .arg("-qp")
            .arg("0")
            .arg("-video_track_timescale")
            .arg(&track_timescale);
        if let Some(skip) = head_skip.as_deref() {
            command.arg("-vf").arg(skip);
        }
        if let Some(limit) = frame_limit.as_deref() {
            command.arg("-frames:v").arg(limit);
        }
        command.arg(lossless_out);
    }
}

/// Frames a batch's outputs may hold: the sum of what every entry owns.
///
/// The cap drops frames from the tail of the concatenated stream, which is
/// only correct because a tail-cut entry is always the batch's last: the
/// dispatcher cuts a chunk at the window's stop, and every chunk that opens
/// after that stop is dropped whole, so no chunk of the trace follows a cut
/// one. A head-cut entry (see [`BatchNutInput::skip_frames`]) never shares a
/// batch at all, so its discarded frames are outside every sum here.
fn batch_frame_count(inputs: &[BatchNutInput]) -> u32 {
    inputs
        .iter()
        .fold(0u32, |total, input| total.saturating_add(input.frame_count))
}

/// Ceiling for the boundary step a declared span may add past a chunk's own
/// content extent. At exactly `i32::MAX` the ultrafast branch silently
/// collapses the following segment's ladder, so the safe maximum is one
/// below. A larger real gap is compressed to this step and stays monotonic.
pub(crate) const MAX_BOUNDARY_DELTA_US: i64 = i32::MAX as i64 - 1;

/// Synthesized-PTS step bounds, mirrored from
/// `data_daemon_bridge/src/writer.rs`: a stamp that fails to advance steps
/// by the stream's observed frame gap, clamped to this range.
const SYNTH_PTS_STEP_MIN_US: u64 = 1_000;
const SYNTH_PTS_STEP_MAX_US: u64 = 100_000;

/// Convert a capture timestamp in seconds to microseconds with the NUT
/// writer's truncation: to nanoseconds first, then divide toward zero. This
/// mirrors the writer's integer microsecond arithmetic, `timestamp_ns /
/// 1_000` in `data_daemon_bridge/src/writer.rs`, so the replayed extent does
/// not undershoot the spooled PTS extent. A naive `round(s * 1e6)` is off by
/// up to 1 us.
pub(crate) fn capture_timestamp_us(timestamp_s: f64) -> i64 {
    ((timestamp_s * 1e9) as i64) / 1000
}

/// Span between two chunks' first capture timestamps, floored at zero so a
/// backwards announcement cannot emit a negative `duration`. The
/// extent-relative ceiling belongs to [`declared_batch_span_us`].
pub(crate) fn declared_span_us(from_timestamp_s: f64, to_timestamp_s: f64) -> i64 {
    (capture_timestamp_us(to_timestamp_s) - capture_timestamp_us(from_timestamp_s)).max(0)
}

/// The chunk's content extent: the last frame's PTS relative to the chunk
/// start, as the spool writer stored it in the NUT.
///
/// Replays the writer's PTS synthesis (`data_daemon_bridge/src/writer.rs`)
/// over the announced stamps. The writer carries its observed frame gap
/// across chunks and the announcement does not, so the replay seeds that
/// unknown with the step ceiling and never undershoots the real extent.
/// Undershooting would let the next chunk start inside this one's content,
/// which makes a B-frame encode store backwards PTS.
pub(crate) fn replayed_chunk_extent_us(frame_timestamps_s: &[f64]) -> i64 {
    let mut origin_us: Option<i64> = None;
    let mut last_pts_us: Option<u64> = None;
    let mut observed_frame_gap_us: Option<u64> = None;
    for &timestamp_s in frame_timestamps_s {
        let timestamp_us = capture_timestamp_us(timestamp_s);
        let origin = *origin_us.get_or_insert(timestamp_us);
        let mut pts = timestamp_us.saturating_sub(origin).max(0) as u64;
        if let Some(previous) = last_pts_us {
            if pts <= previous {
                let step = observed_frame_gap_us
                    .unwrap_or(SYNTH_PTS_STEP_MAX_US)
                    .clamp(SYNTH_PTS_STEP_MIN_US, SYNTH_PTS_STEP_MAX_US);
                pts = previous.saturating_add(step);
                origin_us = Some(timestamp_us.saturating_sub(pts as i64));
            } else if pts - previous <= SYNTH_PTS_STEP_MAX_US {
                observed_frame_gap_us = Some(pts - previous);
            }
        }
        last_pts_us = Some(pts);
    }
    last_pts_us.unwrap_or(0) as i64
}

/// Declared span from a chunk or encoded segment to the next: the capture
/// span between their first announced stamps, floored at the given content
/// extent plus 1 us and capped at the extent plus [`MAX_BOUNDARY_DELTA_US`].
///
/// Without the floor, a backwards clock step at the boundary declares the
/// next input inside this one's content and a B-frame encode stores
/// backwards PTS; the floor degrades that to a 1 us ramp. A well-formed
/// span already sits past the floor and passes through untouched.
pub(crate) fn declared_span_with_extent_us(
    chunk_frame_timestamps_s: &[f64],
    next_first_timestamp_s: f64,
    content_extent_us: i64,
) -> i64 {
    let span_us = match chunk_frame_timestamps_s.first() {
        Some(&first_timestamp_s) => declared_span_us(first_timestamp_s, next_first_timestamp_s),
        None => 0,
    };
    span_us
        .max(content_extent_us + 1)
        .min(content_extent_us + MAX_BOUNDARY_DELTA_US)
}

/// Declared span for a batch entry: [`declared_span_with_extent_us`] on the
/// chunk's replayed content extent.
pub(crate) fn declared_batch_span_us(
    chunk_frame_timestamps_s: &[f64],
    next_first_timestamp_s: f64,
) -> i64 {
    declared_span_with_extent_us(
        chunk_frame_timestamps_s,
        next_first_timestamp_s,
        replayed_chunk_extent_us(chunk_frame_timestamps_s),
    )
}

/// The real mp4 content extent of a batch-encoded segment, as the batch
/// concat list dictated it to ffmpeg: every non-last chunk occupies its
/// declared duration line, so the extent is the stacked spans plus the last
/// chunk's replayed extent.
///
/// The finalise floor needs this rather than a replay over the batch's
/// concatenated stamps, which can undershoot the real placement: the replay
/// cannot see the writer's carried frame gap, and a small healthy boundary
/// delta poisons its observed gap.
pub(crate) fn batch_content_extent_us(
    spans_to_next_us: &[i64],
    last_chunk_frame_timestamps_s: &[f64],
) -> i64 {
    let stacked_spans_us: i64 = spans_to_next_us.iter().sum();
    stacked_spans_us.saturating_add(replayed_chunk_extent_us(last_chunk_frame_timestamps_s))
}

/// Format a microsecond span as the exact decimal seconds a concat-list
/// `duration` directive carries.
fn duration_directive(span_us: i64) -> String {
    format!("{}.{:06}", span_us / 1_000_000, span_us % 1_000_000)
}

/// The file id string every NUT container starts with.
const NUT_FILE_MAGIC: &[u8] = b"nut/multimedia container\0";

/// Verify `raw_nut` starts with the NUT file id string. The concat demuxer
/// treats a file it cannot open as end of stream and ffmpeg still exits 0,
/// so an unchecked bad entry would silently drop the rest of the batch. A
/// truncated file with a healthy header passes, as it does today.
fn verify_nut_header(raw_nut: &Path) -> Result<(), VideoEncodeError> {
    let mut file = std::fs::File::open(raw_nut).map_err(|source| VideoEncodeError::Io {
        path: raw_nut.to_path_buf(),
        source,
    })?;
    let mut header = [0u8; NUT_FILE_MAGIC.len()];
    let header_read = std::io::Read::read_exact(&mut file, &mut header);
    if header_read.is_err() || header != *NUT_FILE_MAGIC {
        return Err(VideoEncodeError::InvalidNutInput {
            path: raw_nut.to_path_buf(),
        });
    }
    Ok(())
}

/// Filter that discards the first `skip_frames` frames and rebases the PTS of
/// what remains to zero, so the segment's extent is the kept frames' extent.
fn head_skip_filter(skip_frames: u32) -> String {
    format!("trim=start_frame={skip_frames},setpts=PTS-STARTPTS")
}

/// Build the ffmpeg `-vf` value that downscales the lossy preview proxy to at
/// most `max_height` lines.
///
/// The scale factor `s = min(1, max_height/ih)` is applied to both axes, so it
/// preserves aspect ratio and **never upscales** (a source already at or below
/// the cap passes through untouched). `trunc(.../2)*2` rounds each axis to an
/// even number of pixels — H.264 `yuv420p` rejects odd dimensions. The comma in
/// `min(1, …)` is escaped (`\,`) because ffmpeg's filtergraph parser otherwise
/// reads it as a filter separator. Works for any resolution or aspect ratio
/// (landscape, portrait, ultrawide); guarded by the `preview_scale_filter_*` tests.
/// `flags=fast_bilinear` replaces the default swscale bicubic scaler: it uses
/// fewer taps per output pixel, and the quality cost is acceptable because
/// this output is only a 480p `-qp 23` proxy.
fn preview_scale_filter(max_height: u32) -> String {
    format!(
        "scale=trunc(iw*min(1\\,{max_height}/ih)/2)*2:trunc(ih*min(1\\,{max_height}/ih)/2)*2:flags=fast_bilinear"
    )
}

/// Build the path to the temporary concat list file used by
/// [`VideoEncoder::concat_segments`]. Placed alongside `out` so concurrent
/// trace concats don't collide.
fn list_file_for(out: &Path) -> PathBuf {
    let mut name = out
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_else(|| OsString::from("concat_list"));
    name.push(".concat.txt");
    match out.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(name),
        _ => PathBuf::from(name),
    }
}

/// Render the ffmpeg `concat` list-file format: one `file '...'` entry per
/// segment, single-quoted with escaped embedded single quotes per the
/// demuxer's own escape rule (`'` → `'\''`). Every entry except the last is
/// followed by its `duration` line from `spans_to_next_us`, which lands each
/// frame on its trace-relative capture timestamp instead of accumulating
/// per-segment probe drift. A single-segment list carries no `duration`
/// line, keeping it byte-identical to the pre-directive shape.
///
/// Relative segment paths are resolved against the current working directory
/// before being written. ffmpeg's concat demuxer interprets `file '...'`
/// entries *relative to the list-file's directory*, not the daemon's CWD —
/// so a relative segment path like `recordings/rec/cam/trace/chunk_0000.mp4`
/// listed in `recordings/rec/cam/trace/lossy.mp4.concat.txt` would expand to
/// `recordings/rec/cam/trace/recordings/rec/cam/trace/chunk_0000.mp4` and
/// fail to open. Absolutising on write side-steps that without forcing
/// callers to pre-canonicalise.
fn write_concat_list(
    path: &Path,
    segments: &[PathBuf],
    spans_to_next_us: &[i64],
) -> Result<(), VideoEncodeError> {
    debug_assert_eq!(
        spans_to_next_us.len(),
        segments.len().saturating_sub(1),
        "one declared span per segment except the last"
    );
    let mut file = std::fs::File::create(path).map_err(|source| VideoEncodeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let write_error = |source| VideoEncodeError::Io {
        path: path.to_path_buf(),
        source,
    };
    for (index, segment) in segments.iter().enumerate() {
        let line = concat_file_line(segment)?;
        writeln!(file, "{line}").map_err(write_error)?;
        if let Some(span_us) = spans_to_next_us.get(index) {
            writeln!(file, "duration {}", duration_directive(*span_us)).map_err(write_error)?;
        }
    }
    Ok(())
}

/// Render one `file '...'` concat-list entry: the path made absolute (see
/// [`write_concat_list`]) and single-quoted with embedded quotes escaped per
/// the demuxer's own rule (`'` -> `'\''`).
fn concat_file_line(segment: &Path) -> Result<String, VideoEncodeError> {
    let absolute = if segment.is_absolute() {
        segment.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|source| VideoEncodeError::Io {
                path: segment.to_path_buf(),
                source,
            })?
            .join(segment)
    };
    let escaped = absolute.to_string_lossy().replace('\'', r"'\''");
    Ok(format!("file '{escaped}'"))
}

/// Render the concat list for a batched encode: each NUT entry followed,
/// for every entry except the last, by its declared `duration`. The demuxer
/// stacks inputs by declared duration, which is what lands each frame on
/// its capture timestamp. A single-entry list carries no `duration` line.
fn write_batch_concat_list(path: &Path, inputs: &[BatchNutInput]) -> Result<(), VideoEncodeError> {
    let mut file = std::fs::File::create(path).map_err(|source| VideoEncodeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let write_error = |source| VideoEncodeError::Io {
        path: path.to_path_buf(),
        source,
    };
    for input in inputs {
        let line = concat_file_line(&input.raw_nut)?;
        writeln!(file, "{line}").map_err(write_error)?;
        if let Some(span_us) = input.span_to_next_us {
            writeln!(file, "duration {}", duration_directive(span_us)).map_err(write_error)?;
        }
    }
    Ok(())
}

/// Ensure the parent directory for `path` exists. The trace actor normally
/// creates the trace directory before any encoder runs, but ffmpeg refuses to
/// emit into a missing directory and the recovery path may have removed it.
fn ensure_parent_dirs(path: &Path) -> Result<(), VideoEncodeError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|source| VideoEncodeError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
    }
    Ok(())
}

/// Stat `path` and return its byte length, erroring if the file is missing or
/// zero bytes. ffmpeg occasionally exits 0 but produces a zero-byte output when
/// the requested codec is unavailable in the local build.
fn non_empty_file_size(path: &Path) -> Result<u64, VideoEncodeError> {
    let metadata = std::fs::metadata(path).map_err(|source| {
        if source.kind() == std::io::ErrorKind::NotFound {
            VideoEncodeError::OutputMissing {
                path: path.to_path_buf(),
            }
        } else {
            VideoEncodeError::Io {
                path: path.to_path_buf(),
                source,
            }
        }
    })?;
    if metadata.len() == 0 {
        return Err(VideoEncodeError::OutputMissing {
            path: path.to_path_buf(),
        });
    }
    Ok(metadata.len())
}

/// Return the trailing portion of `stderr` as a lossy UTF-8 string, capped at
/// 4 KiB. The cap keeps log lines bounded when ffmpeg's diagnostic output runs
/// to megabytes (e.g. one warning per frame).
fn tail_stderr(stderr: &[u8]) -> String {
    const MAX_TAIL: usize = 4 * 1024;
    let start = stderr.len().saturating_sub(MAX_TAIL);
    String::from_utf8_lossy(&stderr[start..]).into_owned()
}

/// Extract the version token from `ffmpeg -version` stdout. The first line is
/// `ffmpeg version <token> ...` (e.g. `ffmpeg version 4.4.2-0ubuntu0.22.04.1
/// Copyright ...`); returns `"unknown"` when that prefix is absent (custom
/// builds occasionally reword it).
fn parse_ffmpeg_version(stdout: &[u8]) -> String {
    String::from_utf8_lossy(stdout)
        .lines()
        .next()
        .and_then(|line| line.strip_prefix("ffmpeg version "))
        .and_then(|rest| rest.split_whitespace().next())
        .map(|token| token.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command as StdCommand;
    use tempfile::TempDir;

    /// Locate an ffmpeg-suite binary on `PATH`. Returns `None` (with a
    /// caller-side skip) so the suite stays green in sandboxes that lack
    /// the FFmpeg toolchain.
    fn locate_binary(name: &str) -> Option<PathBuf> {
        let output = StdCommand::new("which").arg(name).output().ok()?;
        if !output.status.success() {
            return None;
        }
        let path = String::from_utf8(output.stdout).ok()?;
        let trimmed = path.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(PathBuf::from(trimmed))
        }
    }

    /// Synthesise a small NUT chunk via ffmpeg's `testsrc` source so the
    /// encoder tests don't need to pull in the producer crate just for the
    /// NUT writer. `frame_count` frames at the configured rate land in a
    /// NUT-container raw-rgb24 stream that `encode_chunk` can demux.
    fn write_synthetic_nut(ffmpeg: &Path, path: &Path, frame_count: u64) {
        write_synthetic_nut_sized(ffmpeg, path, frame_count, 16, 16);
    }

    /// As [`write_synthetic_nut`] but with an explicit frame geometry, so the
    /// preview-downscale test can feed a source larger than the preview cap.
    fn write_synthetic_nut_sized(
        ffmpeg: &Path,
        path: &Path,
        frame_count: u64,
        width: u32,
        height: u32,
    ) {
        let duration = format!("{}", frame_count); // 1 fps testsrc → frame_count seconds
        let status = StdCommand::new(ffmpeg)
            .args([
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
            ])
            .arg(format!(
                "testsrc=duration={duration}:size={width}x{height}:rate=1"
            ))
            .args(["-c:v", "rawvideo", "-pix_fmt", "rgb24", "-f", "nut"])
            .arg(path)
            .status()
            .expect("ffmpeg synth status");
        assert!(status.success(), "synthetic NUT generation failed");
    }

    #[test]
    fn missing_outputs_classify_as_output_missing() {
        let tempdir = TempDir::new().unwrap();
        let result = non_empty_file_size(&tempdir.path().join("absent.mp4"));
        assert!(matches!(
            result,
            Err(VideoEncodeError::OutputMissing { .. })
        ));
    }

    #[test]
    fn empty_outputs_classify_as_output_missing() {
        let tempdir = TempDir::new().unwrap();
        let path = tempdir.path().join("empty.mp4");
        std::fs::write(&path, []).unwrap();
        let result = non_empty_file_size(&path);
        assert!(matches!(
            result,
            Err(VideoEncodeError::OutputMissing { .. })
        ));
    }

    #[test]
    fn tail_stderr_caps_excessive_output() {
        let bytes = vec![b'x'; 16 * 1024];
        let tail = tail_stderr(&bytes);
        assert_eq!(tail.len(), 4 * 1024);
    }

    #[test]
    fn preview_scale_filter_builds_expected_expression() {
        // The comma inside `min(1, …)` MUST stay escaped (`\,`) — an unescaped
        // comma would be parsed as a filter separator and ffmpeg would reject
        // the graph. Both axes scale by the same `min(1, H/ih)` factor (AR
        // preserved, no upscale) and round to even (`trunc(/2)*2`).
        assert_eq!(
            preview_scale_filter(480),
            "scale=trunc(iw*min(1\\,480/ih)/2)*2:trunc(ih*min(1\\,480/ih)/2)*2:flags=fast_bilinear"
        );
        // The cap is interpolated, so a different target reshapes the filter.
        assert!(preview_scale_filter(720).contains("720/ih"));
    }

    #[test]
    fn parse_version_extracts_token_and_falls_back() {
        assert_eq!(
            parse_ffmpeg_version(b"ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000\n"),
            "4.4.2-0ubuntu0.22.04.1"
        );
        assert_eq!(parse_ffmpeg_version(b"ffmpeg version n6.1\n"), "n6.1");
        assert_eq!(parse_ffmpeg_version(b"some custom banner\n"), "unknown");
        assert_eq!(parse_ffmpeg_version(b""), "unknown");
    }

    #[test]
    fn preflight_reports_not_found_for_missing_binary() {
        let result = VideoEncoder::new()
            .with_binary("nc-definitely-not-a-real-ffmpeg-binary")
            .preflight();
        assert!(
            matches!(result, Err(FfmpegPreflightError::NotFound { .. })),
            "expected NotFound, got {result:?}"
        );
    }

    #[test]
    fn preflight_accepts_a_real_ffmpeg() {
        // Skip where the toolchain is unavailable, matching the encode tests.
        let Some(ffmpeg) = locate_binary("ffmpeg") else {
            return;
        };
        let version = VideoEncoder::new()
            .with_binary(ffmpeg)
            .preflight()
            .expect("system ffmpeg should pass preflight");
        assert!(!version.is_empty(), "version string should be populated");
    }

    #[test]
    fn concat_list_escapes_single_quotes() {
        let tempdir = TempDir::new().unwrap();
        let list = tempdir.path().join("list.txt");
        let segments = vec![
            PathBuf::from("/var/data/recordings/rec/cam/trace/chunks/chunk_0000.nut"),
            PathBuf::from("/var/data/rec'with quote/trace/chunks/chunk_0001.nut"),
        ];
        write_concat_list(&list, &segments, &[16_683]).expect("write list");
        let contents = std::fs::read_to_string(&list).unwrap();
        assert!(
            contents.contains("file '/var/data/recordings/rec/cam/trace/chunks/chunk_0000.nut'")
        );
        assert!(
            contents.contains(r"file '/var/data/rec'\''with quote/trace/chunks/chunk_0001.nut'"),
            "got: {contents}"
        );
    }

    #[test]
    fn concat_list_absolutises_relative_segment_paths() {
        // ffmpeg's concat demuxer resolves entries against the list-file's
        // directory, not the daemon's CWD. Relative segment paths must be
        // joined against the current working directory before being written
        // so the demuxer ends up at the same file the daemon meant to open.
        let tempdir = TempDir::new().unwrap();
        let list = tempdir.path().join("list.txt");
        let cwd = std::env::current_dir().unwrap();
        let segments = vec![PathBuf::from("rel/chunk_0000.mp4")];
        write_concat_list(&list, &segments, &[]).expect("write list");
        let contents = std::fs::read_to_string(&list).unwrap();
        let expected = cwd.join("rel/chunk_0000.mp4");
        assert!(
            contents.contains(&format!("file '{}'", expected.display())),
            "got: {contents}"
        );
    }

    #[test]
    fn concat_list_emits_duration_lines_except_last() {
        let tempdir = TempDir::new().unwrap();
        let list = tempdir.path().join("list.txt");
        let segments = vec![
            PathBuf::from("/data/trace/chunk_0000_lossy.mp4"),
            PathBuf::from("/data/trace/chunk_0001_lossy.mp4"),
            PathBuf::from("/data/trace/chunk_0003_lossy.mp4"),
        ];
        write_concat_list(&list, &segments, &[16_683, MAX_BOUNDARY_DELTA_US]).expect("write list");
        let contents = std::fs::read_to_string(&list).unwrap();
        assert_eq!(
            contents,
            "file '/data/trace/chunk_0000_lossy.mp4'\n\
             duration 0.016683\n\
             file '/data/trace/chunk_0001_lossy.mp4'\n\
             duration 2147.483646\n\
             file '/data/trace/chunk_0003_lossy.mp4'\n"
        );
    }

    #[test]
    fn single_segment_concat_list_stays_byte_identical() {
        // A single-segment trace carries no `duration` line: its list stays
        // byte-identical to the pre-directive format.
        let tempdir = TempDir::new().unwrap();
        let list = tempdir.path().join("list.txt");
        let segments = vec![PathBuf::from("/data/trace/chunk_0000_lossy.mp4")];
        write_concat_list(&list, &segments, &[]).expect("write list");
        let contents = std::fs::read_to_string(&list).unwrap();
        assert_eq!(contents, "file '/data/trace/chunk_0000_lossy.mp4'\n");
    }

    #[test]
    fn concat_segments_rejects_empty_input() {
        let tempdir = TempDir::new().unwrap();
        let out = tempdir.path().join("out.mp4");
        // Sync wrapper so the test body isn't async for this trivial case.
        let result = futures_block(VideoEncoder::new().concat_segments(&[], &[], &out));
        assert!(matches!(result, Err(VideoEncodeError::EmptySegments)));
    }

    /// Drive a future to completion on a single-threaded tokio runtime.
    /// Used by the trivial unit tests that don't need `#[tokio::test]`
    /// scaffolding.
    fn futures_block<T>(future: impl std::future::Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[tokio::test]
    async fn encode_chunk_emits_sealed_mp4_outputs() {
        let ffmpeg = match locate_binary("ffmpeg") {
            Some(path) => path,
            None => {
                eprintln!(
                    "ffmpeg not on PATH — skipping encode_chunk test. Install \
                     `ffmpeg` to enable this test."
                );
                return;
            }
        };
        let ffprobe = match locate_binary("ffprobe") {
            Some(path) => path,
            None => {
                eprintln!("ffprobe not on PATH — skipping encode_chunk test.");
                return;
            }
        };

        let tempdir = TempDir::new().unwrap();
        let raw = tempdir.path().join("chunk_0000.nut");
        let lossy = tempdir.path().join("chunk_0000_lossy.mp4");
        let lossless = tempdir.path().join("chunk_0000_lossless.mp4");

        write_synthetic_nut(&ffmpeg, &raw, 8);

        let encoder = VideoEncoder::new();
        let request = ChunkEncodeRequest {
            raw_nut: raw.clone(),
            lossy_out: lossy.clone(),
            lossless_out: lossless.clone(),
            codec: LossyVideoCodec::LosslessPlusPreview,
            frame_count: 8,
            skip_frames: 0,
        };
        let outcome = encoder
            .encode_chunk(&request, ENCODE_THREADS_PER_OUTPUT)
            .await
            .expect("transcode");

        assert!(outcome.lossy_bytes > 0);
        assert!(outcome.lossless_bytes > 0);
        // The new encode_chunk leaves the source in place — the per-trace
        // actor owns the unlink on its own success path so a partial
        // post-encode failure can still be cleaned up by the recovery sweep.
        assert!(raw.exists(), "encode_chunk must not unlink its source");

        for path in [&lossy, &lossless] {
            let status = StdCommand::new(&ffprobe)
                .args(["-v", "error", "-show_streams", "-of", "json"])
                .arg(path)
                .output()
                .expect("spawn ffprobe");
            assert!(
                status.status.success(),
                "ffprobe rejected {}: stderr={}",
                path.display(),
                String::from_utf8_lossy(&status.stderr)
            );
            let parsed: serde_json::Value =
                serde_json::from_slice(&status.stdout).expect("ffprobe JSON");
            let streams = parsed["streams"].as_array().expect("streams array");
            assert_eq!(
                streams.len(),
                1,
                "{} should contain exactly one stream",
                path.display()
            );
            assert_eq!(streams[0]["codec_type"], "video");
            // 16x16 is far below the preview cap, so the lossy downscale is a
            // no-op here — both outputs keep the source geometry (no upscale).
            assert_eq!(streams[0]["width"], 16);
            assert_eq!(streams[0]["height"], 16);
        }
    }

    #[tokio::test]
    async fn encode_chunk_downscales_lossy_preview_keeps_lossless_native() {
        let (Some(ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping preview-downscale test.");
            return;
        };

        let tempdir = TempDir::new().unwrap();
        let raw = tempdir.path().join("chunk_0000.nut");
        let lossy = tempdir.path().join("chunk_0000_lossy.mp4");
        let lossless = tempdir.path().join("chunk_0000_lossless.mp4");

        // A 1280x720 source: above the 480-line preview cap, 16:9 aspect.
        write_synthetic_nut_sized(&ffmpeg, &raw, 6, 1280, 720);

        let encoder = VideoEncoder::new();
        encoder
            .encode_chunk(
                &ChunkEncodeRequest {
                    raw_nut: raw,
                    lossy_out: lossy.clone(),
                    lossless_out: lossless.clone(),
                    codec: LossyVideoCodec::LosslessPlusPreview,
                    frame_count: 6,
                    skip_frames: 0,
                },
                ENCODE_THREADS_PER_OUTPUT,
            )
            .await
            .expect("transcode");

        let dims = |path: &Path| -> (u64, u64, u64) {
            let out = StdCommand::new(&ffprobe)
                .args([
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=width,height,nb_read_frames",
                    "-of",
                    "json",
                ])
                .arg(path)
                .output()
                .expect("spawn ffprobe");
            let parsed: serde_json::Value =
                serde_json::from_slice(&out.stdout).expect("ffprobe JSON");
            let stream = &parsed["streams"][0];
            let field = |key: &str| -> u64 {
                let value = &stream[key];
                value
                    .as_u64()
                    .or_else(|| value.as_str().and_then(|s| s.parse().ok()))
                    .unwrap_or_else(|| panic!("missing {key}: {stream}"))
            };
            (field("width"), field("height"), field("nb_read_frames"))
        };

        let (lossy_w, lossy_h, lossy_frames) = dims(&lossy);
        let (lossless_w, lossless_h, lossless_frames) = dims(&lossless);

        // Lossy is capped to 480 lines, aspect ratio preserved (1280x720 ->
        // 852x480), and both axes are even (yuv420p requirement).
        assert_eq!(
            (lossy_w, lossy_h),
            (852, 480),
            "lossy should be 480p preview"
        );
        assert_eq!(lossy_w % 2, 0, "lossy width must be even");
        // Lossless keeps the native geometry — it is the archival copy.
        assert_eq!(
            (lossless_w, lossless_h),
            (1280, 720),
            "lossless must stay native resolution"
        );
        // Both outputs carry every source frame, so the per-frame timestamp
        // sidecar stays aligned with each video.
        assert_eq!(
            lossy_frames, lossless_frames,
            "lossy and lossless must hold the same frame count"
        );
        assert_eq!(lossy_frames, 6, "all source frames must be encoded");
    }

    #[tokio::test]
    async fn frame_count_drops_the_tail_and_keeps_the_frames_the_recording_owns() {
        // The NUT cannot be rewritten, so the cut has to reach ffmpeg — on
        // both outputs, or an mp4 outruns the sidecar indexing it.
        let (Some(ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping frame-cap encode test.");
            return;
        };

        let tempdir = TempDir::new().unwrap();
        let raw = tempdir.path().join("chunk_0000.nut");
        let lossy = tempdir.path().join("chunk_0000_lossy.mp4");
        let lossless = tempdir.path().join("chunk_0000_lossless.mp4");
        write_synthetic_nut(&ffmpeg, &raw, 8);

        VideoEncoder::new()
            .encode_chunk(
                &ChunkEncodeRequest {
                    raw_nut: raw,
                    lossy_out: lossy.clone(),
                    lossless_out: lossless.clone(),
                    codec: LossyVideoCodec::LosslessPlusPreview,
                    frame_count: 3,
                    skip_frames: 0,
                },
                ENCODE_THREADS_PER_OUTPUT,
            )
            .await
            .expect("transcode");

        let frame_count = |path: &Path| -> u64 {
            let out = StdCommand::new(&ffprobe)
                .args([
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=nokey=1:noprint_wrappers=1",
                ])
                .arg(path)
                .output()
                .expect("spawn ffprobe");
            String::from_utf8_lossy(&out.stdout).trim().parse().unwrap()
        };

        assert_eq!(frame_count(&lossy), 3, "lossy output must stop at the cut");
        assert_eq!(
            frame_count(&lossless),
            3,
            "lossless output must stop at the same cut"
        );
    }

    #[tokio::test]
    async fn skip_frames_drops_the_head_and_rebases_the_kept_frames() {
        // The head cut has to reach ffmpeg for the same reason the tail cut
        // does, and the kept frames must start at PTS 0 so the segment's
        // extent is the extent the finalise concat floors on.
        let (Some(ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping head-cut encode test.");
            return;
        };

        let tempdir = TempDir::new().unwrap();
        let raw = tempdir.path().join("chunk_0000.nut");
        write_synthetic_nut(&ffmpeg, &raw, 8);

        let probe = |path: &Path, entries: &str, intervals: Option<&str>| -> String {
            let mut command = StdCommand::new(&ffprobe);
            command.args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                entries,
                "-of",
                "default=nokey=1:noprint_wrappers=1",
            ]);
            if let Some(intervals) = intervals {
                command.args(["-read_intervals", intervals]);
            }
            let out = command.arg(path).output().expect("spawn ffprobe");
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        let frame_count =
            |path: &Path| -> u64 { probe(path, "stream=nb_read_frames", None).parse().unwrap() };

        // Head cut only: the 5 frames past the cut survive on both outputs.
        let lossy = tempdir.path().join("chunk_0000_lossy.mp4");
        let lossless = tempdir.path().join("chunk_0000_lossless.mp4");
        VideoEncoder::new()
            .encode_chunk(
                &ChunkEncodeRequest {
                    raw_nut: raw.clone(),
                    lossy_out: lossy.clone(),
                    lossless_out: lossless.clone(),
                    codec: LossyVideoCodec::LosslessPlusPreview,
                    frame_count: 5,
                    skip_frames: 3,
                },
                ENCODE_THREADS_PER_OUTPUT,
            )
            .await
            .expect("transcode");
        assert_eq!(frame_count(&lossy), 5, "lossy output must drop the head");
        assert_eq!(
            frame_count(&lossless),
            5,
            "lossless output must drop the same head"
        );
        // The synthetic NUT is 1 fps, so the kept run starts at 3 s in the
        // source; `setpts` has to bring it back to zero.
        for output in [&lossy, &lossless] {
            assert_eq!(
                probe(output, "frame=pts_time", Some("%+#1")),
                "0.000000",
                "the first kept frame must be rebased to PTS 0"
            );
        }

        // Cut at both ends: `-frames:v` counts frames *after* the filter
        // graph, so the cap bounds the tail of what the head cut left.
        let lossy = tempdir.path().join("chunk_0001_lossy.mp4");
        let lossless = tempdir.path().join("chunk_0001_lossless.mp4");
        VideoEncoder::new()
            .encode_chunk(
                &ChunkEncodeRequest {
                    raw_nut: raw,
                    lossy_out: lossy.clone(),
                    lossless_out: lossless.clone(),
                    codec: LossyVideoCodec::LosslessPlusPreview,
                    frame_count: 2,
                    skip_frames: 3,
                },
                ENCODE_THREADS_PER_OUTPUT,
            )
            .await
            .expect("transcode");
        assert_eq!(frame_count(&lossy), 2, "the cap applies past the head cut");
        assert_eq!(
            frame_count(&lossless),
            2,
            "the cap applies past the head cut on the lossless output too"
        );
    }

    #[tokio::test]
    async fn encode_chunk_lossy_only_writes_single_full_res_h264() {
        let (Some(ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping lossy-only encode test.");
            return;
        };

        let tempdir = TempDir::new().unwrap();
        let raw = tempdir.path().join("chunk_0000.nut");
        let lossy = tempdir.path().join("chunk_0000_lossy.mp4");
        let lossless = tempdir.path().join("chunk_0000_lossless.mp4");

        // 1280x720 source, above the 480-line preview cap. Lossy-only must NOT
        // downscale — the single output is the training-quality video.
        write_synthetic_nut_sized(&ffmpeg, &raw, 6, 1280, 720);

        let encoder = VideoEncoder::new();
        let outcome = encoder
            .encode_chunk(
                &ChunkEncodeRequest {
                    raw_nut: raw,
                    lossy_out: lossy.clone(),
                    lossless_out: lossless.clone(),
                    codec: LossyVideoCodec::H264MediumLossyOnly,
                    frame_count: 6,
                    skip_frames: 0,
                },
                ENCODE_THREADS_PER_OUTPUT,
            )
            .await
            .expect("transcode");

        // No lossless archive is produced in lossy-only mode.
        assert_eq!(outcome.lossless_bytes, 0, "no lossless output expected");
        assert!(!lossless.exists(), "lossless.mp4 must not be written");
        assert!(outcome.lossy_bytes > 0);

        // The single video keeps native resolution, is H.264, and carries every
        // source frame so it stays aligned with the per-frame timestamp sidecar.
        let probe = StdCommand::new(&ffprobe)
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,nb_read_frames,codec_name",
                "-of",
                "json",
            ])
            .arg(&lossy)
            .output()
            .expect("spawn ffprobe");
        assert!(probe.status.success());
        let parsed: serde_json::Value =
            serde_json::from_slice(&probe.stdout).expect("ffprobe JSON");
        let stream = &parsed["streams"][0];
        assert_eq!(stream["codec_name"], "h264");
        assert_eq!(stream["width"], 1280);
        assert_eq!(stream["height"], 720);
        let frames: u64 = stream["nb_read_frames"]
            .as_u64()
            .or_else(|| {
                stream["nb_read_frames"]
                    .as_str()
                    .and_then(|value| value.parse().ok())
            })
            .expect("frame count");
        assert_eq!(frames, 6, "all source frames must be encoded");
    }

    #[tokio::test]
    async fn encode_chunk_pins_the_microsecond_track_timescale() {
        // Without the pinned `-enc_time_base` / `-video_track_timescale`,
        // ffmpeg derives each segment's timescale from a per-chunk guessed
        // frame rate. The guess differs between chunks of one recording
        // (near-constant capture deltas keep the microsecond base, jittery
        // ones normalise to a standard rate), and the stream-copy concat of
        // mixed-timescale segments crams whole chunks onto consecutive
        // single ticks with backwards decoded PTS — the backend's "Video
        // missing logged frames" rejection — from perfectly clean input.
        // Every output must therefore carry the NUT's 1/1000000 clock.
        let ffmpeg = match locate_binary("ffmpeg") {
            Some(path) => path,
            None => {
                eprintln!("ffmpeg not on PATH — skipping timescale test.");
                return;
            }
        };
        let ffprobe = match locate_binary("ffprobe") {
            Some(path) => path,
            None => {
                eprintln!("ffprobe not on PATH — skipping timescale test.");
                return;
            }
        };

        let tempdir = TempDir::new().unwrap();
        let raw = tempdir.path().join("chunk_0000.nut");
        write_synthetic_nut(&ffmpeg, &raw, 8);

        let encoder = VideoEncoder::new();
        let split_lossy = tempdir.path().join("split_lossy.mp4");
        let split_lossless = tempdir.path().join("split_lossless.mp4");
        encoder
            .encode_chunk(
                &ChunkEncodeRequest {
                    raw_nut: raw.clone(),
                    lossy_out: split_lossy.clone(),
                    lossless_out: split_lossless.clone(),
                    codec: LossyVideoCodec::LosslessPlusPreview,
                    frame_count: 8,
                    skip_frames: 0,
                },
                ENCODE_THREADS_PER_OUTPUT,
            )
            .await
            .expect("split transcode");
        let single_lossy = tempdir.path().join("single_lossy.mp4");
        encoder
            .encode_chunk(
                &ChunkEncodeRequest {
                    raw_nut: raw,
                    lossy_out: single_lossy.clone(),
                    lossless_out: tempdir.path().join("unused_lossless.mp4"),
                    codec: LossyVideoCodec::H264MediumLossyOnly,
                    frame_count: 8,
                    skip_frames: 0,
                },
                ENCODE_THREADS_PER_OUTPUT,
            )
            .await
            .expect("lossy-only transcode");

        for path in [&split_lossy, &split_lossless, &single_lossy] {
            let probe = StdCommand::new(&ffprobe)
                .args([
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=time_base",
                    "-of",
                    "csv=p=0",
                ])
                .arg(path)
                .output()
                .expect("spawn ffprobe");
            assert!(probe.status.success());
            let time_base = String::from_utf8_lossy(&probe.stdout).trim().to_string();
            assert_eq!(
                time_base,
                format!("1/{VIDEO_SPOOL_TICKS_PER_SECOND}"),
                "{} must carry the pinned microsecond timescale",
                path.display()
            );
        }
    }

    #[tokio::test]
    async fn concat_of_mixed_cadence_chunks_keeps_monotonic_pts() {
        // The terminal symptom this pipeline must never reproduce: chunks of
        // one recording whose capture cadence differs in character (jittery
        // vs metronome-constant) used to encode to segments with *different*
        // guessed timescales, and the stream-copy concat of those crammed a
        // whole chunk onto consecutive single ticks with backwards decoded
        // PTS. Build exactly that fixture with the real producer NUT writer
        // and assert the merged video stays sound.
        let ffprobe = match locate_binary("ffprobe") {
            Some(path) => path,
            None => {
                eprintln!("ffprobe not on PATH — skipping mixed-cadence test.");
                return;
            }
        };
        if locate_binary("ffmpeg").is_none() {
            eprintln!("ffmpeg not on PATH — skipping mixed-cadence test.");
            return;
        }

        use data_daemon_bridge::nut_writer::{NutVideoConfig, NutWriter};
        let tempdir = TempDir::new().unwrap();
        let frames_per_chunk: i64 = 48;
        let rgb = vec![128u8; 16 * 16 * 3];
        let write_chunk = |path: &Path, jittery: bool| -> Vec<f64> {
            let mut writer = NutWriter::create(
                path,
                NutVideoConfig {
                    width: 16,
                    height: 16,
                    time_base_num: 1,
                    time_base_den: VIDEO_SPOOL_TICKS_PER_SECOND,
                },
            )
            .expect("create NUT");
            let mut timestamps_s = Vec::new();
            let mut timestamp_us: i64 = 0;
            for index in 0..frames_per_chunk {
                // ~59.9 fps; the jittery variant wobbles ±0.5 ms like real
                // capture, the constant variant ticks like a metronome — the
                // exact contrast that used to flip the guessed timescale.
                timestamp_us += if jittery {
                    16_740 + ((index * 7_919) % 1_000) - 500
                } else {
                    16_683
                };
                writer
                    .write_frame(timestamp_us as u64, &rgb)
                    .expect("write frame");
                timestamps_s.push(timestamp_us as f64 / 1e6);
            }
            writer.finish().expect("finish NUT");
            timestamps_s
        };

        let encoder = VideoEncoder::new();
        let mut segments = Vec::new();
        let mut segment_timestamps_s = Vec::new();
        for (chunk_index, jittery) in [true, false, true].into_iter().enumerate() {
            let raw = tempdir.path().join(format!("chunk_{chunk_index:04}.nut"));
            let lossy = tempdir
                .path()
                .join(format!("chunk_{chunk_index:04}_lossy.mp4"));
            segment_timestamps_s.push(write_chunk(&raw, jittery));
            encoder
                .encode_chunk(
                    &ChunkEncodeRequest {
                        raw_nut: raw,
                        lossy_out: lossy.clone(),
                        lossless_out: tempdir
                            .path()
                            .join(format!("chunk_{chunk_index:04}_lossless.mp4")),
                        codec: LossyVideoCodec::LosslessPlusPreview,
                        frame_count: frames_per_chunk as u32,
                        skip_frames: 0,
                    },
                    ENCODE_THREADS_PER_OUTPUT,
                )
                .await
                .expect("transcode chunk");
            segments.push(lossy);
        }

        // These fixture chunks re-anchor at each chunk open, so the spans
        // floor to each segment's extent plus 1 us; the merge must stay sound.
        let spans_to_next_us: Vec<i64> = segment_timestamps_s
            .windows(2)
            .map(|pair| declared_batch_span_us(&pair[0], pair[1][0]))
            .collect();
        let final_lossy = tempdir.path().join("lossy.mp4");
        encoder
            .concat_segments(&segments, &spans_to_next_us, &final_lossy)
            .await
            .expect("concat");
        assert_merged_video_is_sound(&ffprobe, &final_lossy);
    }

    #[test]
    fn lossy_video_codec_resolves_from_config_str() {
        assert_eq!(
            LossyVideoCodec::from_config_str(Some("h264_medium")),
            LossyVideoCodec::H264MediumLossyOnly
        );
        assert!(LossyVideoCodec::from_config_str(Some("h264_medium")).is_lossy_only());
        // h264_lossless is the explicit default; unset/unknown also default.
        for value in [None, Some(""), Some("unknown"), Some("h264_lossless")] {
            assert_eq!(
                LossyVideoCodec::from_config_str(value),
                LossyVideoCodec::LosslessPlusPreview,
                "{value:?} should map to the default codec"
            );
            assert!(!LossyVideoCodec::from_config_str(value).is_lossy_only());
        }
    }

    #[test]
    fn for_trace_gates_lossy_codec_to_rgb_only() {
        // Only RGB honours a lossy codec; depth and every non-RGB stream keep
        // the lossless archive even when h264_medium is configured. This is the
        // core RGB-only invariant of the feature — a regression that dropped a
        // depth lossless archive would corrupt depth training data.
        assert_eq!(
            LossyVideoCodec::for_trace("RGB_IMAGES", Some("h264_medium")),
            LossyVideoCodec::H264MediumLossyOnly
        );
        for data_type in ["DEPTH_IMAGES", "JOINT_POSITIONS", "CUSTOM_1D", ""] {
            assert_eq!(
                LossyVideoCodec::for_trace(data_type, Some("h264_medium")),
                LossyVideoCodec::LosslessPlusPreview,
                "{data_type} must keep lossless regardless of the codec"
            );
            assert!(!LossyVideoCodec::for_trace(data_type, Some("h264_medium")).is_lossy_only());
        }
        // RGB with the default/unset codec stays on the lossless+preview path.
        for value in [None, Some(""), Some("h264_lossless")] {
            assert_eq!(
                LossyVideoCodec::for_trace("RGB_IMAGES", value),
                LossyVideoCodec::LosslessPlusPreview
            );
        }
    }

    #[test]
    fn wire_str_round_trips_through_from_config_str() {
        // These strings are persisted as-is against the recording, so a drift
        // here silently mislabels every one.
        for codec in [
            LossyVideoCodec::LosslessPlusPreview,
            LossyVideoCodec::H264MediumLossyOnly,
        ] {
            assert_eq!(
                LossyVideoCodec::from_config_str(Some(codec.as_wire_str())),
                codec,
                "{codec:?} must survive a wire round trip"
            );
        }
        assert_eq!(
            LossyVideoCodec::LosslessPlusPreview.as_wire_str(),
            "h264_lossless"
        );
        assert_eq!(
            LossyVideoCodec::H264MediumLossyOnly.as_wire_str(),
            "h264_medium"
        );
    }

    #[tokio::test]
    async fn concat_segments_produces_single_mp4() {
        let ffmpeg = match locate_binary("ffmpeg") {
            Some(path) => path,
            None => {
                eprintln!("ffmpeg not on PATH — skipping concat_segments test.");
                return;
            }
        };
        let ffprobe = match locate_binary("ffprobe") {
            Some(path) => path,
            None => {
                eprintln!("ffprobe not on PATH — skipping concat_segments test.");
                return;
            }
        };

        let tempdir = TempDir::new().unwrap();
        let encoder = VideoEncoder::new();
        let mut segments = Vec::new();
        let total_frames: u64 = 4 * 3;
        // Encode three synthetic 4-frame NUT chunks into per-chunk MP4s.
        for chunk_index in 0..3u32 {
            let raw = tempdir.path().join(format!("chunk_{chunk_index:04}.nut"));
            let lossy = tempdir
                .path()
                .join(format!("chunk_{chunk_index:04}_lossy.mp4"));
            let lossless = tempdir
                .path()
                .join(format!("chunk_{chunk_index:04}_lossless.mp4"));
            write_synthetic_nut(&ffmpeg, &raw, 4);
            encoder
                .encode_chunk(
                    &ChunkEncodeRequest {
                        raw_nut: raw,
                        lossy_out: lossy.clone(),
                        lossless_out: lossless,
                        codec: LossyVideoCodec::LosslessPlusPreview,
                        frame_count: 4,
                        skip_frames: 0,
                    },
                    ENCODE_THREADS_PER_OUTPUT,
                )
                .await
                .expect("transcode chunk");
            segments.push(lossy);
        }

        let final_lossy = tempdir.path().join("lossy.mp4");
        // The synthetic chunks run at 1 fps with frames at 0..3 s, so each
        // segment spans 4 s to the next on a contiguous cadence.
        let outcome = encoder
            .concat_segments(&segments, &[4_000_000, 4_000_000], &final_lossy)
            .await
            .expect("concat");
        assert!(outcome.bytes > 0);

        // The concat list file lives next to the output during encoding; the
        // success path unlinks it.
        let list = list_file_for(&final_lossy);
        assert!(!list.exists(), "concat list file should be cleaned up");

        let probe = StdCommand::new(&ffprobe)
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
            ])
            .arg(&final_lossy)
            .output()
            .expect("spawn ffprobe");
        assert!(probe.status.success());
        let trimmed = String::from_utf8(probe.stdout).unwrap();
        let nb_read_frames: u64 = trimmed.trim().parse().unwrap();
        assert_eq!(
            nb_read_frames, total_frames,
            "concat output should contain all {total_frames} frames"
        );
        assert_merged_video_is_sound(&ffprobe, &final_lossy);
    }

    /// Assert the invariants the whole per-chunk pipeline exists to protect on
    /// a concatenated video: the merged track carries the pinned microsecond
    /// timescale and its decoded frames present in strictly increasing PTS
    /// order (the backend's `synchronize_video` rejects the file on the first
    /// backwards step as "Video missing logged frames").
    fn assert_merged_video_is_sound(ffprobe: &Path, video: &Path) {
        let probe = StdCommand::new(ffprobe)
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=time_base",
                "-of",
                "csv=p=0",
            ])
            .arg(video)
            .output()
            .expect("spawn ffprobe");
        assert!(probe.status.success());
        assert_eq!(
            String::from_utf8_lossy(&probe.stdout).trim(),
            format!("1/{VIDEO_SPOOL_TICKS_PER_SECOND}"),
            "{} must keep the pinned microsecond timescale through the concat",
            video.display()
        );

        // Decode-order PTS, exactly as the backend guard walks them.
        let pts_values = decoded_frame_pts(ffprobe, video);
        assert!(
            !pts_values.is_empty(),
            "{} yielded no decoded PTS",
            video.display()
        );
        for pair in pts_values.windows(2) {
            assert!(
                pair[1] > pair[0],
                "{}: decoded PTS must be strictly increasing, got {pair:?}",
                video.display()
            );
        }
    }

    /// Decode `video` and collect its frames' PTS values in the order the
    /// decoder presents them.
    fn decoded_frame_pts(ffprobe: &Path, video: &Path) -> Vec<i64> {
        let probe = StdCommand::new(ffprobe)
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=pts,pkt_pts",
                "-of",
                "default=noprint_wrappers=1",
            ])
            .arg(video)
            .output()
            .expect("spawn ffprobe");
        assert!(probe.status.success());
        let stdout = String::from_utf8_lossy(&probe.stdout);
        stdout
            .lines()
            .filter_map(|line| {
                line.strip_prefix("pts=")
                    .or_else(|| line.strip_prefix("pkt_pts="))
            })
            .filter_map(|value| value.parse().ok())
            .collect()
    }

    #[tokio::test]
    async fn missing_input_yields_non_zero_exit() {
        if locate_binary("ffmpeg").is_none() {
            eprintln!("ffmpeg not on PATH — skipping non-zero-exit test.");
            return;
        }

        let tempdir = TempDir::new().unwrap();
        let request = ChunkEncodeRequest {
            raw_nut: tempdir.path().join("does-not-exist.nut"),
            lossy_out: tempdir.path().join("lossy.mp4"),
            lossless_out: tempdir.path().join("lossless.mp4"),
            codec: LossyVideoCodec::LosslessPlusPreview,
            frame_count: 1,
            skip_frames: 0,
        };
        let encoder = VideoEncoder::new();
        let error = encoder
            .encode_chunk(&request, ENCODE_THREADS_PER_OUTPUT)
            .await
            .expect_err("ffmpeg should fail");
        assert!(
            matches!(error, VideoEncodeError::NonZeroExit { .. }),
            "unexpected error variant: {error:?}"
        );
    }

    #[tokio::test]
    async fn spawn_failure_surfaces_binary_name() {
        let tempdir = TempDir::new().unwrap();
        let raw = tempdir.path().join("raw.nut");
        std::fs::write(&raw, [0u8; 16]).unwrap();
        let request = ChunkEncodeRequest {
            raw_nut: raw,
            lossy_out: tempdir.path().join("lossy.mp4"),
            lossless_out: tempdir.path().join("lossless.mp4"),
            codec: LossyVideoCodec::LosslessPlusPreview,
            frame_count: 1,
            skip_frames: 0,
        };
        let encoder =
            VideoEncoder::new().with_binary("this-binary-definitely-does-not-exist-ffmpeg");
        let error = encoder
            .encode_chunk(&request, ENCODE_THREADS_PER_OUTPUT)
            .await
            .expect_err("spawn should fail");
        match error {
            VideoEncodeError::Spawn { binary, .. } => {
                assert_eq!(
                    binary,
                    OsString::from("this-binary-definitely-does-not-exist-ffmpeg")
                );
            }
            other => panic!("expected Spawn error, got {other:?}"),
        }
    }

    /// Write a NUT chunk with the real producer NUT writer: one 16x16 RGB
    /// frame per entry of `frame_pts_us` (chunk-relative microsecond ticks;
    /// the producer re-anchors every chunk's first frame near 0).
    fn write_nut_chunk(path: &Path, frame_pts_us: &[i64]) {
        use data_daemon_bridge::nut_writer::{NutVideoConfig, NutWriter};
        let rgb = vec![128u8; 16 * 16 * 3];
        let mut writer = NutWriter::create(
            path,
            NutVideoConfig {
                width: 16,
                height: 16,
                time_base_num: 1,
                time_base_den: VIDEO_SPOOL_TICKS_PER_SECOND,
            },
        )
        .expect("create NUT");
        for pts in frame_pts_us {
            writer.write_frame(*pts as u64, &rgb).expect("write frame");
        }
        writer.finish().expect("finish NUT");
    }

    #[test]
    fn capture_timestamp_us_truncates_like_the_nut_writer() {
        assert_eq!(capture_timestamp_us(1.0), 1_000_000);
        assert_eq!(capture_timestamp_us(0.000001), 1);
        // Truncation toward zero, not rounding: 1.9 us of capture time is
        // still tick 1, exactly as the writer's ns-then-divide conversion.
        assert_eq!(capture_timestamp_us(0.0000019), 1);
    }

    #[test]
    fn declared_span_never_goes_negative() {
        assert_eq!(declared_span_us(0.0, 1.0), 1_000_000);
        // A backwards announcement never yields a negative duration.
        assert_eq!(declared_span_us(2.0, 1.0), 0);
    }

    #[test]
    fn replayed_extent_matches_healthy_stamps() {
        // Monotonic stamps trigger no synthesis: the extent is the plain
        // last-minus-first capture span in writer-truncated microseconds.
        assert_eq!(replayed_chunk_extent_us(&[0.0, 0.016683, 0.033366]), 33_366);
        assert_eq!(replayed_chunk_extent_us(&[1.0]), 0);
        assert_eq!(replayed_chunk_extent_us(&[]), 0);
    }

    #[test]
    fn replayed_extent_covers_the_writer_synthesis() {
        // All-duplicate stamps: the writer synthesizes a step per frame from
        // the healthy gap it carried from earlier chunks. That gap is capped
        // at 100 ms, so the replay's worst-case seed never undershoots.
        assert_eq!(replayed_chunk_extent_us(&[1.0, 1.0, 1.0]), 200_000);
        // A healthy gap observed inside the chunk becomes the step for a
        // later duplicate, exactly as the writer applies it.
        assert_eq!(replayed_chunk_extent_us(&[0.0, 0.016683, 0.016683]), 33_366);
        // After a synthesized step the origin re-anchors on the regressed
        // frame, so later stamps resume true capture spacing from it.
        assert_eq!(
            replayed_chunk_extent_us(&[0.0, 0.016683, 0.016683, 0.033366]),
            50_049
        );
        // A regression below the chunk origin clamps to PTS 0 first, then
        // synthesizes past the previous frame.
        assert_eq!(replayed_chunk_extent_us(&[1.0, 0.5]), 100_000);
    }

    #[test]
    fn batch_span_floors_at_the_chunk_extent_plus_one() {
        let capture_s = |us: i64| us as f64 / 1e6;
        let chunk: Vec<f64> = [0, 16_683, 33_366]
            .iter()
            .map(|us| capture_s(*us))
            .collect();
        // Overlap: the next chunk's announced start sits inside this chunk's
        // content span; the declared span floors to the extent plus 1 us.
        assert_eq!(declared_batch_span_us(&chunk, capture_s(20_000)), 33_367);
        // A well-formed span is at least the extent plus one frame interval
        // and passes through unchanged.
        assert_eq!(declared_batch_span_us(&chunk, capture_s(50_049)), 50_049);
        // A synthesized-PTS chunk: the announced extent is zero, but the
        // NUT's real content extends up to one writer step per frame. The
        // floor tracks the replayed extent, not the announced one.
        let duplicates = [1.0, 1.0, 1.0];
        assert_eq!(
            declared_batch_span_us(&duplicates, capture_s(1_005_000)),
            200_001
        );
        // A chunk whose own extent exceeds the boundary ceiling still floors
        // to extent plus one; the boundary delta stays 1 us.
        let long_chunk = [0.0, 5_000.0];
        assert_eq!(
            declared_batch_span_us(&long_chunk, capture_s(20_000)),
            5_000_000_001
        );
        // A chunk with no announced frames contributes extent zero and a
        // 1 us span instead of panicking.
        assert_eq!(declared_batch_span_us(&[], capture_s(20_000)), 1);
    }

    #[test]
    fn batch_span_ceilings_the_boundary_delta() {
        let capture_s = |us: i64| us as f64 / 1e6;
        // A gap past ~35.8 minutes saturates the 32-bit mp4 boundary sample
        // delta: the declared span compresses so the step past the chunk's
        // content extent never exceeds MAX_BOUNDARY_DELTA_US.
        assert_eq!(
            declared_batch_span_us(&[0.0], capture_s(5_000_000_000)),
            MAX_BOUNDARY_DELTA_US
        );
        let chunk = [0.0, 0.016683];
        assert_eq!(
            declared_batch_span_us(&chunk, capture_s(5_000_000_000)),
            16_683 + MAX_BOUNDARY_DELTA_US
        );
    }

    #[test]
    fn span_with_extent_floors_and_ceilings_on_the_carried_extent() {
        let capture_s = |us: i64| us as f64 / 1e6;
        // A segment whose announced stamps replay to 18000 us but whose real
        // placement extent is 33000 us: the raw span (19000 us) sits inside
        // the real content, so the floor binds.
        let segment_stamps = [0.0, 0.016, 0.017, 0.0165];
        assert_eq!(
            declared_span_with_extent_us(&segment_stamps, capture_s(19_000), 33_000),
            33_001
        );
        // A raw span past the real content passes through unchanged.
        assert_eq!(
            declared_span_with_extent_us(&segment_stamps, capture_s(50_000), 33_000),
            50_000
        );
        // The ceiling stays extent-relative.
        assert_eq!(
            declared_span_with_extent_us(&segment_stamps, capture_s(5_000_000_000), 33_000),
            33_000 + MAX_BOUNDARY_DELTA_US
        );
    }

    #[test]
    fn batch_content_extent_stacks_spans_and_last_chunk_extent() {
        let capture_s = |us: i64| us as f64 / 1e6;
        // Batch of three with a floored middle boundary: chunk B's announced
        // start sits inside chunk A's content, so its duration line floors to
        // 33367 and the extent stacks both lines plus the last replayed extent.
        let chunk_a: Vec<f64> = [0, 16_683, 33_366]
            .iter()
            .map(|us| capture_s(*us))
            .collect();
        let chunk_b: Vec<f64> = [20_000, 36_683].iter().map(|us| capture_s(*us)).collect();
        let chunk_c: Vec<f64> = [40_000, 56_683].iter().map(|us| capture_s(*us)).collect();
        let span_a_us = declared_batch_span_us(&chunk_a, chunk_b[0]);
        let span_b_us = declared_batch_span_us(&chunk_b, chunk_c[0]);
        assert_eq!(span_a_us, 33_367, "the overlapped boundary floors");
        assert_eq!(span_b_us, 20_000, "the healthy boundary passes through");
        assert_eq!(
            batch_content_extent_us(&[span_a_us, span_b_us], &chunk_c),
            33_367 + 20_000 + 16_683
        );

        // A batch of one carries no duration lines: the extent is the
        // chunk's own replayed extent.
        assert_eq!(batch_content_extent_us(&[], &[0.0, 0.016683]), 16_683);

        // A last chunk with a regressing stamp: the replay steps by the
        // writer's step ceiling, so the extent cannot undershoot.
        assert_eq!(
            batch_content_extent_us(&[17_000], &[0.017, 0.0165]),
            17_000 + 100_000
        );
    }

    #[test]
    fn duration_directive_formats_exact_decimal_seconds() {
        assert_eq!(duration_directive(16_683), "0.016683");
        assert_eq!(duration_directive(1_000_000), "1.000000");
        assert_eq!(duration_directive(0), "0.000000");
        assert_eq!(duration_directive(MAX_BOUNDARY_DELTA_US), "2147.483646");
    }

    #[test]
    fn batch_concat_list_emits_duration_lines_except_last() {
        let tempdir = TempDir::new().unwrap();
        let list = tempdir.path().join("list.txt");
        let inputs = vec![
            BatchNutInput {
                raw_nut: PathBuf::from("/data/trace/chunks/chunk_0000.nut"),
                span_to_next_us: Some(16_683),
                frame_count: 2,
                skip_frames: 0,
            },
            BatchNutInput {
                raw_nut: PathBuf::from("/data/trace/chunks/chunk_0001.nut"),
                span_to_next_us: Some(MAX_BOUNDARY_DELTA_US),
                frame_count: 2,
                skip_frames: 0,
            },
            BatchNutInput {
                raw_nut: PathBuf::from("/data/trace/chunks/chunk_0002.nut"),
                span_to_next_us: None,
                frame_count: 2,
                skip_frames: 0,
            },
        ];
        write_batch_concat_list(&list, &inputs).expect("write list");
        let contents = std::fs::read_to_string(&list).unwrap();
        assert_eq!(
            contents,
            "file '/data/trace/chunks/chunk_0000.nut'\n\
             duration 0.016683\n\
             file '/data/trace/chunks/chunk_0001.nut'\n\
             duration 2147.483646\n\
             file '/data/trace/chunks/chunk_0002.nut'\n"
        );
    }

    /// Run the batch PTS gate for one fixture, whose `chunk_capture_us` holds
    /// each chunk's frame times as batch-absolute microseconds. Every output
    /// of both codec branches must decode to the batch-relative capture
    /// ladder exactly, frame-complete, monotonic, at the pinned timescale.
    async fn assert_batch_pts_gate(ffprobe: &Path, chunk_capture_us: &[Vec<i64>]) {
        // Mirror the production data flow: the announcement carries per-frame
        // `timestamp_s` seconds; the spooled PTS and the batch spans both
        // derive from them with the writer's truncation.
        let capture_s = |us: i64| us as f64 / 1e6;
        for codec in [
            LossyVideoCodec::LosslessPlusPreview,
            LossyVideoCodec::H264MediumLossyOnly,
        ] {
            let tempdir = TempDir::new().unwrap();
            let mut inputs = Vec::new();
            for (index, chunk) in chunk_capture_us.iter().enumerate() {
                let chunk_origin_us = capture_timestamp_us(capture_s(chunk[0]));
                let relative_pts: Vec<i64> = chunk
                    .iter()
                    .map(|us| capture_timestamp_us(capture_s(*us)) - chunk_origin_us)
                    .collect();
                let raw_nut = tempdir.path().join(format!("chunk_{index:04}.nut"));
                write_nut_chunk(&raw_nut, &relative_pts);
                // Spans go through the same helper the worker uses, so a
                // regression in the production span computation fails the
                // gate.
                let chunk_timestamps_s: Vec<f64> = chunk.iter().map(|us| capture_s(*us)).collect();
                inputs.push(BatchNutInput {
                    raw_nut,
                    span_to_next_us: chunk_capture_us.get(index + 1).map(|next| {
                        declared_batch_span_us(&chunk_timestamps_s, capture_s(next[0]))
                    }),
                    frame_count: chunk.len() as u32,
                    skip_frames: 0,
                });
            }
            let lossy_out = tempdir.path().join("chunk_0000_lossy.mp4");
            let lossless_out = tempdir.path().join("chunk_0000_lossless.mp4");
            let request = BatchEncodeRequest {
                inputs,
                lossy_out: lossy_out.clone(),
                lossless_out: lossless_out.clone(),
                codec,
            };
            VideoEncoder::new()
                .encode_chunk_batch(&request, ENCODE_THREADS_PER_OUTPUT)
                .await
                .expect("batched transcode");

            let batch_origin_us = capture_timestamp_us(capture_s(chunk_capture_us[0][0]));
            let expected: Vec<i64> = chunk_capture_us
                .iter()
                .flatten()
                .map(|us| capture_timestamp_us(capture_s(*us)) - batch_origin_us)
                .collect();
            let mut outputs = vec![lossy_out];
            if !codec.is_lossy_only() {
                outputs.push(lossless_out);
            }
            for video in &outputs {
                assert_merged_video_is_sound(ffprobe, video);
                assert_eq!(
                    decoded_frame_pts(ffprobe, video),
                    expected,
                    "{} ({codec:?}) must decode to the batch-relative capture ladder",
                    video.display()
                );
            }
        }
    }

    #[tokio::test]
    async fn batch_pts_gate_gapped_chunks() {
        let (Some(_ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping batch PTS gate.");
            return;
        };
        // A real capture gap: chunk A at 0/66/133 ms, chunk B at 1000 ms. The
        // gap must survive the batch instead of collapsing to a frame interval.
        assert_batch_pts_gate(&ffprobe, &[vec![0, 66_000, 133_000], vec![1_000_000]]).await;
    }

    #[tokio::test]
    async fn batch_pts_gate_jittered_chunks() {
        let (Some(_ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping batch PTS gate.");
            return;
        };
        // Irregular deltas cycling ~15.4-17.9 ms, continuing across the chunk
        // boundary like real capture jitter.
        let deltas = [15_400i64, 16_250, 17_100, 17_900];
        let mut capture_us = vec![0i64];
        for index in 0..23usize {
            capture_us.push(capture_us[index] + deltas[index % deltas.len()]);
        }
        let (chunk_a, chunk_b) = capture_us.split_at(12);
        assert_batch_pts_gate(&ffprobe, &[chunk_a.to_vec(), chunk_b.to_vec()]).await;
    }

    #[tokio::test]
    async fn batch_pts_gate_contiguous_chunks() {
        let (Some(_ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping batch PTS gate.");
            return;
        };
        // 3 x 48 frames on a metronome 16683 us cadence with no inter-chunk gap.
        let chunks: Vec<Vec<i64>> = (0..3i64)
            .map(|chunk| {
                (0..48i64)
                    .map(|frame| (chunk * 48 + frame) * 16_683)
                    .collect()
            })
            .collect();
        assert_batch_pts_gate(&ffprobe, &chunks).await;
    }

    #[tokio::test]
    async fn batch_of_one_matches_single_chunk_invocation() {
        if locate_binary("ffmpeg").is_none() {
            eprintln!("ffmpeg not on PATH — skipping batch-of-one test.");
            return;
        }
        for codec in [
            LossyVideoCodec::LosslessPlusPreview,
            LossyVideoCodec::H264MediumLossyOnly,
        ] {
            let tempdir = TempDir::new().unwrap();
            let raw = tempdir.path().join("chunk_0000.nut");
            write_nut_chunk(&raw, &[0, 16_683, 33_366, 50_049]);

            let encoder = VideoEncoder::new();
            let single_lossy = tempdir.path().join("single_lossy.mp4");
            let single_lossless = tempdir.path().join("single_lossless.mp4");
            encoder
                .encode_chunk(
                    &ChunkEncodeRequest {
                        raw_nut: raw.clone(),
                        lossy_out: single_lossy.clone(),
                        lossless_out: single_lossless.clone(),
                        codec,
                        frame_count: 4,
                        skip_frames: 0,
                    },
                    ENCODE_THREADS_PER_OUTPUT,
                )
                .await
                .expect("single-chunk transcode");

            let batch_lossy = tempdir.path().join("batch_lossy.mp4");
            let batch_lossless = tempdir.path().join("batch_lossless.mp4");
            let outcome = encoder
                .encode_chunk_batch(
                    &BatchEncodeRequest {
                        inputs: vec![BatchNutInput {
                            raw_nut: raw.clone(),
                            span_to_next_us: None,
                            frame_count: 4,
                            skip_frames: 0,
                        }],
                        lossy_out: batch_lossy.clone(),
                        lossless_out: batch_lossless.clone(),
                        codec,
                    },
                    ENCODE_THREADS_PER_OUTPUT,
                )
                .await
                .expect("batch-of-one transcode");

            // The degenerate batch delegates to the single-chunk invocation,
            // so its outputs are byte-identical to today's.
            assert_eq!(
                std::fs::read(&single_lossy).unwrap(),
                std::fs::read(&batch_lossy).unwrap(),
                "batch-of-one lossy must match the single-chunk encode ({codec:?})"
            );
            if codec.is_lossy_only() {
                assert_eq!(outcome.lossless_bytes, 0);
                assert!(!batch_lossless.exists(), "no lossless output ({codec:?})");
            } else {
                assert_eq!(
                    std::fs::read(&single_lossless).unwrap(),
                    std::fs::read(&batch_lossless).unwrap(),
                    "batch-of-one lossless must match the single-chunk encode"
                );
            }
            assert!(
                !list_file_for(&batch_lossy).exists(),
                "a degenerate batch writes no concat list"
            );
        }
    }

    #[tokio::test]
    async fn lossy_only_batch_produces_single_lossy_segment() {
        let (Some(_ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping lossy-only batch test.");
            return;
        };
        let tempdir = TempDir::new().unwrap();
        let mut inputs = Vec::new();
        for index in 0..3i64 {
            let raw_nut = tempdir.path().join(format!("chunk_{index:04}.nut"));
            write_nut_chunk(&raw_nut, &[0, 16_683]);
            inputs.push(BatchNutInput {
                raw_nut,
                span_to_next_us: (index < 2).then_some(33_366),
                frame_count: 2,
                skip_frames: 0,
            });
        }
        let lossy_out = tempdir.path().join("chunk_0000_lossy.mp4");
        let lossless_out = tempdir.path().join("chunk_0000_lossless.mp4");
        let outcome = VideoEncoder::new()
            .encode_chunk_batch(
                &BatchEncodeRequest {
                    inputs,
                    lossy_out: lossy_out.clone(),
                    lossless_out: lossless_out.clone(),
                    codec: LossyVideoCodec::H264MediumLossyOnly,
                },
                ENCODE_THREADS_PER_OUTPUT,
            )
            .await
            .expect("lossy-only batched transcode");

        assert!(outcome.lossy_bytes > 0);
        assert_eq!(outcome.lossless_bytes, 0);
        assert!(
            !lossless_out.exists(),
            "no lossless segment in lossy-only mode"
        );
        assert_eq!(
            decoded_frame_pts(&ffprobe, &lossy_out).len(),
            6,
            "the single lossy segment must carry every batched frame"
        );
    }

    #[tokio::test]
    async fn batch_frame_count_encodes_only_the_frames_the_recording_owns() {
        // The dispatcher can only cut the trace's last chunk, so the batch cap
        // drops frames from the tail of the concatenated stream: the cut
        // entry's own. Both outputs must stop there, or an mp4 outruns the
        // sidecar indexing it.
        let (Some(_), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe")) else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping batch frame-cap test.");
            return;
        };

        let tempdir = TempDir::new().unwrap();
        let chunk_a = tempdir.path().join("chunk_0000.nut");
        let chunk_b = tempdir.path().join("chunk_0001.nut");
        write_nut_chunk(&chunk_a, &[0, 16_683]);
        write_nut_chunk(&chunk_b, &[0, 16_683, 33_366]);
        let lossy_out = tempdir.path().join("chunk_0000_lossy.mp4");
        let lossless_out = tempdir.path().join("chunk_0000_lossless.mp4");

        VideoEncoder::new()
            .encode_chunk_batch(
                &BatchEncodeRequest {
                    inputs: vec![
                        BatchNutInput {
                            raw_nut: chunk_a,
                            span_to_next_us: Some(33_366),
                            frame_count: 2,
                            skip_frames: 0,
                        },
                        // Cut at the recording boundary: only its first frame
                        // belongs to this recording.
                        BatchNutInput {
                            raw_nut: chunk_b,
                            span_to_next_us: None,
                            frame_count: 1,
                            skip_frames: 0,
                        },
                    ],
                    lossy_out: lossy_out.clone(),
                    lossless_out: lossless_out.clone(),
                    codec: LossyVideoCodec::LosslessPlusPreview,
                },
                ENCODE_THREADS_PER_OUTPUT,
            )
            .await
            .expect("batched transcode");

        assert_eq!(
            decoded_frame_pts(&ffprobe, &lossy_out).len(),
            3,
            "the lossy output must stop at the batch's owned frame count"
        );
        assert_eq!(
            decoded_frame_pts(&ffprobe, &lossless_out).len(),
            3,
            "the lossless output must stop at the same cut"
        );
    }

    #[tokio::test]
    async fn batch_with_corrupt_nut_entry_fails_instead_of_truncating() {
        if locate_binary("ffmpeg").is_none() {
            eprintln!("ffmpeg not on PATH — skipping corrupt batch entry test.");
            return;
        }
        // The concat demuxer treats an entry it cannot open as end of stream and
        // ffmpeg exits 0, so without the header check the batch would
        // silently lose this chunk and every later one.
        let tempdir = TempDir::new().unwrap();
        let good_nut = tempdir.path().join("chunk_0000.nut");
        write_nut_chunk(&good_nut, &[0, 16_683]);
        let corrupt_nut = tempdir.path().join("chunk_0001.nut");
        std::fs::write(&corrupt_nut, b"not a nut container").unwrap();

        let lossy_out = tempdir.path().join("chunk_0000_lossy.mp4");
        let error = VideoEncoder::new()
            .encode_chunk_batch(
                &BatchEncodeRequest {
                    inputs: vec![
                        BatchNutInput {
                            raw_nut: good_nut,
                            span_to_next_us: Some(33_366),
                            frame_count: 2,
                            skip_frames: 0,
                        },
                        BatchNutInput {
                            raw_nut: corrupt_nut.clone(),
                            span_to_next_us: None,
                            frame_count: 2,
                            skip_frames: 0,
                        },
                    ],
                    lossy_out: lossy_out.clone(),
                    lossless_out: tempdir.path().join("chunk_0000_lossless.mp4"),
                    codec: LossyVideoCodec::LosslessPlusPreview,
                },
                ENCODE_THREADS_PER_OUTPUT,
            )
            .await
            .expect_err("a corrupt entry must fail the batch");
        assert!(
            matches!(error, VideoEncodeError::InvalidNutInput { ref path } if *path == corrupt_nut),
            "unexpected error variant: {error:?}"
        );
        assert!(
            !lossy_out.exists(),
            "no output before the batch is validated"
        );
    }

    #[tokio::test]
    async fn batch_span_clamp_stays_monotonic_and_frame_complete() {
        let (Some(_ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping span clamp test.");
            return;
        };
        // A 4000 s gap sits above the mp4 boundary-delta ceiling, so the
        // span clamps to the extent plus the largest safe step. The exact
        // ladder is unattainable; the output must stay monotonic and carry
        // every frame.
        let chunk_a_timestamps = [0.0, 0.016683];
        let span_us = declared_batch_span_us(&chunk_a_timestamps, 4_000.0);
        assert_eq!(
            span_us,
            16_683 + MAX_BOUNDARY_DELTA_US,
            "the fabricated gap must clamp to the extent plus the ceiling"
        );

        let tempdir = TempDir::new().unwrap();
        let chunk_a = tempdir.path().join("chunk_0000.nut");
        let chunk_b = tempdir.path().join("chunk_0001.nut");
        write_nut_chunk(&chunk_a, &[0, 16_683]);
        write_nut_chunk(&chunk_b, &[0, 16_683]);
        let lossy_out = tempdir.path().join("chunk_0000_lossy.mp4");
        let lossless_out = tempdir.path().join("chunk_0000_lossless.mp4");
        VideoEncoder::new()
            .encode_chunk_batch(
                &BatchEncodeRequest {
                    inputs: vec![
                        BatchNutInput {
                            raw_nut: chunk_a,
                            span_to_next_us: Some(span_us),
                            frame_count: 2,
                            skip_frames: 0,
                        },
                        BatchNutInput {
                            raw_nut: chunk_b,
                            span_to_next_us: None,
                            frame_count: 2,
                            skip_frames: 0,
                        },
                    ],
                    lossy_out: lossy_out.clone(),
                    lossless_out: lossless_out.clone(),
                    codec: LossyVideoCodec::LosslessPlusPreview,
                },
                ENCODE_THREADS_PER_OUTPUT,
            )
            .await
            .expect("clamped batched transcode");

        for video in [&lossy_out, &lossless_out] {
            assert_merged_video_is_sound(&ffprobe, video);
            assert_eq!(
                decoded_frame_pts(&ffprobe, video).len(),
                4,
                "{} must keep every frame despite the clamped span",
                video.display()
            );
        }
    }

    #[tokio::test]
    async fn batch_overlap_boundary_stays_monotonic_and_frame_complete() {
        let (Some(_ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping overlap boundary test.");
            return;
        };
        // A backwards clock step at the boundary: chunk B's announced start
        // sits inside chunk A's 33366 us content. The floored span degrades
        // the boundary to a 1 us ramp, so the exact ladder is unattainable;
        // the output must stay monotonic and frame-complete, and chunk A's
        // frames must keep their exact capture PTS.
        let capture_s = |us: i64| us as f64 / 1e6;
        let chunk_a_pts_us = [0i64, 16_683, 33_366];
        let chunk_a_timestamps: Vec<f64> = chunk_a_pts_us.iter().map(|us| capture_s(*us)).collect();
        let span_us = declared_batch_span_us(&chunk_a_timestamps, capture_s(20_000));
        for codec in [
            LossyVideoCodec::LosslessPlusPreview,
            LossyVideoCodec::H264MediumLossyOnly,
        ] {
            let tempdir = TempDir::new().unwrap();
            let chunk_a = tempdir.path().join("chunk_0000.nut");
            let chunk_b = tempdir.path().join("chunk_0001.nut");
            write_nut_chunk(&chunk_a, &chunk_a_pts_us);
            write_nut_chunk(&chunk_b, &[0, 16_683]);
            let lossy_out = tempdir.path().join("chunk_0000_lossy.mp4");
            let lossless_out = tempdir.path().join("chunk_0000_lossless.mp4");
            VideoEncoder::new()
                .encode_chunk_batch(
                    &BatchEncodeRequest {
                        inputs: vec![
                            BatchNutInput {
                                raw_nut: chunk_a,
                                span_to_next_us: Some(span_us),
                                frame_count: 3,
                                skip_frames: 0,
                            },
                            BatchNutInput {
                                raw_nut: chunk_b,
                                span_to_next_us: None,
                                frame_count: 2,
                                skip_frames: 0,
                            },
                        ],
                        lossy_out: lossy_out.clone(),
                        lossless_out: lossless_out.clone(),
                        codec,
                    },
                    ENCODE_THREADS_PER_OUTPUT,
                )
                .await
                .expect("overlap batched transcode");

            let mut outputs = vec![lossy_out];
            if !codec.is_lossy_only() {
                outputs.push(lossless_out);
            }
            for video in &outputs {
                assert_merged_video_is_sound(&ffprobe, video);
                let pts_values = decoded_frame_pts(&ffprobe, video);
                assert_eq!(
                    pts_values.len(),
                    5,
                    "{} ({codec:?}) must keep every frame across the overlap",
                    video.display()
                );
                assert_eq!(
                    &pts_values[..3],
                    &chunk_a_pts_us,
                    "{} ({codec:?}) must keep exact capture PTS before the degraded boundary",
                    video.display()
                );
            }
        }
    }

    #[tokio::test]
    async fn batch_with_synthesized_pts_chunk_stays_monotonic_and_frame_complete() {
        let (Some(_ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping synthesized-PTS batch test.");
            return;
        };
        // A synthesized-PTS chunk: the announced stamps are all the same
        // 1.0 s duplicate, so the writer synthesized the monotonic NUT ladder
        // this test writes directly. The next chunk's announced start sits
        // inside that real content, so flooring only at the announced extent
        // would make the preset-medium encode store backwards PTS. The exact
        // ladder is degraded here; assert positive properties only.
        let chunk_a_announced = [1.0, 1.0, 1.0];
        let chunk_a_nut_pts = [0i64, 16_683, 33_366];
        let span_us = declared_batch_span_us(&chunk_a_announced, 1.005);
        let tempdir = TempDir::new().unwrap();
        let chunk_a = tempdir.path().join("chunk_0000.nut");
        let chunk_b = tempdir.path().join("chunk_0001.nut");
        write_nut_chunk(&chunk_a, &chunk_a_nut_pts);
        write_nut_chunk(&chunk_b, &[0, 16_683]);
        let lossy_out = tempdir.path().join("chunk_0000_lossy.mp4");
        VideoEncoder::new()
            .encode_chunk_batch(
                &BatchEncodeRequest {
                    inputs: vec![
                        BatchNutInput {
                            raw_nut: chunk_a,
                            span_to_next_us: Some(span_us),
                            frame_count: 3,
                            skip_frames: 0,
                        },
                        BatchNutInput {
                            raw_nut: chunk_b,
                            span_to_next_us: None,
                            frame_count: 2,
                            skip_frames: 0,
                        },
                    ],
                    lossy_out: lossy_out.clone(),
                    lossless_out: tempdir.path().join("chunk_0000_lossless.mp4"),
                    codec: LossyVideoCodec::H264MediumLossyOnly,
                },
                ENCODE_THREADS_PER_OUTPUT,
            )
            .await
            .expect("synthesized-PTS batched transcode");

        assert_merged_video_is_sound(&ffprobe, &lossy_out);
        let pts_values = decoded_frame_pts(&ffprobe, &lossy_out);
        assert_eq!(
            pts_values.len(),
            5,
            "every frame must survive the synthesized-PTS boundary"
        );
        assert!(
            pts_values.windows(2).all(|pair| pair[1] > pair[0]),
            "PTS must stay strictly monotonic across the synthesized-PTS boundary: {pts_values:?}"
        );
    }

    /// Encode `chunk_capture_us` into finalise-ready segments for one codec
    /// branch, grouping consecutive chunks per `segment_chunk_counts` entry
    /// (a count of one is a per-chunk encode). Returns the lossy and lossless
    /// segment paths (the latter empty in lossy-only mode), each segment's
    /// announced stamps, and the content extents the worker would carry.
    async fn encode_finalise_segments(
        tempdir: &TempDir,
        chunk_capture_us: &[Vec<i64>],
        segment_chunk_counts: &[usize],
        codec: LossyVideoCodec,
    ) -> (Vec<PathBuf>, Vec<PathBuf>, Vec<Vec<f64>>, Vec<i64>) {
        assert_eq!(
            segment_chunk_counts.iter().sum::<usize>(),
            chunk_capture_us.len(),
            "fixture groups must cover every chunk"
        );
        let capture_s = |us: i64| us as f64 / 1e6;
        let encoder = VideoEncoder::new();
        let mut lossy_segments = Vec::new();
        let mut lossless_segments = Vec::new();
        let mut segment_timestamps_s: Vec<Vec<f64>> = Vec::new();
        let mut segment_extents_us: Vec<i64> = Vec::new();
        let mut chunk_cursor = 0usize;
        for (segment_index, &chunk_count) in segment_chunk_counts.iter().enumerate() {
            let group = &chunk_capture_us[chunk_cursor..chunk_cursor + chunk_count];
            chunk_cursor += chunk_count;
            let mut inputs = Vec::new();
            let mut group_spans_us: Vec<i64> = Vec::new();
            for (offset, chunk) in group.iter().enumerate() {
                // The spooled NUT re-anchors each chunk at its first frame;
                // the announcement keeps the absolute stamps.
                let chunk_origin_us = capture_timestamp_us(capture_s(chunk[0]));
                let relative_pts: Vec<i64> = chunk
                    .iter()
                    .map(|us| capture_timestamp_us(capture_s(*us)) - chunk_origin_us)
                    .collect();
                let raw_nut = tempdir
                    .path()
                    .join(format!("chunk_{segment_index:04}_{offset}.nut"));
                write_nut_chunk(&raw_nut, &relative_pts);
                let chunk_timestamps_s: Vec<f64> = chunk.iter().map(|us| capture_s(*us)).collect();
                let span_to_next_us = group
                    .get(offset + 1)
                    .map(|next| declared_batch_span_us(&chunk_timestamps_s, capture_s(next[0])));
                if let Some(span_us) = span_to_next_us {
                    group_spans_us.push(span_us);
                }
                inputs.push(BatchNutInput {
                    raw_nut,
                    span_to_next_us,
                    frame_count: chunk.len() as u32,
                    skip_frames: 0,
                });
            }
            let lossy_out = tempdir
                .path()
                .join(format!("chunk_{segment_index:04}_lossy.mp4"));
            let lossless_out = tempdir
                .path()
                .join(format!("chunk_{segment_index:04}_lossless.mp4"));
            encoder
                .encode_chunk_batch(
                    &BatchEncodeRequest {
                        inputs,
                        lossy_out: lossy_out.clone(),
                        lossless_out: lossless_out.clone(),
                        codec,
                    },
                    ENCODE_THREADS_PER_OUTPUT,
                )
                .await
                .expect("segment transcode");
            lossy_segments.push(lossy_out);
            if !codec.is_lossy_only() {
                lossless_segments.push(lossless_out);
            }
            segment_timestamps_s.push(group.iter().flatten().map(|us| capture_s(*us)).collect());
            let last_chunk_timestamps_s: Vec<f64> = group
                .last()
                .expect("group covers at least one chunk")
                .iter()
                .map(|us| capture_s(*us))
                .collect();
            segment_extents_us.push(batch_content_extent_us(
                &group_spans_us,
                &last_chunk_timestamps_s,
            ));
        }
        (
            lossy_segments,
            lossless_segments,
            segment_timestamps_s,
            segment_extents_us,
        )
    }

    /// Compute the finalise `duration` spans the way the trace actor does:
    /// each segment's raw first-to-first capture span, floored and capped on
    /// the carried content extent.
    fn finalise_spans_us(
        segment_timestamps_s: &[Vec<f64>],
        segment_extents_us: &[i64],
    ) -> Vec<i64> {
        segment_timestamps_s
            .windows(2)
            .zip(segment_extents_us)
            .map(|(pair, &extent_us)| declared_span_with_extent_us(&pair[0], pair[1][0], extent_us))
            .collect()
    }

    /// Run the finalise PTS gate for one fixture: every final video of both
    /// codec branches must decode to the trace-relative capture ladder
    /// exactly, frame-complete, monotonic, at the pinned timescale.
    async fn assert_finalise_pts_gate(
        ffprobe: &Path,
        chunk_capture_us: &[Vec<i64>],
        segment_chunk_counts: &[usize],
    ) {
        let capture_s = |us: i64| us as f64 / 1e6;
        for codec in [
            LossyVideoCodec::LosslessPlusPreview,
            LossyVideoCodec::H264MediumLossyOnly,
        ] {
            let tempdir = TempDir::new().unwrap();
            let (lossy_segments, lossless_segments, segment_timestamps_s, segment_extents_us) =
                encode_finalise_segments(&tempdir, chunk_capture_us, segment_chunk_counts, codec)
                    .await;
            let spans_to_next_us = finalise_spans_us(&segment_timestamps_s, &segment_extents_us);

            let encoder = VideoEncoder::new();
            let final_lossy = tempdir.path().join("lossy.mp4");
            encoder
                .concat_segments(&lossy_segments, &spans_to_next_us, &final_lossy)
                .await
                .expect("finalise concat");
            let mut outputs = vec![final_lossy];
            if !codec.is_lossy_only() {
                let final_lossless = tempdir.path().join("lossless.mp4");
                encoder
                    .concat_segments(&lossless_segments, &spans_to_next_us, &final_lossless)
                    .await
                    .expect("finalise concat");
                outputs.push(final_lossless);
            }

            let trace_origin_us = capture_timestamp_us(capture_s(chunk_capture_us[0][0]));
            let expected: Vec<i64> = chunk_capture_us
                .iter()
                .flatten()
                .map(|us| capture_timestamp_us(capture_s(*us)) - trace_origin_us)
                .collect();
            for video in &outputs {
                assert_merged_video_is_sound(ffprobe, video);
                assert_eq!(
                    decoded_frame_pts(ffprobe, video),
                    expected,
                    "{} ({codec:?}) must decode to the trace-relative capture ladder",
                    video.display()
                );
            }
        }
    }

    #[tokio::test]
    async fn finalise_pts_gate_gapped_chunks() {
        let (Some(_ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping finalise PTS gate.");
            return;
        };
        // A real capture gap: it must survive the finalise concat instead of
        // collapsing to a frame interval or drifting per boundary.
        let chunks = [vec![0, 66_000, 133_000], vec![1_000_000]];
        // Per-chunk segments, and both chunks batched into one segment so the
        // single-segment finalise path keeps the exact ladder too.
        assert_finalise_pts_gate(&ffprobe, &chunks, &[1, 1]).await;
        assert_finalise_pts_gate(&ffprobe, &chunks, &[2]).await;
    }

    #[tokio::test]
    async fn finalise_pts_gate_jittered_chunks() {
        let (Some(_ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping finalise PTS gate.");
            return;
        };
        // Irregular deltas cycling ~15.4-17.9 ms, continuing across the
        // segment boundary like real capture jitter.
        let deltas = [15_400i64, 16_250, 17_100, 17_900];
        let mut capture_us = vec![0i64];
        for index in 0..23usize {
            capture_us.push(capture_us[index] + deltas[index % deltas.len()]);
        }
        let (chunk_a, chunk_b) = capture_us.split_at(12);
        assert_finalise_pts_gate(&ffprobe, &[chunk_a.to_vec(), chunk_b.to_vec()], &[1, 1]).await;
    }

    #[tokio::test]
    async fn finalise_pts_gate_contiguous_chunks() {
        let (Some(_ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping finalise PTS gate.");
            return;
        };
        // 3 x 48 frames on a metronome 16683 us cadence with no gap between
        // segments.
        let chunks: Vec<Vec<i64>> = (0..3i64)
            .map(|chunk| {
                (0..48i64)
                    .map(|frame| (chunk * 48 + frame) * 16_683)
                    .collect()
            })
            .collect();
        assert_finalise_pts_gate(&ffprobe, &chunks, &[1, 1, 1]).await;
    }

    #[tokio::test]
    async fn finalise_pts_gate_composed_batches() {
        let (Some(_ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping finalise PTS gate.");
            return;
        };
        // 6 chunks of 4 frames at a 16683 us cadence with mixed inter-chunk
        // gaps from ~16 ms to 1.5 s, grouped into batched segments of 3 and
        // 2 chunks plus a single-chunk segment: PR 2's batch-relative
        // exactness and the finalise duration lines must compose to a
        // trace-relative exact final.
        let gaps_to_next_us = [16_683i64, 250_000, 1_500_000, 33_000, 700_000, 0];
        let mut chunks: Vec<Vec<i64>> = Vec::new();
        let mut start_us = 0i64;
        for gap_to_next_us in gaps_to_next_us {
            let chunk: Vec<i64> = (0..4i64).map(|frame| start_us + frame * 16_683).collect();
            start_us = chunk[3] + gap_to_next_us;
            chunks.push(chunk);
        }
        assert_finalise_pts_gate(&ffprobe, &chunks, &[3, 2, 1]).await;
    }

    #[tokio::test]
    async fn finalise_large_gap_keeps_exact_pts() {
        let (Some(_ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping finalise large-gap test.");
            return;
        };
        // A 600 s capture gap at a segment boundary. The batch encode path
        // caps its spans well below this scale, but the finalise concat is
        // stream-copy and must place the next segment exactly even for a
        // gap of several hundred seconds, for both codec branches.
        let chunks = [
            vec![0, 16_683, 33_366],
            vec![600_000_000, 600_016_683, 600_033_366],
        ];
        assert_finalise_pts_gate(&ffprobe, &chunks, &[1, 1]).await;
    }

    #[tokio::test]
    async fn finalise_span_clamp_stays_monotonic_and_frame_complete() {
        let (Some(_ffmpeg), Some(ffprobe)) = (locate_binary("ffmpeg"), locate_binary("ffprobe"))
        else {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping finalise clamp test.");
            return;
        };
        // A 4000 s gap between segments sits above the mp4 boundary-delta
        // ceiling, so the declared finalise span is clamped to the segment
        // extent plus the largest safe boundary step. The exact ladder is
        // unattainable by construction; the final must stay strictly
        // monotonic and carry every frame, for both codec branches, over
        // real batched segments.
        let chunks = [
            vec![0i64, 16_683],
            vec![33_366, 50_049],
            vec![4_000_000_000, 4_000_016_683],
            vec![4_000_033_366, 4_000_050_049],
        ];
        let segment_chunk_counts = [2usize, 2];
        for codec in [
            LossyVideoCodec::LosslessPlusPreview,
            LossyVideoCodec::H264MediumLossyOnly,
        ] {
            let tempdir = TempDir::new().unwrap();
            let (lossy_segments, lossless_segments, segment_timestamps_s, segment_extents_us) =
                encode_finalise_segments(&tempdir, &chunks, &segment_chunk_counts, codec).await;
            let spans_to_next_us = finalise_spans_us(&segment_timestamps_s, &segment_extents_us);
            assert_eq!(
                spans_to_next_us,
                vec![50_049 + MAX_BOUNDARY_DELTA_US],
                "the fabricated gap must clamp to the segment extent plus the ceiling"
            );

            let encoder = VideoEncoder::new();
            let final_lossy = tempdir.path().join("lossy.mp4");
            encoder
                .concat_segments(&lossy_segments, &spans_to_next_us, &final_lossy)
                .await
                .expect("finalise concat");
            let mut outputs = vec![final_lossy];
            if !codec.is_lossy_only() {
                let final_lossless = tempdir.path().join("lossless.mp4");
                encoder
                    .concat_segments(&lossless_segments, &spans_to_next_us, &final_lossless)
                    .await
                    .expect("finalise concat");
                outputs.push(final_lossless);
            }
            for video in &outputs {
                assert_merged_video_is_sound(&ffprobe, video);
                assert_eq!(
                    decoded_frame_pts(&ffprobe, video).len(),
                    8,
                    "{} ({codec:?}) must keep every frame despite the clamped span",
                    video.display()
                );
            }
        }
    }
}
