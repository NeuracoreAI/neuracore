//! Per-chunk `ffmpeg` transcoder and segment concatenator.
//!
//! The producer spools video frames into a sequence of NUT chunk files
//! beneath each trace's `chunks/` directory. As each chunk arrives the
//! per-trace actor calls [`VideoEncoder::encode_chunk`] which shells out to
//! ffmpeg to produce two MP4 segments:
//!
//! - `chunk_NNNN_lossy.mp4` — `libx264` `-pix_fmt yuv420p -preset ultrafast
//!   -qp 23` for fast playback, downscaled to a preview resolution (see
//!   [`LOSSY_PREVIEW_MAX_HEIGHT`]) since it is only a derivable proxy and the
//!   full-resolution encode is the transcoder's dominant cost.
//! - `chunk_NNNN_lossless.mp4` — `libx264rgb` `-pix_fmt rgb24 -preset
//!   ultrafast -qp 0` for mathematically-lossless archival. Encoding the
//!   captured rgb24 frames directly (rather than converting to a YUV format)
//!   keeps the output bit-exact to the captured pixels, encodes ~2.5× faster
//!   than a `yuv444p10le` pass, and matches the Python reference encoder.
//!   `ffv1` would also be lossless but is incompatible with the `.mp4`
//!   container the on-disk layout contract requires.
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

use data_daemon_shared::service_name::VIDEO_SPOOL_TICKS_PER_SECOND;

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
}

/// Inputs to one per-chunk transcode invocation.
#[derive(Debug, Clone)]
pub struct ChunkEncodeRequest {
    /// Source NUT chunk file produced by the producer.
    pub raw_nut: PathBuf,
    /// Destination for the per-chunk lossy mp4 segment.
    pub lossy_out: PathBuf,
    /// Destination for the per-chunk lossless mp4 segment. Unused in lossy-only
    /// mode (no lossless output is produced).
    pub lossless_out: PathBuf,
    /// Lossy codec selection for this trace. Controls whether a lossless
    /// archive is produced and how the lossy output is encoded.
    pub codec: LossyVideoCodec,
}

/// Outcome of a successful per-chunk transcode.
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
    /// `-vsync passthrough` frame-timing mode or the libx264 encoder.
    #[error(
        "ffmpeg at `{}` (version {version}) is incompatible: a required capability was \
         rejected. The daemon needs `-vsync passthrough` (drop-free, frame-accurate \
         encoding — note `-fps_mode passthrough` is ffmpeg >= 5.1 only) and the libx264 / \
         libx264rgb encoders. Install a compatible ffmpeg (>= 4.0 with libx264). ffmpeg \
         reported:\n{stderr_tail}",
        binary.to_string_lossy()
    )]
    Incompatible {
        /// Binary that was probed.
        binary: OsString,
        /// Detected ffmpeg version, or `"unknown"`.
        version: String,
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

    /// Verify the configured ffmpeg is present and supports the capabilities
    /// [`encode_chunk`](Self::encode_chunk) depends on, returning the detected
    /// version string on success.
    ///
    /// Run once at daemon startup so an incompatible install fails fast with a
    /// clear message instead of silently marking every video trace `failed` at
    /// recording time. Two steps: `ffmpeg -version` confirms the binary runs
    /// (and yields a version for diagnostics), then a one-frame synthetic
    /// encode to the null muxer exercises the exact `-vsync passthrough` knob —
    /// the option ffmpeg < 5.1 rejects when spelled `-fps_mode` — together with
    /// the libx264 encoder.
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
    /// pass. A non-zero exit means the local ffmpeg lacks a capability the
    /// encoder needs. The lossless `libx264rgb` path is the one that actually
    /// varies between builds, so probing only the lossy pass (as before) let the
    /// "fail fast at startup" check pass while every real lossless encode failed
    /// at recording time.
    fn probe_passthrough_encode(&self, version: &str) -> Result<(), FfmpegPreflightError> {
        // One 16x16 yuv420p frame (a 16x16 plane plus two 8x8 planes = 384
        // bytes) fed via the rawvideo demuxer on stdin — no lavfi/input-file
        // dependency, so the probe works even on a minimal build. ffmpeg parses
        // (and would reject) the options before reading stdin, so an unsupported
        // `-vsync passthrough`, `-enc_time_base` or `libx264rgb` encode fails
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
            .arg("-vsync")
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
            .arg("-vsync")
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
    /// unlinking it after verifying both outputs landed (the per-trace actor
    /// drops the source as part of its envelope handling so a partial encode
    /// can be retried via the recovery sweep without needing to re-spool).
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
        // truncated mid-frame. `-vsync passthrough` (applied per output) is
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
        // We spell this `-vsync passthrough` rather than the newer
        // `-fps_mode passthrough`: the two select the identical passthrough
        // mode, but `-fps_mode` is unrecognised by ffmpeg < 5.1 (e.g. the 4.4
        // build shipped on Ubuntu 22.04 / the integration host), where it aborts
        // the encode with "Unrecognized option 'fps_mode'". `-vsync` is accepted
        // on both (only deprecated, not removed, on 5.1+). The default branch
        // emits two `-map 0:v -c:v ...` output blocks from a single demux pass;
        // lossy-only emits a single block (no preview/archive split).
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
        let encode_threads = encode_threads.to_string();
        let enc_time_base = format!("1:{VIDEO_SPOOL_TICKS_PER_SECOND}");
        let track_timescale = VIDEO_SPOOL_TICKS_PER_SECOND.to_string();
        let lossy_only = request.codec.is_lossy_only();
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
            .arg(&request.raw_nut)
            .arg("-map")
            .arg("0:v")
            .arg("-vsync")
            .arg("passthrough");
        if lossy_only {
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
                .arg(&track_timescale)
                .arg(&request.lossy_out);
        } else {
            // Downscale the lossy preview proxy (only) to keep this dominant
            // pass cheap at high resolution; the lossless output stays native.
            let preview_filter = preview_scale_filter(LOSSY_PREVIEW_MAX_HEIGHT);
            command
                // Lossy preview proxy only: cap to preview resolution (see
                // `preview_scale_filter`). vsync passthrough still emits every
                // input frame, so the lossy frame count matches the lossless
                // output and the per-frame timestamp sidecar.
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
                .arg(&track_timescale)
                .arg(&request.lossy_out)
                .arg("-map")
                .arg("0:v")
                .arg("-vsync")
                .arg("passthrough")
                .arg("-enc_time_base")
                .arg(&enc_time_base)
                // libx264rgb encodes the rgb24 frames directly: bit-exact to the
                // captured pixels, ~2.5× faster than a yuv444p10le pass, and the
                // format the Python reference encoder writes.
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
                .arg(&track_timescale)
                .arg(&request.lossless_out);
        }
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

        let lossy_bytes = non_empty_file_size(&request.lossy_out)?;
        // In lossy-only mode no lossless archive is produced, so there is no
        // file to size — report zero rather than erroring on a missing output.
        let lossless_bytes = if lossy_only {
            0
        } else {
            non_empty_file_size(&request.lossless_out)?
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
    /// bytes. Caller is responsible for unlinking the source segments after
    /// the concat succeeds.
    pub async fn concat_segments(
        &self,
        segments: &[PathBuf],
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
        write_concat_list(&list_path, segments)?;

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
fn preview_scale_filter(max_height: u32) -> String {
    format!("scale=trunc(iw*min(1\\,{max_height}/ih)/2)*2:trunc(ih*min(1\\,{max_height}/ih)/2)*2")
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
/// demuxer's own escape rule (`'` → `'\''`).
///
/// Relative segment paths are resolved against the current working directory
/// before being written. ffmpeg's concat demuxer interprets `file '...'`
/// entries *relative to the list-file's directory*, not the daemon's CWD —
/// so a relative segment path like `recordings/rec/cam/trace/chunk_0000.mp4`
/// listed in `recordings/rec/cam/trace/lossy.mp4.concat.txt` would expand to
/// `recordings/rec/cam/trace/recordings/rec/cam/trace/chunk_0000.mp4` and
/// fail to open. Absolutising on write side-steps that without forcing
/// callers to pre-canonicalise.
fn write_concat_list(path: &Path, segments: &[PathBuf]) -> Result<(), VideoEncodeError> {
    let mut file = std::fs::File::create(path).map_err(|source| VideoEncodeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    for segment in segments {
        let absolute = if segment.is_absolute() {
            segment.clone()
        } else {
            std::env::current_dir()
                .map_err(|source| VideoEncodeError::Io {
                    path: segment.clone(),
                    source,
                })?
                .join(segment)
        };
        let escaped = absolute.to_string_lossy().replace('\'', r"'\''");
        writeln!(file, "file '{escaped}'").map_err(|source| VideoEncodeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
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
            "scale=trunc(iw*min(1\\,480/ih)/2)*2:trunc(ih*min(1\\,480/ih)/2)*2"
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
        write_concat_list(&list, &segments).expect("write list");
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
        write_concat_list(&list, &segments).expect("write list");
        let contents = std::fs::read_to_string(&list).unwrap();
        let expected = cwd.join("rel/chunk_0000.mp4");
        assert!(
            contents.contains(&format!("file '{}'", expected.display())),
            "got: {contents}"
        );
    }

    #[test]
    fn concat_segments_rejects_empty_input() {
        let tempdir = TempDir::new().unwrap();
        let out = tempdir.path().join("out.mp4");
        // Sync wrapper so the test body isn't async for this trivial case.
        let result = futures_block(VideoEncoder::new().concat_segments(&[], &out));
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
        let write_chunk = |path: &Path, jittery: bool| {
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
            }
            writer.finish().expect("finish NUT");
        };

        let encoder = VideoEncoder::new();
        let mut segments = Vec::new();
        for (chunk_index, jittery) in [true, false, true].into_iter().enumerate() {
            let raw = tempdir.path().join(format!("chunk_{chunk_index:04}.nut"));
            let lossy = tempdir
                .path()
                .join(format!("chunk_{chunk_index:04}_lossy.mp4"));
            write_chunk(&raw, jittery);
            encoder
                .encode_chunk(
                    &ChunkEncodeRequest {
                        raw_nut: raw,
                        lossy_out: lossy.clone(),
                        lossless_out: tempdir
                            .path()
                            .join(format!("chunk_{chunk_index:04}_lossless.mp4")),
                        codec: LossyVideoCodec::LosslessPlusPreview,
                    },
                    ENCODE_THREADS_PER_OUTPUT,
                )
                .await
                .expect("transcode chunk");
            segments.push(lossy);
        }

        let final_lossy = tempdir.path().join("lossy.mp4");
        encoder
            .concat_segments(&segments, &final_lossy)
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
                    },
                    ENCODE_THREADS_PER_OUTPUT,
                )
                .await
                .expect("transcode chunk");
            segments.push(lossy);
        }

        let final_lossy = tempdir.path().join("lossy.mp4");
        let outcome = encoder
            .concat_segments(&segments, &final_lossy)
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
        let pts_values: Vec<i64> = stdout
            .lines()
            .filter_map(|line| {
                line.strip_prefix("pts=")
                    .or_else(|| line.strip_prefix("pkt_pts="))
            })
            .filter_map(|value| value.parse().ok())
            .collect();
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
}
