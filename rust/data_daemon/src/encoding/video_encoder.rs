//! Per-chunk `ffmpeg` transcoder and segment concatenator.
//!
//! The producer spools video frames into a sequence of NUT chunk files
//! beneath each trace's `chunks/` directory. As each chunk arrives the
//! per-trace actor calls [`VideoEncoder::encode_chunk`] which shells out to
//! ffmpeg to produce two MP4 segments:
//!
//! - `chunk_NNNN_lossy.mp4` — `libx264` `-pix_fmt yuv420p -preset ultrafast
//!   -qp 23` for fast playback.
//! - `chunk_NNNN_lossless.mp4` — `libx264` `-pix_fmt yuv444p10le -preset
//!   ultrafast -qp 0` for mathematically-lossless archival. `ffv1` would be
//!   the natural codec for this but is incompatible with the `.mp4`
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

/// Inputs to one per-chunk transcode invocation.
#[derive(Debug, Clone)]
pub struct ChunkEncodeRequest {
    /// Source NUT chunk file produced by the producer.
    pub raw_nut: PathBuf,
    /// Destination for the per-chunk lossy mp4 segment.
    pub lossy_out: PathBuf,
    /// Destination for the per-chunk lossless mp4 segment.
    pub lossless_out: PathBuf,
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
         encoding — note `-fps_mode passthrough` is ffmpeg >= 5.1 only) and the libx264 \
         encoder. Install a compatible ffmpeg (>= 4.0 with libx264). ffmpeg reported:\n{stderr_tail}",
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

    /// Encode one synthetic frame with `-vsync passthrough` + libx264 to the
    /// null muxer; a non-zero exit means the local ffmpeg lacks a capability
    /// the encoder needs.
    fn probe_passthrough_encode(&self, version: &str) -> Result<(), FfmpegPreflightError> {
        // One 16x16 yuv420p frame (a 16x16 plane plus two 8x8 planes = 384
        // bytes) fed via the rawvideo demuxer on stdin — no lavfi/input-file
        // dependency, so the probe works even on a minimal build. ffmpeg parses
        // (and would reject) the options before reading stdin, so an unsupported
        // `-vsync passthrough` fails immediately rather than on a healthy input.
        const PROBE_FRAME_LEN: usize = 16 * 16 * 3 / 2;
        let frame = vec![128u8; PROBE_FRAME_LEN];

        let mut child = std::process::Command::new(&self.binary)
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
            .arg("-vsync")
            .arg("passthrough")
            .arg("-c:v")
            .arg("libx264")
            .arg("-pix_fmt")
            .arg("yuv420p")
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
            })?;

        // The frame is far smaller than a pipe buffer, so writing then dropping
        // stdin cannot deadlock against ffmpeg's reads.
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(&frame);
        }

        let output = child
            .wait_with_output()
            .map_err(|source| FfmpegPreflightError::NotFound {
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
    /// The source `raw.nut` is left in place — the caller is responsible for
    /// unlinking it after verifying both outputs landed (the per-trace actor
    /// drops the source as part of its envelope handling so a partial encode
    /// can be retried via the recovery sweep without needing to re-spool).
    pub async fn encode_chunk(
        &self,
        request: &ChunkEncodeRequest,
    ) -> Result<ChunkEncodeOutcome, VideoEncodeError> {
        ensure_parent_dirs(&request.lossy_out)?;
        ensure_parent_dirs(&request.lossless_out)?;

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
        // on both (only deprecated, not removed, on 5.1+). Two `-map 0:v -c:v ...`
        // blocks emit both outputs from a single demux pass.
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
            .arg("passthrough")
            .arg("-c:v")
            .arg("libx264")
            .arg("-pix_fmt")
            .arg("yuv420p")
            .arg("-preset")
            .arg("ultrafast")
            .arg("-qp")
            .arg("23")
            .arg(&request.lossy_out)
            .arg("-map")
            .arg("0:v")
            .arg("-vsync")
            .arg("passthrough")
            .arg("-c:v")
            .arg("libx264")
            .arg("-pix_fmt")
            .arg("yuv444p10le")
            .arg("-preset")
            .arg("ultrafast")
            .arg("-qp")
            .arg("0")
            .arg(&request.lossless_out)
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
        // `exec`; `setpriority` is async-signal-safe and touches no parent
        // state. A failed renice is non-fatal (ignored), so the encode still
        // runs at default priority.
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
        let lossless_bytes = non_empty_file_size(&request.lossless_out)?;

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
            .arg(format!("testsrc=duration={duration}:size=16x16:rate=1"))
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
        };
        let outcome = encoder.encode_chunk(&request).await.expect("transcode");

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
            assert_eq!(streams[0]["width"], 16);
            assert_eq!(streams[0]["height"], 16);
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
                .encode_chunk(&ChunkEncodeRequest {
                    raw_nut: raw,
                    lossy_out: lossy.clone(),
                    lossless_out: lossless,
                })
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
        };
        let encoder = VideoEncoder::new();
        let error = encoder
            .encode_chunk(&request)
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
        };
        let encoder =
            VideoEncoder::new().with_binary("this-binary-definitely-does-not-exist-ffmpeg");
        let error = encoder
            .encode_chunk(&request)
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
