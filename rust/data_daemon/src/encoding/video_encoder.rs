//! Per-trace `ffmpeg` transcoder.
//!
//! Sub-phase 5d of the rewrite plan (see `docs/data-daemon-rewrite.md`). The
//! trace actor spools captured frames into a `raw.nut` file via
//! [`crate::encoding::nut_writer`]; once the trace ends, this module hands the
//! file to a supervised `ffmpeg` child that produces two mp4s:
//!
//! - `lossy.mp4` — `libx264` `-pix_fmt yuv420p -preset ultrafast -qp 23` for
//!   fast playback.
//! - `lossless.mp4` — `libx264` `-pix_fmt yuv444p10le -preset ultrafast -qp 0`
//!   for mathematically-lossless archival. The Python implementation in
//!   `recording_encoding_disk_manager/encoding/video_trace.py` uses the same
//!   settings; `ffv1` was originally pencilled into the rewrite plan but is
//!   incompatible with the `.mp4` container the on-disk layout contract
//!   requires.
//!
//! Both outputs are verified non-empty before the source `raw.nut` is unlinked;
//! a failed encode leaves the spool intact so the upload coordinator can either
//! retry or surface the partial state on the dashboard. We deliberately keep
//! the ffmpeg invocation in a single command (two `-map 0:v` output streams)
//! so that the raw file is demuxed exactly once.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use tokio::process::Command;

/// Default ffmpeg binary name. Tests override via [`VideoEncoder::with_binary`]
/// when they need to point at a specific build.
pub const DEFAULT_FFMPEG_BINARY: &str = "ffmpeg";

/// Inputs to one transcode invocation.
#[derive(Debug, Clone)]
pub struct VideoEncodeRequest {
    /// Source NUT spool produced by [`crate::encoding::nut_writer::NutWriter`].
    pub raw_nut: PathBuf,
    /// Destination for the H.264 lossy output (matches Python `lossy.mp4`).
    pub lossy_mp4: PathBuf,
    /// Destination for the FFV1 lossless output (matches Python `lossless.mp4`).
    pub lossless_mp4: PathBuf,
}

/// Outcome of a successful transcode.
#[derive(Debug, Clone, Copy)]
pub struct VideoEncodeOutcome {
    /// Bytes written to the lossy mp4.
    pub lossy_bytes: u64,
    /// Bytes written to the lossless mp4.
    pub lossless_bytes: u64,
    /// Whether the source `raw.nut` was unlinked after verification.
    pub raw_nut_removed: bool,
}

/// Errors raised by [`VideoEncoder::run`].
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
    /// An I/O operation around the encode (file metadata, unlink) failed.
    #[error("I/O failure during transcode for {path}: {source}")]
    Io {
        /// Path being inspected when the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Builder for one transcode invocation. Keeps the ffmpeg binary path
/// configurable so unit tests can shim in a wrapper script if needed.
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

    /// Transcode `request.raw_nut` into the configured mp4 outputs.
    ///
    /// On success the source `raw.nut` is unlinked. A failed encode leaves it
    /// in place — the caller decides whether to retry or escalate.
    pub async fn run(
        &self,
        request: &VideoEncodeRequest,
    ) -> Result<VideoEncodeOutcome, VideoEncodeError> {
        ensure_parent_dirs(&request.lossy_mp4)?;
        ensure_parent_dirs(&request.lossless_mp4)?;

        // `-y` overwrites existing outputs (resume safety: a previous failed
        // run may have left a partial mp4). `-fflags +genpts` rebuilds the
        // presentation timestamps from the NUT timing when the spool was
        // truncated mid-frame. `-vsync vfr` (applied per output) is the
        // critical knob here: the NUT spool uses `time_base = 1/1_000_000`
        // so ffmpeg's demuxer reports `r_frame_rate = 1_000_000/1` (one
        // million fps). With the default `vsync auto` cfr policy the encoder
        // would then synthesise an output frame at every microsecond slot
        // between consecutive input PTS values — for a 10 s clip that is
        // ~10 million duplicate output frames, and the encode effectively
        // never completes. `vfr` writes each input frame exactly once at its
        // original PTS, which is what real-time camera capture actually is.
        // Two `-map 0:v -c:v ...` blocks emit both outputs from a single
        // demux pass.
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
            .arg("vfr")
            .arg("-c:v")
            .arg("libx264")
            .arg("-pix_fmt")
            .arg("yuv420p")
            .arg("-preset")
            .arg("ultrafast")
            .arg("-qp")
            .arg("23")
            .arg(&request.lossy_mp4)
            .arg("-map")
            .arg("0:v")
            .arg("-vsync")
            .arg("vfr")
            .arg("-c:v")
            .arg("libx264")
            .arg("-pix_fmt")
            .arg("yuv444p10le")
            .arg("-preset")
            .arg("ultrafast")
            .arg("-qp")
            .arg("0")
            .arg(&request.lossless_mp4)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            // ffmpeg keeps file descriptors open across `fork`/`exec`; the
            // daemon's iceoryx2 sockets must NOT leak into the encoder, so we
            // rely on Tokio's default `cloexec` behaviour and additionally
            // request `kill_on_drop` to clean up if the supervising future is
            // cancelled mid-flight.
            .kill_on_drop(true);

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

        let lossy_bytes = non_empty_file_size(&request.lossy_mp4)?;
        let lossless_bytes = non_empty_file_size(&request.lossless_mp4)?;

        // Source spool can be removed only after both outputs are confirmed
        // non-empty. If the unlink itself fails we still report success on the
        // mp4s — leaving a stale `raw.nut` is preferable to discarding the
        // archived outputs the user is waiting on.
        let raw_nut_removed = match std::fs::remove_file(&request.raw_nut) {
            Ok(()) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => {
                tracing::warn!(
                    %error,
                    raw_nut = %request.raw_nut.display(),
                    "ffmpeg succeeded but raw spool could not be removed"
                );
                false
            }
        };

        Ok(VideoEncodeOutcome {
            lossy_bytes,
            lossless_bytes,
            raw_nut_removed,
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::nut_writer::{NutVideoConfig, NutWriter};
    use std::path::PathBuf;
    use std::process::Command as StdCommand;
    use tempfile::TempDir;

    /// Locate `ffmpeg` on `PATH`. Mirrors the `locate_ffprobe` helper in
    /// [`crate::encoding::nut_writer`] so the suite skips cleanly on sandboxes
    /// that lack the FFmpeg toolchain.
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

    /// Render a small synthetic NUT spool with the given frame count.
    fn write_synthetic_nut(path: &Path, frame_count: u64) -> NutVideoConfig {
        let config = NutVideoConfig {
            width: 16,
            height: 16,
            time_base_num: 1,
            time_base_den: 30,
        };
        let mut writer = NutWriter::create(path, config).expect("create nut");
        for index in 0..frame_count {
            let mut buffer = vec![0u8; 16 * 16 * 3];
            for (pixel_index, chunk) in buffer.chunks_mut(3).enumerate() {
                chunk[0] = ((pixel_index + index as usize) & 0xFF) as u8;
                chunk[1] = ((pixel_index * 3 + index as usize) & 0xFF) as u8;
                chunk[2] = ((pixel_index * 5 + index as usize) & 0xFF) as u8;
            }
            writer.write_frame(index, &buffer).expect("frame");
        }
        writer.finish().expect("finish nut");
        config
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

    #[tokio::test]
    async fn transcode_emits_both_outputs_and_removes_raw() {
        let Some(_) = locate_binary("ffmpeg") else {
            eprintln!(
                "ffmpeg not on PATH — skipping transcode test. Install \
                 `ffmpeg` to enable this test."
            );
            return;
        };
        let ffprobe = match locate_binary("ffprobe") {
            Some(path) => path,
            None => {
                eprintln!("ffprobe not on PATH — skipping transcode test.");
                return;
            }
        };

        let tempdir = TempDir::new().unwrap();
        let raw = tempdir.path().join("raw.nut");
        let lossy = tempdir.path().join("lossy.mp4");
        let lossless = tempdir.path().join("lossless.mp4");

        write_synthetic_nut(&raw, 8);

        let encoder = VideoEncoder::new();
        let request = VideoEncodeRequest {
            raw_nut: raw.clone(),
            lossy_mp4: lossy.clone(),
            lossless_mp4: lossless.clone(),
        };
        let outcome = encoder.run(&request).await.expect("transcode succeeds");

        assert!(outcome.lossy_bytes > 0);
        assert!(outcome.lossless_bytes > 0);
        assert!(outcome.raw_nut_removed);
        assert!(!raw.exists(), "raw.nut should be unlinked on success");

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
    async fn transcode_preserves_frame_count_for_microsecond_timebase() {
        // Reproduces the daemon's runtime configuration: 1/1_000_000
        // time-base with PTS values spaced at 60 Hz (≈16_667 μs apart).
        // Regression guard for the frame-count collapse where every frame
        // PTS was effectively zero and ffmpeg coalesced 600 frames into ~6.
        if locate_binary("ffmpeg").is_none() {
            eprintln!("ffmpeg not on PATH — skipping frame-count regression test.");
            return;
        }
        let ffprobe = match locate_binary("ffprobe") {
            Some(path) => path,
            None => {
                eprintln!("ffprobe not on PATH — skipping frame-count regression test.");
                return;
            }
        };

        let tempdir = TempDir::new().unwrap();
        let raw = tempdir.path().join("raw.nut");
        let lossy = tempdir.path().join("lossy.mp4");
        let lossless = tempdir.path().join("lossless.mp4");

        // 64x64 RGB24 = 12_288 bytes per frame, so 30 frames > MAX_DISTANCE
        // (65_536) — this size forces the periodic-syncpoint emit path that
        // the production daemon hits with the integration matrix's smallest
        // camera (64x64 @ 60 Hz). Reducing this resolution would hide the
        // regression we just fixed.
        let config = NutVideoConfig {
            width: 64,
            height: 64,
            time_base_num: 1,
            time_base_den: 1_000_000,
        };
        let mut writer = NutWriter::create(&raw, config).unwrap();
        // Match the integration matrix's 10s @ 60 Hz workload — 600 frames
        // crosses the syncpoint cadence enough times that any pixel
        // shuffle around a syncpoint boundary will be observable.
        //
        // Timestamps simulate the integration test's actual producer:
        // `int((ts_start + N/60) * 1e9)`. The float64 precision loss at
        // large ts_start (epoch nanoseconds) is intentional — it
        // reproduces the real-world PTS quantisation that triggered the
        // bug, and exercises the `last_pts + 1` monotonicity guard the
        // production trace_actor relies on.
        let frame_count: u64 = 600;
        let ts_start_s: f64 = 1_779_000_000.0;
        for index in 0..frame_count {
            // Mirror the integration test's frame-code encoder: write the
            // 16-byte big-endian representation of `index` into the red channel
            // of the top-left 4x4 pixel grid (the very pattern the data-integrity
            // pass decodes back from cloud-downloaded frames). The rest of the
            // frame is a uniform fill so lossy quantisation can't corrupt the
            // marker.
            const GRID_SIZE: usize = 4;
            const WIDTH: usize = 64;
            let mut buffer = vec![128u8; 64 * 64 * 3];
            let frame_bytes = (index as u128).to_be_bytes();
            for row in 0..GRID_SIZE {
                for col in 0..GRID_SIZE {
                    let byte_value = frame_bytes[row * GRID_SIZE + col];
                    let pixel_offset = (row * WIDTH + col) * 3;
                    buffer[pixel_offset] = byte_value;
                    buffer[pixel_offset + 1] = 255 - byte_value;
                    buffer[pixel_offset + 2] = byte_value / 2;
                }
            }
            // Replicate the integration test producer + production
            // `trace_actor` together: timestamps are derived from float
            // seconds (so PTS spacing reflects 60 Hz quantisation), then
            // re-based to a per-trace origin the way `trace_actor` does,
            // so PTS values stay in the same small range the daemon
            // actually emits in production.
            let timestamp_ns: i64 = ((ts_start_s + (index as f64) / 60.0) * 1_000_000_000.0) as i64;
            let origin_ns: i64 = (ts_start_s * 1_000_000_000.0) as i64;
            let pts_us = ((timestamp_ns - origin_ns) / 1_000) as u64;
            writer.write_frame(pts_us, &buffer).unwrap();
        }
        writer.finish().unwrap();

        let encoder = VideoEncoder::new();
        let request = VideoEncodeRequest {
            raw_nut: raw.clone(),
            lossy_mp4: lossy.clone(),
            lossless_mp4: lossless.clone(),
        };
        encoder.run(&request).await.expect("transcode succeeds");

        for path in [&lossy, &lossless] {
            let output = StdCommand::new(&ffprobe)
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
            assert!(
                output.status.success(),
                "ffprobe rejected {}: {}",
                path.display(),
                String::from_utf8_lossy(&output.stderr)
            );
            let raw = String::from_utf8(output.stdout).unwrap();
            let trimmed = raw.trim();
            let nb_read_frames: u64 = trimmed.parse().unwrap_or_else(|_| {
                panic!("expected integer frame count in ffprobe output, got {trimmed:?}")
            });
            assert_eq!(
                nb_read_frames,
                frame_count,
                "{} should contain all {} frames (got {})",
                path.display(),
                frame_count,
                nb_read_frames
            );
        }

        // The lossless transcode must round-trip the per-frame red marker
        // exactly. If the muxer shuffles or duplicates frames around a
        // syncpoint boundary (the integration-test failure mode), the
        // decoded marker sequence will skip or repeat.
        let ffmpeg = locate_binary("ffmpeg").expect("ffmpeg located above");
        let frames_dir = tempdir.path().join("frames");
        std::fs::create_dir_all(&frames_dir).unwrap();
        let pattern = frames_dir.join("%04d.rgb");
        // Use the LOSSLESS mp4 — `qp=0` round-trips the marker bytes
        // exactly so we can assert on the precise frame index. The lossy
        // mp4 is also produced by the same encoder invocation; the
        // integration test exercises both qualities and any frame-shuffle
        // bug in our muxer would affect both streams equally.
        let extract = StdCommand::new(&ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-i"])
            .arg(&lossless)
            .args(["-f", "image2", "-vcodec", "rawvideo", "-pix_fmt", "rgb24"])
            .arg(&pattern)
            .output()
            .expect("spawn ffmpeg extract");
        assert!(
            extract.status.success(),
            "ffmpeg extract failed: {}",
            String::from_utf8_lossy(&extract.stderr)
        );
        const GRID_SIZE_BACK: usize = 4;
        const WIDTH_BACK: usize = 64;
        for index in 0..frame_count {
            let path = frames_dir.join(format!("{:04}.rgb", index + 1));
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|e| panic!("could not read {}: {e}", path.display()));
            let mut decoded = [0u8; 16];
            for row in 0..GRID_SIZE_BACK {
                for col in 0..GRID_SIZE_BACK {
                    let pixel_offset = (row * WIDTH_BACK + col) * 3;
                    decoded[row * GRID_SIZE_BACK + col] = bytes[pixel_offset];
                }
            }
            let actual = u128::from_be_bytes(decoded) as u64;
            assert_eq!(
                actual, index,
                "decoded frame at output index {} carries marker {} (expected {})",
                index, actual, index
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
        let request = VideoEncodeRequest {
            raw_nut: tempdir.path().join("does-not-exist.nut"),
            lossy_mp4: tempdir.path().join("lossy.mp4"),
            lossless_mp4: tempdir.path().join("lossless.mp4"),
        };
        let encoder = VideoEncoder::new();
        let error = encoder.run(&request).await.expect_err("ffmpeg should fail");
        assert!(
            matches!(error, VideoEncodeError::NonZeroExit { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(
            request.raw_nut.parent().unwrap().exists(),
            "tempdir should still exist after failed encode"
        );
    }

    #[tokio::test]
    async fn spawn_failure_surfaces_binary_name() {
        let tempdir = TempDir::new().unwrap();
        let raw = tempdir.path().join("raw.nut");
        std::fs::write(&raw, [0u8; 16]).unwrap();
        let request = VideoEncodeRequest {
            raw_nut: raw,
            lossy_mp4: tempdir.path().join("lossy.mp4"),
            lossless_mp4: tempdir.path().join("lossless.mp4"),
        };
        let encoder =
            VideoEncoder::new().with_binary("this-binary-definitely-does-not-exist-ffmpeg");
        let error = encoder.run(&request).await.expect_err("spawn should fail");
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
