//! ffmpeg command-line compatibility.
//!
//! The daemon transcodes with the system ffmpeg and the producer's NUT tests
//! decode with it. No single spelling of the passthrough frame-timing option
//! works on every supported ffmpeg, so it is resolved against the installed
//! build.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};

/// Spelling accepted from ffmpeg 5.1 onwards, and the only one accepted from
/// 8.0 where the legacy name was removed.
const FPS_MODE_ARG: &str = "-fps_mode";

/// Spelling accepted before ffmpeg 5.1. Deprecated from 5.1 and removed in 8.0.
const VSYNC_ARG: &str = "-vsync";

/// One 16x16 yuv420p frame: a 16x16 luma plane plus two 8x8 chroma planes.
const PROBE_FRAME_LEN: usize = 16 * 16 * 3 / 2;

/// Start of the message ffmpeg prints when it rejects an unknown option.
const UNKNOWN_OPTION_MESSAGE: &str = "Unrecognized option";

/// Spellings already resolved, keyed by binary. The daemon resolves its binary
/// during startup preflight, before the async runtime exists, so transcodes
/// only ever reach the lookup.
static RESOLVED_ARGS: OnceLock<Mutex<HashMap<OsString, &'static str>>> = OnceLock::new();

/// Return the option name `binary` accepts to select passthrough frame timing
/// (used as `<arg> passthrough`).
///
/// `-fps_mode` does not exist before ffmpeg 5.1 and `-vsync` was removed in
/// 8.0, so builds outside 5.1..=7.x accept only one of the two.
///
/// A probe failure that does not name the option is warned about and resolves
/// to the pre-5.1 spelling, leaving the caller's real invocation to report the
/// underlying problem.
pub fn passthrough_frame_sync_arg(binary: &OsStr) -> &'static str {
    let mut resolved = RESOLVED_ARGS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(arg) = resolved.get(binary) {
        return arg;
    }
    let arg = if accepts_fps_mode(binary) {
        FPS_MODE_ARG
    } else {
        VSYNC_ARG
    };
    tracing::debug!(
        binary = %binary.to_string_lossy(),
        arg,
        "resolved ffmpeg passthrough frame-timing option"
    );
    resolved.insert(binary.to_os_string(), arg);
    arg
}

/// Encode one synthetic frame to the null muxer with [`FPS_MODE_ARG`],
/// reporting whether ffmpeg accepted it.
///
/// rawvideo on stdin and the null muxer need no input file, no `lavfi` and no
/// encoder library, so the option under test is the only build-dependent part
/// left.
fn accepts_fps_mode(binary: &OsStr) -> bool {
    let spawned = Command::new(binary)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-video_size",
            "16x16",
            "-i",
            "-",
            FPS_MODE_ARG,
            "passthrough",
            "-f",
            "null",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) => {
            tracing::warn!(
                binary = %binary.to_string_lossy(),
                %error,
                "could not run ffmpeg to resolve its passthrough frame-timing option"
            );
            return false;
        }
    };
    // The frame is far smaller than a pipe buffer, so writing then dropping
    // stdin cannot deadlock against ffmpeg's reads.
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(&[128u8; PROBE_FRAME_LEN]);
    }
    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(error) => {
            tracing::warn!(
                binary = %binary.to_string_lossy(),
                %error,
                "ffmpeg passthrough probe could not be collected"
            );
            return false;
        }
    };
    if output.status.success() {
        return true;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.contains(UNKNOWN_OPTION_MESSAGE) {
        tracing::warn!(
            binary = %binary.to_string_lossy(),
            stderr = %stderr.trim(),
            "ffmpeg passthrough probe failed for an unexpected reason; assuming the \
             pre-5.1 spelling"
        );
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    /// Write an executable stand-in for ffmpeg that exits with `fps_mode_status`
    /// when it is given `-fps_mode`, and succeeds otherwise. It lets the
    /// selection be tested on a host with one ffmpeg version installed.
    fn write_ffmpeg_stub(directory: &Path, name: &str, fps_mode_status: u8) -> PathBuf {
        let path = directory.join(name);
        std::fs::write(
            &path,
            format!(
                "#!/bin/sh\n\
                 for argument in \"$@\"; do\n\
                 if [ \"$argument\" = \"{FPS_MODE_ARG}\" ]; then\n\
                 echo \"{UNKNOWN_OPTION_MESSAGE} 'fps_mode'.\" >&2\n\
                 exit {fps_mode_status}\n\
                 fi\n\
                 done\n\
                 exit 0\n"
            ),
        )
        .expect("write ffmpeg stub");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .expect("mark ffmpeg stub executable");
        path
    }

    #[test]
    fn a_build_that_rejects_fps_mode_resolves_to_the_legacy_spelling() {
        let directory = TempDir::new().unwrap();
        let stub = write_ffmpeg_stub(directory.path(), "ffmpeg-pre-5-1", 1);
        assert_eq!(passthrough_frame_sync_arg(stub.as_os_str()), VSYNC_ARG);
    }

    #[test]
    fn a_build_that_accepts_fps_mode_resolves_to_the_modern_spelling() {
        let directory = TempDir::new().unwrap();
        let stub = write_ffmpeg_stub(directory.path(), "ffmpeg-5-1-or-newer", 0);
        assert_eq!(passthrough_frame_sync_arg(stub.as_os_str()), FPS_MODE_ARG);
    }

    #[test]
    fn a_binary_is_probed_once_per_process() {
        // Swap the binary after resolving it: a second probe would return the
        // new answer, the memo keeps the first.
        let directory = TempDir::new().unwrap();
        let stub = write_ffmpeg_stub(directory.path(), "ffmpeg-swapped", 1);
        assert_eq!(passthrough_frame_sync_arg(stub.as_os_str()), VSYNC_ARG);

        write_ffmpeg_stub(directory.path(), "ffmpeg-swapped", 0);
        assert_eq!(passthrough_frame_sync_arg(stub.as_os_str()), VSYNC_ARG);
    }

    #[test]
    fn unusable_binary_resolves_to_the_legacy_spelling() {
        assert_eq!(
            passthrough_frame_sync_arg(OsStr::new("nc-definitely-not-a-real-ffmpeg-binary")),
            VSYNC_ARG
        );
    }

    #[test]
    fn resolved_spelling_is_accepted_by_the_local_ffmpeg() {
        // Skip where the toolchain is unavailable, matching the encoder tests.
        let Ok(located) = Command::new("which").arg("ffmpeg").output() else {
            return;
        };
        if !located.status.success() {
            return;
        }
        let binary = OsString::from(String::from_utf8_lossy(&located.stdout).trim());
        let arg = passthrough_frame_sync_arg(&binary);
        let probe = Command::new(&binary)
            .args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i"])
            .arg("testsrc=duration=1:size=16x16:rate=1")
            .args([arg, "passthrough", "-f", "null", "-"])
            .output()
            .expect("spawn ffmpeg");
        assert!(
            probe.status.success(),
            "ffmpeg rejected the resolved `{arg}`: {}",
            String::from_utf8_lossy(&probe.stderr)
        );
    }
}
