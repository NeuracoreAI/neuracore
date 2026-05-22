//! Resolve on-disk paths for a recording, trace, and its artefacts.
//!
//! Mirrors `recording_encoding_disk_manager/core/trace_filesystem.py`. The
//! layout is part of the hard external contract: tests and downstream tools
//! expect `{recordings_root}/{recording_id}/{data_type}/{trace_id}/` and the
//! encoders read/write specific filenames inside that directory.

use std::path::{Path, PathBuf};

/// Filename for the JSON-array trace data, written by both scalar traces and
/// the video-trace sidecar. Matches `video_trace.py::TRACE_FILE` and the
/// scalar `JsonTrace` writer's default.
pub const TRACE_JSON_FILENAME: &str = "trace.json";

/// Filename for the H.264 lossy MP4. Matches `video_trace.py::LOSSY_VIDEO_NAME`.
pub const LOSSY_VIDEO_FILENAME: &str = "lossy.mp4";

/// Filename for the FFV1 lossless MP4. Matches `video_trace.py::LOSSLESS_VIDEO_NAME`.
pub const LOSSLESS_VIDEO_FILENAME: &str = "lossless.mp4";

/// Filename for the raw NUT spool written by the per-trace actor before
/// ffmpeg transcodes it.
pub const RAW_NUT_FILENAME: &str = "raw.nut";

/// Key for an on-disk trace directory.
///
/// The three components map directly to the on-disk path segments:
/// `recording_id` and `trace_id` come from the producer, `data_type` is the
/// wire label carried in `StartTrace`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TracePath {
    /// Recording the trace belongs to.
    pub recording_id: String,
    /// Wire data-type label (e.g. `"video"`, `"joints"`).
    pub data_type: String,
    /// Trace identifier supplied by the SDK.
    pub trace_id: String,
}

impl TracePath {
    /// Compose a new key from owned strings.
    pub fn new(
        recording_id: impl Into<String>,
        data_type: impl Into<String>,
        trace_id: impl Into<String>,
    ) -> Self {
        Self {
            recording_id: recording_id.into(),
            data_type: data_type.into(),
            trace_id: trace_id.into(),
        }
    }

    /// Resolve the trace directory beneath `recordings_root`.
    pub fn directory(&self, recordings_root: &Path) -> PathBuf {
        recordings_root
            .join(&self.recording_id)
            .join(&self.data_type)
            .join(&self.trace_id)
    }

    /// Resolve the `trace.json` path for this trace.
    pub fn trace_json(&self, recordings_root: &Path) -> PathBuf {
        self.directory(recordings_root).join(TRACE_JSON_FILENAME)
    }

    /// Resolve the `lossy.mp4` path for this trace.
    pub fn lossy_video(&self, recordings_root: &Path) -> PathBuf {
        self.directory(recordings_root).join(LOSSY_VIDEO_FILENAME)
    }

    /// Resolve the `lossless.mp4` path for this trace.
    pub fn lossless_video(&self, recordings_root: &Path) -> PathBuf {
        self.directory(recordings_root)
            .join(LOSSLESS_VIDEO_FILENAME)
    }

    /// Resolve the `raw.nut` spool path for this trace.
    pub fn raw_nut(&self, recordings_root: &Path) -> PathBuf {
        self.directory(recordings_root).join(RAW_NUT_FILENAME)
    }
}

/// Sum the byte count of every regular file beneath `root`.
///
/// Returns 0 when `root` does not exist, which is the expected state before
/// the recordings tree has been created. Unreadable entries are silently
/// skipped — the budget tracker treats them as zero.
pub fn directory_bytes(root: &Path) -> u64 {
    let mut total: u64 = 0;
    walk(root, &mut |entry| {
        if let Ok(metadata) = entry.metadata() {
            if metadata.is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    });
    total
}

fn walk(root: &Path, visit: &mut dyn FnMut(&std::fs::DirEntry)) {
    let read_dir = match std::fs::read_dir(root) {
        Ok(iterator) => iterator,
        Err(_) => return,
    };
    for entry in read_dir.flatten() {
        visit(&entry);
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(_) => continue,
        };
        if file_type.is_dir() {
            walk(&entry.path(), visit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn directory_layout_matches_python_convention() {
        let path = TracePath::new("rec-1", "joints", "trace-1");
        let root = Path::new("/var/data/recordings");
        assert_eq!(
            path.directory(root),
            PathBuf::from("/var/data/recordings/rec-1/joints/trace-1")
        );
        assert_eq!(
            path.trace_json(root),
            PathBuf::from("/var/data/recordings/rec-1/joints/trace-1/trace.json")
        );
        assert_eq!(
            path.lossy_video(root),
            PathBuf::from("/var/data/recordings/rec-1/joints/trace-1/lossy.mp4")
        );
        assert_eq!(
            path.lossless_video(root),
            PathBuf::from("/var/data/recordings/rec-1/joints/trace-1/lossless.mp4")
        );
        assert_eq!(
            path.raw_nut(root),
            PathBuf::from("/var/data/recordings/rec-1/joints/trace-1/raw.nut")
        );
    }

    #[test]
    fn directory_bytes_sums_nested_files_and_ignores_missing_roots() {
        let tempdir = TempDir::new().unwrap();
        let root = tempdir.path().join("recordings");

        // Missing root: zero, no error.
        assert_eq!(directory_bytes(&root), 0);

        let trace = TracePath::new("rec-1", "joints", "trace-1");
        let dir = trace.directory(&root);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("trace.json"), vec![0u8; 1024]).unwrap();
        std::fs::write(dir.join("extra.bin"), vec![0u8; 32]).unwrap();

        assert_eq!(directory_bytes(&root), 1024 + 32);
    }
}
