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

/// Directory name (inside a video trace's directory) that holds the
/// producer-spooled NUT chunks awaiting daemon-side encoding.
pub const CHUNKS_DIRNAME: &str = "chunks";

/// Top-level directory (under `recordings_root`) the producer spools video NUT
/// chunks into before the daemon knows which recording they belong to. Keyed
/// by source + sensor because the producer cannot reference a recording. The
/// daemon relinks a chunk under its recording once routing resolves a window,
/// and reclaims the whole tree on startup (a daemon restart mid-recording
/// corrupts that recording).
pub const SPOOL_DIRNAME: &str = ".rgb_spool";

/// Resolve the producer's video spool directory for a `(source, sensor)`
/// stream: `{recordings_root}/.rgb_spool/{robot_id}/{instance}/{data_type}/{sensor_name}/`.
///
/// Both producer and daemon agree on this layout so the daemon can find and
/// relink the producer's spooled NUTs. `sensor_name` is omitted from the path
/// when absent.
pub fn spool_dir(
    recordings_root: &Path,
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    sensor_name: Option<&str>,
) -> PathBuf {
    let mut dir = recordings_root
        .join(SPOOL_DIRNAME)
        .join(robot_id)
        .join(robot_instance.to_string())
        .join(data_type);
    if let Some(sensor_name) = sensor_name {
        dir = dir.join(sensor_name);
    }
    dir
}

/// Build the spool filename for a chunk:
/// `chunk_{publish_ns}_{thread_id}.nut`.
///
/// `publish_ns` (the chunk's `publish_timestamp_ns` — the wall-clock ns the
/// producer opened the chunk) and the producing thread's `thread_id` make the
/// name unique per `(source, sensor)` across recordings — a fresh recording no
/// longer reuses a previous one's filename, so the daemon's relink can never
/// collide with the next recording's spool. The daemon assigns its own
/// per-trace [`chunk_filename`] at relink time, so these values are never
/// otherwise interpreted.
pub fn spool_chunk_filename(publish_ns: i64, thread_id: i64) -> String {
    format!("chunk_{publish_ns}_{thread_id}.nut")
}

/// Resolve the full spool path for one spooled chunk.
pub fn spool_chunk_path(
    recordings_root: &Path,
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    sensor_name: Option<&str>,
    publish_ns: i64,
    thread_id: i64,
) -> PathBuf {
    spool_dir(
        recordings_root,
        robot_id,
        robot_instance,
        data_type,
        sensor_name,
    )
    .join(spool_chunk_filename(publish_ns, thread_id))
}

/// Resolve the top-level spool directory, reclaimed wholesale on daemon start.
pub fn spool_root(recordings_root: &Path) -> PathBuf {
    recordings_root.join(SPOOL_DIRNAME)
}

/// Build the filename for a video chunk at `chunk_index` — `chunk_NNNN.nut`.
///
/// The producer writes directly to this final path; no `.tmp` staging is
/// needed because the daemon only acts on a chunk once the producer has
/// published its [`Envelope::VideoChunkReady`], which happens after the NUT
/// writer has been finished and flushed.
///
/// [`Envelope::VideoChunkReady`]: data_daemon_ipc::Envelope::VideoChunkReady
pub fn chunk_filename(chunk_index: u32) -> String {
    format!("chunk_{chunk_index:04}.nut")
}

/// Build the filename for a per-chunk encoded lossy mp4 segment.
pub fn chunk_lossy_filename(chunk_index: u32) -> String {
    format!("chunk_{chunk_index:04}_lossy.mp4")
}

/// Build the filename for a per-chunk encoded lossless mp4 segment.
pub fn chunk_lossless_filename(chunk_index: u32) -> String {
    format!("chunk_{chunk_index:04}_lossless.mp4")
}

/// Resolve a recording's top-level directory: `{recordings_root}/{recording}`.
///
/// `recording` is the daemon-local `recording_index` stringified — the same
/// value the per-trace [`TracePath`] uses as its first path segment, so this
/// directory contains every trace directory for the recording. Used by the
/// recording reaper to remove a fully-uploaded recording's artefacts in one go.
pub fn recording_dir(recordings_root: &Path, recording_index: i64) -> PathBuf {
    recordings_root.join(recording_index.to_string())
}

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

    /// Resolve the per-trace `chunks/` directory used by the producer to
    /// spool NUT chunks before the daemon encodes them. Both producer and
    /// daemon agree on the layout via this helper so a daemon recovery sweep
    /// can find the producer's leftovers.
    pub fn chunks_dir(&self, recordings_root: &Path) -> PathBuf {
        self.directory(recordings_root).join(CHUNKS_DIRNAME)
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
            path.chunks_dir(root),
            PathBuf::from("/var/data/recordings/rec-1/joints/trace-1/chunks")
        );
    }

    #[test]
    fn chunk_filenames_are_zero_padded() {
        assert_eq!(chunk_filename(0), "chunk_0000.nut");
        assert_eq!(chunk_filename(7), "chunk_0007.nut");
        assert_eq!(chunk_filename(1234), "chunk_1234.nut");
        assert_eq!(chunk_lossy_filename(5), "chunk_0005_lossy.mp4");
        assert_eq!(chunk_lossless_filename(5), "chunk_0005_lossless.mp4");
    }

    #[test]
    fn spool_chunk_filename_is_unique_per_publish_ts_and_thread() {
        // The whole point of keying on `(publish_ns, thread_id)` is that two
        // recordings on the same `(source, sensor)` never collide on a spool
        // filename — distinct opens yield distinct names; identical inputs are
        // stable so the daemon reconstructs exactly what the producer wrote.
        let first = spool_chunk_filename(1_700_000_000_000_000_000, 42);
        let second = spool_chunk_filename(1_700_000_000_000_000_001, 42);
        let other_thread = spool_chunk_filename(1_700_000_000_000_000_000, 43);
        assert_eq!(first, "chunk_1700000000000000000_42.nut");
        assert_ne!(first, second, "a later open must not reuse the filename");
        assert_ne!(first, other_thread, "a different thread disambiguates");
        assert_eq!(
            first,
            spool_chunk_filename(1_700_000_000_000_000_000, 42),
            "identical inputs must be stable"
        );
    }

    #[test]
    fn spool_chunk_path_lives_under_the_spool_dir() {
        let root = Path::new("/var/data/recordings");
        let path = spool_chunk_path(root, "robot-1", 0, "RGB_IMAGES", Some("camera_0"), 150, 7);
        assert_eq!(
            path,
            PathBuf::from(
                "/var/data/recordings/.rgb_spool/robot-1/0/RGB_IMAGES/camera_0/chunk_150_7.nut"
            )
        );
        assert_eq!(
            spool_root(root),
            PathBuf::from("/var/data/recordings/.rgb_spool")
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
