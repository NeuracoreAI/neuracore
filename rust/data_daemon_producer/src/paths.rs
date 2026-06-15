//! Filesystem layout shared with the daemon.
//!
//! The producer spools NUT chunks into a recording-independent inbox under the
//! daemon's recordings root. The path helpers here mirror the daemon's
//! `storage::paths` / `config::env` byte-for-byte so the daemon finds exactly
//! what the producer wrote, and the `stream_key` helpers key the in-progress
//! video-chunk registry by `(source, sensor)`.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Spool directory name — must match `storage::paths::SPOOL_DIRNAME` on the
/// daemon side.
const SPOOL_DIRNAME: &str = ".rgb_spool";

/// Recordings root, resolved once per process via the shared resolver
/// ([`data_daemon_ipc::paths::recordings_root`]) so the producer and daemon
/// always compute the same root from the same inputs.
///
/// The `Err` case — no `$HOME` and no `NEURACORE_DAEMON_RECORDINGS_ROOT`
/// override — is surfaced to Python as a `PyErr` at the `log_frame` boundary
/// (see [`recordings_root`]). It is never allowed to panic across the FFI
/// boundary, nor to silently fall back to a scratch dir the daemon would never
/// read (which would lose the user's video).
static RECORDINGS_ROOT: LazyLock<Result<PathBuf, String>> =
    LazyLock::new(|| data_daemon_ipc::paths::recordings_root().map_err(|error| error.to_string()));

/// The resolved recordings root, or the resolution error message. The
/// `log_frame` pyfunction checks this on the GIL before enqueueing a frame, so
/// an unresolvable root becomes a clear Python exception instead of a
/// writer-thread failure or silent data loss.
pub(crate) fn recordings_root() -> Result<&'static Path, &'static str> {
    match &*RECORDINGS_ROOT {
        Ok(path) => Ok(path.as_path()),
        Err(message) => Err(message.as_str()),
    }
}

/// Composite registry key for one `(source, sensor)` video stream. The NUL
/// separators cannot occur in any component, so the join is unambiguous.
pub(crate) fn stream_key(
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    sensor_name: &str,
) -> String {
    format!("{robot_id}\u{0}{robot_instance}\u{0}{data_type}\u{0}{sensor_name}")
}

/// Prefix matching every video stream belonging to a source.
pub(crate) fn source_prefix(robot_id: &str, robot_instance: i64) -> String {
    format!("{robot_id}\u{0}{robot_instance}\u{0}")
}

/// Split a `stream_key` back into `(data_type, sensor_name)`. The leading
/// `robot_id\0instance\0` is dropped.
pub(crate) fn split_stream_key(key: &str) -> (String, String) {
    let mut parts = key.splitn(4, '\u{0}');
    let _robot_id = parts.next().unwrap_or("");
    let _instance = parts.next().unwrap_or("");
    let data_type = parts.next().unwrap_or("").to_string();
    let sensor_name = parts.next().unwrap_or("").to_string();
    (data_type, sensor_name)
}

/// Build the spool directory for a `(source, sensor)` stream. Mirrors
/// `storage::paths::spool_dir` on the daemon side; the two must agree
/// byte-for-byte so the daemon finds exactly what the producer wrote.
pub(crate) fn spool_dir(
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    sensor_name: &str,
) -> PathBuf {
    // The recordings root is validated at the `log_frame` boundary before any
    // frame is enqueued, so by the time the writer thread builds a spool path
    // resolution has already succeeded.
    recordings_root()
        .expect("recordings root validated at the log_frame boundary")
        .join(SPOOL_DIRNAME)
        .join(robot_id)
        .join(robot_instance.to_string())
        .join(data_type)
        .join(sensor_name)
}

/// Spool chunk filename — must match `storage::paths::spool_chunk_filename`.
pub(crate) fn spool_chunk_filename(publish_ns: i64, thread_id: i64) -> String {
    format!("chunk_{publish_ns}_{thread_id}.nut")
}
