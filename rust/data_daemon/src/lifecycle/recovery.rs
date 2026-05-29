//! Startup recovery from a previous unclean exit.
//!
//! After SIGKILL or a host crash, on-disk artefacts from the previous daemon
//! run can be left behind: a stale PID file containing a PID that is no longer
//! running, partially-written recordings, and iceoryx2 dead-node files. This
//! module exposes the small surface needed by
//! `cli::launch` to bring the host into a consistent state before the new
//! daemon starts.

use std::path::Path;

use iceoryx2::config::Config;
use iceoryx2::node::Node;
use iceoryx2::prelude::ipc;

use crate::lifecycle::pidfile::{pid_is_running, read_pid_from_file};
use crate::state::{SqliteStateStore, StateStore, StateStoreError, TraceWriteStatus};

/// Outcome of [`reclaim_stale_pid_file`], surfaced for logging.
#[derive(Debug, PartialEq, Eq)]
pub enum PidReclaim {
    /// No PID file was present.
    Absent,
    /// A PID file was present and its PID is still alive — the next acquire
    /// attempt will (correctly) report "already running".
    StillRunning(i32),
    /// A stale PID file (PID dead or unparseable) was removed.
    RemovedStale(Option<i32>),
}

/// Remove a PID file left by a previous SIGKILL'd daemon when its PID is no
/// longer running.
///
/// The new launcher's `PidFile::acquire` would itself recover via `flock`
/// alone, but eagerly clearing a stale file makes the `status` command and
/// concurrent diagnostics report accurate state instead of a misleading
/// "daemon running (pid=…)" pointed at a dead PID.
pub fn reclaim_stale_pid_file(pid_path: &Path) -> std::io::Result<PidReclaim> {
    if !pid_path.exists() {
        return Ok(PidReclaim::Absent);
    }

    let pid = read_pid_from_file(pid_path);
    if let Some(pid_value) = pid {
        if pid_is_running(pid_value) {
            return Ok(PidReclaim::StillRunning(pid_value));
        }
    }

    match std::fs::remove_file(pid_path) {
        Ok(()) => Ok(PidReclaim::RemovedStale(pid)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(PidReclaim::Absent),
        Err(error) => Err(error),
    }
}

/// Outcome counters for [`sweep_partial_recordings`], surfaced for logging.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct PartialSweepReport {
    /// Number of recordings whose on-disk artefacts were removed because at
    /// least one trace had not reached the `written` terminal state.
    pub recordings_purged: usize,
    /// Number of recordings inspected and left untouched (either every trace
    /// was `written`, or the recording was already cancelled and has no
    /// further state to clean up).
    pub recordings_preserved: usize,
}

/// Sweep partial recordings left behind by a previous daemon run.
///
/// Producer-side chunk spooling means a SIGKILL between two
/// `VideoChunkReady` envelopes can leave the recording with on-disk
/// artefacts (NUT chunks, half-encoded segments, partial concat outputs)
/// that no current actor will pick up. Mid-encode resume is intentionally
/// out of scope — keeping the lifecycle simple is the point of the
/// per-chunk design — so anything not in the `written` terminal state at
/// startup is purged.
///
/// For each recording:
/// - Already-cancelled recordings are skipped; the dispatcher's cancel
///   handler removed their on-disk state when the cancel originally fired.
/// - Recordings where every trace is `written` are left alone — the upload
///   path picks them up via the existing `TraceWritten` / pending-upload
///   gate.
/// - Anything else: the recording's directory is recursively removed and
///   the recording row is `cancel_recording`'d so the registration /
///   upload / progress coordinators ignore it (and so the in-flight trace
///   rows are burned to terminal `failed`).
pub async fn sweep_partial_recordings(
    store: &SqliteStateStore,
    recordings_root: &Path,
) -> Result<PartialSweepReport, StateStoreError> {
    let mut report = PartialSweepReport::default();
    // Reclaim the producer video inbox up front: any recording in flight at
    // restart is corrupt, so the spooled NUT chunks staged under the inbox
    // are reclaimed wholesale rather than resumed.
    let _ = std::fs::remove_dir_all(crate::storage::paths::inbox_root(recordings_root));
    let recordings = store.list_recordings().await?;
    for recording in recordings {
        if recording.cancelled_at.is_some() {
            // The original dispatcher cancel handler removed the on-disk
            // state when the cancel fired; if anything is left behind on
            // disk it was the cancel handler that failed, not a partial
            // write — and re-cancelling would be a no-op. Leave it.
            report.recordings_preserved += 1;
            continue;
        }
        let traces = store
            .list_traces_for_recording(recording.recording_index)
            .await?;
        let any_non_written = traces
            .iter()
            .any(|trace| trace.write_status != TraceWriteStatus::Written);
        if !any_non_written && !traces.is_empty() {
            // Every trace finished writing — the upload coordinator owns
            // it now.
            report.recordings_preserved += 1;
            continue;
        }

        let dir = recordings_root.join(recording.recording_index.to_string());
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                tracing::warn!(
                    %error,
                    recording_index = recording.recording_index,
                    path = %dir.display(),
                    "failed to purge partial recording directory; continuing"
                );
            }
        }

        if let Err(error) = store.cancel_recording(recording.recording_index).await {
            tracing::warn!(
                %error,
                recording_index = recording.recording_index,
                "failed to mark partial recording cancelled in state store; continuing"
            );
        }

        report.recordings_purged += 1;
    }
    Ok(report)
}

/// Reap stale iceoryx2 node files left by a SIGKILL'd daemon.
///
/// After SIGKILL, iceoryx2's per-node discovery files survive on the
/// filesystem (typically `/tmp/iceoryx2/...`) and prevent a fresh daemon from
/// cleanly attaching to its own services if the OS reuses the killed PID.
/// `Node::cleanup_dead_nodes` walks the global discovery registry, classifies
/// each entry, and removes the artefacts of nodes whose owning process is
/// gone.
///
/// Returns the number of dead nodes successfully reclaimed. The call itself
/// is infallible from our perspective — per-artefact failures are logged here
/// (they typically indicate the current process lacks permission to touch
/// another user's resources, which is expected when iceoryx2 is shared
/// system-wide) and never block daemon startup.
///
/// `NodeBuilder::create` *also* sweeps dead nodes on construction (controlled
/// by `cleanup_dead_nodes_on_creation`), but doing it eagerly here keeps the
/// `status` command's view of the system consistent before the new daemon
/// races to create its own node.
pub fn cleanup_stale_ipc() -> usize {
    let report = Node::<ipc::Service>::cleanup_dead_nodes(Config::global_config());
    if report.failed_cleanups > 0 {
        tracing::warn!(
            failed = report.failed_cleanups,
            "iceoryx2 dead-node sweep left {} artefacts behind (likely permission-denied; continuing)",
            report.failed_cleanups
        );
    }
    report.cleanups
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn reclaim_returns_absent_when_no_pid_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("daemon.pid");
        assert_eq!(reclaim_stale_pid_file(&path).unwrap(), PidReclaim::Absent);
    }

    #[test]
    fn reclaim_removes_stale_pid_file_with_dead_pid() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("daemon.pid");
        // `i32::MAX` is always above the kernel's `pid_max` (default 32768
        // on most distros, 2^22 on tuned hosts) and so is guaranteed not to
        // refer to a running process. Mirrors the trick used by
        // `pid_is_running_true_for_self_and_false_for_unused_pid` in
        // `pidfile::tests`.
        std::fs::write(&path, format!("{}\n", i32::MAX)).unwrap();
        let outcome = reclaim_stale_pid_file(&path).unwrap();
        assert_eq!(outcome, PidReclaim::RemovedStale(Some(i32::MAX)));
        assert!(!path.exists());
    }

    #[test]
    fn reclaim_removes_stale_pid_file_with_garbage_contents() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("daemon.pid");
        std::fs::write(&path, "not-a-pid\n").unwrap();
        let outcome = reclaim_stale_pid_file(&path).unwrap();
        assert_eq!(outcome, PidReclaim::RemovedStale(None));
        assert!(!path.exists());
    }

    #[test]
    fn reclaim_leaves_running_pid_file_in_place() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("daemon.pid");
        let our_pid = std::process::id() as i32;
        std::fs::write(&path, format!("{our_pid}\n")).unwrap();
        let outcome = reclaim_stale_pid_file(&path).unwrap();
        assert_eq!(outcome, PidReclaim::StillRunning(our_pid));
        assert!(path.exists());
    }

    use crate::state::store::TraceUpdate;
    use crate::state::NewRecording;
    use crate::storage::paths::TracePath;

    #[tokio::test]
    async fn partial_recording_is_removed_on_startup() {
        let dir = tempdir().unwrap();
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .expect("open store");
        let recordings_root = dir.path().join("recordings");

        let recording_index = store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                ..Default::default()
            })
            .await
            .unwrap()
            .recording_index;
        store
            .create_trace(recording_index, "trace-1", Some("RGB"), None)
            .await
            .unwrap();
        store
            .update_trace(
                "trace-1",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Writing),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Synthesise leftover on-disk state the previous daemon would have
        // produced — a chunks/ dir and a half-encoded segment.
        let trace_dir = TracePath::new(recording_index.to_string(), "RGB", "trace-1")
            .directory(&recordings_root);
        let chunks_dir = trace_dir.join("chunks");
        std::fs::create_dir_all(&chunks_dir).unwrap();
        std::fs::write(chunks_dir.join("chunk_0000.nut"), b"stale-bytes").unwrap();
        std::fs::write(trace_dir.join("chunk_0000_lossy.mp4"), b"halfway").unwrap();

        let report = sweep_partial_recordings(&store, &recordings_root)
            .await
            .expect("sweep");
        assert_eq!(report.recordings_purged, 1);
        assert_eq!(report.recordings_preserved, 0);
        assert!(!recordings_root.join(recording_index.to_string()).exists());

        let recording = store.get_recording(recording_index).await.unwrap().unwrap();
        assert!(recording.cancelled_at.is_some());
    }

    #[tokio::test]
    async fn completed_recording_is_preserved_for_upload() {
        let dir = tempdir().unwrap();
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .expect("open store");
        let recordings_root = dir.path().join("recordings");

        let recording_index = store
            .create_recording(NewRecording {
                robot_id: Some("robot-2"),
                robot_instance: Some(0),
                ..Default::default()
            })
            .await
            .unwrap()
            .recording_index;
        store
            .create_trace(recording_index, "trace-2", Some("RGB"), None)
            .await
            .unwrap();
        store
            .update_trace(
                "trace-2",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    total_bytes: Some(1024),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let trace_dir = TracePath::new(recording_index.to_string(), "RGB", "trace-2")
            .directory(&recordings_root);
        std::fs::create_dir_all(&trace_dir).unwrap();
        std::fs::write(trace_dir.join("lossy.mp4"), b"keep-me").unwrap();
        std::fs::write(trace_dir.join("lossless.mp4"), b"keep-me-too").unwrap();

        let report = sweep_partial_recordings(&store, &recordings_root)
            .await
            .expect("sweep");
        assert_eq!(report.recordings_purged, 0);
        assert_eq!(report.recordings_preserved, 1);
        assert!(trace_dir.join("lossy.mp4").exists());
        assert!(trace_dir.join("lossless.mp4").exists());

        let recording = store.get_recording(recording_index).await.unwrap().unwrap();
        assert!(recording.cancelled_at.is_none());
    }

    #[test]
    fn cleanup_stale_ipc_is_safe_on_a_clean_host() {
        // Smoke test: the call must return even when there are no dead
        // nodes to reclaim. The real reclamation path is exercised by the
        // end-to-end signal-cleanup integration test; reproducing a SIGKILL'd
        // iceoryx2 node from inside a cargo test would require spawning a
        // child binary, which is out of scope here.
        //
        // We can't assert the exact count because a parallel cargo test
        // process could be creating nodes; we just check the call returned.
        let _ = cleanup_stale_ipc();
    }
}
