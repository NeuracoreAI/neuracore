//! Startup recovery from a previous unclean exit.
//!
//! After SIGKILL or a host crash, on-disk artefacts from the previous daemon
//! run can be left behind: a stale PID file containing a PID that is no longer
//! running, partially-written recordings, and (once Phase 4 lands)
//! iceoryx2 dead-node files. This module exposes the small surface needed by
//! `cli::launch` to bring the host into a consistent state before the new
//! daemon starts.

use std::path::Path;

use iceoryx2::config::Config;
use iceoryx2::node::Node;
use iceoryx2::prelude::ipc;

use crate::lifecycle::pidfile::{pid_is_running, read_pid_from_file};

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

    #[test]
    fn cleanup_stale_ipc_is_safe_on_a_clean_host() {
        // Smoke test: the call must return even when there are no dead
        // nodes to reclaim. The real reclamation path is exercised by
        // `test_signal_cleanup.py` once Phase 4 lands end-to-end; reproducing
        // a SIGKILL'd iceoryx2 node from inside a cargo test would require
        // spawning a child binary, which is out of scope here.
        //
        // We can't assert the exact count because a parallel cargo test
        // process could be creating nodes; we just check the call returned.
        let _ = cleanup_stale_ipc();
    }
}
