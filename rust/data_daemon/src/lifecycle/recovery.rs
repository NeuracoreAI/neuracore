//! Startup recovery from a previous unclean exit.
//!
//! After SIGKILL or a host crash, on-disk artefacts from the previous daemon
//! run can be left behind: a stale PID file containing a PID that is no longer
//! running, partially-written recordings, and (once Phase 4 lands)
//! iceoryx2 dead-node files. This module exposes the small surface needed by
//! `cli::launch` to bring the host into a consistent state before the new
//! daemon starts.

use std::path::Path;

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
/// Placeholder for the Phase 4 IPC bring-up: when the daemon starts using
/// iceoryx2 services this will call `iceoryx2::node::Node::cleanup_dead_nodes`
/// (or equivalent) and remove the resulting empty discovery files. Until then
/// there are no iceoryx2 artefacts on disk, so this is a no-op that returns
/// the count of cleaned artefacts (always zero).
pub fn cleanup_stale_ipc() -> std::io::Result<usize> {
    Ok(0)
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
}
