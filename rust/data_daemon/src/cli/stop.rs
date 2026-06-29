//! `stop` subcommand handler.
//!
//! Reads the daemon's PID from `NEURACORE_DAEMON_PID_PATH`, sends SIGTERM,
//! waits up to 10 s for graceful exit, and escalates to SIGKILL if the
//! daemon is still alive at the deadline.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;

use crate::config::env::pid_path;
use crate::lifecycle::pidfile::{pid_is_running, read_pid_from_file};
use crate::lifecycle::recovery::reclaim_stale_pid_file;

const GRACEFUL_TIMEOUT: Duration = Duration::from_secs(10);
const SIGKILL_REAP_TIMEOUT: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Run the stop command.
pub fn run() -> Result<()> {
    let path = pid_path();
    let Some(pid_value) = read_pid_from_file(&path) else {
        println!("Daemon not running (no pid file at {}).", path.display());
        // Best-effort cleanup: if the file exists but is unreadable, drop it.
        let _ = reclaim_stale_pid_file(&path);
        return Ok(());
    };

    if !pid_is_running(pid_value) {
        println!("Daemon not running (pid={pid_value}); removing stale pid file.");
        let _ = reclaim_stale_pid_file(&path);
        return Ok(());
    }

    let pid = Pid::from_raw(pid_value);
    match kill(pid, Signal::SIGTERM) {
        Ok(()) => {}
        Err(nix::errno::Errno::ESRCH) => {
            println!("Daemon exited before SIGTERM (pid={pid_value}).");
            cleanup(&path);
            return Ok(());
        }
        Err(error) => {
            eprintln!("Failed to send SIGTERM to pid={pid_value}: {error}");
            std::process::exit(1);
        }
    }

    if wait_for_exit(pid_value, GRACEFUL_TIMEOUT) {
        println!("Daemon stopped (pid={pid_value}).");
        cleanup(&path);
        return Ok(());
    }

    eprintln!(
        "Daemon (pid={pid_value}) did not exit within {}s; sending SIGKILL.",
        GRACEFUL_TIMEOUT.as_secs()
    );
    let _ = kill(pid, Signal::SIGKILL);
    if !wait_for_exit(pid_value, SIGKILL_REAP_TIMEOUT) {
        eprintln!("Daemon (pid={pid_value}) still alive after SIGKILL.");
        std::process::exit(1);
    }
    cleanup(&path);
    Ok(())
}

/// Poll until `pid_value` exits or `timeout` elapses.
///
/// Uses `std::thread::sleep` because `stop` runs synchronously without a
/// Tokio runtime — see `cli::run` for the per-command runtime policy. Do not
/// call from async code.
fn wait_for_exit(pid_value: i32, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if !pid_is_running(pid_value) {
            return true;
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    !pid_is_running(pid_value)
}

fn cleanup(pid_path: &Path) {
    // The daemon's `PidFile::Drop` removes the file on a clean exit; this is
    // a defence-in-depth pass for the SIGKILL escalation path.
    let _ = reclaim_stale_pid_file(pid_path);
}
