//! `status` subcommand handler.
//!
//! Reads `NEURACORE_DAEMON_PID_PATH` and reports whether the daemon process
//! it points at is alive. Mirrors the Python `run_status` output shape so
//! existing scripts continue to parse it.

use anyhow::Result;

use crate::config::env::pid_path;
use crate::lifecycle::pidfile::{pid_is_running, read_pid_from_file};

/// Run the status command.
pub fn run() -> Result<()> {
    let path = pid_path();
    let Some(pid_value) = read_pid_from_file(&path) else {
        println!("Daemon not running.");
        return Ok(());
    };

    if pid_is_running(pid_value) {
        println!("Daemon running (pid={pid_value}).");
    } else {
        println!("Daemon not running (stale pid file: {pid_value}).");
    }
    Ok(())
}
