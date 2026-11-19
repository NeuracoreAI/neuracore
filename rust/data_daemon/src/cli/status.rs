//! `status` subcommand handler.
//!
//! Reads `NEURACORE_DAEMON_PID_PATH` and reports whether the daemon process
//! it points at is alive. That first line is a stable contract so external
//! scripts can parse it. A running daemon is then asked what its recordings
//! are waiting on, and its answer follows, one line each.

use std::time::Duration;

use anyhow::Result;

use crate::config::env::pid_path;
use crate::ipc::status_query::query_status;
use crate::lifecycle::pidfile::{pid_is_running, read_pid_from_file};

/// How long to wait for the running daemon to report its status.
const STATUS_TIMEOUT: Duration = Duration::from_secs(2);

/// Run the status command.
pub fn run() -> Result<()> {
    let path = pid_path();
    let Some(pid_value) = read_pid_from_file(&path) else {
        println!("Daemon not running.");
        return Ok(());
    };

    if pid_is_running(pid_value) {
        println!("Daemon running (pid={pid_value}).");
        match query_status(STATUS_TIMEOUT) {
            Ok(Some(status)) => {
                for line in status.summary() {
                    println!("{line}");
                }
            }
            Ok(None) => println!("The daemon did not report its status in time."),
            Err(error) => println!("Could not read the daemon's status: {error:#}."),
        }
    } else {
        println!("Daemon not running (stale pid file: {pid_value}).");
    }
    Ok(())
}
