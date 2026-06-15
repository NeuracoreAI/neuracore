//! Logging and readiness-reporting helpers for the `launch` subcommand.
//!
//! Resolves the background-mode log destination, configures
//! `tracing-subscriber`, and reports startup failures either to the launcher's
//! readiness pipe or stderr.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::env::RuntimeEnv;
use crate::lifecycle::daemonize::ReadinessReporter;

pub(crate) fn report_failure(reporter: Option<ReadinessReporter>, message: &str) {
    if let Some(reporter) = reporter {
        let _ = reporter.fail(message);
    } else {
        eprintln!("{message}");
    }
}

/// Resolve the log-file destination for background mode.
///
/// Defaults to a `daemon.log` sibling of the state database, which is itself
/// configurable via `NEURACORE_DAEMON_DB_PATH`. If the DB path is relative or
/// has no parent (e.g. a user override like `state.db`), falls back to
/// `~/.neuracore/data_daemon/daemon.log` rather than the launcher's CWD —
/// `daemonize` `chdir("/")`s the grandchild, so a relative log path would
/// otherwise land at the filesystem root.
pub(crate) fn log_path_for(runtime_env: &RuntimeEnv) -> PathBuf {
    let candidate = runtime_env
        .db_path
        .parent()
        .map(|parent| parent.join("daemon.log"));
    if let Some(path) = candidate {
        if path.is_absolute() {
            return path;
        }
    }
    if let Some(home) = dirs::home_dir() {
        return home
            .join(".neuracore")
            .join("data_daemon")
            .join("daemon.log");
    }
    PathBuf::from("/tmp/neuracore-data-daemon.log")
}

/// Configure `tracing-subscriber` from `RUST_LOG` / `NDD_DEBUG`.
///
/// In background mode the caller passes `Some(log_path)`; otherwise tracing
/// writes to stderr. `try_init` is used to tolerate test harnesses that have
/// already installed a global subscriber.
pub(crate) fn init_tracing(debug: bool, log_file: Option<&Path>) -> Result<()> {
    let default_level = if debug { "debug" } else { "info" };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));

    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false);

    if let Some(path) = log_file {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create log directory {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("failed to open log file {}", path.display()))?;
        let _ = builder
            .with_writer(std::sync::Mutex::new(file))
            .with_ansi(false)
            .try_init();
    } else {
        // Write to stderr so the parent's stdout=DEVNULL plumbing in
        // background mode does not silently swallow structured log output.
        let _ = builder.with_writer(std::io::stderr).try_init();
    }
    Ok(())
}
