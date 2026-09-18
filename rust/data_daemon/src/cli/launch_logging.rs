//! Logging and readiness-reporting helpers for the `launch` subcommand.
//!
//! Configures `tracing-subscriber` over a rotating [`logroller::LogRoller`]
//! appender, and reports startup failures either to the launcher's readiness
//! pipe or stderr.

use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};

use logroller::{LogRollerBuilder, Rotation, RotationSize};

use crate::lifecycle::daemonize::ReadinessReporter;

/// Bytes the active log may reach before it rolls to `daemon.log.1`.
const DEFAULT_MAX_LOG_BYTES: u64 = 10 * 1024 * 1024;

/// Rolled generations kept beside the active log.
const DEFAULT_MAX_LOG_FILES: u64 = 5;

/// Env var overriding [`DEFAULT_MAX_LOG_BYTES`]; accepts unit suffixes (`10mb`).
const MAX_LOG_BYTES_ENV: &str = "NCD_LOG_MAX_SIZE";

/// Env var overriding [`DEFAULT_MAX_LOG_FILES`].
const MAX_LOG_FILES_ENV: &str = "NCD_LOG_MAX_FILES";

/// Append a launch failure to the daemon's log, and echo it to stderr.
///
/// The failures that happen before [`init_tracing`] have no subscriber to
/// carry them, and an SDK-launched daemon has no terminal reading its stderr,
/// so without this their reason is lost. Appends directly rather than through
/// the rotating writer, which only the process holding the PID lock may open.
pub(crate) fn log_launch_failure(log_file: Option<&Path>, message: &str) {
    eprintln!("{message}");
    let Some(path) = log_file else {
        return;
    };
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    else {
        return;
    };
    let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.6fZ");
    let _ = writeln!(file, "{timestamp} ERROR {message}");
}

pub(crate) fn report_failure(reporter: Option<ReadinessReporter>, message: &str) {
    if let Some(reporter) = reporter {
        let _ = reporter.fail(message);
    } else {
        eprintln!("{message}");
    }
}

/// Rotation threshold in bytes, from [`MAX_LOG_BYTES_ENV`].
fn max_log_bytes() -> u64 {
    let Ok(raw) = std::env::var(MAX_LOG_BYTES_ENV) else {
        return DEFAULT_MAX_LOG_BYTES;
    };
    match crate::config::env::parse_bytes(&raw) {
        Ok(bytes) if bytes > 0 => bytes as u64,
        _ => {
            eprintln!("ignoring invalid {MAX_LOG_BYTES_ENV}={raw}; using the default");
            DEFAULT_MAX_LOG_BYTES
        }
    }
}

/// Number of rolled generations to keep, from [`MAX_LOG_FILES_ENV`].
fn max_log_files() -> u64 {
    let Ok(raw) = std::env::var(MAX_LOG_FILES_ENV) else {
        return DEFAULT_MAX_LOG_FILES;
    };
    match raw.trim().parse::<u64>() {
        Ok(files) if files > 0 => files,
        _ => {
            eprintln!("ignoring invalid {MAX_LOG_FILES_ENV}={raw}; using the default");
            DEFAULT_MAX_LOG_FILES
        }
    }
}

/// Build the rotating appender for `path`.
///
/// `LogRoller` opens with `O_APPEND`, so a restarted daemon continues the
/// existing file instead of replacing it, and rotates *before* each write, so
/// a record is never split across two files.
fn build_appender(path: &Path) -> Result<logroller::LogRoller, String> {
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    let filename = Path::new(path.file_name().unwrap_or_else(|| OsStr::new("daemon.log")));
    // `LogRollerBuilder::new` binds both arguments to one type parameter.
    LogRollerBuilder::new(directory, filename)
        .rotation(Rotation::SizeBased(RotationSize::Bytes(max_log_bytes())))
        .max_keep_files(max_log_files())
        .build()
        .map_err(|error| error.to_string())
}

/// Configure `tracing-subscriber` from `RUST_LOG` / `NDD_DEBUG`.
///
/// Writes to `log_file` when the caller resolved one, otherwise to stderr.
/// A log destination that cannot be opened degrades to stderr rather than
/// failing the launch, so a full disk or an unwritable directory never stops
/// the daemon from running. `try_init` is used to tolerate test harnesses that
/// have already installed a global subscriber.
pub(crate) fn init_tracing(debug: bool, log_file: Option<&Path>) {
    let default_level = if debug { "debug" } else { "warn" };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));

    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false);

    if let Some(path) = log_file {
        match build_appender(path) {
            Ok(appender) => {
                // `with_ansi(false)` keeps escape codes out of the file; the
                // `ansi` feature is on and there is no isatty check.
                let _ = builder
                    .with_writer(std::sync::Mutex::new(appender))
                    .with_ansi(false)
                    .try_init();
                return;
            }
            Err(error) => {
                eprintln!(
                    "failed to open log file {}: {error}; logging to stderr",
                    path.display()
                );
            }
        }
    }
    // Write to stderr so the parent's stdout=DEVNULL plumbing in background
    // mode does not silently swallow structured log output.
    let _ = builder.with_writer(std::io::stderr).try_init();
}

/// Route panics into the daemon's log for the whole life of the process.
///
/// Installed before [`init_tracing`], so it covers a panic raised before there
/// is a subscriber. Replaces the default hook rather than chaining it, since
/// [`log_launch_failure`] already echoes to stderr.
pub(crate) fn install_panic_logger(log_file: Option<PathBuf>) {
    std::panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        log_launch_failure(
            log_file.as_deref(),
            &format!("daemon panicked: {info}\n{backtrace}"),
        );
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn launch_failures_and_panics_append_to_the_daemon_log() {
        let directory = TempDir::new().expect("temp dir");
        let log = directory.path().join("daemon.log");

        log_launch_failure(Some(&log), "Daemon already running (pid=1234)");

        let saved = std::panic::take_hook();
        install_panic_logger(Some(log.clone()));
        let _ = std::panic::catch_unwind(|| panic!("deliberate test panic"));
        std::panic::set_hook(saved);

        let written = std::fs::read_to_string(&log).expect("log written");
        assert!(
            written.contains("Daemon already running (pid=1234)"),
            "{written}"
        );
        assert!(written.contains("deliberate test panic"), "{written}");
    }

    // Mutates process-wide env vars, so a single test drives the matrix (see
    // `data_daemon_shared::paths::resolution_precedence`).
    #[test]
    fn rotation_knobs_read_env_and_reject_nonsense() {
        let saved_size = std::env::var_os(MAX_LOG_BYTES_ENV);
        let saved_files = std::env::var_os(MAX_LOG_FILES_ENV);

        std::env::remove_var(MAX_LOG_BYTES_ENV);
        std::env::remove_var(MAX_LOG_FILES_ENV);
        assert_eq!(max_log_bytes(), DEFAULT_MAX_LOG_BYTES);
        assert_eq!(max_log_files(), DEFAULT_MAX_LOG_FILES);

        std::env::set_var(MAX_LOG_BYTES_ENV, "64kb");
        std::env::set_var(MAX_LOG_FILES_ENV, "2");
        assert_eq!(max_log_bytes(), 64 * 1024);
        assert_eq!(max_log_files(), 2);

        // Unparseable, zero, and negative all fall back rather than failing.
        for value in ["banana", "0", "-1"] {
            std::env::set_var(MAX_LOG_BYTES_ENV, value);
            assert_eq!(max_log_bytes(), DEFAULT_MAX_LOG_BYTES);
            std::env::set_var(MAX_LOG_FILES_ENV, value);
            assert_eq!(max_log_files(), DEFAULT_MAX_LOG_FILES);
        }

        match saved_size {
            Some(value) => std::env::set_var(MAX_LOG_BYTES_ENV, value),
            None => std::env::remove_var(MAX_LOG_BYTES_ENV),
        }
        match saved_files {
            Some(value) => std::env::set_var(MAX_LOG_FILES_ENV, value),
            None => std::env::remove_var(MAX_LOG_FILES_ENV),
        }
    }

    #[test]
    fn appender_appends_to_an_existing_log() {
        let tempdir = TempDir::new().unwrap();
        let path = tempdir.path().join("daemon.log");
        std::fs::write(&path, b"from the previous daemon\n").unwrap();

        {
            use std::io::Write as _;
            let mut appender = build_appender(&path).unwrap();
            appender.write_all(b"from this daemon\n").unwrap();
        }

        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("from the previous daemon"));
        assert!(contents.contains("from this daemon"));
    }

    #[test]
    fn appender_creates_a_missing_log_directory() {
        let tempdir = TempDir::new().unwrap();
        let path = tempdir.path().join("nested").join("daemon.log");

        assert!(build_appender(&path).is_ok());
        assert!(path.exists());
    }
}
