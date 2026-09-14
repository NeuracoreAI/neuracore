//! Logging and readiness-reporting helpers for the `launch` subcommand.
//!
//! Resolves the background-mode log destination, configures
//! `tracing-subscriber`, and reports startup failures either to the launcher's
//! readiness pipe or stderr.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Local;
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::fmt::format::Writer;
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields};
use tracing_subscriber::registry::LookupSpan;

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

/// Render a severity under the name Python's `logging` module uses for it.
///
/// `TRACE` has no Python equivalent and keeps its own name.
fn python_level(level: Level) -> &'static str {
    match level {
        Level::ERROR => "ERROR",
        Level::WARN => "WARNING",
        Level::INFO => "INFO",
        Level::DEBUG => "DEBUG",
        Level::TRACE => "TRACE",
    }
}

/// Event formatter matching the SDK's Python log lines: local timestamp with
/// milliseconds, severity in an 8-wide column, dotted target in a 30-wide column,
/// then the message and its fields.
struct NeuracoreFormat;

impl<S, N> FormatEvent<S, N> for NeuracoreFormat
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
    N: for<'field_writer> FormatFields<'field_writer> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        let metadata = event.metadata();
        write!(
            writer,
            "{} {:<8} {:<30} ",
            Local::now().format("%Y-%m-%d %H:%M:%S,%3f"),
            python_level(*metadata.level()),
            metadata.target().replace("::", "."),
        )?;
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

/// Configure `tracing-subscriber` from `RUST_LOG` / `NDD_DEBUG`.
///
/// In background mode the caller passes `Some(log_path)`; otherwise tracing
/// writes to stderr. `try_init` is used to tolerate test harnesses that have
/// already installed a global subscriber.
pub(crate) fn init_tracing(debug: bool, log_file: Option<&Path>) -> Result<()> {
    let default_level = if debug { "debug" } else { "warn" };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));

    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(false)
        .event_format(NeuracoreFormat);

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
        let _ = builder.with_writer(std::sync::Mutex::new(file)).try_init();
    } else {
        // Write to stderr so the parent's stdout=DEVNULL plumbing in
        // background mode does not silently swallow structured log output.
        let _ = builder.with_writer(std::io::stderr).try_init();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for SharedBuffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'writer> tracing_subscriber::fmt::MakeWriter<'writer> for SharedBuffer {
        type Writer = SharedBuffer;

        fn make_writer(&'writer self) -> Self::Writer {
            self.clone()
        }
    }

    fn capture(emit: impl FnOnce()) -> String {
        let buffer = SharedBuffer::default();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(Level::TRACE)
            .event_format(NeuracoreFormat)
            .with_writer(buffer.clone())
            .finish();
        tracing::subscriber::with_default(subscriber, emit);
        let bytes = buffer.0.lock().unwrap().clone();
        String::from_utf8(bytes).unwrap()
    }

    #[test]
    fn formats_event_into_the_python_column_layout() {
        let output = capture(|| {
            tracing::warn!(target: "data_daemon::upload::worker", attempt = 3, "retry scheduled");
        });

        let (timestamp, rest) = output.split_at(23);
        assert_eq!(&timestamp[4..5], "-");
        assert_eq!(&timestamp[10..11], " ");
        assert_eq!(&timestamp[19..20], ",");
        assert_eq!(
            rest,
            " WARNING  data_daemon.upload.worker      retry scheduled attempt=3\n"
        );
        assert!(!output.contains('\u{1b}'));
    }

    #[test]
    fn renders_every_severity_under_its_python_name() {
        assert_eq!(python_level(Level::ERROR), "ERROR");
        assert_eq!(python_level(Level::WARN), "WARNING");
        assert_eq!(python_level(Level::INFO), "INFO");
        assert_eq!(python_level(Level::DEBUG), "DEBUG");
        assert_eq!(python_level(Level::TRACE), "TRACE");
    }
}
