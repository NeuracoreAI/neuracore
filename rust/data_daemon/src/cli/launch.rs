//! `launch` subcommand handler.
//!
//! Phase 2 wires the daemon lifecycle (PID file, signal handling, optional
//! background detachment) on top of the Phase 1 configuration resolution. The
//! daemon main loop installed here is intentionally minimal — it owns the
//! single-instance lock and waits for SIGTERM/SIGINT to broadcast a shutdown.
//! Per-trace pipelines, encoding, and the iceoryx2 IPC bring-up land in later
//! phases; see `docs/data-daemon-rewrite.md`.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::config::env::RuntimeEnv;
use crate::config::profile::{ProfileError, ProfileManager};
use crate::config::{resolve_effective_config, DaemonConfig, DEFAULT_PROFILE_NAME};
use crate::lifecycle::daemonize::{daemonize, DaemonizeOutcome, Readiness, ReadinessReporter};
use crate::lifecycle::pidfile::{PidFile, PidFileError};
use crate::lifecycle::recovery::{cleanup_stale_ipc, reclaim_stale_pid_file, PidReclaim};
use crate::lifecycle::signals::{install_shutdown_handler, ShutdownSignal};
use crate::state::SqliteStateStore;

/// Run the launch command.
pub fn run(profile: Option<String>, background: bool, debug: bool) -> Result<()> {
    let runtime_env = RuntimeEnv::from_env();
    let profiles = ProfileManager::new();

    // Mirrors `run_launch`: an explicitly named profile must exist on disk.
    if let Some(name) = &profile {
        if let Err(error) = profiles.get_profile(Some(name)) {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }

    // Ensure `default_profile.yaml` exists, mirroring
    // `runtime.py::_resolve_configuration`.
    if let Err(error) = ensure_default_profile_exists(&profiles) {
        eprintln!("Failed to create default profile '{DEFAULT_PROFILE_NAME}': {error}");
        std::process::exit(1);
    }

    let selected_profile = profile
        .or_else(|| runtime_env.profile.clone())
        .unwrap_or_else(|| DEFAULT_PROFILE_NAME.to_string());

    let config = match resolve_effective_config(&profiles, Some(&selected_profile), None) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    let effective_debug = debug || runtime_env.debug;

    if background {
        // Tracing is initialised inside `run_daemon` *after* the fork+stream
        // redirect so the subscriber writes to a real destination instead of
        // /dev/null. The original parent doesn't need tracing — it only
        // prints status.
        let log_path = log_path_for(&runtime_env);
        match daemonize().context("failed to daemonize")? {
            DaemonizeOutcome::Parent(reader) => handle_parent_readiness(reader),
            DaemonizeOutcome::Child(reporter) => run_daemon(
                runtime_env,
                config,
                effective_debug,
                Some(reporter),
                Some(log_path),
            ),
        }
    } else {
        print_preflight(&runtime_env, &config, &selected_profile, background);
        run_daemon(runtime_env, config, effective_debug, None, None)
    }
}

/// Handle the original-caller branch of `daemonize`: block on the readiness
/// pipe, then propagate the grandchild's startup status to the user's shell.
fn handle_parent_readiness(reader: crate::lifecycle::daemonize::ReadinessReader) -> Result<()> {
    match reader.read().context("failed to read daemon readiness")? {
        Readiness::Ready(pid) => {
            println!("{pid}");
            Ok(())
        }
        Readiness::Failed(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
        Readiness::Disconnected => {
            eprintln!("Daemon failed to start (no status reported)");
            std::process::exit(1);
        }
    }
}

/// Run the daemon main loop until a shutdown signal arrives.
///
/// `reporter` is `Some` in background mode and must receive a single ready or
/// fail message before the original caller unblocks. `log_file` is `Some` in
/// background mode and points at the file the grandchild routes tracing to,
/// because its stderr has already been redirected to /dev/null.
fn run_daemon(
    runtime_env: RuntimeEnv,
    config: DaemonConfig,
    debug: bool,
    reporter: Option<ReadinessReporter>,
    log_file: Option<PathBuf>,
) -> Result<()> {
    if let Err(error) = init_tracing(debug, log_file.as_deref()) {
        let message = format!("failed to initialise logging: {error}");
        report_failure(reporter, &message);
        return Err(error.context("failed to initialise logging"));
    }
    if debug {
        tracing::debug!(?config, "effective configuration resolved");
    }

    // Sweep a stale PID file *before* acquire so the next `status` command (or
    // diagnostics that read the file without taking the flock) doesn't report
    // a misleading "running" against a dead PID in the window between SIGKILL
    // and the new daemon's PID being written. `PidFile::acquire` would itself
    // recover via flock + truncate even without this — but doing it eagerly
    // keeps the on-disk PID file consistent for everyone, not just the
    // launcher.
    match reclaim_stale_pid_file(&runtime_env.pid_path).unwrap_or(PidReclaim::Absent) {
        PidReclaim::RemovedStale(prev) => {
            tracing::info!(previous_pid = ?prev, "removed stale pid file from prior unclean exit");
        }
        PidReclaim::StillRunning(_) | PidReclaim::Absent => {}
    }
    match cleanup_stale_ipc() {
        Ok(0) => {}
        Ok(cleaned) => tracing::info!(count = cleaned, "cleaned stale ipc artefacts"),
        Err(error) => tracing::warn!(%error, "failed to clean stale ipc artefacts"),
    }

    // Acquire the single-instance PID file *before* starting the Tokio runtime
    // so a duplicate-launch error is reported promptly and doesn't leak a
    // half-built runtime.
    let pid_file = match PidFile::acquire(&runtime_env.pid_path) {
        Ok(pid_file) => pid_file,
        Err(PidFileError::AlreadyRunning(pid)) => {
            let message = format!("Daemon already running (pid={pid})");
            tracing::error!("{message}");
            report_failure(reporter, &message);
            std::process::exit(1);
        }
        Err(PidFileError::Io(error)) => {
            let context = format!("failed to acquire {}", runtime_env.pid_path.display());
            tracing::error!(%error, "{context}");
            report_failure(reporter, &format!("{context}: {error}"));
            return Err(anyhow::Error::from(error).context(context));
        }
    };

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let message = format!("failed to build tokio runtime: {error}");
            tracing::error!("{message}");
            report_failure(reporter, &message);
            return Err(anyhow::Error::from(error).context("failed to build tokio runtime"));
        }
    };

    // Signal readiness to the launcher *before* the main loop blocks so the
    // user's shell prompt returns as soon as the daemon is actually up.
    if let Some(reporter) = reporter {
        if let Err(error) = reporter.ready(std::process::id()) {
            tracing::warn!(%error, "failed to report readiness to launcher (continuing)");
        }
    }

    let db_path = runtime_env.db_path.clone();
    let outcome = runtime.block_on(async move {
        // Phase 3: open the SQLite state store so the schema exists on first
        // launch. Phase 4 will wire it into the dispatcher and coordinators.
        let state_store = SqliteStateStore::open(&db_path)
            .await
            .with_context(|| format!("failed to open state store at {}", db_path.display()))?;
        tracing::info!(path = %db_path.display(), "state store ready");

        // Run the wait loop in a nested block so the state store can be
        // closed in both the success and error paths before the runtime
        // tears connections down.
        let result: Result<ShutdownSignal> = async {
            // `_shutdown` is the broadcast `Sender` handle. Phase 4 will
            // clone it into each coordinator (dispatcher, registration,
            // upload, etc.) so they can subscribe their own receivers; for
            // now only the primary receiver returned alongside it is
            // awaited.
            let (_shutdown, mut shutdown_rx) =
                install_shutdown_handler().context("failed to install shutdown handlers")?;
            tracing::info!("daemon ready; awaiting shutdown signal");
            // Phase 4 will replace this trivial await with the real
            // dispatcher, per-trace actors, registration/upload
            // coordinators, etc.
            let signal = shutdown_rx.recv().await;
            Ok(signal.ok().unwrap_or(ShutdownSignal::Sigterm))
        }
        .await;

        state_store.close().await;
        result
    });

    drop(pid_file);
    runtime.shutdown_background();

    match outcome {
        Ok(signal) => tracing::info!(?signal, "daemon stopped"),
        Err(error) => {
            tracing::error!(%error, "daemon main loop returned error");
            return Err(error);
        }
    }

    Ok(())
}

fn report_failure(reporter: Option<ReadinessReporter>, message: &str) {
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
fn log_path_for(runtime_env: &RuntimeEnv) -> PathBuf {
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
fn init_tracing(debug: bool, log_file: Option<&Path>) -> Result<()> {
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
        let _ = builder.try_init();
    }
    Ok(())
}

fn ensure_default_profile_exists(profiles: &ProfileManager) -> Result<(), ProfileError> {
    match profiles.create_profile(DEFAULT_PROFILE_NAME) {
        Ok(()) | Err(ProfileError::AlreadyExists(_)) => Ok(()),
        Err(error) => Err(error),
    }
}

fn print_preflight(
    runtime_env: &RuntimeEnv,
    config: &DaemonConfig,
    selected_profile: &str,
    background: bool,
) {
    let offline = config.offline.unwrap_or(false);

    println!("Daemon launch prepared.");
    println!("  profile:            {selected_profile}");
    println!("  offline:            {offline}");
    println!("  background:         {background}");
    println!("  pid file:           {}", runtime_env.pid_path.display());
    println!("  database:           {}", runtime_env.db_path.display());
    println!(
        "  recordings root:    {}",
        runtime_env.recordings_root.display()
    );
    println!("  api url:            {}", runtime_env.api_url);
    println!("  manage pid:         {}", runtime_env.manage_pid);
    println!("  max spooled chunks: {}", runtime_env.max_spooled_chunks);
}
