//! `launch` subcommand handler.
//!
//! Resolves configuration, then wires the daemon lifecycle: the PID-file
//! single-instance lock, signal handling, and optional background
//! detachment. The daemon main loop brings up the iceoryx2 IPC listener, the
//! per-trace pipeline, and the cloud coordinators, then waits for
//! SIGTERM/SIGINT to broadcast a graceful shutdown.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::api::auth::FileAuthProvider;
use crate::api::client::{ApiClient, ApiClientOptions};
use crate::cloud::{
    read_org_id_from_config, spawn_org_watcher, spawn_progress_reporter,
    spawn_recording_cancel_notifier, spawn_recording_start_notifier, spawn_recording_stop_notifier,
    spawn_registration, spawn_status_updater, spawn_uploader, OrgWatcherHandle, StatusUpdate,
};
use crate::config::env::RuntimeEnv;
use crate::config::profile::{ProfileError, ProfileManager};
use crate::config::{resolve_effective_config, DaemonConfig, DEFAULT_PROFILE_NAME};
use crate::connection::{spawn_connection_monitor, spawn_wakelock};
use crate::encoding::video_encoder::VideoEncoder;
use crate::ipc::listener;
use crate::ipc::node::IpcTransport;
use crate::lifecycle::daemonize::{daemonize, DaemonizeOutcome, Readiness, ReadinessReporter};
use crate::lifecycle::pidfile::{PidFile, PidFileError};
use crate::lifecycle::recovery::{
    cleanup_stale_ipc, reclaim_stale_pid_file, sweep_partial_recordings, PidReclaim,
};
use crate::lifecycle::signals::{install_shutdown_handler, ShutdownSignal};
use crate::pipeline::dispatcher::{self, DispatcherContext};
use crate::pipeline::trace_actor::TraceActorContext;
use crate::state::{EventBus, SqliteStateStore, StateStore};
use crate::storage::budget::{StorageBudget, StoragePolicy};

/// Upper bound on how long we wait for the signal-capture task after the
/// listener returns. In normal operation it has already completed; the
/// timeout exists so a future bug that lets the listener exit without a
/// shutdown signal degrades to a `?signal=sigterm` log rather than a hang.
const SIGNAL_CAPTURE_TIMEOUT: Duration = Duration::from_secs(1);

/// `last_updated` age (in seconds) below which a `writing` /
/// `pending_metadata` / `initializing` trace row is left alone by the
/// startup-time sweep. Comfortably larger than the trace_actor's debounce
/// flush interval, so a row a current daemon has just begun writing isn't
/// caught.
const STALE_WRITE_THRESHOLD_SECS: i64 = 30;

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
    let cleaned = cleanup_stale_ipc();
    if cleaned > 0 {
        tracing::info!(count = cleaned, "cleaned stale ipc artefacts");
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
    let recordings_root = runtime_env.recordings_root.clone();
    let storage_policy = StoragePolicy {
        storage_limit_bytes: config
            .storage_limit
            .and_then(|value| u64::try_from(value).ok()),
        ..StoragePolicy::default()
    };
    let api_url = runtime_env.api_url.clone();
    let config_for_runtime = config.clone();
    let outcome = runtime.block_on(async move {
        let state_store = SqliteStateStore::open(&db_path)
            .await
            .with_context(|| format!("failed to open state store at {}", db_path.display()))?;
        tracing::info!(path = %db_path.display(), "state store ready");
        // Re-arm rows stuck in transient `registering` / `uploading` states
        // from a previous unclean exit — the claim/drain queries that drive
        // the coordinators only scan terminal-or-pending rows, so without
        // this sweep a SIGKILL mid-upload would leak traces.
        match state_store.reset_stale_pipeline_states().await {
            Ok(0) => {}
            Ok(count) => tracing::info!(count, "re-armed stale pipeline rows from prior run"),
            Err(error) => {
                tracing::warn!(%error, "failed to reset stale pipeline states (continuing)")
            }
        }
        // Burn trace rows the previous daemon left mid-write.
        // Those rows can never reach `written` (the actor that owned them
        // is gone) and would otherwise pin their recording in the
        // "all traces written" gate the progress reporter waits on. The
        // 30 s threshold gives a future daemon launched on top of an
        // earlier orderly-shutdown's tail a chance to recover; in
        // practice every row caught by this sweep is hours stale.
        match state_store
            .mark_stale_writing_traces_failed(STALE_WRITE_THRESHOLD_SECS)
            .await
        {
            Ok(0) => {}
            Ok(count) => tracing::info!(count, "marked stale writing traces as failed"),
            Err(error) => {
                tracing::warn!(%error, "failed to mark stale writing traces failed (continuing)")
            }
        }
        // Purge any recording the prior daemon left mid-write. Producer-side
        // chunk spooling means we may have stranded NUT chunks, half-encoded
        // segments, and partial concat outputs on disk; mid-encode resume is
        // intentionally out of scope so anything not in the `written`
        // terminal state at startup is removed and the recording marked
        // cancelled.
        match sweep_partial_recordings(&state_store, &recordings_root).await {
            Ok(report) if report == Default::default() => {}
            Ok(report) => tracing::info!(
                purged = report.recordings_purged,
                preserved = report.recordings_preserved,
                "partial-recording sweep completed",
            ),
            Err(error) => {
                tracing::warn!(%error, "partial-recording sweep failed (continuing)")
            }
        }
        let storage_budget = Arc::new(StorageBudget::new(&recordings_root, storage_policy));
        let event_bus = EventBus::new();
        let actor_context = Arc::new(
            TraceActorContext::new(
                recordings_root.clone(),
                storage_budget,
                VideoEncoder::new(),
            )
            .with_event_bus(event_bus.clone()),
        );

        // Run the wait loop in a nested block so the state store can be
        // closed in both the success and error paths before the runtime
        // tears connections down.
        let result: Result<ShutdownSignal> = async {
            let (shutdown_tx, shutdown_rx) =
                install_shutdown_handler().context("failed to install shutdown handlers")?;

            // Bring up iceoryx2 *before* the dispatcher: a failure here is
            // user-visible (the daemon can't accept IPC at all) and must
            // unwind cleanly through the same path as a normal shutdown.
            let transport =
                IpcTransport::bring_up().context("failed to bring up iceoryx2 transport")?;

            // Resolve the initial org_id from the local SDK-managed config,
            // falling back to the daemon profile (NCD_CURRENT_ORG_ID or the
            // YAML profile override). This is only the seed value: the cloud
            // coordinators spawn a watcher over `config_path` and read the
            // *current* org live, so an org selected after launch is picked up
            // without restarting the daemon. The profile override remains the
            // documented escape hatch for tests (the file watcher's fallback).
            let config_path = dirs::home_dir()
                .map(|home| home.join(".neuracore").join("config.json"))
                .unwrap_or_else(|| std::path::PathBuf::from(".neuracore/config.json"));
            let org_id = read_org_id_from_config(&config_path)
                .or_else(|| config_for_runtime.current_org_id.clone());

            // Spawn the cloud-side coordinators *before* the dispatcher so
            // they have an active subscription to the event bus by the time
            // any `TraceWritten` / `RecordingStopped` fires. A late
            // subscriber sees no replay (broadcast channels don't replay),
            // so a coordinator that races behind a fast end-to-end trace
            // would otherwise miss its trigger event and have to wait for
            // the next periodic tick. Order is also load-bearing for
            // ordered shutdown: dropping the dispatcher first guarantees
            // no new `TraceWritten` lands while the coordinators drain.
            let cloud_handles = if config_for_runtime.offline.unwrap_or(false) {
                tracing::info!("offline mode — skipping cloud coordinator spawn");
                None
            } else {
                match build_api_client(&api_url, &config_path) {
                    Ok(api_client) => Some(spawn_cloud_coordinators(
                        state_store.clone(),
                        event_bus.clone(),
                        api_client,
                        Arc::new(recordings_root.clone()),
                        config_path.clone(),
                        org_id.clone(),
                        shutdown_tx.clone(),
                    )),
                    Err(error) => {
                        tracing::warn!(%error, "failed to build API client; cloud uploads disabled");
                        None
                    }
                }
            };

            // Hold a wakelock while at least one trace is queued
            // for upload. Spawned regardless of `offline` so a profile
            // configured "online but flaky network" still keeps the host
            // awake when traces queue up locally; the wakelock task does
            // nothing on hosts without `systemd-inhibit`.
            let wakelock_handle = config_for_runtime
                .keep_wakelock_while_upload
                .unwrap_or(false)
                .then(|| {
                    tracing::info!("wakelock-while-upload enabled");
                    spawn_wakelock(event_bus.clone(), shutdown_tx.subscribe())
                });

            let dispatcher_context = DispatcherContext {
                event_bus: Some(event_bus.clone()),
            };
            let (dispatcher_tx, dispatcher_handle) = dispatcher::spawn_with_context(
                state_store.clone(),
                Arc::clone(&actor_context),
                dispatcher_context,
                shutdown_tx.subscribe(),
            );

            // Capture the actual shutdown signal in a spawned task so we
            // can log which signal triggered the exit *after* the listener
            // returns. The listener itself cannot be `tokio::spawn`'d —
            // iceoryx2 subscriber ports are `!Send` — so we run it inline
            // and let the dispatcher + signal-capture tasks ride the
            // multi-thread runtime in parallel.
            let mut signal_rx = shutdown_tx.subscribe();
            let signal_task = tokio::spawn(async move {
                signal_rx
                    .recv()
                    .await
                    .ok()
                    .unwrap_or(ShutdownSignal::Sigterm)
            });
            // Drain the primary handler-installed receiver so it doesn't
            // accumulate broadcasts behind our back; we no longer need it.
            drop(shutdown_rx);

            tracing::info!(?org_id, "daemon ready; awaiting shutdown signal");
            listener::run(transport, dispatcher_tx.clone(), shutdown_tx.subscribe()).await;

            // Ordered shutdown — by the time `listener::run` has returned
            // the iceoryx2 node has already been dropped (it lived inside
            // the listener task's frame):
            //   1. drop our local dispatcher sender so the dispatcher
            //      inbox closes,
            //   2. dispatcher drains, clears the routing map, and the
            //      per-trace actors observe EOF and exit,
            //   3. wait for the cloud coordinators to finish their
            //      in-flight requests,
            //   4. read the captured shutdown signal for the log line.
            drop(dispatcher_tx);
            dispatcher_handle.shutdown().await;
            if let Some(handles) = cloud_handles {
                handles.join_all().await;
            }
            if let Some(handle) = wakelock_handle {
                handle.join().await;
            }
            // In normal operation the listener returns *because* a shutdown
            // signal fired, so `signal_task` is already complete. Bound the
            // wait so a future code path that lets the listener exit
            // independently (panic, dispatcher dropped) can't hang the
            // daemon's exit on a signal that never arrives.
            let signal = match tokio::time::timeout(SIGNAL_CAPTURE_TIMEOUT, signal_task).await {
                Ok(Ok(captured)) => captured,
                Ok(Err(error)) => {
                    tracing::warn!(?error, "signal-capture task join failed");
                    ShutdownSignal::Sigterm
                }
                Err(_) => {
                    tracing::debug!(
                        "listener exited without a shutdown signal; defaulting to sigterm"
                    );
                    ShutdownSignal::Sigterm
                }
            };

            Ok(signal)
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
        // Write to stderr so the parent's stdout=DEVNULL plumbing in
        // background mode does not silently swallow structured log output.
        let _ = builder.with_writer(std::io::stderr).try_init();
    }
    Ok(())
}

fn ensure_default_profile_exists(profiles: &ProfileManager) -> Result<(), ProfileError> {
    match profiles.create_profile(DEFAULT_PROFILE_NAME) {
        Ok(()) | Err(ProfileError::AlreadyExists(_)) => Ok(()),
        Err(error) => Err(error),
    }
}

/// Bundle of handles for the cloud coordinators.
struct CloudHandles {
    connection: crate::connection::MonitorHandle,
    org_watcher: OrgWatcherHandle,
    registration: crate::cloud::RegistrationCoordinatorHandle,
    uploader: crate::cloud::UploaderHandle,
    status: crate::cloud::StatusUpdaterHandle,
    progress: crate::cloud::ProgressReporterHandle,
    recording_start: crate::cloud::RecordingStartNotifierHandle,
    recording_stop: crate::cloud::RecordingStopNotifierHandle,
    recording_cancel: crate::cloud::RecordingCancelNotifierHandle,
}

impl CloudHandles {
    async fn join_all(self) {
        // Connection monitor drops first because its tick is bounded by the
        // health-check interval; the others have either bus subscriptions
        // or pending requests that may need a moment to wrap up after the
        // shutdown signal fires.
        self.connection.join().await;
        self.org_watcher.join().await;
        self.registration.join().await;
        self.uploader.join().await;
        self.status.join().await;
        self.progress.join().await;
        self.recording_start.join().await;
        self.recording_stop.join().await;
        self.recording_cancel.join().await;
    }
}

fn build_api_client(api_url: &str, config_path: &Path) -> Result<Arc<ApiClient>> {
    let auth = Arc::new(
        FileAuthProvider::new(config_path, api_url.to_string())
            .context("failed to construct auth provider")?,
    );
    let options = ApiClientOptions::new(api_url.to_string());
    let client = ApiClient::new(options, auth).context("failed to build api client")?;
    Ok(Arc::new(client))
}

fn spawn_cloud_coordinators(
    state_store: SqliteStateStore,
    event_bus: EventBus,
    client: Arc<ApiClient>,
    recordings_root: Arc<PathBuf>,
    config_path: PathBuf,
    fallback_org_id: Option<String>,
    shutdown_tx: crate::lifecycle::signals::ShutdownHandle,
) -> CloudHandles {
    let (status_tx, status_rx) = tokio::sync::mpsc::unbounded_channel::<StatusUpdate>();
    // Watch the SDK config for the current org; every coordinator reads the
    // live value at the moment it POSTs rather than a value frozen onto the
    // recording row at creation time.
    let (org_rx, org_watcher) =
        spawn_org_watcher(config_path, fallback_org_id, shutdown_tx.subscribe());
    let connection = spawn_connection_monitor(
        Arc::clone(&client),
        event_bus.clone(),
        shutdown_tx.subscribe(),
    );
    let registration = spawn_registration(
        state_store.clone(),
        event_bus.clone(),
        Arc::clone(&client),
        org_rx.clone(),
        shutdown_tx.subscribe(),
    );
    let uploader = spawn_uploader(
        state_store.clone(),
        event_bus.clone(),
        Arc::clone(&client),
        Arc::clone(&recordings_root),
        org_rx.clone(),
        status_tx.clone(),
        shutdown_tx.subscribe(),
    );
    // Drop the local sender so the channel closes as soon as the uploader
    // exits — the status task uses the close to break out of its select!
    // loop on shutdown.
    drop(status_tx);
    let status = spawn_status_updater(
        state_store.clone(),
        Arc::clone(&client),
        org_rx.clone(),
        status_rx,
        shutdown_tx.subscribe(),
    );
    let progress = spawn_progress_reporter(
        state_store.clone(),
        Arc::clone(&client),
        org_rx.clone(),
        shutdown_tx.subscribe(),
    );
    let recording_start = spawn_recording_start_notifier(
        state_store.clone(),
        event_bus.clone(),
        Arc::clone(&client),
        org_rx.clone(),
        shutdown_tx.subscribe(),
    );
    let recording_stop = spawn_recording_stop_notifier(
        state_store.clone(),
        event_bus.clone(),
        Arc::clone(&client),
        org_rx.clone(),
        shutdown_tx.subscribe(),
    );
    let recording_cancel = spawn_recording_cancel_notifier(
        state_store,
        event_bus,
        Arc::clone(&client),
        org_rx,
        shutdown_tx.subscribe(),
    );

    CloudHandles {
        connection,
        org_watcher,
        registration,
        uploader,
        status,
        progress,
        recording_start,
        recording_stop,
        recording_cancel,
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
