//! `launch` subcommand handler.
//!
//! Resolves configuration, then wires the daemon lifecycle: the PID-file
//! single-instance lock, signal handling, and optional background
//! detachment. The daemon main loop brings up the iceoryx2 IPC listener, the
//! per-trace pipeline, and the cloud coordinators, then waits for
//! SIGTERM/SIGINT to broadcast a graceful shutdown.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::{mpsc, watch};

use crate::cli::coordinators::{build_api_client, spawn_cloud_coordinators};
use crate::cli::launch_logging::{init_tracing, log_path_for, report_failure};
use crate::cloud::{
    read_org_id_from_config, spawn_config_watcher, spawn_recording_reaper, ConfigRefreshRequest,
};
use crate::config::env::RuntimeEnv;
use crate::config::profile::{ProfileError, ProfileManager};
use crate::config::{resolve_effective_config, DaemonConfig, DEFAULT_PROFILE_NAME};
use crate::connection::spawn_wakelock;
use crate::encoding::video_encoder::VideoEncoder;
use crate::ipc::listener;
use crate::ipc::node::IpcTransport;
use crate::lifecycle::daemonize::{daemonize, DaemonizeOutcome, Readiness, ReadinessReporter};
use crate::lifecycle::pidfile::{PidFile, PidFileError};
use crate::lifecycle::recovery::{cleanup_stale_ipc, reclaim_stale_pid_file, PidReclaim};
use crate::lifecycle::shutdown::{install_shutdown_handler, ShutdownSignal};
use crate::pipeline::dispatcher::{self, DispatcherContext};
use crate::pipeline::trace_actor::TraceActorContext;
use crate::state::{EventBus, SqliteStateStore};
use crate::storage::budget::{StorageBudget, StoragePolicy};

/// Upper bound on how long we wait for the signal-capture task after the
/// listener returns. In normal operation it has already completed; the
/// timeout exists so a future bug that lets the listener exit without a
/// shutdown signal degrades to a `?signal=sigterm` log rather than a hang.
const SIGNAL_CAPTURE_TIMEOUT: Duration = Duration::from_secs(1);

/// Capacity of the dispatcher → config-watcher refresh-request channel. Refresh
/// requests are rare and drained promptly; the small buffer just avoids blocking
/// the dispatcher on a burst of back-to-back commands.
const CONFIG_REFRESH_CHANNEL_CAPACITY: usize = 8;

/// Run the launch command.
pub fn run(profile: Option<String>, background: bool, debug: bool) -> Result<()> {
    let runtime_env = RuntimeEnv::from_env();
    let profiles = ProfileManager::new();

    // Ensure the default profile exists before resolving config. A missing
    // *named* profile needs no separate existence pre-check: the
    // `resolve_effective_config` call below surfaces it as
    // `ProfileError::NotFound`, handled in the same match arm.
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
                selected_profile,
                effective_debug,
                Some(reporter),
                Some(log_path),
            ),
        }
    } else {
        print_preflight(&runtime_env, &config, &selected_profile);
        run_daemon(
            runtime_env,
            config,
            selected_profile,
            effective_debug,
            None,
            None,
        )
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
    profile: String,
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
    match reclaim_stale_pid_file(&runtime_env.pid_path) {
        Ok(PidReclaim::RemovedStale(prev)) => {
            tracing::info!(previous_pid = ?prev, "removed stale pid file from prior unclean exit");
        }
        Ok(PidReclaim::StillRunning(_) | PidReclaim::Absent) => {}
        Err(error) => {
            // Non-fatal: `PidFile::acquire` below still recovers via flock +
            // truncate. But a failure here (e.g. a permissions problem on the
            // pid dir) is worth surfacing rather than silently discarding.
            tracing::warn!(
                %error,
                path = %runtime_env.pid_path.display(),
                "failed to reclaim stale pid file at startup; relying on acquire's flock recovery"
            );
        }
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

    // Verify ffmpeg is present and supports the options the encoder depends on
    // *before* standing up the runtime — an incompatible build (e.g. one that
    // lacks `-vsync passthrough`, or is missing libx264) otherwise fails every
    // video encode silently at recording time, marking traces `failed`. Mirrors
    // the fail-fast PID-file acquisition above. Reused as the pipeline's encoder
    // so the probe and the real encodes share one configured binary.
    let video_encoder = VideoEncoder::new();
    match video_encoder.preflight() {
        Ok(version) => tracing::info!(ffmpeg_version = %version, "ffmpeg preflight passed"),
        Err(error) => {
            let message = format!("ffmpeg preflight failed: {error}");
            tracing::error!("{message}");
            report_failure(reporter, &message);
            return Err(anyhow::Error::new(error).context("ffmpeg preflight failed"));
        }
    }

    let mut runtime_builder = tokio::runtime::Builder::new_multi_thread();
    runtime_builder.enable_all();
    // Honour the configured worker-thread count (`NCD_NUM_THREADS` / the YAML
    // `num_threads` / `--num-threads`); a non-positive value falls back to
    // tokio's default (one worker per core).
    match config.num_threads {
        Some(threads) if threads > 0 => {
            runtime_builder.worker_threads(threads as usize);
        }
        Some(threads) => {
            tracing::warn!(
                num_threads = threads,
                "ignoring non-positive num_threads; using default"
            );
        }
        None => {}
    }
    let runtime = match runtime_builder.build() {
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
    // The recordings root is shared with the producer, which lives in a
    // *separate* process and resolves it from `NEURACORE_DAEMON_RECORDINGS_ROOT`
    // (or the db-dir sibling) — it never reads the daemon profile. So a
    // profile `path_to_store_record` that disagrees with the effective root
    // cannot be silently honoured here without stranding the producer's spooled
    // video under a path the daemon never scans. Surface the mismatch loudly
    // instead and point the operator at the knob that actually coordinates both
    // processes.
    if let Some(configured) = config
        .path_to_store_record
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        if Path::new(configured) != recordings_root {
            tracing::warn!(
                configured,
                effective = %recordings_root.display(),
                "profile `path_to_store_record` is ignored; the recordings root is set by \
                 NEURACORE_DAEMON_RECORDINGS_ROOT (read by both daemon and producer). \
                 Set that env var to relocate recordings."
            );
        }
    }
    let storage_policy = StoragePolicy {
        storage_limit_bytes: config
            .storage_limit
            .and_then(|value| u64::try_from(value).ok()),
        ..StoragePolicy::default()
    };
    let api_url = runtime_env.api_url.clone();
    let config_for_runtime = config;
    let outcome = runtime.block_on(async move {
        let state_store = SqliteStateStore::open(&db_path)
            .await
            .with_context(|| format!("failed to open state store at {}", db_path.display()))?;
        tracing::info!(path = %db_path.display(), "state store ready");
        crate::lifecycle::recovery::run_startup_sweeps(&state_store, &recordings_root).await;
        let storage_budget = Arc::new(StorageBudget::new(&recordings_root, storage_policy));
        // Reconcile the storage budget (directory scan + `statvfs`) on a
        // background interval instead of on the trace actors' per-frame
        // `check` path: a raw `statvfs` on the shared spool periodically blocks
        // for hundreds of ms behind an ext4 journal commit, and at the frame
        // rate that stall would back-pressure the whole dispatcher → IPC drain.
        // `spawn_blocking` keeps the scan/syscall off the async runtime threads.
        {
            let refresh_budget = storage_budget.clone();
            let refresh_interval = refresh_budget.policy().refresh_interval;
            if !refresh_interval.is_zero() {
                tokio::spawn(async move {
                    loop {
                        tokio::time::sleep(refresh_interval).await;
                        let budget = refresh_budget.clone();
                        if tokio::task::spawn_blocking(move || budget.refresh())
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                });
            }
        }
        let event_bus = EventBus::new();
        // Write-behind for the per-trace actors' high-frequency progress /
        // status / finalise writes: coalesced per trace and flushed in batches
        // off the actors' hot path so they never contend on the store's single
        // write mutex (see `state::trace_event_database_writer`). Drained + flushed at shutdown
        // below, after the dispatcher (and so every actor) has stopped.
        let (trace_write_handle, trace_writer) =
            crate::state::trace_event_database_writer::spawn(Arc::new(state_store.clone()));
        // Write-behind for the per-trace `trace.json` appends: the blocking JSON
        // `write()` periodically stalls behind an ext4 journal commit on the
        // shared spool, so running it on the actor's hot path back-pressures the
        // dispatcher and IPC listener and spikes producer `log_*` latency. The
        // dedicated thread keeps that disk I/O off the drain path (see
        // `pipeline::json_writer`). The join handle is dropped — the thread exits
        // once the dispatcher and every actor (the last `JsonWriteHandle` holders)
        // are gone at shutdown.
        let (json_write_handle, _json_writer_owner) = crate::pipeline::json_writer::spawn();

        // In-memory daemon config: seed the watch channel with the
        // launch-resolved effective config so its value is available before the
        // watcher's first tick, then let the config watcher (spawned below, once
        // the shutdown broadcaster exists) refresh it on an interval and on
        // demand. The trace actors and registration coordinator read the codec
        // from this channel instead of re-parsing the profile YAML per trace.
        let (config_tx, config_rx) = watch::channel(config_for_runtime.clone());
        let (config_refresh_tx, config_refresh_rx) =
            mpsc::channel::<ConfigRefreshRequest>(CONFIG_REFRESH_CHANNEL_CAPACITY);

        let actor_context = Arc::new(
            TraceActorContext::new(
                recordings_root.clone(),
                storage_budget,
                video_encoder,
                trace_write_handle.clone(),
                json_write_handle,
            )
            .with_event_bus(event_bus.clone())
            .with_config_rx(config_rx.clone()),
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

            // Spawn the daemon-config watcher. Unconditional — offline mode
            // skips the cloud coordinators, but the per-trace actors still need
            // the live codec, so the watcher must run regardless. It owns
            // `config_tx` / `config_refresh_rx`; the seeded `config_rx` is
            // already wired into the actor context and, below, the registration
            // coordinator, and `config_refresh_tx` goes to the dispatcher for
            // the `RefreshConfig` command path.
            let config_watcher = spawn_config_watcher(
                Some(profile.clone()),
                None,
                config_tx,
                config_refresh_rx,
                shutdown_tx.subscribe(),
            );

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
                        trace_write_handle.clone(),
                        event_bus.clone(),
                        api_client,
                        Arc::new(recordings_root.clone()),
                        config_path.clone(),
                        org_id.clone(),
                        config_rx.clone(),
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
                config_refresh_tx: Some(config_refresh_tx),
            };
            let (dispatcher_tx, dispatcher_handle) = dispatcher::spawn_with_context(
                state_store.clone(),
                Arc::clone(&actor_context),
                dispatcher_context,
                shutdown_tx.subscribe(),
            );

            // Reclaim fully-uploaded recordings' files + rows. Spawned
            // regardless of `offline` so a daemon restarted offline still
            // reaps recordings that completed in a prior online session; it
            // only ever acts on recordings the backend already holds in full.
            let reaper_handle = spawn_recording_reaper(
                state_store.clone(),
                Arc::new(recordings_root.clone()),
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
            listener::run(
                transport,
                dispatcher_tx.clone(),
                Arc::new(state_store.clone()),
                shutdown_tx.subscribe(),
            )
            .await;

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
            // Every per-trace actor has now exited, so no further trace writes
            // can be produced. Drain + flush the write-behind's final batch
            // before the store closes so finalise/failed states are durable.
            trace_writer.shutdown().await;
            reaper_handle.join().await;
            if let Some(handles) = cloud_handles {
                handles.join_all().await;
            }
            // The dispatcher (the last `config_refresh_tx` holder) is gone, so
            // the watcher's refresh branch has closed; it exits on the shutdown
            // broadcast.
            config_watcher.join().await;
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

fn ensure_default_profile_exists(profiles: &ProfileManager) -> Result<(), ProfileError> {
    match profiles.create_profile(DEFAULT_PROFILE_NAME) {
        Ok(()) | Err(ProfileError::AlreadyExists(_)) => Ok(()),
        Err(error) => Err(error),
    }
}

fn print_preflight(runtime_env: &RuntimeEnv, config: &DaemonConfig, selected_profile: &str) {
    let offline = config.offline.unwrap_or(false);

    // Only the foreground launch path prints the preflight.
    println!("Daemon launch prepared.");
    println!("  profile:            {selected_profile}");
    println!("  offline:            {offline}");
    println!("  pid file:           {}", runtime_env.pid_path.display());
    println!("  database:           {}", runtime_env.db_path.display());
    println!(
        "  recordings root:    {}",
        runtime_env.recordings_root.display()
    );
    println!("  api url:            {}", runtime_env.api_url);
}
