//! Cloud-coordinator wiring for the daemon main loop.
//!
//! Builds the shared `ApiClient` and spawns the cloud-side coordinators
//! (registration, upload, status, progress, recording notifiers, connection
//! monitor, and org watcher), bundling their handles so `cli::launch` can join
//! them in a defined order at shutdown.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::api::auth::FileAuthProvider;
use crate::api::client::{ApiClient, ApiClientOptions};
use crate::cloud::{
    spawn_org_watcher, spawn_progress_reporter, spawn_recording_cancel_notifier,
    spawn_recording_start_notifier, spawn_recording_stop_notifier, spawn_registration,
    spawn_status_updater, spawn_uploader, OrgWatcherHandle, StatusUpdate,
};
use crate::connection::spawn_connection_monitor;
use crate::state::{EventBus, SqliteStateStore, TraceWriteHandle};

/// Bundle of handles for the cloud coordinators.
pub(crate) struct CloudHandles {
    connection: crate::connection::MonitorHandle,
    org_watcher: OrgWatcherHandle,
    registration: crate::cloud::RegistrationHandle,
    uploader: crate::cloud::UploaderHandle,
    status: crate::cloud::StatusUpdaterHandle,
    progress: crate::cloud::ProgressReporterHandle,
    recording_start: crate::cloud::NotifierHandle,
    recording_stop: crate::cloud::NotifierHandle,
    recording_cancel: crate::cloud::NotifierHandle,
}

impl CloudHandles {
    pub(crate) async fn join_all(self) {
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

pub(crate) fn build_api_client(api_url: &str, config_path: &Path) -> Result<Arc<ApiClient>> {
    let auth = Arc::new(
        FileAuthProvider::new(config_path, api_url.to_string())
            .context("failed to construct auth provider")?,
    );
    let options = ApiClientOptions::new(api_url.to_string());
    let client = ApiClient::new(options, auth).context("failed to build api client")?;
    Ok(Arc::new(client))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_cloud_coordinators(
    state_store: SqliteStateStore,
    trace_writer: TraceWriteHandle,
    event_bus: EventBus,
    client: Arc<ApiClient>,
    recordings_root: Arc<PathBuf>,
    config_path: PathBuf,
    fallback_org_id: Option<String>,
    shutdown_tx: crate::lifecycle::shutdown::ShutdownBroadcaster,
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
        trace_writer,
        event_bus.clone(),
        Arc::clone(&client),
        Arc::clone(&recordings_root),
        org_rx.clone(),
        status_tx.clone(),
        shutdown_tx.subscribe(),
    );
    // Drop the local sender so the only remaining sender is the uploader; once
    // the uploader exits the inbox closes, which (alongside the shutdown
    // broadcast) lets the status task exit cleanly without a dangling sender
    // keeping the channel open.
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
