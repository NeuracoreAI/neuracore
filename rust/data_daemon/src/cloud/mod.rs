//! Cloud-side coordinators: batch registration, resumable uploads,
//! debounced status updates, and the periodic progress reporter.
//!
//! Every coordinator is spawned by the daemon launch routine and subscribes
//! to the broadcast event bus. The flow is:
//!
//! 1. Per-trace actor finishes writing → emits `TraceWritten`.
//! 2. Registration coordinator claims up to `registration::BATCH_SIZE` traces
//!    (or whatever is ready after `registration::MAX_WAIT`), batch-registers,
//!    persists the resumable session URIs, emits `ReadyForUpload`.
//! 3. Upload coordinator opens each on-disk artefact and PUTs it resumably,
//!    emitting `UploadProgress` and `UploadComplete` events as it goes.
//! 4. Status updater debounces `UploadProgress` / `UploadComplete` into
//!    batched backend updates.
//! 5. Progress reporter ticks every `progress::FAST_PROGRESS_TICK` and posts a
//!    per-recording bytes-uploaded snapshot until every trace lands in
//!    `progress_reported`.
//!
//! Each sub-module exposes a single `spawn_*` entry point so the launch
//! routine can drive ordered shutdown by dropping the handle.

pub mod cloud_files;
pub mod notifier;
pub mod org_watcher;
pub mod progress;
pub mod recording_cancel_notifier;
pub mod recording_reaper;
pub mod recording_start_notifier;
pub mod recording_stop_notifier;
pub mod registration;
pub mod status;
mod upload_transfer;
pub mod uploader;

#[allow(unused_imports)]
pub use cloud_files::{cloud_file_list, content_type_for, ContentKind};
pub use notifier::NotifierHandle;
pub use org_watcher::{spawn_org_watcher, OrgIdRx, OrgWatcherHandle};
pub use progress::{spawn_progress_reporter, ProgressReporterHandle};
pub use recording_cancel_notifier::spawn_recording_cancel_notifier;
pub use recording_reaper::spawn_recording_reaper;
pub use recording_start_notifier::spawn_recording_start_notifier;
pub use recording_stop_notifier::spawn_recording_stop_notifier;
pub use registration::{spawn_registration, RegistrationCoordinatorHandle};
pub use status::{spawn_status_updater, StatusUpdate, StatusUpdaterHandle};
pub use uploader::{spawn_uploader, UploaderHandle};

use std::path::Path;

use serde::Deserialize;

/// Read the `current_org_id` field from `~/.neuracore/config.json` (blocking).
///
/// Used for the one-shot resolutions at launch/spawn. Returns `None` when the
/// file is missing, malformed, or the field is unset. The daemon falls back to
/// `NCD_CURRENT_ORG_ID` (via the `DaemonConfig` resolved at launch) when this
/// returns `None`.
pub fn read_org_id_from_config(path: &Path) -> Option<String> {
    match std::fs::read(path) {
        Ok(bytes) => parse_org_id(&bytes, path),
        // Absent / unreadable — fall back silently (the org simply isn't set
        // yet). A *present but corrupt* file is surfaced by `parse_org_id`.
        Err(_) => None,
    }
}

/// Async counterpart of [`read_org_id_from_config`] for the org watcher's
/// periodic poll, so the re-read + parse runs off the runtime worker rather
/// than blocking it once per second for the daemon's whole life.
pub async fn read_org_id_from_config_async(path: &Path) -> Option<String> {
    match tokio::fs::read(path).await {
        Ok(bytes) => parse_org_id(&bytes, path),
        Err(_) => None,
    }
}

/// Parse `current_org_id` out of config bytes. A parse failure on a file that
/// *exists* is logged (rather than silently mapped to `None`) so a corrupt
/// config the user expects to be live doesn't disappear without a trace.
fn parse_org_id(bytes: &[u8], path: &Path) -> Option<String> {
    #[derive(Deserialize)]
    struct ConfigShape {
        #[serde(default)]
        current_org_id: Option<String>,
    }
    match serde_json::from_slice::<ConfigShape>(bytes) {
        Ok(parsed) => parsed.current_org_id,
        Err(error) => {
            tracing::warn!(%error, path = %path.display(), "failed to parse config.json; ignoring org_id until it is valid");
            None
        }
    }
}
