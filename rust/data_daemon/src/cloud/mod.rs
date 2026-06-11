//! Cloud-side coordinators: batch registration, resumable uploads,
//! debounced status updates, and the periodic progress reporter.
//!
//! Every coordinator is spawned by the daemon launch routine and subscribes
//! to the broadcast event bus. The flow is:
//!
//! 1. Per-trace actor finishes writing → emits `TraceWritten`.
//! 2. Registration coordinator buffers up to 50 traces / 1 s, batch-registers,
//!    persists the resumable session URIs, emits `ReadyForUpload`.
//! 3. Upload coordinator opens each on-disk artefact and PUTs it resumably,
//!    emitting `UploadProgress` and `UploadComplete` events as it goes.
//! 4. Status updater debounces `UploadProgress` / `UploadComplete` into
//!    batched backend updates.
//! 5. Progress reporter ticks every 30 s and posts a per-recording bytes-
//!    uploaded snapshot until every trace lands in `progress_reported`.
//!
//! Each sub-module exposes a single `spawn_*` entry point so the launch
//! routine can drive ordered shutdown by dropping the handle.

pub mod cloud_files;
pub mod org_watcher;
pub mod progress;
pub mod recording_cancel_notifier;
pub mod recording_reaper;
pub mod recording_start_notifier;
pub mod recording_stop_notifier;
pub mod registration;
pub mod status;
pub mod uploader;

#[allow(unused_imports)]
pub use cloud_files::{cloud_file_list, content_type_for, ContentKind};
pub use org_watcher::{spawn_org_watcher, OrgIdRx, OrgWatcherHandle};
pub use progress::{spawn_progress_reporter, ProgressReporterHandle};
pub use recording_cancel_notifier::{
    spawn_recording_cancel_notifier, RecordingCancelNotifierHandle,
};
pub use recording_reaper::spawn_recording_reaper;
pub use recording_start_notifier::{spawn_recording_start_notifier, RecordingStartNotifierHandle};
pub use recording_stop_notifier::{spawn_recording_stop_notifier, RecordingStopNotifierHandle};
pub use registration::{spawn_registration, RegistrationCoordinatorHandle};
pub use status::{spawn_status_updater, StatusUpdate, StatusUpdaterHandle};
pub use uploader::{spawn_uploader, UploaderHandle};

use std::path::Path;

use serde::Deserialize;

/// Read the `current_org_id` field from `~/.neuracore/config.json`.
///
/// Returns `None` when the file is missing, malformed, or the field is unset.
/// The daemon falls back to `NCD_CURRENT_ORG_ID` (via the `DaemonConfig`
/// resolved at launch) when this returns `None`.
pub fn read_org_id_from_config(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    #[derive(Deserialize)]
    struct ConfigShape {
        #[serde(default)]
        current_org_id: Option<String>,
    }
    let parsed: ConfigShape = serde_json::from_slice(&bytes).ok()?;
    parsed.current_org_id
}
