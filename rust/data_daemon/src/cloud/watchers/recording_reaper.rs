//! Periodic recording reaper.
//!
//! Reclaims recordings whose local copy is redundant — the daemon owns no other
//! cleanup for a recording that reaches a settled terminal state, so without
//! this task both their files and DB rows leak forever. Three shapes qualify:
//!
//!   * **Stopped + fully uploaded** — every declared trace uploaded and the
//!     backend fully notified (stop POSTed, expected-trace-count + per-trace
//!     progress reported). The cloud holds everything.
//!   * **Cancelled** — the data was discarded; once the backend cancel has been
//!     notified (`backend_cancel_notified_at`) nothing local needs keeping.
//!   * **Never uploads** — it was not registered and its robot or dataset row
//!     was refused, the backend no longer holds it, or a trace failed to
//!     write, register or upload. Each pass stamps
//!     `failed_at` on new ones, and [`FAILED_RECORDING_RETENTION`] later the
//!     recording is deleted. Until then the daemon status says it will not
//!     upload and when it is deleted, so the user hears of it first.
//!
//! For all three, the reaper deletes the on-disk recording directory and then
//! the `recordings` / `traces` rows, keeping local disk and the state DB bounded
//! over a long-running daemon's lifetime.
//!
//! It also deletes the robot and dataset rows nothing needs any more: confirmed
//! or refused, used by no recording, and not connected, selected or recorded
//! with for [`UNUSED_ROW_RETENTION`]. A recording only waiting on a row the
//! server has not confirmed yet is never deleted, and neither is that row. It is the single owner of
//! cancelled-recording file removal — the cancel path no longer unlinks files.
//!
//! The uploaded gate reads the authoritative per-trace `upload_status` rows; a
//! recording with a permanently `failed` trace never satisfies it and is only
//! deleted as one that never uploads. The startup sweep still handles partial
//! (mid-write) recordings separately.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::{RecordingRow, SqliteStateStore, StateStore};
use crate::storage::paths::{is_in_robot_packages, recording_dir, robot_packages_root};

/// How long a recording that can never upload is kept after the reaper first
/// sees it, so the user hears about it before it is deleted.
pub const FAILED_RECORDING_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// How long a robot or dataset row no recording uses is kept after it was
/// last connected, selected or recorded with. A process that connected a
/// robot may start recording with it much later, so its row must stay.
pub const UNUSED_ROW_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Handle returned by [`spawn_recording_reaper`].
pub struct RecordingReaperHandle {
    join: JoinHandle<()>,
}

impl RecordingReaperHandle {
    /// Wait for the reaper task to exit.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "recording reaper join failed");
        }
    }
}

/// Spawn the recording reaper task on the current Tokio runtime.
pub fn spawn_recording_reaper(
    store: SqliteStateStore,
    recordings_root: Arc<PathBuf>,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> RecordingReaperHandle {
    let store = Arc::new(store);
    let join = tokio::spawn(async move {
        let mut ticker = interval(crate::intervals::RECORDING_RECLAIM);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "recording reaper shutting down");
                    break;
                }
                _ = ticker.tick() => {
                    sweep_once(&store, &recordings_root, FAILED_RECORDING_RETENTION).await;
                }
            }
        }
    });
    RecordingReaperHandle { join }
}

/// Run one reclamation pass: list the durably-settled recordings the server-side
/// filter reports as reclaimable and reclaim each (deletes the on-disk artefacts
/// and drops the row), then delete the recordings that can never upload and
/// failed more than `failed_retention` ago. Invoked once per reclaim tick by the
/// spawned reaper task.
async fn sweep_once(
    store: &Arc<SqliteStateStore>,
    recordings_root: &Arc<PathBuf>,
    failed_retention: Duration,
) {
    // Server-side filter returns *only* durably-settled, reclaimable
    // recordings (cancel-notified, or stopped + fully uploaded with the
    // expected trace count met). This walks neither every recording nor the
    // traces of a recording wedged on a permanently-failed upload — both of
    // which the old `list_recordings` + per-row trace fetch re-scanned every
    // sweep, forever.
    let recordings = match store.recordings_pending_reclaim().await {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(%error, "recording reaper could not list reclaimable recordings");
            return;
        }
    };
    for recording in recordings {
        if let Some(traces_deleted) = reclaim(store, recordings_root, &recording).await {
            tracing::info!(
                recording_index = recording.recording_index,
                traces_deleted,
                "reclaimed fully-uploaded recording"
            );
        }
    }
    delete_failed(store, recordings_root, failed_retention).await;
    delete_unused_rows(store, recordings_root, UNUSED_ROW_RETENTION).await;
}

/// Delete the robot and dataset rows nothing has used for `retention`, and the
/// saved packages they still named inside the package folder.
async fn delete_unused_rows(
    store: &Arc<SqliteStateStore>,
    recordings_root: &Path,
    retention: Duration,
) {
    let Ok(retention) = chrono::Duration::from_std(retention) else {
        return;
    };
    let deleted = match store
        .delete_unused_rows(Utc::now().naive_utc() - retention)
        .await
    {
        Ok(deleted) => deleted,
        Err(error) => {
            tracing::warn!(%error, "recording reaper could not delete unused robots and datasets");
            return;
        }
    };
    let packages_root = robot_packages_root(recordings_root);
    for package in deleted.packages.iter().map(Path::new) {
        if is_in_robot_packages(&packages_root, package) {
            crate::cloud::robot_packages::remove_package(package).await;
        }
    }
    if deleted.robots + deleted.datasets > 0 {
        tracing::info!(
            robots = deleted.robots,
            datasets = deleted.datasets,
            "deleted robot and dataset rows nothing has used for the retention"
        );
    }
}

/// Stamp the recordings that can never upload, then delete the ones stamped
/// more than `retention` ago.
async fn delete_failed(
    store: &Arc<SqliteStateStore>,
    recordings_root: &Arc<PathBuf>,
    retention: Duration,
) {
    match store.stamp_failed_recordings().await {
        Ok(0) => {}
        Ok(recordings) => tracing::warn!(
            recordings,
            retention_days = retention.as_secs() / (24 * 60 * 60),
            "recordings can never upload; they are deleted from this machine after the retention"
        ),
        Err(error) => {
            tracing::warn!(%error, "recording reaper could not stamp failed recordings");
        }
    }
    let Ok(retention) = chrono::Duration::from_std(retention) else {
        return;
    };
    let cutoff = Utc::now().naive_utc() - retention;
    let recordings = match store.recordings_failed_before(cutoff).await {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(%error, "recording reaper could not list failed recordings");
            return;
        }
    };
    for recording in recordings {
        if let Some(traces_deleted) = reclaim(store, recordings_root, &recording).await {
            tracing::warn!(
                recording_index = recording.recording_index,
                recording_id = recording.recording_id.as_deref(),
                failed_at = ?recording.failed_at,
                traces_deleted,
                "deleted a recording that could never upload"
            );
        }
    }
}

/// Remove the recording's on-disk directory, then its DB rows. Files are
/// deleted first: if the unlink fails the rows are left in place so the next
/// sweep retries rather than orphaning files with no row pointing at them.
/// Returns how many traces were deleted, or `None` when the recording is left
/// for the next sweep.
async fn reclaim(
    store: &Arc<SqliteStateStore>,
    recordings_root: &Arc<PathBuf>,
    recording: &RecordingRow,
) -> Option<u64> {
    let dir = recording_dir(recordings_root, recording.recording_index);
    // `tokio::fs` so a large directory tree doesn't block a runtime worker
    // (the sweep runs on the async reaper task).
    match tokio::fs::remove_dir_all(&dir).await {
        Ok(()) => {}
        // Already gone (e.g. reclaimed on a prior sweep that crashed before the
        // row delete committed) — fall through and finish removing the rows.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            tracing::warn!(
                %error,
                recording_index = recording.recording_index,
                path = %dir.display(),
                "recording reaper could not remove recording directory; retrying next sweep"
            );
            return None;
        }
    }

    match store
        .delete_recording_cascade(recording.recording_index)
        .await
    {
        Ok(traces_deleted) => Some(traces_deleted),
        Err(error) => {
            tracing::warn!(
                %error,
                recording_index = recording.recording_index,
                "recording reaper removed files but could not delete rows"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::RecordingTimestamp;
    use std::path::Path;

    use tempfile::TempDir;

    use crate::state::{
        NewRecording, ProgressReportStatus, TraceUpdate, TraceUploadStatus, TraceWriteStatus,
    };

    async fn open_store() -> (SqliteStateStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .unwrap();
        (store, dir)
    }

    fn new_recording(instance: i64) -> NewRecording<'static> {
        NewRecording {
            robot_index: None,
            dataset_index: None,
            robot_id: Some("robot-1"),
            robot_instance: Some(instance),
            dataset_id: Some("ds-1"),
            start: RecordingTimestamp::observed_at(1_700_000_000_000_000_000),
        }
    }

    /// Drive a stopped recording with one fully-uploaded trace through every
    /// notify + progress gate so the server-side reclaim filter reports it.
    async fn seed_reclaimable_stopped(store: &SqliteStateStore, instance: i64) -> i64 {
        let index = seed_stopped_unreported(store, instance).await;
        store
            .mark_traces_completion_reported(&[format!("t-{instance}")])
            .await
            .unwrap();
        index
    }

    /// Same as [`seed_reclaimable_stopped`] but the trace's `UPLOAD_COMPLETE`
    /// has not yet been acknowledged by the backend.
    async fn seed_stopped_unreported(store: &SqliteStateStore, instance: i64) -> i64 {
        let index = store
            .create_recording(new_recording(instance))
            .await
            .unwrap()
            .recording_index;
        let trace_id = format!("t-{instance}");
        store
            .mark_recording_start_notified(index, &format!("cloud-{instance}"))
            .await
            .unwrap();
        store
            .create_trace(index, &trace_id, Some("J"), None)
            .await
            .unwrap();
        store
            .update_trace(
                &trace_id,
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    upload_status: Some(TraceUploadStatus::Uploaded),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();
        store
            .mark_recording_stopped(index, RecordingTimestamp::observed_at(1), &[])
            .await
            .unwrap();
        store.mark_recording_stop_notified(index).await.unwrap();
        store.set_expected_trace_count(index, 1).await.unwrap();
        store
            .set_progress_report_status(
                index,
                ProgressReportStatus::Pending,
                ProgressReportStatus::Reported,
            )
            .await
            .unwrap();
        index
    }

    /// The `bbe50b98` incident shape: everything settled locally, but the
    /// status coordinator has not yet flushed the trace's `UPLOAD_COMPLETE`
    /// to the backend. The sweep must leave the recording alone.
    #[tokio::test]
    async fn sweep_spares_a_recording_with_an_unacknowledged_completion() {
        let (store, _db_dir) = open_store().await;
        let root_dir = TempDir::new().unwrap();
        let root = Arc::new(root_dir.path().to_path_buf());
        let index = seed_stopped_unreported(&store, 0).await;

        let dir = recording_dir(&root, index);
        std::fs::create_dir_all(&dir).unwrap();
        touch(&dir.join("trace.json"));

        sweep_once(&Arc::new(store.clone()), &root, FAILED_RECORDING_RETENTION).await;

        assert!(dir.exists(), "files survive until the completion is acked");
        assert!(
            store.get_recording(index).await.unwrap().is_some(),
            "the recording row survives until the completion is acked"
        );

        store
            .mark_traces_completion_reported(&["t-0".to_string()])
            .await
            .unwrap();
        sweep_once(&Arc::new(store.clone()), &root, FAILED_RECORDING_RETENTION).await;
        assert!(!dir.exists(), "acked completion unblocks the reclaim");
    }

    /// A cancelled recording whose backend cancel has been notified — the
    /// reaper is the single owner of removing its files.
    async fn seed_reclaimable_cancelled(store: &SqliteStateStore, instance: i64) -> i64 {
        let index = store
            .create_recording(new_recording(instance))
            .await
            .unwrap()
            .recording_index;
        store
            .mark_recording_start_notified(index, &format!("cloud-{instance}"))
            .await
            .unwrap();
        store
            .cancel_recording(index, RecordingTimestamp::observed_at(1))
            .await
            .unwrap();
        store.mark_recording_cancel_notified(index).await.unwrap();
        index
    }

    fn touch(path: &Path) {
        std::fs::write(path, b"x").expect("write file");
    }

    #[tokio::test]
    async fn sweep_deletes_files_and_rows_of_a_reclaimable_recording() {
        let (store, _db_dir) = open_store().await;
        let root_dir = TempDir::new().unwrap();
        let root = Arc::new(root_dir.path().to_path_buf());
        let index = seed_reclaimable_stopped(&store, 0).await;

        let dir = recording_dir(&root, index);
        std::fs::create_dir_all(&dir).unwrap();
        touch(&dir.join("trace.json"));
        assert!(dir.exists());

        sweep_once(&Arc::new(store.clone()), &root, FAILED_RECORDING_RETENTION).await;

        assert!(!dir.exists(), "the recording directory is removed");
        assert!(
            store.get_recording(index).await.unwrap().is_none(),
            "the recording row is removed"
        );
    }

    #[tokio::test]
    async fn sweep_reclaims_a_cancelled_recording() {
        let (store, _db_dir) = open_store().await;
        let root_dir = TempDir::new().unwrap();
        let root = Arc::new(root_dir.path().to_path_buf());
        let index = seed_reclaimable_cancelled(&store, 0).await;

        let dir = recording_dir(&root, index);
        std::fs::create_dir_all(&dir).unwrap();
        touch(&dir.join("video.mp4"));

        sweep_once(&Arc::new(store.clone()), &root, FAILED_RECORDING_RETENTION).await;

        assert!(!dir.exists(), "cancelled recording files are reclaimed");
        assert!(store.get_recording(index).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn sweep_deletes_a_recording_that_never_uploads_once_its_retention_is_over() {
        let (store, _db_dir) = open_store().await;
        let root_dir = TempDir::new().unwrap();
        let root = Arc::new(root_dir.path().to_path_buf());
        // Stopped, but the backend no longer holds it.
        let index = seed_stopped_unreported(&store, 0).await;
        store.mark_recording_not_found(index).await.unwrap();
        let healthy = store
            .create_recording(new_recording(1))
            .await
            .unwrap()
            .recording_index;
        let dir = recording_dir(&root, index);
        std::fs::create_dir_all(&dir).unwrap();
        touch(&dir.join("trace.json"));

        // Within its retention: stamped, and kept for the user to hear about.
        sweep_once(&Arc::new(store.clone()), &root, FAILED_RECORDING_RETENTION).await;
        let stamped = store.get_recording(index).await.unwrap().expect("kept");
        assert!(stamped.failed_at.is_some());
        assert!(dir.exists(), "files are kept until the retention is over");

        // Retention over: files and rows are deleted; others are left alone.
        sweep_once(&Arc::new(store.clone()), &root, Duration::ZERO).await;
        assert!(!dir.exists(), "the recording directory is removed");
        assert!(store.get_recording(index).await.unwrap().is_none());
        assert!(store.get_recording(healthy).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn a_recording_that_can_upload_again_loses_its_failed_stamp() {
        let (store, _db_dir) = open_store().await;
        let root_dir = TempDir::new().unwrap();
        let root = Arc::new(root_dir.path().to_path_buf());
        let spec = data_daemon_shared::DatasetSpec {
            org_id: "org-1".into(),
            name: "warehouse".into(),
            shared: false,
            description: None,
            tags: Vec::new(),
        };
        let dataset_index = store.select_dataset(&spec).await.unwrap();
        store
            .fail_dataset(dataset_index, crate::state::ResolutionReason::Rejected)
            .await
            .unwrap();
        let index = store
            .create_recording(crate::state::NewRecording {
                dataset_index: Some(dataset_index),
                ..crate::state::NewRecording::default()
            })
            .await
            .unwrap()
            .recording_index;
        let store = Arc::new(store);

        sweep_once(&store, &root, FAILED_RECORDING_RETENTION).await;
        let stamped = store.get_recording(index).await.unwrap().expect("kept");
        assert!(stamped.failed_at.is_some());

        // Selected again: the server is asked again, so it may upload after all.
        store.select_dataset(&spec).await.unwrap();
        sweep_once(&store, &root, FAILED_RECORDING_RETENTION).await;
        let cleared = store.get_recording(index).await.unwrap().expect("kept");
        assert!(cleared.failed_at.is_none());
    }

    #[tokio::test]
    async fn only_rows_unused_for_the_retention_are_deleted() {
        let (store, _db_dir) = open_store().await;
        let arm = |instance: i64| data_daemon_shared::RobotSpec {
            org_id: "org-1".into(),
            name: "arm".into(),
            shared: false,
            instance,
            overwrite_urdf: false,
            package_path: None,
        };
        let unused = store.connect_robot(&arm(0)).await.unwrap().robot_index;
        let recorded = store.connect_robot(&arm(1)).await.unwrap().robot_index;
        let pending = store.connect_robot(&arm(2)).await.unwrap().robot_index;
        for robot_index in [unused, recorded] {
            store
                .resolve_robot(robot_index, "cloud-robot-1", None)
                .await
                .unwrap();
        }
        store
            .create_recording(crate::state::NewRecording {
                robot_index: Some(recorded),
                ..crate::state::NewRecording::default()
            })
            .await
            .unwrap();
        let store = Arc::new(store);

        // Used just now: nothing is old enough.
        delete_unused_rows(&store, Path::new("/nonexistent"), UNUSED_ROW_RETENTION).await;
        assert!(store.get_robot(unused).await.unwrap().is_some());

        delete_unused_rows(&store, Path::new("/nonexistent"), Duration::ZERO).await;
        assert!(store.get_robot(unused).await.unwrap().is_none());
        assert!(
            store.get_robot(recorded).await.unwrap().is_some(),
            "a row a recording uses stays"
        );
        assert!(
            store.get_robot(pending).await.unwrap().is_some(),
            "a row the server has not confirmed stays"
        );
    }

    #[tokio::test]
    async fn sweep_leaves_an_unsettled_recording_intact() {
        // A live (never stopped) recording is not reclaimable: the reaper must
        // touch neither its files nor its rows.
        let (store, _db_dir) = open_store().await;
        let root_dir = TempDir::new().unwrap();
        let root = Arc::new(root_dir.path().to_path_buf());
        let index = store
            .create_recording(new_recording(0))
            .await
            .unwrap()
            .recording_index;

        let dir = recording_dir(&root, index);
        std::fs::create_dir_all(&dir).unwrap();
        touch(&dir.join("trace.json"));

        sweep_once(&Arc::new(store.clone()), &root, FAILED_RECORDING_RETENTION).await;

        assert!(dir.exists(), "a live recording's files are untouched");
        assert!(
            store.get_recording(index).await.unwrap().is_some(),
            "a live recording's row is untouched"
        );
    }

    #[tokio::test]
    async fn sweep_deletes_rows_when_directory_already_gone() {
        // Crash-recovery: a prior sweep removed the files but died before its
        // row delete committed. The next sweep must still drop the rows —
        // NotFound on the directory is treated as success and falls through.
        let (store, _db_dir) = open_store().await;
        let root_dir = TempDir::new().unwrap();
        let root = Arc::new(root_dir.path().to_path_buf());
        let index = seed_reclaimable_stopped(&store, 0).await;
        // Deliberately never create the on-disk directory.
        assert!(!recording_dir(&root, index).exists());

        sweep_once(&Arc::new(store.clone()), &root, FAILED_RECORDING_RETENTION).await;

        assert!(
            store.get_recording(index).await.unwrap().is_none(),
            "rows are reclaimed even when the directory is already gone"
        );
    }

    #[tokio::test]
    async fn sweep_retains_rows_when_directory_removal_fails() {
        // Files are removed before rows. If the unlink fails the rows must be
        // left in place so the next sweep retries — never orphan files with no
        // row pointing at them. We force the failure by planting a regular
        // file where the directory is expected: `remove_dir_all` then fails
        // with ENOTDIR (not NotFound), exercising the retain-and-retry branch.
        let (store, _db_dir) = open_store().await;
        let root_dir = TempDir::new().unwrap();
        let root = Arc::new(root_dir.path().to_path_buf());
        let index = seed_reclaimable_stopped(&store, 0).await;

        let dir_path = recording_dir(&root, index);
        std::fs::write(&dir_path, b"not a directory").unwrap();

        sweep_once(&Arc::new(store.clone()), &root, FAILED_RECORDING_RETENTION).await;

        assert!(
            store.get_recording(index).await.unwrap().is_some(),
            "rows are retained for retry when file removal fails"
        );
        assert!(dir_path.exists(), "the undeletable path is left in place");
    }
}
