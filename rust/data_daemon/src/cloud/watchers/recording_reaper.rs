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
//!   * **Never uploads** — its robot or dataset was refused, the backend no
//!     longer holds it, or a trace failed to write, register or upload. Each pass stamps
//!     `failed_at` on new ones, and [`FAILED_RECORDING_RETENTION`] later the
//!     recording is deleted. Until then the daemon status says it will not
//!     upload and when it is deleted, so the user hears of it first.
//!
//! For all three, the reaper deletes the on-disk recording directory and then
//! the `recordings` / `traces` rows, keeping local disk and the state DB bounded
//! over a long-running daemon's lifetime. It is the single owner of
//! cancelled-recording file removal — the cancel path no longer unlinks files.
//!
//! The uploaded gate reads the authoritative per-trace `upload_status` rows; a
//! recording with a permanently `failed` trace never satisfies it and is only
//! deleted as one that never uploads. The startup sweep still handles partial
//! (mid-write) recordings separately.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::{RecordingRow, SqliteStateStore, StateStore};
use crate::storage::paths::recording_dir;

/// How long a recording that can never upload is kept after the reaper first
/// sees it, so the user hears about it before it is deleted.
pub const FAILED_RECORDING_RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

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
        reclaim(store, recordings_root, &recording, Reclaimed::Settled).await;
    }
    delete_failed(store, recordings_root, failed_retention).await;
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
        reclaim(store, recordings_root, &recording, Reclaimed::NeverUploads).await;
    }
}

/// Why a recording is reclaimed, for the log.
#[derive(Clone, Copy)]
enum Reclaimed {
    /// Fully uploaded, or cancelled and the cancel sent.
    Settled,
    /// It can never upload and its retention is over.
    NeverUploads,
}

/// Remove the recording's on-disk directory, then its DB rows. Files are
/// deleted first: if the unlink fails the rows are left in place so the next
/// sweep retries rather than orphaning files with no row pointing at them.
async fn reclaim(
    store: &Arc<SqliteStateStore>,
    recordings_root: &Arc<PathBuf>,
    recording: &RecordingRow,
    why: Reclaimed,
) {
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
            return;
        }
    }

    match store
        .delete_recording_cascade(recording.recording_index)
        .await
    {
        Ok(traces_deleted) => match why {
            Reclaimed::Settled => tracing::info!(
                recording_index = recording.recording_index,
                traces_deleted,
                "reclaimed fully-uploaded recording"
            ),
            Reclaimed::NeverUploads => tracing::warn!(
                recording_index = recording.recording_index,
                recording_id = recording.recording_id.as_deref(),
                failed_at = ?recording.failed_at,
                traces_deleted,
                "deleted a recording that could never upload"
            ),
        },
        Err(error) => tracing::warn!(
            %error,
            recording_index = recording.recording_index,
            "recording reaper removed files but could not delete rows"
        ),
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
            robot_id: Some("robot-1"),
            robot_instance: Some(instance),
            dataset_id: Some("ds-1"),
            start: RecordingTimestamp::observed_at(1_700_000_000_000_000_000),
            ..NewRecording::default()
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
            .mark_recording_stopped(index, RecordingTimestamp::observed_at(1))
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
