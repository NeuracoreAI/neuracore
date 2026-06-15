//! Periodic recording reaper.
//!
//! Reclaims recordings whose local copy is redundant — the daemon owns no other
//! cleanup for a recording that reaches a settled terminal state, so without
//! this task both their files and DB rows leak forever. Two shapes qualify:
//!
//!   * **Stopped + fully uploaded** — every declared trace uploaded and the
//!     backend fully notified (stop POSTed, expected-trace-count + per-trace
//!     progress reported). The cloud holds everything.
//!   * **Cancelled** — the data was discarded; once the backend cancel has been
//!     notified (`backend_cancel_notified_at`) nothing local needs keeping.
//!
//! For both, the reaper deletes the on-disk recording directory and then the
//! `recordings` / `traces` rows, keeping local disk and the state DB bounded
//! over a long-running daemon's lifetime. It is the single owner of
//! cancelled-recording file removal — the cancel path no longer unlinks files.
//!
//! The uploaded gate reads the authoritative per-trace `upload_status` rows; a
//! recording with a permanently `failed` trace never satisfies it, so data that
//! did not upload is intentionally retained. The startup sweep still handles
//! partial (mid-write) recordings separately.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::lifecycle::signals::ShutdownSignal;
use crate::state::{RecordingRow, SqliteStateStore, StateStore};
use crate::storage::paths::recording_dir;

/// Interval between reclaim sweeps. Reclamation is never latency-sensitive —
/// it only frees space already fully replicated to the cloud — so a relaxed
/// cadence keeps the scan off the hot path.
pub const RECLAIM_INTERVAL: Duration = Duration::from_secs(60);

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
        let mut ticker = interval(RECLAIM_INTERVAL);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "recording reaper shutting down");
                    break;
                }
                _ = ticker.tick() => {
                    sweep_once(&store, &recordings_root).await;
                }
            }
        }
    });
    RecordingReaperHandle { join }
}

async fn sweep_once(store: &Arc<SqliteStateStore>, recordings_root: &Arc<PathBuf>) {
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
        reclaim(store, recordings_root, &recording).await;
    }
}

/// Remove the recording's on-disk directory, then its DB rows. Files are
/// deleted first: if the unlink fails the rows are left in place so the next
/// sweep retries rather than orphaning files with no row pointing at them.
async fn reclaim(
    store: &Arc<SqliteStateStore>,
    recordings_root: &Arc<PathBuf>,
    recording: &RecordingRow,
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
        Ok(traces_deleted) => tracing::info!(
            recording_index = recording.recording_index,
            traces_deleted,
            "reclaimed fully-uploaded recording"
        ),
        Err(error) => tracing::warn!(
            %error,
            recording_index = recording.recording_index,
            "recording reaper removed files but could not delete rows"
        ),
    }
}
