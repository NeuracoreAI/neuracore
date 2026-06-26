//! Hold a wakelock while uploads are in flight.
//!
//! When `NCD_KEEP_WAKELOCK_WHILE_UPLOAD=1` the daemon prevents the host from
//! idling into suspend or hitting a session-idle inhibitor while a recording
//! upload is still going. On Linux this is best
//! expressed through `systemd-inhibit`, which holds the inhibitor for as
//! long as its child process keeps running. The wakelock task:
//!
//! 1. Subscribes to the daemon event bus.
//! 2. Tracks each in-flight trace (by `trace_id`, with its `recording_index`):
//!    a trace is added on [`ReadyForUpload`](DaemonEvent::ReadyForUpload) and
//!    removed on [`UploadComplete`](DaemonEvent::UploadComplete);
//!    [`RecordingCancelled`](DaemonEvent::RecordingCancelled) drops only that
//!    recording's traces.
//! 3. Spawns `systemd-inhibit --what=idle:sleep --mode=block sleep infinity`
//!    on the empty→non-empty transition and kills it on the
//!    non-empty→empty transition.
//!
//! Hosts without `systemd-inhibit` on `$PATH` log a single warning and
//! degrade to a no-op — the upload path itself is untouched. macOS / BSDs
//! would need a per-platform shim (the macOS-equivalent stay-awake CLI);
//! this feature is Linux-only.

use std::collections::HashMap;
use std::process::{Child, Command, Stdio};

use tokio::sync::broadcast;
use tokio::task::JoinHandle;

use crate::lifecycle::shutdown::ShutdownSignal;
use crate::state::{DaemonEvent, EventBus};

/// Handle returned by [`spawn_wakelock`]. Drop or join to tear the task down.
pub struct WakelockHandle {
    join: JoinHandle<()>,
}

impl WakelockHandle {
    /// Wait for the wakelock task to exit. Idempotent: the task itself
    /// releases its inhibitor on shutdown.
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "wakelock task join failed");
        }
    }
}

/// Spawn the wakelock task on the current Tokio runtime.
///
/// Subscribes to `bus` for [`ReadyForUpload`](DaemonEvent::ReadyForUpload),
/// [`UploadComplete`](DaemonEvent::UploadComplete), and
/// [`RecordingCancelled`](DaemonEvent::RecordingCancelled). Holds a
/// `systemd-inhibit` child while at least one trace is in flight.
pub fn spawn_wakelock(
    bus: EventBus,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> WakelockHandle {
    // Subscribe synchronously at spawn time (matching the monitor's deliberate
    // ordering) so an event published between this call returning and the task
    // first being polled is not missed.
    let mut subscriber = bus.subscribe();
    let join = tokio::spawn(async move {
        let mut active = ActiveUploads::default();
        let mut inhibitor = InhibitorChild::new();

        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "wakelock task shutting down");
                    break;
                }
                event = subscriber.recv() => {
                    match event {
                        Ok(DaemonEvent::ReadyForUpload { trace_id, recording_index }) => {
                            if active.add(trace_id, recording_index) {
                                inhibitor.ensure_held();
                            }
                        }
                        Ok(DaemonEvent::UploadComplete { trace_id, .. }) => {
                            if active.complete(&trace_id) {
                                inhibitor.release();
                            }
                        }
                        Ok(DaemonEvent::RecordingCancelled { recording_index }) => {
                            // Drop only the cancelled recording's in-flight
                            // traces (whose actors were torn down, so their
                            // `UploadComplete` will never arrive) — releasing
                            // the inhibitor only if nothing else is uploading.
                            // Clearing *all* traces here would drop the
                            // inhibitor another recording's still-running
                            // upload needs.
                            let released = active.cancel_recording(recording_index);
                            tracing::debug!(
                                recording_index,
                                remaining = active.len(),
                                "wakelock handling recording cancel"
                            );
                            if released {
                                inhibitor.release();
                            }
                        }
                        Ok(_) => {}
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            // We may have dropped `UploadComplete`s, which would
                            // otherwise pin phantom trace-ids in `active` and
                            // leave the inhibitor held forever. Resync
                            // conservatively: clear the bookkeeping and release.
                            // The next `ReadyForUpload` re-acquires — a brief
                            // inhibitor gap beats an inhibitor stuck on.
                            tracing::warn!(
                                skipped,
                                pending = active.len(),
                                "wakelock missed bus events; resyncing (clearing pending + releasing)"
                            );
                            active.clear();
                            inhibitor.release();
                        }
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
            }
        }
        inhibitor.release();
    });
    WakelockHandle { join }
}

/// In-flight upload bookkeeping for the wakelock: each pending trace mapped to
/// the recording it belongs to, so a recording cancel releases only *its* own
/// traces rather than every recording's.
#[derive(Default)]
struct ActiveUploads {
    /// `trace_id → recording_index` for every trace currently uploading.
    by_trace: HashMap<String, i64>,
}

impl ActiveUploads {
    /// Record a trace as in-flight. Returns `true` on the 0→non-empty
    /// transition, i.e. when the caller should acquire the inhibitor.
    fn add(&mut self, trace_id: String, recording_index: i64) -> bool {
        let was_empty = self.by_trace.is_empty();
        self.by_trace.insert(trace_id, recording_index);
        was_empty
    }

    /// Remove a finished trace. Returns `true` when the last in-flight trace
    /// completed, i.e. when the caller should release the inhibitor.
    fn complete(&mut self, trace_id: &str) -> bool {
        self.by_trace.remove(trace_id).is_some() && self.by_trace.is_empty()
    }

    /// Drop every trace belonging to `recording_index` (its actors are gone, so
    /// their `UploadComplete` will never arrive). Returns `true` when nothing
    /// remains in flight, i.e. when the caller should release the inhibitor.
    fn cancel_recording(&mut self, recording_index: i64) -> bool {
        self.by_trace.retain(|_, index| *index != recording_index);
        self.by_trace.is_empty()
    }

    /// Forget every in-flight trace (used on a lagged-bus resync).
    fn clear(&mut self) {
        self.by_trace.clear();
    }

    fn len(&self) -> usize {
        self.by_trace.len()
    }
}

/// Owns the optional `systemd-inhibit` child process. Each `ensure_held` is
/// idempotent (re-entrant) and `release` is a no-op when nothing is held.
struct InhibitorChild {
    child: Option<Child>,
    /// Logged once when the platform doesn't provide `systemd-inhibit`, so
    /// the daemon doesn't repeatedly warn at every transition.
    warned_unavailable: bool,
}

impl InhibitorChild {
    fn new() -> Self {
        Self {
            child: None,
            warned_unavailable: false,
        }
    }

    fn ensure_held(&mut self) {
        if self.child.is_some() {
            return;
        }
        match spawn_systemd_inhibit() {
            Ok(child) => {
                tracing::info!("wakelock acquired (systemd-inhibit)");
                self.child = Some(child);
            }
            Err(InhibitorError::NotInstalled) => {
                if !self.warned_unavailable {
                    self.warned_unavailable = true;
                    tracing::warn!(
                        "NCD_KEEP_WAKELOCK_WHILE_UPLOAD is set but systemd-inhibit \
                         is not available; uploads will run without an inhibitor"
                    );
                }
            }
            Err(InhibitorError::Spawn(error)) => {
                tracing::warn!(%error, "failed to spawn systemd-inhibit");
            }
        }
    }

    fn release(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        if let Err(error) = child.kill() {
            tracing::warn!(%error, "failed to release wakelock (systemd-inhibit)");
        }
        // `wait` reaps the child so we don't leak a zombie process. We
        // don't care about the exit status — the inhibitor we asked it to
        // hold is released the moment the process exits.
        let _ = child.wait();
        tracing::info!("wakelock released");
    }
}

impl Drop for InhibitorChild {
    fn drop(&mut self) {
        self.release();
    }
}

#[derive(Debug, thiserror::Error)]
enum InhibitorError {
    #[error("systemd-inhibit is not installed on this host")]
    NotInstalled,
    #[error("failed to spawn systemd-inhibit: {0}")]
    Spawn(#[from] std::io::Error),
}

fn spawn_systemd_inhibit() -> Result<Child, InhibitorError> {
    let mut command = Command::new("systemd-inhibit");
    command
        .args([
            "--what=idle:sleep",
            "--who=neuracore-data-daemon",
            "--why=Active recording upload in progress",
            "--mode=block",
            // `sleep infinity` is a no-op placeholder that keeps
            // systemd-inhibit alive until we kill it.
            "sleep",
            "infinity",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    match command.spawn() {
        Ok(child) => Ok(child),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Err(InhibitorError::NotInstalled)
        }
        Err(error) => Err(InhibitorError::Spawn(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inhibitor_release_without_held_is_noop() {
        // The `release` path must tolerate the no-child case — the wakelock
        // task calls it once at startup before any traces are in flight.
        let mut inhibitor = InhibitorChild::new();
        inhibitor.release();
    }

    #[test]
    fn add_signals_acquire_only_on_first_trace() {
        let mut active = ActiveUploads::default();
        assert!(
            active.add("a".into(), 1),
            "first trace acquires the inhibitor"
        );
        assert!(
            !active.add("b".into(), 1),
            "a later trace does not re-acquire"
        );
    }

    #[test]
    fn complete_signals_release_only_when_last_trace_finishes() {
        let mut active = ActiveUploads::default();
        active.add("a".into(), 1);
        active.add("b".into(), 1);
        assert!(!active.complete("a"), "one trace still in flight");
        assert!(active.complete("b"), "last trace finishing releases");
        // Completing an unknown trace never signals a release.
        assert!(!active.complete("a"));
    }

    #[test]
    fn cancel_releases_only_the_cancelled_recordings_traces() {
        // M6 regression: cancelling recording A must NOT release the inhibitor
        // while recording B still has an upload in flight.
        let mut active = ActiveUploads::default();
        active.add("a-trace".into(), 1);
        active.add("b-trace".into(), 2);

        let released = active.cancel_recording(1);
        assert!(
            !released,
            "recording B is still uploading, so the inhibitor must stay held"
        );
        assert_eq!(active.len(), 1, "only A's trace was dropped");

        // B finishing now releases.
        assert!(
            active.complete("b-trace"),
            "B's completion releases the inhibitor"
        );
    }

    #[test]
    fn cancel_releases_when_it_empties_the_set() {
        let mut active = ActiveUploads::default();
        active.add("a-trace".into(), 1);
        assert!(
            active.cancel_recording(1),
            "cancelling the only in-flight recording releases the inhibitor"
        );
        assert_eq!(active.len(), 0);
    }
}
