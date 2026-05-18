//! Hold a wakelock while uploads are in flight.
//!
//! Phase 7. When `NCD_KEEP_WAKELOCK_WHILE_UPLOAD=1` the daemon should
//! prevent the host from idling into suspend or hitting a session-idle
//! inhibitor while a recording upload is still going. On Linux this is best
//! expressed through `systemd-inhibit`, which holds the inhibitor for as
//! long as its child process keeps running. The wakelock task:
//!
//! 1. Subscribes to the daemon event bus.
//! 2. Keeps a counter of "in-flight" recordings: incremented on
//!    [`ReadyForUpload`](DaemonEvent::ReadyForUpload), decremented on
//!    [`UploadComplete`](DaemonEvent::UploadComplete).
//! 3. Spawns `systemd-inhibit --what=idle:sleep --mode=block sleep
//!    infinity` when the counter goes 0→positive, kills it when the counter
//!    goes back to 0.
//!
//! Hosts without `systemd-inhibit` on `$PATH` log a single warning and
//! degrade to a no-op — the upload path itself is untouched. macOS / BSDs
//! would need a per-platform shim (the macOS-equivalent stay-awake CLI);
//! the rewrite plan explicitly scopes this to Linux for v1.

use std::collections::HashSet;
use std::process::{Child, Command, Stdio};

use tokio::sync::broadcast;
use tokio::task::JoinHandle;

use crate::lifecycle::signals::ShutdownSignal;
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
    let join = tokio::spawn(async move {
        let mut subscriber = bus.subscribe();
        let mut active: HashSet<String> = HashSet::new();
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
                        Ok(DaemonEvent::ReadyForUpload { trace_id, .. }) => {
                            if active.insert(trace_id) {
                                inhibitor.ensure_held();
                            }
                        }
                        Ok(DaemonEvent::UploadComplete { trace_id, .. }) => {
                            if active.remove(&trace_id) && active.is_empty() {
                                inhibitor.release();
                            }
                        }
                        Ok(DaemonEvent::RecordingCancelled { recording_id }) => {
                            // Cancellation tears down every per-trace actor
                            // we may have been counting; without an explicit
                            // sweep here a cancelled recording's
                            // `ReadyForUpload` events stay in `active` and
                            // the inhibitor never releases. We don't know
                            // the trace ids any more, so we conservatively
                            // drop them all when the cancellation arrives.
                            tracing::debug!(
                                recording_id,
                                pending = active.len(),
                                "wakelock dropping all pending traces on recording cancel"
                            );
                            active.clear();
                            inhibitor.release();
                        }
                        Ok(_) => {}
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            tracing::warn!(
                                skipped,
                                "wakelock task missed bus events; bookkeeping may drift"
                            );
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
}
