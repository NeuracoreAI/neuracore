//! Async SIGTERM / SIGINT handling, fanned out to subscribers over a
//! `tokio::sync::broadcast`.
//!
//! Both signals trigger a graceful shutdown: the broadcast channel is the
//! notification the daemon's main loop awaits. SIGHUP is intentionally not
//! handled.

use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::broadcast;

/// Source of a graceful-shutdown notification, useful for log messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownSignal {
    /// `SIGTERM` (default `kill` signal, CLI `stop` command).
    Sigterm,
    /// `SIGINT` (Ctrl-C from a controlling terminal).
    Sigint,
}

/// Broadcasts the shutdown notification to subscribers: call
/// [`subscribe`](Self::subscribe) for each task that needs to wait for shutdown.
#[derive(Clone)]
pub struct ShutdownBroadcaster {
    sender: broadcast::Sender<ShutdownSignal>,
}

impl ShutdownBroadcaster {
    /// Subscribe to receive a single shutdown notification.
    pub fn subscribe(&self) -> broadcast::Receiver<ShutdownSignal> {
        self.sender.subscribe()
    }

    /// Fire an explicit shutdown (used by `SystemExit`-style flows and
    /// tests). Returns the number of receivers notified.
    #[allow(dead_code)]
    pub fn signal(&self, kind: ShutdownSignal) -> usize {
        self.sender.send(kind).unwrap_or(0)
    }
}

/// Install async SIGTERM and SIGINT handlers, returning a
/// [`ShutdownBroadcaster`] and the *primary* shutdown receiver that the caller
/// must await.
///
/// Returning the primary receiver alongside the handle closes a race that
/// would otherwise exist between the supervisor task's first `send` and the
/// caller's first `subscribe()`: `broadcast::Sender::send` returns
/// `SendError` when there are zero receivers, and `broadcast` does not
/// replay messages for receivers that subscribe later. By constructing the
/// primary receiver up-front via `broadcast::channel`, we guarantee at least
/// one receiver exists from the moment the supervisor task starts.
pub fn install_shutdown_handler(
) -> std::io::Result<(ShutdownBroadcaster, broadcast::Receiver<ShutdownSignal>)> {
    let (sender, primary_receiver) = broadcast::channel(8);
    let supervisor_sender = sender.clone();

    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;

    tokio::spawn(async move {
        loop {
            let received = tokio::select! {
                Some(()) = sigterm.recv() => ShutdownSignal::Sigterm,
                Some(()) = sigint.recv() => ShutdownSignal::Sigint,
                else => return,
            };
            tracing::info!(signal = ?received, "shutdown signal received");
            // The primary receiver returned from this function keeps the
            // channel populated with at least one receiver; further sends
            // only fail if every subscriber has been dropped (typically
            // during shutdown), which we ignore.
            let _ = supervisor_sender.send(received);
        }
    });

    Ok((ShutdownBroadcaster { sender }, primary_receiver))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn explicit_signal_reaches_subscriber() {
        // Construct the channel directly (we cannot install signal handlers
        // in tests because they're a process-global resource) and exercise
        // the `signal` / `subscribe` plumbing the supervisor task relies on.
        let (sender, primary_receiver) = broadcast::channel(8);
        let handle = ShutdownBroadcaster { sender };
        // Drop the primary so we can test that the explicit subscribe path
        // also works for additional listeners.
        drop(primary_receiver);
        let mut subscriber = handle.subscribe();

        let notified = handle.signal(ShutdownSignal::Sigterm);
        assert_eq!(notified, 1);
        let received = subscriber.recv().await.expect("recv");
        assert_eq!(received, ShutdownSignal::Sigterm);
    }

    #[tokio::test]
    async fn signal_with_no_subscribers_returns_zero() {
        let (sender, primary_receiver) = broadcast::channel(8);
        let handle = ShutdownBroadcaster { sender };
        drop(primary_receiver);
        // No live receivers — `send` returns Err, surfaced as `0` from our
        // `signal` wrapper.
        assert_eq!(handle.signal(ShutdownSignal::Sigint), 0);
    }
}
