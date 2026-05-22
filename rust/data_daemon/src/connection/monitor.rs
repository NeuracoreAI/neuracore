//! Periodic `HEAD /status/health` probe.
//!
//! Mirrors `connection_manager.py::ConnectionManager`. Runs on the Tokio
//! runtime as a single supervised task. Each transition is broadcast to the
//! daemon event bus as [`DaemonEvent::ConnectionStateChanged`]; subscribers
//! (currently the upload coordinator) pause when the connection drops and
//! resume on recovery.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};

use crate::api::ApiClient;
use crate::lifecycle::signals::ShutdownSignal;
pub use crate::state::ConnectionState;
use crate::state::{DaemonEvent, EventBus};

/// Interval between health probes — matches `connection_manager.py`.
pub const HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(10);

/// Handle to the spawned connection monitor task.
pub struct MonitorHandle {
    join: JoinHandle<()>,
}

impl MonitorHandle {
    /// Wait for the monitor task to exit (used during ordered shutdown).
    pub async fn join(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "connection monitor join failed");
        }
    }
}

/// Spawn the monitor task on the current Tokio runtime.
///
/// The spawned task publishes `ConnectionState::Down` as its first action so
/// subscribers (uploader / status updater) see a definite initial state
/// regardless of subscription ordering — the first successful probe then
/// flips them to `Up`.
pub fn spawn_connection_monitor(
    client: Arc<ApiClient>,
    bus: EventBus,
    mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> MonitorHandle {
    let join = tokio::spawn(async move {
        // Publish the initial Down state from inside the task so that any
        // task calling `bus.subscribe()` between launch and the next yield
        // point sees the seed event before the first probe runs.
        let mut state = ConnectionState::Down;
        bus.publish(DaemonEvent::ConnectionStateChanged(state));
        let mut ticker = interval(HEALTH_CHECK_INTERVAL);
        // Don't try to "catch up" missed ticks during long pauses — one
        // probe per real-time interval is enough and avoids storming the
        // backend after a daemon stall.
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "connection monitor shutting down");
                    break;
                }
                _ = ticker.tick() => {
                    match client.health_check().await {
                        Ok(true) => {
                            if state != ConnectionState::Up {
                                state = ConnectionState::Up;
                                tracing::info!("backend connection restored");
                                bus.publish(DaemonEvent::ConnectionStateChanged(state));
                            }
                        }
                        Ok(false) | Err(_) => {
                            if state != ConnectionState::Down {
                                state = ConnectionState::Down;
                                tracing::warn!("backend connection lost");
                                bus.publish(DaemonEvent::ConnectionStateChanged(state));
                            }
                        }
                    }
                }
            }
        }
    });
    MonitorHandle { join }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::auth::StaticAuthProvider;
    use crate::api::client::ApiClientOptions;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn build_client(server: &MockServer) -> Arc<ApiClient> {
        let auth = Arc::new(StaticAuthProvider::new("test"));
        let mut options = ApiClientOptions::new(server.uri());
        options.timeout = Duration::from_secs(2);
        Arc::new(ApiClient::new(options, auth).expect("client"))
    }

    #[tokio::test]
    async fn publishes_initial_down_state() {
        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .and(path("/status/health"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;
        let client = build_client(&server).await;
        let bus = EventBus::new();
        let mut subscriber = bus.subscribe();
        let (tx, rx) = broadcast::channel::<ShutdownSignal>(1);

        let handle = spawn_connection_monitor(client, bus, rx);
        // The very first event after subscription is the explicit Down
        // publish from `spawn_connection_monitor`.
        let event = subscriber.recv().await.unwrap();
        assert!(matches!(
            event,
            DaemonEvent::ConnectionStateChanged(ConnectionState::Down)
        ));

        // Tear the monitor down so the test exits.
        let _ = tx.send(ShutdownSignal::Sigterm);
        handle.join.abort();
    }

    #[tokio::test]
    async fn transitions_to_up_when_health_check_passes() {
        // The first probe returns 200 so the monitor flips Down -> Up.
        // The probe runs on a tokio `interval` whose first tick fires
        // immediately, so we only need to wait a couple of polls.
        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .and(path("/status/health"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;
        let client = build_client(&server).await;
        let bus = EventBus::new();
        let mut subscriber = bus.subscribe();
        let (tx, rx) = broadcast::channel::<ShutdownSignal>(1);

        let handle = spawn_connection_monitor(client, bus, rx);

        // Initial Down, then Up once the probe succeeds.
        let first = tokio::time::timeout(Duration::from_secs(2), subscriber.recv())
            .await
            .expect("initial event")
            .unwrap();
        assert!(matches!(
            first,
            DaemonEvent::ConnectionStateChanged(ConnectionState::Down)
        ));
        let second = tokio::time::timeout(Duration::from_secs(5), subscriber.recv())
            .await
            .expect("up event")
            .unwrap();
        assert!(matches!(
            second,
            DaemonEvent::ConnectionStateChanged(ConnectionState::Up)
        ));

        let _ = tx.send(ShutdownSignal::Sigterm);
        handle.join.abort();
    }
}
