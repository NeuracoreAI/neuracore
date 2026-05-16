//! Broadcast event bus driving cross-actor coordination.
//!
//! Mirrors the `tokio::sync::broadcast::channel(256)` described in section 5
//! of the rewrite plan. Phase 3 establishes the variants and the bus type;
//! later phases attach the dispatcher, registration coordinator, upload
//! coordinator, status updater, and progress reporter as subscribers.

use tokio::sync::broadcast;

/// Default capacity of the daemon event channel.
///
/// Sized to match the planning doc (section 5). Subscribers that fall behind
/// receive a [`broadcast::error::RecvError::Lagged`] rather than dropping the
/// stream — the daemon coordinators handle that by re-reading state from the
/// store on the next tick.
pub const EVENT_BUS_CAPACITY: usize = 256;

/// Events the daemon's coordinator tasks react to.
///
/// All payloads are owned `String`/`Copy` types so events can be cloned cheaply
/// across `broadcast` receivers without holding any backing buffer alive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaemonEvent {
    /// A trace finished writing to local disk and is ready for registration.
    TraceWritten {
        trace_id: String,
        recording_id: String,
    },
    /// A trace was successfully registered with the backend.
    TraceRegistered {
        trace_id: String,
        recording_id: String,
    },
    /// Registration completed and the trace is queued for upload.
    ReadyForUpload {
        trace_id: String,
        recording_id: String,
    },
    /// A trace has finished uploading.
    UploadComplete {
        trace_id: String,
        recording_id: String,
    },
    /// A recording was stopped by the producer.
    RecordingStopped { recording_id: String },
}

/// Owns the sender end of the broadcast channel and hands out subscribers.
///
/// Clone the bus to share the sender across tasks; clone the receiver via
/// [`subscribe`](Self::subscribe).
#[derive(Clone)]
pub struct EventBus {
    sender: broadcast::Sender<DaemonEvent>,
}

impl EventBus {
    /// Create a new bus with the default [`EVENT_BUS_CAPACITY`].
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(EVENT_BUS_CAPACITY);
        Self { sender }
    }

    /// Subscribe to events. The returned receiver only sees events published
    /// *after* it was created — replay is intentionally not supported.
    pub fn subscribe(&self) -> broadcast::Receiver<DaemonEvent> {
        self.sender.subscribe()
    }

    /// Publish an event. Returns the number of active receivers reached, or
    /// zero when no task is currently subscribed.
    pub fn publish(&self, event: DaemonEvent) -> usize {
        self.sender.send(event).unwrap_or(0)
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn publish_reaches_each_subscriber() {
        let bus = EventBus::new();
        let mut first = bus.subscribe();
        let mut second = bus.subscribe();

        let event = DaemonEvent::TraceWritten {
            trace_id: "trace-1".to_string(),
            recording_id: "rec-1".to_string(),
        };
        let delivered = bus.publish(event.clone());
        assert_eq!(delivered, 2);

        assert_eq!(first.recv().await.unwrap(), event);
        assert_eq!(second.recv().await.unwrap(), event);
    }

    #[test]
    fn publish_with_no_subscribers_is_zero() {
        let bus = EventBus::new();
        let delivered = bus.publish(DaemonEvent::RecordingStopped {
            recording_id: "rec-2".to_string(),
        });
        assert_eq!(delivered, 0);
    }
}
