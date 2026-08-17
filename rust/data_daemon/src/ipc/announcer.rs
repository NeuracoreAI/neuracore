//! Daemon → producer window-state announcements.
//!
//! The one direction of IPC the daemon initiates. The daemon owns recording
//! windows, so it tells producers when one opens or closes rather than being
//! asked: a producer that did not open a window has nothing to identify it with
//! and could only guess at when to ask.
//!
//! Announcements are idempotent and carry no recording identity — a source and a
//! flag. That is deliberate: a producer needs to know *whether* to spend work,
//! never which recording the work belongs to, so no announcement can be replayed
//! as a claim about a particular window.
//!
//! ## Why it also re-announces
//!
//! Edge-triggered push alone is not self-healing. A producer that attaches after
//! a window opened, or whose subscriber missed the message, would hear nothing
//! until the close — and record nothing for the whole recording, silently,
//! looking exactly like an idle sensor. So the dispatcher restates every live
//! window periodically. Hearing nothing then means only "no window", which is
//! also the safe thing for a producer to assume. A daemon with no live windows
//! announces nothing at all.

use data_daemon_shared::service_name::{
    MAX_NODES_PER_SERVICE, MAX_PUBLISHERS_PER_SERVICE, WINDOW_STATE,
    WINDOW_STATE_MAX_PAYLOAD_BYTES, WINDOW_STATE_MAX_SUBSCRIBERS,
    WINDOW_STATE_SUBSCRIBER_BUFFER_SIZE,
};
use std::sync::mpsc::{channel, Receiver, Sender};

use data_daemon_shared::WindowStateAnnouncement;
use iceoryx2::node::NodeBuilder;
use iceoryx2::prelude::*;

/// Handle the dispatcher holds to announce window state.
///
/// The publisher lives on a thread of its own rather than in the dispatcher,
/// because iceoryx2's ports are `!Send` and the dispatcher is a `tokio::spawn`'d
/// task. (The listener gets away with owning ports because it is `block_on`'d on
/// a dedicated thread.) So the dispatcher holds only a channel, and the thread
/// on the other end owns the node, service and publisher.
pub struct WindowStateAnnouncer {
    tx: Sender<(String, i64, bool)>,
}

impl WindowStateAnnouncer {
    /// Start the announcer thread.
    ///
    /// Returns `None` (having logged) rather than failing the daemon: a window
    /// that cannot be announced costs producers that did not open it, and is not a
    /// reason to refuse recording for the process that did.
    pub fn bring_up() -> Option<Self> {
        let (tx, rx) = channel::<(String, i64, bool)>();
        // Bring the ports up on the thread that will own them, and report back
        // whether it worked so a failure is visible here rather than as silent
        // non-announcement.
        let (ready_tx, ready_rx) = channel::<bool>();
        std::thread::Builder::new()
            .name("nc-window-state".to_string())
            .spawn(move || announce_loop(rx, ready_tx))
            .map_err(|error| tracing::warn!(%error, "window-state announcer thread unavailable"))
            .ok()?;
        if !ready_rx.recv().unwrap_or(false) {
            return None;
        }
        Some(WindowStateAnnouncer { tx })
    }

    /// Announce whether `(robot_id, robot_instance)` has a window open.
    ///
    /// Never blocks the dispatcher: the send is to an unbounded channel, and a
    /// dead announcer thread is ignored. Every announcement is either restated by
    /// the next re-announcement or superseded by the next transition, so one lost
    /// message is not worth surfacing.
    pub fn announce(&self, robot_id: &str, robot_instance: i64, live: bool) {
        let _ = self.tx.send((robot_id.to_string(), robot_instance, live));
    }
}

/// Own the iceoryx2 ports and publish whatever the dispatcher hands over.
fn announce_loop(rx: Receiver<(String, i64, bool)>, ready_tx: Sender<bool>) {
    let Some(publisher) = build_publisher() else {
        let _ = ready_tx.send(false);
        return;
    };
    let _ = ready_tx.send(true);
    while let Ok((robot_id, robot_instance, live)) = rx.recv() {
        let announcement = WindowStateAnnouncement {
            robot_id,
            robot_instance,
            live,
        };
        let bytes = match announcement.encode() {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::debug!(%error, "failed to encode window-state announcement");
                continue;
            }
        };
        let sample = match publisher.loan_slice_uninit(bytes.len()) {
            Ok(sample) => sample,
            Err(error) => {
                tracing::debug!(%error, "failed to loan window-state sample");
                continue;
            }
        };
        if let Err(error) = sample.write_from_slice(&bytes).send() {
            tracing::debug!(%error, "failed to publish window-state announcement");
        }
    }
}

/// Open the `window_state` service and build the daemon's publisher on it.
#[allow(clippy::type_complexity)]
fn build_publisher() -> Option<iceoryx2::port::publisher::Publisher<ipc::Service, [u8], ()>> {
    {
        let node_name = format!("neuracore-window-state-{}", std::process::id());
        let parsed_name = NodeName::new(&node_name).ok()?;
        let node = match NodeBuilder::new()
            .name(&parsed_name)
            .create::<ipc::Service>()
        {
            Ok(node) => node,
            Err(error) => {
                tracing::warn!(%error, "window-state announcer node unavailable");
                return None;
            }
        };
        let service_name = match WINDOW_STATE.try_into() {
            Ok(name) => name,
            Err(error) => {
                tracing::warn!(%error, "window-state service name invalid");
                return None;
            }
        };
        // Overflow is *enabled* here, unlike the commands service: a producer
        // whose gate thread is descheduled must never block the dispatcher, and
        // a dropped announcement is self-correcting via the next
        // re-announcement. Blocking the dispatcher on a slow subscriber would
        // stall routing for every source.
        let service = match node
            .service_builder(&service_name)
            .publish_subscribe::<[u8]>()
            .enable_safe_overflow(true)
            .subscriber_max_buffer_size(WINDOW_STATE_SUBSCRIBER_BUFFER_SIZE)
            .max_publishers(MAX_PUBLISHERS_PER_SERVICE)
            .max_subscribers(WINDOW_STATE_MAX_SUBSCRIBERS)
            .max_nodes(MAX_NODES_PER_SERVICE)
            .open_or_create()
        {
            Ok(service) => service,
            Err(error) => {
                tracing::warn!(%error, "window-state service unavailable");
                return None;
            }
        };
        let publisher = match service
            .publisher_builder()
            .initial_max_slice_len(WINDOW_STATE_MAX_PAYLOAD_BYTES)
            .create()
        {
            Ok(publisher) => publisher,
            Err(error) => {
                tracing::warn!(%error, "window-state publisher unavailable");
                return None;
            }
        };
        tracing::info!(service = WINDOW_STATE, "window-state announcer started");
        // The node and service handle must outlive the publisher, so they are
        // leaked deliberately: this thread lives for the daemon's lifetime, and
        // dropping them would tear the service down under the publisher.
        std::mem::forget(node);
        std::mem::forget(service);
        Some(publisher)
    }
}
