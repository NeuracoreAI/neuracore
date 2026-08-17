//! Whether this process should spend work on a source's data right now.
//!
//! Correctness never needed this gate. The daemon routes by publish stamp and
//! drops what falls outside a window, so a producer that logged unconditionally
//! would still land in the right recording. What it would also do is encode and
//! spool continuously while idle — the writer's NUT chunks are written *here*, in
//! the producer, against a spool budget shared with a real recording (see
//! [`crate::writer`]). So the gate exists to save work, not to decide truth.
//!
//! That split is why a stale answer is survivable: a late open costs a few
//! dropped frames, never a misrouted one.
//!
//! ## Who decides
//!
//! The daemon, in both directions:
//!
//! 1. **This process opened the window.** `start_recording` says so locally, so
//!    the gate opens with zero latency. Waiting for the daemon's announcement
//!    instead would drop the frames logged between the call and the round trip.
//! 2. **Otherwise the daemon's announcements govern** — the only thing available
//!    to a process that did *not* open a window, and the only signal that works
//!    offline, where no cloud recording id exists to name one with.
//!
//! Announcements carry no recording identity, just a source and a flag. A
//! producer learns whether to work, never which recording the work belongs to.
//!
//! ## Why nothing here polls, and why nothing here gets stuck
//!
//! The gate is driven by its own window-state subscriber ([`build_subscriber`]),
//! drained on a thread of its own. No IPC on the logging path at all, and no
//! request-response: the daemon knows the instant a window opens and says so.
//!
//! Push alone would be fragile in both directions, so both are closed:
//!
//! - **Stuck shut** — a producer that attached after the open, or that missed the
//!   message, would hear nothing until the close and record nothing for the whole
//!   recording. The daemon re-announces every live window periodically, so
//!   hearing nothing only ever means "no window".
//! - **Stuck open** — a daemon that dies mid-recording would announce no close,
//!   leaving this process encoding and spooling forever with nothing to route it.
//!   So an open gate is a *lease*: it needs to keep hearing that the window
//!   lives, and expires after [`LIVE_LEASE`] of silence.
//!
//! Both bounds come from the daemon's re-announcement cadence rather than from
//! anything this side chooses, which is what keeps one authority in charge.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use data_daemon_shared::service_name::{
    MAX_NODES_PER_SERVICE, MAX_PUBLISHERS_PER_SERVICE, WINDOW_STATE, WINDOW_STATE_MAX_SUBSCRIBERS,
    WINDOW_STATE_SUBSCRIBER_BUFFER_SIZE,
};
use data_daemon_shared::WindowStateAnnouncement;
use iceoryx2::node::NodeBuilder;
use iceoryx2::port::subscriber::Subscriber;
use iceoryx2::prelude::*;

use crate::publisher::{flush_published_data, now_ns};
use crate::writer::{writer_queue, WriterMsg};

/// How long the refresher blocks waiting for the next announcement before
/// looping to re-check leases.
///
/// Only bounds how promptly an *expiry* is noticed; an arriving announcement
/// wakes the drain immediately.
const DRAIN_POLL: Duration = Duration::from_millis(20);

/// How long an open gate survives without hearing that its window still lives.
///
/// The daemon re-announces every live window on a 100 ms cadence, so this is
/// several missed announcements' worth of grace — long enough that a briefly
/// busy daemon never shuts a healthy gate, short enough that a dead one stops
/// this process spooling into the void.
const LIVE_LEASE: Duration = Duration::from_millis(600);

/// Cadence while waiting for the first announcement of a drain.
const ANNOUNCEMENT_RECEIVE_POLL: Duration = Duration::from_millis(2);

/// A source key, matching the daemon's own `(robot_id, robot_instance)`.
pub(crate) type Source = (String, i64);

/// What one source's gate knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GateState {
    /// Whether frames are currently admitted.
    open: bool,
    /// Whether this process opened the window. While true the daemon's answer is
    /// ignored: local knowledge is both fresher and authoritative for our own
    /// call.
    owned: bool,
    /// When the daemon last confirmed this window lives. Bounds how long an open
    /// gate survives the daemon going away; `None` for a source we have never
    /// heard is live.
    confirmed_at: Option<Instant>,
}

impl GateState {
    /// A source seen for the first time: closed, unowned, never confirmed.
    const fn unknown() -> Self {
        GateState {
            open: false,
            owned: false,
            confirmed_at: None,
        }
    }
}

/// A gate that changed, and what the writer owes as a result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GateTransition {
    /// A window opened for a source this process does not own. The writer must
    /// arm a boundary split so no chunk spans the open.
    Opened { source: Source },
    /// The window closed. The writer must seal and announce the source's tail
    /// chunks — nothing else will, since this process publishes no stop.
    Closed { source: Source },
}

struct GateRegistry {
    owner_pid: u32,
    /// PID whose refresher thread is running, if any. Stored rather than using a
    /// [`std::sync::Once`] because a thread does not survive `fork`: a `Once`
    /// already tripped by the parent would leave a forked child with no
    /// refresher, so its gates would never open.
    refresher_pid: Option<u32>,
    sources: HashMap<Source, GateState>,
}

static GATES: LazyLock<Mutex<GateRegistry>> = LazyLock::new(|| {
    Mutex::new(GateRegistry {
        owner_pid: 0,
        refresher_pid: None,
        sources: HashMap::new(),
    })
});

/// Lock the gate registry and run `operation` against it.
///
/// Heals on fork the same way the chunk registry does: a child inherits the
/// parent's map but none of its windows, so stale entries are cleared before
/// use. Getting this wrong would have a forked camera process believe it was
/// mid-recording because its parent was.
fn with_registry<R>(operation: impl FnOnce(&mut HashMap<Source, GateState>) -> R) -> R {
    let mut registry = GATES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let pid = std::process::id();
    if registry.owner_pid != pid {
        registry.sources.clear();
        // The parent's refresher thread did not come across the fork.
        registry.refresher_pid = None;
        registry.owner_pid = pid;
    }
    operation(&mut registry.sources)
}

/// Start this process's refresher thread if it has none.
///
/// The thread must be the gate's own, not the writer's: the writer thread is
/// spawned by the first frame it is handed, and behind a shut gate no frame is
/// ever handed to it — so hanging the refresh off the writer would mean the gate
/// could only open after it had already opened.
fn ensure_refresher() {
    let pid = std::process::id();
    let needs_spawn = {
        let mut registry = GATES
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if registry.refresher_pid == Some(pid) {
            false
        } else {
            registry.refresher_pid = Some(pid);
            true
        }
    };
    if !needs_spawn {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("nc-gate-refresh".to_string())
        .spawn(|| {
            // One subscriber for the whole process, owned by the thread that
            // drains it — iceoryx2 ports are `!Send`, and a port per thread would
            // multiply the daemon's publisher segment by the thread count.
            let Some(subscriber) = build_subscriber() else {
                tracing::warn!(
                    "no window-state subscriber; video for a source this process \
                     did not open will not be recorded"
                );
                return;
            };
            loop {
                settle_transitions(drain_announcements(&subscriber));
                settle_transitions(expire_stale_leases());
            }
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "failed to start the gate refresher; video for a \
             source this process did not open will not be recorded");
        let mut registry = GATES
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Let a later frame try again rather than failing silently forever.
        registry.refresher_pid = None;
    }
}

/// Act on what changed, through the writer's existing control messages.
///
/// Reuses the paths `start_recording` and `flush_source` already drive rather
/// than reaching into chunk state from this thread, so a gate-driven open or
/// close is indistinguishable to the writer from an owner-driven one.
fn settle_transitions(transitions: Vec<GateTransition>) {
    for transition in transitions {
        match transition {
            GateTransition::Opened { source } => {
                // Arm at now, not at the window's true open: the frames in
                // between were refused, so nothing already written straddles the
                // boundary. This is the leading gap the gate trades for not being
                // pushed to.
                let _ = writer_queue().push(WriterMsg::Boundary {
                    robot_id: source.0,
                    robot_instance: source.1,
                    publish_ns: now_ns(),
                });
            }
            GateTransition::Closed { source } => {
                // Seal what this process owes. It publishes no stop, so without
                // this its tail chunk sits open until exit while the daemon holds
                // the window waiting on a flush marker that never arrives.
                let (ack_tx, ack_rx) = std::sync::mpsc::channel();
                let _ = writer_queue().push(WriterMsg::FlushSource {
                    robot_id: source.0,
                    robot_instance: source.1,
                    ack: ack_tx,
                });
                let _ = ack_rx.recv();
                // The ack means spooled and queued; this puts the announcements
                // on the wire.
                flush_published_data();
            }
        }
    }
}

/// Record that *this* process opened a window for the source.
///
/// Called from `start_recording`, so the gate is open before the call returns
/// and the owner's first frame is never dropped.
pub(crate) fn note_window_opened_locally(robot_id: &str, robot_instance: i64) {
    with_registry(|sources| {
        let state = sources
            .entry((robot_id.to_string(), robot_instance))
            .or_insert_with(GateState::unknown);
        state.open = true;
        state.owned = true;
    });
}

/// Record that this process closed the window it opened.
///
/// Leaves `owned` false so the daemon's answer governs again: another process may
/// hold a window open for this source, and after our own stop we have no local
/// claim on the truth.
pub(crate) fn note_window_closed_locally(robot_id: &str, robot_instance: i64) {
    with_registry(|sources| {
        if let Some(state) = sources.get_mut(&(robot_id.to_string(), robot_instance)) {
            state.open = false;
            state.owned = false;
            // Another process may still hold this source open, so the daemon's
            // announcements decide from here.
            state.confirmed_at = None;
        }
    });
}

/// Whether data for this source should be forwarded to the daemon.
///
/// **Never blocks and never does IPC.** A map read under a short-lived lock, on
/// a path a robot control loop calls at frame rate; the daemon is asked only on
/// the writer thread. A source seen for the first time is registered here so the
/// refresh sweep starts covering it, and refused until that sweep answers — so a
/// process attaching mid-recording begins contributing within one poll interval
/// rather than immediately.
///
/// One gate for every data type, not just video: a process that cannot discover
/// a window cannot log joints into it either, and gating only the expensive path
/// would leave that half-fixed.
///
/// Paying that uniform bound is what buys a logging path with no round trip in
/// it. The alternative — asking once, synchronously, on the first frame — would
/// put a whole IPC round trip inside `log_rgb` for every process that touches a
/// camera, including ones that never record at all.
pub(crate) fn admits_data(robot_id: &str, robot_instance: i64) -> bool {
    ensure_refresher();
    with_registry(|sources| {
        sources
            .entry((robot_id.to_string(), robot_instance))
            .or_insert_with(GateState::unknown)
            .open
    })
}

/// Apply every announcement the daemon has published since the last drain.
///
/// Blocks up to [`DRAIN_POLL`] for the first one so the thread is not spinning,
/// then takes everything queued. Returns what actually changed — a
/// re-announcement of a state already held is a no-op, which is what lets the
/// daemon restate the truth as often as it likes.
fn drain_announcements(subscriber: &Subscriber<ipc::Service, [u8], ()>) -> Vec<GateTransition> {
    let mut transitions = Vec::new();
    let deadline = Instant::now() + DRAIN_POLL;
    loop {
        match next_announcement(subscriber) {
            Some((source, live)) => {
                if let Some(transition) = apply_announcement(source, live) {
                    transitions.push(transition);
                }
                // Keep draining without waiting: a burst is one wake-up.
                continue;
            }
            None => {
                if !transitions.is_empty() || Instant::now() >= deadline {
                    return transitions;
                }
                std::thread::sleep(ANNOUNCEMENT_RECEIVE_POLL);
            }
        }
    }
}

/// Record one announcement, returning the transition if the gate moved.
fn apply_announcement(source: Source, live: bool) -> Option<GateTransition> {
    with_registry(|sources| {
        let state = sources
            .entry(source.clone())
            .or_insert_with(GateState::unknown);
        // Our own window: we know its state first-hand and the daemon's view of
        // it can only be older.
        if state.owned {
            return None;
        }
        if live {
            // Refresh the lease even when already open — that is the whole point
            // of a re-announcement.
            state.confirmed_at = Some(Instant::now());
        } else {
            state.confirmed_at = None;
        }
        if state.open == live {
            return None;
        }
        state.open = live;
        Some(if live {
            GateTransition::Opened { source }
        } else {
            GateTransition::Closed { source }
        })
    })
}

/// Shut any gate whose lease has run out.
///
/// The answer to a daemon that vanished mid-recording: without this the gate
/// stays open and this process spools frames nothing will ever route. Owned
/// windows are exempt — this process knows first-hand that its own window lives,
/// and a daemon restart does not retract that.
pub(crate) fn expire_stale_leases() -> Vec<GateTransition> {
    let now = Instant::now();
    let expired: Vec<Source> = with_registry(|sources| {
        sources
            .iter()
            .filter(|(_, state)| state.open && !state.owned)
            .filter(|(_, state)| match state.confirmed_at {
                None => true,
                Some(confirmed_at) => now.duration_since(confirmed_at) >= LIVE_LEASE,
            })
            .map(|(source, _)| source.clone())
            .collect()
    });
    let mut transitions = Vec::new();
    for source in expired {
        let shut = with_registry(|sources| {
            let state = sources.get_mut(&source)?;
            if !state.open || state.owned {
                return None;
            }
            state.open = false;
            state.confirmed_at = None;
            Some(())
        });
        if shut.is_some() {
            tracing::warn!(
                robot_id = source.0,
                "no window confirmation from the daemon; closing the gate"
            );
            transitions.push(GateTransition::Closed { source });
        }
    }
    transitions
}

/// Take the next queued announcement, or `None` when the queue is empty.
///
/// A decode failure is dropped rather than retried: the next re-announcement
/// restates the same truth.
fn next_announcement(subscriber: &Subscriber<ipc::Service, [u8], ()>) -> Option<(Source, bool)> {
    match subscriber.receive() {
        Ok(Some(sample)) => WindowStateAnnouncement::decode(sample.payload())
            .map(|announcement| {
                (
                    (announcement.robot_id, announcement.robot_instance),
                    announcement.live,
                )
            })
            .ok(),
        Ok(None) => None,
        Err(error) => {
            tracing::debug!(%error, "window-state receive failed");
            None
        }
    }
}

/// Build this process's window-state subscriber, on the gate thread that owns it.
fn build_subscriber() -> Option<Subscriber<ipc::Service, [u8], ()>> {
    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .map_err(|error| tracing::warn!(%error, "window-state node unavailable"))
        .ok()?;
    let service_name = WINDOW_STATE
        .try_into()
        .map_err(|error| tracing::warn!(%error, "window-state service name invalid"))
        .ok()?;
    let service = node
        .service_builder(&service_name)
        .publish_subscribe::<[u8]>()
        .enable_safe_overflow(true)
        .subscriber_max_buffer_size(WINDOW_STATE_SUBSCRIBER_BUFFER_SIZE)
        .max_publishers(MAX_PUBLISHERS_PER_SERVICE)
        .max_subscribers(WINDOW_STATE_MAX_SUBSCRIBERS)
        .max_nodes(MAX_NODES_PER_SERVICE)
        .open_or_create()
        .map_err(|error| tracing::warn!(%error, "window-state service unavailable"))
        .ok()?;
    let subscriber = service
        .subscriber_builder()
        .create()
        .map_err(|error| tracing::warn!(%error, "window-state subscriber unavailable"))
        .ok()?;
    // The node and service must outlive the subscriber, and this thread runs for
    // the life of the process.
    std::mem::forget(node);
    std::mem::forget(service);
    Some(subscriber)
}

/// Forget every gate. Test-only: the registry is process-wide, so cases that
/// assert on transitions must not inherit each other's sources.
#[cfg(test)]
fn reset_for_test() {
    with_registry(|sources| sources.clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The registry is process-wide, so these run under one lock rather than
    /// racing each other's sources.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn source(name: &str) -> Source {
        (name.to_string(), 0)
    }

    #[test]
    fn an_owner_is_open_immediately_and_ignores_announcements() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test();

        note_window_opened_locally("owner-robot", 0);
        assert!(admits_data("owner-robot", 0));

        // The daemon's view of our own window can only be older than ours, so a
        // stale "closed" must not shut a window this process is holding open.
        assert!(apply_announcement(source("owner-robot"), false).is_none());
        assert!(admits_data("owner-robot", 0));

        // And an owned window never expires: we know first-hand that it lives.
        assert!(expire_stale_leases().is_empty());
        assert!(admits_data("owner-robot", 0));
    }

    #[test]
    fn closing_locally_shuts_the_gate_and_returns_the_source_to_the_daemon() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test();

        note_window_opened_locally("stopping-robot", 0);
        note_window_closed_locally("stopping-robot", 0);

        let state = with_registry(|sources| sources[&source("stopping-robot")]);
        assert!(!state.open, "the gate closes with the window");
        assert!(
            !state.owned,
            "another process may still hold this source open"
        );
        assert!(state.confirmed_at.is_none());
    }

    #[test]
    fn a_never_seen_source_is_refused_and_registered() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test();

        // Refused: nothing has been announced, and refusing is the safe reading.
        assert!(!admits_data("fresh-robot", 0));
        let known = with_registry(|sources| sources.contains_key(&source("fresh-robot")));
        assert!(known, "the source is tracked from its first sample");
    }

    #[test]
    fn admitting_data_never_touches_the_daemon() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test();

        // An announced-open source with no daemon attached: a gate that asked
        // would have nothing to ask. This is the hot path's contract — no IPC
        // inside any `log_*`, ever.
        assert_eq!(
            apply_announcement(source("cached-robot"), true),
            Some(GateTransition::Opened {
                source: source("cached-robot")
            })
        );
        assert!(admits_data("cached-robot", 0));
    }

    #[test]
    fn a_repeat_announcement_is_not_a_transition_but_renews_the_lease() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test();

        assert!(apply_announcement(source("held-robot"), true).is_some());
        let first = with_registry(|sources| sources[&source("held-robot")].confirmed_at);

        // Restating the same state must not read as a change — the daemon
        // restates constantly — but it must push the lease out, or a healthy
        // window would expire under us.
        assert!(
            apply_announcement(source("held-robot"), true).is_none(),
            "a re-announcement of a known state is a no-op"
        );
        let renewed = with_registry(|sources| sources[&source("held-robot")].confirmed_at);
        assert!(renewed >= first, "the lease is renewed, not ignored");
        assert!(admits_data("held-robot", 0));
    }

    #[test]
    fn an_open_gate_expires_when_the_daemon_stops_confirming() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test();

        // A window announced open, then a daemon that died: no close ever
        // arrives. Without expiry this process would spool frames forever with
        // nothing to route them.
        with_registry(|sources| {
            sources.insert(
                source("abandoned-robot"),
                GateState {
                    open: true,
                    owned: false,
                    confirmed_at: Some(Instant::now() - LIVE_LEASE - Duration::from_millis(1)),
                },
            );
        });

        assert_eq!(
            expire_stale_leases(),
            vec![GateTransition::Closed {
                source: source("abandoned-robot")
            }]
        );
        assert!(!admits_data("abandoned-robot", 0));
    }

    #[test]
    fn a_fresh_lease_does_not_expire() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test();

        apply_announcement(source("healthy-robot"), true);
        assert!(
            expire_stale_leases().is_empty(),
            "a window confirmed just now must not be shut"
        );
        assert!(admits_data("healthy-robot", 0));
    }

    #[test]
    fn a_close_announcement_shuts_the_gate_once() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_for_test();

        apply_announcement(source("cycling-robot"), true);
        assert_eq!(
            apply_announcement(source("cycling-robot"), false),
            Some(GateTransition::Closed {
                source: source("cycling-robot")
            })
        );
        assert!(!admits_data("cycling-robot", 0));
        assert!(
            apply_announcement(source("cycling-robot"), false).is_none(),
            "a repeated close is not a second transition"
        );
    }
}
