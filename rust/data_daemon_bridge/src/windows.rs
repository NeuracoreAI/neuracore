//! Which of this process's sources have a recording window open.
//!
//! The producer does not decide this and cannot be authoritative about it: a
//! recording is opened and closed by the daemon, and a source can be logged from
//! several processes at once, only one of which called `start_recording`. So the
//! daemon publishes a snapshot of its live windows on
//! [`RECORDING_WINDOWS`](data_daemon_shared::service_name::RECORDING_WINDOWS)
//! and this module keeps the latest one, for the `log_*` entry points to consult
//! before doing any work.
//!
//! That is the whole reason it exists. A camera process that never brackets a
//! recording has no way to know one is in progress, so it either discards every
//! frame of someone else's recording or spools every frame it ever captures.
//! Reading the daemon's own answer is the third option.
//!
//! ## Two sources of truth, deliberately
//!
//! [`is_open`] answers from the snapshot *or* from a claim this process made
//! itself, and the claim is what covers the round trip: `start_recording`
//! publishes its envelope and claims the source here in the same breath, so the
//! very next `log_*` on the calling thread is never dropped waiting for the
//! daemon to answer. The claim then expires on its own — the first snapshot
//! taken *after* it supersedes it, whether or not the source appears in that
//! snapshot. Which means a recording stopped by someone else closes this gate
//! without anything having to tell this process so.
//!
//! ## First call in a process
//!
//! The subscriber starts on the first [`is_open`], so that call answers from
//! claims alone — a process whose very first sample lands inside a recording
//! someone else started drops it, and is correct from the next one. One sample
//! once per process, at process start rather than at a recording boundary; the
//! alternative is blocking a `log_*` under the GIL on an IPC round trip.
//!
//! ## Fork safety
//!
//! The registry stores the pid that owns it and wipes itself on first use from a
//! different process, exactly like the video-chunk registry: a forked
//! `multiprocessing` child inherits the parent's state and its dead subscriber
//! thread, and must start from nothing.

use std::sync::{LazyLock, Mutex};
use std::thread;
use std::time::Duration;

use data_daemon_shared::service_name::{
    MAX_NODES_PER_SERVICE, RECORDING_WINDOWS, RECORDING_WINDOWS_HISTORY_SIZE,
    RECORDING_WINDOWS_MAX_PUBLISHERS, RECORDING_WINDOWS_MAX_SUBSCRIBERS,
    RECORDING_WINDOWS_SUBSCRIBER_BUFFER_SIZE,
};
use data_daemon_shared::RecordingWindows;
use iceoryx2::node::{Node, NodeBuilder};
use iceoryx2::port::subscriber::Subscriber;
use iceoryx2::prelude::ipc;
use iceoryx2::service::port_factory::publish_subscribe::PortFactory;

/// How often the subscriber thread checks for a new snapshot.
///
/// The daemon publishes only when its window set changes, so this is pure
/// overhead while nothing happens — but it is also the delay before a window
/// opened by *another* process becomes visible here, and every frame captured in
/// that delay is dropped. 10 ms keeps it under a third of a frame interval at
/// 30 fps while waking the thread only 100 times a second.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// One source, with the publish time it was opened or claimed at.
///
/// Held in `Vec`s and scanned linearly rather than hashed: a machine hosts a
/// handful of sources, and `is_open` sits on the `log_*` hot path where hashing
/// a `(String, i64)` key would mean allocating one per call.
struct Entry {
    robot_id: String,
    robot_instance: i64,
    at_ns: i64,
}

impl Entry {
    fn matches(&self, robot_id: &str, robot_instance: i64) -> bool {
        self.robot_instance == robot_instance && self.robot_id == robot_id
    }
}

struct Registry {
    owner_pid: u32,
    /// Sources the newest snapshot says are open.
    pushed: Vec<Entry>,
    /// Daemon clock of that snapshot. Zero until the first one arrives, so no
    /// claim is ever superseded by a snapshot that does not exist.
    pushed_at_ns: i64,
    /// Sources this process opened itself by publishing a `StartRecording`.
    claimed: Vec<Entry>,
    /// Whether this process's subscriber thread is running.
    subscribed: bool,
}

static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| {
    Mutex::new(Registry {
        owner_pid: 0,
        pushed: Vec::new(),
        pushed_at_ns: 0,
        claimed: Vec::new(),
        subscribed: false,
    })
});

/// Lock the registry, healing it across a `fork()` first.
fn with_registry<R>(operation: impl FnOnce(&mut Registry) -> R) -> R {
    let mut registry = REGISTRY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let pid = std::process::id();
    if registry.owner_pid != pid {
        registry.owner_pid = pid;
        registry.pushed.clear();
        registry.pushed_at_ns = 0;
        registry.claimed.clear();
        // The parent's thread did not survive the fork; the child needs its own.
        registry.subscribed = false;
    }
    operation(&mut registry)
}

/// Does the source have a recording window open, as far as this process can
/// tell?
///
/// Consulted by every `log_*` before it copies, spools or publishes anything, so
/// it is one mutex acquisition and two short scans and nothing more. Answers
/// `false` when the daemon has never spoken and this process has claimed
/// nothing — the correct answer, and the one that keeps an idle camera from
/// spooling.
pub(crate) fn is_open(robot_id: &str, robot_instance: i64) -> bool {
    let (open, start_subscriber) = with_registry(|registry| {
        let open = registry
            .pushed
            .iter()
            .any(|entry| entry.matches(robot_id, robot_instance))
            // A claim outlives only the snapshots that predate it.
            || registry
                .claimed
                .iter()
                .any(|entry| entry.matches(robot_id, robot_instance) && entry.at_ns >= registry.pushed_at_ns);
        (open, take_subscribe_duty(registry))
    });
    if start_subscriber {
        spawn_subscriber();
    }
    open
}

/// Record that this process has just published a `StartRecording` for the
/// source, so its own logging is not gated on the daemon answering.
///
/// `publish_timestamp_ns` must be the stamp carried on that envelope: it is what
/// decides which snapshots are new enough to supersede the claim.
pub(crate) fn claim(robot_id: &str, robot_instance: i64, publish_timestamp_ns: i64) {
    let start_subscriber = with_registry(|registry| {
        registry
            .claimed
            .retain(|entry| !entry.matches(robot_id, robot_instance));
        registry.claimed.push(Entry {
            robot_id: robot_id.to_string(),
            robot_instance,
            at_ns: publish_timestamp_ns,
        });
        take_subscribe_duty(registry)
    });
    if start_subscriber {
        spawn_subscriber();
    }
}

/// Drop this process's claim on the source, after publishing its stop.
///
/// Only the claim goes: if another process still has the recording open the
/// daemon's snapshot says so, and this process keeps logging into it.
pub(crate) fn release(robot_id: &str, robot_instance: i64) {
    with_registry(|registry| {
        registry
            .claimed
            .retain(|entry| !entry.matches(robot_id, robot_instance));
    });
}

/// Replace the pushed snapshot with a newer one.
fn apply_snapshot(snapshot: RecordingWindows) {
    with_registry(|registry| {
        // iceoryx2 delivers in order per publisher, but the guard is free and
        // keeps the registry monotonic if that ever stops being true.
        if snapshot.published_at_ns < registry.pushed_at_ns {
            return;
        }
        registry.pushed = snapshot
            .windows
            .into_iter()
            .map(|window| Entry {
                robot_id: window.robot_id,
                robot_instance: window.robot_instance,
                at_ns: window.started_at_publish_ns,
            })
            .collect();
        registry.pushed_at_ns = snapshot.published_at_ns;
    });
}

/// Claim the duty of starting this process's subscriber thread, if nobody has.
/// The caller must hold the registry lock, and must spawn once it has released
/// it — the thread's first act is to take the same lock.
fn take_subscribe_duty(registry: &mut Registry) -> bool {
    let duty = !registry.subscribed;
    registry.subscribed = true;
    duty
}

fn spawn_subscriber() {
    if let Err(error) = thread::Builder::new()
        .name("nc-windows".to_string())
        .spawn(subscribe_loop)
    {
        tracing::debug!(%error, "failed to start recording-window subscriber");
        with_registry(|registry| registry.subscribed = false);
    }
}

/// The subscriber's iceoryx2 ports. The node and service handle must outlive the
/// subscriber built off them, so the loop owns all three together.
struct WindowSubscriber {
    _node: Node<ipc::Service>,
    _service: PortFactory<ipc::Service, [u8], ()>,
    subscriber: Subscriber<ipc::Service, [u8], ()>,
}

/// Poll the `recording_windows` service for as long as the process lives.
///
/// A failure to open the service is not retried: the daemon seeds it at startup,
/// so failing here means no daemon is running, and a process logging with no
/// daemon has nowhere for its data to go regardless. Its own claims still open
/// its gate, which is what keeps a locally started recording working while the
/// daemon is still coming up.
fn subscribe_loop() {
    let ports = match open_subscriber() {
        Ok(ports) => ports,
        Err(error) => {
            tracing::warn!(%error, "recording-window subscriber unavailable");
            with_registry(|registry| registry.subscribed = false);
            return;
        }
    };
    loop {
        // Drain everything pending and keep only the last: a snapshot is
        // absolute, so an older one has nothing left to say.
        let mut newest = None;
        loop {
            match ports.subscriber.receive() {
                Ok(Some(sample)) => match RecordingWindows::decode(sample.payload()) {
                    Ok(snapshot) => newest = Some(snapshot),
                    Err(error) => {
                        tracing::warn!(%error, "recording-window snapshot decode failed")
                    }
                },
                Ok(None) => break,
                Err(error) => {
                    tracing::warn!(%error, "recording-window receive failed");
                    break;
                }
            }
        }
        if let Some(snapshot) = newest {
            apply_snapshot(snapshot);
        }
        thread::sleep(POLL_INTERVAL);
    }
}

/// Open (or attach to) the `recording_windows` service and build this process's
/// subscriber. Every attribute must match the daemon's publisher side, which
/// `open_or_create` reconciles against whichever party came up first.
fn open_subscriber() -> Result<WindowSubscriber, String> {
    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .map_err(|error| error.to_string())?;
    let service_name = RECORDING_WINDOWS
        .try_into()
        .map_err(|error| format!("invalid service name: {error}"))?;
    let service = node
        .service_builder(&service_name)
        .publish_subscribe::<[u8]>()
        .history_size(RECORDING_WINDOWS_HISTORY_SIZE)
        .subscriber_max_buffer_size(RECORDING_WINDOWS_SUBSCRIBER_BUFFER_SIZE)
        .max_publishers(RECORDING_WINDOWS_MAX_PUBLISHERS)
        .max_subscribers(RECORDING_WINDOWS_MAX_SUBSCRIBERS)
        .max_nodes(MAX_NODES_PER_SERVICE)
        .open_or_create()
        .map_err(|error| error.to_string())?;
    let subscriber = service
        .subscriber_builder()
        .create()
        .map_err(|error| error.to_string())?;
    Ok(WindowSubscriber {
        _node: node,
        _service: service,
        subscriber,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use data_daemon_shared::OpenWindow;

    /// The registry is process-wide and a snapshot replaces all of it, so these
    /// tests cannot run concurrently with each other. Held for the body of each.
    static SERIALISE: Mutex<()> = Mutex::new(());

    /// Take the test lock and clear the registry behind it.
    fn isolated() -> std::sync::MutexGuard<'static, ()> {
        let guard = SERIALISE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        with_registry(|registry| {
            registry.pushed.clear();
            registry.pushed_at_ns = 0;
            registry.claimed.clear();
        });
        guard
    }

    fn snapshot(sources: &[(&str, i64)], published_at_ns: i64) -> RecordingWindows {
        RecordingWindows {
            windows: sources
                .iter()
                .map(|(robot_id, robot_instance)| OpenWindow {
                    robot_id: (*robot_id).to_string(),
                    robot_instance: *robot_instance,
                    started_at_publish_ns: published_at_ns,
                })
                .collect(),
            published_at_ns,
        }
    }

    /// The daemon's side of the window channel, kept alive by the test.
    ///
    /// The publisher owns the service history, so a short-lived one would take
    /// the snapshot with it and a subscriber that connects a moment later would
    /// see nothing — exactly the daemon's own arrangement, where the publisher
    /// lives as long as the listener loop.
    struct DaemonSide {
        _node: Node<ipc::Service>,
        _service: PortFactory<ipc::Service, [u8], ()>,
        publisher: iceoryx2::port::publisher::Publisher<ipc::Service, [u8], ()>,
    }

    /// Open the daemon's publisher exactly as `ipc::node` does, so the two
    /// service configurations have to reconcile for real.
    fn daemon_side() -> Result<DaemonSide, String> {
        use data_daemon_shared::service_name::RECORDING_WINDOWS_MAX_PAYLOAD_BYTES;
        use iceoryx2::prelude::UnableToDeliverStrategy;

        let node = NodeBuilder::new()
            .create::<ipc::Service>()
            .map_err(|error| error.to_string())?;
        let service_name = RECORDING_WINDOWS
            .try_into()
            .map_err(|error| format!("{error}"))?;
        let service = node
            .service_builder(&service_name)
            .publish_subscribe::<[u8]>()
            .history_size(RECORDING_WINDOWS_HISTORY_SIZE)
            .subscriber_max_buffer_size(RECORDING_WINDOWS_SUBSCRIBER_BUFFER_SIZE)
            .max_publishers(RECORDING_WINDOWS_MAX_PUBLISHERS)
            .max_subscribers(RECORDING_WINDOWS_MAX_SUBSCRIBERS)
            .max_nodes(MAX_NODES_PER_SERVICE)
            .open_or_create()
            .map_err(|error| error.to_string())?;
        let publisher = service
            .publisher_builder()
            .initial_max_slice_len(RECORDING_WINDOWS_MAX_PAYLOAD_BYTES)
            .unable_to_deliver_strategy(UnableToDeliverStrategy::DiscardSample)
            .create()
            // The daemon holds the only publisher slot, so this is what a
            // running data-daemon looks like from here.
            .map_err(|error| format!("{error} (is a data-daemon running?)"))?;
        Ok(DaemonSide {
            _node: node,
            _service: service,
            publisher,
        })
    }

    impl DaemonSide {
        fn publish(&self, snapshot: &RecordingWindows) {
            let bytes = snapshot.encode().expect("encode");
            let sample = self
                .publisher
                .loan_slice_uninit(bytes.len())
                .expect("loan a window sample");
            sample.write_from_slice(&bytes).send().expect("send");
        }
    }

    #[test]
    fn a_snapshot_published_over_real_ipc_opens_the_gate() {
        // Everything above this test drives `apply_snapshot` directly, which
        // proves the decision but not the transport: the daemon's publisher and
        // this subscriber configure the same service independently and
        // `open_or_create` has to reconcile them, so a mismatch would leave
        // every producer permanently gated shut with nothing logged anywhere.
        let _guard = isolated();

        let daemon = daemon_side().expect("open the daemon's publisher");
        let live = snapshot(&[("ipc-robot", 4)], 1_000);

        // Starts this process's subscriber; false, since nothing has been said.
        assert!(!is_open("ipc-robot", 4));

        // Re-sent on a cadence, as the daemon's listener does. A single send
        // would not do: iceoryx2 hands a new subscriber the service history only
        // when the publisher next sends, so a snapshot published before this
        // subscriber's port existed would never arrive at all.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while std::time::Instant::now() < deadline {
            daemon.publish(&live);
            if is_open("ipc-robot", 4) {
                return;
            }
            thread::sleep(POLL_INTERVAL);
        }
        panic!("the subscriber never received the daemon's snapshot");
    }

    #[test]
    fn a_source_nothing_has_said_anything_about_is_closed() {
        let _guard = isolated();
        assert!(!is_open("silent-robot", 0));
    }

    #[test]
    fn a_pushed_window_opens_a_source_this_process_never_started() {
        // The camera child: it published no lifecycle envelope of its own and
        // learns of the recording only from the daemon.
        let _guard = isolated();
        apply_snapshot(snapshot(&[("pushed-robot", 2)], 1_000));
        assert!(is_open("pushed-robot", 2));
        assert!(
            !is_open("pushed-robot", 3),
            "a sibling instance is separate"
        );
    }

    #[test]
    fn a_claim_opens_the_gate_before_any_snapshot_arrives() {
        // The round trip this covers: start_recording has published, the daemon
        // has not answered yet, and the next log_* must not be dropped.
        let _guard = isolated();
        claim("claiming-robot", 0, 5_000);
        assert!(is_open("claiming-robot", 0));
    }

    #[test]
    fn a_snapshot_taken_after_a_claim_supersedes_it() {
        // How a recording stopped by someone else closes this gate: the daemon
        // simply stops naming the source, and the claim no longer counts.
        let _guard = isolated();
        claim("superseded-robot", 0, 5_000);
        apply_snapshot(snapshot(&[], 6_000));
        assert!(!is_open("superseded-robot", 0));
    }

    #[test]
    fn a_snapshot_taken_before_a_claim_does_not_supersede_it() {
        let _guard = isolated();
        apply_snapshot(snapshot(&[], 4_000));
        claim("early-snapshot-robot", 0, 5_000);
        assert!(
            is_open("early-snapshot-robot", 0),
            "the snapshot predates the claim, so it cannot speak for it"
        );
    }

    #[test]
    fn releasing_a_claim_closes_a_gate_no_snapshot_holds_open() {
        let _guard = isolated();
        claim("released-robot", 0, 5_000);
        release("released-robot", 0);
        assert!(!is_open("released-robot", 0));
    }

    #[test]
    fn releasing_a_claim_leaves_a_pushed_window_open() {
        // This process stopped its own recording; another process still has one
        // open for the source, and the daemon says so.
        let _guard = isolated();
        claim("shared-robot", 0, 5_000);
        apply_snapshot(snapshot(&[("shared-robot", 0)], 6_000));
        release("shared-robot", 0);
        assert!(is_open("shared-robot", 0));
    }

    #[test]
    fn an_older_snapshot_cannot_move_the_registry_backwards() {
        let _guard = isolated();
        apply_snapshot(snapshot(&[("ordered-robot", 0)], 9_000));
        apply_snapshot(snapshot(&[], 8_000));
        assert!(is_open("ordered-robot", 0));
    }
}
