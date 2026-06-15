//! IPC plumbing: per-thread iceoryx2 publisher state and the background data
//! publisher thread.
//!
//! ## Threading
//!
//! iceoryx2's [`Publisher`] is neither `Send` nor `Sync`, so it is parked in a
//! [`thread_local`]: each Python thread that calls in lazily builds its own
//! iceoryx2 [`Node`] and a publisher on the commands service.
//!
//! ## Fork safety
//!
//! A one-shot `pthread_atfork` child handler clears the forking thread's
//! `PRODUCER` slot so the next publish rebuilds.
//!
//! ## Background data publisher
//!
//! Synchronous IPC publishes (`BatchedData` joints, `Data` json, and the
//! `VideoChunkReady` chunk announcements) can briefly block on a full commands
//! buffer when the daemon's listener is preempted off-CPU under heavy
//! (multi-context) load. Routing them through a dedicated per-process publisher
//! thread keeps that block off BOTH the caller's `log_*` thread AND the disk
//! writer thread — crucially, the writer's stop/cancel *barrier* then waits only
//! for the durable on-disk seal, never for an IPC publish.
//!
//! All three are held-back *data* on the daemon side (routed by
//! `publish_timestamp_ns` within the holdback + closing-window retention), so —
//! unlike lifecycle envelopes (`Start/Stop/CancelRecording`, which stay on the
//! caller's publisher for strict ordering) — reordering them onto this thread's
//! publisher is safe. The queue is unbounded: messages are small/infrequent and
//! the thread keeps up in steady state; a transient daemon stall buffers only a
//! few hundred small items.

use std::cell::RefCell;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{LazyLock, Mutex, Once};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use data_daemon_ipc::service_name::{
    COMMANDS, COMMANDS_MAX_PAYLOAD_BYTES, LIFECYCLE_SUBSCRIBER_BUFFER_SIZE, MAX_NODES_PER_SERVICE,
    MAX_PUBLISHERS_PER_SERVICE, MAX_QUERY_CLIENTS_PER_SERVICE, MAX_QUERY_SERVERS_PER_SERVICE,
    MAX_SUBSCRIBERS_PER_SERVICE, QUERIES, QUERIES_MAX_PAYLOAD_BYTES,
};
use data_daemon_ipc::{BatchedDataItem, Envelope};
use iceoryx2::node::{Node, NodeBuilder};
use iceoryx2::port::client::Client;
use iceoryx2::port::publisher::Publisher;
use iceoryx2::prelude::{ipc, UnableToDeliverStrategy};
use iceoryx2::service::port_factory::publish_subscribe::PortFactory;
use iceoryx2::service::port_factory::request_response::PortFactory as QueryPortFactory;
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;
use thiserror::Error;

/// Errors raised while publishing envelopes to the daemon.
#[derive(Debug, Error)]
pub(crate) enum ProducerError {
    /// Failed to build the iceoryx2 node.
    #[error("failed to create iceoryx2 node: {0}")]
    NodeCreate(String),
    /// Failed to open or create an iceoryx2 service.
    #[error("failed to open service: {0}")]
    ServiceOpen(String),
    /// Failed to build the publisher port.
    #[error("failed to create publisher: {0}")]
    PublisherCreate(String),
    /// Failed to loan a slice sample.
    #[error("failed to loan sample: {0}")]
    Loan(String),
    /// Failed to send the loaned sample.
    #[error("failed to send sample: {0}")]
    Send(String),
    /// Failed to encode the envelope.
    #[error(transparent)]
    Encode(#[from] data_daemon_ipc::EnvelopeCodecError),
    /// Payload too large for the configured iceoryx2 max slice length.
    #[error("envelope payload {actual} bytes exceeds limit {limit} bytes")]
    PayloadTooLarge {
        /// Actual encoded envelope size.
        actual: usize,
        /// Maximum slice length the publisher was built with.
        limit: usize,
    },
}

impl From<ProducerError> for PyErr {
    fn from(error: ProducerError) -> Self {
        PyRuntimeError::new_err(error.to_string())
    }
}

/// Per-thread iceoryx2 state.
pub(crate) struct ProducerState {
    _node: Node<ipc::Service>,
    _commands_service: PortFactory<ipc::Service, [u8], ()>,
    commands_publisher: Publisher<ipc::Service, [u8], ()>,
    /// Service handle held alongside the query client so port discovery doesn't
    /// race the handle going out of scope.
    _queries_service: QueryPortFactory<ipc::Service, [u8], (), [u8], ()>,
    /// Request-response client used by `get_recording_id` to ask the daemon
    /// for a recording's cloud id.
    pub(crate) queries_client: Client<ipc::Service, [u8], (), [u8], ()>,
}

/// Work item for the publisher thread.
pub(crate) enum PublishMsg {
    /// A batch of joint `(name, value)` samples to serialise + publish as one
    /// `BatchedData`. Serialisation happens on the publisher thread, off the
    /// caller.
    Joint {
        robot_id: String,
        robot_instance: i64,
        data_type: String,
        items: Vec<(String, f64)>,
        timestamp_ns: i64,
        timestamp_s: Option<f64>,
        publish_timestamp_ns: i64,
    },
    /// One JSON sample to publish as a `Data` envelope.
    Json {
        robot_id: String,
        robot_instance: i64,
        data_type: String,
        sensor_name: String,
        payload: Vec<u8>,
        timestamp_ns: i64,
        timestamp_s: Option<f64>,
        publish_timestamp_ns: i64,
    },
    /// A pre-built `VideoChunkReady` envelope to announce (built by the writer
    /// thread once the chunk is sealed on disk).
    Announce(Envelope),
}

/// Process-wide publisher handle, healed across `fork` via `owner_pid` (mirrors
/// `VIDEO_WRITER`).
struct PublisherRegistry {
    owner_pid: u32,
    tx: Option<Sender<PublishMsg>>,
}

static PUBLISHER: LazyLock<Mutex<PublisherRegistry>> = LazyLock::new(|| {
    Mutex::new(PublisherRegistry {
        owner_pid: 0,
        tx: None,
    })
});

thread_local! {
    /// Per-thread cache of the process publisher channel. The hot `log_*` path
    /// hits this slot — a plain TLS load with no global `Mutex` and no
    /// `getpid()` syscall (glibc removed the pid cache, so `process::id()` is a
    /// real syscall on every call). Const-initialised and cleared by
    /// `on_fork_in_child` alongside `PRODUCER`, so a forked child rebuilds.
    static PUBLISHER_TX: RefCell<Option<Sender<PublishMsg>>> = const { RefCell::new(None) };
}

/// Return this process's publisher channel, spawning the publisher thread on
/// first use and re-spawning after a fork.
///
/// Fast path: the thread-local cache (no lock, no syscall). Slow path (first
/// call on a thread, or first call after a fork cleared the slot): heal/spawn
/// the process publisher under the global lock and cache the channel.
pub(crate) fn publisher_tx() -> Sender<PublishMsg> {
    if let Some(tx) = PUBLISHER_TX.with(|cell| cell.borrow().clone()) {
        return tx;
    }
    let tx = publisher_tx_global();
    PUBLISHER_TX.with(|cell| *cell.borrow_mut() = Some(tx.clone()));
    tx
}

/// Heal/spawn the process-wide publisher thread under the global lock and
/// return its channel. Keyed by `owner_pid` so a post-fork child re-spawns.
fn publisher_tx_global() -> Sender<PublishMsg> {
    let mut reg = PUBLISHER.lock().unwrap_or_else(|p| p.into_inner());
    let pid = std::process::id();
    if reg.owner_pid == pid {
        if let Some(tx) = reg.tx.as_ref() {
            return tx.clone();
        }
    }
    let (tx, rx) = std::sync::mpsc::channel();
    match std::thread::Builder::new()
        .name("nc-data-publisher".to_string())
        .spawn(move || publish_loop(rx))
    {
        Ok(_handle) => {
            reg.owner_pid = pid;
            reg.tx = Some(tx.clone());
        }
        Err(error) => {
            tracing::error!(%error, "failed to spawn data publisher thread; dropping sample");
        }
    }
    tx
}

/// The publisher thread's run loop: publish every queued data envelope. Exits
/// when the last [`Sender`] is dropped (the channel closes).
fn publish_loop(rx: Receiver<PublishMsg>) {
    while let Ok(msg) = rx.recv() {
        let result = match msg {
            PublishMsg::Joint {
                robot_id,
                robot_instance,
                data_type,
                items,
                timestamp_ns,
                timestamp_s,
                publish_timestamp_ns,
            } => {
                let timestamp_for_json =
                    timestamp_s.unwrap_or_else(|| timestamp_ns as f64 / 1_000_000_000.0);
                let mut batch_items = Vec::with_capacity(items.len());
                for (name, value) in items {
                    match serde_json::to_vec(&ScalarFrameEntry {
                        timestamp: timestamp_for_json,
                        value,
                    }) {
                        Ok(payload) => batch_items.push(BatchedDataItem {
                            sensor_name: Some(name),
                            payload,
                        }),
                        Err(error) => {
                            tracing::warn!(%error, "failed to encode joint frame JSON; dropping item")
                        }
                    }
                }
                publish(&Envelope::BatchedData {
                    robot_id,
                    robot_instance,
                    data_type,
                    publish_timestamp_ns,
                    timestamp_ns,
                    timestamp_s,
                    items: batch_items,
                })
            }
            PublishMsg::Json {
                robot_id,
                robot_instance,
                data_type,
                sensor_name,
                payload,
                timestamp_ns,
                timestamp_s,
                publish_timestamp_ns,
            } => publish(&Envelope::Data {
                robot_id,
                robot_instance,
                data_type,
                sensor_name: Some(sensor_name),
                publish_timestamp_ns,
                timestamp_ns,
                timestamp_s,
                payload,
            }),
            PublishMsg::Announce(envelope) => publish(&envelope),
        };
        if let Err(error) = result {
            tracing::warn!(%error, "failed to publish data envelope");
        }
    }
}

thread_local! {
    /// One iceoryx2 publisher set per OS thread. Const-initialised so the slot
    /// is a plain TLS load — required for the `pthread_atfork` child handler to
    /// access it without invoking a lazy initializer in a post-fork context.
    static PRODUCER: RefCell<Option<ProducerState>> = const { RefCell::new(None) };
}

/// Run `f` against this thread's producer state, lazily building it on first
/// use.
pub(crate) fn with_producer<R>(
    operation: impl FnOnce(&ProducerState) -> Result<R, ProducerError>,
) -> Result<R, ProducerError> {
    PRODUCER.with(|cell| {
        // One borrow for the whole operation: build-if-empty then run. The
        // previous `borrow().is_none()` + separate `borrow()` took two RefCell
        // borrows per publish.
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(build_producer_state()?);
        }
        operation(
            slot.as_ref()
                .expect("producer state populated immediately above"),
        )
    })
}

fn build_producer_state() -> Result<ProducerState, ProducerError> {
    ensure_fork_handler_registered();

    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .map_err(|error| ProducerError::NodeCreate(error.to_string()))?;

    let (commands_service, commands_publisher) = open_publisher(
        &node,
        COMMANDS,
        LIFECYCLE_SUBSCRIBER_BUFFER_SIZE,
        COMMANDS_MAX_PAYLOAD_BYTES,
    )?;

    let (queries_service, queries_client) = open_query_client(&node, QUERIES)?;

    Ok(ProducerState {
        _node: node,
        _commands_service: commands_service,
        commands_publisher,
        _queries_service: queries_service,
        queries_client,
    })
}

/// Open (or attach to) the `[u8]` request-response `queries` service off `node`
/// and build a client on it. Config mirrors the daemon's `open_query_server`
/// so `open_or_create` reconciles to the same service attributes regardless of
/// which side comes up first.
#[allow(clippy::type_complexity)]
fn open_query_client(
    node: &Node<ipc::Service>,
    service_name: &str,
) -> Result<
    (
        QueryPortFactory<ipc::Service, [u8], (), [u8], ()>,
        Client<ipc::Service, [u8], (), [u8], ()>,
    ),
    ProducerError,
> {
    let parsed_name = service_name
        .try_into()
        .map_err(|error| ProducerError::ServiceOpen(format!("invalid service name: {error}")))?;
    let service = node
        .service_builder(&parsed_name)
        .request_response::<[u8], [u8]>()
        .max_clients(MAX_QUERY_CLIENTS_PER_SERVICE)
        .max_servers(MAX_QUERY_SERVERS_PER_SERVICE)
        .max_nodes(MAX_NODES_PER_SERVICE)
        .open_or_create()
        .map_err(|error| ProducerError::ServiceOpen(error.to_string()))?;
    let client = service
        .client_builder()
        .initial_max_slice_len(QUERIES_MAX_PAYLOAD_BYTES)
        .create()
        .map_err(|error| ProducerError::PublisherCreate(error.to_string()))?;
    Ok((service, client))
}

/// Open (or attach to) one `[u8]` pub/sub service off `node` and build a
/// publisher on it.
#[allow(clippy::type_complexity)]
fn open_publisher(
    node: &Node<ipc::Service>,
    service_name: &str,
    subscriber_buffer_size: usize,
    max_slice_len: usize,
) -> Result<
    (
        PortFactory<ipc::Service, [u8], ()>,
        Publisher<ipc::Service, [u8], ()>,
    ),
    ProducerError,
> {
    let parsed_name = service_name
        .try_into()
        .map_err(|error| ProducerError::ServiceOpen(format!("invalid service name: {error}")))?;
    let service = node
        .service_builder(&parsed_name)
        .publish_subscribe::<[u8]>()
        // Disable iceoryx2's default safe-overflow so a full subscriber buffer
        // makes `Block` take effect rather than silently evicting the oldest
        // sample. Must match the daemon's `open_subscriber`.
        .enable_safe_overflow(false)
        .subscriber_max_buffer_size(subscriber_buffer_size)
        .max_publishers(MAX_PUBLISHERS_PER_SERVICE)
        .max_subscribers(MAX_SUBSCRIBERS_PER_SERVICE)
        .max_nodes(MAX_NODES_PER_SERVICE)
        .open_or_create()
        .map_err(|error| ProducerError::ServiceOpen(error.to_string()))?;
    let publisher = service
        .publisher_builder()
        .initial_max_slice_len(max_slice_len)
        .unable_to_deliver_strategy(UnableToDeliverStrategy::Block)
        .create()
        .map_err(|error| ProducerError::PublisherCreate(error.to_string()))?;
    Ok((service, publisher))
}

/// Install the `pthread_atfork` child handler exactly once per process.
fn ensure_fork_handler_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // SAFETY: `pthread_atfork` is the standard libc primitive for
        // registering fork callbacks. `on_fork_in_child` is `extern "C"`,
        // touches only a const-initialised TLS slot, and the only "work" it
        // does is `mem::forget`.
        let result = unsafe { libc::pthread_atfork(None, None, Some(on_fork_in_child)) };
        if result != 0 {
            tracing::warn!(
                errno = result,
                "pthread_atfork registration failed; fork-safety relies on caller-managed cleanup",
            );
        }
    });
}

/// `pthread_atfork` child callback: clears the surviving thread's `PRODUCER`
/// slot so the next [`with_producer`] rebuilds fresh iceoryx2 publishers. The
/// inherited state is `mem::forget`'d on purpose (running its `Drop` would
/// touch the parent's bookkeeping). The video chunk registry self-heals via the
/// `owner_pid` check.
extern "C" fn on_fork_in_child() {
    PRODUCER.with(|cell| {
        if let Some(stale) = cell.borrow_mut().take() {
            std::mem::forget(stale);
        }
    });
    // Drop the cached publisher channel so the next `publisher_tx` rebuilds
    // through the global (pid-keyed) heal path. The `Sender` is a plain mpsc
    // handle, so a normal drop is safe here (no parent-side bookkeeping to
    // corrupt, unlike `PRODUCER`'s iceoryx2 ports).
    PUBLISHER_TX.with(|cell| {
        cell.borrow_mut().take();
    });
}

/// Producer wall-clock time in nanoseconds since the Unix epoch, stamped onto
/// every published data envelope as its `publish_timestamp_ns`. This is the
/// daemon's sole window-membership key, decoupled from whatever clock the
/// caller timestamps data with. The lifecycle `StartRecording` / `StopRecording`
/// envelopes carry the same publish clock as their `publish_timestamp_ns`, so
/// window boundaries and data are directly comparable.
pub(crate) fn now_ns() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(elapsed) => elapsed.as_nanos() as i64,
        // The system clock is set before the Unix epoch (a mis-set RTC). A 0
        // here is the worst possible value: it routes the datum before *every*
        // window → silent orphan-drop, indistinguishable from a real sample.
        // Fall back to a positive, strictly-increasing monotonic-anchored value
        // so the datum still lands in a window rather than vanishing.
        Err(_) => clock_fallback_ns(),
    }
}

/// Positive, strictly-increasing fallback for [`now_ns`] when the wall clock is
/// unusable. Anchored to a fixed epoch base plus a process-monotonic offset, so
/// every envelope a mis-clocked process emits stays mutually comparable.
fn clock_fallback_ns() -> i64 {
    /// Monotonic anchor captured on first use.
    static ANCHOR: LazyLock<Instant> = LazyLock::new(Instant::now);
    /// 2024-01-01T00:00:00Z in epoch-ns — an arbitrary but sane positive base.
    const BASE_NS: i64 = 1_704_067_200_000_000_000;
    BASE_NS.saturating_add(ANCHOR.elapsed().as_nanos() as i64)
}

/// Encode `envelope` and publish it on the commands service.
pub(crate) fn publish(envelope: &Envelope) -> Result<(), ProducerError> {
    let bytes = envelope.encode()?;
    if bytes.len() > COMMANDS_MAX_PAYLOAD_BYTES {
        return Err(ProducerError::PayloadTooLarge {
            actual: bytes.len(),
            limit: COMMANDS_MAX_PAYLOAD_BYTES,
        });
    }
    with_producer(|state| {
        let publisher = &state.commands_publisher;
        let sample = publisher
            .loan_slice_uninit(bytes.len())
            .map_err(|error| ProducerError::Loan(error.to_string()))?;
        let sample = sample.write_from_slice(&bytes);
        sample
            .send()
            .map_err(|error| ProducerError::Send(error.to_string()))?;
        Ok(())
    })
}

/// Per-item JSON shape written to `trace.json` for scalar joint streams.
#[derive(serde::Serialize)]
struct ScalarFrameEntry {
    timestamp: f64,
    value: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the joint-scalar float serialisation to Python's `json.dumps` shape:
    /// serde_json (ryū) emits the shortest round-trip with at least one
    /// fractional digit, so an integer-valued float keeps its `.0` rather than
    /// collapsing to an int. The cloud-side data verification compares this text
    /// exactly, so a silent change here would break it (see this crate's
    /// Cargo.toml note on why we serialise via serde_json, not `write!`).
    #[test]
    fn scalar_frame_entry_float_repr_matches_python_json_dumps() {
        let cases = [
            (1.0_f64, 0.5_f64, r#"{"timestamp":1.0,"value":0.5}"#),
            (2.0_f64, -0.25_f64, r#"{"timestamp":2.0,"value":-0.25}"#),
            (0.0_f64, 1.0_f64, r#"{"timestamp":0.0,"value":1.0}"#),
        ];
        for (timestamp, value, expected) in cases {
            let bytes = serde_json::to_vec(&ScalarFrameEntry { timestamp, value }).expect("encode");
            assert_eq!(String::from_utf8(bytes).unwrap(), expected);
        }
    }
}
