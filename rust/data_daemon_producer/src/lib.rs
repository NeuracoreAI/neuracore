// PyO3 0.22's `#[pyfunction]` expansion includes an `.into()` on the
// `PyResult<T>` return value that fires clippy's `useless_conversion` lint
// when T resolves to `()`. The lint is correct about the generated code but
// the conversion lives in the macro expansion, not anything we wrote, so we
// silence it at the crate level rather than spraying allows over every
// `#[pyfunction]`.
#![allow(clippy::useless_conversion)]

//! PyO3 producer client for the Neuracore data daemon.
//!
//! This crate ships as `neuracore.data_daemon._native_producer` inside the
//! Python wheel. It exposes a small recording-scoped surface to the SDK's
//! logging layer:
//!
//! - [`start_recording`](crate::start_recording) publishes one
//!   [`StartRecording`](data_daemon_ipc::Envelope::StartRecording) envelope.
//! - [`log_joints`](crate::log_joints) / [`log_frame`](crate::log_frame) /
//!   [`log_scalar`](crate::log_scalar) deliver data. Each *lazily* mints a
//!   trace the first time a stream is seen — publishing its
//!   [`StartTrace`](data_daemon_ipc::Envelope::StartTrace) (and, for video,
//!   [`OpenFrameStream`](data_daemon_ipc::Envelope::OpenFrameStream)) — then
//!   publishes the sample.
//! - [`stop_recording`](crate::stop_recording) ends every trace the recording
//!   minted and publishes one
//!   [`StopRecording`](data_daemon_ipc::Envelope::StopRecording).
//! - [`cancel_recording`](crate::cancel_recording) publishes
//!   [`CancelRecording`](data_daemon_ipc::Envelope::CancelRecording).
//!
//! The crate owns the entire trace lifecycle: trace ids are minted, tracked,
//! and discarded here. Python never sees a trace id — it logs by recording id
//! plus a `(data_type, name)` stream identity.
//!
//! ## Threading
//!
//! iceoryx2's [`Publisher`] uses an `Rc`-backed `ArcSyncPolicy` and is
//! therefore neither `Send` nor `Sync`. We side-step that by parking the
//! publishers in a [`thread_local`]: each Python thread that calls in lazily
//! builds its own iceoryx2 [`Node`] and a publisher per service.
//!
//! The *trace registry* — which streams map to which trace ids — is shared
//! across threads (a publisher thread and the `stop_recording` thread are
//! usually different), so it lives in a process-wide [`Mutex`]
//! ([`TRACE_REGISTRY`]). Publishers stay thread-local; only the small registry
//! map is shared.
//!
//! ## Service split
//!
//! Envelopes travel on one of two iceoryx2 services. Lifecycle envelopes and
//! non-video data ride [`COMMANDS`] (deep buffer, small slice); the
//! pixel-bearing traffic of video traces — `OpenFrameStream`'s sibling
//! `Frame`s and their `EndTrace` — rides [`FRAMES`] (small buffer, 16 MiB
//! slice). The destination is decided by *which* `log_*` function is called;
//! the registry records whether each trace is video so `stop_recording` routes
//! its `EndTrace` to the matching service.
//!
//! ## Fork safety
//!
//! iceoryx2's shared-memory descriptors are bound to the parent PID; a fork
//! child that re-used them would silently drop every envelope. A one-shot
//! `pthread_atfork` child handler clears the forking thread's `PRODUCER` slot
//! so the next publish rebuilds. The process-wide [`TRACE_REGISTRY`] is healed
//! lazily instead: it stores the owning PID and wipes itself on the first
//! access from a process whose PID no longer matches (see [`with_registry`]),
//! so a forked `multiprocessing` worker never inherits stale parent traces.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, Once};

use data_daemon_ipc::service_name::{
    COMMANDS, COMMANDS_MAX_PAYLOAD_BYTES, FRAMES, FRAMES_MAX_PAYLOAD_BYTES,
    FRAMES_SUBSCRIBER_BUFFER_SIZE, LIFECYCLE_SUBSCRIBER_BUFFER_SIZE, MAX_NODES_PER_SERVICE,
    MAX_PUBLISHERS_PER_SERVICE, MAX_SUBSCRIBERS_PER_SERVICE,
};
use data_daemon_ipc::{BatchedFrameItem, Envelope};
use iceoryx2::node::{Node, NodeBuilder};
use iceoryx2::port::publisher::Publisher;
use iceoryx2::prelude::{ipc, UnableToDeliverStrategy};
use iceoryx2::service::port_factory::publish_subscribe::PortFactory;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;
use uuid::Uuid;

/// Errors raised while publishing envelopes to the daemon.
#[derive(Debug, Error)]
enum ProducerError {
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
///
/// Holds the per-thread [`Node`] and, for each service, the [`PortFactory`]
/// handle alongside its [`Publisher`] so the publishers' shared-memory
/// descriptors stay live for the lifetime of the thread.
struct ProducerState {
    _node: Node<ipc::Service>,
    _commands_service: PortFactory<ipc::Service, [u8], ()>,
    commands_publisher: Publisher<ipc::Service, [u8], ()>,
    _frames_service: PortFactory<ipc::Service, [u8], ()>,
    frames_publisher: Publisher<ipc::Service, [u8], ()>,
}

/// Which iceoryx2 service an envelope is published on.
#[derive(Clone, Copy)]
enum Target {
    /// The [`COMMANDS`] lifecycle service.
    Commands,
    /// The [`FRAMES`] video service.
    Frames,
}

/// One trace minted by the producer for a stream within a recording.
struct TraceEntry {
    /// Trace id sent to the daemon; never surfaced to Python.
    trace_id: String,
    /// `true` for video traces — their `Frame`/`EndTrace` ride [`FRAMES`].
    is_video: bool,
}

/// Every trace the producer has minted for one recording, keyed by
/// `stream_key` (`data_type` + `name`).
#[derive(Default)]
struct RecordingTraces {
    traces: HashMap<String, TraceEntry>,
}

/// Process-wide trace registry.
///
/// `owner_pid` lets a forked `multiprocessing` worker detect that the map was
/// inherited from its parent and wipe it on first use — see [`with_registry`].
struct TraceRegistry {
    owner_pid: u32,
    recordings: HashMap<String, RecordingTraces>,
}

/// Process-wide registry of recordings → minted traces. Guarded by a `Mutex`
/// because the publishing threads and the `stop_recording` thread differ;
/// `LazyLock` because `HashMap::new` is not a `const fn`.
static TRACE_REGISTRY: LazyLock<Mutex<TraceRegistry>> = LazyLock::new(|| {
    Mutex::new(TraceRegistry {
        owner_pid: 0,
        recordings: HashMap::new(),
    })
});

thread_local! {
    /// One iceoryx2 publisher set per OS thread. See the module-level note on
    /// threading for the rationale. Const-initialised so the slot is a plain
    /// TLS load — required for the `pthread_atfork` child handler to access
    /// it without invoking a lazy initializer in a post-fork context.
    static PRODUCER: RefCell<Option<ProducerState>> = const { RefCell::new(None) };
}

/// Lock the trace registry and run `operation` against its recordings map.
///
/// Heals the registry across a fork: when the stored `owner_pid` no longer
/// matches the current process the map was inherited from a pre-fork parent,
/// so it is cleared before use. This runs during normal locked execution in
/// the child (never inside the `pthread_atfork` handler), so taking the lock
/// here is safe.
fn with_registry<R>(operation: impl FnOnce(&mut HashMap<String, RecordingTraces>) -> R) -> R {
    let mut registry = TRACE_REGISTRY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let pid = std::process::id();
    if registry.owner_pid != pid {
        registry.recordings.clear();
        registry.owner_pid = pid;
    }
    operation(&mut registry.recordings)
}

/// Build the `data_type`/`name` composite key used to identify a stream
/// within a recording. The NUL separator cannot occur in either component, so
/// the join is unambiguous.
fn stream_key(data_type: &str, name: &str) -> String {
    format!("{data_type}\u{0}{name}")
}

/// Register a recording in the trace registry.
///
/// Returns `true` when the recording was newly registered, `false` when it
/// was already present — i.e. a repeated `start_recording`. This is the
/// single source of truth that keeps `start, start` from emitting a duplicate
/// `StartRecording`.
fn register_recording(recording_id: &str) -> bool {
    with_registry(|recordings| {
        if recordings.contains_key(recording_id) {
            false
        } else {
            recordings.insert(recording_id.to_string(), RecordingTraces::default());
            true
        }
    })
}

/// Remove a recording from the registry, returning the traces it minted.
///
/// Returns `None` when the recording was not registered (already stopped, or
/// never started) — which is what makes a repeated `stop_recording`, or a
/// stray `stop` before `start`, a safe no-op.
fn take_recording(recording_id: &str) -> Option<RecordingTraces> {
    with_registry(|recordings| recordings.remove(recording_id))
}

/// Run `f` against this thread's producer state, lazily building it on first
/// use. The fork-child handler is responsible for clearing the slot in the
/// post-fork child; no per-call PID check is needed here.
fn with_producer<R>(
    operation: impl FnOnce(&ProducerState) -> Result<R, ProducerError>,
) -> Result<R, ProducerError> {
    PRODUCER.with(|cell| {
        if cell.borrow().is_none() {
            let state = build_producer_state()?;
            *cell.borrow_mut() = Some(state);
        }
        let slot = cell.borrow();
        let state = slot
            .as_ref()
            .expect("producer state populated immediately above");
        operation(state)
    })
}

fn build_producer_state() -> Result<ProducerState, ProducerError> {
    // Register the fork-child handler exactly once per process, lazily on the
    // first publisher build. Doing it here (rather than in `_native_producer`
    // module init) keeps the registration co-located with the only state it
    // protects and avoids touching libc from the Python import path.
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
    let (frames_service, frames_publisher) = open_publisher(
        &node,
        FRAMES,
        FRAMES_SUBSCRIBER_BUFFER_SIZE,
        FRAMES_MAX_PAYLOAD_BYTES,
    )?;

    Ok(ProducerState {
        _node: node,
        _commands_service: commands_service,
        commands_publisher,
        _frames_service: frames_service,
        frames_publisher,
    })
}

/// Open (or attach to) one `[u8]` pub/sub service off `node` and build a
/// publisher on it.
///
/// The service attributes are declared explicitly so a producer that races
/// the daemon to `open_or_create` seeds the service with the same
/// configuration the daemon expects rather than iceoryx2's defaults — both
/// sides agree on these via the `data_daemon_ipc` constants.
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
        // Disable iceoryx2's default safe-overflow. With overflow on, a full
        // subscriber buffer silently evicts the oldest sample and the
        // publisher's `Block` strategy below never fires; a dropped
        // `StartTrace` then strands the daemon's per-trace actor. With
        // overflow off a full buffer makes `Block` take effect instead, so
        // delivery is lossless and in-order. Must match the daemon's
        // `open_subscriber`, which sets the same flag.
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
        // Block when the subscriber buffer is full (paired with
        // `enable_safe_overflow(false)` above). Lifecycle envelopes must
        // reach the daemon for the per-trace state machine to advance —
        // silent drops surface in the integration matrix as recordings stuck
        // in `writing` forever.
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
        // does is `mem::forget` — none of which can panic across the FFI
        // boundary, allocate, or take a lock that the parent could hold.
        let result = unsafe { libc::pthread_atfork(None, None, Some(on_fork_in_child)) };
        if result != 0 {
            tracing::warn!(
                errno = result,
                "pthread_atfork registration failed; fork-safety relies on caller-managed cleanup",
            );
        }
    });
}

/// `pthread_atfork` child callback.
///
/// Runs once in the post-fork child, in the single surviving thread (whichever
/// called `fork`). Clears that thread's `PRODUCER` slot so the next call to
/// [`with_producer`] rebuilds fresh iceoryx2 publishers whose descriptors
/// belong to the new process. The inherited `ProducerState` is `mem::forget`'d
/// on purpose: running its `Drop` would notify the daemon's bookkeeping for
/// the *parent's* still-live publisher, and freeing memory in a post-fork
/// child risks an allocator lock the parent held at fork time.
///
/// The process-wide [`TRACE_REGISTRY`] is deliberately *not* touched here —
/// taking its lock in a fork handler could deadlock if a parent thread held it
/// at fork time. It self-heals via the `owner_pid` check in [`with_registry`].
extern "C" fn on_fork_in_child() {
    PRODUCER.with(|cell| {
        if let Some(stale) = cell.borrow_mut().take() {
            std::mem::forget(stale);
        }
    });
}

/// Encode `envelope` and publish it on `target`'s iceoryx2 service.
fn publish(target: Target, envelope: &Envelope) -> Result<(), ProducerError> {
    let bytes = envelope.encode()?;
    let limit = match target {
        Target::Commands => COMMANDS_MAX_PAYLOAD_BYTES,
        Target::Frames => FRAMES_MAX_PAYLOAD_BYTES,
    };
    if bytes.len() > limit {
        return Err(ProducerError::PayloadTooLarge {
            actual: bytes.len(),
            limit,
        });
    }
    with_producer(|state| {
        let publisher = match target {
            Target::Commands => &state.commands_publisher,
            Target::Frames => &state.frames_publisher,
        };
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
///
/// Field order is fixed (`timestamp` before `value`) so byte-level diffs of
/// `trace.json` against historical recordings stay stable.
#[derive(serde::Serialize)]
struct ScalarFrameEntry {
    timestamp: f64,
    value: f64,
}

/// Announce the start of a recording session.
///
/// Idempotent: a second call for an already-registered `recording_id` is a
/// no-op, so a mistaken `start, start` sequence emits no duplicate
/// `StartRecording` and leaves any minted traces intact.
#[pyfunction]
#[pyo3(signature = (recording_id, robot_id = None, robot_name = None, dataset_id = None, dataset_name = None))]
fn start_recording(
    py: Python<'_>,
    recording_id: &str,
    robot_id: Option<String>,
    robot_name: Option<String>,
    dataset_id: Option<String>,
    dataset_name: Option<String>,
) -> PyResult<()> {
    if recording_id.is_empty() {
        return Err(PyValueError::new_err("recording_id must not be empty"));
    }
    let recording_id = recording_id.to_string();
    py.allow_threads(|| -> PyResult<()> {
        if !register_recording(&recording_id) {
            return Ok(());
        }
        publish(
            Target::Commands,
            &Envelope::StartRecording {
                recording_id,
                robot_id,
                robot_name,
                dataset_id,
                dataset_name,
            },
        )?;
        Ok(())
    })
}

/// Resolve a trace id for `(data_type, name)` within `recording`, minting a
/// fresh one when the stream has not been seen yet. Returns the trace id and
/// whether it was newly minted (so the caller can publish its `StartTrace`).
fn resolve_trace(
    recording: &mut RecordingTraces,
    data_type: &str,
    name: &str,
    is_video: bool,
) -> (String, bool) {
    let key = stream_key(data_type, name);
    match recording.traces.get(&key) {
        Some(entry) => (entry.trace_id.clone(), false),
        None => {
            let trace_id = Uuid::new_v4().to_string();
            recording.traces.insert(
                key,
                TraceEntry {
                    trace_id: trace_id.clone(),
                    is_video,
                },
            );
            (trace_id, true)
        }
    }
}

/// Log one scalar sample for each of several joints captured at the same
/// instant. Joints not seen before in this recording have a trace minted (and
/// a `StartTrace` published) before the batch is sent as one `BatchedFrames`.
#[pyfunction]
#[pyo3(signature = (recording_id, data_type, items, timestamp_ns, timestamp_s = None))]
fn log_joints(
    py: Python<'_>,
    recording_id: &str,
    data_type: &str,
    items: Vec<(String, f64)>,
    timestamp_ns: i64,
    timestamp_s: Option<f64>,
) -> PyResult<()> {
    if recording_id.is_empty() || data_type.is_empty() {
        return Err(PyValueError::new_err(
            "recording_id and data_type must not be empty",
        ));
    }
    if items.is_empty() {
        return Ok(());
    }
    if let Some(bad_index) = items.iter().position(|(name, _)| name.is_empty()) {
        return Err(PyValueError::new_err(format!(
            "joint name must not be empty (item index {bad_index})"
        )));
    }
    let recording_id = recording_id.to_string();
    let data_type = data_type.to_string();
    // Fall back to ns→s when the caller omits timestamp_s so the per-frame
    // "timestamp" field still has a sensible value to write into trace.json.
    let timestamp_for_json = timestamp_s.unwrap_or_else(|| timestamp_ns as f64 / 1_000_000_000.0);
    py.allow_threads(|| -> PyResult<()> {
        // Resolve a trace id for every joint, collecting the ones minted now.
        let mut new_traces: Vec<(String, String)> = Vec::new();
        let mut trace_ids: Vec<String> = Vec::with_capacity(items.len());
        with_registry(|recordings| {
            let recording = recordings.entry(recording_id.clone()).or_default();
            for (name, _) in &items {
                let (trace_id, is_new) = resolve_trace(recording, &data_type, name, false);
                if is_new {
                    new_traces.push((trace_id.clone(), name.clone()));
                }
                trace_ids.push(trace_id);
            }
        });
        // A StartTrace for each newly minted trace, ahead of the data batch.
        for (trace_id, name) in new_traces {
            publish(
                Target::Commands,
                &Envelope::StartTrace {
                    recording_id: recording_id.clone(),
                    trace_id,
                    data_type: data_type.clone(),
                    data_type_name: Some(name),
                },
            )?;
        }
        // Pack every joint sample into one BatchedFrames envelope.
        let mut frames = Vec::with_capacity(items.len());
        for ((_, value), trace_id) in items.iter().zip(trace_ids) {
            // serde_json (via ryu) always emits at least one fractional digit
            // for f64 — so integer-valued joint values land on disk as `1.0`,
            // not `1`, keeping the column consistently typed as a float.
            let payload = serde_json::to_vec(&ScalarFrameEntry {
                timestamp: timestamp_for_json,
                value: *value,
            })
            .map_err(|error| {
                PyRuntimeError::new_err(format!("failed to encode joint frame JSON: {error}"))
            })?;
            frames.push(BatchedFrameItem { trace_id, payload });
        }
        publish(
            Target::Commands,
            &Envelope::BatchedFrames {
                timestamp_ns,
                timestamp_s,
                frames,
            },
        )?;
        Ok(())
    })
}

/// Log one video frame for a camera. The first frame for a camera mints its
/// trace and publishes `StartTrace` + `OpenFrameStream`; every frame rides the
/// [`FRAMES`] service.
#[pyfunction]
#[pyo3(signature = (recording_id, data_type, name, width, height, payload, timestamp_ns, timestamp_s = None))]
#[allow(clippy::too_many_arguments)]
fn log_frame(
    py: Python<'_>,
    recording_id: &str,
    data_type: &str,
    name: &str,
    width: u32,
    height: u32,
    payload: &[u8],
    timestamp_ns: i64,
    timestamp_s: Option<f64>,
) -> PyResult<()> {
    if recording_id.is_empty() || data_type.is_empty() || name.is_empty() {
        return Err(PyValueError::new_err(
            "recording_id, data_type and name must not be empty",
        ));
    }
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("width and height must be non-zero"));
    }
    let recording_id = recording_id.to_string();
    let data_type = data_type.to_string();
    let name = name.to_string();
    // Copy the payload into owned Rust memory while the GIL is held — the
    // `&[u8]` borrows a Python buffer.
    let owned_payload = payload.to_vec();
    py.allow_threads(|| -> PyResult<()> {
        let (trace_id, is_new) = with_registry(|recordings| {
            let recording = recordings.entry(recording_id.clone()).or_default();
            resolve_trace(recording, &data_type, &name, true)
        });
        if is_new {
            publish(
                Target::Commands,
                &Envelope::StartTrace {
                    recording_id: recording_id.clone(),
                    trace_id: trace_id.clone(),
                    data_type,
                    data_type_name: Some(name),
                },
            )?;
            publish(
                Target::Commands,
                &Envelope::OpenFrameStream {
                    trace_id: trace_id.clone(),
                    width,
                    height,
                },
            )?;
        }
        publish(
            Target::Frames,
            &Envelope::frame(trace_id, timestamp_ns, timestamp_s, owned_payload),
        )?;
        Ok(())
    })
}

/// Log one scalar/custom sample (e.g. `log_custom_1d`). The first sample for a
/// stream mints its trace and publishes `StartTrace`; the payload is delivered
/// verbatim as a `Frame` on the [`COMMANDS`] service.
#[pyfunction]
#[pyo3(signature = (recording_id, data_type, name, payload, timestamp_ns, timestamp_s = None))]
fn log_scalar(
    py: Python<'_>,
    recording_id: &str,
    data_type: &str,
    name: &str,
    payload: &[u8],
    timestamp_ns: i64,
    timestamp_s: Option<f64>,
) -> PyResult<()> {
    if recording_id.is_empty() || data_type.is_empty() || name.is_empty() {
        return Err(PyValueError::new_err(
            "recording_id, data_type and name must not be empty",
        ));
    }
    let recording_id = recording_id.to_string();
    let data_type = data_type.to_string();
    let name = name.to_string();
    let owned_payload = payload.to_vec();
    py.allow_threads(|| -> PyResult<()> {
        let (trace_id, is_new) = with_registry(|recordings| {
            let recording = recordings.entry(recording_id.clone()).or_default();
            resolve_trace(recording, &data_type, &name, false)
        });
        if is_new {
            publish(
                Target::Commands,
                &Envelope::StartTrace {
                    recording_id: recording_id.clone(),
                    trace_id: trace_id.clone(),
                    data_type,
                    data_type_name: Some(name),
                },
            )?;
        }
        publish(
            Target::Commands,
            &Envelope::frame(trace_id, timestamp_ns, timestamp_s, owned_payload),
        )?;
        Ok(())
    })
}

/// End every trace the recording minted, then announce the recording stopped.
///
/// Idempotent: if the recording is not registered it was already stopped (or
/// never started), so a mistaken `stop, stop` — or a stray `stop` before
/// `start` — is a no-op that publishes no duplicate `StopRecording`.
#[pyfunction]
#[pyo3(signature = (recording_id))]
fn stop_recording(py: Python<'_>, recording_id: &str) -> PyResult<()> {
    if recording_id.is_empty() {
        return Err(PyValueError::new_err("recording_id must not be empty"));
    }
    let recording_id = recording_id.to_string();
    py.allow_threads(|| -> PyResult<()> {
        let Some(recording) = take_recording(&recording_id) else {
            return Ok(());
        };
        for entry in recording.traces.into_values() {
            let target = if entry.is_video {
                Target::Frames
            } else {
                Target::Commands
            };
            publish(
                target,
                &Envelope::EndTrace {
                    trace_id: entry.trace_id,
                },
            )?;
        }
        publish(Target::Commands, &Envelope::StopRecording { recording_id })?;
        Ok(())
    })
}

/// Cancel a recording — the daemon discards every in-flight trace and its
/// on-disk artefacts. The producer drops its trace registry entry (no
/// `EndTrace` is needed; `CancelRecording` supersedes them).
#[pyfunction]
#[pyo3(signature = (recording_id))]
fn cancel_recording(py: Python<'_>, recording_id: &str) -> PyResult<()> {
    if recording_id.is_empty() {
        return Err(PyValueError::new_err("recording_id must not be empty"));
    }
    let recording_id = recording_id.to_string();
    py.allow_threads(|| -> PyResult<()> {
        take_recording(&recording_id);
        publish(
            Target::Commands,
            &Envelope::CancelRecording { recording_id },
        )?;
        Ok(())
    })
}

/// Python module entrypoint registered as `neuracore.data_daemon._native_producer`.
#[pymodule]
fn _native_producer(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(start_recording, module)?)?;
    module.add_function(wrap_pyfunction!(log_joints, module)?)?;
    module.add_function(wrap_pyfunction!(log_frame, module)?)?;
    module.add_function(wrap_pyfunction!(log_scalar, module)?)?;
    module.add_function(wrap_pyfunction!(stop_recording, module)?)?;
    module.add_function(wrap_pyfunction!(cancel_recording, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mistaken lifecycle sequences must degrade to no-ops, never panic or
    /// emit duplicate envelopes. `register_recording` returning `false` is
    /// what suppresses a duplicate `StartRecording`; `take_recording`
    /// returning `None` is what suppresses a duplicate `StopRecording`.
    #[test]
    fn registry_tolerates_repeated_and_out_of_order_lifecycle() {
        // Unique id so the process-wide registry can't collide with a
        // sibling test running in parallel.
        let recording_id = "rec-lifecycle-test-001";

        // start, start -> only the first registers.
        assert!(register_recording(recording_id));
        assert!(!register_recording(recording_id));

        // stop, stop -> only the first drains the registry.
        assert!(take_recording(recording_id).is_some());
        assert!(take_recording(recording_id).is_none());

        // A stray stop for a recording that never started is a no-op.
        assert!(take_recording("rec-never-started-test-001").is_none());
    }
}
