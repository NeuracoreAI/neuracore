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
//!   [`StartTrace`](data_daemon_ipc::Envelope::StartTrace) — then publishes
//!   the sample. Video frames are spooled to per-trace NUT chunk files on
//!   local disk and announced with
//!   [`VideoChunkReady`](data_daemon_ipc::Envelope::VideoChunkReady) once a
//!   chunk fills.
//! - [`stop_recording`](crate::stop_recording) flushes any tail chunk for
//!   each video trace, ends every trace the recording minted, and publishes
//!   one [`StopRecording`](data_daemon_ipc::Envelope::StopRecording).
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
//! publisher in a [`thread_local`]: each Python thread that calls in lazily
//! builds its own iceoryx2 [`Node`] and a publisher on the commands service.
//!
//! The *trace registry* — which streams map to which trace ids — and the
//! *video chunk registry* are shared across threads (a publisher thread and
//! the `stop_recording` thread are usually different), so they live in
//! process-wide [`Mutex`]es. Publishers stay thread-local; only the small
//! registry maps are shared.
//!
//! ## Video chunk spooling
//!
//! Raw video pixels never travel on the IPC bus. The producer writes each
//! camera's frames into a sequence of [`NutWriter`] chunk files at
//! `{recordings_root}/{recording_id}/{data_type}/{trace_id}/chunks/chunk_NNNN.nut`.
//! When a chunk crosses [`CHUNK_FLUSH_BYTES`] the writer is finished and the
//! producer publishes a [`VideoChunkReady`](data_daemon_ipc::Envelope::VideoChunkReady)
//! envelope carrying the per-frame `timestamp_s` values; the daemon then
//! transcodes the file to a sealed MP4 segment. On `EndTrace` the daemon
//! concatenates the per-chunk segments into the final `lossy.mp4` /
//! `lossless.mp4`.
//!
//! ## Fork safety
//!
//! iceoryx2's shared-memory descriptors are bound to the parent PID; a fork
//! child that re-used them would silently drop every envelope. A one-shot
//! `pthread_atfork` child handler clears the forking thread's `PRODUCER` slot
//! so the next publish rebuilds. The process-wide [`TRACE_REGISTRY`] and
//! [`VIDEO_CHUNKS`] are healed lazily instead: they store the owning PID and
//! wipe themselves on the first access from a process whose PID no longer
//! matches, so a forked `multiprocessing` worker never inherits stale parent
//! state.

pub mod nut_writer;

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex, Once};

use data_daemon_ipc::service_name::{
    COMMANDS, COMMANDS_MAX_PAYLOAD_BYTES, LIFECYCLE_SUBSCRIBER_BUFFER_SIZE, MAX_NODES_PER_SERVICE,
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

use pyo3::buffer::PyBuffer;

use crate::nut_writer::{NutVideoConfig, NutWriter};

/// Bytes after which the producer rotates to a fresh NUT chunk file.
///
/// Each chunk pays a fixed per-encode cost on the daemon side: ffmpeg
/// fork+exec + libx264 init for two output codecs is ~100-200ms regardless
/// of chunk size. With 22-frame 1080p chunks at the previous 128 MiB
/// threshold that fixed cost was ~15-25% of the per-chunk wall time —
/// observable as transient backpressure on the producer-side iceoryx2
/// publish when an encode stretched past the chunk-arrival interval.
/// Doubling to 256 MiB halves the fork+exec churn for the same encode
/// throughput. The threshold is checked *after* each frame, so the on-disk
/// file can exceed it by at most one frame.
const CHUNK_FLUSH_BYTES: u64 = 256 * 1024 * 1024;

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
/// Holds the per-thread [`Node`] alongside the commands publisher so the
/// publisher's shared-memory descriptors stay live for the lifetime of the
/// thread.
struct ProducerState {
    _node: Node<ipc::Service>,
    _commands_service: PortFactory<ipc::Service, [u8], ()>,
    commands_publisher: Publisher<ipc::Service, [u8], ()>,
}

/// One trace minted by the producer for a stream within a recording.
struct TraceEntry {
    /// Trace id sent to the daemon; never surfaced to Python.
    trace_id: String,
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

/// In-progress video chunk state for one trace.
///
/// The `NutWriter` is lazy per-chunk: it is created when the first frame of
/// a chunk arrives and consumed by [`flush_chunk`] once the chunk fills (or
/// the trace ends). All other fields persist across chunks for the trace's
/// lifetime.
struct VideoChunkState {
    /// Recording the trace belongs to. Stamped onto every
    /// [`Envelope::VideoChunkReady`] so the daemon can resolve the on-disk
    /// path without a roundtrip through `StartTrace`.
    recording_id: String,
    /// Frame width in pixels (constant across a trace's chunks).
    width: u32,
    /// Frame height in pixels (constant across a trace's chunks).
    height: u32,
    /// `{recordings_root}/{recording_id}/{data_type}/{trace_id}/chunks/`.
    chunks_dir: PathBuf,
    /// Active NUT writer for the in-progress chunk. `None` between chunks
    /// (i.e. immediately after a flush, before the next frame arrives).
    nut_writer: Option<NutWriter>,
    /// Zero-based index of the next chunk to flush.
    chunk_index: u32,
    /// Frames already written into the in-progress chunk.
    frame_count: u32,
    /// Per-trace PTS origin, microseconds since the Unix epoch. Set on the
    /// first frame and kept constant across chunks so PTS values are
    /// comparable across the whole trace.
    pts_origin_us: Option<i64>,
    /// Last PTS written to *any* chunk for the trace; enforces strict
    /// monotonicity even across chunk boundaries.
    last_pts_us: Option<u64>,
    /// Per-frame `timestamp_s` accumulator for the in-progress chunk.
    /// Drained into the `VideoChunkReady` envelope on flush.
    frame_timestamps_s: Vec<f64>,
}

/// Process-wide registry of in-progress per-trace video chunk state.
///
/// The outer [`Mutex`] guards only the `HashMap` shape — inserts when a new
/// video trace starts, removes when it stops, get-or-create on each frame.
/// Per-trace state lives behind its own [`Arc<Mutex<VideoChunkState>>`] so
/// the multi-megabyte NUT write for camera A does **not** block camera B's
/// concurrent write. Camera-A's `log_frame` holds the registry lock only
/// long enough to clone the `Arc`, then releases it and does the actual
/// write under its private per-trace mutex.
///
/// Mirrors [`TRACE_REGISTRY`]: shared between publishing threads (which add
/// frames) and the `stop_recording` thread (which flushes the tail chunk),
/// healed across a fork via [`with_video_chunks`].
type VideoChunkSlot = Arc<Mutex<VideoChunkState>>;

struct VideoChunkRegistry {
    owner_pid: u32,
    traces: HashMap<String, VideoChunkSlot>,
}

static VIDEO_CHUNKS: LazyLock<Mutex<VideoChunkRegistry>> = LazyLock::new(|| {
    Mutex::new(VideoChunkRegistry {
        owner_pid: 0,
        traces: HashMap::new(),
    })
});

/// Recordings root, resolved once per process. Mirrors the daemon's
/// `recordings_root_path()` (`config/env.rs`): `NEURACORE_DAEMON_RECORDINGS_ROOT`
/// when set, otherwise `<NEURACORE_DAEMON_DB_PATH parent>/recordings`,
/// otherwise `~/.neuracore/data_daemon/recordings`.
static RECORDINGS_ROOT: LazyLock<PathBuf> = LazyLock::new(resolve_recordings_root);

fn resolve_recordings_root() -> PathBuf {
    if let Ok(value) = std::env::var("NEURACORE_DAEMON_RECORDINGS_ROOT") {
        if !value.is_empty() {
            return PathBuf::from(value);
        }
    }
    default_recordings_root()
}

fn default_recordings_root() -> PathBuf {
    let db_path = match std::env::var("NEURACORE_DAEMON_DB_PATH") {
        Ok(value) if !value.is_empty() => expand_user(&value),
        _ => home_dir()
            .join(".neuracore")
            .join("data_daemon")
            .join("state.db"),
    };
    db_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("recordings")
}

fn home_dir() -> PathBuf {
    // The daemon's `home_dir()` panics if unresolved; the producer is
    // identically constrained — every on-disk path the SDK writes derives
    // from `$HOME` when no override is set.
    dirs::home_dir().expect("could not determine the user's home directory")
}

fn expand_user(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        return home_dir().join(stripped);
    }
    if path == "~" {
        return home_dir();
    }
    PathBuf::from(path)
}

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

/// Lock the video chunk registry and run `operation` against its traces map.
///
/// The lock is intentionally short-lived — the registry only guards the
/// `HashMap`'s shape, not the per-trace state inside the `Arc<Mutex<...>>`
/// entries. Callers that need to *use* per-trace state (write a frame,
/// flush a chunk) should clone the [`VideoChunkSlot`] under this lock and
/// then release it before doing any blocking work, so concurrent cameras
/// don't serialise on each other.
///
/// Heals on fork the same way [`with_registry`] does.
fn with_video_chunks<R>(operation: impl FnOnce(&mut HashMap<String, VideoChunkSlot>) -> R) -> R {
    let mut registry = VIDEO_CHUNKS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let pid = std::process::id();
    if registry.owner_pid != pid {
        registry.traces.clear();
        registry.owner_pid = pid;
    }
    operation(&mut registry.traces)
}

/// Build the `data_type`/`name` composite key used to identify a stream
/// within a recording. The NUL separator cannot occur in either component, so
/// the join is unambiguous.
fn stream_key(data_type: &str, name: &str) -> String {
    format!("{data_type}\u{0}{name}")
}

/// Build the path to a trace's chunks directory. Mirrors
/// `storage::paths::TracePath::chunks_dir` on the daemon side; the two must
/// agree byte-for-byte so the daemon picks up exactly what the producer
/// wrote.
fn trace_chunks_dir(recording_id: &str, data_type: &str, trace_id: &str) -> PathBuf {
    RECORDINGS_ROOT
        .join(recording_id)
        .join(data_type)
        .join(trace_id)
        .join("chunks")
}

/// Filename for chunk `index` — must match `storage::paths::chunk_filename`.
fn chunk_filename(index: u32) -> String {
    format!("chunk_{index:04}.nut")
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

    Ok(ProducerState {
        _node: node,
        _commands_service: commands_service,
        commands_publisher,
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
/// The process-wide [`TRACE_REGISTRY`] and [`VIDEO_CHUNKS`] are deliberately
/// *not* touched here — taking their locks in a fork handler could deadlock
/// if a parent thread held them at fork time. They self-heal via the
/// `owner_pid` check in [`with_registry`] / [`with_video_chunks`].
extern "C" fn on_fork_in_child() {
    PRODUCER.with(|cell| {
        if let Some(stale) = cell.borrow_mut().take() {
            std::mem::forget(stale);
        }
    });
}

/// Encode `envelope` and publish it on the commands service.
fn publish(envelope: &Envelope) -> Result<(), ProducerError> {
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
        publish(&Envelope::StartRecording {
            recording_id,
            robot_id,
            robot_name,
            dataset_id,
            dataset_name,
        })?;
        Ok(())
    })
}

/// Resolve a trace id for `(data_type, name)` within `recording`, minting a
/// fresh one when the stream has not been seen yet. Returns the trace id and
/// whether it was newly minted (so the caller can publish its `StartTrace`).
fn resolve_trace(recording: &mut RecordingTraces, data_type: &str, name: &str) -> (String, bool) {
    let key = stream_key(data_type, name);
    match recording.traces.get(&key) {
        Some(entry) => (entry.trace_id.clone(), false),
        None => {
            let trace_id = Uuid::new_v4().to_string();
            recording.traces.insert(
                key,
                TraceEntry {
                    trace_id: trace_id.clone(),
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
        // The recording must already be registered (i.e. an earlier
        // `start_recording` is in scope) — if it's not, this call is a
        // silent no-op. Auto-registering on the log path would resurrect a
        // recording the producer has already torn down via `stop_recording`
        // (a real race after the SDK's 5-minute expiry handler), spawning
        // phantom trace ids that the daemon never receives an `EndTrace`
        // for and which stay in `writing` forever.
        let mut new_traces: Vec<(String, String)> = Vec::new();
        let mut trace_ids: Vec<String> = Vec::with_capacity(items.len());
        let recording_known = with_registry(|recordings| {
            let Some(recording) = recordings.get_mut(&recording_id) else {
                return false;
            };
            for (name, _) in &items {
                let (trace_id, is_new) = resolve_trace(recording, &data_type, name);
                if is_new {
                    new_traces.push((trace_id.clone(), name.clone()));
                }
                trace_ids.push(trace_id);
            }
            true
        });
        if !recording_known {
            return Ok(());
        }
        // A StartTrace for each newly minted trace, ahead of the data batch.
        for (trace_id, name) in new_traces {
            publish(&Envelope::StartTrace {
                recording_id: recording_id.clone(),
                trace_id,
                data_type: data_type.clone(),
                data_type_name: Some(name),
            })?;
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
        publish(&Envelope::BatchedFrames {
            timestamp_ns,
            timestamp_s,
            frames,
        })?;
        Ok(())
    })
}

/// Log one video frame for a camera. The first frame for a camera mints its
/// trace and publishes `StartTrace`; the frame is appended to the trace's
/// in-progress NUT chunk on local disk. When the chunk crosses
/// [`CHUNK_FLUSH_BYTES`] a [`Envelope::VideoChunkReady`] is published so the
/// daemon can encode the chunk to a sealed MP4 segment.
#[pyfunction]
#[pyo3(signature = (recording_id, data_type, name, width, height, payload, timestamp_ns, timestamp_s = None))]
#[allow(clippy::too_many_arguments)]
fn log_frame(
    recording_id: &str,
    data_type: &str,
    name: &str,
    width: u32,
    height: u32,
    payload: PyBuffer<u8>,
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
    let expected_bytes = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(3);
    let actual_bytes = payload.item_count();
    if actual_bytes != expected_bytes {
        return Err(PyValueError::new_err(format!(
            "video frame buffer is {actual_bytes} bytes; expected width*height*3 = {expected_bytes}"
        )));
    }
    if !payload.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "video frame buffer must be C-contiguous",
        ));
    }
    let recording_id = recording_id.to_string();
    let data_type = data_type.to_string();
    let name = name.to_string();
    // Resolve `timestamp_s` once. Fall back to ns→s so the chunk envelope's
    // per-frame timestamp matches the integration matrix's exact-match
    // sidecar assertion even when the SDK omitted the f64.
    let resolved_timestamp_s =
        timestamp_s.unwrap_or_else(|| timestamp_ns as f64 / 1_000_000_000.0);

    // SAFETY: PyO3 guarantees the GIL is held for the entire body of a
    // `#[pyfunction]`, the buffer is `u8` (`PyBuffer::<u8>::get` validated
    // the format and item size), the length comes straight from
    // `PyBuffer::item_count`, and we only *read* from the slice. The
    // buffer's underlying Python object stays alive for `payload`'s
    // lifetime, which spans this entire function.
    //
    // We deliberately keep the GIL across the call. Releasing it (via
    // `py.allow_threads`) would require copying `payload` into an owned
    // `Vec<u8>` — for 1080p RGB that's a 6.22 MiB memcpy on *every*
    // frame. Keeping the GIL lets us pass a zero-copy view straight from
    // the numpy buffer through to `NutWriter::write_frame`. The ~2 ms
    // NUT write blocks other Python threads briefly, which joint logging
    // at 15-200 Hz tolerates comfortably.
    let payload_slice: &[u8] =
        unsafe { std::slice::from_raw_parts(payload.buf_ptr() as *const u8, actual_bytes) };

    let Some((trace_id, is_new)) = with_registry(|recordings| {
        recordings
            .get_mut(&recording_id)
            .map(|recording| resolve_trace(recording, &data_type, &name))
    }) else {
        return Ok(());
    };
    if is_new {
        publish(&Envelope::StartTrace {
            recording_id: recording_id.clone(),
            trace_id: trace_id.clone(),
            data_type: data_type.clone(),
            data_type_name: Some(name),
        })?;
    }
    record_video_frame(
        &recording_id,
        &data_type,
        &trace_id,
        width,
        height,
        payload_slice,
        timestamp_ns,
        resolved_timestamp_s,
    )
}

/// Append one frame to the trace's in-progress NUT chunk. Opens the chunk
/// file lazily on the first call for a trace (or after a flush), enforces
/// strict PTS monotonicity, and triggers [`flush_chunk_locked`] once the
/// chunk crosses [`CHUNK_FLUSH_BYTES`].
///
/// NUT-write errors are logged and the frame is dropped — they do not
/// propagate to Python. The producer SDK contract is best-effort delivery
/// for sensor data; a disk write failure in the middle of a recording must
/// not crash the user's training loop.
#[allow(clippy::too_many_arguments)]
fn record_video_frame(
    recording_id: &str,
    data_type: &str,
    trace_id: &str,
    width: u32,
    height: u32,
    payload: &[u8],
    timestamp_ns: i64,
    timestamp_s: f64,
) -> PyResult<()> {
    // The registry lock guards only the HashMap shape; we grab (or create)
    // the per-trace slot and immediately drop it so the actual NUT write
    // runs under the per-trace mutex without serialising other cameras.
    let slot: VideoChunkSlot = with_video_chunks(|traces| {
        traces
            .entry(trace_id.to_string())
            .or_insert_with(|| {
                Arc::new(Mutex::new(VideoChunkState {
                    recording_id: recording_id.to_string(),
                    width,
                    height,
                    chunks_dir: trace_chunks_dir(recording_id, data_type, trace_id),
                    nut_writer: None,
                    chunk_index: 0,
                    frame_count: 0,
                    pts_origin_us: None,
                    last_pts_us: None,
                    frame_timestamps_s: Vec::new(),
                }))
            })
            .clone()
    });

    // The flush envelope is built under the per-trace lock (it consumes the
    // chunk's accumulated state) but published outside it — publish() blocks
    // the calling thread when the daemon falls behind, and holding any
    // per-trace mutex across that block would stall this camera's next
    // frame.
    let flush_envelope = {
        let mut state = slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Compute PTS first; if monotonicity logic rejects the frame we want
        // to bail out before opening the file. Producer-side mirror of the
        // logic previously living in the daemon's trace_actor.
        let origin_us = *state.pts_origin_us.get_or_insert(timestamp_ns / 1_000);
        let relative_us = (timestamp_ns / 1_000).saturating_sub(origin_us).max(0);
        let mut pts = relative_us as u64;
        if let Some(previous) = state.last_pts_us {
            if pts <= previous {
                pts = previous.saturating_add(1);
            }
        }

        // Open the chunk writer lazily — the first frame of every chunk
        // pays the syscall cost so the rest only see appends.
        if state.nut_writer.is_none() {
            let chunk_path = state.chunks_dir.join(chunk_filename(state.chunk_index));
            let config = NutVideoConfig {
                width: state.width,
                height: state.height,
                time_base_num: 1,
                time_base_den: 1_000_000,
            };
            match NutWriter::create(&chunk_path, config) {
                Ok(writer) => state.nut_writer = Some(writer),
                Err(error) => {
                    tracing::warn!(
                        %error,
                        trace_id,
                        path = %chunk_path.display(),
                        "failed to open NUT chunk; dropping frame"
                    );
                    return Ok(());
                }
            }
        }

        let bytes_after_write = {
            let writer = state.nut_writer.as_mut().expect("opened immediately above");
            if let Err(error) = writer.write_frame(pts, payload) {
                tracing::warn!(
                    %error,
                    trace_id,
                    "failed to write video frame to NUT chunk; dropping frame"
                );
                return Ok(());
            }
            writer.bytes_written()
        };
        state.last_pts_us = Some(pts);
        state.frame_count = state.frame_count.saturating_add(1);
        state.frame_timestamps_s.push(timestamp_s);

        if bytes_after_write >= CHUNK_FLUSH_BYTES {
            flush_chunk_locked(trace_id, &mut state)
        } else {
            None
        }
    };

    if let Some(envelope) = flush_envelope {
        publish(&envelope)?;
    }
    Ok(())
}

/// Seal the in-progress chunk and return the envelope to publish.
///
/// The caller must hold the per-trace [`VideoChunkState`] lock. Returns
/// `None` when the trace has no open chunk writer (no frames seen since the
/// last flush, or the writer was never opened) — there's nothing to
/// announce in that case.
fn flush_chunk_locked(trace_id: &str, state: &mut VideoChunkState) -> Option<Envelope> {
    let writer = state.nut_writer.take()?;
    let byte_count = match writer.finish() {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(
                %error,
                trace_id,
                "failed to finalise NUT chunk; dropping chunk"
            );
            // Reset chunk-local state so the next frame opens a fresh chunk
            // — the broken file stays on disk for the recovery sweep to
            // collect.
            state.frame_count = 0;
            state.frame_timestamps_s.clear();
            state.chunk_index = state.chunk_index.saturating_add(1);
            return None;
        }
    };
    let chunk_index = state.chunk_index;
    let frame_count = state.frame_count;
    let frame_timestamps_s = std::mem::take(&mut state.frame_timestamps_s);

    state.frame_count = 0;
    state.chunk_index = state.chunk_index.saturating_add(1);

    Some(Envelope::VideoChunkReady {
        recording_id: state.recording_id.clone(),
        trace_id: trace_id.to_string(),
        chunk_index,
        width: state.width,
        height: state.height,
        byte_count,
        frame_count,
        frame_timestamps_s,
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
        // Silent drop if the recording isn't registered — see the
        // matching no-auto-register comment in `log_joints`.
        let Some((trace_id, is_new)) = with_registry(|recordings| {
            recordings
                .get_mut(&recording_id)
                .map(|recording| resolve_trace(recording, &data_type, &name))
        }) else {
            return Ok(());
        };
        if is_new {
            publish(&Envelope::StartTrace {
                recording_id: recording_id.clone(),
                trace_id: trace_id.clone(),
                data_type,
                data_type_name: Some(name),
            })?;
        }
        publish(&Envelope::frame(
            trace_id,
            timestamp_ns,
            timestamp_s,
            owned_payload,
        ))?;
        Ok(())
    })
}

/// End every trace the recording minted, then announce the recording stopped.
///
/// For each video trace with a partial chunk in [`VIDEO_CHUNKS`] the tail
/// chunk is flushed (and its `VideoChunkReady` published) *before* the
/// trace's `EndTrace` lands, so the in-order delivery contract on the
/// commands service guarantees the daemon sees the chunk first.
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
            // Drain the trace's chunk state — even if there's no partial
            // chunk we want to drop the slot so a re-used trace id (post
            // restart) doesn't inherit stale book-keeping. Removing under
            // the registry lock and flushing under the per-trace lock keeps
            // the global lock short.
            let slot = with_video_chunks(|traces| traces.remove(&entry.trace_id));
            let flush_envelope = slot.and_then(|slot| {
                let mut state = slot
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                flush_chunk_locked(&entry.trace_id, &mut state)
            });
            if let Some(envelope) = flush_envelope {
                publish(&envelope)?;
            }
            publish(&Envelope::EndTrace {
                trace_id: entry.trace_id,
            })?;
        }
        publish(&Envelope::StopRecording { recording_id })?;
        Ok(())
    })
}

/// Cancel a recording — the daemon discards every in-flight trace and its
/// on-disk artefacts. The producer drops its trace registry entry and any
/// in-progress video chunk state without flushing (the daemon's
/// `CancelRecording` handler removes the trace dir).
#[pyfunction]
#[pyo3(signature = (recording_id))]
fn cancel_recording(py: Python<'_>, recording_id: &str) -> PyResult<()> {
    if recording_id.is_empty() {
        return Err(PyValueError::new_err("recording_id must not be empty"));
    }
    let recording_id = recording_id.to_string();
    py.allow_threads(|| -> PyResult<()> {
        if let Some(recording) = take_recording(&recording_id) {
            with_video_chunks(|traces| {
                for entry in recording.traces.values() {
                    // Drop without flushing — the in-progress chunk file
                    // (and any earlier chunks the daemon hasn't yet
                    // encoded) gets removed by the daemon's recording
                    // cancellation sweep.
                    traces.remove(&entry.trace_id);
                }
            });
        }
        publish(&Envelope::CancelRecording { recording_id })?;
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
