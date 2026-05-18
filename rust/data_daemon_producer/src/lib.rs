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
//! Python wheel. It exposes the small lifecycle surface the Python adaptor
//! ([neuracore/data_daemon/communications_management/producer/producer_channel.py](../../../neuracore/data_daemon/communications_management/producer/producer_channel.py))
//! needs to drive the daemon over iceoryx2:
//!
//! - [`start_recording`](crate::start_recording) opens the iceoryx2 commands
//!   service if necessary and publishes a
//!   [`StartRecording`](data_daemon_ipc::Envelope::StartRecording) envelope.
//! - [`start_trace`](crate::start_trace) / [`end_trace`](crate::end_trace) frame
//!   per-trace lifecycle.
//! - [`send_data`](crate::send_data) publishes a
//!   [`Frame`](data_daemon_ipc::Envelope::Frame) envelope for the supplied
//!   trace. Callers manage `trace_id` themselves; the producer is intentionally
//!   stateless so it can be invoked from multiple Python threads without
//!   per-trace bookkeeping inside the Rust layer.
//! - [`open_frame_stream`](crate::open_frame_stream) publishes an
//!   [`OpenFrameStream`](data_daemon_ipc::Envelope::OpenFrameStream) envelope
//!   to switch the daemon-side actor into the video-writer path before the
//!   first pixel frame.
//! - [`stop_recording`](crate::stop_recording) publishes a
//!   [`StopRecording`](data_daemon_ipc::Envelope::StopRecording) envelope.
//!
//! ## Threading
//!
//! iceoryx2's [`Publisher`] uses an `Rc`-backed `ArcSyncPolicy` and is
//! therefore neither `Send` nor `Sync`. We side-step that by parking the
//! per-process state in a [`thread_local`]: each Python thread that calls into
//! the module lazily builds its own iceoryx2 [`Node`] + [`Publisher`] pair on
//! first use and reuses it for the rest of that thread's lifetime. This
//! matches Python's threading model — the GIL serialises Python execution but
//! does *not* prevent multiple OS threads from holding handles into the
//! daemon's IPC namespace — and it keeps every PyO3 entry point lock-free.
//!
//! ## Fork safety
//!
//! iceoryx2's shared-memory descriptors are bound to the parent PID; a fork
//! child that re-used them would silently drop every Frame envelope. We
//! register a one-shot `pthread_atfork` child handler on first publisher
//! construction (see [`ensure_fork_handler_registered`]). After a fork the
//! child resumes in the single thread that called `fork`; the handler clears
//! that thread's `PRODUCER` slot so the next publish rebuilds. The inherited
//! `ProducerState` is `mem::forget`'d — running its `Drop` would notify the
//! daemon's bookkeeping for the *parent's* still-live publisher.

use std::cell::RefCell;
use std::fs::OpenOptions;
use std::path::PathBuf;
use std::sync::Once;

use data_daemon_ipc::service_name::{
    COMMANDS, COMMANDS_MAX_PAYLOAD_BYTES, LIFECYCLE_SUBSCRIBER_BUFFER_SIZE,
    MAX_PUBLISHERS_PER_SERVICE,
};
use data_daemon_ipc::Envelope;
use iceoryx2::node::{Node, NodeBuilder};
use iceoryx2::port::publisher::Publisher;
use iceoryx2::prelude::{ipc, UnableToDeliverStrategy};
use iceoryx2::service::port_factory::publish_subscribe::PortFactory;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;
use tracing_subscriber::EnvFilter;

/// Errors raised while publishing envelopes to the daemon.
#[derive(Debug, Error)]
enum ProducerError {
    /// Failed to build the iceoryx2 node.
    #[error("failed to create iceoryx2 node: {0}")]
    NodeCreate(String),
    /// Failed to open or create the commands service.
    #[error("failed to open commands service: {0}")]
    ServiceOpen(String),
    /// Failed to build the publisher port.
    #[error("failed to create commands publisher: {0}")]
    PublisherCreate(String),
    /// Failed to loan a slice sample.
    #[error("failed to loan command sample: {0}")]
    Loan(String),
    /// Failed to send the loaned sample.
    #[error("failed to send command sample: {0}")]
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
/// Held alongside its [`Node`] and [`PortFactory`] so the publisher's
/// shared-memory descriptors stay live for the lifetime of the thread.
struct ProducerState {
    _node: Node<ipc::Service>,
    _service: PortFactory<ipc::Service, [u8], ()>,
    publisher: Publisher<ipc::Service, [u8], ()>,
}

thread_local! {
    /// One iceoryx2 publisher per OS thread. See the module-level note on
    /// threading for the rationale. Const-initialised so the slot is a plain
    /// TLS load — required for the `pthread_atfork` child handler to access
    /// it without invoking a lazy initializer in a post-fork context.
    static PRODUCER: RefCell<Option<ProducerState>> = const { RefCell::new(None) };
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
    let service_name = COMMANDS
        .try_into()
        .map_err(|error| ProducerError::ServiceOpen(format!("invalid service name: {error}")))?;
    // Declare the same `subscriber_max_buffer_size` and `max_publishers` the
    // daemon configures so a producer that races the daemon to `open_or_create`
    // doesn't seed the service with iceoryx2's default attributes (which would
    // drop bursts of lifecycle envelopes and fail the integration matrix's
    // ~32 worker threads).
    let service = node
        .service_builder(&service_name)
        .publish_subscribe::<[u8]>()
        .subscriber_max_buffer_size(LIFECYCLE_SUBSCRIBER_BUFFER_SIZE)
        .max_publishers(MAX_PUBLISHERS_PER_SERVICE)
        .open_or_create()
        .map_err(|error| ProducerError::ServiceOpen(error.to_string()))?;
    let publisher = service
        .publisher_builder()
        .initial_max_slice_len(COMMANDS_MAX_PAYLOAD_BYTES)
        // Block on a full subscriber buffer instead of iceoryx2's default
        // `DiscardSample`. Lifecycle envelopes must reach the daemon for the
        // per-trace state machine to advance — silent drops surface in the
        // integration matrix as recordings stuck in `writing` forever.
        .unable_to_deliver_strategy(UnableToDeliverStrategy::Block)
        .create()
        .map_err(|error| ProducerError::PublisherCreate(error.to_string()))?;
    Ok(ProducerState {
        _node: node,
        _service: service,
        publisher,
    })
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
/// [`with_producer`] rebuilds a fresh iceoryx2 node + publisher whose
/// descriptors belong to the new process. The inherited `ProducerState` is
/// `mem::forget`'d on purpose: running its `Drop` would notify the daemon's
/// bookkeeping for the *parent's* still-live publisher, unregistering it.
///
/// `thread_local!` slots owned by *other* threads at fork time are
/// inaccessible from the child's surviving thread, but that is fine — those
/// threads no longer exist in the child, and Rust's per-thread TLS allocation
/// means no stale slot can ever be observed from the child.
extern "C" fn on_fork_in_child() {
    PRODUCER.with(|cell| {
        if let Some(stale) = cell.borrow_mut().take() {
            std::mem::forget(stale);
        }
    });
}

fn publish_envelope(envelope: &Envelope) -> Result<(), ProducerError> {
    let bytes = envelope.encode()?;
    if bytes.len() > COMMANDS_MAX_PAYLOAD_BYTES {
        return Err(ProducerError::PayloadTooLarge {
            actual: bytes.len(),
            limit: COMMANDS_MAX_PAYLOAD_BYTES,
        });
    }
    with_producer(|state| {
        let sample = state
            .publisher
            .loan_slice_uninit(bytes.len())
            .map_err(|error| ProducerError::Loan(error.to_string()))?;
        let sample = sample.write_from_slice(&bytes);
        sample
            .send()
            .map_err(|error| ProducerError::Send(error.to_string()))?;
        Ok(())
    })
}

/// Announce the start of a recording session.
#[pyfunction]
#[pyo3(signature = (recording_id, robot_id = None, robot_name = None, dataset_id = None, dataset_name = None))]
fn start_recording(
    recording_id: &str,
    robot_id: Option<String>,
    robot_name: Option<String>,
    dataset_id: Option<String>,
    dataset_name: Option<String>,
) -> PyResult<()> {
    if recording_id.is_empty() {
        return Err(PyValueError::new_err("recording_id must not be empty"));
    }
    let envelope = Envelope::StartRecording {
        recording_id: recording_id.to_string(),
        robot_id,
        robot_name,
        dataset_id,
        dataset_name,
    };
    publish_envelope(&envelope)?;
    Ok(())
}

/// Open a new trace inside an active recording.
///
/// The Python adaptor publishes this once per `(recording_id, trace_id)` pair
/// before the first [`send_data`] for that trace, matching the per-trace state
/// machine described in §5 of the rewrite plan.
#[pyfunction]
#[pyo3(signature = (recording_id, trace_id, data_type, data_type_name = None))]
fn start_trace(
    recording_id: &str,
    trace_id: &str,
    data_type: &str,
    data_type_name: Option<String>,
) -> PyResult<()> {
    if recording_id.is_empty() || trace_id.is_empty() || data_type.is_empty() {
        return Err(PyValueError::new_err(
            "recording_id, trace_id and data_type must not be empty",
        ));
    }
    let envelope = Envelope::StartTrace {
        recording_id: recording_id.to_string(),
        trace_id: trace_id.to_string(),
        data_type: data_type.to_string(),
        data_type_name: data_type_name.filter(|value| !value.is_empty()),
    };
    publish_envelope(&envelope)?;
    Ok(())
}

/// Send one frame/sample for a trace.
///
/// `trace_id` is caller-supplied so the producer stays stateless; the Python
/// adaptor allocates trace IDs (typically once per recording) and keeps the
/// mapping. The adaptor must have already published an
/// [`Envelope::StartTrace`] for `trace_id` before the first frame.
#[pyfunction]
#[pyo3(signature = (trace_id, payload, timestamp_ns = 0, timestamp_s = None))]
fn send_data(
    trace_id: &str,
    payload: &[u8],
    timestamp_ns: i64,
    timestamp_s: Option<f64>,
) -> PyResult<()> {
    if trace_id.is_empty() {
        return Err(PyValueError::new_err("trace_id must not be empty"));
    }
    // Diagnostic per-frame log. Only fires when `NCD_PRODUCER_LOG` is set;
    // the log carries each call's wall-clock duration so producer-throughput
    // analysis can grep + awk the file for per-trace percentiles. When the
    // gate is off the whole block compiles to a single atomic-bool load.
    let log_enabled = producer_log_enabled();
    let started = if log_enabled {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let envelope = Envelope::frame(
        trace_id.to_string(),
        timestamp_ns,
        timestamp_s,
        payload.to_vec(),
    );
    let encode_done = started.map(|t| t.elapsed());
    publish_envelope(&envelope)?;
    if let Some(start) = started {
        let total = start.elapsed();
        let encode_us = encode_done.map(|d| d.as_micros()).unwrap_or(0);
        let publish_us = total.as_micros().saturating_sub(encode_us);
        // Log only every 100th call per thread to keep the file size
        // sensible at 30k+ envelopes/sec, while still giving enough samples
        // for a percentile read. Stats summary is bucketed by thread via the
        // trace_id tail so different traces interleave cleanly.
        let trace_tail = trace_id_tail(trace_id);
        let count = increment_thread_call_counter();
        if count.is_multiple_of(100) {
            tracing::debug!(
                trace_id = trace_tail,
                payload_len = payload.len(),
                encode_us,
                publish_us,
                total_us = total.as_micros() as u64,
                call_count = count,
                "send_data timing"
            );
        }
    }
    Ok(())
}

/// Send a batch of scalar joint samples in one PyO3 call.
///
/// Each `(trace_id, value)` item becomes its own `Frame` envelope, but the
/// JSON payload construction and iceoryx2 publishes all happen inside this
/// single Rust call. Compared to the legacy per-item shim that crossed the
/// PyO3 boundary and ran CPython `json.dumps` for every joint, this collapses:
///
/// - 10× PyO3 boundary crossings → 1
/// - 10× CPython `json.dumps` → 10× `write!` against a small `Vec<u8>`
/// - 10× thread-local publisher lookup → 1 (the inner loop reuses the slot)
///
/// Wire format is intentionally unchanged: the daemon still receives one
/// `Frame` envelope per item carrying the same compact
/// `{"timestamp":<f64>,"value":<f64>}` JSON the legacy shim produced, so no
/// dispatcher or trace-actor changes are required.
#[pyfunction]
#[pyo3(signature = (items, timestamp_ns, timestamp_s = None))]
fn send_batched_joint_data(
    py: Python<'_>,
    items: Vec<(String, f64)>,
    timestamp_ns: i64,
    timestamp_s: Option<f64>,
) -> PyResult<()> {
    if items.is_empty() {
        return Ok(());
    }
    // Fall back to ns→s if the caller omits timestamp_s so the per-frame
    // "timestamp" field still has a sensible value to write into trace.json.
    let timestamp_for_json =
        timestamp_s.unwrap_or_else(|| timestamp_ns as f64 / 1_000_000_000.0);
    let log_enabled = producer_log_enabled();

    // The JSON formatting and iceoryx2 publishes touch no Python state —
    // releasing the GIL here lets the test's other producer threads (camera
    // at 120 Hz, custom_1d at 3.1 kHz, the sibling joint streams) keep
    // making progress while this thread is inside Rust. At 1 kHz × multiple
    // joint streams the per-call GIL hold was costing the workload several
    // hundred microseconds of avoidable serialisation each second.
    // Pre-validate every trace_id before we publish anything. Without this
    // check a malformed item N would leave items 0..N already on the wire
    // and surface the error as an exception — leaving the daemon to
    // process a partial batch the caller assumes it never sent. Hoisting
    // the loop also keeps the validation cost paid once per batch rather
    // than during the GIL-released hot path.
    if let Some(bad_index) = items.iter().position(|(trace_id, _)| trace_id.is_empty()) {
        return Err(PyValueError::new_err(format!(
            "trace_id must not be empty (item index {bad_index})"
        )));
    }
    py.allow_threads(|| -> PyResult<()> {
        for (trace_id, value) in &items {
            let started = if log_enabled {
                Some(std::time::Instant::now())
            } else {
                None
            };
            // serde_json (via ryu) always emits at least one fractional
            // digit for f64 — so integer-valued joint values land on disk
            // as `1.0`, not `1`, matching Python's `json.dumps` shape and
            // keeping the cloud-side data verification happy.
            let payload = serde_json::to_vec(&ScalarFrameEntry {
                timestamp: timestamp_for_json,
                value: *value,
            })
            .map_err(|error| {
                PyRuntimeError::new_err(format!(
                    "failed to encode joint frame JSON: {error}"
                ))
            })?;
            let payload_len = payload.len();
            let envelope =
                Envelope::frame(trace_id.clone(), timestamp_ns, timestamp_s, payload);
            let encode_done = started.map(|t| t.elapsed());
            publish_envelope(&envelope)?;
            if let Some(start) = started {
                let total = start.elapsed();
                let encode_us = encode_done.map(|d| d.as_micros()).unwrap_or(0);
                let publish_us = total.as_micros().saturating_sub(encode_us);
                let trace_tail = trace_id_tail(trace_id);
                let count = increment_thread_call_counter();
                if count.is_multiple_of(100) {
                    tracing::debug!(
                        trace_id = trace_tail,
                        payload_len,
                        encode_us,
                        publish_us,
                        total_us = total.as_micros() as u64,
                        call_count = count,
                        "send_data timing"
                    );
                }
            }
        }
        Ok(())
    })
}

/// Per-item JSON shape written to `trace.json` for scalar joint streams.
///
/// Field order matches the legacy `BATCHED_JOINT_DATA` zmq path
/// ([data_bridge.py:508](../../../neuracore/data_daemon/communications_management/consumer/data_bridge.py#L508))
/// so byte-level diffs against historical recordings stay stable.
#[derive(serde::Serialize)]
struct ScalarFrameEntry {
    timestamp: f64,
    value: f64,
}

thread_local! {
    /// Per-thread monotonic counter of `send_data` calls. Used to decimate
    /// the diagnostic log to one line per 100 calls without locking. Lives
    /// alongside the `PRODUCER` slot so it shares the same thread lifetime.
    static THREAD_CALL_COUNTER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn increment_thread_call_counter() -> u64 {
    THREAD_CALL_COUNTER.with(|cell| {
        let next = cell.get().saturating_add(1);
        cell.set(next);
        next
    })
}

/// Tail of a trace UUID for log readability. The integration test logs the
/// same suffix on the daemon side, so the two timelines join on the same
/// label without a full 36-char UUID per line.
fn trace_id_tail(trace_id: &str) -> &str {
    let len = trace_id.len();
    if len <= 8 {
        trace_id
    } else {
        &trace_id[len - 8..]
    }
}

/// Returns true once `NCD_PRODUCER_LOG=1` (or any non-empty value other than
/// `0`/`false`) has been observed *and* the tracing subscriber has been
/// installed. The first call initialises the subscriber against a file path
/// taken from `NCD_PRODUCER_LOG_FILE` (default `/tmp/ncd_producer.log`); the
/// subscriber is shared across all Python threads in the process.
fn producer_log_enabled() -> bool {
    static INIT: Once = Once::new();
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let raw = std::env::var("NCD_PRODUCER_LOG").unwrap_or_default();
        let enabled = !raw.is_empty() && raw != "0" && !raw.eq_ignore_ascii_case("false");
        if enabled {
            INIT.call_once(install_producer_tracing);
        }
        enabled
    })
}

/// Install a `tracing-subscriber` that writes to the file named by
/// `NCD_PRODUCER_LOG_FILE` (defaults to `<temp-dir>/ncd_producer.log`,
/// resolved via `std::env::temp_dir` so the path is portable across
/// Linux/macOS/Windows hosts).
///
/// Installing a global subscriber from a cdylib is normally a footgun
/// because the embedding process loses control of formatting. We accept
/// that cost here because this code is gated entirely behind
/// `NCD_PRODUCER_LOG`: the subscriber is only constructed when the
/// operator explicitly opts in to per-frame producer diagnostics, and
/// `try_init` short-circuits when the embedder has already installed
/// their own subscriber — our events route through the embedder's
/// subscriber in that case and the file write is a no-op.
fn install_producer_tracing() {
    let path: PathBuf = std::env::var_os("NCD_PRODUCER_LOG_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("ncd_producer.log"));
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path);
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"));
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_ansi(false);
    let result = match file {
        Ok(handle) => builder
            .with_writer(std::sync::Mutex::new(handle))
            .try_init(),
        Err(_) => builder.with_writer(std::io::stderr).try_init(),
    };
    // If `try_init` returns Err the host already installed a subscriber and
    // our events will use it; nothing more to do.
    let _ = result;
}

/// Announce the resolution of an upcoming video trace.
///
/// The daemon-side trace actor uses the presence of this envelope to switch
/// the per-trace writer from JSON to the NUT video pipeline. The producer
/// must send this before the first [`Envelope::Frame`] payload carrying pixel
/// bytes, otherwise the early frames will be wrapped into the JSON sidecar
/// instead of spooled to the muxer.
#[pyfunction]
#[pyo3(signature = (trace_id, width, height))]
fn open_frame_stream(trace_id: &str, width: u32, height: u32) -> PyResult<()> {
    if trace_id.is_empty() {
        return Err(PyValueError::new_err("trace_id must not be empty"));
    }
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("width and height must be non-zero"));
    }
    let envelope = Envelope::OpenFrameStream {
        trace_id: trace_id.to_string(),
        width,
        height,
    };
    publish_envelope(&envelope)?;
    Ok(())
}

/// Close a previously opened trace.
#[pyfunction]
#[pyo3(signature = (trace_id))]
fn end_trace(trace_id: &str) -> PyResult<()> {
    if trace_id.is_empty() {
        return Err(PyValueError::new_err("trace_id must not be empty"));
    }
    let envelope = Envelope::EndTrace {
        trace_id: trace_id.to_string(),
    };
    publish_envelope(&envelope)?;
    Ok(())
}

/// Announce that a recording session has finished.
#[pyfunction]
#[pyo3(signature = (recording_id))]
fn stop_recording(recording_id: &str) -> PyResult<()> {
    if recording_id.is_empty() {
        return Err(PyValueError::new_err("recording_id must not be empty"));
    }
    let envelope = Envelope::StopRecording {
        recording_id: recording_id.to_string(),
    };
    publish_envelope(&envelope)?;
    Ok(())
}

/// Python module entrypoint registered as `neuracore.data_daemon._native_producer`.
#[pymodule]
fn _native_producer(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(start_recording, module)?)?;
    module.add_function(wrap_pyfunction!(start_trace, module)?)?;
    module.add_function(wrap_pyfunction!(send_data, module)?)?;
    module.add_function(wrap_pyfunction!(send_batched_joint_data, module)?)?;
    module.add_function(wrap_pyfunction!(open_frame_stream, module)?)?;
    module.add_function(wrap_pyfunction!(end_trace, module)?)?;
    module.add_function(wrap_pyfunction!(stop_recording, module)?)?;
    Ok(())
}
