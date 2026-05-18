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
    let envelope = Envelope::frame(
        trace_id.to_string(),
        timestamp_ns,
        timestamp_s,
        payload.to_vec(),
    );
    publish_envelope(&envelope)?;
    Ok(())
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
    module.add_function(wrap_pyfunction!(open_frame_stream, module)?)?;
    module.add_function(wrap_pyfunction!(end_trace, module)?)?;
    module.add_function(wrap_pyfunction!(stop_recording, module)?)?;
    Ok(())
}
