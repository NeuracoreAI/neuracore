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
//! - [`stop_recording`](crate::stop_recording) publishes a
//!   [`StopRecording`](data_daemon_ipc::Envelope::StopRecording) envelope.
//!
//! Phase 4 keeps the producer minimal — there is no sequencing, batching, or
//! shared-slot management here. Sub-phase 4h widens the API to the full
//! `ProducerChannel` Python contract once the per-resolution `frames/<WxH>`
//! services from 4f are exercised end-to-end.
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

use std::cell::RefCell;

use data_daemon_ipc::service_name::{COMMANDS, COMMANDS_MAX_PAYLOAD_BYTES};
use data_daemon_ipc::Envelope;
use iceoryx2::node::{Node, NodeBuilder};
use iceoryx2::port::publisher::Publisher;
use iceoryx2::prelude::ipc;
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
    /// threading for the rationale.
    static PRODUCER: RefCell<Option<ProducerState>> = const { RefCell::new(None) };
}

/// Run `f` against this thread's producer state, lazily building it on first
/// use.
fn with_producer<R>(
    operation: impl FnOnce(&ProducerState) -> Result<R, ProducerError>,
) -> Result<R, ProducerError> {
    PRODUCER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(build_producer_state()?);
        }
        let state = slot
            .as_ref()
            .expect("producer state populated immediately above");
        operation(state)
    })
}

fn build_producer_state() -> Result<ProducerState, ProducerError> {
    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .map_err(|error| ProducerError::NodeCreate(error.to_string()))?;
    let service_name = COMMANDS
        .try_into()
        .map_err(|error| ProducerError::ServiceOpen(format!("invalid service name: {error}")))?;
    let service = node
        .service_builder(&service_name)
        .publish_subscribe::<[u8]>()
        .open_or_create()
        .map_err(|error| ProducerError::ServiceOpen(error.to_string()))?;
    let publisher = service
        .publisher_builder()
        .initial_max_slice_len(COMMANDS_MAX_PAYLOAD_BYTES)
        .create()
        .map_err(|error| ProducerError::PublisherCreate(error.to_string()))?;
    Ok(ProducerState {
        _node: node,
        _service: service,
        publisher,
    })
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
        // `write_payload` is reserved for fixed-size payloads; for slices we
        // copy through `write_from_slice` which fills the loaned region in
        // one memcpy.
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
#[pyo3(signature = (recording_id, trace_id, data_type))]
fn start_trace(recording_id: &str, trace_id: &str, data_type: &str) -> PyResult<()> {
    if recording_id.is_empty() || trace_id.is_empty() || data_type.is_empty() {
        return Err(PyValueError::new_err(
            "recording_id, trace_id and data_type must not be empty",
        ));
    }
    let envelope = Envelope::StartTrace {
        recording_id: recording_id.to_string(),
        trace_id: trace_id.to_string(),
        data_type: data_type.to_string(),
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
#[pyo3(signature = (trace_id, payload, timestamp_ns = 0))]
fn send_data(trace_id: &str, payload: &[u8], timestamp_ns: i64) -> PyResult<()> {
    if trace_id.is_empty() {
        return Err(PyValueError::new_err("trace_id must not be empty"));
    }
    let envelope = Envelope::Frame {
        trace_id: trace_id.to_string(),
        timestamp_ns,
        payload: payload.to_vec(),
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
    module.add_function(wrap_pyfunction!(end_trace, module)?)?;
    module.add_function(wrap_pyfunction!(stop_recording, module)?)?;
    Ok(())
}
