// The crate-level docs link to the producer's private `#[pyfunction]` entry
// points and internal modules. CI builds docs with `--document-private-items`,
// so those links resolve; silence the lint that flags them as private.
#![allow(rustdoc::private_intra_doc_links)]

//! PyO3 producer client for the Neuracore data daemon — a *thin shipper*.
//!
//! This crate ships as `neuracore.data_daemon._data_bridge` inside the
//! Python wheel. It knows nothing about recordings: it publishes
//! source/sensor/timestamp-tagged data and three fire-and-forget lifecycle
//! events, and the daemon decides which recording (if any) each datum belongs
//! to. There is no trace registry, no per-frame sequence numbers, and no
//! recording identity on the wire.
//!
//! The surface the SDK's logging layer drives, all keyed by the **source**
//! `(robot_id, robot_instance)`:
//!
//! - [`start_recording`] / [`stop_recording`] / [`cancel_recording`] publish
//!   one lifecycle envelope each, carrying the lifecycle wall-clock
//!   `*_at_ns`.
//! - [`log_joints`] / [`log_json`] publish data envelopes tagged with the
//!   sensor `(data_type, sensor_name)` and capture `timestamp_ns`.
//! - [`log_frame`] spools raw RGB into per-`(source, sensor)` NUT chunk files
//!   under a recording-independent inbox and announces each finished chunk
//!   with [`VideoChunkReady`](data_daemon_shared::Envelope::VideoChunkReady); the
//!   daemon buckets the chunk into a recording by its frame timestamps,
//!   relinks the NUT under that recording, and transcodes it.
//!
//! ## Module layout
//!
//! This file is a thin PyO3 façade: the `#[pyfunction]` wrappers do argument
//! validation, release the GIL, and delegate into the submodules.
//!
//! - [`paths`] — filesystem layout shared with the daemon (recordings root,
//!   spool paths, `(source, sensor)` stream keys).
//! - [`publisher`] — per-thread iceoryx2 publisher state, fork safety, the
//!   synchronous `publish`, and the background data-publisher thread.
//! - [`writer`] — the background video-writer thread, the in-progress
//!   video-chunk registry, and chunk seal/announce/flush logic.
//! - [`query`] — recording-id resolution over the `queries` service.
//! - [`nut_writer`] — minimal NUT-container muxer for raw RGB video.

pub mod nut_writer;

mod paths;
mod publisher;
mod query;
mod writer;

use data_daemon_shared::{Envelope, RecordingIdQuery};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::publisher::{now_ns, publish, publisher_tx, ProducerError, PublishMsg};
use crate::query::{resolve_recording_id, wait_until_ready as wait_until_ready_impl};
use crate::writer::{writer_queue, FrameJob, WriterMsg};

/// Announce that a recording has started for a source. Fire-and-forget: the
/// daemon opens a window and owns all recording identity.
///
/// The producer stamps the window's lower bound on the publish clock
/// (`publish_timestamp_ns`, always wall-clock now) — that, never the caller's
/// timestamp, is what the daemon uses for window membership, so a synthetic
/// capture time can't shift the window or clip data. Separately, the recording's
/// *capture* timestamp (`timestamp_ns` when supplied, else the publish time) is
/// what the daemon stores as `start_timestamp_ns` and POSTs as the backend
/// `start_time`. The capture timestamp is returned so the caller can use it as
/// the marker that resolves the daemon-assigned cloud recording id
/// (`get_recording_id`) for this exact recording.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, robot_name = None, dataset_id = None, dataset_name = None, timestamp_ns = None))]
fn start_recording(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    robot_name: Option<String>,
    dataset_id: Option<String>,
    dataset_name: Option<String>,
    timestamp_ns: Option<i64>,
) -> PyResult<i64> {
    if robot_id.is_empty() {
        return Err(PyValueError::new_err("robot_id must not be empty"));
    }
    let robot_id = robot_id.to_string();
    py.detach(|| -> PyResult<i64> {
        let publish_timestamp_ns = now_ns();
        // Caller-supplied capture time, mirroring the `log_*` timestamp default
        // (publish clock when omitted). Decoupled from the window boundary.
        let capture_timestamp_ns = timestamp_ns.unwrap_or(publish_timestamp_ns);
        publish(&Envelope::StartRecording {
            robot_id,
            robot_instance,
            robot_name,
            dataset_id,
            dataset_name,
            publish_timestamp_ns,
            timestamp_ns: capture_timestamp_ns,
        })?;
        Ok(capture_timestamp_ns)
    })
}

/// Log one scalar sample for each of several joints captured at the same
/// instant, packed into one `BatchedData` envelope.
///
/// **Flattened transfer:** the joint names arrive as a single `\0`-joined
/// string and the values as one flat list, so the GIL-held cost is one string
/// copy plus one `Vec<f64>` extraction. The previous `Vec<(String, f64)>`
/// signature made PyO3 extract N `(name, value)` tuples — N allocations + N
/// downcasts under the GIL — which dominated this call at high joint counts
/// (~1000 joints ≈ 2 ms). The names are split and zipped with the values on the
/// publisher thread, off this path.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, data_type, names, values, timestamp_ns, timestamp_s = None))]
#[allow(clippy::too_many_arguments)]
fn log_joints(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    names: &str,
    values: Vec<f64>,
    timestamp_ns: i64,
    timestamp_s: Option<f64>,
) -> PyResult<()> {
    if robot_id.is_empty() || data_type.is_empty() {
        return Err(PyValueError::new_err(
            "robot_id and data_type must not be empty",
        ));
    }
    if values.is_empty() {
        return Ok(());
    }
    let robot_id = robot_id.to_string();
    let data_type = data_type.to_string();
    let joined_names = names.to_string();
    py.detach(move || {
        // Stamp the window-routing clock at enqueue (inside the recording
        // window). The publisher thread splits the names, zips them with the
        // values, serialises, and publishes the `BatchedData`, keeping that work
        // — and the synchronous IPC publish, which can briefly block on a full
        // commands buffer — off this call.
        let publish_timestamp_ns = now_ns();
        let _ = publisher_tx().send(PublishMsg::Joint {
            robot_id,
            robot_instance,
            data_type,
            joined_names,
            values,
            timestamp_ns,
            timestamp_s,
            publish_timestamp_ns,
        });
    });
    Ok(())
}

/// Log one video frame for a camera. The frame is appended to the
/// `(source, sensor)` in-progress NUT chunk under the inbox; when the chunk
/// crosses the chunk-flush threshold a [`Envelope::VideoChunkReady`] is
/// published.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, data_type, name, width, height, payload, timestamp_ns, timestamp_s = None))]
#[allow(clippy::too_many_arguments)]
fn log_frame(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    name: &str,
    width: u32,
    height: u32,
    payload: PyBuffer<u8>,
    timestamp_ns: i64,
    timestamp_s: Option<f64>,
) -> PyResult<()> {
    if robot_id.is_empty() || data_type.is_empty() || name.is_empty() {
        return Err(PyValueError::new_err(
            "robot_id, data_type and name must not be empty",
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
    // Resolve the recordings root *here*, on the GIL, before copying the frame
    // or handing it to the writer thread. Video is the only path that needs the
    // root (it spools NUT chunks under it), so a host with no `$HOME` and no
    // `NEURACORE_DAEMON_RECORDINGS_ROOT` fails this call with a clear Python
    // error rather than the writer thread spooling somewhere the daemon never
    // looks (silent data loss) or panicking across the FFI boundary.
    crate::paths::recordings_root()
        .map_err(|message| PyRuntimeError::new_err(message.to_string()))?;
    let resolved_timestamp_s = timestamp_s.unwrap_or_else(|| timestamp_ns as f64 / 1_000_000_000.0);

    // SAFETY: PyO3 holds the GIL here, the buffer is validated `u8` and
    // C-contiguous, the length comes from `PyBuffer::item_count`, and we only
    // read. The frame is owned by the caller's numpy array, which may be reused
    // the instant this call returns, so we *copy* it into the job under the GIL
    // (as the buffer protocol requires).
    let data = unsafe {
        std::slice::from_raw_parts(payload.buf_ptr() as *const u8, actual_bytes).to_vec()
    };

    let job = FrameJob {
        robot_id: robot_id.to_string(),
        robot_instance,
        data_type: data_type.to_string(),
        sensor_name: name.to_string(),
        width,
        height,
        timestamp_ns,
        timestamp_s: resolved_timestamp_s,
        data,
    };

    // Hand off to the writer with the GIL released: enqueuing only blocks under
    // sustained overload (the byte caps), and blocking there while holding the
    // GIL would stall every Python thread in the process. A frame that cannot be
    // admitted before the spool-stall window elapses surfaces as an error rather
    // than being silently dropped, so the caller learns the daemon has stalled.
    py.detach(move || writer_queue().push(WriterMsg::Frame(job)))
        .map_err(|_| {
            PyRuntimeError::new_err(
                "video logging stalled: the data daemon is not draining the spool \
                 backlog (frame rejected after 1s of backpressure)",
            )
        })?;
    Ok(())
}

/// Log one JSON sample for any non-joint, non-video data type, delivered
/// verbatim as a `Data` envelope.
///
/// `data_type` is an opaque wire label and `payload` is already-serialized
/// bytes, so this is the generic single-sample path: scalars, poses, gripper
/// amounts, language, point clouds and any future JSON type all flow through
/// here unchanged. The daemon classifies the label downstream
/// (see `content_type_for`); it imposes no allowlist.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, data_type, name, payload, timestamp_ns, timestamp_s = None))]
#[allow(clippy::too_many_arguments)]
fn log_json(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    name: &str,
    payload: &[u8],
    timestamp_ns: i64,
    timestamp_s: Option<f64>,
) -> PyResult<()> {
    if robot_id.is_empty() || data_type.is_empty() || name.is_empty() {
        return Err(PyValueError::new_err(
            "robot_id, data_type and name must not be empty",
        ));
    }
    let robot_id = robot_id.to_string();
    let data_type = data_type.to_string();
    let name = name.to_string();
    let owned_payload = payload.to_vec();
    py.detach(move || {
        // Stamp the window-routing clock at enqueue; the publisher thread
        // publishes the `Data` envelope off this call (see [`PublishMsg::Json`]).
        let publish_timestamp_ns = now_ns();
        let _ = publisher_tx().send(PublishMsg::Json {
            robot_id,
            robot_instance,
            data_type,
            sensor_name: name,
            payload: owned_payload,
            timestamp_ns,
            timestamp_s,
            publish_timestamp_ns,
        });
    });
    Ok(())
}

/// Publish one `StopRecording`, then flush any tail video chunks for the
/// source before returning. The stop is published BEFORE the flush barrier:
/// it shares the calling thread's publisher with `StartRecording`, so sending
/// it first guarantees the daemon sees this stop ahead of the next recording's
/// start. Stamping the boundary early but sending only after a slow flush
/// (~19 ms observed) let the stop reach the wire after the following start —
/// the daemon then retired the wrong window and dropped both recordings' data.
/// Late tail chunks (announced on the writer's port after this stop) route
/// into the just-closed window by the daemon's holdback + closing-window
/// retention, so chunk-before-stop ordering is not required.
///
/// The producer stamps the window's upper bound on the publish clock here
/// (`publish_timestamp_ns`, always wall-clock now at the send), so the whole
/// publish clock is owned by the producer (consistent with the data
/// envelopes). The recording's *capture* stop time (`timestamp_ns` when
/// supplied, else the publish time) is separate — it is stored as
/// `stop_timestamp_ns` and POSTed as the backend `end_time`, never used for
/// window membership.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, timestamp_ns = None))]
fn stop_recording(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    timestamp_ns: Option<i64>,
) -> PyResult<()> {
    if robot_id.is_empty() {
        return Err(PyValueError::new_err("robot_id must not be empty"));
    }
    let robot_id = robot_id.to_string();
    py.detach(|| -> PyResult<()> {
        let publish_timestamp_ns = now_ns();
        // Caller-supplied capture time, mirroring the `log_*` timestamp default
        // (publish clock when omitted). Decoupled from the window boundary.
        let capture_timestamp_ns = timestamp_ns.unwrap_or(publish_timestamp_ns);
        // Publish `StopRecording` FIRST, from THIS (the calling) thread's
        // publisher — the same port as `StartRecording` — stamping the window's
        // upper bound at the actual send. Publishing before the (possibly slow)
        // flush barrier keeps consecutive recordings' start/stop boundaries
        // strictly ordered: the stop can never be reordered behind the next
        // recording's `StartRecording` on this thread.
        publish(&Envelope::StopRecording {
            robot_id: robot_id.clone(),
            robot_instance,
            publish_timestamp_ns,
            timestamp_ns: capture_timestamp_ns,
        })?;
        // Then barrier on the writer: it drains every frame still queued for
        // this source (FIFO), seals the tail chunks and announces them, then
        // acks. Blocking here means `stop_recording` still returns only once
        // those chunks are durably spooled + announced, so a process exit right
        // after the call can't lose them. The tail chunks ride the writer's
        // port and land after this stop; the daemon's holdback + closing-window
        // retention route them into the just-closed window by their in-window
        // open timestamp.
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        // Control messages bypass the frame caps, so this never blocks or stalls.
        let _ = writer_queue().push(WriterMsg::FlushSource {
            robot_id,
            robot_instance,
            ack: ack_tx,
        });
        let _ = ack_rx.recv();
        Ok(())
    })
}

/// Cancel a recording — drop the source's in-progress chunk state without
/// flushing (the daemon's cancel handler removes the relinked artefacts and
/// the recovery sweep reclaims any spooled NUTs).
///
/// A cancel is a recording stop that discards data, so it carries the same
/// capture `timestamp_ns` as `stop_recording` (the caller's value, else the
/// publish clock); the daemon stores it as `stop_timestamp_ns` and POSTs it as
/// the backend `end_time`.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, timestamp_ns = None))]
fn cancel_recording(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    timestamp_ns: Option<i64>,
) -> PyResult<()> {
    if robot_id.is_empty() {
        return Err(PyValueError::new_err("robot_id must not be empty"));
    }
    let robot_id = robot_id.to_string();
    py.detach(|| -> PyResult<()> {
        let capture_timestamp_ns = timestamp_ns.unwrap_or_else(now_ns);
        // Barrier on the writer: it drains any frames still queued for this
        // source (FIFO) and drops the in-progress chunk state without announcing
        // it, then acks. Block until acked so the cancel is ordered after those
        // frames and no late chunk for this recording is announced.
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        // Control messages bypass the frame caps, so this never blocks or stalls.
        let _ = writer_queue().push(WriterMsg::DropSource {
            robot_id: robot_id.clone(),
            robot_instance,
            ack: ack_tx,
        });
        let _ = ack_rx.recv();
        // Publish `CancelRecording` from THIS (the calling) thread's publisher,
        // ordered with Start/Stop on the same port (see the writer module note).
        publish(&Envelope::CancelRecording {
            robot_id,
            robot_instance,
            timestamp_ns: capture_timestamp_ns,
        })?;
        Ok(())
    })
}

/// Resolve the daemon-owned cloud `recording_id` for a recording, blocking with
/// the GIL released until the id is available or `timeout_s` elapses.
///
/// The thin producer never mints recording identity — the daemon allocates the
/// cloud id asynchronously after `/recording/start`. This asks the daemon over
/// the `queries` request-response service (identifying the recording by its
/// source + capture `timestamp_ns` marker) and returns the id once minted, or
/// `None` on timeout / when no daemon is answering. Safe for
/// non-performance-critical paths only (tests, `stop_recording(wait=True)`).
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, timestamp_ns, timeout_s))]
fn get_recording_id(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    timestamp_ns: i64,
    timeout_s: f64,
) -> PyResult<Option<String>> {
    if robot_id.is_empty() {
        return Err(PyValueError::new_err("robot_id must not be empty"));
    }
    let query = RecordingIdQuery {
        robot_id: robot_id.to_string(),
        robot_instance,
        timestamp_ns,
    };
    let request_bytes = query.encode().map_err(ProducerError::from)?;
    py.detach(|| -> PyResult<Option<String>> {
        Ok(resolve_recording_id(&request_bytes, timeout_s)?)
    })
}

/// Wait until the Rust daemon answers a side-effect-free IPC health probe.
#[pyfunction]
#[pyo3(signature = (timeout_s))]
fn wait_until_ready(py: Python<'_>, timeout_s: f64) -> PyResult<Option<u32>> {
    py.detach(|| -> PyResult<Option<u32>> { Ok(wait_until_ready_impl(timeout_s)?) })
}

/// Ask the running daemon to reload its profile config immediately (see
/// [`Envelope::RefreshConfig`]). Published on the caller thread's command port
/// so it is strictly ordered ahead of a subsequent `start_recording` from the
/// same thread.
///
/// Best-effort: the SDK caller ignores failures. With no daemon running the
/// command reaches zero subscribers (a no-op); the profile write already
/// persisted, so the daemon picks it up on its next poll or at launch.
#[pyfunction]
fn refresh_config(py: Python<'_>) -> PyResult<()> {
    py.detach(|| -> PyResult<()> {
        publish(&Envelope::RefreshConfig {})?;
        Ok(())
    })
}

/// Python module entrypoint registered as `neuracore.data_daemon._data_bridge`.
#[pymodule]
fn _data_bridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(start_recording, module)?)?;
    module.add_function(wrap_pyfunction!(log_joints, module)?)?;
    module.add_function(wrap_pyfunction!(log_frame, module)?)?;
    module.add_function(wrap_pyfunction!(log_json, module)?)?;
    module.add_function(wrap_pyfunction!(stop_recording, module)?)?;
    module.add_function(wrap_pyfunction!(cancel_recording, module)?)?;
    module.add_function(wrap_pyfunction!(get_recording_id, module)?)?;
    module.add_function(wrap_pyfunction!(wait_until_ready, module)?)?;
    module.add_function(wrap_pyfunction!(refresh_config, module)?)?;
    Ok(())
}
