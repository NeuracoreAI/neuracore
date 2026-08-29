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

mod depth;
mod paths;
mod publisher;
mod query;
mod windows;
mod writer;

use data_daemon_shared::{Envelope, FrameDtype, RecordingIdQuery};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::publisher::{
    flush_published_data, now_ns, publish, publisher_tx, ProducerError, PublishMsg,
};
use crate::query::{
    daemon_version as daemon_version_impl, resolve_recording_id,
    wait_until_ready as wait_until_ready_impl,
};
use crate::writer::{note_video_activity, writer_queue, FrameJob, WriterMsg};

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
///
/// `cloud_recording_id` is set only when the backend already minted the id
/// itself (a recording started from the web frontend) — the daemon then
/// reuses it instead of POSTing `/recording/start`.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, robot_name = None, dataset_id = None, dataset_name = None, timestamp_ns = None, cloud_recording_id = None))]
#[allow(clippy::too_many_arguments)]
fn start_recording(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    robot_name: Option<String>,
    dataset_id: Option<String>,
    dataset_name: Option<String>,
    timestamp_ns: Option<i64>,
    cloud_recording_id: Option<String>,
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
        // Open this process's logging gate before the envelope goes out, so a
        // `log_*` on the very next line is not dropped while the daemon is still
        // answering. The claim expires against the daemon's own snapshots.
        windows::claim(&robot_id, robot_instance, publish_timestamp_ns);
        publish(&Envelope::StartRecording {
            robot_id,
            robot_instance,
            robot_name,
            dataset_id,
            dataset_name,
            publish_timestamp_ns,
            timestamp_ns: capture_timestamp_ns,
            cloud_recording_id,
        })?;
        // The writer is deliberately not told where the window opened: a chunk
        // may span the boundary, and the daemon splits it per frame.
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
    // Nothing is recording for this source: drop the sample before copying,
    // spooling or publishing anything. This is where the SDK's Python-side
    // recording gate used to sit, and it sits *before* argument validation for
    // the same reason it used to skip this call entirely — a caller logging
    // while idle sees the same nothing it always did.
    if !windows::is_open(robot_id, robot_instance) {
        return Ok(());
    }
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

// A distinct type (rather than a bare `PyRuntimeError`, which `log_frame` also
// raises for an unrelated failure below) so callers can retry on this specific
// condition without inspecting the message. Subclasses `RuntimeError` so
// existing broad catches still work.
pyo3::create_exception!(
    _data_bridge,
    LoggingStalledError,
    PyRuntimeError,
    "The daemon is not draining its on-disk spool backlog fast enough to admit a frame."
);

/// Log one video frame for a camera. The frame is appended to the
/// `(source, sensor)` in-progress NUT chunk under the inbox; when the chunk
/// crosses the chunk-flush threshold a [`Envelope::VideoChunkReady`] is
/// published.
///
/// `dtype` is the Python-facing wire label (`image.dtype.name`): `"uint8"`
/// for RGB24 frames, `"float16"` / `"float32"` for 2D depth frames (metres).
/// It is parsed once, here, at the native boundary — every internal Rust
/// component downstream (the writer, the daemon) works with the strongly
/// typed [`FrameDtype`] rather than re-validating a string.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, data_type, name, width, height, dtype, payload, timestamp_ns, timestamp_s = None))]
#[allow(clippy::too_many_arguments)]
fn log_frame(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    name: &str,
    width: u32,
    height: u32,
    dtype: &str,
    payload: PyBuffer<u8>,
    timestamp_ns: i64,
    timestamp_s: Option<f64>,
) -> PyResult<()> {
    // Not recording: drop it before any work (see `log_joints`).
    if !windows::is_open(robot_id, robot_instance) {
        return Ok(());
    }
    if robot_id.is_empty() || data_type.is_empty() || name.is_empty() {
        return Err(PyValueError::new_err(
            "robot_id, data_type and name must not be empty",
        ));
    }
    let frame_dtype = parse_frame_dtype(data_type, dtype).map_err(PyValueError::new_err)?;
    let actual_bytes = payload.item_count();
    validate_frame_payload(frame_dtype, width, height, actual_bytes)
        .map_err(PyValueError::new_err)?;
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
        dtype: frame_dtype,
        // Stamped on the caller's thread while the owning recording is open,
        // not on the writer thread, which may reach the frame after the stop.
        publish_ns: now_ns(),
        timestamp_ns,
        timestamp_s: resolved_timestamp_s,
        data,
    };

    // Hand off to the writer with the GIL released: enqueuing only blocks under
    // sustained overload (the byte caps), and blocking there while holding the
    // GIL would stall every Python thread in the process. A frame that cannot be
    // admitted before the spool-stall window elapses surfaces as an error rather
    // than being silently dropped, so the caller learns the daemon has stalled.
    py.detach(move || {
        note_video_activity(&job.robot_id, job.robot_instance, job.publish_ns);
        writer_queue().push(WriterMsg::Frame(job))
    })
    .map_err(|_| {
        LoggingStalledError::new_err(
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
    // Not recording: drop it before any work (see `log_joints`).
    if !windows::is_open(robot_id, robot_instance) {
        return Ok(());
    }
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

/// Drain the data publisher, then publish one `StopRecording`.
///
/// Two barriers, in order: the data publisher's queue is drained first, so the
/// recording's joint/JSON tail is on the wire before the window's upper bound
/// is stamped, and only then is the stop published. The writer's tail video
/// chunks are *not* sealed here — [`flush_source`] does that, and every caller
/// must invoke it straight after this call.
///
/// The stop goes out BEFORE the writer flush barrier because it shares the
/// calling thread's publisher with `StartRecording`: sending it first is what
/// keeps the daemon from seeing this stop after the next recording's start.
/// Late tail chunks still route into the just-closed window, via the daemon's
/// holdback and the `SourceFlushed` marker.
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
        // Drain the data publisher before stamping the window's upper bound.
        // `log_joint_*` / `log_json` only hand their envelope to the background
        // publisher thread's unbounded queue, which nothing drains at process
        // exit — so the recording's samples are only durable once this returns.
        // Draining before the stop (rather than after) also puts them ahead of
        // it on the wire, so they never depend on the daemon's closing-window
        // retention to be routed.
        //
        // Ordering with the stop below is unaffected: data rides the background
        // publisher's port, while `StartRecording`/`StopRecording` share the
        // calling thread's own publisher and stay in program order on it.
        flush_published_data();
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
        // Only now: the daemon's window is open until it receives the stop, so
        // closing this process's gate any earlier would discard data the daemon
        // would still have accepted.
        windows::release(&robot_id, robot_instance);
        // Split from [`flush_source`] so the caller can close its logging gate
        // in between, rather than after a backlog-length barrier.
        Ok(())
    })
}

/// Run the writer's stop barrier for a source: drain every frame still queued
/// for it (FIFO), seal and announce the tail chunks, publish the
/// `SourceFlushed` marker, and drain the data publisher's queue onto the wire.
///
/// Mandatory after [`stop_recording`] — the tail chunks are sealed nowhere
/// else. Blocking, so once it returns a process exit cannot lose them; the
/// daemon's holdback and the `SourceFlushed` marker route them into the
/// just-closed window. Idempotent.
#[pyfunction]
fn flush_source(py: Python<'_>, robot_id: &str, robot_instance: i64) -> PyResult<()> {
    if robot_id.is_empty() {
        return Err(PyValueError::new_err("robot_id must not be empty"));
    }
    let robot_id = robot_id.to_string();
    py.detach(|| {
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        // Control messages bypass the frame caps, so this never blocks or stalls.
        let _ = writer_queue().push(WriterMsg::FlushSource {
            robot_id,
            robot_instance,
            ack: ack_tx,
        });
        let _ = ack_rx.recv();
        // The writer announces its sealed tail chunks by pushing `Announce`
        // onto the data publisher's queue, so the ack above means spooled and
        // queued, not published. Drain once more to put them on the wire.
        flush_published_data();
    });
    Ok(())
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
            robot_id: robot_id.clone(),
            robot_instance,
            timestamp_ns: capture_timestamp_ns,
        })?;
        windows::release(&robot_id, robot_instance);
        Ok(())
    })
}

/// Parse a NumPy dtype label and validate it against the declared video type.
///
/// RGB frames must use `uint8`; depth frames must use `float16` or `float32`.
/// Validating the pair at the native boundary prevents incorrectly typed frames
/// from entering the writer and being stored under the wrong trace type.
fn parse_frame_dtype(data_type: &str, dtype: &str) -> Result<FrameDtype, String> {
    let frame_dtype = FrameDtype::from_wire_label(dtype).ok_or_else(|| {
        format!(
            "unsupported video frame dtype {dtype:?}; expected one of: \
             uint8, float16, float32"
        )
    })?;

    let valid_pair = matches!(
        (data_type, frame_dtype),
        ("RGB_IMAGES", FrameDtype::Rgb8)
            | ("DEPTH_IMAGES", FrameDtype::DepthF16)
            | ("DEPTH_IMAGES", FrameDtype::DepthF32)
    );

    if !valid_pair {
        return Err(format!(
            "video data type {data_type:?} is incompatible with dtype {dtype:?}; \
             expected RGB_IMAGES + uint8 or DEPTH_IMAGES + float16/float32"
        ));
    }

    Ok(frame_dtype)
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

/// Ask the running daemon which neuracore version it was built from.
///
/// Returns `None` when nothing answered within `timeout_s`, which the SDK
/// reads as a daemon older than this query.
#[pyfunction]
#[pyo3(signature = (timeout_s))]
fn daemon_version(py: Python<'_>, timeout_s: f64) -> PyResult<Option<String>> {
    py.detach(|| -> PyResult<Option<String>> { Ok(daemon_version_impl(timeout_s)?) })
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
    module.add_function(wrap_pyfunction!(flush_source, module)?)?;
    module.add_function(wrap_pyfunction!(cancel_recording, module)?)?;
    module.add_function(wrap_pyfunction!(get_recording_id, module)?)?;
    module.add_function(wrap_pyfunction!(wait_until_ready, module)?)?;
    module.add_function(wrap_pyfunction!(daemon_version, module)?)?;
    module.add_function(wrap_pyfunction!(refresh_config, module)?)?;
    module.add(
        "LoggingStalledError",
        module.py().get_type::<LoggingStalledError>(),
    )?;
    Ok(())
}

/// Validate a `log_frame` payload against its declared dtype and geometry.
///
/// Pulled out of [`log_frame`] as a pure function (no `PyBuffer`/GIL
/// involvement) so the native bridge's size-validation rules — the "Native
/// bridge validation" contract every frame must satisfy before it is ever
/// enqueued — are directly unit-testable without standing up a Python
/// interpreter.
fn validate_frame_payload(
    dtype: FrameDtype,
    width: u32,
    height: u32,
    actual_bytes: usize,
) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err("width and height must be non-zero".to_string());
    }
    let expected_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(dtype.bytes_per_pixel()))
        .ok_or_else(|| "width * height * bytes_per_pixel overflows".to_string())?;
    if actual_bytes != expected_bytes {
        return Err(format!(
            "video frame buffer is {actual_bytes} bytes; expected width*height*{} = \
             {expected_bytes} bytes for dtype {dtype:?}",
            dtype.bytes_per_pixel(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_rgb_uint8_frame_is_accepted() {
        assert!(validate_frame_payload(FrameDtype::Rgb8, 4, 4, 4 * 4 * 3).is_ok());
    }

    #[test]
    fn valid_depth_f16_frame_is_accepted() {
        assert!(validate_frame_payload(FrameDtype::DepthF16, 4, 4, 4 * 4 * 2).is_ok());
    }

    #[test]
    fn valid_depth_f32_frame_is_accepted() {
        assert!(validate_frame_payload(FrameDtype::DepthF32, 4, 4, 4 * 4 * 4).is_ok());
    }

    #[test]
    fn rgb_frame_with_depth_sized_payload_is_rejected() {
        // Same byte count as a valid 4x4 f32 depth frame, but declared RGB —
        // must be rejected, not silently accepted because the lengths differ
        // from *some* valid combination.
        let err = validate_frame_payload(FrameDtype::Rgb8, 4, 4, 4 * 4 * 4).unwrap_err();
        assert!(err.contains("expected width*height*3"), "got: {err}");
    }

    #[test]
    fn depth_frame_with_rgb_sized_payload_is_rejected() {
        let err = validate_frame_payload(FrameDtype::DepthF16, 4, 4, 4 * 4 * 3).unwrap_err();
        assert!(err.contains("expected width*height*2"), "got: {err}");
    }

    #[test]
    fn zero_dimensions_are_rejected_for_every_dtype() {
        for dtype in [FrameDtype::Rgb8, FrameDtype::DepthF16, FrameDtype::DepthF32] {
            assert!(validate_frame_payload(dtype, 0, 4, 0).is_err());
            assert!(validate_frame_payload(dtype, 4, 0, 0).is_err());
        }
    }

    #[test]
    fn overflowing_dimensions_reject_cleanly_instead_of_wrapping() {
        // u32::MAX * u32::MAX * 4 overflows usize on 32-bit targets and would
        // wrap silently under an unchecked multiply; on 64-bit it still
        // overflows once cubed against bytes_per_pixel for a large enough
        // combination. Either way this must error, never panic or wrap.
        let result = validate_frame_payload(FrameDtype::DepthF32, u32::MAX, u32::MAX, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("overflows"));
    }

    #[test]
    fn wrong_byte_length_produces_a_clear_rgb_error() {
        // Regression guard: the existing RGB error message shape (byte counts
        // + the expected formula) must stay meaningful after adding dtype.
        let err = validate_frame_payload(FrameDtype::Rgb8, 10, 10, 42).unwrap_err();
        assert!(err.contains("42 bytes"), "got: {err}");
        assert!(err.contains("300 bytes"), "got: {err}");
    }

    #[test]
    fn unsupported_dtype_label_is_rejected() {
        assert_eq!(FrameDtype::from_wire_label("int32"), None);
        assert_eq!(FrameDtype::from_wire_label("float64"), None);
    }

    #[test]
    fn valid_data_type_dtype_pairs_are_accepted() {
        assert_eq!(
            parse_frame_dtype("RGB_IMAGES", "uint8"),
            Ok(FrameDtype::Rgb8)
        );
        assert_eq!(
            parse_frame_dtype("DEPTH_IMAGES", "float16"),
            Ok(FrameDtype::DepthF16)
        );
        assert_eq!(
            parse_frame_dtype("DEPTH_IMAGES", "float32"),
            Ok(FrameDtype::DepthF32)
        );
    }

    #[test]
    fn invalid_data_type_dtype_pairs_are_rejected() {
        for (data_type, dtype) in [
            ("RGB_IMAGES", "float16"),
            ("RGB_IMAGES", "float32"),
            ("DEPTH_IMAGES", "uint8"),
        ] {
            let error = parse_frame_dtype(data_type, dtype).unwrap_err();
            assert!(error.contains("incompatible"), "unexpected error: {error}");
        }
    }
}
