// PyO3 0.22's `#[pyfunction]`/`#[pymethods]` expansion includes an `.into()` on
// the `PyResult<T>` return value that fires clippy's `useless_conversion` lint
// when T resolves to `()`. The lint is correct about the generated code but the
// conversion lives in the macro expansion, not anything we wrote, so we silence
// it at the crate level rather than spraying allows over every method.
#![allow(clippy::useless_conversion)]

//! PyO3 WebRTC streaming core for Neuracore — the synchronous, queue-backed
//! replacement for the aiortc stack.
//!
//! This crate ships as `neuracore.core.streaming.p2p._native_webrtc` inside the
//! Python wheel. It exposes two peers, [`Producer`](producer::Producer) (the
//! sole offerer) and [`Consumer`](consumer::Consumer) (answer-only), behind a
//! deliberately **synchronous, thread-safe, queue-backed** API:
//!
//! - Rust owns a tokio runtime and drives it on its own threads (see
//!   [`runtime`]). Python never touches the runtime.
//! - `submit_frame` enqueues onto a bounded queue and returns immediately.
//! - Both peers expose a drainable event queue (see [`events`]).
//!
//! ## Scope
//!
//! PR2 wires the real data plane: a libdatachannel [`RtcPeerConnection`] (via
//! the `datachannel` crate, `media` feature) carrying reliable-ordered data
//! channels with trickle ICE and a control-channel manifest. The producer is
//! the sole offerer; the consumer answers. Video (`add_video_track` /
//! `remove_video_track`) stays stubbed until PR4.
//!
//! [`RtcPeerConnection`]: datachannel::RtcPeerConnection

mod broadcaster;
mod congestion;
mod consumer;
mod events;
mod media;
mod producer;
mod runtime;
mod transport;

use pyo3::prelude::*;

/// The sizes of the three process-global lifecycle registries, as
/// `(producer_fb, media, track_pc)`. The hardening soak test reads this to assert
/// every registry returns to its starting size after add/remove churn — i.e. no
/// leaked entries. Diagnostics only; not part of the streaming API.
#[pyfunction]
fn registry_sizes() -> (usize, usize, usize) {
    (
        producer::producer_fb_len(),
        consumer::media_registry_len(),
        consumer::track_pc_registry_len(),
    )
}

/// Python module entrypoint registered as
/// `neuracore.core.streaming.p2p._native_webrtc`.
#[pymodule]
fn _native_webrtc(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<producer::Producer>()?;
    module.add_class::<consumer::Consumer>()?;
    module.add_class::<broadcaster::Broadcaster>()?;
    // Surface the bounded frame-queue depth so callers/tests can read it
    // without hard-coding the constant.
    module.add("FRAME_QUEUE_CAPACITY", producer::FRAME_QUEUE_CAPACITY)?;
    module.add_function(wrap_pyfunction!(registry_sizes, module)?)?;
    Ok(())
}
