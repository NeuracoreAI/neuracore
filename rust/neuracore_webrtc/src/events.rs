//! The drainable, thread-safe event queue both peers expose to Python.
//!
//! Rust tasks running on the core's tokio runtime [`push`](EventQueue::push)
//! events without ever touching the GIL; Python pulls them with a single
//! synchronous [`drain_events`] call that converts each queued [`Event`] into a
//! plain `dict` under the GIL. This keeps the producer side lock-free of Python
//! and the consumer side cheap: one lock acquisition drains the whole backlog.
//!
//! Each drained dict carries a `"kind"` discriminator. The six kinds mandated
//! by the API contract are:
//!
//! - `on_state` — connection-state transitions (`{"kind", "state"}`)
//! - `on_track_added` — a remote track appeared (`{"kind", "track_id", "mid"}`)
//! - `on_track_removed` — a remote track went away (`{"kind", "mid"}`)
//! - `on_data_channel` — a remote data channel opened (`{"kind", "label", "kind_hint"}`)
//! - `on_message` — a message arrived on a data channel (`{"kind", "label", "data"}`)
//! - `on_frame` — a decoded video frame (`{"kind", "track_id", "mid", "data", "width", "height"}`)
//! - `on_manifest` — the mid→RobotStreamTrack manifest was republished (`{"kind", "json"}`)
//!
//! Two further kinds carry signaling *out* of the core (the producer is the
//! sole offerer, the consumer answers): `on_local_description`
//! (`{"kind", "sdp_type", "sdp"}`) and `on_local_candidate`
//! (`{"kind", "candidate", "mid"}`). They are part of the surface PR1 compiles
//! against; the core only starts emitting them once PR2 wires real signaling.
//!
//! [`drain_events`]: crate::producer::Producer::drain_events

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

/// A single event surfaced to Python via the drainable queue. Variants map
/// one-to-one onto the `"kind"` values documented on the module.
///
/// `dead_code` is allowed because PR0 only ever constructs `State`; the other
/// variants are the agreed event surface and are emitted from PR2 onward. They
/// are already wired through `kind()`/`to_pydict` so the schema is locked now.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Event {
    /// Connection-state transition, e.g. `"new"`, `"connecting"`, `"closed"`.
    State(String),
    /// A remote media track became available.
    TrackAdded { track_id: String, mid: String },
    /// A previously-added remote media track was removed.
    TrackRemoved { mid: String },
    /// A remote data channel opened. `kind_hint` mirrors the producer-side
    /// `add_data_channel(label, kind)` label (e.g. `"json"`, `"control"`).
    DataChannel { label: String, kind_hint: String },
    /// A message arrived on a data channel.
    Message { label: String, data: Vec<u8> },
    /// A decoded video frame ready for the consumer. `data` is the raw picture
    /// as 8-bit HxWx3 (RGB or BGR; the block codec is colour-order agnostic).
    Frame {
        track_id: String,
        mid: String,
        data: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// The mid→RobotStreamTrack manifest, republished verbatim as JSON text.
    Manifest { json: String },
    /// A locally-created SDP offer/answer to relay over signaling.
    LocalDescription { sdp_type: String, sdp: String },
    /// A locally-gathered ICE candidate to trickle over signaling.
    LocalCandidate {
        candidate: String,
        mid: Option<String>,
    },
    /// A recoverable error surfaced from a hot or lifecycle path instead of
    /// panicking across the FFI boundary: a subprocess crash, a chain-attach or
    /// SDP failure, a send on a closed track, or a reconnect-needed signal. Rendered
    /// as `{"kind": "on_error", "where", "detail", "consumer_id"?}`. `consumer_id`
    /// is present only for a broadcaster's per-consumer error (a shared-encode error
    /// has none); `where` is a short location tag (`"encode"`, `"decode"`,
    /// `"negotiate"`, `"connection"`, `"send"`).
    Error {
        consumer_id: Option<String>,
        location: String,
        detail: String,
    },
    /// A per-consumer event from a [`Broadcaster`](crate::broadcaster::Broadcaster):
    /// the wrapped `inner` event rendered with an extra `"consumer_id"` key so a
    /// fan-out signaling layer routes it to the right consumer. The single 1:1
    /// `Producer`/`Consumer` path never constructs this — its events stay
    /// untagged and byte-identical, so the single-consumer suite is unaffected.
    ForConsumer {
        consumer_id: String,
        inner: Box<Event>,
    },
}

impl Event {
    /// An untagged recoverable error (1:1 `Producer`/`Consumer`, or a
    /// broadcaster's shared-encode error that belongs to no single consumer).
    pub(crate) fn error(location: &str, detail: impl Into<String>) -> Event {
        Event::Error {
            consumer_id: None,
            location: location.to_string(),
            detail: detail.into(),
        }
    }

    /// A recoverable error attributed to one broadcaster consumer.
    pub(crate) fn error_for(consumer_id: &str, location: &str, detail: impl Into<String>) -> Event {
        Event::Error {
            consumer_id: Some(consumer_id.to_string()),
            location: location.to_string(),
            detail: detail.into(),
        }
    }

    /// The stable `"kind"` discriminator placed on the Python dict.
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Event::State(_) => "on_state",
            Event::TrackAdded { .. } => "on_track_added",
            Event::TrackRemoved { .. } => "on_track_removed",
            Event::DataChannel { .. } => "on_data_channel",
            Event::Message { .. } => "on_message",
            Event::Frame { .. } => "on_frame",
            Event::Manifest { .. } => "on_manifest",
            Event::LocalDescription { .. } => "on_local_description",
            Event::LocalCandidate { .. } => "on_local_candidate",
            Event::Error { .. } => "on_error",
            // A wrapped per-consumer event keeps the inner event's kind; the
            // consumer_id rides alongside it on the dict.
            Event::ForConsumer { inner, .. } => inner.kind(),
        }
    }

    /// Render this event as a Python `dict`. Caller holds the GIL.
    fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        // A per-consumer event renders exactly as its inner event plus a
        // `consumer_id` key, so the fan-out signaling layer routes it without
        // changing any other field. Done first so the recursion is obvious.
        if let Event::ForConsumer { consumer_id, inner } = self {
            let dict = inner.to_pydict(py)?;
            dict.set_item("consumer_id", consumer_id)?;
            return Ok(dict);
        }
        let dict = PyDict::new_bound(py);
        dict.set_item("kind", self.kind())?;
        match self {
            Event::State(state) => {
                dict.set_item("state", state)?;
            }
            Event::TrackAdded { track_id, mid } => {
                dict.set_item("track_id", track_id)?;
                dict.set_item("mid", mid)?;
            }
            Event::TrackRemoved { mid } => {
                dict.set_item("mid", mid)?;
            }
            Event::DataChannel { label, kind_hint } => {
                dict.set_item("label", label)?;
                dict.set_item("kind_hint", kind_hint)?;
            }
            Event::Message { label, data } => {
                dict.set_item("label", label)?;
                dict.set_item("data", PyBytes::new_bound(py, data))?;
            }
            Event::Frame {
                track_id,
                mid,
                data,
                width,
                height,
            } => {
                dict.set_item("track_id", track_id)?;
                dict.set_item("mid", mid)?;
                dict.set_item("data", PyBytes::new_bound(py, data))?;
                dict.set_item("width", width)?;
                dict.set_item("height", height)?;
            }
            Event::Manifest { json } => {
                dict.set_item("json", json)?;
            }
            Event::LocalDescription { sdp_type, sdp } => {
                dict.set_item("sdp_type", sdp_type)?;
                dict.set_item("sdp", sdp)?;
            }
            Event::LocalCandidate { candidate, mid } => {
                dict.set_item("candidate", candidate)?;
                dict.set_item("mid", mid.clone())?;
            }
            Event::Error {
                consumer_id,
                location,
                detail,
            } => {
                dict.set_item("where", location)?;
                dict.set_item("detail", detail)?;
                // `consumer_id` is optional: present only for a broadcaster's
                // per-consumer error, so the fan-out signaling layer can route it.
                if let Some(id) = consumer_id {
                    dict.set_item("consumer_id", id)?;
                }
            }
            // Handled by the early return above; the match is over the inner
            // (unwrapped) variants only.
            Event::ForConsumer { .. } => unreachable!("ForConsumer rendered above"),
        }
        Ok(dict)
    }
}

/// A cloneable handle onto the shared event backlog. Cloning shares the same
/// underlying queue (it is an `Arc`), so the core's tasks and the Python-facing
/// peer hold the same queue.
#[derive(Clone, Default)]
pub(crate) struct EventQueue {
    inner: Arc<Mutex<VecDeque<Event>>>,
}

impl EventQueue {
    /// Append an event. Cheap, GIL-free, callable from any thread/task.
    pub(crate) fn push(&self, event: Event) {
        // A poisoned lock here only means another thread panicked mid-push;
        // recovering the guard keeps event delivery alive rather than cascading
        // the panic across the FFI boundary.
        let mut queue = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        queue.push_back(event);
    }

    /// Move the whole backlog out in FIFO order, leaving the queue empty. This
    /// is the GIL-free drain primitive [`drain_to_py`](Self::drain_to_py) builds
    /// on; unit tests use it to assert ordering and drain semantics without the
    /// interpreter.
    pub(crate) fn take_all(&self) -> Vec<Event> {
        let mut queue = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        queue.drain(..).collect()
    }

    /// Drain every queued event into a fresh Python list of dicts. The lock is
    /// held only long enough to move the backlog out; the Python objects are
    /// built afterwards so no task is blocked while we touch the interpreter.
    pub(crate) fn drain_to_py(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let drained = self.take_all();
        let list = PyList::empty_bound(py);
        for event in drained {
            list.append(event.to_pydict(py)?)?;
        }
        Ok(list.unbind())
    }
}

/// Emit a single terminal `on_state: "closed"` the first time it is called for
/// a given `closed` flag, and nothing on any later call. Returns `true` exactly
/// once — on that first call — so the caller can run one-shot teardown (drop
/// channels/tracks/the peer connection) under the same guard. Both peers' close
/// paths funnel through here, so "closed" is observed exactly once however many
/// times Python calls `close()`.
pub(crate) fn emit_closed_once(closed: &AtomicBool, events: &EventQueue) -> bool {
    if closed.swap(true, Ordering::SeqCst) {
        return false;
    }
    events.push(Event::State("closed".to_string()));
    true
}

#[cfg(test)]
mod tests {
    //! Peer-free, GIL-free unit tests for the event queue: FIFO drain semantics,
    //! the `"kind"` discriminator schema, and close-once idempotency. The
    //! Event -> `dict` *field* rendering (`to_pydict`) needs the interpreter and
    //! is exercised by the integration relay, which reads each field by name;
    //! here we pin the discriminators (the schema's type tags) and the queueing.

    use super::*;

    #[test]
    fn take_all_preserves_fifo_order_and_drains() {
        let queue = EventQueue::default();
        queue.push(Event::State("new".to_string()));
        queue.push(Event::DataChannel {
            label: "telemetry".to_string(),
            kind_hint: "reliable".to_string(),
        });
        queue.push(Event::State("connected".to_string()));

        let drained = queue.take_all();
        assert_eq!(
            drained,
            vec![
                Event::State("new".to_string()),
                Event::DataChannel {
                    label: "telemetry".to_string(),
                    kind_hint: "reliable".to_string(),
                },
                Event::State("connected".to_string()),
            ]
        );
        // A second drain sees an empty queue: the first drain emptied it.
        assert!(queue.take_all().is_empty());
    }

    #[test]
    fn kind_discriminators_match_the_documented_schema() {
        assert_eq!(Event::State("new".to_string()).kind(), "on_state");
        assert_eq!(
            Event::TrackAdded {
                track_id: "cam".to_string(),
                mid: "v0".to_string(),
            }
            .kind(),
            "on_track_added"
        );
        assert_eq!(
            Event::TrackRemoved {
                mid: "v0".to_string(),
            }
            .kind(),
            "on_track_removed"
        );
        assert_eq!(
            Event::DataChannel {
                label: "telemetry".to_string(),
                kind_hint: "reliable".to_string(),
            }
            .kind(),
            "on_data_channel"
        );
        assert_eq!(
            Event::Message {
                label: "telemetry".to_string(),
                data: vec![1, 2, 3],
            }
            .kind(),
            "on_message"
        );
        assert_eq!(
            Event::Frame {
                track_id: "cam0".to_string(),
                mid: "v0".to_string(),
                data: vec![0, 0, 0],
                width: 640,
                height: 480,
            }
            .kind(),
            "on_frame"
        );
        assert_eq!(
            Event::Manifest {
                json: "{}".to_string(),
            }
            .kind(),
            "on_manifest"
        );
        assert_eq!(
            Event::LocalDescription {
                sdp_type: "offer".to_string(),
                sdp: "v=0".to_string(),
            }
            .kind(),
            "on_local_description"
        );
        assert_eq!(
            Event::LocalCandidate {
                candidate: "candidate:1 ...".to_string(),
                mid: Some("0".to_string()),
            }
            .kind(),
            "on_local_candidate"
        );
        assert_eq!(
            Event::Error {
                consumer_id: None,
                location: "encode".to_string(),
                detail: "ffmpeg died".to_string(),
            }
            .kind(),
            "on_error"
        );
    }

    #[test]
    fn for_consumer_wraps_an_error_keeping_its_kind() {
        // A per-consumer error can also ride the ForConsumer wrapper (the
        // broadcaster tags reconnect/negotiate errors this way); the inner kind is
        // preserved either way.
        let wrapped = Event::ForConsumer {
            consumer_id: "c1".to_string(),
            inner: Box::new(Event::Error {
                consumer_id: None,
                location: "connection".to_string(),
                detail: "reconnect-needed".to_string(),
            }),
        };
        assert_eq!(wrapped.kind(), "on_error");
    }

    #[test]
    fn for_consumer_delegates_kind_to_the_wrapped_event() {
        // A per-consumer wrapper keeps the inner event's "kind" discriminator;
        // the consumer_id is added as a sibling key (rendered under the GIL, so
        // exercised by the multi-consumer relay, not here).
        let wrapped = Event::ForConsumer {
            consumer_id: "c1".to_string(),
            inner: Box::new(Event::LocalDescription {
                sdp_type: "offer".to_string(),
                sdp: "v=0".to_string(),
            }),
        };
        assert_eq!(wrapped.kind(), "on_local_description");
        let wrapped_state = Event::ForConsumer {
            consumer_id: "c2".to_string(),
            inner: Box::new(Event::State("connected".to_string())),
        };
        assert_eq!(wrapped_state.kind(), "on_state");
    }

    #[test]
    fn close_emits_on_state_closed_exactly_once() {
        let closed = AtomicBool::new(false);
        let events = EventQueue::default();

        // The first close does the work; every later close is a no-op. This is
        // the exact guard both Producer::close and Consumer::close run.
        assert!(emit_closed_once(&closed, &events));
        assert!(!emit_closed_once(&closed, &events));
        assert!(!emit_closed_once(&closed, &events));

        let drained = events.take_all();
        assert_eq!(drained, vec![Event::State("closed".to_string())]);
    }
}
