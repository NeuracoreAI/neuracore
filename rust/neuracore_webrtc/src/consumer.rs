//! The consumer peer: answer-only. It never offers and never opens channels; it
//! receives the producer's offer, lets libdatachannel auto-answer it, and
//! surfaces remote data channels, their messages, the control-channel manifest,
//! and connection state on its drainable event queue.
//!
//! ## Video tracks are observed via the manifest, not a track callback
//!
//! datachannel-rs (libdatachannel) exposes **no** incoming-track callback on the
//! peer-connection handler, so the consumer cannot learn of a producer-added video
//! track from the SDP renegotiation directly. Instead the producer republishes the
//! control-channel manifest atomically on every track add/remove, and the consumer
//! derives `on_track_added` / `on_track_removed` by diffing each manifest against
//! the previously-known video-track set. The manifest is the canonical
//! stream-identity channel (see the locked design), so this is the authoritative
//! signal — the SDP renegotiation still happens underneath and brings the media
//! m-line up; PR4's receive/decode path keys off the same mid.

use std::collections::HashMap;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use crate::media::RestartPolicy;

use datachannel::{
    ConnectionState, DataChannelHandler, DataChannelInfo, IceCandidate, IceState,
    PeerConnectionHandler, RtcDataChannel, RtcPeerConnection, SdpType, SessionDescription,
};
use datachannel_sys as sys;
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use serde_json::Value;

use crate::events::{emit_closed_once, Event, EventQueue};
use crate::media::{H264Decoder, RtpDepacketizer};
use crate::runtime::ensure_started;
use crate::transport::{
    connection_state_str, debug_trace, lock, loopback_config, map_err, parse_session, raw_pc_id,
    reliability_kind_hint, sdp_type_str, CONTROL_LABEL,
};

/// The bitrate (bits/sec) the consumer requests via `rtcRequestBitrate`, which is
/// what makes its receiving session emit REMB toward the producer. On the
/// libdatachannel loopback this is a fixed echoed number (the library does no
/// bandwidth estimation); a real browser computes its own REMB. We request a high
/// ceiling so REMB never *itself* throttles — the producer's estimator owns the
/// adaptation. See `reports/SPIKE-pr5-media-chain.md` §3.
const REQUESTED_BITRATE_BPS: u32 = 8_000_000;

/// Decoded-frame dimensions. The synthetic source is fixed 640x480 rgb24; the
/// producer encodes that and the consumer decodes it back to the same shape. A
/// later PR can carry per-track dimensions in the manifest if sources vary.
const FRAME_WIDTH: u32 = 640;
const FRAME_HEIGHT: u32 = 480;

/// Shared mid -> track_id view, written by the control-channel manifest diff and
/// read by an inbound track's frame emitter so `on_frame` carries the app track id.
type MidToTrack = Arc<Mutex<HashMap<String, String>>>;

/// Inbound channels kept alive for the connection's lifetime. The `Box` is
/// mandatory and cannot be elided: libdatachannel stores a raw pointer to each
/// `RtcDataChannel`'s heap location (`rtcSetUserPointer`), so the value must not
/// move — clippy's `vec_box` suggestion to unbox is wrong here.
#[allow(clippy::vec_box)]
type IncomingChannels = Vec<Box<RtcDataChannel<ConsumerChannelHandler>>>;

/// Per-data-channel handler for the consumer's inbound channels. Surfaces
/// application messages as `on_message`, and control-channel payloads as
/// `on_manifest` plus the derived `on_track_added` / `on_track_removed`.
pub(crate) struct ConsumerChannelHandler {
    events: EventQueue,
    label: String,
    is_control: bool,
    /// Video tracks known from the last manifest, mid -> track_id. Only the
    /// control channel's handler uses this; it persists for the connection's
    /// lifetime (the control channel opens once), and on_message is `&mut self`.
    known_tracks: HashMap<String, String>,
    /// The shared mid -> track_id view the inbound-track frame emitter reads, so
    /// each decoded `on_frame` carries the application track id. Updated on every
    /// manifest from the control channel's handler.
    mid_to_track: MidToTrack,
}

impl DataChannelHandler for ConsumerChannelHandler {
    fn on_message(&mut self, msg: &[u8]) {
        if self.is_control {
            // The manifest is JSON text; ignore anything non-UTF-8 on control.
            if let Ok(json) = std::str::from_utf8(msg) {
                self.diff_tracks(json);
                self.events.push(Event::Manifest {
                    json: json.to_string(),
                });
            }
        } else {
            self.events.push(Event::Message {
                label: self.label.clone(),
                data: msg.to_vec(),
            });
        }
    }
}

impl ConsumerChannelHandler {
    /// Diff this manifest's video-track entries against the previously-known set
    /// and emit `on_track_added` (new mids) and `on_track_removed` (vanished
    /// mids). Idempotent: a manifest republished for a non-track change (e.g. a
    /// data-channel add) produces no track events.
    fn diff_tracks(&mut self, json: &str) {
        let Ok(Value::Object(map)) = serde_json::from_str::<Value>(json) else {
            // Not a manifest object; leave the known set untouched.
            return;
        };
        let mut current: HashMap<String, String> = HashMap::new();
        for (key, value) in &map {
            if value.get("type").and_then(Value::as_str) == Some("video_track") {
                let track_id = value
                    .get("track_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                current.insert(key.clone(), track_id);
            }
        }

        for (mid, track_id) in &current {
            if !self.known_tracks.contains_key(mid) {
                self.events.push(Event::TrackAdded {
                    track_id: track_id.clone(),
                    mid: mid.clone(),
                });
            }
        }
        for mid in self.known_tracks.keys() {
            if !current.contains_key(mid) {
                self.events.push(Event::TrackRemoved { mid: mid.clone() });
            }
        }
        // Publish the latest mid -> track_id view for the inbound-track emitter.
        *lock(&self.mid_to_track) = current.clone();
        self.known_tracks = current;
    }
}

/// Peer-connection handler for the consumer: relays signaling/state callbacks
/// and adopts inbound data channels.
pub(crate) struct ConsumerHandler {
    events: EventQueue,
    /// Inbound channels are kept alive here; dropping a channel deletes it in
    /// libdatachannel, which would stop delivering its messages.
    incoming: Arc<Mutex<IncomingChannels>>,
    /// Shared mid -> track_id view, handed to each channel handler so the control
    /// channel's manifest diff can publish it for the inbound-track emitter.
    mid_to_track: MidToTrack,
}

impl PeerConnectionHandler for ConsumerHandler {
    type DCH = ConsumerChannelHandler;

    fn data_channel_handler(&mut self, info: DataChannelInfo) -> Self::DCH {
        ConsumerChannelHandler {
            events: self.events.clone(),
            label: info.label.clone(),
            is_control: info.label == CONTROL_LABEL,
            known_tracks: HashMap::new(),
            mid_to_track: self.mid_to_track.clone(),
        }
    }

    fn on_description(&mut self, sess_desc: SessionDescription) {
        // The consumer is answer-only, so this is the auto-generated answer.
        self.events.push(Event::LocalDescription {
            sdp_type: sdp_type_str(&sess_desc.sdp_type).to_string(),
            sdp: sess_desc.sdp.to_string(),
        });
    }

    fn on_candidate(&mut self, cand: IceCandidate) {
        self.events.push(Event::LocalCandidate {
            candidate: cand.candidate,
            mid: Some(cand.mid),
        });
    }

    fn on_connection_state_change(&mut self, state: ConnectionState) {
        crate::transport::debug_trace("C", connection_state_str(&state));
        // The constructor emits the initial "new"; skip the duplicate.
        if state == ConnectionState::New {
            return;
        }
        self.events
            .push(Event::State(connection_state_str(&state).to_string()));
    }

    fn on_ice_state_change(&mut self, state: IceState) {
        crate::transport::debug_trace("C", &format!("ice:{state:?}"));
    }

    fn on_data_channel(&mut self, data_channel: Box<RtcDataChannel<Self::DCH>>) {
        let label = data_channel.label();
        // The control channel is the manifest transport, not an application
        // stream; do not surface it as a data channel.
        if label != CONTROL_LABEL {
            let kind_hint = reliability_kind_hint(&data_channel.reliability());
            self.events.push(Event::DataChannel { label, kind_hint });
        }
        lock(&self.incoming).push(data_channel);
    }
}

// ---------------------------------------------------------------------------
// Inbound media: receive, depacketize, decode, emit on_frame
// ---------------------------------------------------------------------------
//
// datachannel-rs 0.16's `PeerConnectionHandler` has no `on_track`, so the
// consumer cannot learn of a producer-added media track through the safe API.
// We register libdatachannel's C track callback directly via the sys layer
// (`rtcSetTrackCallback`) to adopt each inbound track by its raw id, set a
// message callback on it to receive its RTP, depacketize FU-A back into NAL
// units, feed them to a per-track ffmpeg decoder, and surface each decoded
// picture as an `on_frame` event.
//
// Both C callbacks are plain `extern "C"` functions with no closure environment,
// so they route through process-global registries keyed by the libdatachannel
// integer ids: `MEDIA` maps a peer-connection id to its `ConsumerMedia`, and
// `TRACK_PC` maps an inbound track id back to its peer-connection id. We never
// touch the peer connection's user pointer (datachannel-rs owns it for its own
// callbacks), so the registries are how the callbacks find their context.

/// Process-global: peer-connection id -> its consumer media context.
static MEDIA: Lazy<Mutex<HashMap<i32, Arc<ConsumerMedia>>>> = Lazy::new(Default::default);
/// Process-global: inbound track id -> the peer-connection id that owns it.
static TRACK_PC: Lazy<Mutex<HashMap<i32, i32>>> = Lazy::new(Default::default);

/// The number of consumer media contexts in `MEDIA`. A diagnostics accessor for
/// the soak test's registry-baseline check (no leaked entries after churn).
pub(crate) fn media_registry_len() -> usize {
    lock(&MEDIA).len()
}

/// The number of inbound-track->pc mappings in `TRACK_PC`. A diagnostics accessor
/// for the soak test's registry-baseline check.
pub(crate) fn track_pc_registry_len() -> usize {
    lock(&TRACK_PC).len()
}

/// One inbound track's receive pipeline: a stateful RTP depacketizer feeding a
/// persistent ffmpeg decoder. Dropping it stops the decoder (kills its ffmpeg).
/// The decoder is behind a `Mutex` so a crashed one can be replaced in place
/// (bounded by `restart`) without losing the depacketizer's reassembly state.
struct TrackReceiver {
    depacketizer: Mutex<RtpDepacketizer>,
    decoder: Mutex<H264Decoder>,
    /// The track's mid, kept so a restart can rebuild the decoder's frame emitter.
    mid: String,
    /// Bounded crash-restart budget for this track's decoder.
    restart: Mutex<RestartPolicy>,
}

/// The consumer's media context, shared between the Python-facing `Consumer` and
/// the global registry the C callbacks consult.
struct ConsumerMedia {
    events: EventQueue,
    mid_to_track: MidToTrack,
    receivers: Mutex<HashMap<i32, TrackReceiver>>,
}

impl ConsumerMedia {
    /// Build a fresh decoder for `mid` whose decoded frames become `on_frame`
    /// events keyed to the app track id via the manifest. Used by both [`adopt`] and
    /// the crash-restart path, so the frame emitter is identical across a restart.
    fn build_decoder(&self, mid: &str) -> std::io::Result<H264Decoder> {
        let events = self.events.clone();
        let mid_to_track = self.mid_to_track.clone();
        let emit_mid = mid.to_string();
        let on_frame = move |data: Vec<u8>| {
            let track = lock(&mid_to_track)
                .get(&emit_mid)
                .cloned()
                .unwrap_or_default();
            events.push(Event::Frame {
                track_id: track,
                mid: emit_mid.clone(),
                data,
                width: FRAME_WIDTH,
                height: FRAME_HEIGHT,
            });
        };
        H264Decoder::new(FRAME_WIDTH, FRAME_HEIGHT, on_frame)
    }

    /// Adopt an inbound track: stand up its decoder and register its receive
    /// pipeline. Idempotent per track id.
    fn adopt(&self, track_id: i32, mid: String) {
        if lock(&self.receivers).contains_key(&track_id) {
            return;
        }
        match self.build_decoder(&mid) {
            Ok(decoder) => {
                lock(&self.receivers).insert(
                    track_id,
                    TrackReceiver {
                        depacketizer: Mutex::new(RtpDepacketizer::new()),
                        decoder: Mutex::new(decoder),
                        mid: mid.clone(),
                        restart: Mutex::new(RestartPolicy::default()),
                    },
                );
            }
            Err(err) => debug_trace("C", &format!("decoder spawn failed for mid {mid}: {err}")),
        }
    }

    /// Restart a crashed decoder in place, within the per-track bounded budget,
    /// surfacing an `on_error` with ffmpeg's stderr tail. The new decoder recovers
    /// on the producer's next periodic IDR. The depacketizer's reassembly state is
    /// preserved (it self-recovers on the next clean NAL).
    fn restart_decoder(&self, track_id: i32) {
        let receivers = lock(&self.receivers);
        let Some(receiver) = receivers.get(&track_id) else {
            return;
        };
        let mid = receiver.mid.clone();
        let detail = lock(&receiver.decoder).stderr_tail();
        if !lock(&receiver.restart).should_restart() {
            self.events.push(Event::error(
                "decode",
                format!("decoder for mid {mid:?} crashed and exceeded the restart budget"),
            ));
            return;
        }
        match self.build_decoder(&mid) {
            Ok(decoder) => {
                // Installing the new decoder drops the old one, killing its ffmpeg.
                *lock(&receiver.decoder) = decoder;
                self.events.push(Event::error(
                    "decode",
                    format!(
                        "decoder for mid {mid:?} crashed; restarting (ffmpeg: {})",
                        crate::producer::last_stderr_line(&detail)
                    ),
                ));
            }
            Err(err) => self.events.push(Event::error(
                "decode",
                format!("could not respawn decoder for mid {mid:?}: {err}"),
            )),
        }
    }

    /// Feed one inbound RTP packet through the track's depacketizer and into its
    /// decoder as Annex-B NAL units, restarting the decoder first if it has crashed.
    fn feed(&self, track_id: i32, packet: &[u8]) {
        // Detect a crashed decoder (ffmpeg exited) and restart it before feeding,
        // rather than silently dropping every inbound packet into a dead pipe.
        let dead = {
            let receivers = lock(&self.receivers);
            receivers
                .get(&track_id)
                .map(|r| !lock(&r.decoder).is_alive())
                .unwrap_or(false)
        };
        if dead {
            self.restart_decoder(track_id);
        }
        let receivers = lock(&self.receivers);
        let Some(receiver) = receivers.get(&track_id) else {
            return;
        };
        if !dead {
            // A healthy live decoder: clear the consecutive-crash budget.
            lock(&receiver.restart).reset();
        }
        let nals = lock(&receiver.depacketizer).depacketize(packet);
        let decoder = lock(&receiver.decoder);
        for nal in nals {
            decoder.feed_nal(&nal);
        }
    }
}

/// libdatachannel's track callback: an inbound media track was created. Look up
/// its peer connection's media context, adopt the track by mid, and wire its
/// message callback so its RTP starts flowing to the decoder.
unsafe extern "C" fn on_track_cb(pc: c_int, track: c_int, _ptr: *mut c_void) {
    let Some(media) = lock(&MEDIA).get(&pc).cloned() else {
        return;
    };
    let mid = track_mid(track);
    debug_trace("C", &format!("adopt track tr={track} mid={mid}"));
    media.adopt(track, mid);
    lock(&TRACK_PC).insert(track, pc);
    // Attach the built-in RTCP receiving session so the consumer answers the
    // producer's SR with RR (loss/jitter) and processes NACK/PLI, and request a
    // bitrate so the session emits REMB toward the producer. Inbound media still
    // arrives as whole RTP packets to the message callback (the C API has no
    // depacketizer), so the FU-A depacketizer is kept; the chain only adds the
    // RTCP feedback the producer's estimator reads.
    if sys::rtcChainRtcpReceivingSession(track) < 0 {
        debug_trace("C", &format!("receiving-session attach failed tr={track}"));
    }
    if sys::rtcRequestBitrate(track, REQUESTED_BITRATE_BPS) < 0 {
        debug_trace("C", &format!("requestBitrate failed tr={track}"));
    }
    sys::rtcSetMessageCallback(track, Some(on_message_cb));
}

/// libdatachannel's per-track message callback: one inbound RTP packet. Binary
/// messages carry a non-negative size; string messages (size < 0) are not media.
unsafe extern "C" fn on_message_cb(id: c_int, msg: *const c_char, size: c_int, _ptr: *mut c_void) {
    if size < 0 || msg.is_null() {
        return;
    }
    let packet = std::slice::from_raw_parts(msg as *const u8, size as usize);
    let Some(pc) = lock(&TRACK_PC).get(&id).copied() else {
        return;
    };
    let Some(media) = lock(&MEDIA).get(&pc).cloned() else {
        return;
    };
    media.feed(id, packet);
}

/// Read an inbound track's mid via the sys layer (the same two-call size-then-fill
/// pattern datachannel-rs uses internally).
fn track_mid(track: i32) -> String {
    unsafe {
        let size = sys::rtcGetTrackMid(track, std::ptr::null_mut(), 0);
        if size <= 0 {
            return String::new();
        }
        let mut buf = vec![0u8; size as usize];
        if sys::rtcGetTrackMid(track, buf.as_mut_ptr() as *mut c_char, size) < 0 {
            return String::new();
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    }
}

/// The consumer-side WebRTC peer exposed to Python. Answer-only by design.
#[pyclass]
pub struct Consumer {
    events: EventQueue,
    closed: Arc<AtomicBool>,
    pc: Mutex<Option<Box<RtcPeerConnection<ConsumerHandler>>>>,
    incoming: Arc<Mutex<IncomingChannels>>,
    /// The raw peer-connection id this consumer registered media callbacks under,
    /// used to deregister and drop its media context on close.
    pc_id: Option<i32>,
}

#[pymethods]
impl Consumer {
    /// Create an answer-only consumer. `connection_id` is an opaque label used
    /// only for logging/correlation.
    #[new]
    #[pyo3(signature = (connection_id=None))]
    fn new(connection_id: Option<String>) -> PyResult<Self> {
        let _ = connection_id;
        ensure_started();

        let events = EventQueue::default();
        let incoming = Arc::new(Mutex::new(Vec::new()));
        let mid_to_track: MidToTrack = Arc::new(Mutex::new(HashMap::new()));
        let handler = ConsumerHandler {
            events: events.clone(),
            incoming: incoming.clone(),
            mid_to_track: mid_to_track.clone(),
        };
        let pc = RtcPeerConnection::new(&loopback_config(), handler).map_err(map_err)?;
        events.push(Event::State("new".to_string()));

        // Register the media receive path under this peer connection's raw id:
        // publish its context and install libdatachannel's track callback so
        // inbound media tracks are adopted (the only way to receive — the safe
        // handler has no on_track).
        let pc_id = raw_pc_id(&pc);
        if let Some(id) = pc_id {
            let media = Arc::new(ConsumerMedia {
                events: events.clone(),
                mid_to_track,
                receivers: Mutex::new(HashMap::new()),
            });
            lock(&MEDIA).insert(id, media);
            // SAFETY: `id` is this live peer connection's id; the callback only
            // touches the process-global registries.
            unsafe { sys::rtcSetTrackCallback(id, Some(on_track_cb)) };
        } else {
            debug_trace("C", "could not recover pc id; inbound media disabled");
        }

        Ok(Self {
            events,
            closed: Arc::new(AtomicBool::new(false)),
            pc: Mutex::new(Some(pc)),
            incoming,
            pc_id,
        })
    }

    /// Apply the producer's SDP offer. libdatachannel auto-generates the answer,
    /// delivered as an `on_local_description` event.
    fn set_remote_offer(&self, sdp: &str) -> PyResult<()> {
        let sess = parse_session(sdp, SdpType::Offer)?;
        let mut guard = lock(&self.pc);
        let pc = guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("consumer is closed"))?;
        pc.set_remote_description(&sess).map_err(map_err)
    }

    /// Apply a remote ICE candidate trickled from the producer.
    #[pyo3(signature = (candidate, mid=None))]
    fn add_remote_candidate(&self, candidate: &str, mid: Option<String>) -> PyResult<()> {
        let cand = IceCandidate {
            candidate: candidate.to_string(),
            mid: mid.unwrap_or_default(),
        };
        let mut guard = lock(&self.pc);
        let pc = guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("consumer is closed"))?;
        pc.add_remote_candidate(&cand).map_err(map_err)
    }

    /// Drain and return all queued events as a list of dicts. See the
    /// [`events`](crate::events) module for the dict schema.
    fn drain_events(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        self.events.drain_to_py(py)
    }

    /// Close the consumer. Idempotent: the first call deregisters the media
    /// callbacks, drops its decoders (killing their ffmpeg), drops inbound
    /// channels and the peer connection, then emits a final `on_state: "closed"`.
    fn close(&self) -> PyResult<()> {
        if emit_closed_once(&self.closed, &self.events) {
            if let Some(id) = self.pc_id {
                // Stop new track callbacks, then drop this pc's media context
                // (its decoders/ffmpeg) and forget its tracks. A message callback
                // racing teardown finds no context and no-ops. The peer connection
                // itself is still alive here; it is dropped just below.
                unsafe { sys::rtcSetTrackCallback(id, None) };
                lock(&MEDIA).remove(&id);
                lock(&TRACK_PC).retain(|_, owner| *owner != id);
            }
            lock(&self.incoming).clear();
            *lock(&self.pc) = None;
        }
        Ok(())
    }
}
