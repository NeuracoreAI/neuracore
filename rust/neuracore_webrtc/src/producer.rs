//! The producer peer: the sole offerer in the negotiation model.
//!
//! The producer owns a libdatachannel [`RtcPeerConnection`] and is the only peer
//! that creates data channels and video tracks. Creating the first data channel
//! makes libdatachannel emit an offer (`on_local_description`); candidates trickle
//! out as `on_local_candidate`; the consumer's answer comes back through
//! [`Producer::set_remote_answer`]. Auto-negotiation is left at the libdatachannel
//! default for **data channels** — additional channels open over the existing SCTP
//! association without further SDP.
//!
//! ## Video tracks and the renegotiation queue (PR3)
//!
//! Unlike `createDataChannel`, libdatachannel's `addTrack` does **not** auto-offer
//! (verified in `impl/peerconnection.cpp`): a track add only changes the local
//! state, and the producer must call `set_local_description(Offer)` itself to
//! renegotiate. libdatachannel also silently drops a track that is added while a
//! prior offer is still in flight (no answer applied yet). The producer therefore
//! serialises all track mutations through a single-writer **negotiation queue**:
//!
//!  * `add_video_track` / `remove_video_track` allocate identity, enqueue a
//!    [`Mutation`], and signal the pump. They return immediately (the mid is
//!    allocated synchronously, so `add_video_track` can return it).
//!  * The pump task applies at most **one** mutation per offer/answer cycle: it
//!    pops a mutation, applies it (`add_track_ex` + `set_local_description(Offer)`,
//!    or drop-the-track + `set_local_description(Offer)`), and marks the cycle
//!    in-flight. It does nothing more until the consumer's answer is applied and
//!    `on_signaling_state_change(Stable)` fires, which clears the in-flight flag
//!    and re-signals the pump to advance.
//!
//! This serialises a burst of adds/removes into one mutation per offer and never
//! mutates tracks mid-cycle, so no track is silently dropped. The transport is
//! provisioned with `force_media_transport` up front (see
//! [`crate::transport::loopback_config`]) so the first track reuses the existing
//! BUNDLE/DTLS transport — a track add never triggers a second DTLS handshake.
//!
//! The pump never calls into libdatachannel from inside a libdatachannel callback:
//! the signaling-state callback only sets a flag and pings the pump channel; the
//! pump runs on the tokio runtime, outside any callback (the same discipline the
//! data-channel flusher uses for `on_open`). It also never holds the negotiation
//! lock while taking the peer-connection lock, so it cannot deadlock against the
//! callback (which takes the negotiation lock while libdatachannel holds the PC).
//!
//! ## Pre-open send gate
//!
//! libdatachannel rejects a send before the channel's SCTP stream is open, so
//! each outgoing channel buffers sends until its `on_open` fires. The handler
//! cannot safely send from inside its own `on_open` callback (that would alias
//! the channel libdatachannel is mid-callback on), so it instead signals a
//! per-producer flusher task over a channel; the task takes the channels lock —
//! outside any callback — and replays the buffer in order via the safe
//! `RtcDataChannel::send`.
//!
//! Still real from PR0: the Rust-owned tokio runtime, the bounded `submit_frame`
//! queue, and the drainable event queue. PR4 feeds encoded RTP into the per-track
//! [`RtcTrack`] handles registered here.

use std::collections::{HashMap, VecDeque};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use datachannel::{
    ConnectionState, DataChannelHandler, DataChannelInfo, IceCandidate, IceState,
    PeerConnectionHandler, RtcDataChannel, RtcPeerConnection, SdpType, SessionDescription,
    SignalingState,
};
use datachannel_sys as sys;
use once_cell::sync::Lazy;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use tokio::sync::mpsc;

use crate::congestion::{CongestionController, TrackControl, LADDER, TOP_STEP};
use crate::events::{emit_closed_once, Event, EventQueue};
use crate::media::{
    annexb_access_unit, is_open, open_flag, packetizer_cname, vcl_nal_count, DropPolicy,
    EncodeParams, H264Encoder, OpenFlag, RestartPolicy, MAX_FRAGMENT_SIZE, VIDEO_CLOCK_HZ,
};
use crate::runtime::{ensure_started, runtime};
use crate::transport::{
    chrome_sdp_enabled, connection_state_str, lock, loopback_config, map_err, munge_ssrc_cname,
    parse_session, raw_pc_id, sdp_type_str, ManifestState, CONTROL_LABEL,
};

/// The NACK responder's stored-packet history depth (packets it can retransmit on
/// a NACK). One second at the source rate is ample for loopback/LAN RTT.
const NACK_HISTORY: c_uint = 512;

// ---------------------------------------------------------------------------
// Producer-side RTCP feedback routing (sys-level chain callbacks)
// ---------------------------------------------------------------------------
//
// The producer's video track is created through `rtcAddTrackEx` (see
// `apply_mutation`), so unlike a datachannel-rs `RtcTrack` it has no safe handler
// — its chain callbacks are bare `extern "C"` functions. The REMB handler, the
// PLI handler, and the inbound-RTCP message callback (which carries RR) all route
// through this process-global registry keyed by libdatachannel's integer track
// id, exactly as the consumer's inbound-track callbacks do. We never touch the
// PC's user pointer (datachannel-rs owns it).

/// Process-global: producer track id -> its congestion controller. Shared with
/// the broadcaster (its per-consumer tracks register here too) — the registry is
/// keyed by libdatachannel's globally-unique integer track id, so one map serves
/// every peer connection in the process.
pub(crate) static PRODUCER_FB: Lazy<Mutex<HashMap<i32, Arc<CongestionController>>>> =
    Lazy::new(Default::default);

/// Cap on a track's capture-timestamp queue. A healthy 1:1 pipeline keeps it tiny
/// (one push per frame written, one pop per access unit emitted); the cap is a
/// belt-and-braces bound so a transiently-stalled or mid-crash encoder (writes
/// accepted, no access units emitted) can never grow it without bound. This is the
/// only place besides the ingress queue that could grow under fault, so it is
/// explicitly bounded. See [`push_capture_ts`].
pub(crate) const TS_QUEUE_CAP: usize = 256;

/// Push a capture timestamp onto a track's queue, dropping the oldest once the cap
/// is reached so the queue stays bounded under any fault.
pub(crate) fn push_capture_ts(queue: &Mutex<VecDeque<u32>>, ts: u32) {
    let mut q = lock(queue);
    if q.len() >= TS_QUEUE_CAP {
        q.pop_front();
    }
    q.push_back(ts);
}

/// Deregister a producer/broadcaster sys track from the process-global feedback
/// registry. Split from the libdatachannel teardown ([`teardown_sys_track`]) so the
/// registry-hygiene invariant is unit-testable without a live track.
pub(crate) fn deregister_feedback(raw_id: i32) {
    lock(&PRODUCER_FB).remove(&raw_id);
}

/// The number of live feedback controllers in `PRODUCER_FB`. A diagnostics
/// accessor the soak test reads to assert the registry returns to baseline after
/// churn (no leaked entries).
pub(crate) fn producer_fb_len() -> usize {
    lock(&PRODUCER_FB).len()
}

/// Fully tear down a producer/broadcaster sys track: deregister its feedback
/// controller, clear its RTCP message callback, and delete the track in
/// libdatachannel. Every remove, every close, AND every mid-setup error path
/// funnels through here, so no `PRODUCER_FB` entry or chain callback is ever
/// leaked — even when setup fails partway.
pub(crate) fn teardown_sys_track(raw_id: i32) {
    deregister_feedback(raw_id);
    // SAFETY: `raw_id` is a sys track this peer created and still owns; clearing
    // the callback before deletion stops any in-flight RTCP routing.
    unsafe {
        sys::rtcSetMessageCallback(raw_id, None);
        sys::rtcDeleteTrack(raw_id);
    }
}

/// REMB handler callback: the receiver-estimated max bitrate for this track.
pub(crate) unsafe extern "C" fn on_remb_cb(tr: c_int, bitrate: c_uint, _ptr: *mut c_void) {
    if let Some(ctrl) = lock(&PRODUCER_FB).get(&tr).cloned() {
        ctrl.on_remb(bitrate);
    }
}

/// PLI handler callback: the receiver wants a keyframe. Coalesced into a single
/// pending request the feed satisfies via an encoder restart.
pub(crate) unsafe extern "C" fn on_pli_cb(tr: c_int, _ptr: *mut c_void) {
    if let Some(ctrl) = lock(&PRODUCER_FB).get(&tr).cloned() {
        ctrl.on_pli();
    }
}

/// Inbound-RTCP message callback on the producer's (send-only) track. The chain's
/// SR reporter is outgoing-only, so inbound RR/REMB pass through to here; we
/// hand-parse the RR report blocks (the C API decodes none). Binary messages
/// carry a non-negative size.
pub(crate) unsafe extern "C" fn on_rtcp_cb(
    id: c_int,
    msg: *const c_char,
    size: c_int,
    _ptr: *mut c_void,
) {
    if size < 0 || msg.is_null() {
        return;
    }
    let buf = std::slice::from_raw_parts(msg as *const u8, size as usize);
    if let Some(ctrl) = lock(&PRODUCER_FB).get(&id).cloned() {
        ctrl.on_rtcp(buf);
    }
}

/// Persistent per-track ffmpeg encoders, keyed by `track_id`. Created lazily on
/// the first frame for a track (when its dimensions are known) and dropped on
/// close (which kills the subprocess).
type Encoders = Arc<Mutex<HashMap<String, Arc<H264Encoder>>>>;

/// How many pre-open frames to stash per track before the track's SRTP is up.
/// The feed thread drains the bounded ingress queue into this stash (so
/// `submit_frame` never drops while the connection is still coming up) and
/// flushes it in order — IDR first — the moment the track opens. Generous enough
/// to cover the sub-second open window at the source rate; the oldest frames are
/// the IDR we most want, so an over-long open sheds the newest.
pub(crate) const PREOPEN_STASH_FRAMES: usize = 32;

/// Capacity of the bounded queue behind [`Producer::submit_frame`]. Frames are
/// dropped (never block the caller) once this many are in flight; PR4 replaces
/// the stub drain with the real encoder feed and may revisit the depth.
pub(crate) const FRAME_QUEUE_CAPACITY: usize = 16;

/// The H.264 dynamic payload type advertised on every video track. Constrained
/// baseline / packetization-mode 1 is the first browser target; for the PR3
/// loopback (both peers libdatachannel) only consistency matters.
pub(crate) const VIDEO_PAYLOAD_TYPE: i32 = 96;

/// One raw frame handed off by `submit_frame`: its owned bytes, the routing
/// track id, and the dimensions the encoder needs to start ffmpeg. The pixel
/// format is fixed rgb24 (8-bit, 3-channel, the synthetic source shape).
pub(crate) struct Frame {
    pub(crate) track_id: String,
    pub(crate) data: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Capture time in 90 kHz RTP units (from the producer's epoch at
    /// `submit_frame`). Carried through to the encoder output so each access
    /// unit's RTP timestamp reflects when the frame was *captured*, not when the
    /// (bursty) encoder happened to emit it — a strict receiver (Chrome) needs a
    /// timestamp cadence that matches real arrival or it discards frames.
    pub(crate) capture_ts: u32,
}

/// The slice of a raw frame buffer `submit_frame` needs: whether it is
/// C-contiguous, its shape, and a copy of its bytes. Abstracted behind a trait so
/// the contiguity/shape gate and the copy can be unit-tested with a fake, without
/// constructing a live `PyBuffer` (which needs the interpreter). Production
/// implements it over `PyBuffer<u8>` (the buffer-protocol view of the caller's
/// numpy array); the bytes are copied because that array may be reused the
/// instant `submit_frame` returns.
pub(crate) trait FrameBytes {
    fn is_contiguous(&self) -> bool;
    fn shape(&self) -> Vec<usize>;
    fn to_owned_bytes(&self) -> Vec<u8>;
}

impl FrameBytes for PyBuffer<u8> {
    fn is_contiguous(&self) -> bool {
        self.is_c_contiguous()
    }

    fn shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }

    fn to_owned_bytes(&self) -> Vec<u8> {
        let len = self.item_count();
        // SAFETY: the GIL is held (this runs inside a `#[pymethods]` call), the
        // buffer is a validated `u8` C-contiguous buffer, the length comes from
        // `PyBuffer::item_count`, and we only read.
        unsafe { std::slice::from_raw_parts(self.buf_ptr() as *const u8, len).to_vec() }
    }
}

/// A validated frame: its bytes plus the dimensions extracted from the buffer's
/// shape. Channels are fixed at 3 (rgb24/bgr24).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FrameData {
    pub(crate) data: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// Why a frame buffer was rejected by [`read_frame`]. Kept as an enum rather
/// than a `PyErr` so the validation decision is testable without the
/// interpreter; `submit_frame` maps each to the `ValueError` callers see.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum FrameError {
    /// The buffer is not C-contiguous (the encoder needs a packed row layout).
    NotContiguous,
    /// The buffer is not an 8-bit HxWx3 image (the only format the encoder feeds).
    BadShape,
}

/// Validate and copy a frame buffer: reject a non-C-contiguous buffer (the same
/// methodology as the disk recording path's `log_frame`), require an `HxWx3`
/// 8-bit shape so the encoder knows the input format, then copy its bytes out.
/// Pure given the [`FrameBytes`] seam.
pub(crate) fn read_frame<B: FrameBytes>(buffer: &B) -> Result<FrameData, FrameError> {
    if !buffer.is_contiguous() {
        return Err(FrameError::NotContiguous);
    }
    let shape = buffer.shape();
    let [height, width, channels] = shape[..] else {
        return Err(FrameError::BadShape);
    };
    if channels != 3 || width == 0 || height == 0 {
        return Err(FrameError::BadShape);
    }
    Ok(FrameData {
        data: buffer.to_owned_bytes(),
        width: width as u32,
        height: height as u32,
    })
}

/// Shared map of the producer's outgoing data channels, keyed by label.
pub(crate) type Channels = Arc<Mutex<HashMap<String, OutgoingEntry>>>;

/// Shared map of the producer's outgoing video tracks, keyed by `track_id`.
type Tracks = Arc<Mutex<HashMap<String, TrackEntry>>>;

/// A producer-owned outgoing data channel plus its pre-open send buffer.
pub(crate) struct OutgoingEntry {
    /// The live channel; dropping it deletes the channel in libdatachannel.
    pub(crate) channel: Box<RtcDataChannel<ProducerChannelHandler>>,
    /// True once the channel's SCTP stream is open.
    pub(crate) open: bool,
    /// Bytes submitted before the channel opened, replayed in order on open.
    pub(crate) pending: VecDeque<Vec<u8>>,
}

impl OutgoingEntry {
    /// Send now if open, else buffer. Flushes any backlog first to preserve order.
    pub(crate) fn send(&mut self, bytes: Vec<u8>) {
        if self.open {
            self.flush();
            let _ = self.channel.send(&bytes);
        } else {
            self.pending.push_back(bytes);
        }
    }

    pub(crate) fn flush(&mut self) {
        while let Some(message) = self.pending.pop_front() {
            let _ = self.channel.send(&message);
        }
    }
}

/// A producer-owned outgoing video track. PR5 creates the track through the sys
/// layer (`rtcAddTrackEx`) so it can attach libdatachannel's built-in H.264
/// chain (packetizer + SR/NACK/PLI/REMB) to the raw id — datachannel-rs keeps a
/// safe `RtcTrack`'s id private. We therefore own the track's lifecycle manually:
/// `rtcDeleteTrack(raw_id)` on remove/close (see [`apply_mutation`] /
/// [`Producer::close`]), in place of PR4's drop-the-`Box<RtcTrack>` lifecycle.
struct TrackEntry {
    mid: String,
    /// libdatachannel's integer id for the track, from `rtcAddTrackEx`. The
    /// encoder feed sends NAL units on it via `rtcSendMessage`; remove/close
    /// deletes it via `rtcDeleteTrack`.
    raw_id: i32,
    /// Flipped true when the track's renegotiation completes (its answer applied).
    /// The encoder feed holds frames until this is set so the first sent access
    /// unit (always an IDR) is the first thing the consumer receives.
    open: OpenFlag,
    /// The adaptation effect surface: the estimator (driven by the RTCP callbacks)
    /// publishes a ladder rung here and the feed thread applies it (fps cap +
    /// encoder restart). Shared with the [`CongestionController`] in
    /// [`PRODUCER_FB`].
    control: Arc<TrackControl>,
    /// Capture timestamps (90 kHz) queued in submit order, one pushed per frame
    /// written to the encoder and popped per emitted access unit. Baseline H.264
    /// is in-order, so the Nth output frame carries the Nth input frame's capture
    /// time. Shared (`Arc`) so it survives encoder restarts.
    ts_queue: Arc<Mutex<VecDeque<u32>>>,
}

/// The SDP m-line mid for a video source: the caller-supplied `track_id`, used
/// verbatim. The caller (the Python provider for the broadcaster, the loopback
/// harness for the 1:1 producer) owns the mid and registers the same value for
/// browser identity (available_robots), so the offer's a=mid and that record
/// always agree. There is deliberately no per-connection "v{n}" counter: that
/// produced a wire mid the identity record never matched. The caller keeps mids
/// unique within a peer connection and must not reuse the data channel's m-line
/// mid ("0"); libdatachannel silently drops a video track whose mid collides with
/// the data channel (the Python provider prefixes "v" to stay clear of it).
pub(crate) fn track_mid(track_id: &str) -> String {
    track_id.to_string()
}

/// One queued track mutation. Removal is keyed by `track_id` (symmetric with the
/// add); the mid is recovered from the registry when the removal is applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Mutation {
    Add {
        track_id: String,
        mid: String,
        ssrc: u32,
    },
    Remove {
        track_id: String,
    },
}

impl Mutation {
    /// The track id this mutation targets (the queue's stable identity for both
    /// add and remove). Used by the negotiation-queue unit tests to assert the
    /// order mutations are applied in.
    #[cfg(test)]
    fn track_id(&self) -> &str {
        match self {
            Mutation::Add { track_id, .. } | Mutation::Remove { track_id } => track_id,
        }
    }
}

/// The single-writer negotiation queue state. `in_flight` is true between
/// applying a mutation (offer sent) and the consumer's answer being applied
/// (signaling back to Stable). At most one offer/answer cycle is ever in flight.
#[derive(Default)]
pub(crate) struct NegState {
    pub(crate) in_flight: bool,
    pub(crate) pending: VecDeque<Mutation>,
}

/// Per-data-channel handler for the producer's outgoing channels. On open it
/// signals the flusher task to drain the channel's pre-open send buffer. The
/// channels are send-only here, so inbound messages are ignored.
pub(crate) struct ProducerChannelHandler {
    label: String,
    flush_tx: mpsc::UnboundedSender<String>,
}

impl ProducerChannelHandler {
    /// Build a channel handler that signals `flush_tx` (with `label`) when its
    /// SCTP stream opens. Reused by the broadcaster's per-consumer control
    /// channels, which share the producer's pre-open send/flush discipline.
    pub(crate) fn new(label: String, flush_tx: mpsc::UnboundedSender<String>) -> Self {
        Self { label, flush_tx }
    }
}

impl DataChannelHandler for ProducerChannelHandler {
    fn on_open(&mut self) {
        // Do not send from inside on_open: that would re-enter the channel
        // libdatachannel is mid-callback on. Defer the flush to the task, which
        // sends from outside any callback.
        let _ = self.flush_tx.send(self.label.clone());
    }
}

/// Peer-connection handler for the producer: relays libdatachannel's signaling
/// and state callbacks onto the drainable event queue, and drives the
/// renegotiation queue forward when a cycle completes. All callbacks fire on
/// libdatachannel threads and only touch mutex-backed shared state (never the PC
/// itself — the pump does that off-callback).
pub(crate) struct ProducerHandler {
    events: EventQueue,
    /// Cleared to advance the queue when a negotiation cycle returns to Stable.
    neg: Arc<Mutex<NegState>>,
    /// The open flag of the track whose add-renegotiation is in flight, if any.
    /// Flipped true when that cycle returns to Stable — the precise point the
    /// consumer is provably ready to receive the track's RTP. See
    /// [`on_signaling_state_change`](Self::on_signaling_state_change).
    pending_open: Arc<Mutex<Option<OpenFlag>>>,
    /// Pinged after `in_flight` is cleared so the pump applies the next mutation.
    pump_tx: mpsc::UnboundedSender<()>,
    /// Set once a reconnect-needed error has been surfaced for the current outage,
    /// cleared on `Connected`, so a Disconnected->Failed sequence surfaces it once
    /// per outage rather than on every transition.
    reconnect_surfaced: Arc<AtomicBool>,
}

impl PeerConnectionHandler for ProducerHandler {
    type DCH = ProducerChannelHandler;

    fn data_channel_handler(&mut self, _info: DataChannelInfo) -> Self::DCH {
        // The consumer never opens channels back to the producer, so this
        // factory is effectively unused; hand back a detached handler.
        let (flush_tx, _flush_rx) = mpsc::unbounded_channel();
        ProducerChannelHandler {
            label: String::new(),
            flush_tx,
        }
    }

    fn on_description(&mut self, sess_desc: SessionDescription) {
        // Chrome-only SDP munge (gated, so the loopback path is byte-identical):
        // give libdatachannel's bare `a=ssrc:<n>` line a cname, which Chrome's
        // stricter parser requires before it will set up the receive track.
        let mut sdp = sess_desc.sdp.to_string();
        if sess_desc.sdp_type == SdpType::Offer && chrome_sdp_enabled() {
            sdp = munge_ssrc_cname(&sdp, crate::media::PACKETIZER_CNAME);
        }
        self.events.push(Event::LocalDescription {
            sdp_type: sdp_type_str(&sess_desc.sdp_type).to_string(),
            sdp,
        });
    }

    fn on_candidate(&mut self, cand: IceCandidate) {
        self.events.push(Event::LocalCandidate {
            candidate: cand.candidate,
            mid: Some(cand.mid),
        });
    }

    fn on_connection_state_change(&mut self, state: ConnectionState) {
        crate::transport::debug_trace("P", connection_state_str(&state));
        // The constructor emits the initial "new"; skip the duplicate so a
        // single new->connecting->connected sequence is observed.
        if state == ConnectionState::New {
            return;
        }
        if state == ConnectionState::Connected {
            // A fresh connection re-arms the one-shot reconnect surface.
            self.reconnect_surfaced.store(false, Ordering::SeqCst);
        }
        self.events
            .push(Event::State(connection_state_str(&state).to_string()));
        // Reconnect handling: this binding cannot restart ICE (libjuice
        // single-shot agent), so a Disconnected/Failed connection surfaces a clear
        // reconnect-needed error once per outage and the app removes + re-adds the
        // peer. If the binding ever gains ICE restart, the seam returns IceRestart.
        match crate::transport::reconnect_action(state, crate::transport::ICE_RESTART_SUPPORTED) {
            crate::transport::ReconnectAction::SurfaceReconnect => {
                if !self.reconnect_surfaced.swap(true, Ordering::SeqCst) {
                    self.events.push(Event::error(
                        "connection",
                        "reconnect-needed: connection failed and ICE restart is \
                         unsupported on this binding — remove and re-add the peer",
                    ));
                }
            }
            crate::transport::ReconnectAction::IceRestart
            | crate::transport::ReconnectAction::None => {}
        }
    }

    fn on_signaling_state_change(&mut self, state: SignalingState) {
        crate::transport::debug_trace("P", &format!("sig:{state:?}"));
        // A return to Stable means the in-flight offer's answer has been applied
        // (or there was nothing in flight). Clear the gate and wake the pump so
        // it applies the next queued mutation. Done off the PC: we only touch the
        // neg/pending locks here, never call back into libdatachannel from its
        // callback.
        if state == SignalingState::Stable {
            // If a track add just completed, the consumer has processed the media
            // offer and answered it, so it is now ready to receive. Open the
            // track for the encoder feed at exactly this point (not at the track's
            // own too-early SRTP on_open).
            if let Some(open) = lock(&self.pending_open).take() {
                open.store(true, Ordering::SeqCst);
                crate::transport::debug_trace("P", "gate-open (renegotiation complete)");
            }
            lock(&self.neg).in_flight = false;
            let _ = self.pump_tx.send(());
        }
    }

    fn on_ice_state_change(&mut self, state: IceState) {
        crate::transport::debug_trace("P", &format!("ice:{state:?}"));
    }
}

/// The producer-side WebRTC peer exposed to Python.
#[pyclass]
pub struct Producer {
    events: EventQueue,
    /// The bounded ingress sender, behind an `Option` so [`close`](Self::close) can
    /// drop it: dropping the last sender ends the feed thread's `blocking_recv`, so
    /// close stops the feed deterministically instead of leaking it until the
    /// `Producer` is garbage-collected.
    frame_tx: Mutex<Option<mpsc::Sender<Frame>>>,
    /// The encoder feed thread, joined on close so no thread outlives the producer.
    feed_handle: Mutex<Option<JoinHandle<()>>>,
    closed: Arc<AtomicBool>,
    /// Producer epoch: `submit_frame` stamps each frame's capture time as
    /// `elapsed_since(epoch)` in 90 kHz units.
    epoch: Instant,
    /// The libdatachannel peer connection. Shared (`Arc`) so the pump task can
    /// apply track mutations on it off-callback; dropped on `close`.
    pc: Arc<Mutex<Option<Box<RtcPeerConnection<ProducerHandler>>>>>,
    /// Outgoing data channels keyed by label (includes the control channel).
    channels: Channels,
    /// Outgoing video tracks keyed by track_id (the RTP send registry).
    tracks: Tracks,
    /// Persistent per-track ffmpeg encoders, created lazily by the feed thread.
    encoders: Encoders,
    /// The published stream manifest (data channels keyed by label, video tracks
    /// keyed by mid). Shared with the flusher and the pump's manifest republish.
    manifest: Arc<Mutex<ManifestState>>,
    /// The single-writer renegotiation queue, shared with the pump and handler.
    neg: Arc<Mutex<NegState>>,
    /// Allocates a unique RTP SSRC per track.
    ssrc_counter: AtomicU64,
    /// Channels signal this on open so the flusher drains their send buffers.
    flush_tx: mpsc::UnboundedSender<String>,
    /// Pinged on every track mutation (and from the handler on Stable) to wake
    /// the pump.
    pump_tx: mpsc::UnboundedSender<()>,
}

#[pymethods]
impl Producer {
    /// Create a producer. `connection_id` is an opaque label used only for
    /// logging/correlation; the Python signaling layer owns connection
    /// identity. `frame_queue_capacity` sizes the bounded `submit_frame` queue.
    #[new]
    #[pyo3(signature = (connection_id=None, frame_queue_capacity=FRAME_QUEUE_CAPACITY))]
    fn new(connection_id: Option<String>, frame_queue_capacity: usize) -> PyResult<Self> {
        let _ = connection_id;
        ensure_started();

        let (frame_tx, frame_rx) = mpsc::channel::<Frame>(frame_queue_capacity.max(1));

        let channels: Channels = Arc::new(Mutex::new(HashMap::new()));
        let tracks: Tracks = Arc::new(Mutex::new(HashMap::new()));
        let encoders: Encoders = Arc::new(Mutex::new(HashMap::new()));
        let events = EventQueue::default();
        // The encoder feed: drains the bounded ingress queue, lazily spins up one
        // ffmpeg encoder per track, gates on track-open, restarts a crashed encoder
        // (surfacing on_error), and hands each encoded access unit to the
        // packetize+send stage. Its handle is joined on close.
        let feed_handle = spawn_feed(frame_rx, encoders.clone(), tracks.clone(), events.clone());
        let manifest = Arc::new(Mutex::new(ManifestState::default()));
        let neg = Arc::new(Mutex::new(NegState::default()));
        let pending_open: Arc<Mutex<Option<OpenFlag>>> = Arc::new(Mutex::new(None));
        let (flush_tx, flush_rx) = mpsc::unbounded_channel::<String>();
        let (pump_tx, pump_rx) = mpsc::unbounded_channel::<()>();
        Self::spawn_flusher(channels.clone(), manifest.clone(), flush_rx);

        let handler = ProducerHandler {
            events: events.clone(),
            neg: neg.clone(),
            pending_open: pending_open.clone(),
            pump_tx: pump_tx.clone(),
            reconnect_surfaced: Arc::new(AtomicBool::new(false)),
        };
        let pc = Arc::new(Mutex::new(Some(
            RtcPeerConnection::new(&loopback_config(), handler).map_err(map_err)?,
        )));
        events.push(Event::State("new".to_string()));

        Self::spawn_pump(
            pc.clone(),
            tracks.clone(),
            encoders.clone(),
            manifest.clone(),
            channels.clone(),
            pending_open.clone(),
            neg.clone(),
            events.clone(),
            pump_rx,
        );

        Ok(Self {
            events,
            frame_tx: Mutex::new(Some(frame_tx)),
            feed_handle: Mutex::new(Some(feed_handle)),
            closed: Arc::new(AtomicBool::new(false)),
            epoch: Instant::now(),
            pc,
            channels,
            tracks,
            encoders,
            manifest,
            neg,
            ssrc_counter: AtomicU64::new(1),
            flush_tx,
            pump_tx,
        })
    }

    /// Add a video track and return its negotiated mid. The caller owns the mid:
    /// the supplied `track_id` is used verbatim as the SDP m-line mid, so the value
    /// the caller registers for identity (e.g. available_robots) and the offer's
    /// a=mid are the same. The caller must keep mids unique within the peer
    /// connection and must not reuse the data m-line mid ("0"); libdatachannel drops
    /// a track whose mid collides with the data channel. The mid is returned
    /// synchronously (so callers learn it immediately); the actual `add_track_ex`
    /// and the renegotiation it triggers are serialised through the queue, so a
    /// burst of adds never overlaps an in-flight offer.
    fn add_video_track(&self, track_id: &str) -> PyResult<String> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(PyValueError::new_err("producer is closed"));
        }
        let mid = track_mid(track_id);
        let ssrc = self.ssrc_counter.fetch_add(1, Ordering::SeqCst) as u32;
        lock(&self.neg).pending.push_back(Mutation::Add {
            track_id: track_id.to_string(),
            mid: mid.clone(),
            ssrc,
        });
        let _ = self.pump_tx.send(());
        Ok(mid)
    }

    /// Remove a previously-added video track by its `track_id`, routed through the
    /// queue. The consumer learns of the removal by mid via the republished
    /// manifest (libdatachannel surfaces no incoming-track callback).
    fn remove_video_track(&self, track_id: &str) -> PyResult<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(PyValueError::new_err("producer is closed"));
        }
        lock(&self.neg).pending.push_back(Mutation::Remove {
            track_id: track_id.to_string(),
        });
        let _ = self.pump_tx.send(());
        Ok(())
    }

    /// Open a reliable-ordered data channel. The first channel triggers the
    /// offer (negotiation-needed); later channels open over the existing SCTP
    /// association. `kind` is an opaque label hint recorded in the manifest.
    /// The reserved `"control"` label carries the manifest and is not itself
    /// listed in it.
    fn add_data_channel(&self, label: &str, kind: &str) -> PyResult<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(PyValueError::new_err("producer is closed"));
        }
        let handler = ProducerChannelHandler {
            label: label.to_string(),
            flush_tx: self.flush_tx.clone(),
        };
        let channel = {
            let mut guard = lock(&self.pc);
            let pc = guard
                .as_mut()
                .ok_or_else(|| PyValueError::new_err("producer is closed"))?;
            pc.create_data_channel(label, handler).map_err(map_err)?
        };
        lock(&self.channels).insert(
            label.to_string(),
            OutgoingEntry {
                channel,
                open: false,
                pending: VecDeque::new(),
            },
        );

        if label != CONTROL_LABEL {
            lock(&self.manifest).upsert_data_channel(label, kind);
        }
        // Republish the manifest on every change (atomic full-state message).
        // Buffers until the control channel opens, after which the flusher
        // replays it, so the consumer always converges on the latest set.
        republish(&self.channels, &self.manifest);
        Ok(())
    }

    /// Send a JSON payload (already-serialised text) over the named data
    /// channel. This is the single channel send path both `send_json` and the
    /// recording-context `log_*` bridge reach.
    fn send_json(&self, label: &str, payload: &str) -> PyResult<()> {
        let mut map = lock(&self.channels);
        let entry = map
            .get_mut(label)
            .ok_or_else(|| PyValueError::new_err(format!("no data channel labelled {label:?}")))?;
        entry.send(payload.as_bytes().to_vec());
        Ok(())
    }

    /// Enqueue one raw frame for `track_id` onto the bounded queue and return
    /// immediately. Never blocks: under overload the frame is dropped rather
    /// than back-pressuring the caller. The frame buffer must be C-contiguous
    /// (same zero-copy methodology as the disk recording path); the bytes are
    /// copied under the GIL because the caller's numpy array may be reused the
    /// instant this returns.
    #[pyo3(signature = (track_id, frame))]
    fn submit_frame(&self, track_id: &str, frame: PyBuffer<u8>) -> PyResult<()> {
        let FrameData {
            data,
            width,
            height,
        } = read_frame(&frame).map_err(|err| match err {
            FrameError::NotContiguous => PyValueError::new_err("frame buffer must be C-contiguous"),
            FrameError::BadShape => PyValueError::new_err("frame must be an 8-bit HxWx3 image"),
        })?;
        let capture_ts = (self.epoch.elapsed().as_secs_f64() * VIDEO_CLOCK_HZ as f64) as u32;
        let job = Frame {
            track_id: track_id.to_string(),
            data,
            width,
            height,
            capture_ts,
        };
        // Drop policy: admit only while the bounded ingress queue has room. With
        // room (the steady state at or below the encoder's throughput) nothing is
        // shed; once the encoder backs the queue to capacity, frames are dropped
        // here. Never blocks the caller. After close the sender is gone -> no-op.
        let guard = lock(&self.frame_tx);
        let Some(frame_tx) = guard.as_ref() else {
            return Ok(());
        };
        let capacity = frame_tx.max_capacity();
        let backlog = capacity - frame_tx.capacity();
        if !DropPolicy::new(capacity).admit(backlog) {
            return Ok(());
        }
        let _ = frame_tx.try_send(job);
        Ok(())
    }

    /// Apply the consumer's SDP answer to complete a negotiation round.
    fn set_remote_answer(&self, sdp: &str) -> PyResult<()> {
        let sess = parse_session(sdp, SdpType::Answer)?;
        let mut guard = lock(&self.pc);
        let pc = guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("producer is closed"))?;
        pc.set_remote_description(&sess).map_err(map_err)
    }

    /// Apply a remote ICE candidate trickled from the consumer.
    #[pyo3(signature = (candidate, mid=None))]
    fn add_remote_candidate(&self, candidate: &str, mid: Option<String>) -> PyResult<()> {
        let cand = IceCandidate {
            candidate: candidate.to_string(),
            mid: mid.unwrap_or_default(),
        };
        let mut guard = lock(&self.pc);
        let pc = guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("producer is closed"))?;
        pc.add_remote_candidate(&cand).map_err(map_err)
    }

    /// Drain and return all queued events as a list of dicts. See the
    /// [`events`](crate::events) module for the dict schema.
    fn drain_events(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        self.events.drain_to_py(py)
    }

    /// The congestion ladder rung a track is currently encoded at (0 = finest,
    /// higher = coarser), or `None` for an unknown track. The structured signal a
    /// constrained-link test reads to confirm the estimator is adapting.
    fn congestion_step(&self, track_id: &str) -> Option<usize> {
        lock(&self.tracks)
            .get(track_id)
            .map(|entry| entry.control.desired_step())
    }

    /// The coarsest ladder rung a track ever reached (a high-water mark), so a
    /// test sees adaptation fired even after the link recovered and the rung
    /// stepped back up. `None` for an unknown track.
    fn congestion_max_step(&self, track_id: &str) -> Option<usize> {
        lock(&self.tracks)
            .get(track_id)
            .map(|entry| entry.control.max_step())
    }

    /// Close the producer. Idempotent: the first call drops every data channel,
    /// every track, and the peer connection, then emits a final
    /// `on_state: "closed"`. Dropping the channels/tracks/PC releases the
    /// handler-held flush/pump senders, so the flusher and pump tasks end.
    fn close(&self) -> PyResult<()> {
        if emit_closed_once(&self.closed, &self.events) {
            // Stop the feed: dropping the ingress sender ends the feed thread's
            // blocking_recv. Join it so no thread (and, once it drops its encoder
            // clones, no ffmpeg subprocess) outlives the producer.
            *lock(&self.frame_tx) = None;
            if let Some(handle) = lock(&self.feed_handle).take() {
                let _ = handle.join();
            }
            // Drop encoders first so their ffmpeg subprocesses are killed before
            // the tracks they send on go away.
            lock(&self.encoders).clear();
            lock(&self.channels).clear();
            // Deregister each sys track's feedback controller and clear its
            // message callback before the PC drop frees the tracks, so no chain
            // callback races teardown with a stale registry entry.
            {
                let tracks = lock(&self.tracks);
                let mut fb = lock(&PRODUCER_FB);
                for entry in tracks.values() {
                    fb.remove(&entry.raw_id);
                    // SAFETY: raw_id is this producer's track, still alive until
                    // the PC is dropped just below.
                    unsafe { sys::rtcSetMessageCallback(entry.raw_id, None) };
                }
            }
            lock(&self.tracks).clear();
            *lock(&self.pc) = None;
        }
        Ok(())
    }
}

impl Producer {
    /// Spawn the task that drains a channel's pre-open send buffer once it opens.
    /// For the control channel it also re-sends the current manifest so a freshly
    /// connected consumer gets the up-to-date stream set.
    fn spawn_flusher(
        channels: Channels,
        manifest: Arc<Mutex<ManifestState>>,
        mut flush_rx: mpsc::UnboundedReceiver<String>,
    ) {
        runtime().spawn(async move {
            while let Some(label) = flush_rx.recv().await {
                {
                    let mut map = lock(&channels);
                    if let Some(entry) = map.get_mut(&label) {
                        entry.open = true;
                        entry.flush();
                    }
                }
                if label == CONTROL_LABEL {
                    let json = lock(&manifest).to_json();
                    let mut map = lock(&channels);
                    if let Some(entry) = map.get_mut(CONTROL_LABEL) {
                        entry.send(json.into_bytes());
                    }
                }
            }
        });
    }

    /// Spawn the single-writer negotiation pump. It is woken by `pump_tx` from
    /// both the Python track methods (a mutation was enqueued) and the handler
    /// (a cycle returned to Stable), and applies at most one mutation per cycle.
    #[allow(clippy::too_many_arguments)]
    fn spawn_pump(
        pc: Arc<Mutex<Option<Box<RtcPeerConnection<ProducerHandler>>>>>,
        tracks: Tracks,
        encoders: Encoders,
        manifest: Arc<Mutex<ManifestState>>,
        channels: Channels,
        pending_open: Arc<Mutex<Option<OpenFlag>>>,
        neg: Arc<Mutex<NegState>>,
        events: EventQueue,
        mut pump_rx: mpsc::UnboundedReceiver<()>,
    ) {
        runtime().spawn(async move {
            let negotiator = ProducerNegotiator {
                pc,
                tracks,
                encoders,
                manifest,
                channels,
                pending_open,
                events,
            };
            while pump_rx.recv().await.is_some() {
                pump_step(&neg, &negotiator);
            }
        });
    }
}

/// The track facts the encoder feed needs, cloned out of the [`TrackEntry`] so it
/// can build a send closure and read the adaptation state without holding the
/// tracks lock across an ffmpeg restart.
struct TrackMeta {
    raw_id: i32,
    open: OpenFlag,
    control: Arc<TrackControl>,
    ts_queue: Arc<Mutex<VecDeque<u32>>>,
}

/// The encoder feed's per-track state, local to the feed thread: which ladder
/// rung is currently encoded, the token-bucket pacer that enforces the rung's
/// input fps cap, and the pre-open stash.
#[derive(Default)]
struct FeedState {
    applied_step: Option<usize>,
    /// Token-bucket pacer: `allowance` frames are available to send, refilled at
    /// `fps_cap` per second. A simple "skip if too soon" gate would beat against
    /// the source's fixed frame grid (e.g. a 45 fps source through a 33 fps gate
    /// only lands on 22 ms boundaries, quantising to ~22.7 fps); the bucket hits
    /// the target rate regardless of the input cadence.
    allowance: f64,
    last_tick: Option<Instant>,
    stash: VecDeque<Frame>,
    /// Bounded crash-restart budget for this track's encoder. Reset whenever a
    /// healthy live encoder is observed, so only repeated back-to-back crashes
    /// exhaust it.
    restart: RestartPolicy,
}

impl FeedState {
    /// Stash a pre-open (or pre-encoder) frame, keeping the earliest
    /// [`PREOPEN_STASH_FRAMES`] so the IDR-first prefix survives and memory stays
    /// bounded if the track never opens.
    fn stash(&mut self, frame: Frame) {
        if self.stash.len() < PREOPEN_STASH_FRAMES {
            self.stash.push_back(frame);
        }
    }
}

/// Spawn the producer's encoder feed on a dedicated OS thread (off the tokio
/// pool, because it makes blocking ffmpeg-stdin writes). It drains the bounded
/// ingress queue, holds frames until the track's renegotiation completes, then
/// encodes them — restarting the per-track ffmpeg encoder whenever the congestion
/// estimator moves the ladder rung (a coarser bitrate/resolution) or a PLI is
/// pending, and capping the input fps to the rung's floor. The blocking write
/// propagates back-pressure to the bounded ingress queue, which is the single
/// place steady-state overload sheds frames (drop-on-full in `submit_frame`).
fn spawn_feed(
    mut frame_rx: mpsc::Receiver<Frame>,
    encoders: Encoders,
    tracks: Tracks,
    events: EventQueue,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("ncwebrtc-feed".into())
        .spawn(move || {
            let adapt_disabled = std::env::var_os("NCD_WEBRTC_DISABLE_ADAPT").is_some();
            let mut feeds: HashMap<String, FeedState> = HashMap::new();
            while let Some(frame) = frame_rx.blocking_recv() {
                let track_id = frame.track_id.clone();
                // The track must be registered and its renegotiation complete
                // before we encode/send; until then, stash and wait.
                let Some(meta) = track_meta(&tracks, &track_id) else {
                    feeds.entry(track_id).or_default().stash(frame);
                    continue;
                };
                if !is_open(&meta.open) {
                    feeds.entry(track_id).or_default().stash(frame);
                    continue;
                }
                let feed = feeds.entry(track_id.clone()).or_default();

                // (Re)build the encoder when the rung changes, a PLI is pending
                // (restart = clean keyframe), or none exists yet. `NCD_WEBRTC_
                // DISABLE_ADAPT` pins the finest rung so the constrained-link
                // gate can prove it fails *without* adaptation (the estimator
                // still observes, but the feed ignores it).
                let desired = if adapt_disabled {
                    TOP_STEP
                } else {
                    meta.control.desired_step()
                };
                let pli = meta.control.take_pli();

                // Crash detection: a crashed ffmpeg encoder (its stdout hit EOF)
                // must be restarted, not silently stalled. Surface an on_error with
                // ffmpeg's stderr tail, resync the capture-timestamp queue (the dead
                // encoder emitted none of its queued stamps), and rebuild — within a
                // bounded budget so a permanently-broken ffmpeg surfaces a terminal
                // error instead of hot-looping.
                let dead = lock(&encoders)
                    .get(&track_id)
                    .map(|e| !e.is_alive())
                    .unwrap_or(false);
                if dead {
                    let detail = lock(&encoders)
                        .get(&track_id)
                        .map(|e| e.stderr_tail())
                        .unwrap_or_default();
                    lock(&encoders).remove(&track_id);
                    lock(&meta.ts_queue).clear();
                    feed.applied_step = None;
                    if feed.restart.should_restart() {
                        events.push(Event::error(
                            "encode",
                            format!(
                                "encoder for {track_id:?} crashed; restarting (ffmpeg: {})",
                                last_stderr_line(&detail)
                            ),
                        ));
                    } else {
                        events.push(Event::error(
                            "encode",
                            format!(
                                "encoder for {track_id:?} crashed and exceeded the restart \
                                 budget; dropping frames (ffmpeg: {})",
                                last_stderr_line(&detail)
                            ),
                        ));
                        feed.stash(frame);
                        continue;
                    }
                } else if lock(&encoders).contains_key(&track_id) {
                    // A healthy live encoder: clear the consecutive-crash budget so
                    // only repeated back-to-back crashes ever exhaust it.
                    feed.restart.reset();
                }

                let missing = !lock(&encoders).contains_key(&track_id);
                if feed.applied_step != Some(desired) || pli || missing {
                    match make_encoder(frame.width, frame.height, desired, &meta, events.clone()) {
                        Some(encoder) => {
                            // Dropping the previous encoder kills its ffmpeg.
                            lock(&encoders).insert(track_id.clone(), encoder);
                            feed.applied_step = Some(desired);
                        }
                        None => {
                            // Spawn failed; surface once per budget so a missing
                            // ffmpeg does not emit an unbounded error stream.
                            if !dead && feed.restart.should_restart() {
                                events.push(Event::error(
                                    "encode",
                                    format!("could not spawn ffmpeg encoder for {track_id:?}"),
                                ));
                            }
                            feed.stash(frame);
                            continue;
                        }
                    }
                }

                let Some(encoder) = lock(&encoders).get(&track_id).cloned() else {
                    feed.stash(frame);
                    continue;
                };
                // Flush the pre-open stash IDR-first (unpaced: the startup burst).
                // Queue each frame's capture timestamp so the encoder callback
                // stamps the matching output access unit with it.
                for held in feed.stash.drain(..) {
                    push_capture_ts(&meta.ts_queue, held.capture_ts);
                    encoder.write_frame(&held.data);
                }
                // Input fps cap via a token bucket: refill `allowance` at the
                // rung's fps_cap, spend one token per frame, drop when empty. This
                // sheds toward the floor without beating against the source grid.
                let fps_cap = LADDER[desired].fps_cap.max(1) as f64;
                let now = Instant::now();
                if let Some(last) = feed.last_tick {
                    feed.allowance += now.duration_since(last).as_secs_f64() * fps_cap;
                } else {
                    feed.allowance = 1.0; // first frame after a (re)start may send
                }
                feed.last_tick = Some(now);
                // Cap the bucket at ~1s of frames so a stall does not let a burst
                // through afterwards.
                if feed.allowance > fps_cap {
                    feed.allowance = fps_cap;
                }
                if feed.allowance < 1.0 {
                    continue; // over the cap -> drop this frame
                }
                feed.allowance -= 1.0;
                push_capture_ts(&meta.ts_queue, frame.capture_ts);
                encoder.write_frame(&frame.data);
            }
        })
        .expect("spawn encoder feed thread")
}

/// The last non-empty line of an ffmpeg stderr tail, trimmed — the concise
/// diagnostic put in a crash `on_error`. Empty when stderr captured nothing.
pub(crate) fn last_stderr_line(tail: &str) -> String {
    tail.lines()
        .rev()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .to_string()
}

/// Clone the feed-relevant facts out of a track's registry entry.
fn track_meta(tracks: &Tracks, track_id: &str) -> Option<TrackMeta> {
    lock(tracks).get(track_id).map(|entry| TrackMeta {
        raw_id: entry.raw_id,
        open: entry.open.clone(),
        control: entry.control.clone(),
        ts_queue: entry.ts_queue.clone(),
    })
}

/// The encoder's frame rate at a ladder rung: the rung's fps cap, but never above
/// the synthetic source's nominal 45 fps, so libx264's CBR bit budget matches the
/// frames it actually receives.
fn rung_encoder_fps(step: usize) -> u32 {
    LADDER[step].fps_cap.min(45)
}

/// Build a fresh per-track ffmpeg encoder for `step`. Its per-access-unit callback
/// frames the encoded NAL units as Annex-B and sends them on the sys track via
/// `rtcSendMessage` — the attached built-in packetizer does the RTP. Each access
/// unit advances the track's 90 kHz RTP timestamp (pushed via
/// `rtcSetTrackRtpTimestamp`) so a frame's packets share a timestamp across
/// restarts. Returns `None` if ffmpeg could not be spawned.
fn make_encoder(
    width: u32,
    height: u32,
    step: usize,
    meta: &TrackMeta,
    events: EventQueue,
) -> Option<Arc<H264Encoder>> {
    let params = EncodeParams {
        fps: rung_encoder_fps(step),
        bitrate: LADDER[step].bitrate,
        scale: LADDER[step].scale,
    };
    let raw_id = meta.raw_id;
    let open = meta.open.clone();
    let ts_queue = meta.ts_queue.clone();
    let on_access_unit = move |access_unit: Vec<Vec<u8>>| {
        if !is_open(&open) {
            return;
        }
        // Invariant: the producer sends one access unit as exactly one RTP frame
        // under one capture timestamp, so the access unit must carry exactly one
        // VCL NAL. A slicing or NAL-aggregation change that emits more than one
        // would desync the capture-timestamp queue and fabricate per-slice
        // timestamps — precisely the Chrome-only, loopback-invisible defect in
        // reports/SPIKE-chrome-pframe.md. Fail loud and drop the frame rather than
        // send a malformed one. (x264 `threads=1` keeps this true; see
        // media::x264_params.)
        let vcl_count = vcl_nal_count(&access_unit);
        if vcl_count != 1 {
            eprintln!(
                "[ncwebrtc] INVARIANT VIOLATED: access unit has {vcl_count} VCL NAL(s), \
                 expected exactly 1 (one slice per frame). Dropping rather than \
                 fabricating RTP timestamps — see reports/SPIKE-chrome-pframe.md."
            );
            return;
        }
        let buf = annexb_access_unit(&access_unit);
        if buf.is_empty() {
            return;
        }
        // Stamp this access unit with the matching input frame's *capture* time.
        // The queue is pushed once per frame written to the encoder and popped
        // once per emitted access unit (in-order baseline, so the head is this
        // frame's), so a healthy 1:1 pipeline never underflows. An underflow means
        // more access units than input frames (a multi-slice encode the VCL guard
        // above did not catch) — fail loud and drop rather than fabricate a
        // backward-running timestamp (the exact regression the spike found). No
        // silent fallback: a capture-time RTP clock is what a strict receiver
        // (Chrome's jitter buffer) needs to assemble inter-keyframe frames.
        let Some(ts) = lock(&ts_queue).pop_front() else {
            eprintln!(
                "[ncwebrtc] INVARIANT VIOLATED: capture-timestamp queue underflow \
                 (more access units than input frames — multi-slice encode?). \
                 Dropping rather than fabricating a timestamp — see \
                 reports/SPIKE-chrome-pframe.md."
            );
            return;
        };
        // SAFETY: `raw_id` is a live sys track id this producer created and owns
        // until remove/close; both calls are libdatachannel-internally locked.
        let sent = unsafe {
            sys::rtcSetTrackRtpTimestamp(raw_id, ts);
            sys::rtcSendMessage(raw_id, buf.as_ptr() as *const c_char, buf.len() as c_int)
        };
        if sent < 0 {
            // The consumer's track went away under us (closed / SRTP torn down).
            // Stop sending (close the gate so this does not spam) and surface once;
            // the Failed connection state separately drives reconnect-needed.
            open.store(false, Ordering::SeqCst);
            events.push(Event::error(
                "send",
                "send on a closed track; suppressing further sends",
            ));
        }
    };
    match H264Encoder::new(width, height, params, on_access_unit) {
        Ok(encoder) => Some(Arc::new(encoder)),
        Err(err) => {
            crate::transport::debug_trace("P", &format!("encoder spawn failed: {err}"));
            None
        }
    }
}

/// Build and attach libdatachannel's built-in H.264 send chain to a sys track id:
/// the packetizer (FU-A / sequence / marker / SSRC) plus the SR reporter, NACK
/// responder, PLI handler and REMB handler. The packetizer init's `cname` **must**
/// be non-null or `rtcSetH264Packetizer` returns -1 (and the SR reporter then
/// fails) — [`packetizer_cname`] guarantees that. The cname is copied into the
/// config during the call, so it only needs to outlive `rtcSetH264Packetizer`.
pub(crate) fn attach_producer_chain(raw_id: i32, ssrc: u32) -> Result<(), String> {
    let cname = packetizer_cname();
    let init = sys::rtcPacketizerInit {
        ssrc,
        cname: cname.as_ptr(),
        payloadType: VIDEO_PAYLOAD_TYPE as u8,
        clockRate: VIDEO_CLOCK_HZ,
        sequenceNumber: 0,
        timestamp: 0,
        maxFragmentSize: MAX_FRAGMENT_SIZE,
        nalSeparator: sys::rtcNalUnitSeparator_RTC_NAL_SEPARATOR_LONG_START_SEQUENCE,
        obuPacketization: sys::rtcObuPacketization_RTC_OBU_PACKETIZED_OBU,
        playoutDelayId: 0,
        playoutDelayMin: 0,
        playoutDelayMax: 0,
    };
    // SAFETY: `raw_id` is a freshly created sys track id; `init.cname` is a
    // non-null C string that lives until the end of this function.
    unsafe {
        if sys::rtcSetH264Packetizer(raw_id, &init) < 0 {
            return Err("rtcSetH264Packetizer failed".into());
        }
        if sys::rtcChainRtcpSrReporter(raw_id) < 0 {
            return Err("rtcChainRtcpSrReporter failed".into());
        }
        if sys::rtcChainRtcpNackResponder(raw_id, NACK_HISTORY) < 0 {
            return Err("rtcChainRtcpNackResponder failed".into());
        }
        if sys::rtcChainPliHandler(raw_id, Some(on_pli_cb)) < 0 {
            return Err("rtcChainPliHandler failed".into());
        }
        if sys::rtcChainRembHandler(raw_id, Some(on_remb_cb)) < 0 {
            return Err("rtcChainRembHandler failed".into());
        }
    }
    drop(cname); // copied by rtcSetH264Packetizer; nothing else holds the pointer
    Ok(())
}

/// The side-effecting half of one negotiation cycle: apply a single track
/// mutation against the real transport (add/drop the track, drive the offer,
/// update the registry and manifest). Abstracted behind a trait so the queue's
/// *control logic* in [`pump_step`] — one cycle in flight, in-order draining,
/// error recovery — can be driven by a fake in unit tests without a live
/// `PeerConnection`. Production is [`ProducerNegotiator`].
pub(crate) trait NegotiationApply {
    fn apply(&self, mutation: Mutation) -> Result<(), String>;
}

/// Production negotiator: applies a mutation against the real peer connection
/// (the `pc`/`tracks`/`manifest`/`channels` the pump previously took directly).
struct ProducerNegotiator {
    pc: Arc<Mutex<Option<Box<RtcPeerConnection<ProducerHandler>>>>>,
    tracks: Tracks,
    encoders: Encoders,
    manifest: Arc<Mutex<ManifestState>>,
    channels: Channels,
    pending_open: Arc<Mutex<Option<OpenFlag>>>,
    events: EventQueue,
}

impl NegotiationApply for ProducerNegotiator {
    fn apply(&self, mutation: Mutation) -> Result<(), String> {
        let result = apply_mutation(
            &self.pc,
            &self.tracks,
            &self.encoders,
            &self.manifest,
            &self.channels,
            &self.pending_open,
            mutation,
        );
        // Surface a negotiation failure (chain-attach / SDP / add-track error) on
        // the event queue rather than only tracing it; the queue's control logic
        // still clears the in-flight gate and drains on, so one bad mutation never
        // wedges the pump.
        if let Err(err) = &result {
            self.events
                .push(Event::error("negotiate", format!("track mutation failed: {err}")));
        }
        result
    }
}

/// Republish the current manifest over the control channel, if it exists.
/// Buffers (via the pre-open gate) until the control channel is open.
pub(crate) fn republish(channels: &Channels, manifest: &Arc<Mutex<ManifestState>>) {
    let json = lock(manifest).to_json();
    let mut map = lock(channels);
    if let Some(entry) = map.get_mut(CONTROL_LABEL) {
        entry.send(json.into_bytes());
    }
}

/// Advance the negotiation queue. Starts at most one offer/answer cycle: if a
/// cycle is already in flight, returns immediately (the Stable callback will wake
/// the pump again). On an apply error the gate is cleared and the next mutation is
/// attempted, so one bad mutation never wedges the queue.
///
/// Generic over [`NegotiationApply`] so the control logic here is exercised by a
/// fake in unit tests; production passes [`ProducerNegotiator`].
///
/// Lock discipline: the negotiation lock is only ever held to pop a mutation and
/// is released before the negotiator runs (which takes the peer-connection lock).
/// The signaling callback takes the negotiation lock while libdatachannel holds
/// the PC, so taking them in the opposite order here would deadlock — hence the
/// release-before-apply.
pub(crate) fn pump_step(neg: &Arc<Mutex<NegState>>, negotiator: &impl NegotiationApply) {
    loop {
        let mutation = {
            let mut state = lock(neg);
            if state.in_flight {
                return;
            }
            match state.pending.pop_front() {
                Some(mutation) => {
                    state.in_flight = true;
                    mutation
                }
                None => return,
            }
        };

        match negotiator.apply(mutation) {
            Ok(()) => return, // cycle in flight; the Stable callback resumes us
            Err(err) => {
                crate::transport::debug_trace("P", &format!("mutation failed: {err}"));
                lock(neg).in_flight = false;
                // Try the next mutation rather than wedging on a bad one.
            }
        }
    }
}

/// Apply one track mutation: add or remove a media m-line and trigger the offer.
/// Never holds the tracks/manifest lock across the PC lock.
#[allow(clippy::too_many_arguments)]
fn apply_mutation(
    pc: &Arc<Mutex<Option<Box<RtcPeerConnection<ProducerHandler>>>>>,
    tracks: &Tracks,
    encoders: &Encoders,
    manifest: &Arc<Mutex<ManifestState>>,
    channels: &Channels,
    pending_open: &Arc<Mutex<Option<OpenFlag>>>,
    mutation: Mutation,
) -> Result<(), String> {
    match mutation {
        Mutation::Add {
            track_id,
            mid,
            ssrc,
        } => {
            let open = open_flag();
            let control = Arc::new(TrackControl::default());
            let ts_queue = Arc::new(Mutex::new(VecDeque::new()));
            // Create the track through the sys layer so we keep its raw id and can
            // attach the built-in chain to it (datachannel-rs `add_track_ex`
            // swallows the id). `rtcAddTrackEx` does NOT auto-offer, so we drive
            // the offer ourselves — same as PR3.
            let raw_id = {
                let mut guard = lock(pc);
                let pc = guard.as_mut().ok_or("producer is closed")?;
                let pc_id = raw_pc_id(pc).ok_or("cannot recover pc id for rtcAddTrackEx")?;
                let mid_c = CString::new(mid.clone()).map_err(|e| e.to_string())?;
                let track_c = CString::new(track_id.clone()).map_err(|e| e.to_string())?;
                let init = sys::rtcTrackInit {
                    direction: sys::rtcDirection_RTC_DIRECTION_SENDONLY,
                    codec: sys::rtcCodec_RTC_CODEC_H264,
                    payloadType: VIDEO_PAYLOAD_TYPE,
                    ssrc,
                    mid: mid_c.as_ptr(),
                    name: std::ptr::null(),
                    msid: std::ptr::null(),
                    trackId: track_c.as_ptr(),
                    profile: std::ptr::null(),
                };
                // SAFETY: `pc_id` is this live PC's id; the CString pointers live
                // until the end of this block.
                let raw_id = unsafe { sys::rtcAddTrackEx(pc_id, &init) };
                if raw_id < 0 {
                    return Err(format!("rtcAddTrackEx failed: {raw_id}"));
                }
                // From here the track exists in libdatachannel; any later failure
                // must tear it down (deregister the controller, clear the callback,
                // delete the track) so a mid-setup error leaves no PRODUCER_FB entry
                // or callback leaked. Run the fallible setup, then clean up on Err.
                let setup = (|| -> Result<(), String> {
                    attach_producer_chain(raw_id, ssrc)?;
                    // Register the feedback controller, then route the chain's RR
                    // (raw inbound RTCP) to it via the track's message callback.
                    lock(&PRODUCER_FB).insert(
                        raw_id,
                        Arc::new(CongestionController::new(ssrc, control.clone())),
                    );
                    // SAFETY: `raw_id` is the just-created track id.
                    unsafe { sys::rtcSetMessageCallback(raw_id, Some(on_rtcp_cb)) };
                    pc.set_local_description(SdpType::Offer)
                        .map_err(|e| e.to_string())?;
                    Ok(())
                })();
                if let Err(err) = setup {
                    teardown_sys_track(raw_id);
                    return Err(err);
                }
                raw_id
            };
            lock(tracks).insert(
                track_id.clone(),
                TrackEntry {
                    mid: mid.clone(),
                    raw_id,
                    open: open.clone(),
                    control,
                    ts_queue,
                },
            );
            // Arm the open flag to flip when this offer's answer is applied (the
            // Stable callback). That is the point the consumer is ready, so the
            // encoder feed may start sending without losing the first IDR.
            *lock(pending_open) = Some(open);
            lock(manifest).upsert_video_track(&mid, &track_id);
            republish(channels, manifest);
            Ok(())
        }
        Mutation::Remove { track_id } => {
            let entry = lock(tracks).remove(&track_id);
            let Some(entry) = entry else {
                // Unknown track id: nothing to negotiate, treat as a no-op so the
                // queue keeps draining.
                return Ok(());
            };
            let mid = entry.mid.clone();
            let raw_id = entry.raw_id;
            drop(entry);
            // Reap the encoder (kills its ffmpeg), then deregister the controller,
            // clear the callback, and delete the sys track via the shared teardown —
            // the same path the close and mid-setup-error paths use.
            lock(encoders).remove(&track_id);
            teardown_sys_track(raw_id);
            {
                let mut guard = lock(pc);
                let pc = guard.as_mut().ok_or("producer is closed")?;
                pc.set_local_description(SdpType::Offer)
                    .map_err(|e| e.to_string())?;
            }
            lock(manifest).remove_entry(&mid);
            republish(channels, manifest);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    //! Peer-free unit tests for the producer's frame ingress and the negotiation
    //! queue's control logic. The queue is driven through a fake
    //! [`NegotiationApply`] and frame ingress through a fake [`FrameBytes`], so
    //! neither needs a live `PeerConnection`, a socket, or the GIL.

    use super::*;
    use std::collections::HashSet;
    use tokio::sync::mpsc::error::TrySendError;

    // --- frame ingress: the contiguity gate ----------------------------------

    struct FakeFrame {
        contiguous: bool,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    impl FrameBytes for FakeFrame {
        fn is_contiguous(&self) -> bool {
            self.contiguous
        }
        fn shape(&self) -> Vec<usize> {
            self.shape.clone()
        }
        fn to_owned_bytes(&self) -> Vec<u8> {
            self.bytes.clone()
        }
    }

    #[test]
    fn read_frame_rejects_a_non_contiguous_buffer() {
        let frame = FakeFrame {
            contiguous: false,
            shape: vec![2, 2, 3],
            bytes: vec![0; 12],
        };
        // submit_frame maps this exact error onto the ValueError callers see.
        assert_eq!(read_frame(&frame), Err(FrameError::NotContiguous));
    }

    #[test]
    fn read_frame_rejects_a_non_hwc3_shape() {
        // Wrong dimensionality (a flat buffer) and wrong channel count both fail
        // the shape gate before any encoder sees them.
        let flat = FakeFrame {
            contiguous: true,
            shape: vec![12],
            bytes: vec![0; 12],
        };
        assert_eq!(read_frame(&flat), Err(FrameError::BadShape));
        let four_channel = FakeFrame {
            contiguous: true,
            shape: vec![2, 2, 4],
            bytes: vec![0; 16],
        };
        assert_eq!(read_frame(&four_channel), Err(FrameError::BadShape));
    }

    #[test]
    fn read_frame_extracts_dimensions_and_copies_a_contiguous_buffer() {
        let frame = FakeFrame {
            contiguous: true,
            shape: vec![480, 640, 3],
            bytes: vec![9, 8, 7],
        };
        assert_eq!(
            read_frame(&frame),
            Ok(FrameData {
                data: vec![9, 8, 7],
                width: 640,
                height: 480,
            })
        );
    }

    // --- frame ingress: the bounded queue ------------------------------------

    fn frame(tag: u8) -> Frame {
        Frame {
            track_id: "cam0".to_string(),
            data: vec![tag],
            width: 640,
            height: 480,
            capture_ts: tag as u32,
        }
    }

    #[test]
    fn frame_queue_capacity_is_sixteen() {
        assert_eq!(FRAME_QUEUE_CAPACITY, 16);
    }

    #[test]
    fn submit_drops_on_overflow_and_never_blocks() {
        // The producer's bounded channel, exactly as submit_frame builds it.
        // `try_send` is the non-blocking enqueue: it returns immediately whether
        // or not the queue is full.
        let (tx, mut rx) = mpsc::channel::<Frame>(FRAME_QUEUE_CAPACITY);
        for i in 0..FRAME_QUEUE_CAPACITY {
            assert!(tx.try_send(frame(i as u8)).is_ok(), "frame {i} should fit");
        }
        // Full queue: the next enqueue is dropped (Full), never blocked.
        match tx.try_send(frame(0xff)) {
            Err(TrySendError::Full(_)) => {}
            other => panic!("expected a Full drop on overflow, got {other:?}"),
        }
        // Draining one slot admits exactly one more (bounded depth N, FIFO).
        assert!(rx.try_recv().is_ok());
        assert!(tx.try_send(frame(0x01)).is_ok());
    }

    // --- negotiation queue: control logic behind a fake ----------------------

    /// Records every mutation it applies, in order. A track id listed in `fail`
    /// returns an error, to drive the error-recovery path.
    #[derive(Default)]
    struct FakeNegotiator {
        applied: Mutex<Vec<Mutation>>,
        fail: Mutex<HashSet<String>>,
    }

    impl FakeNegotiator {
        fn applied_ids(&self) -> Vec<String> {
            self.applied
                .lock()
                .unwrap()
                .iter()
                .map(|m| m.track_id().to_string())
                .collect()
        }
    }

    impl NegotiationApply for FakeNegotiator {
        fn apply(&self, mutation: Mutation) -> Result<(), String> {
            if self.fail.lock().unwrap().contains(mutation.track_id()) {
                return Err(format!("forced failure for {}", mutation.track_id()));
            }
            self.applied.lock().unwrap().push(mutation);
            Ok(())
        }
    }

    fn queue_with(mutations: Vec<Mutation>) -> Arc<Mutex<NegState>> {
        let neg = Arc::new(Mutex::new(NegState::default()));
        lock(&neg).pending.extend(mutations);
        neg
    }

    /// Models `on_signaling_state_change(Stable)`: the in-flight offer's answer
    /// has been applied, so clear the gate. (The real callback also pings the
    /// pump; here the test drives `pump_step` directly.)
    fn complete_cycle(neg: &Arc<Mutex<NegState>>) {
        lock(neg).in_flight = false;
    }

    fn add(track_id: &str) -> Mutation {
        Mutation::Add {
            track_id: track_id.to_string(),
            mid: format!("v_{track_id}"),
            ssrc: 1,
        }
    }
    fn remove(track_id: &str) -> Mutation {
        Mutation::Remove {
            track_id: track_id.to_string(),
        }
    }

    #[test]
    fn applies_at_most_one_mutation_per_cycle_and_holds_the_rest() {
        let neg = queue_with(vec![add("a"), add("b"), remove("c")]);
        let negotiator = FakeNegotiator::default();

        // One pump starts exactly one cycle; the rest stay queued (serialized,
        // not batched — the PR3 contract).
        pump_step(&neg, &negotiator);
        assert_eq!(negotiator.applied_ids(), vec!["a"]);
        assert!(lock(&neg).in_flight, "a cycle must be in flight");
        assert_eq!(lock(&neg).pending.len(), 2);
    }

    #[test]
    fn no_mutation_is_applied_while_a_cycle_is_in_flight() {
        let neg = queue_with(vec![add("a"), add("b")]);
        let negotiator = FakeNegotiator::default();

        pump_step(&neg, &negotiator); // applies "a"; in_flight = true
                                      // Further pumps are no-ops until the cycle completes.
        pump_step(&neg, &negotiator);
        pump_step(&neg, &negotiator);
        assert_eq!(negotiator.applied_ids(), vec!["a"]);
    }

    #[test]
    fn queued_mutations_apply_in_order_after_each_completion() {
        let neg = queue_with(vec![add("a"), add("b"), remove("c")]);
        let negotiator = FakeNegotiator::default();

        pump_step(&neg, &negotiator); // a
        complete_cycle(&neg);
        pump_step(&neg, &negotiator); // b
        complete_cycle(&neg);
        pump_step(&neg, &negotiator); // c

        assert_eq!(negotiator.applied_ids(), vec!["a", "b", "c"]);
    }

    #[test]
    fn the_queue_drains_to_a_converged_final_state() {
        let mutations = vec![add("a"), add("b"), remove("a"), add("c")];
        let neg = queue_with(mutations.clone());
        let negotiator = FakeNegotiator::default();

        for _ in 0..mutations.len() {
            pump_step(&neg, &negotiator);
            complete_cycle(&neg);
        }
        // Every mutation applied exactly once, in order; queue empty, not in flight.
        assert_eq!(*negotiator.applied.lock().unwrap(), mutations);
        assert!(lock(&neg).pending.is_empty());
        assert!(!lock(&neg).in_flight);
    }

    #[test]
    fn a_failed_mutation_clears_the_gate_and_the_queue_keeps_draining() {
        let neg = queue_with(vec![add("bad"), add("good")]);
        let negotiator = FakeNegotiator::default();
        negotiator.fail.lock().unwrap().insert("bad".to_string());

        // One pump: "bad" fails (gate cleared, the loop continues), then "good"
        // applies and leaves its cycle in flight — one bad mutation never wedges
        // the queue.
        pump_step(&neg, &negotiator);
        assert_eq!(negotiator.applied_ids(), vec!["good"]);
        assert!(lock(&neg).in_flight);
        assert!(lock(&neg).pending.is_empty());
    }

    // --- registry hygiene: deregistration leaves no leak ---------------------

    #[test]
    fn deregister_feedback_removes_only_the_keyed_track() {
        // The producer-feedback registry is process-global, so assert on the
        // specific ids this test owns (parallel-test-safe) rather than emptiness.
        // Two tracks are registered (an add); deregistering one — the remove or a
        // mid-setup-failure cleanup — leaves no entry for it and does not disturb
        // the other.
        let id_a = 0x07_70_00_01;
        let id_b = 0x07_70_00_02;
        let ctrl = || Arc::new(CongestionController::new(1, Arc::new(TrackControl::default())));
        lock(&PRODUCER_FB).insert(id_a, ctrl());
        lock(&PRODUCER_FB).insert(id_b, ctrl());

        deregister_feedback(id_a);
        assert!(!lock(&PRODUCER_FB).contains_key(&id_a), "added-then-removed leaves no entry");
        assert!(lock(&PRODUCER_FB).contains_key(&id_b), "the other track is untouched");

        // Simulate the mid-setup-failure cleanup path (register, then fail, then
        // deregister): the partially-set-up track also leaves no leaked entry.
        deregister_feedback(id_b);
        assert!(!lock(&PRODUCER_FB).contains_key(&id_b), "setup-failure cleanup leaves no entry");
    }

    // --- the capture-timestamp queue stays bounded ---------------------------

    #[test]
    fn push_capture_ts_caps_the_queue_and_keeps_the_newest() {
        let queue = Mutex::new(VecDeque::new());
        for ts in 0..(TS_QUEUE_CAP as u32 + 50) {
            push_capture_ts(&queue, ts);
        }
        let q = lock(&queue);
        assert_eq!(q.len(), TS_QUEUE_CAP, "the queue never grows past the cap");
        // The oldest stamps were shed; the most recent are retained.
        assert_eq!(*q.back().unwrap(), TS_QUEUE_CAP as u32 + 49);
        assert_eq!(*q.front().unwrap(), 50);
    }

    // --- crash diagnostic extraction -----------------------------------------

    #[test]
    fn last_stderr_line_picks_the_last_non_empty_line() {
        assert_eq!(
            last_stderr_line("Input #0\n[libx264] fatal: out of memory\n\n"),
            "[libx264] fatal: out of memory"
        );
        assert_eq!(last_stderr_line(""), "");
        assert_eq!(last_stderr_line("   \n  \n"), "");
    }

    // --- the caller-owned mid is used verbatim (no per-producer v-counter) -----

    #[test]
    fn track_mid_is_the_supplied_track_id_verbatim() {
        // The single source of truth for the video m-line mid: the caller-supplied
        // track_id, used verbatim. The old per-producer / per-link "v{n}" counter is
        // gone, so the value a caller registers for identity (available_robots) and
        // the offer's a=mid are the one and same producer-owned value. A regression
        // that reintroduced a counter would fail here.
        assert_eq!(track_mid("wrist_cam"), "wrist_cam");
        assert_eq!(track_mid("v0"), "v0");
        assert_eq!(track_mid("0"), "0");
    }
}
