//! The broadcaster: one producer serving **many** consumers from a single shared
//! encode per video source.
//!
//! The 1:1 [`Producer`](crate::producer::Producer) owns one peer connection and
//! one ffmpeg encode per track that doubles as that single consumer's track. The
//! [`Broadcaster`] keeps the same encode/packetize separation PR4 introduced but
//! fans **one** encode out to N consumers:
//!
//!  * Each consumer is its own answer-only peer connection (the broadcaster is the
//!    sole offerer to each), with its own PR3 negotiation queue, its own control
//!    channel + manifest (its own mids), and its own [`CongestionController`] per
//!    track registered in [`PRODUCER_FB`].
//!  * Exactly one ffmpeg encode runs per video source (`track_id`), never per
//!    consumer. Its NAL access units fan out to every consumer's track for that
//!    source via that track's own send handle (`rtcSendMessage` on the raw track
//!    id); each consumer's built-in chain packetizes independently with its own
//!    SSRC and sequence space. Re-encoding per consumer is exactly what this
//!    module exists to avoid.
//!  * The shared encoder rung is the **minimum estimate** across all consumers'
//!    controllers — the worst link caps everyone (a single-encoder tradeoff: no
//!    per-consumer quality). A lower estimate is a *coarser* ladder rung (higher
//!    index), so the min-fold over estimates is a `max` over ladder indices
//!    ([`fold_rung`]).
//!
//! ## Join / leave
//!
//! A join adds a consumer peer connection and, for each existing source, a
//! per-consumer track, then negotiates that consumer only. It does **not** force a
//! shared-encode keyframe — a forced IDR would blip every existing consumer — so a
//! joiner waits for the next periodic IDR (the encoder's `keyint`). A joiner's
//! early PLIs (it has no decodable frame until that IDR) are coalesced and
//! suppressed for a grace window ([`should_honor_pli`]) so one joining consumer
//! cannot restart the shared encode and disrupt the rest.
//!
//! A leave tears down only that consumer's peer connection, its tracks, and its
//! controllers (deregistered from [`PRODUCER_FB`]) without disturbing the others
//! or the shared encode. Removing the last consumer is graceful: the encode idles
//! (its encoder is reaped) and the min-fold over zero consumers does not panic.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use datachannel::{
    ConnectionState, DataChannelInfo, IceCandidate, IceState, PeerConnectionHandler,
    RtcPeerConnection, SdpType, SessionDescription, SignalingState,
};
use datachannel_sys as sys;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use tokio::sync::mpsc;

use crate::congestion::{CongestionController, TrackControl, LADDER, TOP_STEP};
use crate::events::{emit_closed_once, Event, EventQueue};
use crate::media::{
    annexb_access_unit, is_open, open_flag, vcl_nal_count, DropPolicy, EncodeParams, H264Encoder,
    OpenFlag, RestartPolicy, VIDEO_CLOCK_HZ,
};
use crate::producer::{
    attach_producer_chain, deregister_feedback, last_stderr_line, on_rtcp_cb, push_capture_ts,
    read_frame, republish, teardown_sys_track, Channels, Frame, FrameData, FrameError, Mutation,
    NegState, OutgoingEntry, ProducerChannelHandler, FRAME_QUEUE_CAPACITY, PREOPEN_STASH_FRAMES,
    PRODUCER_FB, VIDEO_PAYLOAD_TYPE,
};
use crate::runtime::{ensure_started, runtime};
use crate::transport::{
    chrome_sdp_enabled, connection_state_str, lock, loopback_config, map_err, munge_ssrc_cname,
    parse_session, raw_pc_id, sdp_type_str, ManifestState, CONTROL_LABEL,
};

/// How long after a consumer's track joins its PLIs are suppressed. A joiner has
/// no decodable frame until the next periodic IDR (`keyint`, ~1 s at the source
/// rate), so it will spew PLIs; honouring them would restart the shared encode and
/// blip every other consumer. The grace window is comfortably longer than one
/// keyint so the joiner becomes decodable on the natural periodic IDR before its
/// PLIs are ever honoured. After the window a PLI is real loss and is honoured.
const JOINER_PLI_GRACE_S: f64 = 2.0;

// ---------------------------------------------------------------------------
// Pure governance helpers (unit-tested without a peer)
// ---------------------------------------------------------------------------

/// The shared encoder rung from the per-consumer ladder steps: the **min-fold**
/// over consumer estimates, which (because a lower estimate is a coarser, higher
/// ladder index) is the `max` over ladder indices — the worst link caps everyone.
/// Zero consumers does not panic: it returns the finest rung [`TOP_STEP`].
pub(crate) fn fold_rung(steps: &[usize]) -> usize {
    steps.iter().copied().max().unwrap_or(TOP_STEP)
}

/// Whether a PLI from a track that joined `joined_elapsed_s` ago should be honoured
/// (i.e. trigger a shared-encode keyframe restart). A freshly joined track's PLIs
/// are suppressed until the grace window passes so a joiner cannot disrupt the
/// established consumers; an established track's PLI is real loss and is honoured.
pub(crate) fn should_honor_pli(joined_elapsed_s: f64, grace_s: f64) -> bool {
    joined_elapsed_s >= grace_s
}

/// The encoder's frame rate at a ladder rung: the rung's fps cap, but never above
/// the synthetic source's nominal 45 fps (so libx264's bit budget matches the
/// frames it actually receives). Mirrors the producer's `rung_encoder_fps`.
fn rung_encoder_fps(step: usize) -> u32 {
    LADDER[step].fps_cap.min(45)
}

// ---------------------------------------------------------------------------
// Per-source fan-out state
// ---------------------------------------------------------------------------

/// One consumer's track for a source — the unit the shared encode fans out to.
/// Created when that consumer's add-track renegotiation is applied; removed on
/// leave or source removal.
struct FanTrack {
    /// libdatachannel's integer id for this consumer's track. The shared encode
    /// sends NAL units on it via `rtcSendMessage`; teardown deletes it. The
    /// consumer's mid for the source lives in the link's `fan_refs` (the
    /// source-independent teardown record), not here.
    raw_id: i32,
    /// Flipped true when this consumer's add-track renegotiation completes, so the
    /// fan-out only sends to a track whose remote peer is ready to receive.
    open: OpenFlag,
    /// This track's congestion controller's published rung. The shared rung is the
    /// min-fold over all open fan tracks' rungs.
    control: Arc<TrackControl>,
    /// When this track joined, for the PLI suppression grace window.
    joined_at: Instant,
}

/// A shared video source: exactly one ffmpeg encode, fanned out to every
/// consumer's [`FanTrack`] for the source. The encode reads `ts_queue`,
/// `frames_encoded`, and `fanout` directly (cloned `Arc`s) so it never has to take
/// the broadcaster-wide `sources` lock on the per-frame hot path.
#[derive(Clone, Default)]
struct VideoSource {
    /// Capture timestamps (90 kHz) in submit order, one per frame written to the
    /// encoder, popped per emitted access unit (in-order baseline H.264). Shared so
    /// it survives encoder restarts.
    ts_queue: Arc<Mutex<VecDeque<u32>>>,
    /// Access units emitted by the single encoder for this source. The
    /// observability stat that proves fan-out: it counts one encode regardless of
    /// how many consumers receive it, so it does not scale with consumer count.
    frames_encoded: Arc<AtomicU64>,
    /// The fan-out set, keyed by consumer id. The feed governs the shared rung as a
    /// min-fold over the open entries' controllers; the encode sends each access
    /// unit to every open entry's track.
    fanout: Arc<Mutex<HashMap<String, FanTrack>>>,
}

/// Persistent per-source ffmpeg encoders, keyed by `track_id`. Exactly one per
/// source (never per consumer); created lazily when the first consumer for a
/// source is ready, reaped when the source idles (no open consumers).
type Encoders = Arc<Mutex<HashMap<String, Arc<H264Encoder>>>>;

/// Shared map of every video source, keyed by `track_id`.
type Sources = Arc<Mutex<HashMap<String, VideoSource>>>;

/// Shared map of every consumer link, keyed by `consumer_id`.
type Consumers = Arc<Mutex<HashMap<String, Arc<ConsumerLink>>>>;

// ---------------------------------------------------------------------------
// Per-consumer peer connection
// ---------------------------------------------------------------------------

/// One consumer's peer connection and its negotiation state. The broadcaster is
/// the sole offerer to it; it answers. Each link owns its own PR3 negotiation
/// queue (so a burst of track adds for this consumer never overlaps an in-flight
/// offer), its own control channel + manifest (its own mids), and the set of
/// source ids it currently carries (for leave teardown).
struct ConsumerLink {
    consumer_id: String,
    /// The libdatachannel peer connection; dropped (set to `None`) on leave/close.
    pc: Arc<Mutex<Option<Box<RtcPeerConnection<BroadcastConsumerHandler>>>>>,
    /// This consumer's single-writer negotiation queue.
    neg: Arc<Mutex<NegState>>,
    /// This consumer's outgoing data channels keyed by label (its control channel
    /// plus every json/joints channel fanned to it). Shared (`Arc`) with this
    /// consumer's flusher and negotiator; the broadcaster reaches into it to fan a
    /// `send_json` across consumers and to open a new channel on a live consumer.
    channels: Channels,
    /// This consumer's published manifest (its own mids + data-channel labels).
    /// Shared with the flusher and negotiator; a data-channel add upserts it here
    /// and republishes over this consumer's control channel.
    manifest: Arc<Mutex<ManifestState>>,
    /// Source ids this consumer currently carries a track for (the *desired* set,
    /// updated synchronously at enqueue time so a remove can decide whether to queue
    /// a teardown even before the matching add has been applied).
    sources: Mutex<HashSet<String>>,
    /// The *applied* per-source track identities, `track_id -> (raw_id, mid)`,
    /// populated when an add is applied and consulted on remove/leave/close. This is
    /// the source-independent record of what to deregister: removing a video source
    /// drops the shared `VideoSource` (and its fan-out), so teardown can no longer
    /// recover a track's raw id from there — it recovers it from here instead, which
    /// is what keeps `PRODUCER_FB` from leaking when a source is removed. Shared
    /// (`Arc`) with this consumer's negotiator, which writes it.
    fan_refs: Arc<Mutex<HashMap<String, (i32, String)>>>,
    /// Wakes this consumer's negotiation pump.
    pump_tx: mpsc::UnboundedSender<()>,
    /// Channels signal this on open so the flusher drains their send buffers.
    flush_tx: mpsc::UnboundedSender<String>,
}

/// Peer-connection handler for one consumer link. Identical in shape to the
/// producer's handler, except every surfaced event is wrapped with this
/// consumer's id ([`Event::ForConsumer`]) so the fan-out signaling layer routes
/// it to the right consumer.
struct BroadcastConsumerHandler {
    consumer_id: String,
    events: EventQueue,
    neg: Arc<Mutex<NegState>>,
    /// The open flag of the track whose add-renegotiation is in flight, flipped
    /// true when that cycle returns to Stable — the point the consumer is provably
    /// ready to receive the track's RTP.
    pending_open: Arc<Mutex<Option<OpenFlag>>>,
    pump_tx: mpsc::UnboundedSender<()>,
    /// One-shot reconnect-needed surface for this consumer's current outage;
    /// cleared on Connected.
    reconnect_surfaced: Arc<AtomicBool>,
}

impl BroadcastConsumerHandler {
    /// Push an event tagged with this consumer's id.
    fn emit(&self, inner: Event) {
        self.events.push(Event::ForConsumer {
            consumer_id: self.consumer_id.clone(),
            inner: Box::new(inner),
        });
    }
}

impl PeerConnectionHandler for BroadcastConsumerHandler {
    type DCH = ProducerChannelHandler;

    fn data_channel_handler(&mut self, _info: DataChannelInfo) -> Self::DCH {
        // A consumer never opens channels back to the broadcaster, so this factory
        // is effectively unused; hand back a detached handler (its flush_tx goes
        // nowhere, exactly like the producer's).
        let (flush_tx, _flush_rx) = mpsc::unbounded_channel();
        ProducerChannelHandler::new(String::new(), flush_tx)
    }

    fn on_description(&mut self, sess_desc: SessionDescription) {
        // Chrome-only SDP munge (gated, so the loopback path is byte-identical),
        // mirroring the producer: give the bare a=ssrc its required cname on offers.
        let mut sdp = sess_desc.sdp.to_string();
        if sess_desc.sdp_type == SdpType::Offer && chrome_sdp_enabled() {
            sdp = munge_ssrc_cname(&sdp, crate::media::PACKETIZER_CNAME);
        }
        self.emit(Event::LocalDescription {
            sdp_type: sdp_type_str(&sess_desc.sdp_type).to_string(),
            sdp,
        });
    }

    fn on_candidate(&mut self, cand: IceCandidate) {
        self.emit(Event::LocalCandidate {
            candidate: cand.candidate,
            mid: Some(cand.mid),
        });
    }

    fn on_connection_state_change(&mut self, state: ConnectionState) {
        crate::transport::debug_trace(&self.consumer_id, connection_state_str(&state));
        if state == ConnectionState::New {
            return; // the constructor already emitted the initial "new"
        }
        if state == ConnectionState::Connected {
            self.reconnect_surfaced.store(false, Ordering::SeqCst);
        }
        self.emit(Event::State(connection_state_str(&state).to_string()));
        // A failed consumer is torn down and re-added per consumer (this binding
        // cannot ICE-restart); surface reconnect-needed once per outage, tagged
        // with the consumer id, without disturbing the other consumers or the
        // shared encode.
        if let crate::transport::ReconnectAction::SurfaceReconnect = crate::transport::reconnect_action(
            state,
            crate::transport::ICE_RESTART_SUPPORTED,
        ) {
            if !self.reconnect_surfaced.swap(true, Ordering::SeqCst) {
                self.emit(Event::error_for(
                    &self.consumer_id,
                    "connection",
                    "reconnect-needed: consumer connection failed and ICE restart is \
                     unsupported — remove and re-add this consumer",
                ));
            }
        }
    }

    fn on_signaling_state_change(&mut self, state: SignalingState) {
        crate::transport::debug_trace(&self.consumer_id, &format!("sig:{state:?}"));
        // A return to Stable means the in-flight offer's answer has been applied
        // (the bootstrap data-channel cycle, or a track-add cycle). Open the track
        // whose add just completed, clear the gate, and wake the pump — all off the
        // PC (we only touch the neg/pending locks here).
        if state == SignalingState::Stable {
            if let Some(open) = lock(&self.pending_open).take() {
                open.store(true, Ordering::SeqCst);
            }
            lock(&self.neg).in_flight = false;
            let _ = self.pump_tx.send(());
        }
    }

    fn on_ice_state_change(&mut self, state: IceState) {
        crate::transport::debug_trace(&self.consumer_id, &format!("ice:{state:?}"));
    }
}

/// Production negotiator for one consumer: applies a single track mutation against
/// that consumer's peer connection (add/drop the track on its PC, drive its offer,
/// register/unregister the [`FanTrack`] in the shared source's fan-out). Drives the
/// reused [`pump_step`](crate::producer::pump_step) control logic.
struct ConsumerNegotiator {
    consumer_id: String,
    pc: Arc<Mutex<Option<Box<RtcPeerConnection<BroadcastConsumerHandler>>>>>,
    channels: Channels,
    manifest: Arc<Mutex<ManifestState>>,
    pending_open: Arc<Mutex<Option<OpenFlag>>>,
    sources: Sources,
    /// Shared with the [`ConsumerLink`]: `track_id -> (raw_id, mid)` for every
    /// applied track, so remove/leave/close can tear a track down without the
    /// shared `VideoSource` (which a source removal drops).
    fan_refs: Arc<Mutex<HashMap<String, (i32, String)>>>,
    events: EventQueue,
}

impl crate::producer::NegotiationApply for ConsumerNegotiator {
    fn apply(&self, mutation: Mutation) -> Result<(), String> {
        let result = match mutation {
            Mutation::Add {
                track_id,
                mid,
                ssrc,
            } => self.apply_add(track_id, mid, ssrc),
            Mutation::Remove { track_id } => self.apply_remove(track_id),
        };
        // Surface a per-consumer negotiation failure on the event queue (tagged
        // with the consumer id) rather than only tracing it; the pump still clears
        // the gate and drains on, so one bad mutation never wedges this consumer.
        if let Err(err) = &result {
            self.events.push(Event::error_for(
                &self.consumer_id,
                "negotiate",
                format!("track mutation failed: {err}"),
            ));
        }
        result
    }
}

impl ConsumerNegotiator {
    /// Add this consumer's track for `track_id`: create the sys track on its PC,
    /// attach the built-in H.264 chain, register its controller + fan-out entry,
    /// and drive the offer. Mirrors the producer's `apply_mutation` add arm.
    fn apply_add(&self, track_id: String, mid: String, ssrc: u32) -> Result<(), String> {
        let open = open_flag();
        let control = Arc::new(TrackControl::default());
        let raw_id = {
            let mut guard = lock(&self.pc);
            let pc = guard.as_mut().ok_or("consumer is closed")?;
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
            // SAFETY: `pc_id` is this live PC's id; the CString pointers live until
            // the end of this block.
            let raw_id = unsafe { sys::rtcAddTrackEx(pc_id, &init) };
            if raw_id < 0 {
                return Err(format!("rtcAddTrackEx failed: {raw_id}"));
            }
            // From here the track exists; any later failure tears it down so a
            // mid-setup error leaves no PRODUCER_FB entry or callback leaked.
            let setup = (|| -> Result<(), String> {
                attach_producer_chain(raw_id, ssrc)?;
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
        // Record the applied identity on the link (source-independent), so a later
        // remove/leave/close can deregister this track even after the shared source
        // is gone.
        lock(&self.fan_refs)
            .insert(track_id.clone(), (raw_id, mid.clone()));
        // Register the fan-out entry on the shared source so the encode reaches it.
        if let Some(source) = lock(&self.sources).get(&track_id).cloned() {
            lock(&source.fanout).insert(
                self.consumer_id.clone(),
                FanTrack {
                    raw_id,
                    open: open.clone(),
                    control,
                    joined_at: Instant::now(),
                },
            );
        }
        // Arm the open flag to flip when this offer's answer is applied (Stable).
        *lock(&self.pending_open) = Some(open);
        lock(&self.manifest).upsert_video_track(&mid, &track_id);
        republish(&self.channels, &self.manifest);
        Ok(())
    }

    /// Remove this consumer's track for `track_id`: tear it down on its PC,
    /// deregister its controller, drop the fan-out entry, and renegotiate. Mirrors
    /// the producer's `apply_mutation` remove arm. The track's raw id comes from the
    /// link's `fan_refs` (not the shared source), so this still tears the track down
    /// and deregisters `PRODUCER_FB` even when the source was already removed.
    fn apply_remove(&self, track_id: String) -> Result<(), String> {
        let Some((raw_id, mid)) = lock(&self.fan_refs).remove(&track_id) else {
            return Ok(()); // unknown track for this consumer: no-op, keep draining
        };
        // Drop the fan-out entry if the source still exists (the send path); if the
        // source was removed, its fan-out went with it.
        if let Some(source) = lock(&self.sources).get(&track_id).cloned() {
            lock(&source.fanout).remove(&self.consumer_id);
        }
        // Deregister the controller, clear the callback, and delete the sys track
        // via the shared teardown (the same path remove/close/error-cleanup use).
        teardown_sys_track(raw_id);
        {
            let mut guard = lock(&self.pc);
            let pc = guard.as_mut().ok_or("consumer is closed")?;
            pc.set_local_description(SdpType::Offer)
                .map_err(|e| e.to_string())?;
        }
        lock(&self.manifest).remove_entry(&mid);
        republish(&self.channels, &self.manifest);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The broadcaster pyclass
// ---------------------------------------------------------------------------

/// One producer fanning a single shared encode per source out to many consumers.
#[pyclass]
pub struct Broadcaster {
    events: EventQueue,
    /// The single shared ingress sender, behind an `Option` so [`close`](Self::close)
    /// can drop it and end the shared feed thread's `blocking_recv` deterministically.
    frame_tx: Mutex<Option<mpsc::Sender<Frame>>>,
    /// The shared encoder feed thread, joined on close so no thread outlives the
    /// broadcaster.
    feed_handle: Mutex<Option<JoinHandle<()>>>,
    closed: Arc<AtomicBool>,
    /// Broadcaster epoch: `submit_frame` stamps each frame's capture time as
    /// `elapsed_since(epoch)` in 90 kHz units.
    epoch: Instant,
    sources: Sources,
    consumers: Consumers,
    encoders: Encoders,
    /// Allocates a process-wide-unique RTP SSRC for every track (per-consumer
    /// tracks for the same source still get distinct SSRCs).
    ssrc_counter: AtomicU64,
    /// The data channels to open on every consumer, in insertion order, as
    /// `(label, kind)`. `add_data_channel` records each here (so a future consumer
    /// opens them at bootstrap) and opens it on every current consumer. The
    /// reserved `control` label is never recorded here. Mirrors the 1:1
    /// `Producer`'s data channels, fanned across consumers.
    data_channels: Mutex<Vec<(String, String)>>,
}

#[pymethods]
impl Broadcaster {
    /// Create a broadcaster with no consumers and no sources. `connection_id` is an
    /// opaque label for logging/correlation. `frame_queue_capacity` sizes the
    /// single shared bounded ingress queue feeding the per-source encode.
    #[new]
    #[pyo3(signature = (connection_id=None, frame_queue_capacity=FRAME_QUEUE_CAPACITY))]
    fn new(connection_id: Option<String>, frame_queue_capacity: usize) -> PyResult<Self> {
        let _ = connection_id;
        ensure_started();

        let (frame_tx, frame_rx) = mpsc::channel::<Frame>(frame_queue_capacity.max(1));
        let sources: Sources = Arc::new(Mutex::new(HashMap::new()));
        let encoders: Encoders = Arc::new(Mutex::new(HashMap::new()));
        let events = EventQueue::default();
        let feed_handle =
            spawn_broadcast_feed(frame_rx, encoders.clone(), sources.clone(), events.clone());

        events.push(Event::State("new".to_string()));

        Ok(Self {
            events,
            frame_tx: Mutex::new(Some(frame_tx)),
            feed_handle: Mutex::new(Some(feed_handle)),
            closed: Arc::new(AtomicBool::new(false)),
            epoch: Instant::now(),
            sources,
            consumers: Arc::new(Mutex::new(HashMap::new())),
            encoders,
            ssrc_counter: AtomicU64::new(1),
            data_channels: Mutex::new(Vec::new()),
        })
    }

    /// Add a consumer: stand up its answer-only peer connection (the broadcaster is
    /// the sole offerer), open its control channel (which triggers the offer), and
    /// add a per-consumer track for every existing source, then negotiate that
    /// consumer only. The track adds are queued behind the bootstrap offer so they
    /// do not race it.
    fn add_consumer(&self, consumer_id: &str) -> PyResult<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(PyValueError::new_err("broadcaster is closed"));
        }
        if lock(&self.consumers).contains_key(consumer_id) {
            return Err(PyValueError::new_err(format!(
                "consumer {consumer_id:?} already added"
            )));
        }

        let channels: Channels = Arc::new(Mutex::new(HashMap::new()));
        let manifest = Arc::new(Mutex::new(ManifestState::default()));
        let neg = Arc::new(Mutex::new(NegState::default()));
        let pending_open: Arc<Mutex<Option<OpenFlag>>> = Arc::new(Mutex::new(None));
        let fan_refs: Arc<Mutex<HashMap<String, (i32, String)>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let (flush_tx, flush_rx) = mpsc::unbounded_channel::<String>();
        let (pump_tx, pump_rx) = mpsc::unbounded_channel::<()>();

        spawn_flusher(channels.clone(), manifest.clone(), flush_rx);

        let handler = BroadcastConsumerHandler {
            consumer_id: consumer_id.to_string(),
            events: self.events.clone(),
            neg: neg.clone(),
            pending_open: pending_open.clone(),
            pump_tx: pump_tx.clone(),
            reconnect_surfaced: Arc::new(AtomicBool::new(false)),
        };
        let pc = Arc::new(Mutex::new(Some(
            RtcPeerConnection::new(&loopback_config(), handler).map_err(map_err)?,
        )));
        // The constructor's connection state is not surfaced (libdatachannel emits
        // New first); emit the per-consumer "new" so the signaling layer sees it.
        self.events.push(Event::ForConsumer {
            consumer_id: consumer_id.to_string(),
            inner: Box::new(Event::State("new".to_string())),
        });

        spawn_consumer_pump(
            consumer_id.to_string(),
            pc.clone(),
            channels.clone(),
            manifest.clone(),
            pending_open.clone(),
            self.sources.clone(),
            fan_refs.clone(),
            neg.clone(),
            self.events.clone(),
            pump_rx,
        );

        let link = Arc::new(ConsumerLink {
            consumer_id: consumer_id.to_string(),
            pc: pc.clone(),
            neg: neg.clone(),
            channels: channels.clone(),
            manifest: manifest.clone(),
            sources: Mutex::new(HashSet::new()),
            fan_refs,
            pump_tx: pump_tx.clone(),
            flush_tx,
        });

        // Gate any track adds behind the bootstrap offer: mark a cycle in flight so
        // the pump will not apply a track mutation (a second set_local_description)
        // until the control-channel offer's answer brings signaling back to Stable.
        lock(&neg).in_flight = true;
        // Open the control channel — this triggers the bootstrap offer.
        {
            let dch = ProducerChannelHandler::new(CONTROL_LABEL.to_string(), link.flush_tx.clone());
            let channel = {
                let mut guard = lock(&pc);
                let pc = guard
                    .as_mut()
                    .ok_or_else(|| PyValueError::new_err("consumer is closed"))?;
                pc.create_data_channel(CONTROL_LABEL, dch)
                    .map_err(map_err)?
            };
            lock(&channels).insert(
                CONTROL_LABEL.to_string(),
                OutgoingEntry {
                    channel,
                    open: false,
                    pending: VecDeque::new(),
                },
            );
        }

        // Open every registered json/joints data channel on this fresh consumer so
        // a late joiner gets the existing channels at bootstrap (DCEP over the same
        // SCTP the control channel brings up — no renegotiation; see PR2).
        let registered: Vec<(String, String)> = lock(&self.data_channels).clone();
        for (label, kind) in registered {
            if let Err(err) = open_consumer_channel(&link, &label, &kind) {
                self.events.push(Event::error_for(
                    consumer_id,
                    "data-channel",
                    format!("failed to open data channel {label:?} at bootstrap: {err}"),
                ));
            }
        }

        lock(&self.consumers).insert(consumer_id.to_string(), link.clone());

        // Add a track for each existing source (queued behind the bootstrap gate).
        let existing: Vec<String> = lock(&self.sources).keys().cloned().collect();
        for track_id in existing {
            self.enqueue_add(&link, &track_id);
        }
        let _ = pump_tx.send(());
        Ok(())
    }

    /// Remove a consumer: tear down only its peer connection, its tracks, and its
    /// controllers, without disturbing the other consumers or the shared encode.
    fn remove_consumer(&self, consumer_id: &str) -> PyResult<()> {
        let link = lock(&self.consumers).remove(consumer_id);
        let Some(link) = link else {
            return Ok(()); // unknown / already removed: idempotent
        };
        teardown_consumer(&self.sources, &link);
        Ok(())
    }

    /// Add a video source visible to all consumers (current and future).
    /// `submit_frame(track_id, ...)` encodes it once and fans it out. For each
    /// current consumer this queues a per-consumer track add (renegotiated per
    /// consumer); a future consumer picks the source up when it is added.
    fn add_video_track(&self, track_id: &str) -> PyResult<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(PyValueError::new_err("broadcaster is closed"));
        }
        lock(&self.sources).entry(track_id.to_string()).or_default();
        let links: Vec<Arc<ConsumerLink>> = lock(&self.consumers).values().cloned().collect();
        for link in links {
            self.enqueue_add(&link, track_id);
            let _ = link.pump_tx.send(());
        }
        Ok(())
    }

    /// Remove a video source from every consumer and stop its shared encode.
    fn remove_video_track(&self, track_id: &str) -> PyResult<()> {
        let links: Vec<Arc<ConsumerLink>> = lock(&self.consumers).values().cloned().collect();
        for link in links {
            if lock(&link.sources).remove(track_id) {
                lock(&link.neg).pending.push_back(Mutation::Remove {
                    track_id: track_id.to_string(),
                });
                let _ = link.pump_tx.send(());
            }
        }
        // Stop the shared encode immediately (kills its ffmpeg); drop the source.
        lock(&self.encoders).remove(track_id);
        lock(&self.sources).remove(track_id);
        Ok(())
    }

    /// Enqueue one raw frame for `track_id` onto the single shared bounded ingress
    /// queue and return immediately. Never blocks: under overload the frame is
    /// dropped. Same drop policy and contiguity contract as the producer.
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
        let guard = lock(&self.frame_tx);
        let Some(frame_tx) = guard.as_ref() else {
            return Ok(()); // closed: no-op
        };
        let capacity = frame_tx.max_capacity();
        let backlog = capacity - frame_tx.capacity();
        if !DropPolicy::new(capacity).admit(backlog) {
            return Ok(());
        }
        let _ = frame_tx.try_send(job);
        Ok(())
    }

    /// Open a reliable-ordered data channel with `label` on every consumer
    /// (current and future). For a current consumer the channel opens over its
    /// existing SCTP association (DCEP, no renegotiation — PR2); a future consumer
    /// opens it at bootstrap (see [`add_consumer`](Self::add_consumer)). `kind` is
    /// an opaque label hint recorded in each consumer's manifest. The reserved
    /// `control` label carries the per-consumer manifest and is never a json/joints
    /// label. Idempotent per label: re-adding a known label is a no-op so a live
    /// consumer never gets the same channel twice. Mirrors the 1:1
    /// [`Producer::add_data_channel`](crate::producer::Producer), fanned across
    /// consumers.
    fn add_data_channel(&self, label: &str, kind: &str) -> PyResult<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(PyValueError::new_err("broadcaster is closed"));
        }
        if label == CONTROL_LABEL {
            return Err(PyValueError::new_err(
                "'control' is reserved for the per-consumer manifest transport",
            ));
        }
        // Record once so a future consumer opens it at bootstrap; skip a known
        // label so a live consumer is never handed a duplicate channel.
        {
            let mut registry = lock(&self.data_channels);
            if registry.iter().any(|(existing, _)| existing == label) {
                return Ok(());
            }
            registry.push((label.to_string(), kind.to_string()));
        }
        // Open it on every current consumer over its existing association.
        let links: Vec<Arc<ConsumerLink>> = lock(&self.consumers).values().cloned().collect();
        for link in links {
            if let Err(err) = open_consumer_channel(&link, label, kind) {
                self.events.push(Event::error_for(
                    &link.consumer_id,
                    "data-channel",
                    format!("failed to open data channel {label:?}: {err}"),
                ));
            }
        }
        Ok(())
    }

    /// Send a JSON payload (already-serialised text) over the named data channel of
    /// **every** consumer that has it. A consumer still mid-bootstrap buffers it
    /// behind its pre-open gate and replays it in order on open, so no consumer
    /// loses or reorders a message. A label no consumer carries (e.g. one never
    /// added, or sent before any browser connected) simply reaches no one — the
    /// broadcast analogue of the 1:1 [`Producer::send_json`](crate::producer::Producer).
    fn send_json(&self, label: &str, payload: &str) -> PyResult<()> {
        let links: Vec<Arc<ConsumerLink>> = lock(&self.consumers).values().cloned().collect();
        for link in links {
            let mut map = lock(&link.channels);
            if let Some(entry) = map.get_mut(label) {
                entry.send(payload.as_bytes().to_vec());
            }
        }
        Ok(())
    }

    /// Apply a consumer's SDP answer, routed by `consumer_id`.
    fn set_remote_answer(&self, consumer_id: &str, sdp: &str) -> PyResult<()> {
        let link = lock(&self.consumers)
            .get(consumer_id)
            .cloned()
            .ok_or_else(|| PyValueError::new_err(format!("unknown consumer {consumer_id:?}")))?;
        let sess = parse_session(sdp, SdpType::Answer)?;
        let mut guard = lock(&link.pc);
        let pc = guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("consumer is closed"))?;
        pc.set_remote_description(&sess).map_err(map_err)
    }

    /// Apply a remote ICE candidate trickled from a consumer, routed by
    /// `consumer_id`.
    #[pyo3(signature = (consumer_id, candidate, mid=None))]
    fn add_remote_candidate(
        &self,
        consumer_id: &str,
        candidate: &str,
        mid: Option<String>,
    ) -> PyResult<()> {
        let link = lock(&self.consumers)
            .get(consumer_id)
            .cloned()
            .ok_or_else(|| PyValueError::new_err(format!("unknown consumer {consumer_id:?}")))?;
        let cand = IceCandidate {
            candidate: candidate.to_string(),
            mid: mid.unwrap_or_default(),
        };
        let mut guard = lock(&link.pc);
        let pc = guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("consumer is closed"))?;
        pc.add_remote_candidate(&cand).map_err(map_err)
    }

    /// Drain and return all queued events as a list of dicts. Per-consumer events
    /// carry a `"consumer_id"` key so the signaling layer routes them.
    fn drain_events(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        self.events.drain_to_py(py)
    }

    /// The number of live shared encoders. Exactly one per active source,
    /// **independent of the consumer count** — the observable that proves the
    /// encode is shared (fan-out) rather than re-encoded per consumer.
    fn encoder_count(&self) -> usize {
        lock(&self.encoders).len()
    }

    /// Access units the shared encoder has emitted for `track_id` (one encode,
    /// fanned out), or `None` for an unknown source. Does not scale with the
    /// consumer count.
    fn frames_encoded(&self, track_id: &str) -> Option<u64> {
        lock(&self.sources)
            .get(track_id)
            .map(|source| source.frames_encoded.load(Ordering::SeqCst))
    }

    /// The shared ladder rung a source is currently encoded at (the min-fold over
    /// its open consumers' controllers; finest = 0). `None` for an unknown source.
    fn congestion_step(&self, track_id: &str) -> Option<usize> {
        lock(&self.sources).get(track_id).map(|source| {
            let steps = open_rungs(&source);
            fold_rung(&steps)
        })
    }

    /// The number of consumers currently attached.
    fn consumer_count(&self) -> usize {
        lock(&self.consumers).len()
    }

    /// Close the broadcaster. Idempotent: tears down every consumer, kills every
    /// encoder, drops every source, then emits a final `on_state: "closed"`.
    fn close(&self) -> PyResult<()> {
        if emit_closed_once(&self.closed, &self.events) {
            // Stop the shared feed first (drop the ingress sender, join the thread)
            // so no thread or ffmpeg subprocess outlives the broadcaster.
            *lock(&self.frame_tx) = None;
            if let Some(handle) = lock(&self.feed_handle).take() {
                let _ = handle.join();
            }
            let links: Vec<Arc<ConsumerLink>> =
                lock(&self.consumers).drain().map(|(_, v)| v).collect();
            for link in links {
                teardown_consumer(&self.sources, &link);
            }
            lock(&self.encoders).clear();
            lock(&self.sources).clear();
        }
        Ok(())
    }
}

impl Broadcaster {
    /// Allocate this consumer's identity for a source and queue an add mutation on
    /// its negotiation queue (the caller pings the pump).
    fn enqueue_add(&self, link: &Arc<ConsumerLink>, track_id: &str) {
        // The Producer owns the mid: use the supplied track_id verbatim (see
        // crate::producer::track_mid) as the SDP m-line mid. The Producer registers
        // available_robots under this same value, so the offer's a=mid and the SSE
        // manifest agree and the browser's identityForMid succeeds. Every consumer
        // link receives the same source set in the same order, so one producer-owned
        // mid per source is globally consistent across consumers. The old per-link
        // "v{n}" counter produced "v0" on the wire while available_robots held "0",
        // which never matched, so the browser rendered no tile.
        let mid = crate::producer::track_mid(track_id);
        let ssrc = self.ssrc_counter.fetch_add(1, Ordering::SeqCst) as u32;
        lock(&link.sources).insert(track_id.to_string());
        lock(&link.neg).pending.push_back(Mutation::Add {
            track_id: track_id.to_string(),
            mid,
            ssrc,
        });
    }
}

/// Open one reliable-ordered data channel on a single consumer link: create it on
/// that consumer's peer connection, register it in the link's channel map (so a
/// later `send_json` finds it and the pre-open gate buffers until it opens), then
/// upsert its label into that consumer's manifest and republish over its control
/// channel. Mirrors the 1:1 [`Producer::add_data_channel`](crate::producer::Producer)
/// for one consumer; the broadcaster calls it for every consumer.
fn open_consumer_channel(link: &ConsumerLink, label: &str, kind: &str) -> Result<(), String> {
    let dch = ProducerChannelHandler::new(label.to_string(), link.flush_tx.clone());
    let channel = {
        let mut guard = lock(&link.pc);
        let pc = guard.as_mut().ok_or("consumer is closed")?;
        pc.create_data_channel(label, dch)
            .map_err(|e| e.to_string())?
    };
    lock(&link.channels).insert(
        label.to_string(),
        OutgoingEntry {
            channel,
            open: false,
            pending: VecDeque::new(),
        },
    );
    lock(&link.manifest).upsert_data_channel(label, kind);
    republish(&link.channels, &link.manifest);
    Ok(())
}

/// Tear down one consumer: drop its fan-out entries from every source (deregister
/// their controllers), then drop its peer connection (which closes its tracks and
/// fires its Closed callback). The shared encode and other consumers are untouched.
fn teardown_consumer(sources: &Sources, link: &ConsumerLink) {
    // Deregister every applied track from the link's own record, so cleanup is
    // independent of whether the shared source still exists (a removed source has
    // already dropped its fan-out). This is what keeps PRODUCER_FB from leaking on
    // leave/close after a source removal.
    let refs: Vec<(String, (i32, String))> = lock(&link.fan_refs).drain().collect();
    let source_map = lock(sources).clone();
    for (track_id, (raw_id, _mid)) in refs {
        if let Some(source) = source_map.get(&track_id) {
            lock(&source.fanout).remove(&link.consumer_id);
        }
        // Deregister the controller and clear the track's RTCP callback before the
        // PC drop frees the track, so no chain callback races teardown with a stale
        // registry entry and no entry/callback is leaked.
        deregister_feedback(raw_id);
        // SAFETY: `raw_id` is this consumer's track, alive until the PC drop just
        // below. The PC drop deletes the track itself.
        unsafe { sys::rtcSetMessageCallback(raw_id, None) };
    }
    // Drop this consumer's data channels before the PC (the producer-close order),
    // releasing their flush_tx handles so the flusher task ends, and tearing the
    // channels down cleanly with no per-consumer leak.
    lock(&link.channels).clear();
    // Dropping the PC frees its tracks/SRTP and fires the consumer's Closed; the
    // pump task ends once every pump_tx sender (handler + link) is gone.
    *lock(&link.pc) = None;
}

/// Spawn one consumer's negotiation pump (reusing the producer's `pump_step`).
#[allow(clippy::too_many_arguments)]
fn spawn_consumer_pump(
    consumer_id: String,
    pc: Arc<Mutex<Option<Box<RtcPeerConnection<BroadcastConsumerHandler>>>>>,
    channels: Channels,
    manifest: Arc<Mutex<ManifestState>>,
    pending_open: Arc<Mutex<Option<OpenFlag>>>,
    sources: Sources,
    fan_refs: Arc<Mutex<HashMap<String, (i32, String)>>>,
    neg: Arc<Mutex<NegState>>,
    events: EventQueue,
    mut pump_rx: mpsc::UnboundedReceiver<()>,
) {
    runtime().spawn(async move {
        let negotiator = ConsumerNegotiator {
            consumer_id,
            pc,
            channels,
            manifest,
            pending_open,
            sources,
            fan_refs,
            events,
        };
        while pump_rx.recv().await.is_some() {
            crate::producer::pump_step(&neg, &negotiator);
        }
    });
}

/// Spawn the per-consumer flusher: drains a channel's pre-open send buffer once it
/// opens, and re-sends the current manifest when the control channel opens.
/// Mirrors the producer's flusher.
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

// ---------------------------------------------------------------------------
// The shared encoder feed (one ffmpeg encode per source, fanned to N consumers)
// ---------------------------------------------------------------------------

/// The per-source feed state local to the feed thread.
#[derive(Default)]
struct FeedState {
    applied_step: Option<usize>,
    allowance: f64,
    last_tick: Option<Instant>,
    stash: VecDeque<Frame>,
    /// Bounded crash-restart budget for this source's shared encoder.
    restart: RestartPolicy,
}

impl FeedState {
    fn stash(&mut self, frame: Frame) {
        if self.stash.len() < PREOPEN_STASH_FRAMES {
            self.stash.push_back(frame);
        }
    }
}

/// The desired-rung steps of a source's currently-open fan tracks.
fn open_rungs(source: &VideoSource) -> Vec<usize> {
    lock(&source.fanout)
        .values()
        .filter(|ft| is_open(&ft.open))
        .map(|ft| ft.control.desired_step())
        .collect()
}

/// Coalesce a source's pending PLIs across its fan tracks, draining each track's
/// request (so a suppressed joiner's PLI does not accumulate) and honouring a
/// keyframe restart only for tracks past the join grace window.
fn coalesce_pli(source: &VideoSource) -> bool {
    let mut honor = false;
    for ft in lock(&source.fanout).values() {
        let requested = ft.control.take_pli();
        if requested && should_honor_pli(ft.joined_at.elapsed().as_secs_f64(), JOINER_PLI_GRACE_S) {
            honor = true;
        }
    }
    honor
}

/// Spawn the shared encoder feed on a dedicated OS thread (off the tokio pool: it
/// makes blocking ffmpeg-stdin writes). It drains the single shared ingress queue,
/// runs exactly one ffmpeg encode per source, governs that encode's rung by the
/// min-fold over the source's open consumers, and fans each encoded access unit out
/// to every open consumer's track. The blocking write propagates back-pressure to
/// the shared ingress queue (the single shed point in `submit_frame`).
fn spawn_broadcast_feed(
    mut frame_rx: mpsc::Receiver<Frame>,
    encoders: Encoders,
    sources: Sources,
    events: EventQueue,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("ncwebrtc-fanout-feed".into())
        .spawn(move || {
            let adapt_disabled = std::env::var_os("NCD_WEBRTC_DISABLE_ADAPT").is_some();
            let mut feeds: HashMap<String, FeedState> = HashMap::new();
            while let Some(frame) = frame_rx.blocking_recv() {
                let track_id = frame.track_id.clone();
                // The source must exist and have at least one open consumer before
                // we encode; otherwise stash (bounded) so a first consumer's IDR is
                // not lost while it is still negotiating.
                let Some(source) = lock(&sources).get(&track_id).cloned() else {
                    feeds.entry(track_id).or_default().stash(frame);
                    continue;
                };
                let steps = open_rungs(&source);
                if steps.is_empty() {
                    // No open consumers: idle the encode (reap its ffmpeg) and hold
                    // the frame. The min-fold over zero consumers does not panic.
                    lock(&encoders).remove(&track_id);
                    let feed = feeds.entry(track_id).or_default();
                    feed.applied_step = None;
                    feed.stash(frame);
                    continue;
                }
                let feed = feeds.entry(track_id.clone()).or_default();

                let desired = if adapt_disabled {
                    TOP_STEP
                } else {
                    fold_rung(&steps)
                };
                let pli = coalesce_pli(&source);

                // Crash detection for the shared encode: a crashed ffmpeg is
                // restarted (not stalled) so it does not silently stall every
                // consumer. A shared-encode crash belongs to no single consumer, so
                // the on_error carries no consumer_id. Resync the source's capture-
                // timestamp queue and rebuild within the bounded budget.
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
                    lock(&source.ts_queue).clear();
                    feed.applied_step = None;
                    if feed.restart.should_restart() {
                        events.push(Event::error(
                            "encode",
                            format!(
                                "shared encoder for {track_id:?} crashed; restarting (ffmpeg: {})",
                                last_stderr_line(&detail)
                            ),
                        ));
                    } else {
                        events.push(Event::error(
                            "encode",
                            format!(
                                "shared encoder for {track_id:?} crashed and exceeded the restart \
                                 budget; dropping frames (ffmpeg: {})",
                                last_stderr_line(&detail)
                            ),
                        ));
                        feed.stash(frame);
                        continue;
                    }
                } else if lock(&encoders).contains_key(&track_id) {
                    feed.restart.reset();
                }

                let missing = !lock(&encoders).contains_key(&track_id);
                if feed.applied_step != Some(desired) || pli || missing {
                    match make_broadcast_encoder(
                        frame.width,
                        frame.height,
                        desired,
                        &source,
                        events.clone(),
                    ) {
                        Some(encoder) => {
                            lock(&encoders).insert(track_id.clone(), encoder);
                            feed.applied_step = Some(desired);
                        }
                        None => {
                            if !dead && feed.restart.should_restart() {
                                events.push(Event::error(
                                    "encode",
                                    format!("could not spawn shared ffmpeg encoder for {track_id:?}"),
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
                // Flush the pre-open stash IDR-first (the startup burst, unpaced).
                for held in feed.stash.drain(..) {
                    push_capture_ts(&source.ts_queue, held.capture_ts);
                    encoder.write_frame(&held.data);
                }
                // Input fps cap via a token bucket at the rung's fps cap.
                let fps_cap = LADDER[desired].fps_cap.max(1) as f64;
                let now = Instant::now();
                if let Some(last) = feed.last_tick {
                    feed.allowance += now.duration_since(last).as_secs_f64() * fps_cap;
                } else {
                    feed.allowance = 1.0;
                }
                feed.last_tick = Some(now);
                if feed.allowance > fps_cap {
                    feed.allowance = fps_cap;
                }
                if feed.allowance < 1.0 {
                    continue; // over the cap -> drop this frame
                }
                feed.allowance -= 1.0;
                push_capture_ts(&source.ts_queue, frame.capture_ts);
                encoder.write_frame(&frame.data);
            }
        })
        .expect("spawn broadcast encoder feed thread")
}

/// Build a fresh shared ffmpeg encoder for `step`. Its per-access-unit callback
/// fans the encoded NAL units out to every currently-open consumer track for the
/// source: it stamps the access unit's shared capture timestamp on each track and
/// sends the same Annex-B bytes on each track's raw id. Each consumer's own chain
/// packetizes independently (its own SSRC/sequence). Returns `None` if ffmpeg could
/// not be spawned. The PR5.6 invariant guards are preserved (one VCL NAL / one
/// capture timestamp per access unit).
fn make_broadcast_encoder(
    width: u32,
    height: u32,
    step: usize,
    source: &VideoSource,
    events: EventQueue,
) -> Option<Arc<H264Encoder>> {
    let params = EncodeParams {
        fps: rung_encoder_fps(step),
        bitrate: LADDER[step].bitrate,
        scale: LADDER[step].scale,
    };
    let ts_queue = source.ts_queue.clone();
    let frames_encoded = source.frames_encoded.clone();
    let fanout = source.fanout.clone();
    let on_access_unit = move |access_unit: Vec<Vec<u8>>| {
        // Invariant: one access unit -> exactly one RTP frame under one capture
        // timestamp, so exactly one VCL NAL. A multi-slice/aggregation change would
        // desync the timestamp queue and fabricate per-slice timestamps — the
        // Chrome-only, loopback-invisible defect in reports/SPIKE-chrome-pframe.md.
        // Fail loud and drop rather than send a malformed/fabricated frame.
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
        // One capture timestamp per access unit (shared across all consumer tracks
        // for the frame). No silent fallback: an underflow means more access units
        // than input frames (a multi-slice encode), so fail loud and drop.
        let Some(ts) = lock(&ts_queue).pop_front() else {
            eprintln!(
                "[ncwebrtc] INVARIANT VIOLATED: capture-timestamp queue underflow \
                 (more access units than input frames — multi-slice encode?). \
                 Dropping rather than fabricating a timestamp — see \
                 reports/SPIKE-chrome-pframe.md."
            );
            return;
        };
        // One encode -> fan out to every open consumer's track. This counter is the
        // observable that proves the encode is shared (one bump per access unit,
        // regardless of how many consumers it is sent to).
        frames_encoded.fetch_add(1, Ordering::SeqCst);
        let fan = lock(&fanout);
        for (consumer_id, ft) in fan.iter() {
            if !is_open(&ft.open) {
                continue;
            }
            // SAFETY: `raw_id` is a live sys track id this broadcaster created and
            // owns until the consumer leaves / the source is removed; both sys calls
            // are libdatachannel-internally locked.
            let sent = unsafe {
                sys::rtcSetTrackRtpTimestamp(ft.raw_id, ts);
                sys::rtcSendMessage(
                    ft.raw_id,
                    buf.as_ptr() as *const c_char,
                    buf.len() as c_int,
                )
            };
            if sent < 0 {
                // The track went away under us (consumer closed / SRTP torn down).
                // Stop sending to it (flip its open flag) so this does not spam each
                // frame, and surface one per-consumer error; the leave path reclaims
                // the entry. The other consumers and the shared encode are untouched.
                ft.open.store(false, Ordering::SeqCst);
                events.push(Event::error_for(
                    consumer_id,
                    "send",
                    format!("send on a closed track for consumer {consumer_id:?}; suppressing"),
                ));
            }
        }
    };
    match H264Encoder::new(width, height, params, on_access_unit) {
        Ok(encoder) => Some(Arc::new(encoder)),
        Err(err) => {
            crate::transport::debug_trace("B", &format!("encoder spawn failed: {err}"));
            None
        }
    }
}

#[cfg(test)]
mod tests {
    //! Peer-free unit tests for the broadcaster's governance and fan-out
    //! bookkeeping: the min-fold, join PLI suppression, and the fan-out set's
    //! add/remove routing. None touch a socket, a peer, ffmpeg, or the GIL.

    use super::*;
    use crate::congestion::bottom_step;

    // --- min-fold: the worst link caps everyone ------------------------------

    #[test]
    fn fold_rung_is_the_max_ladder_index_the_worst_link_caps_everyone() {
        // A lower estimate is a coarser (higher-index) rung, so the min estimate is
        // the max index. Two consumers on the finest rung and one on a coarse rung
        // -> everyone encodes at the coarse rung.
        assert_eq!(fold_rung(&[0, 0, 3]), 3);
        assert_eq!(fold_rung(&[1, 2, 2]), 2);
    }

    #[test]
    fn a_newly_worst_consumer_lowers_the_shared_rung() {
        let mut steps = vec![0usize, 0, 0];
        assert_eq!(fold_rung(&steps), 0);
        // A consumer's link degrades to step 4 (the coarsest) -> shared rung = 4.
        steps.push(4);
        assert_eq!(fold_rung(&steps), bottom_step().min(4));
    }

    #[test]
    fn fold_rung_over_zero_consumers_does_not_panic() {
        // The last-leave case: an empty fold returns the finest rung, no panic.
        assert_eq!(fold_rung(&[]), TOP_STEP);
    }

    // --- join PLI suppression ------------------------------------------------

    #[test]
    fn a_joining_consumers_early_pli_is_suppressed() {
        // Inside the grace window a joiner's PLI must not trigger a shared restart
        // (which would blip every other consumer); the joiner waits for the next
        // periodic IDR.
        assert!(!should_honor_pli(0.0, JOINER_PLI_GRACE_S));
        assert!(!should_honor_pli(
            JOINER_PLI_GRACE_S - 0.5,
            JOINER_PLI_GRACE_S
        ));
        // An established track past the window: a PLI is real loss and is honoured.
        assert!(should_honor_pli(JOINER_PLI_GRACE_S, JOINER_PLI_GRACE_S));
        assert!(should_honor_pli(10.0, JOINER_PLI_GRACE_S));
    }

    // --- fan-out set routing -------------------------------------------------

    fn fan_track(raw_id: i32, open: bool) -> FanTrack {
        let flag = open_flag();
        flag.store(open, Ordering::SeqCst);
        FanTrack {
            raw_id,
            open: flag,
            control: Arc::new(TrackControl::default()),
            joined_at: Instant::now(),
        }
    }

    #[test]
    fn the_fanout_set_updates_on_add_and_remove() {
        let source = VideoSource::default();
        // Two consumers subscribe; both are in the fan-out set.
        lock(&source.fanout).insert("c1".into(), fan_track(1, true));
        lock(&source.fanout).insert("c2".into(), fan_track(2, true));
        assert_eq!(lock(&source.fanout).len(), 2);

        // One consumer leaves -> it stops receiving (drops out of the set); the
        // other is untouched.
        lock(&source.fanout).remove("c1");
        let fan = lock(&source.fanout);
        assert_eq!(fan.len(), 1);
        assert!(fan.contains_key("c2"));
        assert!(!fan.contains_key("c1"));
    }

    #[test]
    fn open_rungs_only_counts_open_tracks_and_folds_to_the_worst() {
        let source = VideoSource::default();
        let a = fan_track(1, true);
        a.control.set_step(1);
        let b = fan_track(2, true);
        b.control.set_step(3);
        let c = fan_track(3, false); // still negotiating: excluded from the fold
        c.control.set_step(4);
        lock(&source.fanout).insert("a".into(), a);
        lock(&source.fanout).insert("b".into(), b);
        lock(&source.fanout).insert("c".into(), c);

        let steps = open_rungs(&source);
        assert_eq!(steps.len(), 2, "the not-yet-open track is excluded");
        // The shared rung is the worst (coarsest) of the open tracks.
        assert_eq!(fold_rung(&steps), 3);
    }

    #[test]
    fn coalesce_pli_suppresses_a_joiner_but_honours_an_established_track() {
        let source = VideoSource::default();
        // An established track (joined long ago) with a pending PLI: honoured.
        let mut established = fan_track(1, true);
        established.joined_at = Instant::now() - std::time::Duration::from_secs(10);
        established.control.request_pli();
        lock(&source.fanout).insert("old".into(), established);
        assert!(
            coalesce_pli(&source),
            "an established track's PLI is honoured"
        );

        // A fresh joiner with a pending PLI: suppressed (drained, not honoured).
        let joiner = fan_track(2, true);
        joiner.control.request_pli();
        lock(&source.fanout).insert("new".into(), joiner);
        assert!(
            !coalesce_pli(&source),
            "a freshly joined consumer's PLI must not restart the shared encode"
        );
    }
}
