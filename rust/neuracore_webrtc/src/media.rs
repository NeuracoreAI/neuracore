//! H.264 encode/decode, the Annex-B framing the producer feeds the built-in
//! packetizer, and the FU-A depacketizer the consumer still needs.
//!
//! ## Why the producer no longer hand-rolls RTP (PR5)
//!
//! PR4 hand-rolled the producer's RTP (FU-A) because datachannel-rs keeps a
//! track's integer id private. PR5 creates the producer track through the sys
//! layer (`rtcAddTrackEx`, see [`crate::producer`]), recovering that raw id, so it
//! can attach libdatachannel's **built-in** media chain to it:
//! `rtcSetH264Packetizer` (does the FU-A framing, sequence numbers, marker bit and
//! SSRC), plus `rtcChainRtcpSrReporter` / `rtcChainRtcpNackResponder` /
//! `rtcChainPliHandler` / `rtcChainRembHandler`. The producer therefore sends
//! **raw NAL units** (as an Annex-B access unit) and the library packetizes — the
//! hand-rolled `RtpPacketizer` and its unit tests are gone.
//!
//! The **consumer** keeps the FU-A depacketizer: the C API exposes no
//! depacketizer (`rtcChainRtcpReceivingSession` only validates and passes whole
//! RTP packets through to the message callback — see
//! `reports/SPIKE-pr5-media-chain.md`), so inbound media still arrives as raw RTP
//! and [`RtpDepacketizer`] reassembles NAL units for the decoder.
//!
//! The encode (`numpy -> H.264 NAL units`) is kept separate from the send so a
//! later PR can fan one encode out to many consumers. Both ends shell out to a
//! **persistent** ffmpeg subprocess (one per track) — spawning per frame would
//! blow the glass-to-glass budget — exactly as the disk recording path does. The
//! encoder is restartable at a coarser [`crate::congestion::Step`] (lower
//! bitrate, then downscale) so the queue-driven adaptation can degrade under
//! congestion; the restart's first IDR carries SPS/PPS (`repeat-headers=1`).
//!
//! ## What lives here
//!
//! - [`NalSplitter`] — streaming Annex-B byte stream → NAL units.
//! - [`AccessUnitAssembler`] — NAL units → per-frame access units (flush on VCL).
//! - [`annexb_access_unit`] — NAL units → one Annex-B buffer the built-in
//!   packetizer splits (the producer's send payload).
//! - [`RtpDepacketizer`] — the consumer's FU-A reassembly (kept; pure seam).
//! - [`DropPolicy`] — the shed-on-backlog decision behind a fake-clock seam.
//! - [`H264Encoder`] / [`H264Decoder`] — the persistent ffmpeg subprocesses.

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// How long the encoder reader waits for more ffmpeg output before deciding the
/// frame is complete and flushing the trailing NAL. Annex-B carries no NAL length
/// prefix, so a NAL is only "complete" when the next start code arrives — which
/// never happens for the final frame of a paused stream. An idle this short
/// (below the inter-frame gap at 45 fps) flushes that trailing NAL promptly while
/// staying clear of a single frame's atomic ffmpeg write.
const ENCODER_FLUSH_IDLE: Duration = Duration::from_millis(12);

/// How long the decoder feed waits with no new NAL before nudging ffmpeg to emit
/// its last held frame. The h264 decoder only outputs a frame once it sees the
/// *next* access unit, so the final frame of a paused stream is stuck inside
/// ffmpeg until an Access Unit Delimiter (or EOF) arrives. This must clear the
/// inter-frame gap (33 ms at 30 fps) *and* any startup or mid-stream jitter,
/// because a spuriously-early AUD makes ffmpeg re-emit the frame it is still
/// holding — a duplicate. It must also stay below the test collector's quiet
/// window (0.5 s) so a genuinely paused stream's last frame still flushes in
/// time. 250 ms sits comfortably between the two.
const DECODER_FLUSH_IDLE: Duration = Duration::from_millis(250);

/// A bare H.264 Access Unit Delimiter NAL (start code + type-9 + primary_pic
/// payload), fed to the decoder on an idle to flush its last held frame.
const AUD_NAL: [u8; 6] = [0, 0, 0, 1, 0x09, 0x10];

/// How many trailing bytes of a subprocess's stderr to retain for the
/// `on_error` detail when it crashes. ffmpeg's last error line(s) are the useful
/// diagnostic; an unbounded capture would be a slow leak on a chatty process, so
/// the tail is ring-trimmed to this cap.
const STDERR_TAIL_CAP: usize = 2048;

/// The default bounded crash-restart budget for a persistent subprocess: how many
/// consecutive deaths the feed will try to recover from before surfacing a
/// terminal error rather than spinning forever (e.g. ffmpeg permanently missing).
/// A healthy run resets the budget. See [`RestartPolicy`].
pub(crate) const DEFAULT_RESTART_BUDGET: u32 = 5;

/// The built-in H.264 packetizer's max RTP fragment size (bytes). Capped well
/// below the 64 KiB loopback datagram so a 640x480 keyframe actually fragments
/// into FU-A packets rather than riding as one datagram — the fragmentation path
/// must be exercised on loopback, not just under a real MTU.
pub(crate) const MAX_FRAGMENT_SIZE: u16 = 1200;

/// RTP fixed header length (V/P/X/CC + M/PT + seq + timestamp + ssrc), no CSRCs.
/// The consumer's depacketizer skips this prefix on every inbound packet.
const RTP_HEADER_LEN: usize = 12;

/// The FU-A NAL type (RFC 6184 §5.8): fragmentation unit without DON.
const FU_A_TYPE: u8 = 28;

/// The RTCP CNAME the built-in packetizer's SR reporter advertises. libdatachannel
/// **throws** (`rtcSetH264Packetizer` returns -1) if `rtcPacketizerInit.cname` is
/// null — and then `rtcChainRtcpSrReporter` fails too because it chains onto the
/// packetizer's RTP config that was never created. So the cname must always be a
/// non-null, non-empty C string; [`packetizer_cname`] is the single source and
/// `cname_is_non_null_and_non_empty` guards the invariant.
pub(crate) const PACKETIZER_CNAME: &str = "neuracore";

/// The packetizer's non-null CNAME as a `CString`, ready for
/// `rtcPacketizerInit.cname`. Panics only if [`PACKETIZER_CNAME`] ever contains an
/// interior NUL (a compile-time-constant string that never will), so the live
/// path can `.expect` it and the guard test pins it.
pub(crate) fn packetizer_cname() -> std::ffi::CString {
    std::ffi::CString::new(PACKETIZER_CNAME).expect("packetizer cname has no interior NUL")
}

/// 90 kHz is the RTP clock for video. The producer steps the track's RTP
/// timestamp by `VIDEO_CLOCK_HZ / fps` per access unit (via
/// `rtcSetTrackRtpTimestamp`) so every packet of one frame shares a timestamp and
/// the SR reporter sees a coherent clock.
pub(crate) const VIDEO_CLOCK_HZ: u32 = 90_000;

/// The ffmpeg binary to shell out to: `NEURACORE_WEBRTC_FFMPEG` if set, else
/// `ffmpeg` on `PATH` (the same provisioning the disk recording path uses).
pub(crate) fn ffmpeg_bin() -> String {
    std::env::var("NEURACORE_WEBRTC_FFMPEG").unwrap_or_else(|_| "ffmpeg".to_string())
}

// ---------------------------------------------------------------------------
// Annex-B NAL splitting and access-unit grouping
// ---------------------------------------------------------------------------

/// Streaming Annex-B splitter: feed it arbitrary byte chunks from the encoder's
/// stdout and it yields complete NAL units (start codes stripped). A NAL is only
/// emitted once the *next* start code is seen, so the trailing partial NAL stays
/// buffered until more bytes arrive or [`NalSplitter::flush`] is called at EOF.
pub(crate) struct NalSplitter {
    buf: Vec<u8>,
}

impl NalSplitter {
    pub(crate) fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Indices of every `00 00 01` start-code prefix in `buf`. A 4-byte
    /// `00 00 00 01` start code contains a `00 00 01` at offset 1; the extra
    /// leading zero is stripped from the preceding NAL's tail instead.
    fn start_codes(buf: &[u8]) -> Vec<usize> {
        let mut positions = Vec::new();
        if buf.len() < 3 {
            return positions;
        }
        let mut i = 0;
        while i + 2 < buf.len() {
            if buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1 {
                positions.push(i);
                i += 3;
            } else {
                i += 1;
            }
        }
        positions
    }

    /// Feed more bytes; return every NAL unit that became complete.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<Vec<u8>> {
        self.buf.extend_from_slice(bytes);
        let positions = Self::start_codes(&self.buf);
        let mut out = Vec::new();
        if positions.len() < 2 {
            return out;
        }
        for pair in positions.windows(2) {
            let start = pair[0] + 3;
            let mut end = pair[1];
            // Strip the trailing zero(s) that belong to the next 4-byte start
            // code (H.264 RBSP never legitimately ends in 0x00 after the stop
            // bit, so this cannot truncate real payload).
            while end > start && self.buf[end - 1] == 0 {
                end -= 1;
            }
            if end > start {
                out.push(self.buf[start..end].to_vec());
            }
        }
        // Retain from the final start code onward (its NAL is not yet complete).
        let last = *positions.last().unwrap();
        self.buf.drain(..last);
        out
    }

    /// Emit the final buffered NAL at end of stream (if any).
    pub(crate) fn flush(&mut self) -> Option<Vec<u8>> {
        let positions = Self::start_codes(&self.buf);
        let start = *positions.first()? + 3;
        let mut end = self.buf.len();
        while end > start && self.buf[end - 1] == 0 {
            end -= 1;
        }
        let nal = (end > start).then(|| self.buf[start..end].to_vec());
        self.buf.clear();
        nal
    }
}

/// Returns whether a NAL header byte denotes a VCL (coded-slice) NAL — types 1
/// (non-IDR slice) through 5 (IDR slice). A frame's single slice is its last
/// NAL, so a VCL NAL completes the access unit.
fn is_vcl(nal: &[u8]) -> bool {
    matches!(nal.first().map(|b| b & 0x1F), Some(1..=5))
}

/// Groups NAL units into access units (one decoded frame each). The encoder is
/// configured single-slice, so a VCL NAL is the last NAL of its frame; non-VCL
/// NALs (SPS/PPS/SEI) accumulate as the prefix of the access unit they precede.
pub(crate) struct AccessUnitAssembler {
    nals: Vec<Vec<u8>>,
}

impl AccessUnitAssembler {
    pub(crate) fn new() -> Self {
        Self { nals: Vec::new() }
    }

    /// Append a NAL; return the completed access unit when this NAL is a VCL slice.
    pub(crate) fn push(&mut self, nal: Vec<u8>) -> Option<Vec<Vec<u8>>> {
        let vcl = is_vcl(&nal);
        self.nals.push(nal);
        vcl.then(|| std::mem::take(&mut self.nals))
    }
}

/// The number of VCL (coded-slice) NAL units in an access unit. The producer
/// sends one access unit as exactly **one** RTP frame under one capture
/// timestamp, so a well-formed access unit carries exactly one VCL NAL. More than
/// one means a slicing or NAL-aggregation change (e.g. x264 multi-slice threading,
/// or STAP-A) silently broke the one-VCL-per-frame invariant the timestamping
/// depends on — the exact defect in `reports/SPIKE-chrome-pframe.md`. The send
/// path asserts `== 1` and drops loudly rather than fabricate timestamps; see
/// [`x264_params`] for the encoder lever that keeps it true. Pure and testable.
pub(crate) fn vcl_nal_count(nals: &[Vec<u8>]) -> usize {
    nals.iter().filter(|nal| is_vcl(nal)).count()
}

// ---------------------------------------------------------------------------
// Producer send framing (the built-in packetizer does the RTP)
// ---------------------------------------------------------------------------

/// Join an access unit's NAL units (start codes stripped) into one Annex-B
/// buffer with 4-byte long start codes, the payload the producer hands
/// `rtcSendMessage`. The attached `rtcSetH264Packetizer` is configured with the
/// long-start-sequence NAL separator, so it splits this back into NAL units and
/// does the RTP framing (single-NAL or FU-A), sequence numbers, marker bit and
/// SSRC itself. Pure so the framing is unit-testable without the chain.
pub(crate) fn annexb_access_unit(nals: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in nals {
        if nal.is_empty() {
            continue;
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    }
    out
}

/// Reassembles H.264 NAL units from inbound RTP. Single NAL packets pass through;
/// FU-A fragments are stitched back into the original NAL. A sequence gap mid-FU-A
/// drops the partial NAL rather than emitting a corrupt one. Owned and pure.
pub(crate) struct RtpDepacketizer {
    fu_buffer: Vec<u8>,
    in_fu: bool,
    last_seq: Option<u16>,
}

impl RtpDepacketizer {
    pub(crate) fn new() -> Self {
        Self {
            fu_buffer: Vec::new(),
            in_fu: false,
            last_seq: None,
        }
    }

    /// Feed one RTP packet; return any NAL unit(s) that became complete.
    pub(crate) fn depacketize(&mut self, packet: &[u8]) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        if packet.len() <= RTP_HEADER_LEN {
            return out;
        }
        let seq = u16::from_be_bytes([packet[2], packet[3]]);
        // Drop a non-advancing sequence: the producer's NACK responder
        // retransmits lost packets, and a retransmit (or a reordered duplicate)
        // arrives with a sequence we have already processed. Re-feeding it to the
        // decoder would surface a duplicate frame, so only strictly-newer
        // sequences pass. "Newer" is the forward half of the 16-bit sequence
        // space (1..=32767 ahead); a duplicate (0) or an old packet is dropped.
        if let Some(prev) = self.last_seq {
            let ahead = seq.wrapping_sub(prev);
            if ahead == 0 || ahead >= 0x8000 {
                return out;
            }
        }
        let gap = matches!(self.last_seq, Some(prev) if seq != prev.wrapping_add(1));
        self.last_seq = Some(seq);

        let payload = &packet[RTP_HEADER_LEN..];
        let nal_type = payload[0] & 0x1F;
        match nal_type {
            FU_A_TYPE => {
                if payload.len() < 2 {
                    return out;
                }
                let fu_indicator = payload[0];
                let fu_header = payload[1];
                let start = fu_header & 0x80 != 0;
                let end = fu_header & 0x40 != 0;
                let orig_type = fu_header & 0x1F;
                if start {
                    self.fu_buffer.clear();
                    self.fu_buffer.push((fu_indicator & 0xE0) | orig_type);
                    self.fu_buffer.extend_from_slice(&payload[2..]);
                    self.in_fu = true;
                } else if !self.in_fu || gap {
                    // Missing the FU start, or a packet was lost mid-fragment:
                    // abandon the partial NAL rather than emit a corrupt one.
                    self.in_fu = false;
                    self.fu_buffer.clear();
                    return out;
                } else {
                    self.fu_buffer.extend_from_slice(&payload[2..]);
                }
                if end && self.in_fu {
                    out.push(std::mem::take(&mut self.fu_buffer));
                    self.in_fu = false;
                }
            }
            1..=23 => {
                // A single, complete NAL unit. Any in-flight FU-A is broken.
                self.in_fu = false;
                self.fu_buffer.clear();
                out.push(payload.to_vec());
            }
            _ => {
                // STAP-A/MTAP/etc. are never produced by our packetizer; ignore.
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Drop policy
// ---------------------------------------------------------------------------

/// The shed-on-backlog decision. The encoder is a fixed-rate sink fed from the
/// bounded ingress queue; while the queue has room (the steady state at or below
/// the encoder's throughput) nothing is ever dropped, so at or below 30 fps the
/// stream is loss-free. Frames are shed only once the encoder has backed the
/// queue up to capacity — which only happens above the sustainable rate.
///
/// Pure so a fake clock/queue can drive the shed-above-30 / never-below-30 /
/// zero-drop-at-or-below-30 contract without a live encoder.
pub(crate) struct DropPolicy {
    capacity: usize,
}

impl DropPolicy {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
        }
    }

    /// Admit a newly submitted frame iff the ingress queue has room for it.
    pub(crate) fn admit(&self, backlog: usize) -> bool {
        backlog < self.capacity
    }
}

// ---------------------------------------------------------------------------
// Crash-restart policy (the bounded-retry decision behind a seam)
// ---------------------------------------------------------------------------

/// The bounded decision a feed makes when a persistent subprocess dies: try to
/// restart it, up to a budget, then give up and surface a terminal error rather
/// than spinning. The budget guards against a permanently-broken process (e.g.
/// ffmpeg missing or a fatal arg) turning a crash into a hot loop. A subprocess
/// that comes back healthy (produces output again) calls [`reset`](Self::reset)
/// to restore the full budget, so transient crashes never exhaust it.
///
/// Pure and clock-free, so the crash-restart decision is unit-tested with a fake
/// "subprocess that dies" without a live ffmpeg.
#[derive(Debug)]
pub(crate) struct RestartPolicy {
    max_consecutive: u32,
    consecutive: u32,
}

impl RestartPolicy {
    pub(crate) fn new(max_consecutive: u32) -> Self {
        Self {
            max_consecutive,
            consecutive: 0,
        }
    }

    /// Record a death and decide whether to attempt another restart. Returns
    /// `true` (counting the attempt) while the budget remains; `false` once it is
    /// exhausted.
    pub(crate) fn should_restart(&mut self) -> bool {
        if self.consecutive >= self.max_consecutive {
            return false;
        }
        self.consecutive += 1;
        true
    }

    /// Mark the subprocess healthy again, clearing the consecutive-failure count
    /// so a later transient crash gets the full budget.
    pub(crate) fn reset(&mut self) {
        self.consecutive = 0;
    }

    /// Whether the restart budget is spent (no further restart will be attempted
    /// until a [`reset`](Self::reset)). Exercised by the crash-restart unit tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn exhausted(&self) -> bool {
        self.consecutive >= self.max_consecutive
    }
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self::new(DEFAULT_RESTART_BUDGET)
    }
}

/// Append `chunk` to a bounded stderr-tail buffer, trimming the front so it never
/// grows past [`STDERR_TAIL_CAP`]. Shared by the encoder/decoder stderr drains.
fn append_stderr_tail(tail: &Mutex<String>, chunk: &str) {
    let mut buf = tail.lock().unwrap_or_else(|e| e.into_inner());
    buf.push_str(chunk);
    if buf.len() > STDERR_TAIL_CAP {
        let cut = buf.len() - STDERR_TAIL_CAP;
        // Trim on a char boundary so the retained tail is valid UTF-8.
        let cut = (cut..=buf.len())
            .find(|&i| buf.is_char_boundary(i))
            .unwrap_or(buf.len());
        buf.drain(..cut);
    }
}

/// Drain a subprocess's stderr to a bounded tail buffer (and echo it when
/// `NEURACORE_WEBRTC_DEBUG` is set). Reused for both ffmpeg subprocesses; runs on
/// its own thread and ends on stderr EOF, so it is joined on `Drop` like the other
/// reader threads.
fn spawn_stderr_drain(
    name: &str,
    mut stderr: std::process::ChildStderr,
    tail: Arc<Mutex<String>>,
) -> std::io::Result<JoinHandle<()>> {
    let echo = std::env::var_os("NEURACORE_WEBRTC_DEBUG").is_some();
    std::thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let mut buf = [0u8; 1 << 12];
            loop {
                match stderr.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let chunk = String::from_utf8_lossy(&buf[..n]);
                        if echo {
                            eprint!("{chunk}");
                        }
                        append_stderr_tail(&tail, &chunk);
                    }
                }
            }
        })
}

// ---------------------------------------------------------------------------
// Persistent ffmpeg encode / decode subprocesses
// ---------------------------------------------------------------------------

/// The encoder's rate-control / resolution parameters for one ladder rung. The
/// input is always the full-resolution rgb24 source; `scale` (1 = full, 2 = half
/// each axis) downscales **inside** ffmpeg, and `bitrate` is the libx264 target.
/// Restarting the encoder with a coarser [`EncodeParams`] is how the queue-driven
/// adaptation degrades under congestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EncodeParams {
    pub fps: u32,
    pub bitrate: u32,
    pub scale: u32,
}

/// The libx264 `-x264-params` string for an encoder with the given `keyint`.
///
/// ## `threads=1` is load-bearing for Chrome (single slice per frame, low latency)
///
/// `-tune zerolatency` turns on x264 **slice-based threading** (`sliced-threads=1`),
/// which splits every coded frame into one slice *per worker thread* — on a many-core
/// host that is ~7+ slices per frame, each its own VCL NAL. Our pipeline assumes
/// **one VCL NAL per frame**: [`AccessUnitAssembler`] flushes an access unit on each
/// VCL slice, and the producer sends each access unit as a separate `rtcSendMessage`
/// with its own RTP timestamp and marker bit. With multi-slice frames that turns one
/// captured frame into N RTP "frames" — N timestamps (the capture-timestamp queue
/// underflows ~N:1 and fabricates the rest, so they run *backwards*), N markers, and
/// each carrying only 1/N of the macroblocks. The loopback ffmpeg consumer reassembles
/// the Annex-B NALs regardless and masks it, but **Chrome's RTP frame assembler keys
/// frames on the timestamp**: it sees N partial pseudo-frames per real frame and never
/// completes the inter-keyframe P-frames (`framesReceived` stalls at the keyframe rate
/// with `packetsLost == 0`). See `reports/SPIKE-chrome-pframe.md`.
///
/// `threads=1` (not merely `sliced-threads=0`) is the fix: a single thread emits a
/// single slice per frame. Disabling *only* slice threading would let x264 fall back
/// to **frame-based** threading, whose pipeline delays output by one frame per worker
/// (~14 frames / ~310 ms on a 14-core host — a glass-to-glass SLO blowout). A fully
/// serial encoder has no such pipeline and still clears the source rate with room to
/// spare at `ultrafast`. `slices=1` is kept belt-and-braces (a single thread cannot
/// slice-parallelise anyway).
pub(crate) fn x264_params(keyint: &str) -> String {
    // Test-only hook (`NCD_WEBRTC_FORCE_SLICES=N`): deliberately emit N slices per
    // frame so the one-VCL-NAL-per-access-unit invariant guard can be exercised
    // end-to-end — it must DROP the malformed access unit and shout, never panic.
    // Unset (the only production value) keeps the single-slice invariant below.
    if let Some(n) = std::env::var_os("NCD_WEBRTC_FORCE_SLICES") {
        let n = n.to_string_lossy();
        return format!(
            "keyint={keyint}:min-keyint={keyint}:scenecut=0:bframes=0:\
             repeat-headers=1:slices={n}:annexb=1"
        );
    }
    format!(
        "keyint={keyint}:min-keyint={keyint}:scenecut=0:bframes=0:\
         repeat-headers=1:slices=1:threads=1:annexb=1"
    )
}

/// A persistent ffmpeg encoder: raw rgb24 frames in on stdin, H.264 Annex-B out
/// on stdout. A reader thread splits the output into NAL units, groups them into
/// per-frame access units, and hands each access unit to the `on_access_unit`
/// callback (which frames and sends it on the track). One per track; restarted
/// (a fresh instance) when the ladder rung changes.
pub(crate) struct H264Encoder {
    child: Mutex<Child>,
    stdin: Mutex<Option<ChildStdin>>,
    reader: Mutex<Option<JoinHandle<()>>>,
    splitter: Mutex<Option<JoinHandle<()>>>,
    stderr: Mutex<Option<JoinHandle<()>>>,
    frame_len: usize,
    /// Cleared by the reader thread when ffmpeg's stdout reaches EOF — i.e. the
    /// subprocess exited. The feed reads [`is_alive`](Self::is_alive) to detect a
    /// crash and trigger a bounded restart instead of stalling silently.
    alive: Arc<AtomicBool>,
    /// The tail of ffmpeg's stderr, the diagnostic surfaced in the `on_error`
    /// event when the encoder crashes.
    stderr_tail: Arc<Mutex<String>>,
}

impl H264Encoder {
    /// Spawn ffmpeg for a `width`x`height` rgb24 source at the given
    /// [`EncodeParams`]. `on_access_unit` runs on the reader thread for every
    /// encoded frame. The first output IDR always carries SPS/PPS
    /// (`repeat-headers=1`), so a restart at a new rung is itself a clean
    /// keyframe — which is how a coalesced PLI is satisfied.
    pub(crate) fn new(
        width: u32,
        height: u32,
        params: EncodeParams,
        mut on_access_unit: impl FnMut(Vec<Vec<u8>>) + Send + 'static,
    ) -> std::io::Result<Self> {
        let fps = params.fps.max(1);
        let keyint = fps.to_string();
        let scale = params.scale.max(1);
        // Even dimensions for yuv420p chroma. Downscale inside ffmpeg so the wire
        // resolution drops while the source stays full-res; the consumer scales
        // any rung back to its fixed decode size.
        let out_w = ((width / scale) & !1).max(2);
        let out_h = ((height / scale) & !1).max(2);
        // VBV-capped CRF rate control: CRF keeps low-complexity content small (so
        // a clean link never bursts the loopback socket — strict CBR padding
        // would, and did), while `maxrate`/`bufsize` cap the peak at the ladder
        // rung so a coarser rung genuinely throttles the stream under a
        // constrained link. A small `bufsize` keeps the cap responsive (low
        // latency) rather than letting a big VBV buffer absorb a whole second.
        let kbps = (params.bitrate / 1000).max(50);
        let maxrate = format!("{kbps}k");
        let bufsize = format!("{}k", (kbps / 2).max(32));
        let scale_filter = format!("scale={out_w}:{out_h}:flags=fast_bilinear");
        let mut args: Vec<String> = [
            "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        args.extend([
            "-s".into(), format!("{width}x{height}"),
            "-r".into(), fps.to_string(),
            "-i".into(), "pipe:0".into(),
            "-an".into(),
            "-vf".into(), scale_filter,
            "-c:v".into(), "libx264".into(),
            "-profile:v".into(), "baseline".into(),
            "-pix_fmt".into(), "yuv420p".into(),
            "-preset".into(), "ultrafast".into(),
            "-tune".into(), "zerolatency".into(),
            "-bf".into(), "0".into(),
            "-g".into(), keyint.clone(),
            "-crf".into(), "26".into(),
            "-maxrate".into(), maxrate,
            "-bufsize".into(), bufsize,
            "-x264-params".into(),
            x264_params(&keyint),
            "-f".into(), "h264".into(), "pipe:1".into(),
        ]);
        // Pipe stderr (always) into a bounded tail buffer so a crash carries
        // ffmpeg's last error line in the surfaced `on_error`; the drain thread
        // also echoes it under NEURACORE_WEBRTC_DEBUG, preserving the old inherit
        // behaviour for interactive debugging.
        let mut child = Command::new(ffmpeg_bin())
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdin = child.stdin.take();
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("ffmpeg encoder stdout unavailable"))?;
        let child_stderr = child
            .stderr
            .take()
            .ok_or_else(|| std::io::Error::other("ffmpeg encoder stderr unavailable"))?;
        let stderr_tail = Arc::new(Mutex::new(String::new()));
        let stderr = spawn_stderr_drain("ncwebrtc-encode-err", child_stderr, stderr_tail.clone())?;

        // Two threads: the reader does blocking stdout reads and forwards raw
        // chunks; the splitter feeds them through the NAL splitter/assembler with
        // an idle-flush so the trailing NAL of the final (or any paused) frame is
        // not stranded waiting for a start code that never comes.
        let alive = Arc::new(AtomicBool::new(true));
        let reader_alive = alive.clone();
        let (chunk_tx, chunk_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        let reader = std::thread::Builder::new()
            .name("ncwebrtc-encode-read".into())
            .spawn(move || {
                let mut buf = [0u8; 1 << 16];
                loop {
                    match stdout.read(&mut buf) {
                        // stdout EOF/err == the subprocess exited: mark it dead so
                        // the feed restarts it rather than stalling.
                        Ok(0) | Err(_) => {
                            reader_alive.store(false, Ordering::SeqCst);
                            break;
                        }
                        Ok(n) => {
                            if chunk_tx.send(buf[..n].to_vec()).is_err() {
                                break;
                            }
                        }
                    }
                }
            })?;
        let splitter = std::thread::Builder::new()
            .name("ncwebrtc-encode-split".into())
            .spawn(move || {
                let mut splitter = NalSplitter::new();
                let mut assembler = AccessUnitAssembler::new();
                let mut emit = |nal: Vec<u8>, assembler: &mut AccessUnitAssembler| {
                    if let Some(au) = assembler.push(nal) {
                        on_access_unit(au);
                    }
                };
                loop {
                    match chunk_rx.recv_timeout(ENCODER_FLUSH_IDLE) {
                        Ok(chunk) => {
                            for nal in splitter.push(&chunk) {
                                emit(nal, &mut assembler);
                            }
                        }
                        // Idle: ffmpeg produced nothing for a frame period, so the
                        // buffered trailing NAL is a complete frame — flush it.
                        Err(RecvTimeoutError::Timeout) => {
                            if let Some(nal) = splitter.flush() {
                                emit(nal, &mut assembler);
                            }
                        }
                        Err(RecvTimeoutError::Disconnected) => {
                            if let Some(nal) = splitter.flush() {
                                emit(nal, &mut assembler);
                            }
                            break;
                        }
                    }
                }
            })?;

        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            reader: Mutex::new(Some(reader)),
            splitter: Mutex::new(Some(splitter)),
            stderr: Mutex::new(Some(stderr)),
            frame_len: (width as usize) * (height as usize) * 3,
            alive,
            stderr_tail,
        })
    }

    /// Write one raw rgb24 frame to ffmpeg's stdin. Blocks if ffmpeg has backed
    /// up — that back-pressure propagates to the bounded ingress queue, which is
    /// where frames are shed. Returns false once stdin has gone away (a crash):
    /// the feed treats that, like [`is_alive`](Self::is_alive), as a death signal.
    pub(crate) fn write_frame(&self, data: &[u8]) -> bool {
        if data.len() != self.frame_len {
            return true; // wrong shape for this encoder; skip, do not wedge
        }
        let mut guard = self.stdin.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_mut() {
            Some(stdin) => stdin.write_all(data).is_ok(),
            None => false,
        }
    }

    /// Whether ffmpeg is still running (its stdout has not hit EOF). The feed
    /// polls this to detect a crash and trigger a bounded restart.
    pub(crate) fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// A snapshot of ffmpeg's stderr tail, for the surfaced crash diagnostic.
    pub(crate) fn stderr_tail(&self) -> String {
        self.stderr_tail
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

impl Drop for H264Encoder {
    fn drop(&mut self) {
        // Close stdin (EOF -> ffmpeg flushes and exits), then make sure the
        // process is gone and join the reader so no thread outlives the encoder.
        *self.stdin.lock().unwrap_or_else(|e| e.into_inner()) = None;
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
        // Reader exits on stdout EOF and drops its chunk sender; the splitter then
        // sees Disconnected and exits; the stderr drain exits on stderr EOF. Join
        // all three so no thread (and no zombie child) outlives the encoder.
        for slot in [&self.reader, &self.splitter, &self.stderr] {
            if let Some(handle) = slot.lock().unwrap_or_else(|e| e.into_inner()).take() {
                let _ = handle.join();
            }
        }
    }
}

/// A persistent ffmpeg decoder: H.264 Annex-B in on stdin, raw rgb24 frames out
/// on stdout. A writer thread feeds NAL units (start-code framed) to stdin; a
/// reader thread reads fixed-size decoded frames and hands each to `on_frame`.
/// One per inbound track.
pub(crate) struct H264Decoder {
    input_tx: Mutex<Option<std::sync::mpsc::Sender<Vec<u8>>>>,
    child: Mutex<Child>,
    writer: Mutex<Option<JoinHandle<()>>>,
    reader: Mutex<Option<JoinHandle<()>>>,
    stderr: Mutex<Option<JoinHandle<()>>>,
    /// Cleared by the reader thread when ffmpeg's stdout reaches EOF (the
    /// subprocess exited). The consumer polls [`is_alive`](Self::is_alive) to
    /// detect a decoder crash and restart the receive pipeline.
    alive: Arc<AtomicBool>,
    /// The tail of ffmpeg's stderr for the surfaced crash diagnostic.
    stderr_tail: Arc<Mutex<String>>,
}

impl H264Decoder {
    pub(crate) fn new(
        width: u32,
        height: u32,
        mut on_frame: impl FnMut(Vec<u8>) + Send + 'static,
    ) -> std::io::Result<Self> {
        // Normalise every rung back to the fixed decode size: the producer may
        // downscale under congestion (a coarser ladder rung), so scale whatever
        // resolution arrives up to width x height. The scale filter reconfigures
        // on a mid-stream resolution change, and the block-coded header band
        // survives the rescale, so the consumer's fixed-size frame reader stays
        // correct across an adaptation step.
        let scale_filter = format!("scale={width}:{height}:flags=fast_bilinear");
        let mut child = Command::new(ffmpeg_bin())
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                // Minimal probe + low_delay so the decoder emits each frame as
                // soon as it is decoded, starting from the first IDR, rather than
                // buffering megabytes to analyse the stream. Baseline has no
                // B-frames, so output order == input order.
                "-probesize",
                "32",
                "-analyzeduration",
                "0",
                "-flags",
                "low_delay",
                "-f",
                "h264",
                "-i",
                "pipe:0",
                "-an",
                "-vf",
                &scale_filter,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("ffmpeg decoder stdin unavailable"))?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("ffmpeg decoder stdout unavailable"))?;
        let child_stderr = child
            .stderr
            .take()
            .ok_or_else(|| std::io::Error::other("ffmpeg decoder stderr unavailable"))?;
        let stderr_tail = Arc::new(Mutex::new(String::new()));
        let stderr = spawn_stderr_drain("ncwebrtc-decode-err", child_stderr, stderr_tail.clone())?;
        let alive = Arc::new(AtomicBool::new(true));

        let (input_tx, input_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        let writer = std::thread::Builder::new()
            .name("ncwebrtc-decode-in".into())
            .spawn(move || {
                // `flushed` tracks whether the last real NAL has already been
                // followed by an AUD, so a long pause emits exactly one AUD (which
                // flushes ffmpeg's last held frame) rather than a stream of them.
                let mut flushed = true;
                loop {
                    match input_rx.recv_timeout(DECODER_FLUSH_IDLE) {
                        Ok(annexb) => {
                            if stdin.write_all(&annexb).is_err() {
                                break;
                            }
                            flushed = false;
                        }
                        Err(RecvTimeoutError::Timeout) => {
                            if !flushed {
                                if stdin.write_all(&AUD_NAL).is_err() {
                                    break;
                                }
                                flushed = true;
                            }
                        }
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }
                // Dropping stdin signals EOF so ffmpeg drains and exits.
            })?;

        let frame_len = (width as usize) * (height as usize) * 3;
        let reader_alive = alive.clone();
        let reader = std::thread::Builder::new()
            .name("ncwebrtc-decode-out".into())
            .spawn(move || {
                let mut frame = vec![0u8; frame_len];
                // Drop a decoded picture byte-identical to the one just emitted.
                // libdatachannel's RTCP receiving session re-delivers the first
                // frame's media at startup (re-packetized, so its new RTP
                // sequences slip past the depacketizer's sequence de-dup), making
                // the decoder emit the identical first picture twice. A real
                // encoder never produces two bit-exact consecutive frames, so
                // suppressing an exact duplicate is information-preserving and
                // costs one frame comparison.
                let mut prev: Option<Vec<u8>> = None;
                // Each decoded picture is exactly frame_len bytes; a short read
                // means EOF/shutdown.
                while stdout.read_exact(&mut frame).is_ok() {
                    if prev.as_deref() == Some(frame.as_slice()) {
                        continue;
                    }
                    on_frame(frame.clone());
                    prev = Some(frame.clone());
                }
                // A short read means EOF/shutdown — the subprocess exited. Mark it
                // dead so the consumer restarts the receive pipeline.
                reader_alive.store(false, Ordering::SeqCst);
            })?;

        Ok(Self {
            input_tx: Mutex::new(Some(input_tx)),
            child: Mutex::new(child),
            writer: Mutex::new(Some(writer)),
            reader: Mutex::new(Some(reader)),
            stderr: Mutex::new(Some(stderr)),
            alive,
            stderr_tail,
        })
    }

    /// Whether ffmpeg is still running (its stdout has not hit EOF). The consumer
    /// polls this to detect a decoder crash and restart the receive pipeline.
    pub(crate) fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// A snapshot of ffmpeg's stderr tail, for the surfaced crash diagnostic.
    pub(crate) fn stderr_tail(&self) -> String {
        self.stderr_tail
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Feed one NAL unit (no start code) to the decoder as Annex-B.
    pub(crate) fn feed_nal(&self, nal: &[u8]) {
        let mut annexb = Vec::with_capacity(nal.len() + 4);
        annexb.extend_from_slice(&[0, 0, 0, 1]);
        annexb.extend_from_slice(nal);
        if let Some(tx) = self
            .input_tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            let _ = tx.send(annexb);
        }
    }
}

impl Drop for H264Decoder {
    fn drop(&mut self) {
        // Drop the sender (writer thread ends, closes stdin -> ffmpeg EOF), then
        // kill/reap the process and join both threads.
        *self.input_tx.lock().unwrap_or_else(|e| e.into_inner()) = None;
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
        for slot in [&self.writer, &self.reader, &self.stderr] {
            if let Some(handle) = slot.lock().unwrap_or_else(|e| e.into_inner()).take() {
                let _ = handle.join();
            }
        }
    }
}

/// A live track-open flag shared between the producer's track handler (which
/// flips it on `on_open`) and the encoder feed thread (which holds frames until
/// it is set, so the very first encoded access unit — always an IDR — is the
/// first thing sent once SRTP is ready).
pub(crate) type OpenFlag = Arc<AtomicBool>;

/// Convenience: a fresh, closed open-flag.
pub(crate) fn open_flag() -> OpenFlag {
    Arc::new(AtomicBool::new(false))
}

/// Whether `flag` has been flipped open.
pub(crate) fn is_open(flag: &OpenFlag) -> bool {
    flag.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    //! Peer-free, ffmpeg-free unit tests for the framing we own: Annex-B
    //! splitting, access-unit grouping, RTP packetize/depacketize (including
    //! FU-A and gap handling), and the drop policy behind a fake clock/queue.

    use super::*;

    fn start_code(four: bool, nal: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(if four { &[0, 0, 0, 1] } else { &[0, 0, 1] });
        v.extend_from_slice(nal);
        v
    }

    // --- Annex-B splitting ----------------------------------------------------

    #[test]
    fn nal_splitter_separates_units_and_strips_4byte_start_codes() {
        let mut s = NalSplitter::new();
        let mut stream = Vec::new();
        stream.extend(start_code(true, &[0x67, 1, 2, 3])); // SPS (4-byte code)
        stream.extend(start_code(false, &[0x68, 4, 5])); // PPS (3-byte code)
        stream.extend(start_code(true, &[0x65, 9, 9, 9])); // IDR slice

        // Feed in two arbitrary chunks to exercise the cross-read buffering.
        let mut out = s.push(&stream[..7]);
        out.extend(s.push(&stream[7..]));
        if let Some(tail) = s.flush() {
            out.push(tail);
        }
        assert_eq!(
            out,
            vec![vec![0x67, 1, 2, 3], vec![0x68, 4, 5], vec![0x65, 9, 9, 9],]
        );
    }

    #[test]
    fn access_unit_assembler_flushes_on_the_vcl_slice() {
        let mut a = AccessUnitAssembler::new();
        assert!(a.push(vec![0x67, 1]).is_none()); // SPS: non-VCL, accumulates
        assert!(a.push(vec![0x68, 2]).is_none()); // PPS: non-VCL, accumulates
        let au = a
            .push(vec![0x65, 3])
            .expect("IDR completes the access unit");
        assert_eq!(au, vec![vec![0x67, 1], vec![0x68, 2], vec![0x65, 3]]);
        // A bare non-IDR slice is its own access unit.
        let au2 = a
            .push(vec![0x61, 4])
            .expect("non-IDR slice completes its AU");
        assert_eq!(au2, vec![vec![0x61, 4]]);
    }

    // --- producer send framing (Annex-B for the built-in packetizer) ---------

    #[test]
    fn annexb_access_unit_prefixes_each_nal_with_a_long_start_code() {
        let sps = vec![0x67, 1, 2, 3];
        let idr = vec![0x65, 9, 9];
        let buf = annexb_access_unit(&[sps.clone(), idr.clone()]);
        assert_eq!(
            buf,
            vec![0, 0, 0, 1, 0x67, 1, 2, 3, 0, 0, 0, 1, 0x65, 9, 9],
            "each NAL is preceded by a 4-byte start code so the chain's \
             long-start-sequence separator can split them"
        );
    }

    #[test]
    fn annexb_access_unit_skips_empty_nals() {
        assert!(annexb_access_unit(&[vec![]]).is_empty());
        assert_eq!(
            annexb_access_unit(&[vec![], vec![0x41, 7]]),
            vec![0, 0, 0, 1, 0x41, 7]
        );
    }

    // --- single-slice encoder invariant (Chrome P-frame assembly) ------------

    #[test]
    fn x264_params_force_a_single_slice_per_frame() {
        // The producer sends each access unit as one RTP frame (its own timestamp
        // + marker), and AccessUnitAssembler flushes one access unit per VCL NAL,
        // so the encoder MUST emit exactly one slice per frame. `-tune zerolatency`
        // otherwise enables slice-based threading (one slice per worker thread),
        // which Chrome's timestamp-keyed frame assembler cannot reassemble — it
        // sees N partial pseudo-frames per real frame and never completes the
        // P-frames. Pin both levers that guarantee a single slice.
        let params = x264_params("45");
        assert!(
            params.contains("slices=1"),
            "must request a single slice: {params}"
        );
        assert!(
            params.contains("threads=1"),
            "must run x264 fully serial: one slice/frame and no frame-thread \
             pipeline latency (zerolatency's slice threading would multi-slice): {params}"
        );
        // The keyint is interpolated into both keyint and min-keyint.
        assert!(params.contains("keyint=45:min-keyint=45"), "keyint wired: {params}");
    }

    // --- one-VCL-NAL-per-access-unit invariant -------------------------------

    #[test]
    fn vcl_nal_count_is_one_for_a_normal_access_unit_and_trips_on_multi_slice() {
        // A normal access unit — SPS + PPS + a single IDR slice (the assembler's
        // steady-state output) — carries exactly one VCL NAL and so maps to one
        // RTP frame under one capture timestamp.
        let normal = vec![vec![0x67, 1, 2], vec![0x68, 3], vec![0x65, 9, 9]];
        assert_eq!(vcl_nal_count(&normal), 1, "SPS+PPS+IDR is one VCL NAL");
        // A bare non-IDR slice is also a single VCL NAL.
        assert_eq!(vcl_nal_count(&[vec![0x41, 7]]), 1);
        // Two coded slices grouped into one access unit — what a multi-slice /
        // NAL-aggregation change would produce — has >1 VCL NAL, so the send
        // path's `!= 1` guard trips and drops the AU rather than emitting
        // out-of-order or fabricated per-slice timestamps.
        let multi_slice = vec![vec![0x65, 1], vec![0x65, 2], vec![0x65, 3]];
        assert_eq!(vcl_nal_count(&multi_slice), 3, "multi-slice trips the guard");
        // Parameter sets alone are not a complete frame (zero VCL NALs).
        assert_eq!(vcl_nal_count(&[vec![0x67, 1], vec![0x68, 2]]), 0);
    }

    // --- packetizer-init cname guard -----------------------------------------

    #[test]
    fn cname_is_non_null_and_non_empty() {
        // rtcSetH264Packetizer returns -1 (and the SR reporter then fails) if the
        // packetizer init's cname is null. Pin that the single source is always a
        // valid, non-empty C string, so the live path never passes null.
        let cname = packetizer_cname();
        assert!(!cname.as_bytes().is_empty(), "cname must be non-empty");
        // CString guarantees NUL termination and no interior NUL.
        assert_eq!(cname.to_str(), Ok(PACKETIZER_CNAME));
    }

    // --- RTP depacketizer (kept; the C API has no depacketizer) ---------------

    /// Build the RTP packets for one access unit the way libdatachannel's
    /// built-in H.264 packetizer does (single-NAL under the fragment cap, FU-A
    /// above it), so the kept depacketizer tests have wire-shaped input without
    /// the deleted producer packetizer. Sequence numbers are monotonic from
    /// `seq0`; the marker is set on the access unit's final packet.
    fn build_rtp(seq0: u16, ssrc: u32, nals: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let cap = MAX_FRAGMENT_SIZE as usize;
        let mut seq = seq0;
        let mut header = |seq: &mut u16| {
            let mut h = vec![0x80u8, 96];
            h.extend_from_slice(&seq.to_be_bytes());
            h.extend_from_slice(&0u32.to_be_bytes()); // timestamp (unused here)
            h.extend_from_slice(&ssrc.to_be_bytes());
            *seq = seq.wrapping_add(1);
            h
        };
        let mut out: Vec<Vec<u8>> = Vec::new();
        for nal in nals {
            if nal.len() <= cap {
                let mut pkt = header(&mut seq);
                pkt.extend_from_slice(nal);
                out.push(pkt);
            } else {
                let f_nri = nal[0] & 0xE0;
                let nal_type = nal[0] & 0x1F;
                let chunks: Vec<&[u8]> = nal[1..].chunks(cap - 2).collect();
                let last = chunks.len() - 1;
                for (i, chunk) in chunks.iter().enumerate() {
                    let fu_header = ((i == 0) as u8) << 7 | ((i == last) as u8) << 6 | nal_type;
                    let mut pkt = header(&mut seq);
                    pkt.push(f_nri | FU_A_TYPE);
                    pkt.push(fu_header);
                    pkt.extend_from_slice(chunk);
                    out.push(pkt);
                }
            }
        }
        if let Some(last) = out.last_mut() {
            last[1] |= 0x80;
        }
        out
    }

    #[test]
    fn round_trip_reassembles_a_fragmented_nal_exactly() {
        let mut nal = vec![0x65];
        nal.extend((0..(MAX_FRAGMENT_SIZE as usize * 2 + 37)).map(|i| (i % 253) as u8));
        let pkts = build_rtp(0, 42, &[nal.clone()]);

        let mut d = RtpDepacketizer::new();
        let mut got = Vec::new();
        for pkt in &pkts {
            got.extend(d.depacketize(pkt));
        }
        assert_eq!(got, vec![nal], "FU-A round trip is byte-exact");
    }

    #[test]
    fn round_trip_preserves_a_multi_nal_access_unit() {
        let sps = vec![0x67, 1, 2, 3];
        let pps = vec![0x68, 4, 5];
        let mut idr = vec![0x65];
        idr.extend(std::iter::repeat(9u8).take(MAX_FRAGMENT_SIZE as usize + 5)); // forces FU-A
        let pkts = build_rtp(0, 7, &[sps.clone(), pps.clone(), idr.clone()]);

        let mut d = RtpDepacketizer::new();
        let mut got = Vec::new();
        for pkt in &pkts {
            got.extend(d.depacketize(pkt));
        }
        assert_eq!(got, vec![sps, pps, idr]);
    }

    #[test]
    fn depacketizer_drops_a_retransmitted_duplicate_sequence() {
        // The producer's NACK responder retransmits packets; a retransmit (or a
        // reordered duplicate) carries a sequence already processed and must not
        // surface a second copy of the NAL.
        let pkts = build_rtp(10, 7, &[vec![0x67, 1], vec![0x41, 2], vec![0x41, 3]]);
        let mut d = RtpDepacketizer::new();
        let mut got = Vec::new();
        for pkt in &pkts {
            got.extend(d.depacketize(pkt));
        }
        assert_eq!(got.len(), 3, "three distinct NALs");
        // Replay the middle packet (a retransmit): no new NAL is emitted.
        assert!(
            d.depacketize(&pkts[1]).is_empty(),
            "an already-seen sequence is dropped, not re-emitted"
        );
    }

    #[test]
    fn depacketizer_drops_a_partial_nal_on_a_sequence_gap() {
        let mut nal = vec![0x65];
        nal.extend(std::iter::repeat(3u8).take(MAX_FRAGMENT_SIZE as usize * 3));
        let pkts = build_rtp(0, 7, &[nal]);
        assert!(pkts.len() >= 3);

        let mut d = RtpDepacketizer::new();
        // Deliver the FU-A start, then SKIP a middle fragment, then the rest.
        let mut got = Vec::new();
        got.extend(d.depacketize(&pkts[0])); // start
        for pkt in &pkts[2..] {
            got.extend(d.depacketize(pkt)); // gap: pkts[1] dropped
        }
        assert!(got.is_empty(), "a mid-FU gap abandons the corrupt NAL");

        // The depacketizer recovers cleanly on the next complete single NAL.
        let single = build_rtp(500, 7, &[vec![0x67, 1, 2, 3]]);
        let recovered = d.depacketize(&single[0]);
        assert_eq!(recovered, vec![vec![0x67, 1, 2, 3]]);
    }

    // --- drop policy ----------------------------------------------------------

    /// Drive the drop policy over a fake clock: submit at `fps`, drain (encode)
    /// at `encoder_fps`, and report (delivered, dropped) across `seconds`.
    fn simulate(capacity: usize, fps: f64, encoder_fps: f64, seconds: f64) -> (usize, usize) {
        let policy = DropPolicy::new(capacity);
        let submit_dt = 1.0 / fps;
        let drain_dt = 1.0 / encoder_fps;
        let mut backlog = 0usize;
        let mut next_drain = drain_dt;
        let (mut delivered, mut dropped) = (0usize, 0usize);
        let total = (fps * seconds) as usize;
        for i in 0..total {
            let now = i as f64 * submit_dt;
            while next_drain <= now {
                if backlog > 0 {
                    backlog -= 1;
                    delivered += 1;
                }
                next_drain += drain_dt;
            }
            if policy.admit(backlog) {
                backlog += 1;
            } else {
                dropped += 1;
            }
        }
        delivered += backlog; // the encoder drains the remainder
        (delivered, dropped)
    }

    #[test]
    fn zero_drops_at_or_below_thirty_when_the_encoder_keeps_up() {
        // Encoder comfortably faster than a 30fps source -> the queue never fills.
        let (delivered, dropped) = simulate(16, 30.0, 45.0, 4.0);
        assert_eq!(dropped, 0, "no deliberate drops at or below 30fps");
        assert_eq!(delivered, (30.0f64 * 4.0) as usize, "every frame delivered");
    }

    #[test]
    fn sheds_above_thirty_but_holds_the_delivered_floor() {
        // 45fps source, encoder sustains ~35fps -> excess is shed, floor holds.
        let (delivered, dropped) = simulate(16, 45.0, 35.0, 4.0);
        assert!(dropped > 0, "over-rate source sheds the excess");
        let delivered_fps = delivered as f64 / 4.0;
        assert!(
            delivered_fps >= 30.0,
            "delivered {delivered_fps:.1}fps must stay at/above the 30 floor"
        );
    }

    #[test]
    fn admit_tracks_queue_occupancy() {
        let policy = DropPolicy::new(16);
        assert!(policy.admit(0), "room -> admit");
        assert!(policy.admit(15), "last slot -> admit");
        assert!(!policy.admit(16), "full -> shed");
    }

    // --- crash-restart policy -------------------------------------------------

    #[test]
    fn restart_policy_permits_up_to_the_budget_then_gives_up() {
        // A "subprocess that keeps dying": each death asks should_restart. The
        // feed gets the budget of restart attempts, then a terminal give-up so a
        // permanently-broken encoder surfaces an error instead of hot-looping.
        let mut policy = RestartPolicy::new(3);
        assert!(policy.should_restart(), "1st death -> restart");
        assert!(policy.should_restart(), "2nd death -> restart");
        assert!(policy.should_restart(), "3rd death -> restart");
        assert!(!policy.should_restart(), "budget spent -> give up");
        assert!(policy.exhausted());
    }

    #[test]
    fn restart_policy_reset_after_a_healthy_run_restores_the_budget() {
        // A recovered subprocess (it produced output again) resets the budget, so
        // a much later, unrelated transient crash still gets the full retry count.
        let mut policy = RestartPolicy::new(2);
        assert!(policy.should_restart());
        assert!(policy.should_restart());
        assert!(!policy.should_restart(), "budget spent");
        policy.reset();
        assert!(!policy.exhausted(), "reset clears the exhausted state");
        assert!(policy.should_restart(), "full budget again after a healthy run");
    }

    // --- bounded stderr tail --------------------------------------------------

    #[test]
    fn stderr_tail_is_ring_trimmed_to_the_cap() {
        let tail = Mutex::new(String::new());
        // Write well past the cap; only the trailing STDERR_TAIL_CAP bytes survive.
        let chunk = "x".repeat(STDERR_TAIL_CAP);
        append_stderr_tail(&tail, &chunk);
        append_stderr_tail(&tail, "ERROR: ffmpeg died");
        let got = tail.lock().unwrap().clone();
        assert!(got.len() <= STDERR_TAIL_CAP, "tail stays bounded: {}", got.len());
        assert!(
            got.ends_with("ERROR: ffmpeg died"),
            "the most recent (diagnostic) bytes are retained"
        );
    }
}
