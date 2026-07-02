//! Queue-driven congestion adaptation: the RTCP feedback the producer reads, and
//! the lightweight estimator that turns it into a rung on the [`LADDER`].
//!
//! libdatachannel 0.23.2 implements **no** transport-cc and does **no** bandwidth
//! estimation — see `reports/SPIKE-pr5-media-chain.md`. The only two real signals
//! the producer can read are:
//!
//!  * **REMB** — the receiver-estimated max bitrate, delivered through the chain's
//!    [`rtcChainRembHandler`] callback. A real browser (Chrome) computes this from
//!    its own receive-side bandwidth estimator; the libdatachannel loopback
//!    consumer only echoes whatever `rtcRequestBitrate` was set to, so REMB is the
//!    *Chrome-path* driver.
//!  * **RTCP RR** — receiver reports carrying `fraction_lost` and `jitter`, which
//!    the consumer's receiving session computes from the real sequence-number gaps.
//!    RR is the *loopback-path* driver (it reacts to netem rate/loss for real,
//!    where loopback REMB cannot).
//!
//! Neither signal is transport-cc and there is no full GCC here: the estimator is
//! a deliberately small loss/headroom controller over a fixed [`LADDER`]. It is a
//! pure seam — every decision is a function of the samples it is fed and a clock
//! passed in — so it is exercised by a fake clock in the unit tests below without
//! a peer, a socket, or live media.
//!
//! ## What lives here
//!
//! - [`Step`] / [`LADDER`] — the adaptation rungs (input-fps cap, resolution
//!   scale, target encoder bitrate).
//! - [`parse_rtcp_reports`] — hand-parses fraction-lost/jitter out of a compound
//!   RTCP RR (or SR) packet, because the C API exposes no RR decoder.
//! - [`Estimator`] — folds REMB + RR samples into a ladder step, degrading on
//!   pressure and recovering conservatively (sticky).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// One rung of the adaptation ladder. `fps_cap` caps the *input* frame rate fed
/// to the encoder (shed toward the 30 fps floor first); `scale` divides each
/// spatial axis (1 = full resolution, 2 = half); `bitrate` is the encoder's
/// target bits/sec. Coarser rungs combine a lower fps, a lower bitrate, and
/// eventually a downscale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Step {
    pub fps_cap: u32,
    pub scale: u32,
    pub bitrate: u32,
}

/// The fixed adaptation ladder, finest (index 0) to coarsest. Step 0 is
/// effectively unconstrained for a 45 fps source; step 1 sheds input fps to the
/// 30 fps floor (the cheap first move); steps 2+ lower the bitrate and finally
/// downscale for larger or sustained pressure. The 30 fps floor is held from
/// step 1 on, so adaptation never sheds *delivered* fps below the contract floor.
pub(crate) const LADDER: &[Step] = &[
    Step { fps_cap: 60, scale: 1, bitrate: 2_500_000 },
    Step { fps_cap: 40, scale: 1, bitrate: 1_500_000 },
    Step { fps_cap: 36, scale: 1, bitrate: 900_000 },
    Step { fps_cap: 36, scale: 2, bitrate: 500_000 },
    Step { fps_cap: 36, scale: 2, bitrate: 300_000 },
];

/// The step the encoder starts on (finest).
pub(crate) const TOP_STEP: usize = 0;

/// Highest (coarsest) ladder index.
pub(crate) fn bottom_step() -> usize {
    LADDER.len() - 1
}

// --- tuning ----------------------------------------------------------------

/// REMB below `committed_bitrate * REMB_PRESSURE` signals the link cannot carry
/// the current rung: degrade. (Headroom for protocol overhead/burstiness.)
const REMB_PRESSURE: f64 = 0.85;
/// REMB below `committed_bitrate * REMB_SEVERE` is acute under-provisioning:
/// degrade immediately rather than waiting out the degrade window.
const REMB_SEVERE: f64 = 0.5;
/// RR fraction-lost above this sustained for the degrade window is pressure.
const LOSS_PRESSURE: f64 = 0.02;
/// RR fraction-lost above this is acute loss: degrade immediately.
const LOSS_SEVERE: f64 = 0.10;
/// Sustained pressure must persist this long before a mild degrade fires (so a
/// single noisy sample does not move the ladder).
const DEGRADE_WINDOW_S: f64 = 1.5;
/// Recovery is deliberately slow and sticky: the link must look clear this much
/// longer than a degrade before the estimator steps back up one rung. It is an
/// order of magnitude longer than the degrade window so the ladder settles on a
/// fitting rung and does not oscillate back into loss — a recovery into a rung
/// the link cannot carry re-loses and re-degrades, and that lossy excursion shows
/// as corruption (inter-frame error propagation). Recovery is especially cautious
/// because the only headroom signal here (loopback REMB) is a fixed echo that
/// cannot veto a premature step-up.
const RECOVERY_WINDOW_S: f64 = 20.0;
/// Only recover when REMB shows headroom for the *finer* rung we would move to
/// (its bitrate times this margin), so we do not immediately re-degrade.
const RECOVERY_HEADROOM: f64 = 1.25;

/// One feedback observation. Either field may be absent: the loopback path only
/// has useful RR, the Chrome path drives REMB. The estimator uses whichever is
/// present, taking the worse of the two when both are.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Sample {
    /// Receiver-estimated max bitrate in bits/sec, if a REMB arrived.
    pub remb_bps: Option<u32>,
    /// RR fraction lost in 0.0..=1.0, if an RR arrived.
    pub fraction_lost: Option<f64>,
}

/// The pure congestion estimator. Folds [`Sample`]s into a [`LADDER`] index,
/// degrading promptly under pressure and recovering slowly when the link looks
/// clear. Holds no clock of its own — every method takes `now_s` — so the unit
/// tests drive it deterministically.
#[derive(Debug)]
pub(crate) struct Estimator {
    step: usize,
    /// When the link first looked pressured since the last clear sample.
    pressured_since: Option<f64>,
    /// When the link first looked clear since the last pressured sample.
    clear_since: Option<f64>,
    /// Last REMB seen, carried so a recovery decision can check headroom even on
    /// an RR-only sample.
    last_remb: Option<u32>,
}

impl Default for Estimator {
    fn default() -> Self {
        Self {
            step: TOP_STEP,
            pressured_since: None,
            clear_since: None,
            last_remb: None,
        }
    }
}

impl Estimator {
    /// The current ladder index. (Read by the tests; the live path uses the
    /// value [`observe`](Self::observe) returns.)
    #[allow(dead_code)]
    pub(crate) fn step(&self) -> usize {
        self.step
    }

    /// Fold one observation in at time `now_s` and return the (possibly changed)
    /// ladder index. Degrades immediately on severe pressure, after
    /// [`DEGRADE_WINDOW_S`] on mild pressure, and recovers one rung only after a
    /// longer clear window with REMB headroom.
    pub(crate) fn observe(&mut self, sample: Sample, now_s: f64) -> usize {
        if let Some(remb) = sample.remb_bps {
            self.last_remb = Some(remb);
        }
        let committed = LADDER[self.step].bitrate as f64;

        let remb_ratio = sample.remb_bps.map(|r| r as f64 / committed);
        let loss = sample.fraction_lost.unwrap_or(0.0);

        let severe = matches!(remb_ratio, Some(r) if r < REMB_SEVERE) || loss > LOSS_SEVERE;
        let mild = severe
            || matches!(remb_ratio, Some(r) if r < REMB_PRESSURE)
            || loss > LOSS_PRESSURE;

        if mild {
            self.clear_since = None;
            let since = *self.pressured_since.get_or_insert(now_s);
            let sustained = now_s - since >= DEGRADE_WINDOW_S;
            if (severe || sustained) && self.step < bottom_step() {
                self.step += 1;
                // Reset the windows: re-measure pressure/clearness against the
                // new rung rather than carrying the old timer across a move.
                self.pressured_since = None;
                self.clear_since = None;
            }
        } else {
            self.pressured_since = None;
            let since = *self.clear_since.get_or_insert(now_s);
            let clear_long = now_s - since >= RECOVERY_WINDOW_S;
            if clear_long && self.step > TOP_STEP && self.has_recovery_headroom() {
                self.step -= 1;
                self.pressured_since = None;
                self.clear_since = None;
            }
        }
        self.step
    }

    /// Whether REMB (if known) shows enough headroom to move up to the next finer
    /// rung without immediately re-degrading. With no REMB (the loopback path),
    /// a clear loss window alone authorises recovery.
    fn has_recovery_headroom(&self) -> bool {
        match self.last_remb {
            Some(remb) => {
                let finer = LADDER[self.step - 1].bitrate as f64;
                remb as f64 >= finer * RECOVERY_HEADROOM
            }
            None => true,
        }
    }
}

// ---------------------------------------------------------------------------
// RTCP RR / SR report-block parsing (hand-rolled; the C API decodes none)
// ---------------------------------------------------------------------------

/// One RTCP report block's loss + jitter figures, as parsed off the wire.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ReportBlock {
    /// SSRC the report is *about* (our outgoing media SSRC).
    pub ssrc: u32,
    /// Loss fraction since the last report, in 0.0..=1.0 (the wire byte / 256).
    pub fraction_lost: f64,
    /// Cumulative packets lost (24-bit signed on the wire; widened here).
    pub cumulative_lost: u32,
    /// Interarrival jitter in RTP timestamp units.
    pub jitter: u32,
}

const RTCP_HEADER_LEN: usize = 4;
const PT_SR: u8 = 200;
const PT_RR: u8 = 201;
const REPORT_BLOCK_LEN: usize = 24;
/// Bytes from an RR packet's start to its first report block: the 4-byte common
/// header plus the 4-byte reporter SSRC.
const RR_REPORTS_OFFSET: usize = 8;
/// Bytes from an SR packet's start to its first report block: the 4-byte common
/// header, the 4-byte reporter SSRC, and the 20-byte sender info.
const SR_REPORTS_OFFSET: usize = 28;

/// Parse every report block out of a (possibly compound) RTCP packet. Walks each
/// sub-packet by its length field and pulls report blocks from RR (PT=201) and
/// SR (PT=200) packets; all other packet types (REMB/PLI/NACK/BYE/SDES) are
/// skipped. Tolerates a truncated or non-RTCP buffer by returning what it could
/// read. Pure — unit-tested against synthetic compound RTCP.
pub(crate) fn parse_rtcp_reports(buf: &[u8]) -> Vec<ReportBlock> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + RTCP_HEADER_LEN <= buf.len() {
        let rc = (buf[off] & 0x1F) as usize; // reception report count
        let pt = buf[off + 1];
        // length is in 32-bit words minus one; total packet bytes = (len+1)*4.
        let len_words = u16::from_be_bytes([buf[off + 2], buf[off + 3]]) as usize;
        let packet_len = (len_words + 1) * 4;
        if packet_len < RTCP_HEADER_LEN || off + packet_len > buf.len() {
            break;
        }
        let reports_at = match pt {
            PT_RR => Some(off + RR_REPORTS_OFFSET),
            PT_SR => Some(off + SR_REPORTS_OFFSET),
            _ => None,
        };
        if let Some(mut block_off) = reports_at {
            for _ in 0..rc {
                if block_off + REPORT_BLOCK_LEN > off + packet_len {
                    break;
                }
                out.push(parse_report_block(&buf[block_off..block_off + REPORT_BLOCK_LEN]));
                block_off += REPORT_BLOCK_LEN;
            }
        }
        off += packet_len;
    }
    out
}

/// Decode one 24-byte RTCP report block.
fn parse_report_block(b: &[u8]) -> ReportBlock {
    let ssrc = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
    let fraction_lost = b[4] as f64 / 256.0;
    let cumulative_lost = u32::from_be_bytes([0, b[5], b[6], b[7]]);
    let jitter = u32::from_be_bytes([b[12], b[13], b[14], b[15]]);
    ReportBlock {
        ssrc,
        fraction_lost,
        cumulative_lost,
        jitter,
    }
}

// ---------------------------------------------------------------------------
// Shared effect surface
// ---------------------------------------------------------------------------

/// The per-track effect surface the estimator writes and the encoder feed reads.
/// The estimator (driven on libdatachannel's RTCP callback threads) only flips
/// these atomics; the feed thread applies them — caps input fps and restarts the
/// ffmpeg subprocess at the new rung — so no libdatachannel callback ever blocks
/// on an ffmpeg restart. `pli_pending` coalesces PLIs into a single keyframe
/// request the feed satisfies via a restart.
#[derive(Debug)]
pub(crate) struct TrackControl {
    desired_step: AtomicU32,
    /// The coarsest rung ever requested (a high-water mark), so a test or operator
    /// can see that adaptation fired even after the link recovered and the rung
    /// stepped back up.
    max_step: AtomicU32,
    pli_pending: std::sync::atomic::AtomicBool,
}

impl Default for TrackControl {
    fn default() -> Self {
        Self {
            desired_step: AtomicU32::new(TOP_STEP as u32),
            max_step: AtomicU32::new(TOP_STEP as u32),
            pli_pending: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl TrackControl {
    /// Publish the ladder rung the estimator now wants, advancing the high-water
    /// mark if this is the coarsest rung seen so far.
    pub(crate) fn set_step(&self, step: usize) {
        self.desired_step.store(step as u32, Ordering::SeqCst);
        self.max_step.fetch_max(step as u32, Ordering::SeqCst);
    }

    /// The rung the feed thread should be encoding at.
    pub(crate) fn desired_step(&self) -> usize {
        self.desired_step.load(Ordering::SeqCst) as usize
    }

    /// The coarsest rung the estimator ever requested.
    pub(crate) fn max_step(&self) -> usize {
        self.max_step.load(Ordering::SeqCst) as usize
    }

    /// Coalesce a PLI: record that a keyframe is wanted.
    pub(crate) fn request_pli(&self) {
        self.pli_pending.store(true, Ordering::SeqCst);
    }

    /// Take the coalesced PLI request (true at most once per burst).
    pub(crate) fn take_pli(&self) -> bool {
        self.pli_pending.swap(false, Ordering::SeqCst)
    }
}

/// The estimator plus its effect surface, shared between the RTCP callbacks
/// (which observe) and the feed thread (which applies). The estimator is behind a
/// `Mutex` because the REMB callback, the RR callback, and the PLI callback all
/// fire on libdatachannel threads.
/// How long after the first feedback sample the controller ignores REMB/RR. The
/// connection's startup (the stash flush, the first IDR, ICE settling) produces a
/// transient loss/jitter spike on an otherwise clean link; acting on it would
/// degrade quality on a link that is actually fine. A real sustained constraint
/// (netem) persists well past this, so the warmup only suppresses the transient.
const WARMUP_S: f64 = 2.5;

pub(crate) struct CongestionController {
    estimator: std::sync::Mutex<Estimator>,
    control: Arc<TrackControl>,
    /// Our committed outgoing SSRC, so RR report blocks for other SSRCs are
    /// ignored.
    ssrc: u32,
    /// Seconds the controller suppresses feedback after its first sample.
    warmup_s: f64,
    /// Wall-clock seconds of the first observed sample, set lazily.
    started_at: std::sync::Mutex<Option<f64>>,
}

impl CongestionController {
    pub(crate) fn new(ssrc: u32, control: Arc<TrackControl>) -> Self {
        Self::with_warmup(ssrc, control, WARMUP_S)
    }

    /// Construct with an explicit warmup (0.0 disables it, for deterministic
    /// unit tests of the REMB/RR wiring).
    pub(crate) fn with_warmup(ssrc: u32, control: Arc<TrackControl>, warmup_s: f64) -> Self {
        Self {
            estimator: std::sync::Mutex::new(Estimator::default()),
            control,
            ssrc,
            warmup_s,
            started_at: std::sync::Mutex::new(None),
        }
    }

    /// Whether the controller is still inside its startup warmup at `now`.
    fn in_warmup(&self, now: f64) -> bool {
        let mut started = self.started_at.lock().unwrap_or_else(|e| e.into_inner());
        let start = *started.get_or_insert(now);
        now - start < self.warmup_s
    }

    /// A monotonic seconds clock for the live path (the unit tests pass their own).
    fn now_s() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Feed a REMB estimate (bits/sec) in.
    pub(crate) fn on_remb(&self, bitrate_bps: u32) {
        if self.in_warmup(Self::now_s()) {
            return;
        }
        let step = {
            let mut est = self.estimator.lock().unwrap_or_else(|e| e.into_inner());
            est.observe(
                Sample {
                    remb_bps: Some(bitrate_bps),
                    ..Default::default()
                },
                Self::now_s(),
            )
        };
        self.control.set_step(step);
    }

    /// Feed raw inbound RTCP in: parse RR report blocks for our SSRC and fold
    /// their loss/jitter into the estimator.
    pub(crate) fn on_rtcp(&self, buf: &[u8]) {
        if self.in_warmup(Self::now_s()) {
            return;
        }
        let mut applied = None;
        for block in parse_rtcp_reports(buf) {
            if block.ssrc != self.ssrc {
                continue;
            }
            // Jitter is parsed for observability (the report records it); loss
            // fraction is what actually drives the ladder.
            crate::transport::debug_trace(
                "P",
                &format!(
                    "rr ssrc={:#x} loss={:.3} jitter={}",
                    block.ssrc, block.fraction_lost, block.jitter
                ),
            );
            let mut est = self.estimator.lock().unwrap_or_else(|e| e.into_inner());
            applied = Some(est.observe(
                Sample {
                    fraction_lost: Some(block.fraction_lost),
                    ..Default::default()
                },
                Self::now_s(),
            ));
        }
        if let Some(step) = applied {
            self.control.set_step(step);
        }
    }

    /// A PLI arrived: coalesce it into the feed's keyframe request.
    pub(crate) fn on_pli(&self) {
        self.control.request_pli();
    }
}

#[cfg(test)]
mod tests {
    //! Peer-free, clock-injected tests for the estimator, the RR parser, and the
    //! REMB-consumption path. None touch a socket, a peer, or live media.

    use super::*;

    // --- RR / SR parsing -----------------------------------------------------

    /// Build one RTCP RR with a single report block carrying `fraction_lost`
    /// (a raw byte) and `jitter`, reporting on `ssrc`.
    fn rr_packet(reporter: u32, ssrc: u32, fraction_lost: u8, cumulative: u32, jitter: u32) -> Vec<u8> {
        let mut p = Vec::new();
        p.push(0x80 | 1); // V=2, P=0, RC=1
        p.push(PT_RR);
        // length in words minus one: header(1) + reporter ssrc(1) + 6 words block = 8 -> len 7
        p.extend_from_slice(&7u16.to_be_bytes());
        p.extend_from_slice(&reporter.to_be_bytes());
        // report block
        p.extend_from_slice(&ssrc.to_be_bytes());
        p.push(fraction_lost);
        p.extend_from_slice(&cumulative.to_be_bytes()[1..]); // 24-bit
        p.extend_from_slice(&0u32.to_be_bytes()); // ext highest seq
        p.extend_from_slice(&jitter.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes()); // lsr
        p.extend_from_slice(&0u32.to_be_bytes()); // dlsr
        p
    }

    #[test]
    fn parses_fraction_lost_and_jitter_from_an_rr() {
        // 13% loss -> byte 33 (33/256 ~= 0.129).
        let pkt = rr_packet(0x1111, 0xCAFE, 33, 7, 4096);
        let blocks = parse_rtcp_reports(&pkt);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].ssrc, 0xCAFE);
        assert!((blocks[0].fraction_lost - 33.0 / 256.0).abs() < 1e-9);
        assert_eq!(blocks[0].cumulative_lost, 7);
        assert_eq!(blocks[0].jitter, 4096);
    }

    #[test]
    fn parses_report_blocks_out_of_a_compound_packet_after_an_sr() {
        // Compound: a minimal SR (no report blocks) followed by an RR with one.
        let mut sr = Vec::new();
        sr.push(0x80); // V=2, RC=0
        sr.push(PT_SR);
        sr.extend_from_slice(&6u16.to_be_bytes()); // header+ssrc+sender info = 7 words -> len 6
        sr.extend_from_slice(&0x2222u32.to_be_bytes()); // ssrc
        sr.extend_from_slice(&[0u8; 20]); // sender info
        let rr = rr_packet(0x2222, 0xBEEF, 8, 2, 100);
        let mut compound = sr;
        compound.extend_from_slice(&rr);

        let blocks = parse_rtcp_reports(&compound);
        assert_eq!(blocks.len(), 1, "the SR carried no blocks; the RR carried one");
        assert_eq!(blocks[0].ssrc, 0xBEEF);
        assert_eq!(blocks[0].jitter, 100);
    }

    #[test]
    fn tolerates_reordered_and_truncated_rtcp_without_panicking() {
        // A truncated tail after a valid RR: parser returns the good block and
        // stops at the bad length rather than reading out of bounds.
        let mut pkt = rr_packet(1, 2, 5, 1, 9);
        pkt.extend_from_slice(&[0x81, PT_RR, 0xFF, 0xFF]); // claims a huge length
        let blocks = parse_rtcp_reports(&pkt);
        assert_eq!(blocks.len(), 1);
        // Pure garbage parses to nothing, no panic.
        assert!(parse_rtcp_reports(&[0xAA, 0xBB]).is_empty());
    }

    // --- estimator: degrade on loss (the loopback/RR path) -------------------

    fn loss(f: f64) -> Sample {
        Sample {
            fraction_lost: Some(f),
            ..Default::default()
        }
    }
    fn remb(bps: u32) -> Sample {
        Sample {
            remb_bps: Some(bps),
            ..Default::default()
        }
    }

    #[test]
    fn severe_loss_degrades_immediately_one_rung() {
        let mut est = Estimator::default();
        assert_eq!(est.step(), 0);
        // 20% loss is above LOSS_SEVERE -> immediate single-rung degrade.
        assert_eq!(est.observe(loss(0.20), 0.0), 1);
        assert_eq!(est.observe(loss(0.20), 0.1), 2);
    }

    #[test]
    fn mild_loss_only_degrades_after_the_sustained_window() {
        let mut est = Estimator::default();
        // 3% loss is mild (> LOSS_PRESSURE, < LOSS_SEVERE): no move yet.
        assert_eq!(est.observe(loss(0.03), 0.0), 0);
        assert_eq!(est.observe(loss(0.03), 1.0), 0, "still inside the degrade window");
        // Past DEGRADE_WINDOW_S of sustained mild pressure -> one rung down.
        assert_eq!(est.observe(loss(0.03), 1.6), 1);
    }

    #[test]
    fn a_single_clear_sample_resets_the_degrade_window() {
        let mut est = Estimator::default();
        assert_eq!(est.observe(loss(0.03), 0.0), 0);
        // A clear sample mid-window cancels the pending degrade.
        assert_eq!(est.observe(loss(0.0), 1.0), 0);
        assert_eq!(est.observe(loss(0.03), 1.4), 0, "window restarts from the new pressure");
        assert_eq!(est.observe(loss(0.03), 3.0), 1);
    }

    // --- estimator: REMB path (the Chrome path) ------------------------------

    #[test]
    fn remb_below_severe_fraction_degrades_immediately() {
        let mut est = Estimator::default();
        // committed at step 0 is 2.5Mbit; REMB 1.0Mbit is < 0.5x -> severe.
        assert_eq!(est.observe(remb(1_000_000), 0.0), 1);
    }

    #[test]
    fn remb_mild_pressure_waits_for_the_window() {
        let mut est = Estimator::default();
        // step1 committed 1.5Mbit; REMB 1.2Mbit is 0.8x (< 0.85 pressure, > 0.5).
        est.observe(loss(0.20), 0.0); // -> step 1 fast
        assert_eq!(est.step(), 1);
        assert_eq!(est.observe(remb(1_200_000), 10.0), 1, "mild, window restarts");
        assert_eq!(est.observe(remb(1_200_000), 11.6), 2, "sustained -> degrade");
    }

    // --- estimator: conservative, sticky recovery ----------------------------

    #[test]
    fn recovery_is_slow_sticky_and_needs_headroom() {
        let mut est = Estimator::default();
        // Drive down to step 2 on severe loss.
        est.observe(loss(0.20), 0.0);
        est.observe(loss(0.20), 0.1);
        assert_eq!(est.step(), 2);

        // Clear loss but no REMB headroom info yet: with no REMB, a long clear
        // window recovers one rung (loopback path).
        assert_eq!(est.observe(loss(0.0), 1.0), 2, "inside the recovery window");
        assert_eq!(est.observe(loss(0.0), 15.0), 2, "still inside recovery window");
        assert_eq!(
            est.observe(loss(0.0), 21.5),
            1,
            "recovers exactly one rung only after the long clear window (since 1.0)"
        );
        // Recovery is one rung at a time: another full clear window for the next.
        assert_eq!(est.observe(loss(0.0), 22.0), 1);
        assert_eq!(est.observe(loss(0.0), 42.5), 0);
    }

    #[test]
    fn remb_recovery_requires_headroom_for_the_finer_rung() {
        let mut est = Estimator::default();
        est.observe(loss(0.20), 0.0);
        est.observe(loss(0.20), 0.1);
        assert_eq!(est.step(), 2); // step 1 bitrate is 1.5Mbit

        // Clear loss, but REMB only 1.6Mbit: finer rung (step 1) needs
        // 1.5M * 1.25 = 1.875M of headroom, so even past the clear window
        // recovery is withheld.
        est.observe(remb(1_600_000), 1.0);
        assert_eq!(
            est.observe(remb(1_600_000), 22.0),
            2,
            "past the window but no headroom -> stay put"
        );
        // REMB rises above the headroom bar -> recover one rung.
        est.observe(remb(2_000_000), 23.0);
        assert_eq!(est.observe(remb(2_000_000), 44.0), 1);
    }

    #[test]
    fn never_degrades_past_the_bottom_rung() {
        let mut est = Estimator::default();
        for t in 0..20 {
            est.observe(loss(0.5), t as f64 * 0.1);
        }
        assert_eq!(est.step(), bottom_step());
    }

    // --- controller wiring: REMB + RR -> the shared control ------------------

    #[test]
    fn controller_publishes_the_estimator_step_from_remb() {
        let control = Arc::new(TrackControl::default());
        let ctrl = CongestionController::with_warmup(0xABCD, control.clone(), 0.0);
        assert_eq!(control.desired_step(), 0);
        ctrl.on_remb(900_000); // < 0.5x of step0's 2.5Mbit -> degrade
        assert_eq!(control.desired_step(), 1);
    }

    #[test]
    fn controller_only_acts_on_rr_blocks_for_our_ssrc() {
        let control = Arc::new(TrackControl::default());
        let ctrl = CongestionController::with_warmup(0xABCD, control.clone(), 0.0);
        // RR about a different ssrc is ignored.
        ctrl.on_rtcp(&rr_packet(1, 0x9999, 200, 50, 0));
        assert_eq!(control.desired_step(), 0);
        // RR about our ssrc with severe loss degrades.
        ctrl.on_rtcp(&rr_packet(1, 0xABCD, 200, 50, 0));
        assert_eq!(control.desired_step(), 1);
    }

    #[test]
    fn pli_coalesces_into_a_single_keyframe_request() {
        let control = Arc::new(TrackControl::default());
        let ctrl = CongestionController::with_warmup(1, control.clone(), 0.0);
        ctrl.on_pli();
        ctrl.on_pli();
        ctrl.on_pli();
        assert!(control.take_pli(), "a burst of PLIs is one pending request");
        assert!(!control.take_pli(), "taken exactly once");
    }
}
