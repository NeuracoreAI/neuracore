//! Shared transport plumbing for the producer and consumer peers.
//!
//! Both peers wrap a libdatachannel [`RtcPeerConnection`] (via datachannel-rs)
//! and translate between its callback surface and the synchronous, queue-backed
//! Python API. This module holds the pieces both sides need:
//!
//! - [`loopback_config`] — the ICE configuration used in-process (host
//!   candidates only; no STUN/TURN).
//! - SDP / state translation ([`parse_session`], [`sdp_type_str`],
//!   [`connection_state_str`], [`reliability_kind_hint`]).
//! - [`ManifestState`] — the control-channel manifest model (a flat map keyed by
//!   data-channel label now, video-track mid later) and its JSON rendering.
//!
//! libdatachannel refuses (`outgoing()` throws) any message sent before a data
//! channel's SCTP stream is open, so the producer buffers outgoing bytes per
//! channel until `on_open` fires and then flushes them in order; that send gate
//! lives in [`crate::producer`] next to the channels it owns.
//!
//! All shared state is interior-mutable behind `Arc`/`Mutex` so the libdatachannel
//! callback threads and the Python-facing methods can touch it concurrently while
//! the peers stay `Send + Sync`.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use datachannel::{
    ConnectionState, DataChannelHandler, PeerConnectionHandler, Reliability, RtcConfig,
    RtcPeerConnection, SdpType, SessionDescription,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;
use serde_json::{json, Map, Value};

/// Recover the raw libdatachannel peer-connection id that the sys-level calls
/// (`rtcSetTrackCallback`, `rtcAddTrackEx`) need. datachannel-rs 0.16 surfaces it
/// only as the opaque `PeerConnectionId`, whose inner `i32` is private; its
/// derived `Debug` renders as `PeerConnectionId(<n>)`, so we parse the integer
/// out. The alternative is forking the binding; this keeps the workaround
/// contained to one function (used by both peers). If parsing ever fails, the
/// caller disables its sys-level path gracefully.
pub(crate) fn raw_pc_id<P>(pc: &RtcPeerConnection<P>) -> Option<i32>
where
    P: PeerConnectionHandler + Send,
    P::DCH: DataChannelHandler + Send,
{
    let dbg = format!("{:?}", pc.id());
    let start = dbg.find('(')? + 1;
    let end = dbg.rfind(')')?;
    dbg.get(start..end)?.trim().parse().ok()
}

/// The reserved control-channel label. It carries the manifest and is never
/// itself listed in the manifest nor surfaced to the consumer as a data channel.
pub(crate) const CONTROL_LABEL: &str = "control";

/// Stderr trace of a transport event, gated on `NEURACORE_WEBRTC_DEBUG`. Used to
/// time the ICE/DTLS phases when diagnosing connect latency; a no-op otherwise.
pub(crate) fn debug_trace(peer: &str, what: &str) {
    if std::env::var_os("NEURACORE_WEBRTC_DEBUG").is_some() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        eprintln!("[ncwebrtc {now:.3}] {peer} {what}");
    }
}

/// Recover a mutex guard even if a holder panicked mid-update. A poisoned lock
/// here only means a callback thread panicked while pushing; keeping delivery
/// alive beats cascading the panic across the FFI boundary.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Map a libdatachannel error into a Python exception.
pub(crate) fn map_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyRuntimeError::new_err(format!("datachannel error: {err}"))
}

/// In-process ICE configuration: no ICE servers (host candidates only) and the
/// ICE agent bound to a single address so exactly one host candidate is
/// gathered. Two peers in the same process then connect over that one address
/// with no STUN/TURN round trip.
///
/// Binding matters: left unbound, libdatachannel gathers a candidate per local
/// interface and ICE tries the highest-priority pair first. In containers the
/// highest-priority candidate is often an unreachable IPv6 ULA, so the agent
/// stalls a full second on the STUN retransmit before falling back to IPv4 —
/// blowing the connect SLO. Binding to one reachable address removes the dud
/// pair. The address is overridable via `NEURACORE_WEBRTC_BIND_ADDRESS` (default
/// `127.0.0.1`, correct for the in-process peers this PR ships; the production
/// cutover supplies its own RtcConfig with real ICE servers).
///
/// `force_media_transport` is mandatory from PR3 on **both** peers. libdatachannel
/// only stands up the DTLS-SRTP transport when the initial connection already has
/// media or this flag is set; otherwise a track added by a *later* renegotiation
/// hits `iterateRemoteTracks` with no SRTP transport and the track is errored
/// ("The connection has no media transport" — see libdatachannel
/// `impl/peerconnection.cpp`). Forcing it up front means the first video-track add
/// reuses the existing BUNDLE transport with no second DTLS handshake, so
/// connect-latency is paid once during bootstrap and `connect_ms` does not regress.
pub(crate) fn loopback_config() -> RtcConfig {
    let no_servers: [&str; 0] = [];
    RtcConfig::new(&no_servers)
        .bind_address(&bind_address())
        .force_media_transport()
}

/// The single ICE bind address used in-process: `NEURACORE_WEBRTC_BIND_ADDRESS`
/// if set, else `127.0.0.1`. Split out of [`loopback_config`] so the selection
/// is unit-testable without building an `RtcConfig` (which would pull in the ICE
/// agent and bind a socket).
pub(crate) fn bind_address() -> String {
    std::env::var("NEURACORE_WEBRTC_BIND_ADDRESS").unwrap_or_else(|_| "127.0.0.1".to_string())
}

/// The wire `sdp_type` string for an SDP. Mirrors datachannel-rs's private
/// `SdpType::val`, which we cannot call.
pub(crate) fn sdp_type_str(sdp_type: &SdpType) -> &'static str {
    match sdp_type {
        SdpType::Answer => "answer",
        SdpType::Offer => "offer",
        SdpType::Pranswer => "pranswer",
        SdpType::Rollback => "rollback",
    }
}

/// Whether outgoing offers should be munged for Chrome's stricter SDP parser.
/// Gated by `NCD_WEBRTC_CHROME_SDP` so the libdatachannel-to-libdatachannel
/// loopback path (which parses the bare `a=ssrc` line fine) is untouched.
pub(crate) fn chrome_sdp_enabled() -> bool {
    std::env::var_os("NCD_WEBRTC_CHROME_SDP").is_some()
}

/// Make a libdatachannel offer acceptable to Chrome's stricter SDP parser by
/// giving every bare `a=ssrc:<n>` line a `cname` attribute.
///
/// libdatachannel emits a bare `a=ssrc:<id>` (no source attribute); Chrome rejects
/// it ("a=ssrc Expects 2 fields") and never sets up the receive track, so the
/// producer's H.264 never reaches the decoder. RFC 5576 requires the SSRC carry at
/// least a `cname`. We append `cname:<value>` (the packetizer's RTCP CNAME, so the
/// SDP and the RTCP SR agree) to any bare `a=ssrc:` line, leaving an already-
/// qualified line (`a=ssrc:<n> <attr>...`) untouched and idempotent. Pure so it is
/// unit-testable without a peer; applied only when [`chrome_sdp_enabled`].
pub(crate) fn munge_ssrc_cname(sdp: &str, cname: &str) -> String {
    // Preserve the original line endings: SDP is CRLF on the wire, but the munge
    // must not rewrite an `\n`-only document into CRLF (or vice versa).
    let mut out = String::with_capacity(sdp.len() + 32);
    let mut rest = sdp;
    while let Some(nl) = rest.find('\n') {
        let (line_with_cr, tail) = rest.split_at(nl + 1);
        out.push_str(&munge_ssrc_line(line_with_cr, cname));
        rest = tail;
    }
    if !rest.is_empty() {
        out.push_str(&munge_ssrc_line(rest, cname));
    }
    out
}

/// Munge a single SDP line (which may carry a trailing `\r\n`/`\n`). Appends
/// ` cname:<cname>` to a bare `a=ssrc:<n>` line; everything else passes through.
fn munge_ssrc_line(line: &str, cname: &str) -> String {
    let trimmed = line.trim_end_matches(['\r', '\n']);
    let eol = &line[trimmed.len()..];
    let Some(value) = trimmed.strip_prefix("a=ssrc:") else {
        return line.to_string();
    };
    // Already qualified (`a=ssrc:<n> cname:...`) -> leave it alone (idempotent).
    if value.split_whitespace().count() != 1 {
        return line.to_string();
    }
    format!("a=ssrc:{value} cname:{cname}{eol}")
}

// ---------------------------------------------------------------------------
// Reconnect / ICE-restart decision (a pure seam)
// ---------------------------------------------------------------------------

/// Whether this binding can perform an ICE restart. **It cannot.** libdatachannel
/// builds ICE on libjuice, whose agent is single-shot: it cannot regather with new
/// credentials, so there is no `restart_ice`/`setLocalDescription({iceRestart})`
/// path (upstream libjuice #130; recorded in `reports/PR8-*` from the spike line).
/// On a Disconnected/Failed connection the peer therefore surfaces a clear
/// reconnect-needed signal and the application removes and re-adds the peer (for a
/// broadcaster, the single failed consumer) rather than attempting an in-place
/// restart. This constant documents the binding capability in one place; if a
/// future libdatachannel/libjuice gains real ICE restart, flip it and
/// [`reconnect_action`] starts returning [`ReconnectAction::IceRestart`].
pub(crate) const ICE_RESTART_SUPPORTED: bool = false;

/// What to do when a peer connection changes state, given whether the binding can
/// restart ICE. The decision is pure so it is unit-tested behind a fake without a
/// live peer (see the reconnect-decision test).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReconnectAction {
    /// A healthy/transitional state: do nothing.
    None,
    /// Disconnected/Failed and the binding supports ICE restart: attempt a bounded
    /// in-place restart. Unreachable while [`ICE_RESTART_SUPPORTED`] is false.
    IceRestart,
    /// Disconnected/Failed and the binding cannot restart ICE: surface a
    /// reconnect-needed error so the app tears down and re-adds the peer/consumer.
    SurfaceReconnect,
}

/// Map a connection state to a reconnect action. `Failed` and `Disconnected` are
/// the only states that need recovery; everything else is `None`.
pub(crate) fn reconnect_action(state: ConnectionState, ice_restart_supported: bool) -> ReconnectAction {
    match state {
        ConnectionState::Failed | ConnectionState::Disconnected => {
            if ice_restart_supported {
                ReconnectAction::IceRestart
            } else {
                ReconnectAction::SurfaceReconnect
            }
        }
        _ => ReconnectAction::None,
    }
}

/// The lowercase wire string for a connection state (the `on_state` payload).
pub(crate) fn connection_state_str(state: &ConnectionState) -> &'static str {
    match state {
        ConnectionState::New => "new",
        ConnectionState::Connecting => "connecting",
        ConnectionState::Connected => "connected",
        ConnectionState::Disconnected => "disconnected",
        ConnectionState::Failed => "failed",
        ConnectionState::Closed => "closed",
    }
}

/// A coarse reliability hint surfaced with a newly observed data channel. All
/// channels in this PR are reliable-ordered, but we report what the channel
/// actually negotiated so the hint stays honest if that changes.
pub(crate) fn reliability_kind_hint(reliability: &Reliability) -> String {
    if reliability.unreliable {
        "unreliable".to_string()
    } else if reliability.unordered {
        "unordered".to_string()
    } else {
        "reliable".to_string()
    }
}

/// Parse a wire SDP string into the [`SessionDescription`] datachannel-rs wants
/// for `set_remote_description`. The SDP round-trips through webrtc_sdp's
/// parser, so it is semantically — not byte — faithful, which is fine when both
/// peers are libdatachannel.
pub(crate) fn parse_session(sdp: &str, sdp_type: SdpType) -> Result<SessionDescription, PyErr> {
    let parsed = datachannel::sdp::parse_sdp(sdp, false)
        .map_err(|e| PyValueError::new_err(format!("invalid SDP: {e}")))?;
    Ok(SessionDescription {
        sdp: parsed,
        sdp_type,
    })
}

/// The control-channel manifest: the producer's published view of the streams a
/// consumer can expect on this connection.
///
/// It is a flat JSON object keyed by stream identity — data-channel **label**
/// now, video-track **mid** once PR4 adds tracks — so a consumer reads the whole
/// stream set from the object's keys. Each value is a small descriptor object
/// carrying a `type` discriminator plus type-specific fields. The control
/// channel itself is never an entry. See `reports/PR2-data-path.md` for the
/// schema PR4 extends.
#[derive(Default)]
pub(crate) struct ManifestState {
    entries: BTreeMap<String, Value>,
}

impl ManifestState {
    /// Insert or replace a data-channel entry keyed by its label.
    pub(crate) fn upsert_data_channel(&mut self, label: &str, kind: &str) {
        self.entries.insert(
            label.to_string(),
            json!({ "type": "data_channel", "kind": kind }),
        );
    }

    /// Insert or replace a video-track entry keyed by its negotiated `mid`. The
    /// shape is the one PR2 reserved and the video test asserts: `mid` is the
    /// key, the descriptor carries `type: "video_track"`, the producer-side
    /// `track_id`, and the `mid` itself.
    pub(crate) fn upsert_video_track(&mut self, mid: &str, track_id: &str) {
        self.entries.insert(
            mid.to_string(),
            json!({ "type": "video_track", "track_id": track_id, "mid": mid }),
        );
    }

    /// Remove an entry by its key (a data-channel label or a video-track mid).
    pub(crate) fn remove_entry(&mut self, key: &str) {
        self.entries.remove(key);
    }

    /// Render the current manifest as a JSON object string.
    pub(crate) fn to_json(&self) -> String {
        let object: Map<String, Value> = self
            .entries
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        serde_json::to_string(&Value::Object(object)).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(test)]
mod tests {
    //! Peer-free unit tests for the transport translation and the manifest
    //! model. None of these touch a `PeerConnection`, a socket, or the GIL: they
    //! pin the pure SDP/state/reliability translation and the control-channel
    //! manifest JSON schema deterministically.

    use super::*;
    use datachannel::{IceCandidate, Reliability};
    use std::collections::BTreeSet;

    // The SDP tests drive `datachannel::sdp::parse_sdp` directly (the same parser
    // `parse_session` wraps) rather than `parse_session` itself: the wrapper
    // returns a `PyErr`, and referencing pyo3's runtime from a `cargo test`
    // binary would need libpython linked (the crate is an `extension-module`).
    // The wrapper is a one-line `map_err`; the semantics under test are the
    // parser's round trip, exercised here without the interpreter.

    fn reliability(unordered: bool, unreliable: bool) -> Reliability {
        Reliability {
            unordered,
            unreliable,
            max_packet_life_time: 0,
            max_retransmits: 0,
        }
    }

    // --- Chrome a=ssrc cname munge -------------------------------------------

    #[test]
    fn munge_adds_cname_to_a_bare_ssrc_line() {
        let sdp = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=ssrc:1\r\n";
        let out = munge_ssrc_cname(sdp, "neuracore");
        assert!(out.contains("a=ssrc:1 cname:neuracore\r\n"), "{out}");
        // Untouched lines pass through unchanged, CRLF preserved.
        assert!(out.starts_with("v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n"));
    }

    #[test]
    fn munge_is_idempotent_and_leaves_qualified_ssrc_lines() {
        // An already-qualified ssrc line (Chrome's own, or a second munge pass) is
        // left exactly as-is.
        let already = "a=ssrc:1 cname:neuracore\r\n";
        assert_eq!(munge_ssrc_cname(already, "neuracore"), already);
        let other_attr = "a=ssrc:42 msid:stream track\r\n";
        assert_eq!(munge_ssrc_cname(other_attr, "neuracore"), other_attr);
    }

    #[test]
    fn munge_preserves_lf_only_documents_and_the_final_unterminated_line() {
        // `\n`-only input stays `\n`-only (no spurious CR), and a trailing line
        // without an EOL is still munged.
        let sdp = "a=ssrc:7\nc=IN IP4 0.0.0.0\na=ssrc:8";
        let out = munge_ssrc_cname(sdp, "nc");
        assert_eq!(out, "a=ssrc:7 cname:nc\nc=IN IP4 0.0.0.0\na=ssrc:8 cname:nc");
    }

    // --- SDP / state / reliability translation -------------------------------

    #[test]
    fn sdp_type_maps_to_wire_strings() {
        assert_eq!(sdp_type_str(&SdpType::Offer), "offer");
        assert_eq!(sdp_type_str(&SdpType::Answer), "answer");
        assert_eq!(sdp_type_str(&SdpType::Pranswer), "pranswer");
        assert_eq!(sdp_type_str(&SdpType::Rollback), "rollback");
    }

    #[test]
    fn connection_state_maps_to_on_state_strings() {
        assert_eq!(connection_state_str(&ConnectionState::New), "new");
        assert_eq!(connection_state_str(&ConnectionState::Connecting), "connecting");
        assert_eq!(connection_state_str(&ConnectionState::Connected), "connected");
        assert_eq!(
            connection_state_str(&ConnectionState::Disconnected),
            "disconnected"
        );
        assert_eq!(connection_state_str(&ConnectionState::Failed), "failed");
        assert_eq!(connection_state_str(&ConnectionState::Closed), "closed");
    }

    // --- reconnect / ICE-restart decision ------------------------------------

    #[test]
    fn reconnect_action_surfaces_when_ice_restart_is_unsupported() {
        // The binding cannot restart ICE (libjuice single-shot agent), so a
        // failed/disconnected connection surfaces reconnect-needed.
        assert_eq!(
            reconnect_action(ConnectionState::Failed, false),
            ReconnectAction::SurfaceReconnect
        );
        assert_eq!(
            reconnect_action(ConnectionState::Disconnected, false),
            ReconnectAction::SurfaceReconnect
        );
        // Healthy/transitional states need no recovery.
        for state in [
            ConnectionState::New,
            ConnectionState::Connecting,
            ConnectionState::Connected,
            ConnectionState::Closed,
        ] {
            assert_eq!(reconnect_action(state, false), ReconnectAction::None);
        }
    }

    #[test]
    fn reconnect_action_would_restart_if_the_binding_supported_it() {
        // Behind the fake "supported" flag the same states choose an in-place ICE
        // restart — the branch that goes live only if libjuice ever gains it.
        assert_eq!(
            reconnect_action(ConnectionState::Failed, true),
            ReconnectAction::IceRestart
        );
        assert_eq!(
            reconnect_action(ConnectionState::Disconnected, true),
            ReconnectAction::IceRestart
        );
        // The shipped binding pins the unsupported path.
        assert!(!ICE_RESTART_SUPPORTED);
    }

    #[test]
    fn reliability_kind_hint_reports_what_was_negotiated() {
        assert_eq!(reliability_kind_hint(&reliability(false, false)), "reliable");
        assert_eq!(reliability_kind_hint(&reliability(true, false)), "unordered");
        // unreliable wins over unordered (it is checked first).
        assert_eq!(
            reliability_kind_hint(&reliability(true, true)),
            "unreliable"
        );
    }

    // --- SDP round trip ------------------------------------------------------

    /// A minimal but valid data-channel offer SDP. `parse_sdp(.., false)` is the
    /// lenient (non-local) mode the consumer/producer use for remote SDPs.
    const SAMPLE_SDP: &str = "v=0\r\n\
o=- 0 0 IN IP4 127.0.0.1\r\n\
s=-\r\n\
c=IN IP4 127.0.0.1\r\n\
t=0 0\r\n\
m=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n\
a=mid:0\r\n\
a=sctp-port:5000\r\n";

    #[test]
    fn parse_sdp_round_trip_is_semantically_faithful() {
        let parsed = datachannel::sdp::parse_sdp(SAMPLE_SDP, false)
            .expect("sample data-channel SDP should parse");
        // This is exactly what parse_session wraps into a SessionDescription;
        // the sdp_type comes from the caller, not the wire SDP.
        let session = SessionDescription {
            sdp: parsed,
            sdp_type: SdpType::Offer,
        };
        assert_eq!(sdp_type_str(&session.sdp_type), "offer");

        // Render back out and re-parse: a faithful round trip parses again and
        // preserves the application m-line and the origin address.
        let rendered = session.sdp.to_string();
        assert!(
            rendered.contains("m=application"),
            "data m-line lost on round trip: {rendered}"
        );
        assert!(
            rendered.contains("IN IP4 127.0.0.1"),
            "origin address lost on round trip: {rendered}"
        );
        datachannel::sdp::parse_sdp(&rendered, false)
            .expect("rendered SDP should parse again (idempotent)");
    }

    #[test]
    fn parse_sdp_rejects_garbage() {
        // parse_session maps this Err into a ValueError for callers; here we pin
        // the underlying rejection.
        assert!(datachannel::sdp::parse_sdp("not an sdp at all", false).is_err());
    }

    // --- ICE candidate parse/format -----------------------------------------

    #[test]
    fn ice_candidate_carries_candidate_and_mid_faithfully() {
        let wire = "candidate:1 1 udp 2113937151 127.0.0.1 54321 typ host";
        let candidate = IceCandidate {
            candidate: wire.to_string(),
            mid: "0".to_string(),
        };
        // Both fields survive construction (the shape on_candidate emits and
        // add_remote_candidate consumes).
        assert_eq!(candidate.candidate, wire);
        assert_eq!(candidate.mid, "0");
    }

    // --- ManifestState schema ------------------------------------------------

    fn manifest_object(state: &ManifestState) -> Map<String, Value> {
        serde_json::from_str::<Value>(&state.to_json())
            .expect("manifest renders valid JSON")
            .as_object()
            .expect("manifest is a flat JSON object")
            .clone()
    }

    #[test]
    fn data_channel_entry_matches_the_fixed_schema() {
        let mut state = ManifestState::default();
        state.upsert_data_channel("telemetry", "reliable");
        let object = manifest_object(&state);
        // Keyed by label; descriptor carries the type discriminator + kind.
        assert_eq!(
            object.get("telemetry"),
            Some(&json!({ "type": "data_channel", "kind": "reliable" }))
        );
    }

    #[test]
    fn video_track_entry_matches_the_fixed_schema() {
        let mut state = ManifestState::default();
        state.upsert_video_track("v0", "wrist_cam");
        let object = manifest_object(&state);
        // Keyed by mid; descriptor carries type, track_id, and the mid itself.
        assert_eq!(
            object.get("v0"),
            Some(&json!({ "type": "video_track", "track_id": "wrist_cam", "mid": "v0" }))
        );
    }

    #[test]
    fn manifest_is_a_flat_object_with_no_envelope_or_version_key() {
        let mut state = ManifestState::default();
        state.upsert_data_channel("telemetry", "reliable");
        state.upsert_video_track("v0", "wrist_cam");
        state.upsert_data_channel("joints", "reliable");
        let object = manifest_object(&state);
        assert_eq!(
            object.keys().cloned().collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "joints".to_string(),
                "telemetry".to_string(),
                "v0".to_string(),
            ])
        );
        // No top-level envelope/version key: the keys ARE the stream set.
        assert!(!object.contains_key("version"));
        assert!(!object.contains_key("streams"));
    }

    #[test]
    fn control_is_never_a_manifest_entry() {
        // Mirror the producer's rule: the control channel carries the manifest
        // and is never itself listed in it.
        let mut state = ManifestState::default();
        for label in ["control", "telemetry", "joints"] {
            if label != CONTROL_LABEL {
                state.upsert_data_channel(label, "reliable");
            }
        }
        assert!(!manifest_object(&state).contains_key(CONTROL_LABEL));
    }

    #[test]
    fn remove_entry_drops_only_the_keyed_stream() {
        let mut state = ManifestState::default();
        state.upsert_data_channel("telemetry", "reliable");
        state.upsert_video_track("v0", "wrist_cam");
        state.remove_entry("v0");
        let object = manifest_object(&state);
        assert!(!object.contains_key("v0"));
        assert!(object.contains_key("telemetry"));
    }

    #[test]
    fn to_json_republishes_the_full_state_every_call() {
        // Each render is an atomic full-state message: every entry, every time.
        let mut state = ManifestState::default();
        state.upsert_data_channel("a", "reliable");
        assert_eq!(manifest_object(&state).len(), 1);
        state.upsert_data_channel("b", "reliable");
        let object = manifest_object(&state);
        assert_eq!(object.len(), 2);
        assert!(object.contains_key("a") && object.contains_key("b"));
    }

    // --- bind-address selection ----------------------------------------------

    #[test]
    fn bind_address_defaults_to_loopback_and_honours_the_override() {
        // This is the only test that touches NEURACORE_WEBRTC_BIND_ADDRESS, so
        // the set/remove cannot race another test reading the same variable.
        std::env::remove_var("NEURACORE_WEBRTC_BIND_ADDRESS");
        assert_eq!(bind_address(), "127.0.0.1");
        std::env::set_var("NEURACORE_WEBRTC_BIND_ADDRESS", "10.1.2.3");
        assert_eq!(bind_address(), "10.1.2.3");
        std::env::remove_var("NEURACORE_WEBRTC_BIND_ADDRESS");
    }
}
