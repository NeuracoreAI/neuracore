//! Shared IPC wire format for the Neuracore data daemon.
//!
//! Both the daemon binary and the PyO3 producer crate
//! (`data_daemon_producer`) depend on this crate so they agree on:
//!
//! - the iceoryx2 service-name conventions ([`service_name`]),
//! - the [`Envelope`] enum carried over the `commands` service, and
//! - the helpers to (de)serialize that envelope to/from the byte slice payload
//!   iceoryx2 transports.
//!
//! Envelopes are encoded with [`postcard`], a compact length-prefixed binary
//! format. Payload bytes travel raw (length-prefix + bytes — no base64 or
//! `[u8]→[i32]` expansion that JSON would force), and `f64` fields round-trip
//! bit-exact because postcard writes the IEEE-754 byte pattern directly. The
//! schema is forward-compatible: postcard's enum representation tags variants
//! with a u32 discriminant, so new envelope variants append cleanly.
//!
//! Phase 4 carries every lifecycle message — including the frame payloads used
//! by the smoke-test deliverable — over the single `[u8]` slice `commands`
//! service. Phase 5 introduces typed per-resolution `frames/<WxH>` zero-copy
//! services for real video traffic; that addition does not require a breaking
//! change to the existing command/lifecycle messages.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// iceoryx2 service-name conventions shared by daemon and producer.
pub mod service_name {
    /// Pub/sub service carrying lifecycle envelopes
    /// (`start_recording`, `start_trace`, `end_trace`, `stop_recording`,
    /// `open_frame_stream`) and, in phase 4, every `frame` payload.
    pub const COMMANDS: &str = "neuracore/data_daemon/commands";

    /// Maximum size of a single command-stream sample.
    ///
    /// Sized so a 1920×1080 RGB24 frame (6,220,800 bytes) plus the postcard
    /// envelope scaffolding fits comfortably while still under iceoryx2's slice
    /// budget. Phase 4h ships per-resolution `frames/<WxH>` services for the
    /// real video path; once that lands this cap can drop back to the
    /// lifecycle-only footprint (~64 KiB).
    // TODO(sub-phase 4h): drop this back to lifecycle-only once the
    // per-resolution `frames/<WxH>` services take over the video path.
    pub const COMMANDS_MAX_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

    /// Subscriber buffer depth for the lifecycle service.
    ///
    /// iceoryx2's default subscriber buffer is 2 samples with safe-overflow
    /// enabled — fine for high-rate sensor streams where dropping old data is
    /// the right answer, but catastrophic for lifecycle envelopes. A single
    /// `nc.log_joint_positions` call publishes one `StartRecording` +
    /// `StartTrace` per joint plus one `Frame` per joint via the native
    /// producer (typically 14+ envelopes back-to-back); with the default the
    /// listener only ever sees the trailing two, and the earlier `StartTrace`
    /// envelopes are silently dropped — which would otherwise surface
    /// downstream as missing per-trace state on disk because the per-trace
    /// actor never learns the recording or data-type identifiers.
    ///
    /// 1024 covers the busiest burst observed in the data-integrity test
    /// matrix (multi-camera multi-joint recordings at 1000 Hz joint logging)
    /// with comfortable headroom; combined with iceoryx2's default
    /// `Block`-on-unable-to-deliver strategy this provides reliable in-order
    /// delivery for every lifecycle envelope.
    pub const LIFECYCLE_SUBSCRIBER_BUFFER_SIZE: usize = 1024;

    /// Maximum number of concurrent publishers per service.
    ///
    /// iceoryx2's default cap of 2 is unworkable for the SDK's threading
    /// model: the native producer parks its iceoryx2 publisher in a
    /// `thread_local!` (publishers are `!Sync`), so each Python OS thread
    /// that calls into the producer builds its own. The integration matrix
    /// fans up to ~32 worker threads (`parallel_contexts=8` × three joint
    /// roles + one RGB role) and the orchestrator thread also publishes
    /// lifecycle envelopes, comfortably exceeding the default. Hitting the
    /// cap surfaces as
    /// `PublisherCreateError::ExceedsMaxSupportedPublishers` from
    /// `publisher_builder().create()` and the SDK can't drain the trace.
    ///
    /// Both sides agree on this constant via `open_or_create`, so the first
    /// party in (the daemon at startup) seeds the service with the larger
    /// cap and the producer's later open observes the same attribute set.
    pub const MAX_PUBLISHERS_PER_SERVICE: usize = 128;
}

/// A single message exchanged between the producer and the daemon.
///
/// Variants mirror the lifecycle described in §4 of the rewrite plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Envelope {
    /// Producer announces a new recording session.
    StartRecording {
        /// Recording identifier supplied by the SDK.
        recording_id: String,
        /// Optional robot identifier.
        robot_id: Option<String>,
        /// Optional robot human-readable name.
        robot_name: Option<String>,
        /// Optional dataset identifier.
        dataset_id: Option<String>,
        /// Optional dataset human-readable name.
        dataset_name: Option<String>,
    },
    /// Producer opens a new trace within an active recording.
    StartTrace {
        /// Recording this trace belongs to.
        recording_id: String,
        /// Trace identifier supplied by the SDK.
        trace_id: String,
        /// Wire data-type label (e.g. `"video"`, `"joints"`).
        data_type: String,
        /// Optional per-stream label (e.g. joint name or camera id). When
        /// present it disambiguates traces that share a `data_type` — joint
        /// streams produce one trace per joint name, video streams one per
        /// camera id — so downstream tooling (and the integration test
        /// matrix) can identify the producing stream from the DB row alone.
        data_type_name: Option<String>,
    },
    /// Producer delivers one frame/sample for a trace.
    ///
    /// The payload is opaque to the IPC layer; the per-trace actor parses it
    /// according to `data_type`. Phase 4 only counts frames in the database;
    /// phase 5 wires the bytes into the JSON / NUT writers.
    Frame {
        /// Trace this frame belongs to.
        trace_id: String,
        /// Caller-supplied capture time in nanoseconds since the Unix epoch.
        timestamp_ns: i64,
        /// Optional caller-supplied capture time in seconds (f64) since the
        /// Unix epoch. Postcard writes this as 8 raw IEEE-754 bytes so the
        /// value round-trips bit-exact — required for the integration
        /// matrix's exact-match timestamp assertion on the video sidecar.
        timestamp_s: Option<f64>,
        /// Opaque per-frame bytes. Postcard transports these as
        /// length-prefix + raw bytes (no expansion).
        payload: Vec<u8>,
    },
    /// Producer closes a trace; no further frames will follow.
    EndTrace {
        /// Trace identifier supplied earlier in [`Envelope::StartTrace`].
        trace_id: String,
    },
    /// Producer requests the daemon stop accepting recording data.
    StopRecording {
        /// Recording identifier supplied earlier in
        /// [`Envelope::StartRecording`].
        recording_id: String,
    },
    /// Producer cancels a recording — the daemon drops every in-flight
    /// per-trace actor, deletes the on-disk artefacts, marks the recording
    /// row as cancelled, and does not upload any of its traces.
    ///
    /// Sent in lieu of (or after) a [`StopRecording`] envelope when the
    /// producer decides the recording should be discarded entirely. The
    /// daemon is idempotent: a CancelRecording after a successful upload is
    /// a no-op (the recording row already records what was uploaded).
    CancelRecording {
        /// Recording identifier supplied earlier in
        /// [`Envelope::StartRecording`].
        recording_id: String,
    },
    /// Producer announces that it is about to publish video frames of
    /// `(width, height)` for `trace_id`.
    ///
    /// Sent on the [`COMMANDS`](service_name::COMMANDS) service so the daemon
    /// can lazily open the matching per-resolution iceoryx2 service before
    /// the first frame arrives. In phase 4 the frame payloads also travel
    /// over `COMMANDS`; phase 4h moves them onto dedicated zero-copy
    /// services keyed by resolution.
    OpenFrameStream {
        /// Trace these frames belong to.
        trace_id: String,
        /// Frame width in pixels.
        width: u32,
        /// Frame height in pixels.
        height: u32,
    },
}

impl Envelope {
    /// Convenience constructor for [`Envelope::Frame`].
    pub fn frame(
        trace_id: String,
        timestamp_ns: i64,
        timestamp_s: Option<f64>,
        payload: Vec<u8>,
    ) -> Self {
        Envelope::Frame {
            trace_id,
            timestamp_ns,
            timestamp_s,
            payload,
        }
    }

    /// Variant name used in tracing/logging.
    pub fn kind(&self) -> &'static str {
        match self {
            Envelope::StartRecording { .. } => "start_recording",
            Envelope::StartTrace { .. } => "start_trace",
            Envelope::Frame { .. } => "frame",
            Envelope::EndTrace { .. } => "end_trace",
            Envelope::StopRecording { .. } => "stop_recording",
            Envelope::CancelRecording { .. } => "cancel_recording",
            Envelope::OpenFrameStream { .. } => "open_frame_stream",
        }
    }

    /// Encode the envelope as a postcard byte vector ready for an iceoryx2
    /// sample.
    pub fn encode(&self) -> Result<Vec<u8>, EnvelopeCodecError> {
        postcard::to_allocvec(self).map_err(EnvelopeCodecError::Encode)
    }

    /// Decode an envelope from the byte slice carried in an iceoryx2 sample.
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvelopeCodecError> {
        postcard::from_bytes(bytes).map_err(EnvelopeCodecError::Decode)
    }
}

/// Errors raised while encoding or decoding an [`Envelope`].
#[derive(Debug, Error)]
pub enum EnvelopeCodecError {
    /// Failed to serialize the envelope.
    #[error("failed to encode envelope: {0}")]
    Encode(#[source] postcard::Error),
    /// Failed to deserialize the envelope.
    #[error("failed to decode envelope: {0}")]
    Decode(#[source] postcard::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_round_trips_through_postcard() {
        let original = Envelope::StartTrace {
            recording_id: "rec-1".into(),
            trace_id: "trace-1".into(),
            data_type: "video".into(),
            data_type_name: Some("camera_0".into()),
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn frame_envelope_preserves_payload_bytes() {
        let original = Envelope::frame("trace-1".into(), 1_000_000, None, vec![1, 2, 3, 4, 5, 6]);
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn frame_timestamp_s_is_bit_exact_over_postcard_wire() {
        // Postcard writes `f64` as 8 raw IEEE-754 bytes, so values that
        // would shift under a decimal parser (e.g. `7/60`) round-trip
        // bit-identically — required for the integration matrix's
        // exact-match assertion on the video sidecar timestamps.
        let original = Envelope::frame(
            "trace-1".into(),
            116_666_666,
            Some(7.0_f64 / 60.0_f64),
            vec![0xAA, 0xBB],
        );
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        if let Envelope::Frame { timestamp_s, .. } = decoded {
            assert_eq!(
                timestamp_s.map(f64::to_bits),
                Some((7.0_f64 / 60.0_f64).to_bits()),
            );
        } else {
            panic!("decoded envelope was not a Frame");
        }
    }

    #[test]
    fn frame_payload_does_not_expand_under_postcard() {
        // The whole point of moving off JSON is that `Vec<u8>` no longer
        // expands ~3× as a JSON array of integers. Encode a 1 MiB payload
        // and check the wire form is within a small constant of the raw
        // bytes (variant tag + length prefix + trace_id + timestamps).
        const PAYLOAD_LEN: usize = 1024 * 1024;
        let original = Envelope::frame("trace-1".into(), 0, None, vec![0xAB; PAYLOAD_LEN]);
        let bytes = original.encode().expect("encode");
        // Empirically postcard adds well under 256 bytes of framing on top
        // of the raw payload for this envelope. Allow a generous 4 KiB
        // headroom so the test stays stable across minor postcard / serde
        // revisions without losing its teeth.
        assert!(
            bytes.len() <= PAYLOAD_LEN + 4096,
            "postcard wire form ({} bytes) is too far from raw payload ({} bytes)",
            bytes.len(),
            PAYLOAD_LEN,
        );
        assert!(
            bytes.len() >= PAYLOAD_LEN,
            "wire form must contain the raw bytes"
        );
    }

    #[test]
    fn envelope_kind_label_is_stable() {
        let env = Envelope::StopRecording {
            recording_id: "rec-1".into(),
        };
        assert_eq!(env.kind(), "stop_recording");
    }

    #[test]
    fn cancel_recording_round_trips() {
        let original = Envelope::CancelRecording {
            recording_id: "rec-1".into(),
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        assert_eq!(original.kind(), "cancel_recording");
    }

    #[test]
    fn open_frame_stream_round_trips() {
        let original = Envelope::OpenFrameStream {
            trace_id: "trace-1".into(),
            width: 1920,
            height: 1080,
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        assert_eq!(original.kind(), "open_frame_stream");
    }
}
