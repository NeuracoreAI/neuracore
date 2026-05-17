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
//! Phase 4 (see `docs/data-daemon-rewrite.md`) intentionally carries every
//! lifecycle message — including the frame payloads used by the smoke-test
//! deliverable — over a single `[u8]` slice service. Phase 5 introduces the
//! typed per-resolution `frames/<WxH>` zero-copy services for real video
//! traffic; the envelope schema is designed so that addition does not require
//! a breaking wire change for the existing command/lifecycle messages.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// iceoryx2 service-name conventions shared by daemon and producer.
pub mod service_name {
    /// Pub/sub service carrying lifecycle envelopes
    /// (`start_recording`, `start_trace`, `end_trace`, `stop_recording`,
    /// `open_frame_stream`).
    ///
    /// The payload type is `[u8]`; each message is a JSON-encoded
    /// [`Envelope`](super::Envelope). Sub-phase 4f split scalar samples and
    /// video frames onto dedicated services for backpressure isolation; this
    /// service stays small and infrequent — at most one message per trace
    /// lifecycle transition.
    pub const COMMANDS: &str = "neuracore/data_daemon/commands";

    /// Pub/sub service carrying scalar / sensor frame payloads.
    ///
    /// Joint poses, IMU samples, language tokens, and other low-rate JSON
    /// payloads land here. The throughput bound is loose (sub-MB/s in
    /// practice), so JSON-encoded envelopes are fine; the dedicated service
    /// just keeps a slow consumer from back-pressuring lifecycle commands.
    pub const SCALARS: &str = "neuracore/data_daemon/scalars";

    /// Maximum size of a single command-stream sample.
    ///
    /// Sized for lifecycle envelopes (`StartRecording`, `StartTrace`,
    /// `EndTrace`, `StopRecording`, `OpenFrameStream`) — all under a few
    /// hundred bytes of JSON in practice. 8 MiB is deliberate headroom for
    /// the phase 4 smoke-test path where `Frame` envelopes also travel over
    /// `commands` (carrying small test payloads of a few KiB each); a real
    /// RGB frame would not fit because `serde_json::serialize_bytes` emits
    /// `[u8]` as a JSON array of integers (~3× expansion), pushing a 6 MiB
    /// raw frame past this cap. Phase 4h routes frame traffic onto the
    /// dedicated `frames/<WxH>` services with `loan_uninit` zero-copy, at
    /// which point this budget can drop to the lifecycle footprint.
    pub const COMMANDS_MAX_PAYLOAD_BYTES: usize = 8 * 1024 * 1024;

    /// Maximum size of a single scalar-stream sample.
    ///
    /// 1 MiB comfortably covers the largest joint pose or language-token
    /// payload observed in the integration tests, with headroom for the JSON
    /// envelope scaffolding.
    pub const SCALARS_MAX_PAYLOAD_BYTES: usize = 1024 * 1024;

    /// Bytes-per-pixel assumed for the per-resolution `frames/<WxH>`
    /// services. Phase 4 ships RGB only; RGBA / depth variants would extend
    /// this constant once they're modelled.
    pub const FRAME_BYTES_PER_PIXEL: usize = 3;

    /// Headroom added on top of `width * height * FRAME_BYTES_PER_PIXEL` for
    /// the JSON envelope scaffolding (variant tag, trace_id, timestamp).
    /// 1 KiB is comfortably above the empirical worst case of a few hundred
    /// bytes.
    pub const FRAME_ENVELOPE_OVERHEAD_BYTES: usize = 1024;

    /// Build the iceoryx2 service name for raw RGB frames of the given
    /// resolution.
    ///
    /// One service per `(width, height)` pair so the iceoryx2 publisher /
    /// subscriber sample size is fixed and the loan pool can be tuned
    /// per-resolution. The daemon opens these lazily on the first
    /// [`OpenFrameStream`](super::Envelope::OpenFrameStream) envelope it
    /// observes for a new resolution.
    pub fn frames(width: u32, height: u32) -> String {
        format!("neuracore/data_daemon/frames/{width}x{height}")
    }

    /// Maximum size of a single video-frame sample for the given resolution.
    pub fn frames_max_payload_bytes(width: u32, height: u32) -> usize {
        (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(FRAME_BYTES_PER_PIXEL)
            .saturating_add(FRAME_ENVELOPE_OVERHEAD_BYTES)
    }
}

/// A single message exchanged between the producer and the daemon.
///
/// Variants mirror the lifecycle described in §4 of the rewrite plan. The
/// internal-tagged representation (`#[serde(tag = "kind")]`) keeps the wire
/// format self-describing and forward-compatible — adding a new variant
/// doesn't displace existing fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Envelope {
    /// Producer announces a new recording session.
    StartRecording {
        /// Recording identifier supplied by the SDK.
        recording_id: String,
        /// Optional robot identifier.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        robot_id: Option<String>,
        /// Optional robot human-readable name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        robot_name: Option<String>,
        /// Optional dataset identifier.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dataset_id: Option<String>,
        /// Optional dataset human-readable name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
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
        #[serde(default)]
        timestamp_ns: i64,
        /// Opaque per-frame bytes.
        #[serde(with = "frame_payload")]
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
    /// Producer announces that it is about to publish video frames of
    /// `(width, height)` for `trace_id`.
    ///
    /// Sent on the [`COMMANDS`](service_name::COMMANDS) service so the daemon
    /// can lazily open the matching
    /// [`frames(width, height)`](service_name::frames) iceoryx2 service
    /// before the first frame arrives. The producer must wait for the daemon
    /// to acknowledge this (or, in phase 4, simply allow enough time for the
    /// daemon's listener tick) before publishing the first frame, otherwise
    /// the early samples land in the per-resolution service's queue with no
    /// subscriber attached and are dropped.
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
    /// Variant name used in tracing/logging.
    pub fn kind(&self) -> &'static str {
        match self {
            Envelope::StartRecording { .. } => "start_recording",
            Envelope::StartTrace { .. } => "start_trace",
            Envelope::Frame { .. } => "frame",
            Envelope::EndTrace { .. } => "end_trace",
            Envelope::StopRecording { .. } => "stop_recording",
            Envelope::OpenFrameStream { .. } => "open_frame_stream",
        }
    }

    /// Encode the envelope as a JSON byte vector ready for an iceoryx2 sample.
    pub fn encode(&self) -> Result<Vec<u8>, EnvelopeCodecError> {
        serde_json::to_vec(self).map_err(EnvelopeCodecError::Encode)
    }

    /// Decode an envelope from the byte slice carried in an iceoryx2 sample.
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvelopeCodecError> {
        serde_json::from_slice(bytes).map_err(EnvelopeCodecError::Decode)
    }
}

/// Frame-payload byte vector codec.
///
/// `serde_json` represents Rust byte slices as JSON arrays of integers, which
/// is wire-compatible but verbose. This shim makes the (de)serialize calls
/// explicit so a future migration to a binary envelope (e.g. msgpack) only
/// needs to touch this module.
mod frame_payload {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(bytes)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        // Matches what `serde_json` produces for `serialize_bytes` today: a
        // JSON array of integers, which `Vec::<u8>::deserialize` decodes
        // directly. A binary encoder that emits a true byte string would
        // need a custom `Visitor` here.
        Vec::<u8>::deserialize(deserializer)
    }
}

/// Errors raised while encoding or decoding an [`Envelope`].
#[derive(Debug, Error)]
pub enum EnvelopeCodecError {
    /// Failed to serialize the envelope to JSON.
    #[error("failed to encode envelope: {0}")]
    Encode(#[source] serde_json::Error),
    /// Failed to deserialize the envelope from JSON.
    #[error("failed to decode envelope: {0}")]
    Decode(#[source] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_round_trips_through_json() {
        let original = Envelope::StartTrace {
            recording_id: "rec-1".into(),
            trace_id: "trace-1".into(),
            data_type: "video".into(),
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn frame_envelope_preserves_payload_bytes() {
        let original = Envelope::Frame {
            trace_id: "trace-1".into(),
            timestamp_ns: 1_000_000,
            payload: vec![1, 2, 3, 4, 5, 6],
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn envelope_kind_label_is_stable() {
        let env = Envelope::StopRecording {
            recording_id: "rec-1".into(),
        };
        assert_eq!(env.kind(), "stop_recording");
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

    #[test]
    fn frame_service_name_is_resolution_keyed() {
        assert_eq!(
            service_name::frames(1920, 1080),
            "neuracore/data_daemon/frames/1920x1080"
        );
        assert_eq!(
            service_name::frames(256, 256),
            "neuracore/data_daemon/frames/256x256"
        );
    }

    #[test]
    fn frame_payload_budget_covers_rgb_plus_overhead() {
        let budget = service_name::frames_max_payload_bytes(1920, 1080);
        assert!(budget >= 1920 * 1080 * 3);
        assert!(budget <= 1920 * 1080 * 3 + 4096);
    }
}
