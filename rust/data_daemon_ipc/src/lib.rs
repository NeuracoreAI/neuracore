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
//! All envelopes — lifecycle, joints/scalars, and the chunk-ready notifications
//! for video traces — travel over a single `commands` service. Video pixel
//! buffers themselves are *not* on the IPC bus: the producer spools them to
//! disk as NUT chunks and announces each finished chunk with a
//! [`Envelope::VideoChunkReady`] envelope. See [`service_name`].

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// iceoryx2 service-name conventions shared by daemon and producer.
pub mod service_name {
    /// Pub/sub service carrying every IPC envelope: lifecycle
    /// (`start_recording`, `start_trace`, `stop_recording`,
    /// `cancel_recording`), non-video `frame` / `batched_frames` envelopes
    /// (joints, scalars, custom streams), and the
    /// [`Envelope::VideoChunkReady`] notifications that hand off
    /// disk-spooled video chunks to the daemon.
    ///
    /// There is no longer a dedicated video service — the producer writes
    /// pixel data straight to disk, so the IPC bus only ever carries
    /// metadata-sized payloads.
    pub const COMMANDS: &str = "neuracore/data_daemon/commands";

    /// Maximum size of a single `commands`-service sample.
    ///
    /// All envelope payloads are now metadata-sized: non-video frames are
    /// small JSON, the integration matrix's 1000-joint batch encodes to
    /// ~90 KiB, and `VideoChunkReady`'s `frame_timestamps_s` vector is
    /// ~30 KiB even for a 128 MiB 1080p chunk. 1 MiB leaves generous
    /// headroom for the worst case.
    pub const COMMANDS_MAX_PAYLOAD_BYTES: usize = 1024 * 1024;

    /// Subscriber buffer depth for the lifecycle service.
    ///
    /// Lossless, in-order delivery is *not* a function of this depth: the
    /// service is opened with `enable_safe_overflow(false)`, so a full
    /// buffer makes the producer's `Block` strategy wait rather than silently
    /// evict the oldest sample. (Were overflow left at iceoryx2's default the
    /// oldest sample — typically a `StartTrace` — would be dropped, stranding
    /// the per-trace actor.) The depth therefore only trades producer-blocking
    /// frequency against memory.
    ///
    /// The depth is bounded from *above* by memory, not just throughput.
    /// iceoryx2 sizes a publisher's data segment as
    /// `max_subscribers × (buffer + borrowed) × initial_max_slice_len`, and
    /// the resident footprint is `buffer × actual_sample_size`. The largest
    /// `commands` sample is now a [`Envelope::BatchedFrames`] envelope — the
    /// integration matrix's 1000-joint worst case encodes to ~90 KiB — so a
    /// 1024-deep buffer would retain ~94 MiB of pages per publisher and
    /// exhaust the 64 MiB devcontainer `/dev/shm`.
    ///
    /// 64 keeps that worst case at ~6 MiB per publisher while staying deep
    /// enough for steady state: the daemon drains every 10 ms and batched
    /// joint logging emits one envelope per timestep, so the buffer never
    /// fills under normal load. The one-time `StartTrace` burst at recording
    /// setup (one envelope per trace) briefly blocks the producer once the
    /// buffer is full — acceptable for a setup-path cost.
    pub const LIFECYCLE_SUBSCRIBER_BUFFER_SIZE: usize = 64;

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

    /// Maximum number of concurrent subscribers per service.
    ///
    /// The daemon opens exactly one subscriber per service; producers never
    /// subscribe. iceoryx2 sizes every publisher's data segment as
    /// `max_subscribers × (buffer + borrowed) × slice`, so the default of 8
    /// inflates each segment 8× for subscribers that never exist. Pinning
    /// this to 1 keeps the segment proportional to the real topology.
    pub const MAX_SUBSCRIBERS_PER_SERVICE: usize = 1;

    /// Maximum number of concurrent iceoryx2 nodes attached to any service.
    ///
    /// One node is built per **thread** (the `thread_local!` PRODUCER slot in
    /// the native producer). The integration matrix fans to 8 parallel worker
    /// subprocesses each running 5+ threads (main + RGB + joint roles), giving
    /// 40+ nodes plus the daemon. 512 gives enough headroom that the cap is
    /// never approached in any test configuration.
    ///
    /// The scalable fix (one node shared per process, not per thread) is tracked
    /// separately and would reduce the live count to single digits.
    pub const MAX_NODES_PER_SERVICE: usize = 512;
}

/// A single message exchanged between the producer and the daemon.
///
/// Variants cover the full recording / trace lifecycle.
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
    /// according to `data_type` and writes it through the JSON writer.
    ///
    /// Video frames do *not* travel as `Frame` envelopes — they are spooled
    /// to disk by the producer and announced via
    /// [`Envelope::VideoChunkReady`] instead.
    Frame {
        /// Trace this frame belongs to.
        trace_id: String,
        /// Zero-based per-trace counter, assigned by the producer's per-trace
        /// `AtomicU64` at publish time. The producer's
        /// [`Envelope::StopRecording`] payload reports the post-stop counter
        /// value (== total envelopes sent) per trace; the daemon's per-trace
        /// actor counts received envelopes and finalises when the count
        /// matches that promised total, closing the historical race where
        /// `EndTrace` from one publisher could arrive before in-flight
        /// `Frame`s from another.
        sequence_number: u64,
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
    /// Producer delivers one sample for each of several traces captured at
    /// the same instant — used by scalar joint logging, where a robot's N
    /// joints are sampled together.
    ///
    /// Collapsing N [`Envelope::Frame`] envelopes into one IPC message cuts
    /// the per-call iceoryx2 publish count (and the pressure on the deep
    /// lifecycle buffer) by a factor of N. The daemon's listener unpacks each
    /// item into a standalone [`Envelope::Frame`] before dispatch, so nothing
    /// downstream of the wire boundary has to know this variant exists.
    ///
    /// Joint payloads are tiny (a `{"timestamp":..,"value":..}` JSON object,
    /// ~50 bytes) so even the integration matrix's 1000-joint worst case
    /// encodes to ~90 KiB — comfortably inside [`COMMANDS_MAX_PAYLOAD_BYTES`].
    ///
    /// [`COMMANDS_MAX_PAYLOAD_BYTES`]: service_name::COMMANDS_MAX_PAYLOAD_BYTES
    BatchedFrames {
        /// Capture time in nanoseconds since the Unix epoch, shared by every
        /// item in the batch.
        timestamp_ns: i64,
        /// Optional capture time in seconds since the Unix epoch, shared by
        /// every item. See [`Envelope::Frame`]'s `timestamp_s` for the
        /// bit-exactness contract postcard provides.
        timestamp_s: Option<f64>,
        /// Per-trace samples; each unpacks into one [`Envelope::Frame`].
        frames: Vec<BatchedFrameItem>,
    },
    /// Producer requests the daemon stop accepting recording data and reports
    /// how many envelopes were sent for every trace the recording minted.
    ///
    /// This envelope replaces the legacy per-trace `EndTrace`: each trace's
    /// promised envelope count gates finalisation in the daemon, eliminating
    /// the race where an `EndTrace` published from the stop thread could
    /// reach the daemon before in-flight `Frame`s from another publisher
    /// thread.
    StopRecording {
        /// Recording identifier supplied earlier in
        /// [`Envelope::StartRecording`].
        recording_id: String,
        /// One [`TraceEnding`] per trace the producer minted within this
        /// recording. Empty when the producer minted no traces.
        trace_endings: Vec<TraceEnding>,
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
    /// Producer announces a finished NUT chunk for a video trace.
    ///
    /// The producer spools captured RGB frames to disk as a sequence of NUT
    /// chunks. When a chunk crosses the flush threshold (or the trace ends)
    /// the producer renames `chunk_NNNN.nut.tmp` → `chunk_NNNN.nut`, then
    /// publishes this envelope so the daemon can encode the chunk to a
    /// sealed MP4 segment. Per-frame `timestamp_s` values are carried
    /// inline so the daemon-side `trace.json` sidecar matches the bit-exact
    /// assertion that today travels through [`Envelope::Frame::timestamp_s`].
    VideoChunkReady {
        /// Recording this trace belongs to. Carried so the daemon can resolve
        /// the on-disk path without a roundtrip through `StartTrace`.
        recording_id: String,
        /// Trace these frames belong to.
        trace_id: String,
        /// Zero-based chunk index within the trace.
        chunk_index: u32,
        /// Frame width in pixels (constant across a trace).
        width: u32,
        /// Frame height in pixels (constant across a trace).
        height: u32,
        /// Size of the NUT file in bytes.
        byte_count: u64,
        /// Number of frames packed into this chunk.
        frame_count: u32,
        /// Per-frame `timestamp_s` (Unix seconds, f64) in arrival order.
        /// Length equals `frame_count`; values round-trip bit-exact through
        /// postcard.
        frame_timestamps_s: Vec<f64>,
    },
}

/// One trace's sample inside an [`Envelope::BatchedFrames`] batch.
///
/// Carries only the fields that differ between items — the `timestamp_ns` /
/// `timestamp_s` are hoisted onto the parent envelope because every joint in
/// a batch is captured at the same instant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchedFrameItem {
    /// Trace this sample belongs to.
    pub trace_id: String,
    /// Per-trace zero-based counter at publish time. The IPC listener carries
    /// this through into the unpacked [`Envelope::Frame::sequence_number`].
    pub sequence_number: u64,
    /// Opaque per-frame bytes. Transported length-prefix + raw, exactly as
    /// [`Envelope::Frame`]'s `payload`.
    pub payload: Vec<u8>,
}

/// One trace's stop record inside an [`Envelope::StopRecording`] payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceEnding {
    /// Trace identifier supplied earlier in [`Envelope::StartTrace`].
    pub trace_id: String,
    /// Total envelopes the producer published for this trace
    /// ([`Envelope::Frame`] count for scalar / joint traces,
    /// [`Envelope::VideoChunkReady`] count for video traces). The daemon's
    /// per-trace actor finalises the trace once its observed envelope count
    /// reaches this value; a fixed wait deadline marks the trace failed if
    /// promised envelopes never arrive (e.g. producer crash mid-publish).
    pub final_sequence_number: u64,
}

impl Envelope {
    /// Convenience constructor for [`Envelope::Frame`].
    pub fn frame(
        trace_id: String,
        sequence_number: u64,
        timestamp_ns: i64,
        timestamp_s: Option<f64>,
        payload: Vec<u8>,
    ) -> Self {
        Envelope::Frame {
            trace_id,
            sequence_number,
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
            Envelope::BatchedFrames { .. } => "batched_frames",
            Envelope::StopRecording { .. } => "stop_recording",
            Envelope::CancelRecording { .. } => "cancel_recording",
            Envelope::VideoChunkReady { .. } => "video_chunk_ready",
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
        let original =
            Envelope::frame("trace-1".into(), 0, 1_000_000, None, vec![1, 2, 3, 4, 5, 6]);
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn frame_sequence_number_round_trips() {
        let original = Envelope::frame("trace-1".into(), 42, 1, None, vec![0]);
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        match decoded {
            Envelope::Frame {
                sequence_number, ..
            } => assert_eq!(sequence_number, 42),
            other => panic!("decoded envelope was not a Frame: {other:?}"),
        }
        assert_eq!(original, Envelope::decode(&bytes).expect("decode"));
    }

    #[test]
    fn frame_timestamp_s_is_bit_exact_over_postcard_wire() {
        // Postcard writes `f64` as 8 raw IEEE-754 bytes, so values that
        // would shift under a decimal parser (e.g. `7/60`) round-trip
        // bit-identically — required for the integration matrix's
        // exact-match assertion on the video sidecar timestamps.
        let original = Envelope::frame(
            "trace-1".into(),
            0,
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
        let original = Envelope::frame("trace-1".into(), 0, 0, None, vec![0xAB; PAYLOAD_LEN]);
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
    fn batched_frames_round_trips() {
        let original = Envelope::BatchedFrames {
            timestamp_ns: 1_700_000_000_000_000_000,
            timestamp_s: Some(1_700_000_000.5),
            frames: vec![
                BatchedFrameItem {
                    trace_id: "joint-0".into(),
                    sequence_number: 0,
                    payload: br#"{"timestamp":1.0,"value":0.5}"#.to_vec(),
                },
                BatchedFrameItem {
                    trace_id: "joint-1".into(),
                    sequence_number: 7,
                    payload: br#"{"timestamp":1.0,"value":-0.25}"#.to_vec(),
                },
            ],
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        assert_eq!(original.kind(), "batched_frames");
    }

    #[test]
    fn batched_frames_worst_case_fits_commands_slice() {
        // The integration matrix's high-dimensionality case logs 1000 joints
        // per call. Each joint payload is a small `{"timestamp":..,"value":..}`
        // JSON object plus a UUID trace_id; the whole batch must fit inside a
        // single `commands` sample so the producer can publish it in one go.
        let frames: Vec<BatchedFrameItem> = (0..1000)
            .map(|index| BatchedFrameItem {
                trace_id: format!("11111111-2222-3333-4444-{index:012}"),
                sequence_number: index as u64,
                payload: br#"{"timestamp":1747740000.1234567,"value":-1.234567890123}"#.to_vec(),
            })
            .collect();
        let envelope = Envelope::BatchedFrames {
            timestamp_ns: 1_747_740_000_123_456_700,
            timestamp_s: Some(1_747_740_000.123_456_7),
            frames,
        };
        let bytes = envelope.encode().expect("encode");
        assert!(
            bytes.len() <= service_name::COMMANDS_MAX_PAYLOAD_BYTES,
            "1000-joint batch ({} bytes) must fit the commands slice ({} bytes)",
            bytes.len(),
            service_name::COMMANDS_MAX_PAYLOAD_BYTES,
        );
    }

    #[test]
    fn envelope_kind_label_is_stable() {
        let env = Envelope::StopRecording {
            recording_id: "rec-1".into(),
            trace_endings: Vec::new(),
        };
        assert_eq!(env.kind(), "stop_recording");
    }

    #[test]
    fn stop_recording_round_trips_with_trace_endings() {
        let original = Envelope::StopRecording {
            recording_id: "rec-1".into(),
            trace_endings: vec![
                TraceEnding {
                    trace_id: "trace-a".into(),
                    final_sequence_number: 17,
                },
                TraceEnding {
                    trace_id: "trace-b".into(),
                    final_sequence_number: 0,
                },
            ],
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
    }

    #[test]
    fn stop_recording_with_thousand_trace_endings_fits_commands_slice() {
        // The integration matrix's 1000-joint recording mints up to 1000
        // traces per recording. The combined StopRecording envelope must
        // fit a single `commands` sample.
        let trace_endings: Vec<TraceEnding> = (0..1000)
            .map(|index| TraceEnding {
                trace_id: format!("11111111-2222-3333-4444-{index:012}"),
                final_sequence_number: index as u64,
            })
            .collect();
        let envelope = Envelope::StopRecording {
            recording_id: "11111111-2222-3333-4444-555555555555".into(),
            trace_endings,
        };
        let bytes = envelope.encode().expect("encode");
        assert!(
            bytes.len() <= service_name::COMMANDS_MAX_PAYLOAD_BYTES,
            "1000-trace stop envelope ({} bytes) must fit the commands slice ({} bytes)",
            bytes.len(),
            service_name::COMMANDS_MAX_PAYLOAD_BYTES,
        );
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
    fn video_chunk_ready_round_trips() {
        let original = Envelope::VideoChunkReady {
            recording_id: "rec-1".into(),
            trace_id: "trace-1".into(),
            chunk_index: 3,
            width: 1920,
            height: 1080,
            byte_count: 128 * 1024 * 1024,
            frame_count: 7,
            frame_timestamps_s: vec![
                1_700_000_000.0,
                1_700_000_000.016_666_7,
                1_700_000_000.033_333_3,
                7.0_f64 / 60.0_f64,
                1_700_000_000.066_666_7,
                1_700_000_000.083_333_3,
                1_700_000_000.1,
            ],
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        assert_eq!(original.kind(), "video_chunk_ready");
    }

    #[test]
    fn video_chunk_ready_worst_case_fits_commands_slice() {
        // A 128 MiB 1080p chunk holds ~3800 frames; carry one f64 per frame.
        // Even at 10_000 timestamps the envelope is comfortably under
        // COMMANDS_MAX_PAYLOAD_BYTES.
        let frame_timestamps_s: Vec<f64> = (0..10_000).map(|i| i as f64 * 1e-3).collect();
        let envelope = Envelope::VideoChunkReady {
            recording_id: "11111111-2222-3333-4444-555555555555".into(),
            trace_id: "66666666-7777-8888-9999-aaaaaaaaaaaa".into(),
            chunk_index: 42,
            width: 1920,
            height: 1080,
            byte_count: 128 * 1024 * 1024,
            frame_count: frame_timestamps_s.len() as u32,
            frame_timestamps_s,
        };
        let bytes = envelope.encode().expect("encode");
        assert!(
            bytes.len() <= service_name::COMMANDS_MAX_PAYLOAD_BYTES,
            "10k-frame chunk envelope ({} bytes) must fit the commands slice ({} bytes)",
            bytes.len(),
            service_name::COMMANDS_MAX_PAYLOAD_BYTES,
        );
    }
}
