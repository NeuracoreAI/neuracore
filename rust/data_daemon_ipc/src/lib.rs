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
//! # The thin-shipper model
//!
//! The producer is a *thin shipper*: it knows nothing about recordings. Every
//! envelope is tagged only with its **source** (`robot_id`, `robot_instance`)
//! and — for data — its **sensor** (`data_type`, `sensor_name`) and capture
//! `timestamp_ns`. The producer publishes three fire-and-forget lifecycle
//! events ([`Envelope::StartRecording`] / [`Envelope::StopRecording`] /
//! [`Envelope::CancelRecording`]) carrying the lifecycle wall-clock timestamp,
//! and the daemon decides — from its per-source active-window map — which
//! recording (if any) each datum belongs to. There is **no** `recording_id`,
//! `recording_index`, `trace_id`, or `sequence_number` on the wire; the daemon
//! assigns and stores those after routing.
//!
//! All envelopes — lifecycle, joints/scalars, and the chunk-ready
//! notifications for video traces — travel over a single `commands` service.
//! Video pixel buffers themselves are *not* on the IPC bus: the producer
//! spools them to disk as NUT chunks and announces each finished chunk with an
//! [`Envelope::VideoChunkReady`] envelope. See [`service_name`].

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// iceoryx2 service-name conventions shared by daemon and producer.
pub mod service_name {
    /// Pub/sub service carrying every IPC envelope: lifecycle
    /// (`start_recording`, `stop_recording`, `cancel_recording`), non-video
    /// `data` / `batched_data` envelopes (joints, scalars, custom streams),
    /// and the [`Envelope::VideoChunkReady`] notifications that hand off
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

    /// Worst-case postcard size of one frame's contribution to a
    /// [`Envelope::VideoChunkReady`] announcement: a `frame_timestamps_ns`
    /// element is an `i64` zigzag varint (≤10 bytes for a full-range Unix-ns
    /// value) and a `frame_timestamps_s` element is a fixed 8-byte `f64`.
    pub const VIDEO_CHUNK_BYTES_PER_FRAME: usize = 10 + 8;

    /// Bytes held back from [`COMMANDS_MAX_PAYLOAD_BYTES`] for a
    /// `VideoChunkReady` envelope's fixed fields — the enum tag, source ids,
    /// dimensions, counts and the two vector length prefixes — so the frame cap
    /// below is computed against only the room left for the per-frame vectors.
    pub const VIDEO_CHUNK_HEADER_RESERVE: usize = 4 * 1024;

    /// Maximum number of frames a single video chunk may carry.
    ///
    /// The producer seals a chunk at the **lower** of its byte threshold and
    /// this frame cap. The cap exists so a [`Envelope::VideoChunkReady`]
    /// announcement always fits one [`COMMANDS_MAX_PAYLOAD_BYTES`] sample: the
    /// per-frame `frame_timestamps_{ns,s}` vectors are the only unbounded part
    /// of the envelope, so a long recording of small frames — which never
    /// reaches the byte threshold mid-recording — would otherwise accumulate
    /// enough frames in a single chunk to overflow the slice. The announcement
    /// then fails to publish and the whole recording's video is lost. Guarded
    /// by `video_chunk_ready_at_frame_cap_fits_commands_slice`.
    pub const MAX_VIDEO_CHUNK_FRAMES: u32 = ((COMMANDS_MAX_PAYLOAD_BYTES
        - VIDEO_CHUNK_HEADER_RESERVE)
        / VIDEO_CHUNK_BYTES_PER_FRAME) as u32;

    /// Subscriber buffer depth for the lifecycle service.
    ///
    /// Lossless, in-order delivery is *not* a function of this depth: the
    /// service is opened with `enable_safe_overflow(false)`, so a full
    /// buffer makes the producer's `Block` strategy wait rather than silently
    /// evict the oldest sample. (Were overflow left at iceoryx2's default the
    /// oldest sample would be dropped, stranding the daemon's per-source
    /// routing.) The depth therefore only trades producer-blocking frequency
    /// against memory.
    ///
    /// The depth is bounded from *above* by memory, not just throughput.
    /// iceoryx2 sizes a publisher's data segment as
    /// `max_subscribers × (buffer + borrowed) × initial_max_slice_len`, and
    /// the resident footprint is `buffer × actual_sample_size`. The largest
    /// `commands` sample is a [`Envelope::BatchedData`] envelope — the
    /// integration matrix's 1000-joint worst case encodes to ~90 KiB — so a
    /// 1024-deep buffer would retain ~94 MiB of pages per publisher and
    /// exhaust the 64 MiB devcontainer `/dev/shm`.
    ///
    /// 64 keeps that worst case at ~6 MiB per publisher while staying deep
    /// enough for steady state: the daemon drains every 1 ms and batched
    /// joint logging emits one envelope per timestep, so the buffer never
    /// fills under normal load.
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
/// Every variant is tagged with its **source** (`robot_id`, `robot_instance`).
/// Data variants additionally carry their **sensor** (`data_type`,
/// `sensor_name`) and capture `timestamp_ns`. No recording or trace identity
/// travels on the wire — the daemon owns it (see the crate-level docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Envelope {
    /// Producer announces that a recording has started for a source.
    ///
    /// The daemon opens an active window for `(robot_id, robot_instance)` at
    /// `publish_timestamp_ns`, allocates the local `recording_index`, and
    /// inserts the recording row. Processed immediately on arrival (bypasses
    /// the holdback).
    StartRecording {
        /// Robot identifier — the first half of the source key.
        robot_id: String,
        /// Robot instance — the second half of the source key.
        robot_instance: i64,
        /// Optional robot human-readable name.
        robot_name: Option<String>,
        /// Optional dataset identifier.
        dataset_id: Option<String>,
        /// Optional dataset human-readable name.
        dataset_name: Option<String>,
        /// Producer wall-clock publish time (Unix nanoseconds) at which the
        /// recording window opens — the inclusive lower bound of the window's
        /// membership range, on the same publish clock as every `Data`
        /// envelope. The **only** key used for window membership, so routing
        /// never depends on the caller's capture clock.
        publish_timestamp_ns: i64,
        /// Caller-supplied capture time (Unix nanoseconds) for the recording's
        /// start — the recording's *own* clock, or the publish time when the
        /// caller supplied none. Stored as the row's `start_timestamp_ns` and
        /// POSTed to the backend as `start_time`; never used for routing.
        timestamp_ns: i64,
    },
    /// Producer announces that the source's active recording has stopped.
    ///
    /// The daemon sets the window's exclusive upper bound and begins the
    /// drain/finalise countdown. Processed immediately on arrival.
    StopRecording {
        /// Robot identifier — the first half of the source key.
        robot_id: String,
        /// Robot instance — the second half of the source key.
        robot_instance: i64,
        /// Producer wall-clock publish time (Unix nanoseconds) at which the
        /// recording window closes — the exclusive upper bound of the
        /// membership range, on the same publish clock as the data envelopes.
        publish_timestamp_ns: i64,
        /// Caller-supplied capture time (Unix nanoseconds) for the recording's
        /// stop — or the publish time when the caller supplied none. Stored as
        /// the row's `stop_timestamp_ns` and POSTed to the backend as
        /// `end_time`; never used for routing.
        timestamp_ns: i64,
    },
    /// Producer cancels the source's active recording — the daemon drops every
    /// in-flight per-trace actor, deletes the on-disk artefacts, marks the
    /// recording row cancelled, and uploads nothing. Processed immediately on
    /// arrival; the daemon is idempotent.
    CancelRecording {
        /// Robot identifier — the first half of the source key.
        robot_id: String,
        /// Robot instance — the second half of the source key.
        robot_instance: i64,
        /// Caller-supplied capture time (Unix nanoseconds) for the cancel — or
        /// the publish time when the caller supplied none. A cancel is a
        /// recording stop that discards data, so the daemon stores this as the
        /// row's `stop_timestamp_ns` and POSTs it as the backend `end_time`,
        /// exactly like `StopRecording`. No window-boundary `publish_timestamp_ns`
        /// is carried because cancelling drops the window outright.
        timestamp_ns: i64,
    },
    /// Producer delivers one sensor sample.
    ///
    /// The payload is opaque to the IPC layer; the per-trace actor parses it
    /// according to `data_type` and writes it through the JSON writer. The
    /// daemon holds the datum for the configured holdback, then routes it into
    /// the source's window whose `[started_at_ns, stopped_at_ns)` contains
    /// `timestamp_ns`.
    ///
    /// Video frames do *not* travel as `Data` envelopes — they are spooled to
    /// disk by the producer and announced via [`Envelope::VideoChunkReady`]
    /// instead.
    Data {
        /// Robot identifier — the first half of the source key.
        robot_id: String,
        /// Robot instance — the second half of the source key.
        robot_instance: i64,
        /// Wire data-type label (e.g. `"JOINT_POSITIONS"`, `"RGB_IMAGES"`).
        data_type: String,
        /// Per-stream sensor label (joint name, camera id, …) — disambiguates
        /// traces that share a `data_type`. Persisted to the trace row's
        /// `data_type_name` column.
        sensor_name: Option<String>,
        /// Producer wall-clock time (Unix nanoseconds) stamped at the moment
        /// this envelope is published. This is the **only** key used for
        /// window membership — it is decoupled from the data's own capture
        /// time, so the daemon's routing never depends on what clock the
        /// caller timestamps data with. Lifecycle events carry the same kind
        /// of publish-clock timestamp, so a datum belongs to the window whose
        /// `[started_at_ns, stopped_at_ns)` brackets its publish time.
        publish_timestamp_ns: i64,
        /// Caller-supplied capture time in nanoseconds since the Unix epoch —
        /// the data's *own* clock, written into the trace content. Not used
        /// for routing.
        timestamp_ns: i64,
        /// Optional caller-supplied capture time in seconds (f64). Postcard
        /// writes this bit-exact.
        timestamp_s: Option<f64>,
        /// Opaque per-sample bytes. Postcard transports these as
        /// length-prefix + raw bytes (no expansion).
        payload: Vec<u8>,
    },
    /// Producer delivers one sample for each of several sensors captured at the
    /// same instant — used by scalar joint logging, where a robot's N joints
    /// are sampled together.
    ///
    /// Collapsing N [`Envelope::Data`] envelopes into one IPC message cuts the
    /// per-call iceoryx2 publish count (and the pressure on the lifecycle
    /// buffer) by a factor of N. Because every item shares the batch's
    /// `timestamp_ns`, the whole batch belongs to one window — the daemon
    /// holds and routes it as a single unit.
    BatchedData {
        /// Robot identifier — the first half of the source key.
        robot_id: String,
        /// Robot instance — the second half of the source key.
        robot_instance: i64,
        /// Producer wall-clock publish time (Unix nanoseconds), shared by every
        /// item. The sole key for window membership (see [`Envelope::Data`]).
        publish_timestamp_ns: i64,
        /// Caller-supplied capture time (ns), shared by every item — content,
        /// not routing.
        timestamp_ns: i64,
        /// Optional caller-supplied capture time in seconds, shared by every
        /// item.
        timestamp_s: Option<f64>,
        /// Per-sensor samples; each routes to one trace actor.
        items: Vec<BatchedDataItem>,
    },
    /// Producer announces a finished NUT chunk for a video trace.
    ///
    /// The producer spools captured RGB frames to disk as a sequence of NUT
    /// chunks under a recording-independent spool dir keyed by source + sensor,
    /// each named `chunk_{spool_ns}_{thread_id}.nut` so two recordings on the
    /// same source never collide on a filename. When a chunk crosses the flush
    /// threshold (or a lifecycle event rolls it) the producer finishes the NUT
    /// and publishes this envelope so the daemon can route the chunk into the
    /// right recording window (by `publish_timestamp_ns`), relink the NUT under
    /// the recording, and encode it to a sealed MP4 segment. Per-frame `timestamp_s` values are
    /// carried inline so the daemon-side `trace.json` sidecar matches the
    /// bit-exact assertion.
    VideoChunkReady {
        /// Robot identifier — the first half of the source key.
        robot_id: String,
        /// Robot instance — the second half of the source key.
        robot_instance: i64,
        /// Wire data-type label (e.g. `"RGB_IMAGES"`).
        data_type: String,
        /// Per-stream sensor label (camera id).
        sensor_name: Option<String>,
        /// Producer wall-clock ns stamped when the chunk's NUT file was opened
        /// (its first frame). Serves two purposes: it is the key that routes
        /// the whole chunk into a recording window — the open moment lies
        /// strictly inside the recording, so membership is unambiguous — and,
        /// with `thread_id`, it forms the chunk's spool filename
        /// `chunk_{publish_timestamp_ns}_{thread_id}.nut` so the daemon can
        /// reconstruct the spool path.
        publish_timestamp_ns: i64,
        /// OS thread id (`gettid`) of the producer thread that spooled the
        /// chunk. Disambiguates the spool filename across threads and is a
        /// useful breadcrumb when inspecting the spool directory.
        thread_id: i64,
        /// Frame width in pixels (constant across a trace).
        width: u32,
        /// Frame height in pixels (constant across a trace).
        height: u32,
        /// Size of the NUT file in bytes.
        byte_count: u64,
        /// Number of frames packed into this chunk.
        frame_count: u32,
        /// Per-frame capture time in nanoseconds since the Unix epoch, in
        /// arrival order. Length equals `frame_count`. Used to bucket the
        /// chunk's frames against the source's active-window map.
        frame_timestamps_ns: Vec<i64>,
        /// Per-frame `timestamp_s` (Unix seconds, f64) in arrival order.
        /// Length equals `frame_count`; values round-trip bit-exact through
        /// postcard for the metadata sidecar.
        frame_timestamps_s: Vec<f64>,
    },
}

/// One sensor's sample inside an [`Envelope::BatchedData`] batch.
///
/// Carries only the fields that differ between items — the `timestamp_ns` /
/// `timestamp_s` are hoisted onto the parent envelope because every sensor in
/// a batch is captured at the same instant. Each item self-tags its sensor
/// because there is no pre-registered trace to look up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchedDataItem {
    /// Wire data-type label for this item.
    pub data_type: String,
    /// Per-stream sensor label (joint name, …).
    pub sensor_name: Option<String>,
    /// Opaque per-sample bytes. Transported length-prefix + raw, exactly as
    /// [`Envelope::Data`]'s `payload`.
    pub payload: Vec<u8>,
}

impl Envelope {
    /// Convenience constructor for [`Envelope::Data`].
    #[allow(clippy::too_many_arguments)]
    pub fn data(
        robot_id: String,
        robot_instance: i64,
        data_type: String,
        sensor_name: Option<String>,
        publish_timestamp_ns: i64,
        timestamp_ns: i64,
        timestamp_s: Option<f64>,
        payload: Vec<u8>,
    ) -> Self {
        Envelope::Data {
            robot_id,
            robot_instance,
            data_type,
            sensor_name,
            publish_timestamp_ns,
            timestamp_ns,
            timestamp_s,
            payload,
        }
    }

    /// Variant name used in tracing/logging.
    pub fn kind(&self) -> &'static str {
        match self {
            Envelope::StartRecording { .. } => "start_recording",
            Envelope::StopRecording { .. } => "stop_recording",
            Envelope::CancelRecording { .. } => "cancel_recording",
            Envelope::Data { .. } => "data",
            Envelope::BatchedData { .. } => "batched_data",
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
    fn start_recording_round_trips_through_postcard() {
        let original = Envelope::StartRecording {
            robot_id: "robot-1".into(),
            robot_instance: 3,
            robot_name: Some("arm".into()),
            dataset_id: Some("ds-1".into()),
            dataset_name: Some("warehouse".into()),
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            timestamp_ns: 1_700_000_000_000_000_000,
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        assert_eq!(original.kind(), "start_recording");
    }

    #[test]
    fn data_envelope_preserves_payload_bytes() {
        let original = Envelope::data(
            "robot-1".into(),
            0,
            "JOINT_POSITIONS".into(),
            Some("waist".into()),
            1_700_000_000_000_000_000,
            1_000_000,
            None,
            vec![1, 2, 3, 4, 5, 6],
        );
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        assert_eq!(original.kind(), "data");
    }

    #[test]
    fn data_timestamp_s_is_bit_exact_over_postcard_wire() {
        // Postcard writes `f64` as 8 raw IEEE-754 bytes, so values that
        // would shift under a decimal parser (e.g. `7/60`) round-trip
        // bit-identically — required for the integration matrix's
        // exact-match assertion on the video sidecar timestamps.
        let original = Envelope::data(
            "robot-1".into(),
            0,
            "RGB_IMAGES".into(),
            Some("camera_right".into()),
            1_700_000_000_000_000_000,
            116_666_666,
            Some(7.0_f64 / 60.0_f64),
            vec![0xAA, 0xBB],
        );
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        if let Envelope::Data { timestamp_s, .. } = decoded {
            assert_eq!(
                timestamp_s.map(f64::to_bits),
                Some((7.0_f64 / 60.0_f64).to_bits()),
            );
        } else {
            panic!("decoded envelope was not Data");
        }
    }

    #[test]
    fn data_payload_does_not_expand_under_postcard() {
        // The whole point of moving off JSON is that `Vec<u8>` no longer
        // expands ~3× as a JSON array of integers. Encode a 1 MiB payload
        // and check the wire form is within a small constant of the raw
        // bytes (variant tag + length prefix + source/sensor + timestamps).
        const PAYLOAD_LEN: usize = 1024 * 1024;
        let original = Envelope::data(
            "robot-1".into(),
            0,
            "RGB_IMAGES".into(),
            None,
            0,
            0,
            None,
            vec![0xAB; PAYLOAD_LEN],
        );
        let bytes = original.encode().expect("encode");
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
    fn batched_data_round_trips() {
        let original = Envelope::BatchedData {
            robot_id: "robot-1".into(),
            robot_instance: 0,
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            timestamp_ns: 1_700_000_000_000_000_000,
            timestamp_s: Some(1_700_000_000.5),
            items: vec![
                BatchedDataItem {
                    data_type: "JOINT_POSITIONS".into(),
                    sensor_name: Some("joint-0".into()),
                    payload: br#"{"timestamp":1.0,"value":0.5}"#.to_vec(),
                },
                BatchedDataItem {
                    data_type: "JOINT_POSITIONS".into(),
                    sensor_name: Some("joint-1".into()),
                    payload: br#"{"timestamp":1.0,"value":-0.25}"#.to_vec(),
                },
            ],
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        assert_eq!(original.kind(), "batched_data");
    }

    #[test]
    fn batched_data_worst_case_fits_commands_slice() {
        // The integration matrix's high-dimensionality case logs 1000 joints
        // per call. Each joint payload is a small `{"timestamp":..,"value":..}`
        // JSON object plus a data_type label and sensor name; the whole batch
        // must fit inside a single `commands` sample so the producer can
        // publish it in one go.
        let items: Vec<BatchedDataItem> = (0..1000)
            .map(|index| BatchedDataItem {
                data_type: "JOINT_POSITIONS".into(),
                sensor_name: Some(format!("vx300s_left_joint_{index:04}")),
                payload: br#"{"timestamp":1747740000.1234567,"value":-1.234567890123}"#.to_vec(),
            })
            .collect();
        let envelope = Envelope::BatchedData {
            robot_id: "11111111-2222-3333-4444-555555555555".into(),
            robot_instance: 0,
            publish_timestamp_ns: 1_747_740_000_123_456_700,
            timestamp_ns: 1_747_740_000_123_456_700,
            timestamp_s: Some(1_747_740_000.123_456_7),
            items,
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
    fn stop_and_cancel_round_trip() {
        let stop = Envelope::StopRecording {
            robot_id: "robot-1".into(),
            robot_instance: 2,
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            timestamp_ns: 1_700_000_000_000_000_000,
        };
        let bytes = stop.encode().expect("encode");
        assert_eq!(stop, Envelope::decode(&bytes).expect("decode"));
        assert_eq!(stop.kind(), "stop_recording");

        let cancel = Envelope::CancelRecording {
            robot_id: "robot-1".into(),
            robot_instance: 2,
            timestamp_ns: 1_700_000_000_000_000_000,
        };
        let bytes = cancel.encode().expect("encode");
        assert_eq!(cancel, Envelope::decode(&bytes).expect("decode"));
        assert_eq!(cancel.kind(), "cancel_recording");
    }

    #[test]
    fn video_chunk_ready_round_trips() {
        let original = Envelope::VideoChunkReady {
            robot_id: "robot-1".into(),
            robot_instance: 0,
            data_type: "RGB_IMAGES".into(),
            sensor_name: Some("camera_right".into()),
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            thread_id: 4242,
            width: 1920,
            height: 1080,
            byte_count: 128 * 1024 * 1024,
            frame_count: 4,
            frame_timestamps_ns: vec![
                1_700_000_000_000_000_000,
                1_700_000_000_016_666_700,
                1_700_000_000_033_333_300,
                1_700_000_000_050_000_000,
            ],
            frame_timestamps_s: vec![
                1_700_000_000.0,
                1_700_000_000.016_666_7,
                1_700_000_000.033_333_3,
                7.0_f64 / 60.0_f64,
            ],
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        assert_eq!(original.kind(), "video_chunk_ready");
    }

    #[test]
    fn video_chunk_ready_worst_case_fits_commands_slice() {
        // A 128 MiB 1080p chunk holds ~3800 frames; carry two timestamps per
        // frame (ns + s). Even at 10_000 frames the envelope is comfortably
        // under COMMANDS_MAX_PAYLOAD_BYTES.
        let frame_timestamps_ns: Vec<i64> = (0..10_000).map(|i| i as i64 * 1_000_000).collect();
        let frame_timestamps_s: Vec<f64> = (0..10_000).map(|i| i as f64 * 1e-3).collect();
        let envelope = Envelope::VideoChunkReady {
            robot_id: "11111111-2222-3333-4444-555555555555".into(),
            robot_instance: 0,
            data_type: "RGB_IMAGES".into(),
            sensor_name: Some("camera_right".into()),
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            thread_id: 42,
            width: 1920,
            height: 1080,
            byte_count: 128 * 1024 * 1024,
            frame_count: frame_timestamps_ns.len() as u32,
            frame_timestamps_ns,
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

    #[test]
    fn video_chunk_ready_at_frame_cap_fits_commands_slice() {
        // The producer caps a chunk at MAX_VIDEO_CHUNK_FRAMES frames so its
        // announcement always fits one commands sample. Prove the cap holds at
        // the absolute worst case: every per-frame ns timestamp a full-range
        // i64 (10-byte postcard zigzag varint) and every fixed field maxed out.
        // Without the cap a long recording of tiny frames overflows the slice
        // and the whole recording's video announcement fails to publish.
        let count = service_name::MAX_VIDEO_CHUNK_FRAMES as usize;
        let frame_timestamps_ns: Vec<i64> = (0..count).map(|i| i64::MAX - i as i64).collect();
        let frame_timestamps_s: Vec<f64> = (0..count).map(|i| i as f64).collect();
        let envelope = Envelope::VideoChunkReady {
            robot_id: "11111111-2222-3333-4444-555555555555".into(),
            robot_instance: i64::MAX,
            data_type: "RGB_IMAGES".into(),
            sensor_name: Some("camera_with_a_deliberately_long_sensor_label".into()),
            publish_timestamp_ns: i64::MAX,
            thread_id: i64::MAX,
            width: u32::MAX,
            height: u32::MAX,
            byte_count: u64::MAX,
            frame_count: count as u32,
            frame_timestamps_ns,
            frame_timestamps_s,
        };
        let bytes = envelope.encode().expect("encode");
        assert!(
            bytes.len() <= service_name::COMMANDS_MAX_PAYLOAD_BYTES,
            "chunk at frame cap ({count} frames, {} bytes) must fit the commands slice ({} bytes)",
            bytes.len(),
            service_name::COMMANDS_MAX_PAYLOAD_BYTES,
        );
    }
}
