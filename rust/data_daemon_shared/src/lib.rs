//! Shared definitions for the Neuracore data daemon.
//!
//! Both the daemon binary and the PyO3 producer crate
//! (`data_daemon_bridge`) depend on this crate so they agree on everything
//! that crosses the process boundary:
//!
//! - the iceoryx2 service-name conventions ([`service_name`]),
//! - the [`Envelope`] enum carried over the `commands` service, and
//! - the helpers to (de)serialize that envelope to/from the byte slice payload
//!   iceoryx2 transports.
//!
//! It also owns the resolution the two processes must compute identically off
//! the same inputs: the daemon configuration model ([`config`]) and the
//! filesystem layout ([`paths`]). Keeping these here is what stops the daemon
//! and producer from drifting on, say, the spool-backlog cap or the recordings
//! root. [`ffmpeg`] lives here for a weaker reason: both crates shell out to
//! ffmpeg (the daemon to transcode, the producer only in its tests) and must
//! spell its options the same way.
//!
//! Envelopes are encoded with [`postcard`], a compact length-prefixed binary
//! format. Payload bytes travel raw (length-prefix + bytes — no base64 or
//! `[u8]→[i32]` expansion that JSON would force), and `f64` fields round-trip
//! bit-exact because postcard writes the IEEE-754 byte pattern directly. The
//! schema is forward-compatible: postcard's enum representation tags variants
//! with a varint-encoded u32 discriminant (one byte for the first 128
//! variants), so new envelope variants append cleanly.
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
//! recording (if any) each datum belongs to. There is no `recording_index`,
//! `trace_id`, or `sequence_number` on the wire; the daemon assigns and
//! stores those after routing. `StartRecording` carries an optional
//! `cloud_recording_id` for the one case where the backend, not the daemon,
//! already minted it — a recording started from the web frontend.
//!
//! All envelopes — lifecycle, joints/scalars, and the chunk-ready
//! notifications for video traces — travel over a single `commands` service.
//! Video pixel buffers themselves are *not* on the IPC bus: the producer
//! spools them to disk as NUT chunks and announces each finished chunk with an
//! [`Envelope::VideoChunkReady`] envelope. See [`service_name`].

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod config;
pub mod ffmpeg;
pub mod paths;

/// Recording-window membership for the frames *inside* one video chunk.
///
/// A chunk is a NUT file appended to until something seals it, so frames logged
/// either side of a boundary can share one and its open stamp speaks only for
/// the first. Membership is therefore resolved per frame, at both bounds, and
/// the **daemon** resolves it: from the per-frame offsets this module encodes it
/// keeps the run of frames each window owns, and one chunk can be claimed by
/// several windows.
///
/// The **producer** seals on nothing but its own caps — size and frame count —
/// and is never told where a window opened, so a chunk spanning one (or several)
/// is the normal case rather than the exception. Nothing aligns a chunk to a
/// window: that is what lets a camera process which opens no window of its own
/// still keep every frame it logged inside someone else's recording.
pub mod video_boundary {
    /// Nanoseconds per microsecond, the wire resolution of a frame offset.
    ///
    /// Microseconds, not milliseconds, because the offset is **floored**: a
    /// frame's reconstructed publish time is up to one unit earlier than its
    /// true one, which can push a frame published just inside a window back
    /// across its start bound. Adjacent windows do not abut — a recording stops
    /// before the next starts — so a frame pushed into that gap is claimed by
    /// neither and its pixels are dropped. At millisecond resolution and a fast
    /// camera that is a frame lost at a boundary crossing; a microsecond is far
    /// finer than any frame interval, which closes it.
    ///
    /// `u32` microseconds saturates at ~71 minutes of chunk span, far beyond any
    /// chunk a producer holds open, and [`publish_offset_us`] clamps rather than
    /// wrapping. It also matches the NUT spool time base
    /// ([`crate::service_name::VIDEO_SPOOL_TICKS_PER_SECOND`]), so every stage of
    /// the video path measures in the same unit.
    const NS_PER_US: i64 = 1_000;

    /// Is `frame_publish_ns` at or past the recording boundary `bound_ns`?
    ///
    /// Argument order is boundary first, frame second.
    pub fn at_or_past_boundary(bound_ns: i64, frame_publish_ns: i64) -> bool {
        frame_publish_ns >= bound_ns
    }

    /// One frame's publish time as µs after its chunk's open stamp, floored.
    pub fn publish_offset_us(chunk_open_ns: i64, frame_publish_ns: i64) -> u32 {
        let offset_ns = frame_publish_ns.saturating_sub(chunk_open_ns).max(0);
        (offset_ns / NS_PER_US).try_into().unwrap_or(u32::MAX)
    }

    /// The inverse of [`publish_offset_us`], paired with it so neither can
    /// change alone.
    pub fn frame_publish_ns(chunk_open_ns: i64, offset_us: u32) -> i64 {
        chunk_open_ns.saturating_add(i64::from(offset_us) * NS_PER_US)
    }

    /// Leading frames published before `bound_ns`, i.e. the index of the first
    /// that is [`at_or_past_boundary`].
    pub fn frames_before_boundary(offsets_us: &[u32], chunk_open_ns: i64, bound_ns: i64) -> usize {
        offsets_us
            .iter()
            .position(|offset| {
                at_or_past_boundary(bound_ns, frame_publish_ns(chunk_open_ns, *offset))
            })
            .unwrap_or(offsets_us.len())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const CHUNK_OPEN_NS: i64 = 1_700_000_000_000_000_000;

        #[test]
        fn a_frame_on_the_bound_is_already_past_it() {
            // The half-open convention both sides must agree on.
            assert!(!at_or_past_boundary(CHUNK_OPEN_NS, CHUNK_OPEN_NS - 1));
            assert!(at_or_past_boundary(CHUNK_OPEN_NS, CHUNK_OPEN_NS));
            assert!(at_or_past_boundary(CHUNK_OPEN_NS, CHUNK_OPEN_NS + 1));
        }

        #[test]
        fn offsets_round_trip_at_microsecond_resolution() {
            assert_eq!(publish_offset_us(CHUNK_OPEN_NS, CHUNK_OPEN_NS), 0);
            assert_eq!(
                publish_offset_us(CHUNK_OPEN_NS, CHUNK_OPEN_NS + 1_500),
                1,
                "a sub-microsecond remainder floors away"
            );
            assert_eq!(
                frame_publish_ns(CHUNK_OPEN_NS, 33_333),
                CHUNK_OPEN_NS + 33_333_000
            );
        }

        #[test]
        fn a_frame_just_inside_a_window_is_not_floored_out_of_it() {
            // The hole microsecond resolution closes. A recording stops, the
            // next starts a millisecond later, and a frame lands just inside the
            // new window. Floored to the millisecond it would reconstruct
            // *before* that start — into the gap between the two recordings,
            // owned by neither — and its pixels would be dropped.
            let stop_ns = CHUNK_OPEN_NS + 4_000_000;
            let start_ns = stop_ns + 1_000_000;
            let frame_ns = start_ns + 400_000; // 0.4 ms into the new recording

            let offset = publish_offset_us(CHUNK_OPEN_NS, frame_ns);
            let reconstructed = frame_publish_ns(CHUNK_OPEN_NS, offset);

            assert!(
                reconstructed >= start_ns,
                "the frame must still read as inside the window it was logged in"
            );
            assert!(
                reconstructed > stop_ns,
                "and must not fall back into the gap before it"
            );
        }

        #[test]
        fn publish_offset_saturates_at_the_chunk_open() {
            // A backwards caller clock reads as 0 rather than wrapping.
            assert_eq!(publish_offset_us(CHUNK_OPEN_NS, CHUNK_OPEN_NS - 5), 0);
        }

        #[test]
        fn frames_before_boundary_cuts_on_the_bound() {
            // The frames before the boundary are kept; the rest are cut.
            let offsets = [0, 10, 20, 30, 40];
            let bound = frame_publish_ns(CHUNK_OPEN_NS, 25);
            assert_eq!(frames_before_boundary(&offsets, CHUNK_OPEN_NS, bound), 3);

            // And the bound itself is exclusive.
            let on_boundary = frame_publish_ns(CHUNK_OPEN_NS, 20);
            assert_eq!(
                frames_before_boundary(&offsets, CHUNK_OPEN_NS, on_boundary),
                2
            );
        }

        #[test]
        fn frames_entirely_before_the_boundary_are_all_kept() {
            let offsets = [0, 10, 20];
            let bound = frame_publish_ns(CHUNK_OPEN_NS, 60);
            assert_eq!(frames_before_boundary(&offsets, CHUNK_OPEN_NS, bound), 3);
        }

        #[test]
        fn the_cut_is_a_prefix_under_disordered_stamps() {
            // One file feeds one encode, so the first frame at or past the
            // boundary takes its successors with it.
            let offsets = [0, 30, 10, 20];
            let bound = frame_publish_ns(CHUNK_OPEN_NS, 25);
            assert_eq!(frames_before_boundary(&offsets, CHUNK_OPEN_NS, bound), 1);
        }
    }
}

/// iceoryx2 service-name conventions shared by daemon and producer.
pub mod service_name {
    /// Pub/sub service carrying every IPC envelope: lifecycle
    /// (`start_recording`, `stop_recording`, `cancel_recording`), non-video
    /// `data` / `batched_data` envelopes (joints, scalars, custom streams),
    /// and the [`crate::Envelope::VideoChunkReady`] notifications that hand off
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
    /// [`crate::Envelope::VideoChunkReady`] announcement: a `frame_timestamps_ns`
    /// element is an `i64` zigzag varint (≤10 bytes for a full-range Unix-ns
    /// value), a `frame_timestamps_s` element is a fixed 8-byte `f64`, and a
    /// `frame_publish_offsets_us` element is a `u32` varint (≤5 bytes).
    pub const VIDEO_CHUNK_BYTES_PER_FRAME: usize = 10 + 8 + 5;

    /// Bytes held back from [`COMMANDS_MAX_PAYLOAD_BYTES`] for a
    /// `VideoChunkReady` envelope's fixed fields — the enum tag, source ids,
    /// dimensions, counts and the two vector length prefixes — so the frame cap
    /// below is computed against only the room left for the per-frame vectors.
    pub const VIDEO_CHUNK_HEADER_RESERVE: usize = 4 * 1024;

    /// Maximum number of frames a single video chunk may carry.
    ///
    /// The producer seals a chunk at the **lower** of its byte threshold and
    /// this frame cap. The cap exists so a [`crate::Envelope::VideoChunkReady`]
    /// announcement always fits one [`COMMANDS_MAX_PAYLOAD_BYTES`] sample: the
    /// per-frame `frame_timestamps_{ns,s}` and `frame_publish_offsets_us`
    /// vectors are the only unbounded part
    /// of the envelope, so a long recording of small frames — which never
    /// reaches the byte threshold mid-recording — would otherwise accumulate
    /// enough frames in a single chunk to overflow the slice. The announcement
    /// then fails to publish and the whole recording's video is lost. Guarded
    /// by `video_chunk_ready_at_frame_cap_fits_commands_slice`.
    pub const MAX_VIDEO_CHUNK_FRAMES: u32 = ((COMMANDS_MAX_PAYLOAD_BYTES
        - VIDEO_CHUNK_HEADER_RESERVE)
        / VIDEO_CHUNK_BYTES_PER_FRAME) as u32;

    /// How long the producer's writer leaves a video chunk open on a stream
    /// that has stopped receiving frames before sealing it anyway.
    ///
    /// Shared because it is the daemon's only handle on when a quiet producer
    /// has finished: a stopped window owed a flush marker no producer will
    /// send waits out this bound and settles on the source's silence instead.
    pub const VIDEO_CHUNK_MAX_OPEN_NS: i64 = 5 * 1_000_000_000;

    /// The microsecond clock shared by every stage of the video path: the
    /// producer writes spool NUT chunks with a `1/1_000_000` time base, and
    /// the daemon pins its per-chunk encode outputs to the same clock
    /// (`-enc_time_base` / `-video_track_timescale`). Sharing one constant
    /// keeps the two in lockstep — if they diverge, ffmpeg falls back to a
    /// per-chunk *guessed* frame rate, chunks of one recording land on
    /// different timescales, and the stream-copy concat corrupts the merged
    /// video's PTS.
    pub const VIDEO_SPOOL_TICKS_PER_SECOND: u32 = 1_000_000;

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
    /// `commands` sample is a [`crate::Envelope::BatchedData`] envelope — the
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
    /// model: the data bridge parks its iceoryx2 publisher in a
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
    /// the data bridge). The integration matrix fans to 8 parallel worker
    /// subprocesses each running 5+ threads (main + RGB + joint roles), giving
    /// 40+ nodes plus the daemon. 512 gives enough headroom that the cap is
    /// never approached in any test configuration.
    ///
    /// **Failure mode when the cap *is* reached** (a long-lived process that
    /// churns through >512 distinct OS threads, each lazily building its own
    /// node on first `log_*`): `open_or_create` on the service returns
    /// `ExceedsMaxNumberOfNodes`. In the producer that surfaces as a
    /// `ProducerError` the publish path swallows (the sample is dropped, logged
    /// once) — data from new threads silently stops flowing; in the daemon a
    /// failed attach is fatal to that service. The node count never shrinks
    /// while the process lives (nodes are released only on process exit / fork),
    /// so a thread-churning workload leaks toward the cap monotonically.
    ///
    /// The scalable fix (one node shared per process, not per thread) is tracked
    /// separately and would reduce the live count to single digits and remove
    /// the cliff entirely.
    pub const MAX_NODES_PER_SERVICE: usize = 512;

    /// Request-response service the SDK uses to probe daemon readiness.
    ///
    /// A successful health reply proves the daemon has opened IPC and entered
    /// the listener loop, so launchers can wait on this service instead of
    /// treating the PID file as readiness.
    pub const HEALTH: &str = "neuracore/data_daemon/health";

    /// Maximum size of a single health-service sample.
    pub const HEALTH_MAX_PAYLOAD_BYTES: usize = 1024;

    /// Request-response service the SDK uses to read the neuracore version the
    /// running daemon was built from.
    ///
    /// The SDK compares that version with its own installed version, so a
    /// daemon left behind by an earlier install is reported instead of being
    /// adopted silently. It is a service of its own rather than a field on
    /// [`crate::HealthReply`] because postcard does not tag fields: a daemon
    /// built before this service simply never answers here, while a changed
    /// health reply would fail to decode and break the readiness contract.
    /// See [`crate::VersionRequest`] / [`crate::VersionReply`].
    pub const VERSION: &str = "neuracore/data_daemon/version";

    /// Maximum size of a single version-service sample. Both the request and
    /// the reply are a nonce plus a short version string.
    pub const VERSION_MAX_PAYLOAD_BYTES: usize = 1024;

    /// Request-response service the SDK uses to resolve a recording's
    /// daemon-owned cloud `recording_id`.
    ///
    /// The cloud id is minted asynchronously by the start notifier, so the SDK
    /// (`nc.start_recording(wait=True)`, tests) asks the daemon for it over this
    /// service instead of reading the daemon's private SQLite DB directly — the
    /// daemon answers authoritatively from its own state. A request carries the
    /// source + the recording's capture marker; the reply carries the id once
    /// minted (or "not yet"). See [`crate::RecordingIdQuery`] / [`crate::RecordingIdReply`].
    pub const RECORDING_IDS: &str = "neuracore/data_daemon/recording_ids";

    /// Maximum size of a single `recording_ids` service sample. Both the request and
    /// the reply are a handful of UUID strings + integers; 4 KiB is generous.
    pub const RECORDING_ID_MAX_PAYLOAD_BYTES: usize = 4 * 1024;

    /// Request-response service the SDK uses to read a source's *current*
    /// recording state.
    ///
    /// The daemon is the only party subscribed to the backend's org-wide
    /// notification stream, so it is the only one that knows a recording
    /// nothing local bracketed. Distinct from [`RECORDING_IDS`], which resolves
    /// the cloud id of *one named* recording: this asks which recording, if
    /// any, is live right now.
    ///
    /// See [`crate::RecordingStateQuery`] / [`crate::RecordingStateReply`].
    pub const RECORDING_STATE: &str = "neuracore/data_daemon/recording_state";

    /// Maximum size of a single `recording_state` service sample. Mirrors
    /// [`RECORDING_ID_MAX_PAYLOAD_BYTES`].
    pub const RECORDING_STATE_MAX_PAYLOAD_BYTES: usize = 4 * 1024;

    /// Maximum number of concurrent request-response clients. Mirrors
    /// [`MAX_PUBLISHERS_PER_SERVICE`]: the data bridge parks one client port
    /// per OS thread (iceoryx2 ports are `!Sync`), so the cap must cover the
    /// integration matrix's full thread fan-out.
    pub const MAX_REQUEST_RESPONSE_CLIENTS_PER_SERVICE: usize = 128;

    /// Maximum number of concurrent request-response servers. The daemon opens exactly one.
    pub const MAX_REQUEST_RESPONSE_SERVERS_PER_SERVICE: usize = 1;
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
        robot_id: String,
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
        /// Optional cloud recording id the backend already minted.
        cloud_recording_id: Option<String>,
    },
    /// Producer announces that the source's active recording has stopped.
    ///
    /// The daemon sets the window's exclusive upper bound and begins the
    /// drain/finalise countdown. Processed immediately on arrival.
    StopRecording {
        robot_id: String,
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
        robot_id: String,
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
        robot_id: String,
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
        robot_id: String,
        robot_instance: i64,
        /// Wire data-type label shared by every item (e.g. `"JOINT_POSITIONS"`).
        /// A batch is one `log_*` call for a single sensor group, so the type
        /// is constant across the batch — carried once here rather than
        /// duplicated into every [`BatchedDataItem`] (which, for the 1000-joint
        /// worst case, was ~16% of the envelope's wire size).
        data_type: String,
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
        robot_id: String,
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
        /// OS process id of the producer that sealed this chunk, matched
        /// against [`Envelope::SourceFlushed`]'s `producer_pid` so one process's
        /// marker never vouches for another's pending video.
        producer_pid: u32,
        /// Frame width in pixels (constant across a trace).
        width: u32,
        /// Frame height in pixels (constant across a trace).
        height: u32,
        /// Size of the NUT file in bytes.
        byte_count: u64,
        /// Number of frames packed into this chunk.
        frame_count: u32,
        /// Per-frame capture time in nanoseconds since the Unix epoch, in
        /// arrival order. Length equals `frame_count`. Capture-clock content for
        /// the trace sidecar; routing uses `frame_publish_offsets_us`.
        frame_timestamps_ns: Vec<i64>,
        /// Per-frame `timestamp_s` (Unix seconds, f64) in arrival order.
        /// Length equals `frame_count`; values round-trip bit-exact through
        /// postcard for the metadata sidecar.
        frame_timestamps_s: Vec<f64>,
        /// Original dtype of every frame in this chunk (never mixed — the
        /// producer seals and reopens a chunk on a dtype change, mirroring a
        /// geometry change). The daemon never decodes pixels; it threads this
        /// straight into the trace's `trace.json` sidecar for depth frames.
        dtype: FrameDtype,
        /// Per-frame publish time as µs after this chunk's own
        /// `publish_timestamp_ns`, in arrival order. What makes chunk membership
        /// per-frame rather than atomic; see [`video_boundary`].
        frame_publish_offsets_us: Vec<u32>,
    },
    /// Force the daemon to re-read its profile config immediately, rather than
    /// waiting for the config watcher's next poll. Sent by the SDK's
    /// `set_video_encoding_options` after it writes the profile so a
    /// `set → start_recording → log_frame` sequence observes the new codec.
    /// Carries no fields — the config is daemon-global — and touches no
    /// recording window; the dispatcher handles it by refreshing the in-memory
    /// config (see `cloud::config_watcher`).
    RefreshConfig {},
    /// End-of-stream marker for a just-stopped window's late data.
    ///
    /// Published on the same port as the chunk announcements, so it is ordered
    /// strictly behind every chunk it vouches for and the dispatcher can evict
    /// the closing window with no timing assumption.
    SourceFlushed {
        robot_id: String,
        robot_instance: i64,
        /// Publish time at the marker's send. Diagnostic only: deliberately
        /// not used for window membership.
        publish_timestamp_ns: i64,
        /// OS process id of the producer whose barrier this reports on.
        producer_pid: u32,
    },
    /// Producer asserts that it is currently logging video for a source —
    /// published *before* any of that video has been sealed into a chunk.
    VideoProducerActive {
        robot_id: String,
        robot_instance: i64,
        /// Publish time of the frame that triggered the claim, stamped on the
        /// logging thread, so it is the window-membership key here too.
        publish_timestamp_ns: i64,
        /// OS process id of the claiming producer, the same identity
        /// [`Envelope::SourceFlushed`] reports under.
        producer_pid: u32,
    },
}

/// Original pixel/sample representation of one video-family frame.
///
/// Carried on [`Envelope::VideoChunkReady`] so the daemon can record a depth
/// frame's original dtype in its `trace.json` sidecar without ever seeing the
/// pixels themselves — the daemon never decodes or converts frame data, it
/// only threads this label through to metadata (see the crate-level docs on
/// the thin-shipper model). The producer is the only side that interprets the
/// raw bytes: RGB frames are already packed RGB24 for the NUT/PNG pipeline;
/// depth frames are converted to RGB24 storage bytes by the producer before
/// ever reaching the NUT writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameDtype {
    /// Packed RGB24, one byte per channel — the existing RGB/NUT contract.
    Rgb8,
    /// 2D depth frame, IEEE-754 binary16 (metres).
    DepthF16,
    /// 2D depth frame, IEEE-754 binary32 (metres).
    DepthF32,
}

impl FrameDtype {
    /// Bytes occupied by one pixel/sample of this representation, before any
    /// depth-to-RGB24 conversion.
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            FrameDtype::Rgb8 => 3,
            FrameDtype::DepthF16 => 2,
            FrameDtype::DepthF32 => 4,
        }
    }

    /// Parse the Python-facing wire label — `numpy.dtype.name`, i.e.
    /// `"uint8"` for RGB and `"float16"` / `"float32"` for depth. `None` for
    /// anything else, so the native boundary rejects an unsupported dtype
    /// with a clear error rather than silently misinterpreting the buffer.
    pub fn from_wire_label(label: &str) -> Option<Self> {
        match label {
            "uint8" => Some(FrameDtype::Rgb8),
            "float16" => Some(FrameDtype::DepthF16),
            "float32" => Some(FrameDtype::DepthF32),
            _ => None,
        }
    }

    /// The canonical `trace.json` dtype string for a depth frame — `None` for
    /// RGB, which keeps the existing RGB `trace.json` schema untouched (no
    /// consumer or established schema calls for an RGB dtype field).
    pub fn depth_label(self) -> Option<&'static str> {
        match self {
            FrameDtype::Rgb8 => None,
            FrameDtype::DepthF16 => Some("float16"),
            FrameDtype::DepthF32 => Some("float32"),
        }
    }
}

/// One sensor's sample inside an [`Envelope::BatchedData`] batch.
///
/// Carries only the fields that differ between items — `data_type`,
/// `timestamp_ns` and `timestamp_s` are hoisted onto the parent envelope
/// because every sensor in a batch shares them (one `log_*` call, one sensor
/// group, one capture instant). Each item self-tags its `sensor_name` because
/// there is no pre-registered trace to look up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchedDataItem {
    /// Per-stream sensor label (joint name, …).
    pub sensor_name: Option<String>,
    /// Opaque per-sample bytes. Transported length-prefix + raw, exactly as
    /// [`Envelope::Data`]'s `payload`.
    pub payload: Vec<u8>,
}

impl Envelope {
    /// Variant name used in tracing/logging.
    pub fn kind(&self) -> &'static str {
        match self {
            Envelope::StartRecording { .. } => "start_recording",
            Envelope::StopRecording { .. } => "stop_recording",
            Envelope::CancelRecording { .. } => "cancel_recording",
            Envelope::Data { .. } => "data",
            Envelope::BatchedData { .. } => "batched_data",
            Envelope::VideoChunkReady { .. } => "video_chunk_ready",
            Envelope::RefreshConfig {} => "refresh_config",
            Envelope::SourceFlushed { .. } => "source_flushed",
            Envelope::VideoProducerActive { .. } => "video_producer_active",
        }
    }

    /// Encode the envelope as a postcard byte vector ready for an iceoryx2
    /// sample.
    pub fn encode(&self) -> Result<Vec<u8>, EnvelopeCodecError> {
        encode_postcard(self)
    }

    /// Decode an envelope from the byte slice carried in an iceoryx2 sample.
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvelopeCodecError> {
        decode_postcard(bytes)
    }
}

/// Encode a wire type as a postcard byte vector.
fn encode_postcard<T: Serialize>(value: &T) -> Result<Vec<u8>, EnvelopeCodecError> {
    postcard::to_allocvec(value).map_err(EnvelopeCodecError::Encode)
}

/// Decode a wire type from a postcard byte slice.
fn decode_postcard<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, EnvelopeCodecError> {
    postcard::from_bytes(bytes).map_err(EnvelopeCodecError::Decode)
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

/// Request sent by the SDK on the [`service_name::RECORDING_IDS`] service to resolve a
/// recording's daemon-owned cloud `recording_id`.
///
/// The recording is identified exactly the way the daemon stored it: the
/// `(robot_id, robot_instance)` source plus `timestamp_ns` — the producer's
/// capture marker returned by `start_recording`, persisted verbatim as the
/// recording row's `start_timestamp_ns`. Matching on the marker (not `<=`)
/// resolves precisely that recording, never an earlier one for the same source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingIdQuery {
    pub robot_id: String,
    pub robot_instance: i64,
    /// The recording's capture marker (Unix nanoseconds).
    pub timestamp_ns: i64,
}

/// Reply to a [`RecordingIdQuery`].
///
/// `recording_id` is `None` while the start notifier has not yet minted the
/// cloud id (or no matching, non-cancelled recording exists); the SDK re-asks
/// until it is `Some` or its own timeout elapses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingIdReply {
    /// The daemon-owned cloud recording id, once available.
    pub recording_id: Option<String>,
}

/// Request sent over [`service_name::RECORDING_STATE`] asking which recording,
/// if any, is currently open for a source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingStateQuery {
    pub robot_id: String,
    pub robot_instance: i64,
}

/// The recording a source has open, as the daemon sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveRecording {
    /// The daemon's local primary key. It exists before [`Self::recording_id`]
    /// does, so it is what tells two consecutive recordings apart.
    pub recording_index: i64,
    /// The cloud handle, once `/recording/start` has been notified. `None`
    /// while the recording is still local-only.
    pub recording_id: Option<String>,
    /// The recording's capture-clock start (Unix nanoseconds), when known.
    pub start_timestamp_ns: Option<i64>,
}

/// Reply to a [`RecordingStateQuery`].
///
/// `recording` is `None` when the source has no open recording, however it
/// came to have none: the caller acts on "recording" versus "not recording".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingStateReply {
    pub recording: Option<LiveRecording>,
}

/// Side-effect-free readiness probe sent over [`service_name::HEALTH`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthRequest {
    /// Caller-generated token echoed by the reply.
    pub nonce: u64,
}

/// Reply to a [`HealthRequest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthReply {
    /// PID of the daemon process that answered.
    pub pid: u32,
    /// Echo of [`HealthRequest::nonce`].
    pub nonce: u64,
}

/// Request sent over [`service_name::VERSION`] asking the daemon which
/// neuracore version it was built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionRequest {
    /// Caller-generated token echoed by the reply.
    pub nonce: u64,
}

/// Reply to a [`VersionRequest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionReply {
    /// Echo of [`VersionRequest::nonce`].
    pub nonce: u64,
    /// The neuracore version the answering daemon was built from.
    pub version: String,
}

impl RecordingIdQuery {
    /// Encode as a postcard byte vector for a `recording_ids` service request sample.
    pub fn encode(&self) -> Result<Vec<u8>, EnvelopeCodecError> {
        encode_postcard(self)
    }

    /// Decode from the byte slice carried in a `recording_ids` service request sample.
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvelopeCodecError> {
        decode_postcard(bytes)
    }
}

impl RecordingIdReply {
    /// Encode as a postcard byte vector for a `recording_ids` service response sample.
    pub fn encode(&self) -> Result<Vec<u8>, EnvelopeCodecError> {
        encode_postcard(self)
    }

    /// Decode from the byte slice carried in a `recording_ids` service response sample.
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvelopeCodecError> {
        decode_postcard(bytes)
    }
}

impl RecordingStateQuery {
    /// Encode as a postcard byte vector for a `recording_state` request sample.
    pub fn encode(&self) -> Result<Vec<u8>, EnvelopeCodecError> {
        encode_postcard(self)
    }

    /// Decode from the byte slice carried in a `recording_state` request sample.
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvelopeCodecError> {
        decode_postcard(bytes)
    }
}

impl RecordingStateReply {
    /// Encode as a postcard byte vector for a `recording_state` response sample.
    pub fn encode(&self) -> Result<Vec<u8>, EnvelopeCodecError> {
        encode_postcard(self)
    }

    /// Decode from the byte slice carried in a `recording_state` response sample.
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvelopeCodecError> {
        decode_postcard(bytes)
    }
}

impl HealthRequest {
    /// Encode as a postcard byte vector for a health-service request sample.
    pub fn encode(&self) -> Result<Vec<u8>, EnvelopeCodecError> {
        encode_postcard(self)
    }

    /// Decode from the byte slice carried in a health-service request sample.
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvelopeCodecError> {
        decode_postcard(bytes)
    }
}

impl HealthReply {
    /// Encode as a postcard byte vector for a health-service response sample.
    pub fn encode(&self) -> Result<Vec<u8>, EnvelopeCodecError> {
        encode_postcard(self)
    }

    /// Decode from the byte slice carried in a health-service response sample.
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvelopeCodecError> {
        decode_postcard(bytes)
    }
}

impl VersionRequest {
    /// Encode as a postcard byte vector for a version-service request sample.
    pub fn encode(&self) -> Result<Vec<u8>, EnvelopeCodecError> {
        encode_postcard(self)
    }

    /// Decode from the byte slice carried in a version-service request sample.
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvelopeCodecError> {
        decode_postcard(bytes)
    }
}

impl VersionReply {
    /// Encode as a postcard byte vector for a version-service response sample.
    pub fn encode(&self) -> Result<Vec<u8>, EnvelopeCodecError> {
        encode_postcard(self)
    }

    /// Decode from the byte slice carried in a version-service response sample.
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvelopeCodecError> {
        decode_postcard(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_request_and_reply_round_trip() {
        let request = HealthRequest { nonce: 42 };
        assert_eq!(
            HealthRequest::decode(&request.encode().unwrap()).unwrap(),
            request
        );

        let reply = HealthReply {
            pid: 1234,
            nonce: 42,
        };
        assert_eq!(
            HealthReply::decode(&reply.encode().unwrap()).unwrap(),
            reply
        );
    }

    #[test]
    fn version_request_and_reply_round_trip() {
        let request = VersionRequest { nonce: 42 };
        assert_eq!(
            VersionRequest::decode(&request.encode().unwrap()).unwrap(),
            request
        );

        let reply = VersionReply {
            nonce: 42,
            version: "13.7.2".into(),
        };
        assert_eq!(
            VersionReply::decode(&reply.encode().unwrap()).unwrap(),
            reply
        );
    }

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
            cloud_recording_id: None,
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        assert_eq!(original.kind(), "start_recording");
    }

    #[test]
    fn data_envelope_preserves_payload_bytes() {
        let original = Envelope::Data {
            robot_id: "robot-1".into(),
            robot_instance: 0,
            data_type: "JOINT_POSITIONS".into(),
            sensor_name: Some("waist".into()),
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            timestamp_ns: 1_000_000,
            timestamp_s: None,
            payload: vec![1, 2, 3, 4, 5, 6],
        };
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
        let original = Envelope::Data {
            robot_id: "robot-1".into(),
            robot_instance: 0,
            data_type: "RGB_IMAGES".into(),
            sensor_name: Some("camera_right".into()),
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            timestamp_ns: 116_666_666,
            timestamp_s: Some(7.0_f64 / 60.0_f64),
            payload: vec![0xAA, 0xBB],
        };
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
        let original = Envelope::Data {
            robot_id: "robot-1".into(),
            robot_instance: 0,
            data_type: "RGB_IMAGES".into(),
            sensor_name: None,
            publish_timestamp_ns: 0,
            timestamp_ns: 0,
            timestamp_s: None,
            payload: vec![0xAB; PAYLOAD_LEN],
        };
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
            data_type: "JOINT_POSITIONS".into(),
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            timestamp_ns: 1_700_000_000_000_000_000,
            timestamp_s: Some(1_700_000_000.5),
            items: vec![
                BatchedDataItem {
                    sensor_name: Some("joint-0".into()),
                    payload: br#"{"timestamp":1.0,"value":0.5}"#.to_vec(),
                },
                BatchedDataItem {
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
                sensor_name: Some(format!("vx300s_left_joint_{index:04}")),
                payload: br#"{"timestamp":1747740000.1234567,"value":-1.234567890123}"#.to_vec(),
            })
            .collect();
        let envelope = Envelope::BatchedData {
            robot_id: "11111111-2222-3333-4444-555555555555".into(),
            robot_instance: 0,
            data_type: "JOINT_POSITIONS".into(),
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
    fn refresh_config_round_trips() {
        let refresh = Envelope::RefreshConfig {};
        let bytes = refresh.encode().expect("encode");
        assert_eq!(refresh, Envelope::decode(&bytes).expect("decode"));
        assert_eq!(refresh.kind(), "refresh_config");
    }

    #[test]
    fn video_producer_active_round_trips() {
        let claim = Envelope::VideoProducerActive {
            robot_id: "robot-1".into(),
            robot_instance: 3,
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            producer_pid: 4242,
        };
        let bytes = claim.encode().expect("encode");
        assert_eq!(claim, Envelope::decode(&bytes).expect("decode"));
        assert_eq!(claim.kind(), "video_producer_active");
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
            producer_pid: 99,
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
            dtype: FrameDtype::Rgb8,
            frame_publish_offsets_us: vec![0, 16, 33, 50],
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        assert_eq!(original.kind(), "video_chunk_ready");
    }

    #[test]
    fn video_chunk_ready_round_trips_for_every_frame_dtype() {
        // The whole point of carrying dtype on the wire is that a depth
        // chunk's announcement survives the daemon's postcard round trip —
        // guard every variant, not just the RGB default above.
        for (data_type, dtype) in [
            ("RGB_IMAGES", FrameDtype::Rgb8),
            ("DEPTH_IMAGES", FrameDtype::DepthF16),
            ("DEPTH_IMAGES", FrameDtype::DepthF32),
        ] {
            let original = Envelope::VideoChunkReady {
                robot_id: "robot-1".into(),
                robot_instance: 0,
                data_type: data_type.into(),
                sensor_name: Some("depth_camera".into()),
                publish_timestamp_ns: 1_700_000_000_000_000_000,
                thread_id: 7,
                producer_pid: 99,
                width: 128,
                height: 128,
                byte_count: 4096,
                frame_count: 1,
                frame_timestamps_ns: vec![1_700_000_000_000_000_000],
                frame_timestamps_s: vec![1_700_000_000.0],
                dtype,
                frame_publish_offsets_us: vec![0],
            };
            let bytes = original.encode().expect("encode");
            let decoded = Envelope::decode(&bytes).expect("decode");
            assert_eq!(original, decoded, "dtype {dtype:?} did not round-trip");
        }
    }

    #[test]
    fn frame_dtype_wire_labels_round_trip() {
        assert_eq!(FrameDtype::from_wire_label("uint8"), Some(FrameDtype::Rgb8));
        assert_eq!(
            FrameDtype::from_wire_label("float16"),
            Some(FrameDtype::DepthF16)
        );
        assert_eq!(
            FrameDtype::from_wire_label("float32"),
            Some(FrameDtype::DepthF32)
        );
        assert_eq!(FrameDtype::from_wire_label("int32"), None);
        assert_eq!(FrameDtype::from_wire_label(""), None);
    }

    #[test]
    fn frame_dtype_depth_label_is_none_for_rgb() {
        // RGB must not gain a `trace.json` dtype field — only depth does.
        assert_eq!(FrameDtype::Rgb8.depth_label(), None);
        assert_eq!(FrameDtype::DepthF16.depth_label(), Some("float16"));
        assert_eq!(FrameDtype::DepthF32.depth_label(), Some("float32"));
    }

    #[test]
    fn frame_dtype_bytes_per_pixel() {
        assert_eq!(FrameDtype::Rgb8.bytes_per_pixel(), 3);
        assert_eq!(FrameDtype::DepthF16.bytes_per_pixel(), 2);
        assert_eq!(FrameDtype::DepthF32.bytes_per_pixel(), 4);
    }

    #[test]
    fn video_chunk_ready_worst_case_fits_commands_slice() {
        // Two timestamps plus a publish offset per frame stays well under
        // COMMANDS_MAX_PAYLOAD_BYTES even at an implausible frame count.
        let frame_timestamps_ns: Vec<i64> = (0..10_000).map(|i| i as i64 * 1_000_000).collect();
        let frame_timestamps_s: Vec<f64> = (0..10_000).map(|i| i as f64 * 1e-3).collect();
        let frame_publish_offsets_us: Vec<u32> = (0..10_000).collect();
        let envelope = Envelope::VideoChunkReady {
            robot_id: "11111111-2222-3333-4444-555555555555".into(),
            robot_instance: 0,
            data_type: "RGB_IMAGES".into(),
            sensor_name: Some("camera_right".into()),
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            thread_id: 42,
            producer_pid: 99,
            width: 1920,
            height: 1080,
            byte_count: 128 * 1024 * 1024,
            frame_count: frame_timestamps_ns.len() as u32,
            frame_timestamps_ns,
            frame_timestamps_s,
            dtype: FrameDtype::Rgb8,
            frame_publish_offsets_us,
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
        // i64 (10-byte postcard zigzag varint), every publish offset a
        // full-range u32 (5-byte varint), and every fixed field maxed out.
        // Without the cap a long recording of tiny frames overflows the slice
        // and the whole recording's video announcement fails to publish.
        let count = service_name::MAX_VIDEO_CHUNK_FRAMES as usize;
        let frame_timestamps_ns: Vec<i64> = (0..count).map(|i| i64::MAX - i as i64).collect();
        let frame_timestamps_s: Vec<f64> = (0..count).map(|i| i as f64).collect();
        let frame_publish_offsets_us: Vec<u32> = (0..count).map(|_| u32::MAX).collect();
        let envelope = Envelope::VideoChunkReady {
            robot_id: "11111111-2222-3333-4444-555555555555".into(),
            robot_instance: i64::MAX,
            data_type: "RGB_IMAGES".into(),
            sensor_name: Some("camera_with_a_deliberately_long_sensor_label".into()),
            publish_timestamp_ns: i64::MAX,
            thread_id: i64::MAX,
            producer_pid: u32::MAX,
            width: u32::MAX,
            height: u32::MAX,
            byte_count: u64::MAX,
            frame_count: count as u32,
            frame_timestamps_ns,
            frame_timestamps_s,
            dtype: FrameDtype::Rgb8,
            frame_publish_offsets_us,
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
