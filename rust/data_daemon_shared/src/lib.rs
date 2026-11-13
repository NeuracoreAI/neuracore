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
//! `timestamp_us`. The producer publishes three fire-and-forget lifecycle
//! events ([`Envelope::StartRecording`] / [`Envelope::StopRecording`] /
//! [`Envelope::CancelRecording`]) carrying the lifecycle wall-clock timestamp,
//! and the daemon decides — from its per-source active-window map — which
//! recording (if any) each datum belongs to. There is no `recording_index`,
//! `trace_id`, or `sequence_number` on the wire; the daemon assigns and
//! stores those after routing.
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

/// Nanoseconds in one microsecond, the unit of the caller timestamps the
/// daemon stores.
pub const NANOSECONDS_PER_MICROSECOND: i64 = 1_000;

/// Microseconds in one second.
pub const MICROSECONDS_PER_SECOND: f64 = 1_000_000.0;

/// A timestamp in microseconds as float seconds.
pub fn microseconds_to_seconds(timestamp_us: i64) -> f64 {
    timestamp_us as f64 / MICROSECONDS_PER_SECOND
}

/// Float seconds as whole microseconds, rounded to nearest with ties to even
/// as the Python SDK rounds them.
pub fn seconds_to_microseconds(seconds: f64) -> i64 {
    (seconds * MICROSECONDS_PER_SECOND).round_ties_even() as i64
}

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
    /// about 110 KiB, and `VideoChunkReady`'s `frame_timestamps_us` vector is
    /// ~30 KiB even for a 128 MiB 1080p chunk. 1 MiB leaves generous
    /// headroom for the worst case.
    pub const COMMANDS_MAX_PAYLOAD_BYTES: usize = 1024 * 1024;

    /// Worst-case postcard size of one frame's contribution to a
    /// [`crate::Envelope::VideoChunkReady`] announcement: a `frame_timestamps_us`
    /// element is an `i64` zigzag varint (at most 10 bytes for a full-range value), a
    /// `frame_publish_offsets_us` element is a `u32` varint (at most 5 bytes), and a
    /// `frame_calibrations` element is at most [`FRAME_CALIBRATION_MAX_BYTES`].
    pub const VIDEO_CHUNK_BYTES_PER_FRAME: usize = 10 + 5 + FRAME_CALIBRATION_MAX_BYTES;

    /// Worst-case postcard size of one [`crate::FrameCalibration`]: two option
    /// tags plus nine and sixteen fixed 4-byte `f32` values.
    pub const FRAME_CALIBRATION_MAX_BYTES: usize = 1 + 9 * 4 + 1 + 16 * 4;

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
    /// per-frame `frame_timestamps_us` and `frame_publish_offsets_us`
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

    /// The microsecond clock of every capture timestamp and of every stage of
    /// the video path: a frame's PTS is its `timestamp_us` minus the chunk
    /// origin, the producer writes spool NUT chunks with a `1/1_000_000` time
    /// base, and the daemon pins its per-chunk encode outputs to the same
    /// clock (`-enc_time_base` / `-video_track_timescale`). Sharing one constant
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
    /// integration matrix's 1000-joint worst case encodes to about 110 KiB, so
    /// a 1024-deep buffer would retain about 110 MiB of pages per publisher and
    /// exhaust the 64 MiB devcontainer `/dev/shm`.
    ///
    /// 64 keeps that worst case at about 7 MiB per publisher while staying deep
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

    /// Request-response service the SDK and `data-daemon status` use to ask
    /// what the daemon is waiting on before recordings can upload.
    ///
    /// A service of its own rather than more of [`crate::VersionReply`]: the
    /// version check has to decode replies from daemons of every version, and
    /// it runs on every `start_recording`, while answering this one reads the
    /// state store. See [`crate::StatusRequest`] / [`crate::StatusReply`].
    pub const STATUS: &str = "neuracore/data_daemon/status";

    /// Maximum size of a single status-service sample. The reply is a few
    /// counts and at most a handful of short reason codes.
    pub const STATUS_MAX_PAYLOAD_BYTES: usize = 4 * 1024;

    /// Request-response service the SDK uses to read a source's *current*
    /// recording state.
    ///
    /// The daemon is the only party subscribed to the backend's org-wide
    /// notification stream, so it is the only one that knows a recording
    /// nothing local bracketed. It answers both "is this source recording?" and
    /// "under what cloud id?", which is why the SDK needs no second service to
    /// resolve an id.
    ///
    /// See [`crate::RecordingStateQuery`] / [`crate::RecordingStateReply`].
    pub const RECORDING_STATE: &str = "neuracore/data_daemon/recording_state";

    /// Maximum size of a single `recording_state` service sample. Both the
    /// request and the reply are a handful of UUID strings + integers; 4 KiB is
    /// generous.
    pub const RECORDING_STATE_MAX_PAYLOAD_BYTES: usize = 4 * 1024;

    /// Maximum number of concurrent request-response clients. Mirrors
    /// [`MAX_PUBLISHERS_PER_SERVICE`]: the data bridge parks one client port
    /// per OS thread (iceoryx2 ports are `!Sync`), so the cap must cover the
    /// integration matrix's full thread fan-out.
    pub const MAX_REQUEST_RESPONSE_CLIENTS_PER_SERVICE: usize = 128;

    /// Maximum number of concurrent request-response servers. The daemon opens exactly one.
    pub const MAX_REQUEST_RESPONSE_SERVERS_PER_SERVICE: usize = 1;
}

/// Who a message or a recording belongs to: `(robot_id, robot_instance)`.
///
/// The one identity the producer and the daemon both speak — every [`Envelope`]
/// is tagged with it and [`RecordingStateQuery`] carries both halves — so it
/// belongs here rather than being re-declared privately on each side of the
/// wire.
///
/// An alias rather than a struct because the two sides use it differently: the
/// daemon keys a map with it, while the bridge passes it as a message and
/// deliberately nests its maps instead of keying on it (a tuple key cannot be
/// looked up by borrow, which would put a `String` allocation on the log path).
/// A struct would inherit that same limitation without paying for itself.
pub type Source = (String, i64);

/// A single message exchanged between the producer and the daemon.
///
/// Every variant is tagged with its **source** (`robot_id`, `robot_instance`).
/// Data variants additionally carry their **sensor** (`data_type`,
/// `sensor_name`) and capture `timestamp_us`. No recording or trace identity
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
        /// Caller-supplied capture time (microseconds) for the recording's
        /// start — the recording's *own* clock, or the publish time when the
        /// caller supplied none. Stored as the row's start and POSTed to the
        /// backend as `start_time`; never used for routing.
        timestamp_us: i64,
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
        /// Caller-supplied capture time (microseconds) for the recording's
        /// stop, or the publish time when the caller supplied none. Stored as
        /// the row's stop and POSTed to the backend as `end_time`; never used
        /// for routing.
        timestamp_us: i64,
    },
    /// Producer cancels the source's active recording — the daemon drops every
    /// in-flight per-trace actor, deletes the on-disk artefacts, marks the
    /// recording row cancelled, and uploads nothing. Processed immediately on
    /// arrival; the daemon is idempotent.
    CancelRecording {
        robot_id: String,
        robot_instance: i64,
        /// Caller-supplied capture time (microseconds) for the cancel, or
        /// the publish time when the caller supplied none. A cancel is a
        /// recording stop that discards data, so the daemon stores this as the
        /// row's stop and POSTs it as the backend `end_time`, exactly like
        /// `StopRecording`. No window-boundary `publish_timestamp_ns`
        /// is carried because cancelling drops the window outright.
        timestamp_us: i64,
    },
    /// Producer delivers one sensor sample.
    ///
    /// The payload is opaque to the IPC layer; the per-trace actor parses it
    /// according to `data_type` and writes it through the JSON writer. The
    /// daemon holds the datum for the configured holdback, then routes it into
    /// the source's window whose `[started_at_ns, stopped_at_ns)` contains
    /// `publish_timestamp_ns`.
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
        /// Caller-supplied capture time in microseconds on the data's *own*
        /// clock, written into the trace content. Not used for routing.
        timestamp_us: i64,
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
    /// `publish_timestamp_ns`, the whole batch belongs to one window, and the
    /// daemon holds and routes it as a single unit.
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
        /// Caller-supplied capture time in microseconds, shared by every item.
        /// It is content, not routing.
        timestamp_us: i64,
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
    /// the recording, and encode it to a sealed MP4 segment. Per-frame capture
    /// times in microseconds are carried inline for the daemon-side
    /// `trace.json` sidecar.
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
        /// Per-frame capture time in microseconds, in arrival order. Length equals
        /// `frame_count`. Capture-clock content for the trace sidecar and the
        /// spooled PTS; routing uses `frame_publish_offsets_us`.
        frame_timestamps_us: Vec<i64>,
        /// Per-frame camera calibration in arrival order. Length equals
        /// `frame_count`.
        frame_calibrations: Vec<FrameCalibration>,
        /// Stored representation of every frame in this chunk (never mixed:
        /// the producer seals and reopens a chunk on a dtype or depth scale
        /// change, like a geometry change). Depth chunks carry
        /// [`FrameDtype::DepthU16`] with their scale, which picks the depth
        /// encode and lands in the trace's `trace.json` sidecar.
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
    /// [`Envelope::StartRecording`] for a recording whose robot or dataset
    /// was made offline and so has only a local id.
    ///
    /// Opens the window and inserts the row exactly as `StartRecording` does,
    /// and also carries what the daemon needs to find or create the robot and
    /// dataset on the backend once it can reach it. A separate variant rather
    /// than new `StartRecording` fields because postcard does not tag fields:
    /// appending a variant leaves every existing envelope's bytes unchanged.
    StartLocalRecording {
        /// The source's robot id: a local id when `local_robot` is set.
        robot_id: String,
        robot_instance: i64,
        /// The dataset id: a local id when `local_dataset` is set.
        dataset_id: Option<String>,
        /// As [`Envelope::StartRecording`]'s `publish_timestamp_ns`.
        publish_timestamp_ns: i64,
        /// As [`Envelope::StartRecording`]'s `timestamp_us`.
        timestamp_us: i64,
        /// Set when `robot_id` is a local id.
        local_robot: Option<LocalRobot>,
        /// Set when `dataset_id` is a local id.
        local_dataset: Option<LocalDataset>,
    },
}

/// A robot connected offline, as the backend's robot upsert needs it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalRobot {
    /// The robot name.
    pub name: String,
    /// Whether the robot is shared.
    pub shared: bool,
    /// Whether its saved URDF package replaces one the backend already has,
    /// as `connect_robot`'s `overwrite_existing_urdf`.
    pub overwrite_urdf: bool,
}

/// A dataset selected offline, as the backend's dataset create needs it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalDataset {
    /// The dataset name.
    pub name: String,
    /// Whether the dataset is shared.
    pub shared: bool,
    /// Description, used if the dataset is created.
    pub description: Option<String>,
    /// Tags, used if the dataset is created.
    pub tags: Vec<String>,
}

/// Camera calibration logged with one video frame.
///
/// `intrinsics` is the row-major 3x3 pinhole matrix and `extrinsics` the
/// row-major 4x4 camera pose, both exactly as the caller logged them. The
/// daemon writes them into the frame's `trace.json` entry as nested lists.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameCalibration {
    /// Row-major 3x3 intrinsics matrix.
    pub intrinsics: Option<[f32; 9]>,
    /// Row-major 4x4 extrinsics matrix.
    pub extrinsics: Option<[f32; 16]>,
}

/// Pixel or sample representation of one video-family frame.
///
/// On the producer side it describes the buffer a caller logs. On
/// [`Envelope::VideoChunkReady`] it describes what the chunk stores: packed
/// RGB24 for RGB, and uint16 grey samples with their metres-per-unit scale for
/// depth. The daemon reads only this label and never the pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameDtype {
    /// Packed RGB24, one byte per channel.
    Rgb8,
    /// 2D depth frame, little-endian uint16 sensor units where 0 means no
    /// return. `scale_bits` holds the metres per unit as `f64::to_bits`, so
    /// the dtype stays `Eq` and a scale change seals a chunk like a dtype
    /// change does.
    DepthU16 {
        /// Metres per unit as `f64::to_bits`.
        scale_bits: u64,
    },
}

impl FrameDtype {
    /// uint16 depth at `depth_scale_m` metres per unit.
    pub fn depth_u16(depth_scale_m: f64) -> Self {
        FrameDtype::DepthU16 {
            scale_bits: depth_scale_m.to_bits(),
        }
    }

    /// Metres per unit of a uint16 depth frame, `None` for every other dtype.
    pub fn depth_scale_m(self) -> Option<f64> {
        match self {
            FrameDtype::DepthU16 { scale_bits } => Some(f64::from_bits(scale_bits)),
            FrameDtype::Rgb8 => None,
        }
    }

    /// Whether this is a depth representation.
    pub fn is_depth(self) -> bool {
        !matches!(self, FrameDtype::Rgb8)
    }

    /// Bytes occupied by one pixel or sample of this representation.
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            FrameDtype::Rgb8 => 3,
            FrameDtype::DepthU16 { .. } => 2,
        }
    }

    /// Parse the Python-facing wire label, which is `numpy.dtype.name`:
    /// `"uint8"` for RGB and `"uint16"` for depth in sensor units, which also
    /// needs a positive finite `depth_scale_m`. `None` for anything else, so
    /// the native boundary rejects an unsupported dtype instead of misreading
    /// the buffer.
    pub fn from_wire_label(label: &str, depth_scale_m: Option<f64>) -> Option<Self> {
        match label {
            "uint8" => Some(FrameDtype::Rgb8),
            "uint16" => depth_scale_m
                .filter(|scale| scale.is_finite() && *scale > 0.0)
                .map(Self::depth_u16),
            _ => None,
        }
    }
}

/// One sensor's sample inside an [`Envelope::BatchedData`] batch.
///
/// Carries only the fields that differ between items: `data_type` and
/// `timestamp_us` are hoisted onto the parent envelope
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
            Envelope::StartLocalRecording { .. } => "start_local_recording",
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
    /// The daemon's local primary key, once data has opened the recording's
    /// window. It exists before [`Self::recording_id`] does, so it is what
    /// tells two consecutive recordings apart. `None` for a recording the
    /// backend has announced that this host has not yet published for.
    pub recording_index: Option<i64>,
    /// The cloud handle, once `/recording/start` has been notified. `None`
    /// while the recording is still local-only.
    pub recording_id: Option<String>,
    /// The recording's capture-clock start (microseconds), when known.
    pub start_timestamp_us: Option<i64>,
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

/// Why a robot or dataset made offline is stuck `pending`, or why a recording
/// will never upload: the codes the daemon stores in `local_robots.detail` and
/// `local_datasets.detail`, and the ones it reports in [`StatusReply`].
pub mod resolution_detail {
    /// Pending: the robot is archived. Unarchiving it lets it register.
    pub const ROBOT_ARCHIVED: &str = "robot_archived";
    /// Pending: the user is not in the org. Selecting another org, or being
    /// added to this one, lets it through.
    pub const NOT_IN_ORG: &str = "not_in_org";
    /// Pending: the backend rejected the API key, or none is saved. Logging
    /// in fixes it.
    pub const CREDENTIALS: &str = "credentials";
    /// Pending: a dataset of the other kind (shared or private) has the
    /// dataset's name. Renaming or deleting that one lets it through.
    pub const NAME_TAKEN: &str = "name_taken";
    /// Failed: the user may not create shared datasets.
    pub const SHARED_NOT_ALLOWED: &str = "shared_not_allowed";
    /// Failed: the backend rejected the request itself.
    pub const REJECTED: &str = "rejected";
    /// Failed: part of the recording could not be saved or uploaded. Only
    /// reported, never stored.
    pub const UPLOAD_FAILED: &str = "upload_failed";
    /// Failed: the server no longer holds the recording. Only reported, never
    /// stored.
    pub const NOT_ON_SERVER: &str = "not_on_server";

    /// What a reason code means to the user.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Explanation {
        /// Why the recordings are held back or failed, as a clause:
        /// "the robot is archived".
        pub cause: &'static str,
        /// What the user can do about it, as a sentence, if anything.
        pub fix: Option<&'static str>,
    }

    /// What `code` means to the user, or `None` for a code this build does
    /// not know.
    pub fn explain(code: &str) -> Option<Explanation> {
        let (cause, fix) = match code {
            ROBOT_ARCHIVED => (
                "the robot is archived",
                Some("Unarchive it on the Archived Robots page of the Neuracore dashboard."),
            ),
            NOT_IN_ORG => (
                "the account is not in the selected organization",
                Some("Select another with `neuracore select-org`."),
            ),
            CREDENTIALS => (
                "the API key was rejected or is missing",
                Some("Log in with `neuracore login`."),
            ),
            NAME_TAKEN => (
                "a dataset of the other kind, shared or private, has the dataset's name",
                Some("Rename or delete that dataset on the Neuracore dashboard."),
            ),
            SHARED_NOT_ALLOWED => ("the account may not create shared datasets", None),
            REJECTED => ("the server rejected the robot or dataset", None),
            UPLOAD_FAILED => ("part of the recording could not be saved or uploaded", None),
            NOT_ON_SERVER => ("the server no longer holds the recording", None),
            _ => return None,
        };
        Some(Explanation { cause, fix })
    }
}

/// Request sent over [`service_name::STATUS`] asking what the daemon is
/// waiting on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatusRequest {
    /// Caller-generated token echoed by the reply.
    pub nonce: u64,
}

/// Reply to a [`StatusRequest`]: what stands between the recordings on this
/// machine and the server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatusReply {
    /// Echo of [`StatusRequest::nonce`].
    pub nonce: u64,
    /// Whether the daemon can reach the server.
    pub connection: Connection,
    /// Whether an API key is saved for the daemon to upload with.
    pub api_key_saved: bool,
    /// Whether an organization is selected to upload to.
    pub org_selected: bool,
    /// Recordings the server has not heard of yet that can still upload,
    /// including the blocked ones.
    pub waiting: RecordingCount,
    /// Recordings the server has started whose data is still uploading.
    pub uploading: RecordingCount,
    /// Why waiting recordings are held back until the user fixes something,
    /// by [`resolution_detail`] code.
    pub blocked: Vec<ReasonCount>,
    /// Why recordings will never upload, by [`resolution_detail`] code.
    pub failed: Vec<ReasonCount>,
}

/// Whether the daemon can reach the server, as far as it knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Connection {
    /// The daemon has just started and has not checked yet.
    NotChecked,
    /// The latest check reached the server.
    Reachable,
    /// The latest check did not reach the server.
    Unreachable,
    /// The daemon's profile is set to offline, so it does not try.
    OfflineProfile,
}

impl Connection {
    /// The name the SDK sees: `"not_checked"`, `"reachable"`,
    /// `"unreachable"` or `"offline_profile"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotChecked => "not_checked",
            Self::Reachable => "reachable",
            Self::Unreachable => "unreachable",
            Self::OfflineProfile => "offline_profile",
        }
    }
}

/// How many recordings are in one state, and how much of their data is left
/// to upload.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingCount {
    /// Number of recordings.
    pub recordings: u64,
    /// Bytes of their data not uploaded yet.
    pub bytes: u64,
}

/// How many recordings share one reason code, and the robot or dataset it
/// is about.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasonCount {
    /// The reason code.
    pub reason: String,
    /// Number of recordings with it.
    pub recordings: u64,
    /// The robot or dataset the reason is about, so the user knows which.
    /// `None` for a reason about the account (a rejected key, not being in
    /// the org) or one past [`MAX_NAMED_REASONS`].
    pub subject: Option<Subject>,
}

/// A robot or dataset made offline, as the user named it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Subject {
    /// Whether it is a robot or a dataset.
    pub kind: SubjectKind,
    /// Its name, cut to at most [`MAX_SUBJECT_NAME_BYTES`].
    pub name: String,
}

/// What a [`Subject`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubjectKind {
    /// A robot.
    Robot,
    /// A dataset.
    Dataset,
}

impl SubjectKind {
    /// "robot" or "dataset", as stored and as shown.
    pub fn as_str(self) -> &'static str {
        match self {
            SubjectKind::Robot => "robot",
            SubjectKind::Dataset => "dataset",
        }
    }

    /// Parse [`Self::as_str`].
    pub fn parse(kind: &str) -> Option<Self> {
        match kind {
            "robot" => Some(SubjectKind::Robot),
            "dataset" => Some(SubjectKind::Dataset),
            _ => None,
        }
    }
}

/// How many reasons a [`StatusReply`] list names a robot or dataset for. The
/// rest are merged, one per reason code, so the reply stays within
/// [`service_name::STATUS_MAX_PAYLOAD_BYTES`].
pub const MAX_NAMED_REASONS: usize = 10;

/// How much of a robot or dataset name a [`StatusReply`] carries.
pub const MAX_SUBJECT_NAME_BYTES: usize = 64;

impl Subject {
    /// A subject named `name`, cut on a character boundary to
    /// [`MAX_SUBJECT_NAME_BYTES`].
    pub fn new(kind: SubjectKind, name: &str) -> Self {
        let mut end = name.len().min(MAX_SUBJECT_NAME_BYTES);
        while !name.is_char_boundary(end) {
            end -= 1;
        }
        Subject {
            kind,
            name: name[..end].to_string(),
        }
    }
}

/// Keep the first [`MAX_NAMED_REASONS`] named reasons of `reasons` and merge
/// the rest into one unnamed entry per reason code, so a [`StatusReply`]
/// fits its service. Order is kept.
pub fn limit_reasons(reasons: Vec<ReasonCount>) -> Vec<ReasonCount> {
    let mut named = 0;
    let mut kept: Vec<ReasonCount> = Vec::new();
    for reason in reasons {
        if reason.subject.is_some() {
            named += 1;
            if named <= MAX_NAMED_REASONS {
                kept.push(reason);
                continue;
            }
        }
        match kept
            .iter_mut()
            .find(|kept| kept.subject.is_none() && kept.reason == reason.reason)
        {
            Some(merged) => {
                merged.recordings = merged.recordings.saturating_add(reason.recordings);
            }
            None => kept.push(ReasonCount {
                subject: None,
                ..reason
            }),
        }
    }
    kept
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

impl StatusRequest {
    /// Encode as a postcard byte vector for a status-service request sample.
    pub fn encode(&self) -> Result<Vec<u8>, EnvelopeCodecError> {
        encode_postcard(self)
    }

    /// Decode from the byte slice carried in a status-service request sample.
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvelopeCodecError> {
        decode_postcard(bytes)
    }
}

impl StatusReply {
    /// Encode as a postcard byte vector for a status-service response sample.
    pub fn encode(&self) -> Result<Vec<u8>, EnvelopeCodecError> {
        encode_postcard(self)
    }

    /// Decode from the byte slice carried in a status-service response sample.
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvelopeCodecError> {
        decode_postcard(bytes)
    }

    /// What the user needs to hear about, one sentence each: what they must
    /// fix for recordings to upload, recordings that never will and when they
    /// are deleted, and recordings held on this machine because the server
    /// cannot be reached. Empty when nothing is wrong.
    pub fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if !self.api_key_saved {
            problems.push(
                "No API key is saved: log in with `neuracore login` so recordings can upload."
                    .to_string(),
            );
        }
        if !self.org_selected {
            problems.push(
                "No organization is selected: select one with `neuracore select-org` so \
                 recordings can upload."
                    .to_string(),
            );
        }
        for blocked in &self.blocked {
            problems.push(reason_sentence(blocked, "is waiting", "are waiting"));
        }
        if !self.blocked.is_empty() {
            // Blocked recordings are kept until the problem is fixed, however
            // long that takes; reset is the only way to give up on them.
            problems.push(DISCARD_BLOCKED.to_string());
        }
        for failed in &self.failed {
            problems.push(reason_sentence(
                failed,
                "will not upload",
                "will not upload",
            ));
        }
        let held = self
            .waiting
            .recordings
            .saturating_add(self.uploading.recordings);
        let until = match self.connection {
            Connection::Unreachable => "until the Neuracore server can be reached.",
            Connection::OfflineProfile => {
                "because the daemon's profile is set to offline. Relaunch the daemon \
                 without offline mode to upload them."
            }
            Connection::NotChecked | Connection::Reachable => "",
        };
        if held > 0 && !until.is_empty() {
            let bytes = self.waiting.bytes.saturating_add(self.uploading.bytes);
            problems.push(format!(
                "{} ({}) {} kept on this machine {until}",
                recordings(held),
                format_bytes(bytes),
                if held == 1 { "is" } else { "are" },
            ));
        }
        problems
    }

    /// Everything `neuracore data-daemon status` shows: the connection, the
    /// recordings still to upload, then the [`problems`](Self::problems).
    pub fn summary(&self) -> Vec<String> {
        let connection = match self.connection {
            Connection::Reachable => "Neuracore server: reachable.",
            Connection::Unreachable => "Neuracore server: not reachable.",
            Connection::NotChecked => "Neuracore server: not checked yet.",
            Connection::OfflineProfile => {
                "Neuracore server: not contacted (the daemon's profile is set to offline)."
            }
        };
        let mut summary = vec![
            connection.to_string(),
            format!(
                "Waiting to start on the server: {} ({}).",
                recordings(self.waiting.recordings),
                format_bytes(self.waiting.bytes),
            ),
            format!(
                "Uploading: {} ({} left).",
                recordings(self.uploading.recordings),
                format_bytes(self.uploading.bytes),
            ),
        ];
        summary.extend(self.problems());
        summary
    }
}

/// Shown after the recordings waiting on a problem the user must fix, which
/// are kept until it is fixed.
const DISCARD_BLOCKED: &str = "Waiting recordings are kept until the problem is fixed. To \
     discard them instead, run `neuracore data-daemon reset`: it deletes every recording on \
     this machine that has not finished uploading, not only these.";

/// "2 recordings are waiting because the robot is archived. Unarchive it ...".
fn reason_sentence(reason: &ReasonCount, one: &str, many: &str) -> String {
    let verb = if reason.recordings == 1 { one } else { many };
    let count = recordings(reason.recordings);
    let sentence = match resolution_detail::explain(&reason.reason) {
        Some(explanation) => {
            let fix = explanation
                .fix
                .map(|fix| format!(" {fix}"))
                .unwrap_or_default();
            let cause = match &reason.subject {
                Some(subject) => named_cause(&reason.reason, subject)
                    .unwrap_or_else(|| explanation.cause.to_string()),
                None => explanation.cause.to_string(),
            };
            format!("{count} {verb} because {cause}.{fix}")
        }
        None => match &reason.subject {
            Some(subject) => format!(
                "{count} {verb} (reason: {}, {} \"{}\").",
                reason.reason,
                subject.kind.as_str(),
                subject.name
            ),
            None => format!("{count} {verb} (reason: {}).", reason.reason),
        },
    };
    sentence
}

/// [`resolution_detail::explain`]'s cause naming the robot or dataset, for
/// the codes about one.
fn named_cause(code: &str, subject: &Subject) -> Option<String> {
    let Subject { kind, name } = subject;
    let kind = kind.as_str();
    Some(match code {
        resolution_detail::ROBOT_ARCHIVED => format!("the robot \"{name}\" is archived"),
        resolution_detail::NAME_TAKEN => {
            format!("a dataset of the other kind, shared or private, has the name \"{name}\"")
        }
        resolution_detail::SHARED_NOT_ALLOWED => {
            format!("the account may not create the shared dataset \"{name}\"")
        }
        resolution_detail::REJECTED => format!("the server rejected the {kind} \"{name}\""),
        _ => return None,
    })
}

/// "1 recording", "3 recordings".
fn recordings(count: u64) -> String {
    if count == 1 {
        "1 recording".to_string()
    } else {
        format!("{count} recordings")
    }
}

/// A byte count in decimal units: "512 B", "1.5 MB".
fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["KB", "MB", "GB", "TB", "PB"];
    if bytes < 1_000 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1_000.0;
    let mut unit = 0;
    while value >= 1_000.0 && unit + 1 < UNITS.len() {
        value /= 1_000.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seconds_to_microseconds_rounds_where_truncation_falls_short() {
        assert_eq!(
            seconds_to_microseconds(1_771_669_371.383_135),
            1_771_669_371_383_135
        );
    }

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

    fn status_reply(blocked: Vec<ReasonCount>, failed: Vec<ReasonCount>) -> StatusReply {
        StatusReply {
            nonce: 7,
            connection: Connection::Unreachable,
            api_key_saved: true,
            org_selected: false,
            waiting: RecordingCount {
                recordings: 3,
                bytes: 4_096,
            },
            uploading: RecordingCount::default(),
            blocked,
            failed,
        }
    }

    #[test]
    fn status_request_and_reply_round_trip() {
        let request = StatusRequest { nonce: 7 };
        assert_eq!(
            StatusRequest::decode(&request.encode().unwrap()).unwrap(),
            request
        );

        let reply = status_reply(
            vec![ReasonCount {
                reason: "robot_archived".into(),
                recordings: 2,
                subject: Some(Subject::new(SubjectKind::Robot, "arm")),
            }],
            vec![ReasonCount {
                reason: "shared_not_allowed".into(),
                recordings: 1,
                subject: None,
            }],
        );
        assert_eq!(
            StatusReply::decode(&reply.encode().unwrap()).unwrap(),
            reply
        );

        let not_checked_yet = StatusReply {
            connection: Connection::NotChecked,
            ..reply
        };
        assert_eq!(
            StatusReply::decode(&not_checked_yet.encode().unwrap()).unwrap(),
            not_checked_yet
        );
    }

    fn healthy() -> StatusReply {
        StatusReply {
            nonce: 1,
            connection: Connection::Reachable,
            api_key_saved: true,
            org_selected: true,
            waiting: RecordingCount::default(),
            uploading: RecordingCount::default(),
            blocked: Vec::new(),
            failed: Vec::new(),
        }
    }

    fn reason(reason: &str, recordings: u64) -> ReasonCount {
        ReasonCount {
            reason: reason.into(),
            recordings,
            subject: None,
        }
    }

    fn about(reason: ReasonCount, kind: SubjectKind, name: &str) -> ReasonCount {
        ReasonCount {
            subject: Some(Subject::new(kind, name)),
            ..reason
        }
    }

    #[test]
    fn nothing_to_report_when_recordings_are_uploading_normally() {
        let reply = StatusReply {
            waiting: RecordingCount {
                recordings: 2,
                bytes: 10,
            },
            uploading: RecordingCount {
                recordings: 1,
                bytes: 10,
            },
            ..healthy()
        };

        assert!(reply.problems().is_empty());
    }

    #[test]
    fn problems_say_what_to_fix_for_each_reason() {
        let reply = StatusReply {
            api_key_saved: false,
            org_selected: false,
            blocked: vec![
                reason(resolution_detail::ROBOT_ARCHIVED, 2),
                reason(resolution_detail::CREDENTIALS, 1),
            ],
            failed: vec![reason(resolution_detail::SHARED_NOT_ALLOWED, 1)],
            ..healthy()
        };

        assert_eq!(
            reply.problems(),
            vec![
                "No API key is saved: log in with `neuracore login` so recordings can upload."
                    .to_string(),
                "No organization is selected: select one with `neuracore select-org` so \
                 recordings can upload."
                    .to_string(),
                "2 recordings are waiting because the robot is archived. Unarchive it on the \
                 Archived Robots page of the Neuracore dashboard."
                    .to_string(),
                "1 recording is waiting because the API key was rejected or is missing. Log in \
                 with `neuracore login`."
                    .to_string(),
                "Waiting recordings are kept until the problem is fixed. To discard them \
                 instead, run `neuracore data-daemon reset`: it deletes every recording on this \
                 machine that has not finished uploading, not only these."
                    .to_string(),
                "1 recording will not upload because the account may not create shared datasets."
                    .to_string(),
            ]
        );
    }

    #[test]
    fn recordings_held_offline_are_reported_with_their_size() {
        let reply = StatusReply {
            connection: Connection::Unreachable,
            waiting: RecordingCount {
                recordings: 3,
                bytes: 1_500_000,
            },
            uploading: RecordingCount {
                recordings: 1,
                bytes: 2_000_000_000,
            },
            ..healthy()
        };

        assert_eq!(
            reply.problems(),
            vec![
                "4 recordings (2.0 GB) are kept on this machine until the Neuracore server can \
                 be reached."
                    .to_string()
            ]
        );
        // Nothing to say before the daemon has checked, or with nothing held.
        let not_checked = StatusReply {
            connection: Connection::NotChecked,
            ..reply.clone()
        };
        assert!(not_checked.problems().is_empty());
        let nothing_held = StatusReply {
            connection: Connection::Unreachable,
            ..healthy()
        };
        assert!(nothing_held.problems().is_empty());
    }

    #[test]
    fn recordings_held_by_an_offline_profile_say_how_to_upload_them() {
        let reply = StatusReply {
            connection: Connection::OfflineProfile,
            waiting: RecordingCount {
                recordings: 1,
                bytes: 2_000,
            },
            ..healthy()
        };

        assert_eq!(
            reply.problems(),
            vec![
                "1 recording (2.0 KB) is kept on this machine because the daemon's profile is \
                 set to offline. Relaunch the daemon without offline mode to upload them."
                    .to_string()
            ]
        );
        assert_eq!(
            reply.summary()[0],
            "Neuracore server: not contacted (the daemon's profile is set to offline)."
        );
    }

    #[test]
    fn problems_name_the_robot_or_dataset() {
        let reply = StatusReply {
            blocked: vec![
                about(
                    reason(resolution_detail::ROBOT_ARCHIVED, 2),
                    SubjectKind::Robot,
                    "arm",
                ),
                about(
                    reason(resolution_detail::NAME_TAKEN, 1),
                    SubjectKind::Dataset,
                    "warehouse",
                ),
            ],
            failed: vec![about(
                reason(resolution_detail::REJECTED, 1),
                SubjectKind::Dataset,
                "pick-test",
            )],
            ..healthy()
        };

        let problems = reply.problems();

        assert_eq!(
            problems[0],
            "2 recordings are waiting because the robot \"arm\" is archived. Unarchive it on \
             the Archived Robots page of the Neuracore dashboard."
        );
        assert_eq!(
            problems[1],
            "1 recording is waiting because a dataset of the other kind, shared or private, \
             has the name \"warehouse\". Rename or delete that dataset on the Neuracore \
             dashboard."
        );
        assert_eq!(
            problems[3],
            "1 recording will not upload because the server rejected the dataset \"pick-test\"."
        );
    }

    #[test]
    fn names_past_the_limit_are_merged_by_reason() {
        let named = |name: &str| {
            about(
                reason(resolution_detail::REJECTED, 1),
                SubjectKind::Robot,
                name,
            )
        };
        let reasons: Vec<ReasonCount> = (0..MAX_NAMED_REASONS + 3)
            .map(|index| named(&format!("arm-{index}")))
            .collect();

        let limited = limit_reasons(reasons);

        assert_eq!(limited.len(), MAX_NAMED_REASONS + 1);
        assert!(limited[..MAX_NAMED_REASONS]
            .iter()
            .all(|reason| reason.subject.is_some()));
        let merged = &limited[MAX_NAMED_REASONS];
        assert_eq!(merged.subject, None);
        assert_eq!(merged.recordings, 3);
    }

    #[test]
    fn a_long_name_is_cut_on_a_character_boundary() {
        let name = format!("{}\u{e9}", "a".repeat(MAX_SUBJECT_NAME_BYTES - 1));

        let subject = Subject::new(SubjectKind::Robot, &name);

        assert_eq!(subject.name, "a".repeat(MAX_SUBJECT_NAME_BYTES - 1));
    }

    #[test]
    fn a_reason_this_build_does_not_know_is_shown_by_its_code() {
        let reply = StatusReply {
            blocked: vec![reason("something_new", 1)],
            ..healthy()
        };

        assert_eq!(
            reply.problems()[0],
            "1 recording is waiting (reason: something_new)."
        );
    }

    #[test]
    fn every_reason_code_is_explained() {
        for code in [
            resolution_detail::ROBOT_ARCHIVED,
            resolution_detail::NOT_IN_ORG,
            resolution_detail::CREDENTIALS,
            resolution_detail::NAME_TAKEN,
            resolution_detail::SHARED_NOT_ALLOWED,
            resolution_detail::REJECTED,
            resolution_detail::UPLOAD_FAILED,
            resolution_detail::NOT_ON_SERVER,
        ] {
            assert!(resolution_detail::explain(code).is_some(), "{code}");
        }
    }

    #[test]
    fn the_summary_always_shows_the_connection_and_the_counts() {
        let reply = StatusReply {
            connection: Connection::NotChecked,
            waiting: RecordingCount {
                recordings: 1,
                bytes: 512,
            },
            ..healthy()
        };

        assert_eq!(
            reply.summary(),
            vec![
                "Neuracore server: not checked yet.".to_string(),
                "Waiting to start on the server: 1 recording (512 B).".to_string(),
                "Uploading: 0 recordings (0 B left).".to_string(),
            ]
        );
    }

    #[test]
    fn bytes_are_shown_in_decimal_units() {
        assert_eq!(format_bytes(999), "999 B");
        assert_eq!(format_bytes(1_000), "1.0 KB");
        assert_eq!(format_bytes(1_250_000), "1.2 MB");
        assert_eq!(format_bytes(3_000_000_000_000), "3.0 TB");
        assert_eq!(format_bytes(u64::MAX), "18446.7 PB");
    }

    #[test]
    fn the_largest_status_reply_fits_in_one_sample() {
        // Every reason code the daemon has, in both lists, with the largest
        // counts.
        // counts, and as many named ones as a reply keeps, with the longest
        // names: `limit_reasons` merges any more.
        let codes = [
            resolution_detail::ROBOT_ARCHIVED,
            resolution_detail::NOT_IN_ORG,
            resolution_detail::CREDENTIALS,
            resolution_detail::NAME_TAKEN,
            resolution_detail::SHARED_NOT_ALLOWED,
            resolution_detail::REJECTED,
            resolution_detail::UPLOAD_FAILED,
            resolution_detail::NOT_ON_SERVER,
        ];
        let reason = |code: &str, subject: Option<Subject>| ReasonCount {
            reason: code.into(),
            recordings: u64::MAX,
            subject,
        };
        let longest = "\u{1F916}".repeat(MAX_SUBJECT_NAME_BYTES);
        let reasons: Vec<ReasonCount> = (0..MAX_NAMED_REASONS * 2)
            .map(|_| {
                reason(
                    resolution_detail::NAME_TAKEN,
                    Some(Subject::new(SubjectKind::Dataset, &longest)),
                )
            })
            .chain(codes.into_iter().map(|code| reason(code, None)))
            .collect();
        let reasons = limit_reasons(reasons);
        let mut reply = status_reply(reasons.clone(), reasons);
        reply.nonce = u64::MAX;
        reply.waiting = RecordingCount {
            recordings: u64::MAX,
            bytes: u64::MAX,
        };
        reply.uploading = reply.waiting;

        let encoded = reply.encode().unwrap();

        assert!(
            encoded.len() <= service_name::STATUS_MAX_PAYLOAD_BYTES,
            "{} bytes",
            encoded.len()
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
            timestamp_us: 1_700_000_000_000_000,
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        assert_eq!(original.kind(), "start_recording");
    }

    #[test]
    fn start_recording_bytes_are_unchanged() {
        // Captured before `StartLocalRecording` was added: a daemon from an
        // earlier install must keep decoding every online start.
        let original = Envelope::StartRecording {
            robot_id: "robot-1".into(),
            robot_instance: 3,
            robot_name: Some("arm".into()),
            dataset_id: Some("ds-1".into()),
            dataset_name: Some("warehouse".into()),
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            timestamp_us: 1_700_000_000_000_000,
        };
        assert_eq!(
            original.encode().expect("encode"),
            vec![
                0, 7, 114, 111, 98, 111, 116, 45, 49, 6, 1, 3, 97, 114, 109, 1, 4, 100, 115, 45,
                49, 1, 9, 119, 97, 114, 101, 104, 111, 117, 115, 101, 128, 128, 208, 226, 198, 191,
                206, 151, 47, 128, 128, 242, 129, 131, 137, 133, 6,
            ],
        );
    }

    #[test]
    fn start_local_recording_round_trips_through_postcard() {
        let original = Envelope::StartLocalRecording {
            robot_id: "local-robot-1".into(),
            robot_instance: 2,
            dataset_id: Some("local-dataset-1".into()),
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            timestamp_us: 1_700_000_000_000_000,
            local_robot: Some(LocalRobot {
                name: "arm".into(),
                shared: false,
                overwrite_urdf: true,
            }),
            local_dataset: Some(LocalDataset {
                name: "warehouse".into(),
                shared: true,
                description: Some("pick and place".into()),
                tags: vec!["day-1".into(), "shelf-a".into()],
            }),
        };
        let bytes = original.encode().expect("encode");
        assert_eq!(Envelope::decode(&bytes).expect("decode"), original);
        assert_eq!(original.kind(), "start_local_recording");
    }

    #[test]
    fn start_local_recording_is_appended_after_every_existing_variant() {
        // The first byte is the variant's index; `VideoProducerActive` is 8.
        let original = Envelope::StartLocalRecording {
            robot_id: "local-robot-1".into(),
            robot_instance: 0,
            dataset_id: None,
            publish_timestamp_ns: 0,
            timestamp_us: 0,
            local_robot: None,
            local_dataset: None,
        };
        assert_eq!(original.encode().expect("encode")[0], 9);
    }

    #[test]
    fn data_envelope_preserves_payload_bytes() {
        let original = Envelope::Data {
            robot_id: "robot-1".into(),
            robot_instance: 0,
            data_type: "JOINT_POSITIONS".into(),
            sensor_name: Some("waist".into()),
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            timestamp_us: 1_000_000,
            payload: vec![1, 2, 3, 4, 5, 6],
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        assert_eq!(original.kind(), "data");
    }

    #[test]
    fn data_timestamp_us_round_trips_over_postcard_wire() {
        let timestamp_us = 1_747_740_000_123_457_i64;
        let original = Envelope::Data {
            robot_id: "robot-1".into(),
            robot_instance: 0,
            data_type: "RGB_IMAGES".into(),
            sensor_name: Some("camera_right".into()),
            publish_timestamp_ns: 1_700_000_000_000_000_000,
            timestamp_us,
            payload: vec![0xAA, 0xBB],
        };
        let bytes = original.encode().expect("encode");
        let decoded = Envelope::decode(&bytes).expect("decode");
        assert_eq!(original, decoded);
        let Envelope::Data {
            timestamp_us: decoded_timestamp_us,
            ..
        } = decoded
        else {
            panic!("decoded envelope was not Data");
        };
        assert_eq!(decoded_timestamp_us, timestamp_us);
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
            timestamp_us: 0,
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
            timestamp_us: 1_700_000_000_500_000,
            items: vec![
                BatchedDataItem {
                    sensor_name: Some("joint-0".into()),
                    payload: br#"{"timestamp":1.0,"timestamp_us":1000000,"value":0.5}"#.to_vec(),
                },
                BatchedDataItem {
                    sensor_name: Some("joint-1".into()),
                    payload: br#"{"timestamp":1.0,"timestamp_us":1000000,"value":-0.25}"#.to_vec(),
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
        // per call. Each joint payload is a small
        // `{"timestamp":..,"timestamp_us":..,"value":..}` JSON object plus a
        // data_type label and sensor name; the whole batch must fit inside a
        // single `commands` sample so the producer can publish it in one go.
        let items: Vec<BatchedDataItem> = (0..1000)
            .map(|index| BatchedDataItem {
                sensor_name: Some(format!("vx300s_left_joint_{index:04}")),
                payload:
                    br#"{"timestamp":1747740000.123457,"timestamp_us":1747740000123457,"value":-1.234567890123}"#
                        .to_vec(),
            })
            .collect();
        let envelope = Envelope::BatchedData {
            robot_id: "11111111-2222-3333-4444-555555555555".into(),
            robot_instance: 0,
            data_type: "JOINT_POSITIONS".into(),
            publish_timestamp_ns: 1_747_740_000_123_456_700,
            timestamp_us: 1_747_740_000_123_457,
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
            timestamp_us: 1_700_000_000_000_000,
        };
        let bytes = stop.encode().expect("encode");
        assert_eq!(stop, Envelope::decode(&bytes).expect("decode"));
        assert_eq!(stop.kind(), "stop_recording");

        let cancel = Envelope::CancelRecording {
            robot_id: "robot-1".into(),
            robot_instance: 2,
            timestamp_us: 1_700_000_000_000_000,
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
            frame_timestamps_us: vec![
                1_700_000_000_000_000,
                1_700_000_000_016_667,
                1_700_000_000_033_333,
                1_700_000_000_050_000,
            ],
            frame_calibrations: vec![
                FrameCalibration {
                    intrinsics: Some([600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]),
                    extrinsics: Some([
                        1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0,
                        1.0,
                    ]),
                },
                FrameCalibration {
                    intrinsics: Some([600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]),
                    extrinsics: None,
                },
                FrameCalibration::default(),
                FrameCalibration::default(),
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
            ("DEPTH_IMAGES", FrameDtype::depth_u16(1e-4)),
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
                frame_timestamps_us: vec![1_700_000_000_000_000],
                frame_calibrations: vec![FrameCalibration::default()],
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
        assert_eq!(
            FrameDtype::from_wire_label("uint8", None),
            Some(FrameDtype::Rgb8)
        );
        assert_eq!(
            FrameDtype::from_wire_label("uint16", Some(1e-4)),
            Some(FrameDtype::depth_u16(1e-4))
        );
        assert_eq!(FrameDtype::from_wire_label("float32", None), None);
        assert_eq!(FrameDtype::from_wire_label("int32", None), None);
        assert_eq!(FrameDtype::from_wire_label("", None), None);
    }

    #[test]
    fn uint16_wire_label_needs_a_positive_finite_scale() {
        for scale in [
            None,
            Some(0.0),
            Some(-1e-4),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            assert_eq!(
                FrameDtype::from_wire_label("uint16", scale),
                None,
                "{scale:?}"
            );
        }
    }

    #[test]
    fn frame_dtype_bytes_per_pixel() {
        assert_eq!(FrameDtype::Rgb8.bytes_per_pixel(), 3);
        assert_eq!(FrameDtype::depth_u16(1e-4).bytes_per_pixel(), 2);
    }

    #[test]
    fn only_uint16_depth_has_a_scale() {
        assert_eq!(FrameDtype::depth_u16(1e-4).depth_scale_m(), Some(1e-4));
        assert_eq!(FrameDtype::Rgb8.depth_scale_m(), None);
    }

    #[test]
    fn video_chunk_ready_worst_case_fits_commands_slice() {
        // A capture time plus a publish offset per frame stays well under
        // COMMANDS_MAX_PAYLOAD_BYTES even at an implausible frame count.
        let frame_timestamps_us: Vec<i64> = (0..10_000).map(|i| i as i64 * 1_000).collect();
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
            frame_count: frame_timestamps_us.len() as u32,
            frame_timestamps_us,
            frame_calibrations: vec![FrameCalibration::default(); 10_000],
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
        // the absolute worst case: every per-frame capture time a full-range
        // i64 (10-byte postcard zigzag varint), every publish offset a
        // full-range u32 (5-byte varint), every frame carrying both
        // calibration matrices, and every fixed field maxed out.
        // Without the cap a long recording of tiny frames overflows the slice
        // and the whole recording's video announcement fails to publish.
        let count = service_name::MAX_VIDEO_CHUNK_FRAMES as usize;
        let frame_timestamps_us: Vec<i64> = (0..count).map(|i| i64::MAX - i as i64).collect();
        let frame_calibrations: Vec<FrameCalibration> = (0..count)
            .map(|_| FrameCalibration {
                intrinsics: Some([f32::MAX; 9]),
                extrinsics: Some([f32::MAX; 16]),
            })
            .collect();
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
            frame_timestamps_us,
            frame_calibrations,
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
