// PyO3 0.22's `#[pyfunction]` expansion includes an `.into()` on the
// `PyResult<T>` return value that fires clippy's `useless_conversion` lint
// when T resolves to `()`. The lint is correct about the generated code but
// the conversion lives in the macro expansion, not anything we wrote, so we
// silence it at the crate level rather than spraying allows over every
// `#[pyfunction]`.
#![allow(clippy::useless_conversion)]

//! PyO3 producer client for the Neuracore data daemon — a *thin shipper*.
//!
//! This crate ships as `neuracore.data_daemon._native_producer` inside the
//! Python wheel. It knows nothing about recordings: it publishes
//! source/sensor/timestamp-tagged data and three fire-and-forget lifecycle
//! events, and the daemon decides which recording (if any) each datum belongs
//! to. There is no trace registry, no per-frame sequence numbers, and no
//! recording identity on the wire.
//!
//! The surface the SDK's logging layer drives, all keyed by the **source**
//! `(robot_id, robot_instance)`:
//!
//! - [`start_recording`] / [`stop_recording`] / [`cancel_recording`] publish
//!   one lifecycle envelope each, carrying the lifecycle wall-clock
//!   `*_at_ns`.
//! - [`log_joints`] / [`log_json`] publish data envelopes tagged with the
//!   sensor `(data_type, sensor_name)` and capture `timestamp_ns`.
//! - [`log_frame`] spools raw RGB into per-`(source, sensor)` NUT chunk files
//!   under a recording-independent inbox and announces each finished chunk
//!   with [`VideoChunkReady`](data_daemon_ipc::Envelope::VideoChunkReady); the
//!   daemon buckets the chunk into a recording by its frame timestamps,
//!   relinks the NUT under that recording, and transcodes it.
//!
//! ## Threading
//!
//! iceoryx2's [`Publisher`] is neither `Send` nor `Sync`, so it is parked in a
//! [`thread_local`]: each Python thread that calls in lazily builds its own
//! iceoryx2 [`Node`] and a publisher on the commands service. The only shared
//! state is the small per-`(source, sensor)` video-chunk registry (publishing
//! threads add frames; the `stop_recording` thread flushes tail chunks), which
//! lives in a process-wide [`Mutex`].
//!
//! ## Fork safety
//!
//! A one-shot `pthread_atfork` child handler clears the forking thread's
//! `PRODUCER` slot so the next publish rebuilds. The process-wide
//! [`VIDEO_CHUNKS`] registry stores the owning PID and wipes itself on the
//! first access from a process whose PID no longer matches, so a forked
//! `multiprocessing` worker never inherits stale parent state.

pub mod nut_writer;

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, LazyLock, Mutex, Once};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use data_daemon_ipc::service_name::{
    COMMANDS, COMMANDS_MAX_PAYLOAD_BYTES, LIFECYCLE_SUBSCRIBER_BUFFER_SIZE,
    MAX_NODES_PER_SERVICE, MAX_PUBLISHERS_PER_SERVICE, MAX_QUERY_CLIENTS_PER_SERVICE,
    MAX_QUERY_SERVERS_PER_SERVICE, MAX_SUBSCRIBERS_PER_SERVICE, MAX_VIDEO_CHUNK_FRAMES, QUERIES,
    QUERIES_MAX_PAYLOAD_BYTES,
};
use data_daemon_ipc::{BatchedDataItem, Envelope, RecordingIdQuery, RecordingIdReply};
use iceoryx2::node::{Node, NodeBuilder};
use iceoryx2::port::client::Client;
use iceoryx2::port::publisher::Publisher;
use iceoryx2::prelude::{ipc, UnableToDeliverStrategy};
use iceoryx2::service::port_factory::publish_subscribe::PortFactory;
use iceoryx2::service::port_factory::request_response::PortFactory as QueryPortFactory;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

use pyo3::buffer::PyBuffer;

use crate::nut_writer::{NutVideoConfig, NutWriter};

/// Bytes after which the producer rotates to a fresh NUT chunk file.
///
/// Each chunk pays a fixed per-encode cost on the daemon side (~100-200 ms of
/// ffmpeg fork+exec + libx264 init for two output codecs). 256 MiB keeps that
/// fixed cost a small fraction of the per-chunk wall time. The threshold is
/// checked *after* each frame, so the on-disk file can exceed it by at most
/// one frame. A chunk is also rolled at every lifecycle event so a single NUT
/// only ever holds frames from one recording window.
///
/// This byte threshold has a companion frame-count cap
/// ([`MAX_VIDEO_CHUNK_FRAMES`]): a chunk is sealed at whichever bound is hit
/// first (see [`should_flush_chunk`]). Small frames never reach 256 MiB
/// mid-recording, so the frame cap is what bounds the chunk's announcement
/// envelope to one commands slice.
const CHUNK_FLUSH_BYTES: u64 = 256 * 1024 * 1024;

/// Spool directory name — must match `storage::paths::SPOOL_DIRNAME` on the
/// daemon side.
const SPOOL_DIRNAME: &str = ".rgb_spool";

/// Errors raised while publishing envelopes to the daemon.
#[derive(Debug, Error)]
enum ProducerError {
    /// Failed to build the iceoryx2 node.
    #[error("failed to create iceoryx2 node: {0}")]
    NodeCreate(String),
    /// Failed to open or create an iceoryx2 service.
    #[error("failed to open service: {0}")]
    ServiceOpen(String),
    /// Failed to build the publisher port.
    #[error("failed to create publisher: {0}")]
    PublisherCreate(String),
    /// Failed to loan a slice sample.
    #[error("failed to loan sample: {0}")]
    Loan(String),
    /// Failed to send the loaned sample.
    #[error("failed to send sample: {0}")]
    Send(String),
    /// Failed to encode the envelope.
    #[error(transparent)]
    Encode(#[from] data_daemon_ipc::EnvelopeCodecError),
    /// Payload too large for the configured iceoryx2 max slice length.
    #[error("envelope payload {actual} bytes exceeds limit {limit} bytes")]
    PayloadTooLarge {
        /// Actual encoded envelope size.
        actual: usize,
        /// Maximum slice length the publisher was built with.
        limit: usize,
    },
}

impl From<ProducerError> for PyErr {
    fn from(error: ProducerError) -> Self {
        PyRuntimeError::new_err(error.to_string())
    }
}

/// Per-thread iceoryx2 state.
struct ProducerState {
    _node: Node<ipc::Service>,
    _commands_service: PortFactory<ipc::Service, [u8], ()>,
    commands_publisher: Publisher<ipc::Service, [u8], ()>,
    /// Service handle held alongside the query client so port discovery doesn't
    /// race the handle going out of scope.
    _queries_service: QueryPortFactory<ipc::Service, [u8], (), [u8], ()>,
    /// Request-response client used by [`get_recording_id`] to ask the daemon
    /// for a recording's cloud id.
    queries_client: Client<ipc::Service, [u8], (), [u8], ()>,
}

/// In-progress video chunk state for one `(source, sensor)` stream.
///
/// The producer does not know which recording the frames belong to, so chunks
/// are spooled into a recording-independent inbox keyed by source + sensor.
/// The daemon relinks them under a recording once routing resolves a window.
struct VideoChunkState {
    /// Frame width in pixels (constant across a stream's chunks).
    width: u32,
    /// Frame height in pixels (constant across a stream's chunks).
    height: u32,
    /// `{recordings_root}/.rgb_spool/{robot_id}/{instance}/{data_type}/{sensor_name}/`.
    spool_dir: PathBuf,
    /// Active NUT writer for the in-progress chunk. `None` between chunks.
    nut_writer: Option<NutWriter>,
    /// `publish_timestamp_ns` of the in-progress chunk — captured with
    /// `chunk_thread_id` when the chunk opened (its first frame). Keys both the
    /// spool filename `chunk_{publish_ns}_{thread_id}.nut` and the window
    /// routing on the announcement, so the daemon can reconstruct the spool
    /// path. Re-stamped on every chunk open, so each chunk is named uniquely
    /// and no two recordings collide on a filename. `0` between chunks.
    chunk_publish_ns: i64,
    /// OS thread id (`gettid`) of the thread that opened the in-progress chunk.
    chunk_thread_id: i64,
    /// Frames already written into the in-progress chunk.
    frame_count: u32,
    /// Per-stream PTS origin, microseconds since the Unix epoch.
    pts_origin_us: Option<i64>,
    /// Last PTS written to any chunk for the stream; enforces monotonicity.
    last_pts_us: Option<u64>,
    /// Per-frame capture time in ns for the in-progress chunk — drained into
    /// the announcement so the daemon can bucket frames into a window.
    frame_timestamps_ns: Vec<i64>,
    /// Per-frame `timestamp_s` accumulator for the in-progress chunk.
    frame_timestamps_s: Vec<f64>,
}

/// Process-wide registry of in-progress per-`(source, sensor)` video chunk
/// state. Per-stream state lives behind its own [`Arc<Mutex<VideoChunkState>>`]
/// so a multi-megabyte NUT write for camera A does not block camera B.
type VideoChunkSlot = Arc<Mutex<VideoChunkState>>;

struct VideoChunkRegistry {
    owner_pid: u32,
    streams: HashMap<String, VideoChunkSlot>,
}

static VIDEO_CHUNKS: LazyLock<Mutex<VideoChunkRegistry>> = LazyLock::new(|| {
    Mutex::new(VideoChunkRegistry {
        owner_pid: 0,
        streams: HashMap::new(),
    })
});

// ---------------------------------------------------------------------------
// Background video writer
// ---------------------------------------------------------------------------
//
// `log_frame` must never block the caller on disk I/O. The producer spool lives
// on the same filesystem as the daemon's SQLite WAL and the ffmpeg transcode
// outputs, so on ext4 (`data=ordered`) a frame `write()` can stall for hundreds
// of ms behind an unrelated `fsync`/journal commit — and because `log_frame`
// holds the GIL across that write, the stall freezes the whole producer. To
// keep the robot-facing ingest path latency-bounded, `log_frame` copies the
// frame and hands it to a dedicated per-process writer thread, returning at
// once. The writer owns *every* NUT write, chunk seal, and
// `VideoChunkReady`/`StopRecording`/`CancelRecording` publish for video, so a
// disk stall blocks only the writer — never a `log_*` caller.
//
// Publishing the tail chunks *and* the `StopRecording`/`CancelRecording` from
// the one writer thread (its own iceoryx2 publisher).
//
// Lifecycle envelopes (`StartRecording` / `StopRecording` / `CancelRecording`)
// are emphatically NOT routed through the writer: they stay on the *calling*
// thread's publisher, the same port `StartRecording` uses, so consecutive
// recordings' start/stop boundaries keep their strict in-order delivery. (Were
// `StopRecording` published from the writer's port instead, it could be
// reordered against the next recording's `StartRecording` on the main port —
// the daemon then sees a start while the prior window is still live and drops
// the overlapping window's data.) The stop/cancel paths only *barrier* on the
// writer — seal + announce (or drop) the source's tail chunks, ack — and then
// publish the lifecycle envelope themselves. Chunk-before-stop ordering is not
// a same-port guarantee here but is safe anyway: the daemon holds every
// `VideoChunkReady` back (`NCD_HOLDBACK_MS`, default 500 ms) and retains a
// just-closed window, so a tail chunk announced just before the stop still
// routes into the (closing) window by its in-window open timestamp.

/// Backpressure cap for the writer's frame queue. A transient disk stall is
/// absorbed by buffering frames up to this many bytes before `log_frame`
/// blocks; only a *sustained* overload (the writer genuinely can't keep up)
/// propagates backpressure to the caller. 64 MiB holds ~a second of a
/// multi-camera 256×256@30 workload while staying small next to a worker's RSS.
const WRITER_QUEUE_MAX_BYTES: usize = 64 * 1024 * 1024;

/// One frame handed to the background writer. Owns its pixel bytes (copied out
/// of the caller's buffer under the GIL) so the caller can return immediately.
struct FrameJob {
    robot_id: String,
    robot_instance: i64,
    data_type: String,
    sensor_name: String,
    width: u32,
    height: u32,
    timestamp_ns: i64,
    timestamp_s: f64,
    data: Vec<u8>,
}

/// Work item for the writer thread.
enum WriterMsg {
    /// Append one frame to its `(source, sensor)` in-progress chunk.
    Frame(FrameJob),
    /// Stop barrier: drain every frame queued ahead for the source (FIFO), seal
    /// + announce its open chunks, then acknowledge. The caller publishes
    /// `StopRecording` itself once acked. No lifecycle envelope is published
    /// here — see the module note on lifecycle ordering.
    FlushSource {
        robot_id: String,
        robot_instance: i64,
        ack: Sender<()>,
    },
    /// Cancel barrier: drop every open chunk for the source without announcing
    /// it (the daemon's cancel + recovery sweep reclaim the spooled NUTs), then
    /// acknowledge. The caller publishes `CancelRecording` itself once acked.
    DropSource {
        robot_id: String,
        robot_instance: i64,
        ack: Sender<()>,
    },
}

impl WriterMsg {
    /// Bytes this message contributes to the queue's backpressure budget. Only
    /// frame payloads are throttled; control messages must always enqueue so a
    /// stop/cancel can drain even a full queue.
    fn queue_bytes(&self) -> usize {
        match self {
            WriterMsg::Frame(job) => job.data.len(),
            _ => 0,
        }
    }
}

/// Byte-bounded MPSC queue between the logging threads and the writer thread.
struct FrameQueue {
    inner: Mutex<FrameQueueInner>,
    not_full: Condvar,
    not_empty: Condvar,
}

struct FrameQueueInner {
    msgs: VecDeque<WriterMsg>,
    bytes: usize,
}

impl FrameQueue {
    fn new() -> Self {
        FrameQueue {
            inner: Mutex::new(FrameQueueInner {
                msgs: VecDeque::new(),
                bytes: 0,
            }),
            not_full: Condvar::new(),
            not_empty: Condvar::new(),
        }
    }

    /// Enqueue a message, blocking the caller only while a *frame* would push
    /// the buffered bytes over the cap (control messages never block). A lone
    /// frame larger than the cap is still admitted once the queue is empty, so
    /// forward progress is always possible.
    fn push(&self, msg: WriterMsg) {
        let add = msg.queue_bytes();
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if add > 0 {
            while inner.bytes > 0 && inner.bytes + add > WRITER_QUEUE_MAX_BYTES {
                inner = self
                    .not_full
                    .wait(inner)
                    .unwrap_or_else(|p| p.into_inner());
            }
        }
        inner.bytes += add;
        inner.msgs.push_back(msg);
        self.not_empty.notify_one();
    }

    /// Block until a message is available, then pop it (FIFO).
    fn pop(&self) -> WriterMsg {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(msg) = inner.msgs.pop_front() {
                inner.bytes -= msg.queue_bytes();
                self.not_full.notify_one();
                return msg;
            }
            inner = self
                .not_empty
                .wait(inner)
                .unwrap_or_else(|p| p.into_inner());
        }
    }
}

/// Process-wide writer handle, healed across `fork` via `owner_pid` (mirrors
/// [`VIDEO_CHUNKS`]). The parent's writer thread does not survive into a forked
/// child, so the child re-spawns one on first use.
struct WriterRegistry {
    owner_pid: u32,
    queue: Option<Arc<FrameQueue>>,
}

static VIDEO_WRITER: LazyLock<Mutex<WriterRegistry>> = LazyLock::new(|| {
    Mutex::new(WriterRegistry {
        owner_pid: 0,
        queue: None,
    })
});

/// Return this process's writer queue, spawning the writer thread on first use
/// and re-spawning after a fork. On the (near-impossible) spawn failure we log
/// and return a detached queue *without* recording it, so the next call retries
/// rather than the caller blocking forever on a consumer-less queue.
fn writer_queue() -> Arc<FrameQueue> {
    let mut reg = VIDEO_WRITER.lock().unwrap_or_else(|p| p.into_inner());
    let pid = std::process::id();
    if reg.owner_pid == pid {
        if let Some(queue) = reg.queue.as_ref() {
            return queue.clone();
        }
    }

    let queue = Arc::new(FrameQueue::new());
    let worker_queue = queue.clone();
    match std::thread::Builder::new()
        .name("nc-video-writer".to_string())
        .spawn(move || writer_loop(&worker_queue))
    {
        Ok(_handle) => {
            reg.owner_pid = pid;
            reg.queue = Some(queue.clone());
        }
        Err(error) => {
            // Leave the registry unset so the next call retries the spawn. The
            // returned queue has no consumer; a single `push` of a frame far
            // under the cap won't block, so the frame is simply dropped when the
            // queue is freed rather than the producer hanging.
            tracing::error!(%error, "failed to spawn video writer thread; dropping frame");
        }
    }
    queue
}

/// The writer thread's run loop. Sole accessor of the in-progress chunk state
/// and sole publisher of video chunk + stop/cancel envelopes for this process.
fn writer_loop(queue: &FrameQueue) {
    loop {
        match queue.pop() {
            WriterMsg::Frame(job) => {
                if let Err(error) = record_video_frame(
                    &job.robot_id,
                    job.robot_instance,
                    &job.data_type,
                    &job.sensor_name,
                    job.width,
                    job.height,
                    &job.data,
                    job.timestamp_ns,
                    job.timestamp_s,
                ) {
                    tracing::warn!(%error, sensor_name = job.sensor_name, "failed to spool video frame");
                }
            }
            WriterMsg::FlushSource {
                robot_id,
                robot_instance,
                ack,
            } => {
                if let Err(error) = flush_source_chunks(&robot_id, robot_instance) {
                    tracing::warn!(%error, "failed to flush tail video chunks on stop");
                }
                let _ = ack.send(());
            }
            WriterMsg::DropSource {
                robot_id,
                robot_instance,
                ack,
            } => {
                let prefix = source_prefix(&robot_id, robot_instance);
                with_video_chunks(|streams| {
                    streams.retain(|key, _| !key.starts_with(&prefix));
                });
                let _ = ack.send(());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Background data publisher
// ---------------------------------------------------------------------------
//
// Synchronous IPC publishes (`BatchedData` joints, `Data` json, and the
// `VideoChunkReady` chunk announcements) can briefly block on a full commands
// buffer when the daemon's listener is preempted off-CPU under heavy
// (multi-context) load. Routing them through a dedicated per-process publisher
// thread keeps that block off BOTH the caller's `log_*` thread AND the disk
// writer thread — crucially, the writer's stop/cancel *barrier* then waits only
// for the durable on-disk seal, never for an IPC publish.
//
// All three are held-back *data* on the daemon side (routed by
// `publish_timestamp_ns` within the holdback + closing-window retention), so —
// unlike lifecycle envelopes (`Start/Stop/CancelRecording`, which stay on the
// caller's publisher for strict ordering) — reordering them onto this thread's
// publisher is safe. The queue is unbounded: messages are small/infrequent and
// the thread keeps up in steady state; a transient daemon stall buffers only a
// few hundred small items.

/// Work item for the publisher thread.
enum PublishMsg {
    /// A batch of joint `(name, value)` samples to serialise + publish as one
    /// `BatchedData`. Serialisation happens on the publisher thread, off the
    /// caller.
    Joint {
        robot_id: String,
        robot_instance: i64,
        data_type: String,
        items: Vec<(String, f64)>,
        timestamp_ns: i64,
        timestamp_s: Option<f64>,
        publish_timestamp_ns: i64,
    },
    /// One JSON sample to publish as a `Data` envelope.
    Json {
        robot_id: String,
        robot_instance: i64,
        data_type: String,
        sensor_name: String,
        payload: Vec<u8>,
        timestamp_ns: i64,
        timestamp_s: Option<f64>,
        publish_timestamp_ns: i64,
    },
    /// A pre-built `VideoChunkReady` envelope to announce (built by the writer
    /// thread once the chunk is sealed on disk).
    Announce(Envelope),
}

/// Process-wide publisher handle, healed across `fork` via `owner_pid` (mirrors
/// [`VIDEO_WRITER`]).
struct PublisherRegistry {
    owner_pid: u32,
    tx: Option<Sender<PublishMsg>>,
}

static PUBLISHER: LazyLock<Mutex<PublisherRegistry>> = LazyLock::new(|| {
    Mutex::new(PublisherRegistry {
        owner_pid: 0,
        tx: None,
    })
});

/// Return this process's publisher channel, spawning the publisher thread on
/// first use and re-spawning after a fork.
fn publisher_tx() -> Sender<PublishMsg> {
    let mut reg = PUBLISHER.lock().unwrap_or_else(|p| p.into_inner());
    let pid = std::process::id();
    if reg.owner_pid == pid {
        if let Some(tx) = reg.tx.as_ref() {
            return tx.clone();
        }
    }
    let (tx, rx) = std::sync::mpsc::channel();
    match std::thread::Builder::new()
        .name("nc-data-publisher".to_string())
        .spawn(move || publish_loop(rx))
    {
        Ok(_handle) => {
            reg.owner_pid = pid;
            reg.tx = Some(tx.clone());
        }
        Err(error) => {
            tracing::error!(%error, "failed to spawn data publisher thread; dropping sample");
        }
    }
    tx
}

/// The publisher thread's run loop: publish every queued data envelope. Exits
/// when the last [`Sender`] is dropped (the channel closes).
fn publish_loop(rx: Receiver<PublishMsg>) {
    while let Ok(msg) = rx.recv() {
        let result = match msg {
            PublishMsg::Joint {
                robot_id,
                robot_instance,
                data_type,
                items,
                timestamp_ns,
                timestamp_s,
                publish_timestamp_ns,
            } => {
                let timestamp_for_json =
                    timestamp_s.unwrap_or_else(|| timestamp_ns as f64 / 1_000_000_000.0);
                let mut batch_items = Vec::with_capacity(items.len());
                for (name, value) in items {
                    match serde_json::to_vec(&ScalarFrameEntry {
                        timestamp: timestamp_for_json,
                        value,
                    }) {
                        Ok(payload) => batch_items.push(BatchedDataItem {
                            data_type: data_type.clone(),
                            sensor_name: Some(name),
                            payload,
                        }),
                        Err(error) => {
                            tracing::warn!(%error, "failed to encode joint frame JSON; dropping item")
                        }
                    }
                }
                publish(&Envelope::BatchedData {
                    robot_id,
                    robot_instance,
                    publish_timestamp_ns,
                    timestamp_ns,
                    timestamp_s,
                    items: batch_items,
                })
            }
            PublishMsg::Json {
                robot_id,
                robot_instance,
                data_type,
                sensor_name,
                payload,
                timestamp_ns,
                timestamp_s,
                publish_timestamp_ns,
            } => publish(&Envelope::Data {
                robot_id,
                robot_instance,
                data_type,
                sensor_name: Some(sensor_name),
                publish_timestamp_ns,
                timestamp_ns,
                timestamp_s,
                payload,
            }),
            PublishMsg::Announce(envelope) => publish(&envelope),
        };
        if let Err(error) = result {
            tracing::warn!(%error, "failed to publish data envelope");
        }
    }
}

/// Recordings root, resolved once per process. Mirrors the daemon's
/// `recordings_root_path()` (`config/env.rs`).
static RECORDINGS_ROOT: LazyLock<PathBuf> = LazyLock::new(resolve_recordings_root);

fn resolve_recordings_root() -> PathBuf {
    if let Ok(value) = std::env::var("NEURACORE_DAEMON_RECORDINGS_ROOT") {
        if !value.is_empty() {
            return PathBuf::from(value);
        }
    }
    default_recordings_root()
}

fn default_recordings_root() -> PathBuf {
    let db_path = match std::env::var("NEURACORE_DAEMON_DB_PATH") {
        Ok(value) if !value.is_empty() => expand_user(&value),
        _ => home_dir()
            .join(".neuracore")
            .join("data_daemon")
            .join("state.db"),
    };
    db_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("recordings")
}

fn home_dir() -> PathBuf {
    dirs::home_dir().expect("could not determine the user's home directory")
}

fn expand_user(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        return home_dir().join(stripped);
    }
    if path == "~" {
        return home_dir();
    }
    PathBuf::from(path)
}

thread_local! {
    /// One iceoryx2 publisher set per OS thread. Const-initialised so the slot
    /// is a plain TLS load — required for the `pthread_atfork` child handler to
    /// access it without invoking a lazy initializer in a post-fork context.
    static PRODUCER: RefCell<Option<ProducerState>> = const { RefCell::new(None) };
}

/// Lock the video chunk registry and run `operation` against its streams map.
///
/// Heals on fork: when the stored `owner_pid` no longer matches the current
/// process the map was inherited from a pre-fork parent, so it is cleared
/// before use.
fn with_video_chunks<R>(operation: impl FnOnce(&mut HashMap<String, VideoChunkSlot>) -> R) -> R {
    let mut registry = VIDEO_CHUNKS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let pid = std::process::id();
    if registry.owner_pid != pid {
        registry.streams.clear();
        registry.owner_pid = pid;
    }
    operation(&mut registry.streams)
}

/// Composite registry key for one `(source, sensor)` video stream. The NUL
/// separators cannot occur in any component, so the join is unambiguous.
fn stream_key(robot_id: &str, robot_instance: i64, data_type: &str, sensor_name: &str) -> String {
    format!("{robot_id}\u{0}{robot_instance}\u{0}{data_type}\u{0}{sensor_name}")
}

/// Prefix matching every video stream belonging to a source.
fn source_prefix(robot_id: &str, robot_instance: i64) -> String {
    format!("{robot_id}\u{0}{robot_instance}\u{0}")
}

/// Build the spool directory for a `(source, sensor)` stream. Mirrors
/// `storage::paths::spool_dir` on the daemon side; the two must agree
/// byte-for-byte so the daemon finds exactly what the producer wrote.
fn spool_dir(robot_id: &str, robot_instance: i64, data_type: &str, sensor_name: &str) -> PathBuf {
    RECORDINGS_ROOT
        .join(SPOOL_DIRNAME)
        .join(robot_id)
        .join(robot_instance.to_string())
        .join(data_type)
        .join(sensor_name)
}

/// Spool chunk filename — must match `storage::paths::spool_chunk_filename`.
fn spool_chunk_filename(publish_ns: i64, thread_id: i64) -> String {
    format!("chunk_{publish_ns}_{thread_id}.nut")
}

/// Run `f` against this thread's producer state, lazily building it on first
/// use.
fn with_producer<R>(
    operation: impl FnOnce(&ProducerState) -> Result<R, ProducerError>,
) -> Result<R, ProducerError> {
    PRODUCER.with(|cell| {
        if cell.borrow().is_none() {
            let state = build_producer_state()?;
            *cell.borrow_mut() = Some(state);
        }
        let slot = cell.borrow();
        let state = slot
            .as_ref()
            .expect("producer state populated immediately above");
        operation(state)
    })
}

fn build_producer_state() -> Result<ProducerState, ProducerError> {
    ensure_fork_handler_registered();

    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .map_err(|error| ProducerError::NodeCreate(error.to_string()))?;

    let (commands_service, commands_publisher) = open_publisher(
        &node,
        COMMANDS,
        LIFECYCLE_SUBSCRIBER_BUFFER_SIZE,
        COMMANDS_MAX_PAYLOAD_BYTES,
    )?;

    let (queries_service, queries_client) = open_query_client(&node, QUERIES)?;

    Ok(ProducerState {
        _node: node,
        _commands_service: commands_service,
        commands_publisher,
        _queries_service: queries_service,
        queries_client,
    })
}

/// Open (or attach to) the `[u8]` request-response `queries` service off `node`
/// and build a client on it. Config mirrors the daemon's `open_query_server`
/// so `open_or_create` reconciles to the same service attributes regardless of
/// which side comes up first.
#[allow(clippy::type_complexity)]
fn open_query_client(
    node: &Node<ipc::Service>,
    service_name: &str,
) -> Result<
    (
        QueryPortFactory<ipc::Service, [u8], (), [u8], ()>,
        Client<ipc::Service, [u8], (), [u8], ()>,
    ),
    ProducerError,
> {
    let parsed_name = service_name
        .try_into()
        .map_err(|error| ProducerError::ServiceOpen(format!("invalid service name: {error}")))?;
    let service = node
        .service_builder(&parsed_name)
        .request_response::<[u8], [u8]>()
        .max_clients(MAX_QUERY_CLIENTS_PER_SERVICE)
        .max_servers(MAX_QUERY_SERVERS_PER_SERVICE)
        .max_nodes(MAX_NODES_PER_SERVICE)
        .open_or_create()
        .map_err(|error| ProducerError::ServiceOpen(error.to_string()))?;
    let client = service
        .client_builder()
        .initial_max_slice_len(QUERIES_MAX_PAYLOAD_BYTES)
        .create()
        .map_err(|error| ProducerError::PublisherCreate(error.to_string()))?;
    Ok((service, client))
}

/// Open (or attach to) one `[u8]` pub/sub service off `node` and build a
/// publisher on it.
#[allow(clippy::type_complexity)]
fn open_publisher(
    node: &Node<ipc::Service>,
    service_name: &str,
    subscriber_buffer_size: usize,
    max_slice_len: usize,
) -> Result<
    (
        PortFactory<ipc::Service, [u8], ()>,
        Publisher<ipc::Service, [u8], ()>,
    ),
    ProducerError,
> {
    let parsed_name = service_name
        .try_into()
        .map_err(|error| ProducerError::ServiceOpen(format!("invalid service name: {error}")))?;
    let service = node
        .service_builder(&parsed_name)
        .publish_subscribe::<[u8]>()
        // Disable iceoryx2's default safe-overflow so a full subscriber buffer
        // makes `Block` take effect rather than silently evicting the oldest
        // sample. Must match the daemon's `open_subscriber`.
        .enable_safe_overflow(false)
        .subscriber_max_buffer_size(subscriber_buffer_size)
        .max_publishers(MAX_PUBLISHERS_PER_SERVICE)
        .max_subscribers(MAX_SUBSCRIBERS_PER_SERVICE)
        .max_nodes(MAX_NODES_PER_SERVICE)
        .open_or_create()
        .map_err(|error| ProducerError::ServiceOpen(error.to_string()))?;
    let publisher = service
        .publisher_builder()
        .initial_max_slice_len(max_slice_len)
        .unable_to_deliver_strategy(UnableToDeliverStrategy::Block)
        .create()
        .map_err(|error| ProducerError::PublisherCreate(error.to_string()))?;
    Ok((service, publisher))
}

/// Install the `pthread_atfork` child handler exactly once per process.
fn ensure_fork_handler_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // SAFETY: `pthread_atfork` is the standard libc primitive for
        // registering fork callbacks. `on_fork_in_child` is `extern "C"`,
        // touches only a const-initialised TLS slot, and the only "work" it
        // does is `mem::forget`.
        let result = unsafe { libc::pthread_atfork(None, None, Some(on_fork_in_child)) };
        if result != 0 {
            tracing::warn!(
                errno = result,
                "pthread_atfork registration failed; fork-safety relies on caller-managed cleanup",
            );
        }
    });
}

/// `pthread_atfork` child callback: clears the surviving thread's `PRODUCER`
/// slot so the next [`with_producer`] rebuilds fresh iceoryx2 publishers. The
/// inherited state is `mem::forget`'d on purpose (running its `Drop` would
/// touch the parent's bookkeeping). [`VIDEO_CHUNKS`] self-heals via the
/// `owner_pid` check.
extern "C" fn on_fork_in_child() {
    PRODUCER.with(|cell| {
        if let Some(stale) = cell.borrow_mut().take() {
            std::mem::forget(stale);
        }
    });
}

/// Producer wall-clock time in nanoseconds since the Unix epoch, stamped onto
/// every published data envelope as its `publish_timestamp_ns`. This is the
/// daemon's sole window-membership key, decoupled from whatever clock the
/// caller timestamps data with. The lifecycle `StartRecording` / `StopRecording`
/// envelopes carry the same publish clock as their `publish_timestamp_ns`, so
/// window boundaries and data are directly comparable.
fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as i64)
        .unwrap_or(0)
}

/// A chunk-open timestamp that is strictly increasing within this process.
///
/// The spool filename is `chunk_{publish_ns}_{thread_id}.nut`. All video chunks
/// are now opened by the single background writer thread, so they share one
/// `thread_id` and uniqueness rests entirely on `publish_ns`. `now_ns()` reads
/// `CLOCK_REALTIME`, whose granularity can repeat across two opens issued back
/// to back, which would collide two cameras' chunk files. Bumping past the last
/// value returned keeps every chunk's name distinct while staying within the
/// recording window (the window spans seconds; a few ns of skew is irrelevant
/// to membership). Only the writer thread calls this, but the atomic keeps it
/// correct regardless.
fn next_chunk_open_ns() -> i64 {
    use std::sync::atomic::{AtomicI64, Ordering};
    static LAST: AtomicI64 = AtomicI64::new(0);
    let mut candidate = now_ns();
    loop {
        let last = LAST.load(Ordering::Relaxed);
        if candidate <= last {
            candidate = last + 1;
        }
        match LAST.compare_exchange_weak(last, candidate, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return candidate,
            Err(_) => candidate = now_ns(),
        }
    }
}

/// OS thread id of the calling thread (Linux `gettid`). Used to disambiguate a
/// video chunk's spool filename across producer threads and as a breadcrumb
/// when inspecting the spool directory.
fn current_thread_id() -> i64 {
    // SAFETY: `gettid` takes no arguments and cannot fail.
    unsafe { libc::gettid() as i64 }
}

/// Encode `envelope` and publish it on the commands service.
fn publish(envelope: &Envelope) -> Result<(), ProducerError> {
    let bytes = envelope.encode()?;
    if bytes.len() > COMMANDS_MAX_PAYLOAD_BYTES {
        return Err(ProducerError::PayloadTooLarge {
            actual: bytes.len(),
            limit: COMMANDS_MAX_PAYLOAD_BYTES,
        });
    }
    with_producer(|state| {
        let publisher = &state.commands_publisher;
        let sample = publisher
            .loan_slice_uninit(bytes.len())
            .map_err(|error| ProducerError::Loan(error.to_string()))?;
        let sample = sample.write_from_slice(&bytes);
        sample
            .send()
            .map_err(|error| ProducerError::Send(error.to_string()))?;
        Ok(())
    })
}

/// Per-item JSON shape written to `trace.json` for scalar joint streams.
#[derive(serde::Serialize)]
struct ScalarFrameEntry {
    timestamp: f64,
    value: f64,
}

/// Announce that a recording has started for a source. Fire-and-forget: the
/// daemon opens a window and owns all recording identity.
///
/// The producer stamps the window's lower bound on the publish clock
/// (`publish_timestamp_ns`, always wall-clock now) — that, never the caller's
/// timestamp, is what the daemon uses for window membership, so a synthetic
/// capture time can't shift the window or clip data. Separately, the recording's
/// *capture* timestamp (`timestamp_ns` when supplied, else the publish time) is
/// what the daemon stores as `start_timestamp_ns` and POSTs as the backend
/// `start_time`. The capture timestamp is returned so the caller can use it as
/// the marker that resolves the daemon-assigned cloud recording id
/// (`get_recording_id`) for this exact recording.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, robot_name = None, dataset_id = None, dataset_name = None, timestamp_ns = None))]
fn start_recording(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    robot_name: Option<String>,
    dataset_id: Option<String>,
    dataset_name: Option<String>,
    timestamp_ns: Option<i64>,
) -> PyResult<i64> {
    if robot_id.is_empty() {
        return Err(PyValueError::new_err("robot_id must not be empty"));
    }
    let robot_id = robot_id.to_string();
    py.allow_threads(|| -> PyResult<i64> {
        let publish_timestamp_ns = now_ns();
        // Caller-supplied capture time, mirroring the `log_*` timestamp default
        // (publish clock when omitted). Decoupled from the window boundary.
        let capture_timestamp_ns = timestamp_ns.unwrap_or(publish_timestamp_ns);
        publish(&Envelope::StartRecording {
            robot_id,
            robot_instance,
            robot_name,
            dataset_id,
            dataset_name,
            publish_timestamp_ns,
            timestamp_ns: capture_timestamp_ns,
        })?;
        Ok(capture_timestamp_ns)
    })
}

/// Log one scalar sample for each of several joints captured at the same
/// instant, packed into one `BatchedData` envelope.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, data_type, items, timestamp_ns, timestamp_s = None))]
fn log_joints(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    items: Vec<(String, f64)>,
    timestamp_ns: i64,
    timestamp_s: Option<f64>,
) -> PyResult<()> {
    if robot_id.is_empty() || data_type.is_empty() {
        return Err(PyValueError::new_err(
            "robot_id and data_type must not be empty",
        ));
    }
    if items.is_empty() {
        return Ok(());
    }
    if let Some(bad_index) = items.iter().position(|(name, _)| name.is_empty()) {
        return Err(PyValueError::new_err(format!(
            "joint name must not be empty (item index {bad_index})"
        )));
    }
    let robot_id = robot_id.to_string();
    let data_type = data_type.to_string();
    py.allow_threads(move || {
        // Stamp the window-routing clock at enqueue (inside the recording
        // window). The publisher thread serialises the items and publishes the
        // `BatchedData`, keeping the synchronous IPC publish — which can briefly
        // block on a full commands buffer — off this call.
        let publish_timestamp_ns = now_ns();
        let _ = publisher_tx().send(PublishMsg::Joint {
            robot_id,
            robot_instance,
            data_type,
            items,
            timestamp_ns,
            timestamp_s,
            publish_timestamp_ns,
        });
    });
    Ok(())
}

/// Log one video frame for a camera. The frame is appended to the
/// `(source, sensor)` in-progress NUT chunk under the inbox; when the chunk
/// crosses [`CHUNK_FLUSH_BYTES`] a [`Envelope::VideoChunkReady`] is published.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, data_type, name, width, height, payload, timestamp_ns, timestamp_s = None))]
#[allow(clippy::too_many_arguments)]
fn log_frame(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    name: &str,
    width: u32,
    height: u32,
    payload: PyBuffer<u8>,
    timestamp_ns: i64,
    timestamp_s: Option<f64>,
) -> PyResult<()> {
    if robot_id.is_empty() || data_type.is_empty() || name.is_empty() {
        return Err(PyValueError::new_err(
            "robot_id, data_type and name must not be empty",
        ));
    }
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("width and height must be non-zero"));
    }
    let expected_bytes = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(3);
    let actual_bytes = payload.item_count();
    if actual_bytes != expected_bytes {
        return Err(PyValueError::new_err(format!(
            "video frame buffer is {actual_bytes} bytes; expected width*height*3 = {expected_bytes}"
        )));
    }
    if !payload.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "video frame buffer must be C-contiguous",
        ));
    }
    let resolved_timestamp_s = timestamp_s.unwrap_or_else(|| timestamp_ns as f64 / 1_000_000_000.0);

    // SAFETY: PyO3 holds the GIL here, the buffer is validated `u8` and
    // C-contiguous, the length comes from `PyBuffer::item_count`, and we only
    // read. The frame is owned by the caller's numpy array, which may be reused
    // the instant this call returns, so we *copy* it into the job under the GIL
    // (as the buffer protocol requires).
    let data = unsafe {
        std::slice::from_raw_parts(payload.buf_ptr() as *const u8, actual_bytes).to_vec()
    };

    let job = FrameJob {
        robot_id: robot_id.to_string(),
        robot_instance,
        data_type: data_type.to_string(),
        sensor_name: name.to_string(),
        width,
        height,
        timestamp_ns,
        timestamp_s: resolved_timestamp_s,
        data,
    };

    // Hand off to the writer with the GIL released: enqueuing only blocks under
    // sustained overload (the byte cap), and blocking there while holding the
    // GIL would stall every Python thread in the process.
    py.allow_threads(move || writer_queue().push(WriterMsg::Frame(job)));
    Ok(())
}

/// Whether the in-progress chunk should be sealed now, checked after each
/// appended frame. A chunk is rolled at the **lower** of two bounds:
///
/// * [`CHUNK_FLUSH_BYTES`] — keeps the daemon's per-chunk encode cost amortised.
/// * [`MAX_VIDEO_CHUNK_FRAMES`] — keeps the chunk's `VideoChunkReady`
///   announcement within one [`COMMANDS_MAX_PAYLOAD_BYTES`] sample. Small frames
///   never reach the byte threshold mid-recording, so without the frame cap a
///   long recording accumulates one ever-growing chunk whose per-frame
///   timestamp vectors eventually overflow the commands slice — the
///   announcement then fails to publish and the recording's video is lost.
fn should_flush_chunk(chunk_bytes: u64, frame_count: u32) -> bool {
    chunk_bytes >= CHUNK_FLUSH_BYTES || frame_count >= MAX_VIDEO_CHUNK_FRAMES
}

/// Append one frame to the `(source, sensor)` in-progress NUT chunk, opening
/// the chunk lazily, enforcing PTS monotonicity, and flushing once the chunk
/// crosses [`CHUNK_FLUSH_BYTES`] or [`MAX_VIDEO_CHUNK_FRAMES`]. Best-effort:
/// NUT-write errors are logged and the frame dropped, never propagated to
/// Python.
#[allow(clippy::too_many_arguments)]
fn record_video_frame(
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    sensor_name: &str,
    width: u32,
    height: u32,
    payload: &[u8],
    timestamp_ns: i64,
    timestamp_s: f64,
) -> Result<(), ProducerError> {
    let key = stream_key(robot_id, robot_instance, data_type, sensor_name);
    let slot: VideoChunkSlot = with_video_chunks(|streams| {
        streams
            .entry(key)
            .or_insert_with(|| {
                Arc::new(Mutex::new(VideoChunkState {
                    width,
                    height,
                    spool_dir: spool_dir(robot_id, robot_instance, data_type, sensor_name),
                    nut_writer: None,
                    chunk_publish_ns: 0,
                    chunk_thread_id: 0,
                    frame_count: 0,
                    pts_origin_us: None,
                    last_pts_us: None,
                    frame_timestamps_ns: Vec::new(),
                    frame_timestamps_s: Vec::new(),
                }))
            })
            .clone()
    });

    // The flush envelope is built under the per-stream lock but published
    // outside it — `publish()` blocks the calling thread when the daemon falls
    // behind, and holding the mutex across that block would stall this
    // camera's next frame.
    let flush_envelope = {
        let mut state = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        let origin_us = *state.pts_origin_us.get_or_insert(timestamp_ns / 1_000);
        let relative_us = (timestamp_ns / 1_000).saturating_sub(origin_us).max(0);
        let mut pts = relative_us as u64;
        if let Some(previous) = state.last_pts_us {
            if pts <= previous {
                pts = previous.saturating_add(1);
            }
        }

        if state.nut_writer.is_none() {
            // Stamp the chunk's identity at open: its `publish_timestamp_ns`
            // (this instant — inside the active recording window) plus the
            // opening thread's id. These name the spool file and ride the
            // announcement so the daemon can both route and locate the chunk.
            state.chunk_publish_ns = next_chunk_open_ns();
            state.chunk_thread_id = current_thread_id();
            let chunk_path = state.spool_dir.join(spool_chunk_filename(
                state.chunk_publish_ns,
                state.chunk_thread_id,
            ));
            let config = NutVideoConfig {
                width: state.width,
                height: state.height,
                time_base_num: 1,
                time_base_den: 1_000_000,
            };
            match NutWriter::create(&chunk_path, config) {
                Ok(writer) => state.nut_writer = Some(writer),
                Err(error) => {
                    tracing::warn!(
                        %error,
                        sensor_name,
                        path = %chunk_path.display(),
                        "failed to open NUT chunk; dropping frame"
                    );
                    return Ok(());
                }
            }
        }

        let bytes_after_write = {
            let writer = state.nut_writer.as_mut().expect("opened immediately above");
            if let Err(error) = writer.write_frame(pts, payload) {
                tracing::warn!(
                    %error,
                    sensor_name,
                    "failed to write video frame to NUT chunk; dropping frame"
                );
                return Ok(());
            }
            writer.bytes_written()
        };
        state.last_pts_us = Some(pts);
        state.frame_count = state.frame_count.saturating_add(1);
        state.frame_timestamps_ns.push(timestamp_ns);
        state.frame_timestamps_s.push(timestamp_s);

        if should_flush_chunk(bytes_after_write, state.frame_count) {
            flush_chunk_locked(robot_id, robot_instance, data_type, sensor_name, &mut state)
        } else {
            None
        }
    };

    if let Some(envelope) = flush_envelope {
        // Hand the sealed chunk's announcement to the publisher thread rather
        // than publishing inline: this runs on the writer thread, which must
        // never block on an IPC publish (the stop/cancel barrier waits on it).
        let _ = publisher_tx().send(PublishMsg::Announce(envelope));
    }
    Ok(())
}

/// Seal the in-progress chunk and return the announcement envelope. The caller
/// must hold the per-stream lock. Returns `None` when there is no open chunk
/// writer (no frames since the last flush).
fn flush_chunk_locked(
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    sensor_name: &str,
    state: &mut VideoChunkState,
) -> Option<Envelope> {
    let writer = state.nut_writer.take()?;
    let byte_count = match writer.finish() {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(
                %error,
                sensor_name,
                "failed to finalise NUT chunk; dropping chunk"
            );
            state.frame_count = 0;
            state.frame_timestamps_ns.clear();
            state.frame_timestamps_s.clear();
            return None;
        }
    };
    // The chunk's open-time identity, stamped when its writer was created.
    let publish_timestamp_ns = state.chunk_publish_ns;
    let thread_id = state.chunk_thread_id;
    let frame_count = state.frame_count;
    let frame_timestamps_ns = std::mem::take(&mut state.frame_timestamps_ns);
    let frame_timestamps_s = std::mem::take(&mut state.frame_timestamps_s);

    state.frame_count = 0;

    Some(Envelope::VideoChunkReady {
        robot_id: robot_id.to_string(),
        robot_instance,
        data_type: data_type.to_string(),
        sensor_name: Some(sensor_name.to_string()),
        publish_timestamp_ns,
        thread_id,
        width: state.width,
        height: state.height,
        byte_count,
        frame_count,
        frame_timestamps_ns,
        frame_timestamps_s,
    })
}

/// Log one JSON sample for any non-joint, non-video data type, delivered
/// verbatim as a `Data` envelope.
///
/// `data_type` is an opaque wire label and `payload` is already-serialized
/// bytes, so this is the generic single-sample path: scalars, poses, gripper
/// amounts, language, point clouds and any future JSON type all flow through
/// here unchanged. The daemon classifies the label downstream
/// (see `content_type_for`); it imposes no allowlist.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, data_type, name, payload, timestamp_ns, timestamp_s = None))]
#[allow(clippy::too_many_arguments)]
fn log_json(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    name: &str,
    payload: &[u8],
    timestamp_ns: i64,
    timestamp_s: Option<f64>,
) -> PyResult<()> {
    if robot_id.is_empty() || data_type.is_empty() || name.is_empty() {
        return Err(PyValueError::new_err(
            "robot_id, data_type and name must not be empty",
        ));
    }
    let robot_id = robot_id.to_string();
    let data_type = data_type.to_string();
    let name = name.to_string();
    let owned_payload = payload.to_vec();
    py.allow_threads(move || {
        // Stamp the window-routing clock at enqueue; the publisher thread
        // publishes the `Data` envelope off this call (see [`PublishMsg::Json`]).
        let publish_timestamp_ns = now_ns();
        let _ = publisher_tx().send(PublishMsg::Json {
            robot_id,
            robot_instance,
            data_type,
            sensor_name: name,
            payload: owned_payload,
            timestamp_ns,
            timestamp_s,
            publish_timestamp_ns,
        });
    });
    Ok(())
}

/// Flush any tail video chunks for the source, then publish one
/// `StopRecording`. The flush happens before the stop publish so the in-order
/// delivery contract on this thread's publisher delivers the chunk first.
///
/// The producer stamps the window's upper bound on the publish clock here
/// (`publish_timestamp_ns`, always wall-clock now), so the whole publish clock
/// is owned by the producer (consistent with the data envelopes). Every video
/// chunk routes by its *open* time, which is strictly inside the recording, so
/// the exact value of this boundary no longer has to be reconciled with a tail
/// chunk. The recording's *capture* stop time (`timestamp_ns` when supplied,
/// else the publish time) is separate — it is stored as `stop_timestamp_ns` and
/// POSTed as the backend `end_time`, never used for window membership.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, timestamp_ns = None))]
fn stop_recording(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    timestamp_ns: Option<i64>,
) -> PyResult<()> {
    if robot_id.is_empty() {
        return Err(PyValueError::new_err("robot_id must not be empty"));
    }
    let robot_id = robot_id.to_string();
    py.allow_threads(|| -> PyResult<()> {
        let publish_timestamp_ns = now_ns();
        // Caller-supplied capture time, mirroring the `log_*` timestamp default
        // (publish clock when omitted). Decoupled from the window boundary.
        let capture_timestamp_ns = timestamp_ns.unwrap_or(publish_timestamp_ns);
        // Barrier on the writer: it drains every frame still queued for this
        // source (FIFO), seals the tail chunks and announces them, then acks.
        // Blocking here means the stop never returns until those chunks are
        // durably spooled + announced, so a process exit right after
        // `stop_recording` can't lose them.
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        writer_queue().push(WriterMsg::FlushSource {
            robot_id: robot_id.clone(),
            robot_instance,
            ack: ack_tx,
        });
        let _ = ack_rx.recv();
        // Publish `StopRecording` from THIS (the calling) thread's publisher —
        // the same port as `StartRecording` — so consecutive recordings' start
        // and stop boundaries stay strictly ordered for the daemon. The tail
        // chunks were announced on the writer's port just above; the daemon's
        // holdback + closing-window retention route them into this window even
        // though they ride a different port.
        publish(&Envelope::StopRecording {
            robot_id,
            robot_instance,
            publish_timestamp_ns,
            timestamp_ns: capture_timestamp_ns,
        })?;
        Ok(())
    })
}

/// Flush and remove every open video chunk for a source. Each flushed chunk is
/// announced so the daemon can route it before the `StopRecording` lands.
fn flush_source_chunks(robot_id: &str, robot_instance: i64) -> Result<(), ProducerError> {
    let prefix = source_prefix(robot_id, robot_instance);
    let slots: Vec<(String, VideoChunkSlot)> = with_video_chunks(|streams| {
        let keys: Vec<String> = streams
            .keys()
            .filter(|key| key.starts_with(&prefix))
            .cloned()
            .collect();
        keys.into_iter()
            .filter_map(|key| streams.remove(&key).map(|slot| (key, slot)))
            .collect()
    });
    for (key, slot) in slots {
        let (data_type, sensor_name) = split_stream_key(&key);
        let flush_envelope = {
            let mut state = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            flush_chunk_locked(
                robot_id,
                robot_instance,
                &data_type,
                &sensor_name,
                &mut state,
            )
        };
        if let Some(envelope) = flush_envelope {
            // Announce via the publisher thread so the stop barrier (which awaits
            // this flush) only ever waits on the on-disk seal, never an IPC send.
            let _ = publisher_tx().send(PublishMsg::Announce(envelope));
        }
    }
    Ok(())
}

/// Split a `stream_key` back into `(data_type, sensor_name)`. The leading
/// `robot_id\0instance\0` is dropped.
fn split_stream_key(key: &str) -> (String, String) {
    let mut parts = key.splitn(4, '\u{0}');
    let _robot_id = parts.next().unwrap_or("");
    let _instance = parts.next().unwrap_or("");
    let data_type = parts.next().unwrap_or("").to_string();
    let sensor_name = parts.next().unwrap_or("").to_string();
    (data_type, sensor_name)
}

/// Cancel a recording — drop the source's in-progress chunk state without
/// flushing (the daemon's cancel handler removes the relinked artefacts and
/// the recovery sweep reclaims any spooled NUTs).
///
/// A cancel is a recording stop that discards data, so it carries the same
/// capture `timestamp_ns` as `stop_recording` (the caller's value, else the
/// publish clock); the daemon stores it as `stop_timestamp_ns` and POSTs it as
/// the backend `end_time`.
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, timestamp_ns = None))]
fn cancel_recording(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    timestamp_ns: Option<i64>,
) -> PyResult<()> {
    if robot_id.is_empty() {
        return Err(PyValueError::new_err("robot_id must not be empty"));
    }
    let robot_id = robot_id.to_string();
    py.allow_threads(|| -> PyResult<()> {
        let capture_timestamp_ns = timestamp_ns.unwrap_or_else(now_ns);
        // Barrier on the writer: it drains any frames still queued for this
        // source (FIFO) and drops the in-progress chunk state without announcing
        // it, then acks. Block until acked so the cancel is ordered after those
        // frames and no late chunk for this recording is announced.
        let (ack_tx, ack_rx) = std::sync::mpsc::channel();
        writer_queue().push(WriterMsg::DropSource {
            robot_id: robot_id.clone(),
            robot_instance,
            ack: ack_tx,
        });
        let _ = ack_rx.recv();
        // Publish `CancelRecording` from THIS (the calling) thread's publisher,
        // ordered with Start/Stop on the same port (see the writer module note).
        publish(&Envelope::CancelRecording {
            robot_id,
            robot_instance,
            timestamp_ns: capture_timestamp_ns,
        })?;
        Ok(())
    })
}

/// Python module entrypoint registered as `neuracore.data_daemon._native_producer`.
/// Interval between successive recording-id requests to the daemon.
const RECORDING_ID_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// How long a single request waits for the daemon's reply before re-asking.
const RECORDING_ID_RESPONSE_WAIT: Duration = Duration::from_millis(40);
/// Poll cadence while waiting for a single request's reply.
const RECORDING_ID_RECEIVE_POLL: Duration = Duration::from_millis(2);

/// Resolve the daemon-owned cloud `recording_id` for a recording, blocking with
/// the GIL released until the id is available or `timeout_s` elapses.
///
/// The thin producer never mints recording identity — the daemon allocates the
/// cloud id asynchronously after `/recording/start`. This asks the daemon over
/// the `queries` request-response service (identifying the recording by its
/// source + capture `timestamp_ns` marker) and returns the id once minted, or
/// `None` on timeout / when no daemon is answering. Safe for
/// non-performance-critical paths only (tests, `stop_recording(wait=True)`).
#[pyfunction]
#[pyo3(signature = (robot_id, robot_instance, timestamp_ns, timeout_s))]
fn get_recording_id(
    py: Python<'_>,
    robot_id: &str,
    robot_instance: i64,
    timestamp_ns: i64,
    timeout_s: f64,
) -> PyResult<Option<String>> {
    if robot_id.is_empty() {
        return Err(PyValueError::new_err("robot_id must not be empty"));
    }
    let query = RecordingIdQuery {
        robot_id: robot_id.to_string(),
        robot_instance,
        timestamp_ns,
    };
    let request_bytes = query.encode().map_err(ProducerError::from)?;
    py.allow_threads(|| -> PyResult<Option<String>> {
        let deadline = Instant::now() + Duration::from_secs_f64(timeout_s.max(0.0));
        loop {
            let resolved =
                with_producer(|state| resolve_recording_id_once(state, &request_bytes))?;
            if resolved.is_some() {
                return Ok(resolved);
            }
            if Instant::now() >= deadline {
                return Ok(None);
            }
            std::thread::sleep(RECORDING_ID_POLL_INTERVAL);
        }
    })
}

/// Send one recording-id request and wait briefly for the daemon's reply.
///
/// Returns `Ok(Some(id))` once the daemon has minted the cloud id, or `Ok(None)`
/// when it replied "not yet" or did not reply within the per-request window
/// (e.g. no daemon is up). The caller re-asks until its overall timeout.
fn resolve_recording_id_once(
    state: &ProducerState,
    request_bytes: &[u8],
) -> Result<Option<String>, ProducerError> {
    let request = state
        .queries_client
        .loan_slice_uninit(request_bytes.len())
        .map_err(|error| ProducerError::Loan(error.to_string()))?;
    let request = request.write_from_slice(request_bytes);
    let pending = request
        .send()
        .map_err(|error| ProducerError::Send(error.to_string()))?;

    let response_deadline = Instant::now() + RECORDING_ID_RESPONSE_WAIT;
    loop {
        match pending.receive() {
            Ok(Some(response)) => {
                let reply = RecordingIdReply::decode(response.payload())?;
                return Ok(reply.recording_id);
            }
            Ok(None) => {}
            Err(error) => return Err(ProducerError::Send(error.to_string())),
        }
        if Instant::now() >= response_deadline {
            return Ok(None);
        }
        std::thread::sleep(RECORDING_ID_RECEIVE_POLL);
    }
}

#[pymodule]
fn _native_producer(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(start_recording, module)?)?;
    module.add_function(wrap_pyfunction!(log_joints, module)?)?;
    module.add_function(wrap_pyfunction!(log_frame, module)?)?;
    module.add_function(wrap_pyfunction!(log_json, module)?)?;
    module.add_function(wrap_pyfunction!(stop_recording, module)?)?;
    module.add_function(wrap_pyfunction!(cancel_recording, module)?)?;
    module.add_function(wrap_pyfunction!(get_recording_id, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flushes_when_byte_threshold_reached() {
        // The byte threshold seals the chunk regardless of frame count (large
        // frames hit 256 MiB long before the frame cap).
        assert!(should_flush_chunk(CHUNK_FLUSH_BYTES, 1));
        assert!(should_flush_chunk(CHUNK_FLUSH_BYTES + 1, 1));
    }

    #[test]
    fn flushes_when_frame_cap_reached() {
        // Small frames never reach the byte threshold, so the frame cap is what
        // bounds the chunk — it seals at MAX_VIDEO_CHUNK_FRAMES even with a
        // near-empty NUT file.
        assert!(should_flush_chunk(0, MAX_VIDEO_CHUNK_FRAMES));
        assert!(should_flush_chunk(1, MAX_VIDEO_CHUNK_FRAMES + 1));
    }

    #[test]
    fn does_not_flush_below_both_bounds() {
        assert!(!should_flush_chunk(0, 0));
        assert!(!should_flush_chunk(
            CHUNK_FLUSH_BYTES - 1,
            MAX_VIDEO_CHUNK_FRAMES - 1
        ));
    }

    #[test]
    fn flush_is_the_lower_of_the_two_bounds() {
        // A chunk of tiny frames is sealed by the frame cap with bytes still far
        // under the byte threshold — i.e. whichever bound is hit first wins.
        assert!(should_flush_chunk(1, MAX_VIDEO_CHUNK_FRAMES));
        // ...and a byte-heavy chunk is sealed before the frame cap.
        assert!(should_flush_chunk(CHUNK_FLUSH_BYTES, 1));
    }
}
