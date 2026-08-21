//! Per-trace actor task.
//!
//! Owns the SQLite lifecycle and the on-disk encoders for one trace. The
//! daemon owns trace identity: a trace is `(recording_index, data_type,
//! sensor_name)` and the dispatcher mints its `trace_id` (a UUID, the DB
//! primary key and on-disk directory name) when it first routes data for that
//! key. The actor therefore knows its full identity at spawn time — there is
//! no `StartTrace` and no pre-`StartTrace` buffering.
//!
//! Scalar / sensor traces stream into a [`crate::encoding::json_trace::JsonTraceWriter`]; video traces
//! consume [`TraceActorMessage::Video`] notifications that hand off
//! daemon-relinked NUT chunks for ffmpeg-side transcoding into MP4 segments
//! (one batch of up to [`ENCODE_BATCH_MAX_CHUNKS`] queued chunks per
//! invocation under backlog), then on finalise stitch the segments into the
//! final `lossy.mp4` / `lossless.mp4` and flush the
//! [`VideoMetadataAccumulator`] sidecar.
//!
//! `TraceWriterKind::Video` owns the `completed_chunks` map, the worker
//! `JoinSet`, and the shared encode queue and un-encoded chunk counter. The
//! `EncodeWorker` holds `Arc` clones of the queue, the counter and the ffmpeg
//! permit pool plus the encoder, paths and codec, and borrows nothing from the
//! actor state, so it runs independently of the actor task. `handle_video`
//! pushes the arriving chunk onto the queue and spawns a worker on the
//! `JoinSet`. The worker that wins an ffmpeg permit drains the queue front
//! for up to [`ENCODE_BATCH_MAX_CHUNKS`] chunks. The drain stops at a dtype
//! change and before a chunk whose declared span exceeds
//! [`ENCODE_BATCH_MAX_SPAN_US`].
//! The worker relinks its batch from the producer spool only after it wins
//! that permit, so the fixed spool cap bounds the un-encoded backlog. A worker
//! reports `NoOp` when an earlier worker took its chunk, `Completed` with one
//! `CompletedChunk`, or `Failed`. On the arrival path
//! `drain_completed_encodes` reaps finished workers with `try_join_next` and
//! marks the trace failed on `Failed`. At finalise `finalise_writer` awaits
//! every remaining worker with `join_next` and returns `Err` on `Failed`,
//! which skips the concat. `completed_chunks` is keyed by the batch's first
//! chunk index, so segment indices are not dense, and finalise concatenates
//! the segments in key order.
//!
//! Finalisation is driven by a single [`TraceActorMessage::WindowClosing`]
//! signal: the dispatcher sends every routed datum to the actor's FIFO inbox
//! *before* `WindowClosing`, so by the time the actor sees it every frame has
//! been applied — completeness without counting sequence numbers.
//!
//! Database writes never touch the store's single write mutex on the actor's
//! hot path: the row creation *and* every subsequent progress / status /
//! finalise / failed update are fired into the coalescing write-behind
//! ([`crate::state::trace_event_database_writer`]) and never awaited — the actor's first write
//! carries the create fields, which are enqueued before any update; the
//! batcher's coalescing plus the `ON CONFLICT DO NOTHING` insert keep the row
//! correct even if the create and its updates land in different flush batches.
//! Because creation is fire-and-forget too, the actor starts draining its inbox the
//! instant it spawns, even during a boundary's spawn burst. Per-frame
//! `bytes_written` updates are still debounced ([`BYTES_WRITTEN_DEBOUNCE_FRAMES`])
//! before being enqueued, and the batcher further coalesces them per trace and
//! flushes them in batched transactions.

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use data_daemon_shared::FrameDtype;
use serde_json::Value;
use tokio::sync::{mpsc, watch, Semaphore};
use tokio::task::{self, JoinSet};

use crate::cloud::ConfigRx;
use crate::config::DaemonConfig;
use crate::encoding::json_trace::JsonTraceError;
use crate::encoding::metadata::{MetadataError, VideoMetadataAccumulator};
use crate::encoding::video_encoder::{
    batch_content_extent_us, declared_batch_span_us, declared_span_with_extent_us,
    BatchEncodeRequest, BatchNutInput, LossyVideoCodec, VideoEncodeError, VideoEncoder,
    ENCODE_THREADS_PER_OUTPUT,
};
use crate::pipeline::json_writer::JsonWriteHandle;
use crate::state::TraceWriteHandle;
use crate::storage::budget::StorageBudget;
use crate::storage::paths::{self, TracePath};

/// Routing key identifying one per-trace actor.
///
/// `Data` and `VideoChunkReady` envelopes carry their source + sensor on the
/// wire; the dispatcher resolves the source's active window to a
/// `recording_index` and routes by this key. Two recordings of the same sensor
/// get distinct actors automatically because `recording_index` differs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraceKey {
    /// Parent recording's local index.
    pub recording_index: i64,
    /// Wire data-type label (e.g. `"JOINT_POSITIONS"`, `"RGB_IMAGES"`).
    pub data_type: String,
    /// Per-stream sensor label (joint name, camera id). Persisted to the trace
    /// row's `data_type_name` column.
    pub sensor_name: Option<String>,
}

/// Full identity handed to a spawned actor: its routing key plus the
/// daemon-minted `trace_id` used as the DB primary key and on-disk directory.
#[derive(Debug, Clone)]
pub struct TraceIdentity {
    /// Daemon-minted UUID — DB primary key and on-disk directory name.
    pub trace_id: String,
    /// Routing key (`recording_index`, `data_type`, `sensor_name`).
    pub key: TraceKey,
}

/// Flush `bytes_written` to the DB every N frames instead of every frame.
///
/// At 30 fps video and 200 Hz scalars this keeps the SQLite write rate well
/// under 10 Hz per trace, which the WAL handles comfortably while still giving
/// the upload coordinator a recent enough byte count for its progress reports.
/// A finalise always issues a fresh UPDATE so the terminal row is exact.
const BYTES_WRITTEN_DEBOUNCE_FRAMES: u64 = 32;

/// Cap on how many queued chunks one worker drains into a single batched
/// ffmpeg invocation. Bounds the latency of any one batch and keeps the
/// shared permit pool fair across traces under backlog.
pub(crate) const ENCODE_BATCH_MAX_CHUNKS: usize = 8;

/// Cap on the declared span between two chunks inside one batch. ffmpeg
/// silently corrupts a preset-medium batched encode once a `duration` line
/// passes a version-dependent cliff (bisected at 120 s on 5.1; CI runs 4.4).
/// A gap this large under backlog is a recording stall, so ending the batch
/// instead costs nothing.
pub(crate) const ENCODE_BATCH_MAX_SPAN_US: i64 = 30_000_000;

/// Cap on concurrent ffmpeg transcodes.
///
/// Each batch encode bounds its libx264 thread pool to
/// [`ENCODE_THREADS_PER_OUTPUT`] per output stream, so to keep the encode fleet
/// near — not far past — the host's core count we run roughly
/// `cores / threads_per_output` invocations at once. Letting the permit count
/// *and* each child's thread pool both scale with the core count (as an earlier
/// revision did, with libx264 defaulting to one frame-thread per core)
/// oversubscribed a 14-core host to ~200 encode threads, which thrashed the
/// scheduler and stole cycles from the latency-critical `nc.log_*` threads.
/// Dividing here holds the total encode-thread count near the core count while
/// still letting bigger hosts transcode multi-camera 8-context workloads in
/// parallel.
///
/// Floor at 2 so single-core hosts still get a useful permit pool.
pub(crate) fn default_ffmpeg_concurrency() -> usize {
    (encode_host_cores() / ENCODE_THREADS_PER_OUTPUT).max(2)
}

/// Host parallelism (logical cores), floored at 1. Cached-cheap syscall; read
/// once when the actor context is built.
pub(crate) fn encode_host_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// libx264 threads for each output of an encode, given the host core count and
/// how many transcodes are running right now. The permit pool caps concurrency
/// at `cores / ENCODE_THREADS_PER_OUTPUT`; when fewer than that run (a few-camera
/// load on a many-core host) the rest of the box is idle. `cores / active` keeps
/// total encoder threads ≈ `2 × cores` at any concurrency — it collapses to
/// [`ENCODE_THREADS_PER_OUTPUT`] at full load (the original fixed cap) and fills
/// the idle cores otherwise.
pub(crate) fn adaptive_encode_threads(cores: usize, active_encodes: usize) -> usize {
    let cores = cores.max(1);
    let active = active_encodes.max(1);
    let ceiling = cores.max(ENCODE_THREADS_PER_OUTPUT);
    (cores / active).clamp(ENCODE_THREADS_PER_OUTPUT, ceiling)
}

/// Shared context passed to every per-trace actor.
///
/// Cheap to clone (everything inside is an `Arc` or `Copy`-like config), so
/// the dispatcher hands each actor its own handle without contention. The
/// storage budget is shared across actors so reservations accumulate.
#[derive(Clone)]
pub struct TraceActorContext {
    /// Filesystem root under which trace artefacts are written.
    pub recordings_root: Arc<std::path::PathBuf>,
    /// Shared storage-budget tracker. Reserved here so the budget can refuse
    /// frames when the configured quota is exhausted.
    pub storage_budget: Arc<StorageBudget>,
    /// Encoder used to transcode batches of NUT files into MP4 segments and
    /// to stream-copy concatenate the segments into the final outputs on
    /// finalise. Cloning a [`VideoEncoder`] is cheap (it carries only the
    /// configured ffmpeg binary path).
    pub video_encoder: VideoEncoder,
    /// Bounds concurrent ffmpeg children. Shared across actors so the
    /// integration matrix's parallel encode storms don't fork-bomb the
    /// transcoder.
    pub ffmpeg_permits: Arc<Semaphore>,
    /// Total permits in [`ffmpeg_permits`](Self::ffmpeg_permits), captured at
    /// construction (before any are held). Subtracting the live
    /// `available_permits()` yields how many transcodes are running right now,
    /// which sizes each encode's thread pool (see [`adaptive_encode_threads`]).
    pub ffmpeg_permit_total: usize,
    /// Host parallelism, paired with the live permit count to scale each
    /// encode's libx264 thread pool to the idle cores.
    pub encode_cores: usize,
    /// Optional daemon event bus. When present, the trace actor publishes a
    /// [`crate::state::DaemonEvent::TraceWritten`] on finalise so the
    /// registration coordinator can wake immediately. Optional so unit tests
    /// can exercise the actor without standing up a bus.
    pub event_bus: Option<crate::state::EventBus>,
    /// Write-behind handle for this actor's create / progress / status /
    /// finalise updates. Routing these through the coalescing batcher keeps the
    /// actor's hot path — including row creation — off the store's single write
    /// mutex entirely (see [`crate::state::trace_event_database_writer`]).
    pub trace_writer: TraceWriteHandle,
    /// Write-behind handle for this actor's `trace.json` appends. Keeps the
    /// blocking JSON `write()` — which periodically stalls behind an ext4
    /// journal commit on the shared spool — off the actor's hot path, so a disk
    /// stall can't back-pressure the dispatcher / IPC listener (see
    /// [`crate::pipeline::json_writer`]).
    pub json_writer: JsonWriteHandle,
    /// Live view of the effective daemon config, published by the config
    /// watcher. The actor reads `video_codec` from here at a trace's first
    /// chunk instead of re-parsing the profile YAML. Seeded with the default
    /// config; production overrides it via [`TraceActorContext::with_config_rx`].
    pub config_rx: ConfigRx,
}

impl TraceActorContext {
    /// Build a context with the default ffmpeg concurrency cap. Suitable for
    /// production wiring; tests that need a deterministic transcode order may
    /// prefer [`TraceActorContext::with_ffmpeg_permits`].
    pub fn new(
        recordings_root: impl Into<std::path::PathBuf>,
        storage_budget: Arc<StorageBudget>,
        video_encoder: VideoEncoder,
        trace_writer: TraceWriteHandle,
        json_writer: JsonWriteHandle,
    ) -> Self {
        Self::with_ffmpeg_permits(
            recordings_root,
            storage_budget,
            video_encoder,
            Arc::new(Semaphore::new(default_ffmpeg_concurrency())),
            trace_writer,
            json_writer,
        )
    }

    /// Build a context with an externally-provided ffmpeg permit pool.
    pub fn with_ffmpeg_permits(
        recordings_root: impl Into<std::path::PathBuf>,
        storage_budget: Arc<StorageBudget>,
        video_encoder: VideoEncoder,
        ffmpeg_permits: Arc<Semaphore>,
        trace_writer: TraceWriteHandle,
        json_writer: JsonWriteHandle,
    ) -> Self {
        // Captured before any permit is acquired, so this is the pool's total.
        let ffmpeg_permit_total = ffmpeg_permits.available_permits();
        // Seed the config view with defaults. The sender is dropped
        // immediately; a `watch::Receiver` still serves its last value via
        // `borrow()` after the sender is gone, which is all the actor needs
        // when no watcher is wired (tests / offline construction). Production
        // replaces this via `with_config_rx`.
        let (_seed_tx, config_rx) = watch::channel(DaemonConfig::default());
        Self {
            recordings_root: Arc::new(recordings_root.into()),
            storage_budget,
            video_encoder,
            ffmpeg_permits,
            ffmpeg_permit_total,
            encode_cores: encode_host_cores(),
            event_bus: None,
            trace_writer,
            json_writer,
            config_rx,
        }
    }

    /// Attach a daemon event bus to this context. Returns `self` so it
    /// composes cleanly with [`TraceActorContext::new`] /
    /// [`TraceActorContext::with_ffmpeg_permits`].
    pub fn with_event_bus(mut self, bus: crate::state::EventBus) -> Self {
        self.event_bus = Some(bus);
        self
    }

    /// Attach the live config view published by the config watcher. Returns
    /// `self` so it composes with the constructors and [`Self::with_event_bus`].
    pub fn with_config_rx(mut self, config_rx: ConfigRx) -> Self {
        self.config_rx = config_rx;
        self
    }
}

/// Message accepted by a per-trace actor.
#[derive(Debug)]
pub enum TraceActorMessage {
    /// One sensor sample routed to this trace after its holdback elapsed.
    Data {
        /// Caller-supplied capture time in nanoseconds since the Unix epoch.
        timestamp_ns: i64,
        /// Optional caller-supplied capture time in seconds.
        timestamp_s: Option<f64>,
        /// Opaque per-sample bytes.
        payload: Vec<u8>,
    },
    /// One finished NUT chunk. The actor relinks it from the producer spool
    /// into this trace's `chunks/chunk_NNNN.nut` (on a blocking thread, inside
    /// the background encode task) so the rename's possible journal-commit stall
    /// stays off the dispatcher's routing path.
    Video {
        /// Daemon-assigned, per-trace monotonic chunk index.
        chunk_index: u32,
        /// Producer-spooled source NUT to relink into this trace's chunks dir.
        spool_nut: PathBuf,
        /// Frame width in pixels (constant across a trace).
        width: u32,
        /// Frame height in pixels.
        height: u32,
        /// Size of the spooled NUT file in bytes.
        byte_count: u64,
        /// Number of frames in the chunk.
        frame_count: u32,
        /// Per-frame `timestamp_s` for the metadata sidecar, in capture order.
        frame_timestamps_s: Vec<f64>,
        /// Original dtype of every frame in this chunk. The daemon never
        /// decodes pixels — this is threaded straight into the completed
        /// chunk and, for depth, the trace's `trace.json` sidecar.
        dtype: FrameDtype,
    },
    /// The recording window has closed and its holdback has drained: finalise
    /// the trace. Every routed datum has already been delivered ahead of this
    /// message by the single-owner dispatcher.
    WindowClosing,
    /// Drop the in-flight writer and delete the on-disk artefacts. Sent by
    /// the dispatcher when the parent recording is cancelled.
    Cancel,
}

/// Internal state of a per-trace actor.
///
/// Encoders are opened lazily: a scalar trace doesn't need a `trace.json` file
/// until the first frame arrives, and a video trace's segment / metadata
/// state is allocated when the first `Video` message lands.
enum TraceWriterKind {
    /// No frames yet observed; the writer is decided on the first frame or
    /// chunk message.
    Pending,
    /// Scalar trace streaming into a single `trace.json` array. The actual
    /// [`crate::encoding::json_trace::JsonTraceWriter`] lives on the write-behind thread
    /// ([`crate::pipeline::json_writer`]); the actor only holds this marker and
    /// drives it by `trace_id` through [`TraceActorContext::json_writer`].
    Json,
    /// Video trace whose encode workers run as concurrent background tasks.
    /// A worker drains at most one batch from the shared queue, and encodes
    /// nothing if an earlier worker took its chunk.
    Video {
        /// Frame width in pixels (recorded from the first chunk message).
        width: u32,
        /// Frame height in pixels.
        height: u32,
        /// Lossy codec for this trace, resolved once at the first chunk and
        /// applied uniformly to every chunk encode and the finalise concat.
        codec: LossyVideoCodec,
        /// Encodes completed so far, keyed by the batch's first `chunk_index`
        /// so the finalise concat can iterate in order regardless of
        /// completion order.
        completed_chunks: BTreeMap<u32, CompletedChunk>,
        /// Spawned encode workers still running.
        pending_encodes: JoinSet<EncodeWorkerOutcome>,
        /// Chunks waiting for a worker to win an ffmpeg permit and drain
        /// them into a batch. Shared with the spawned workers.
        encode_queue: Arc<Mutex<VecDeque<QueuedChunk>>>,
        /// Count of un-encoded chunks: the queue length plus the chunks
        /// covered by in-flight batches. Reported as `pending_encode_count`
        /// at finalise.
        unencoded_chunks: Arc<AtomicUsize>,
    },
}

/// One un-encoded chunk waiting in a trace's encode queue for a worker to
/// drain it into a batch.
struct QueuedChunk {
    /// Daemon-assigned, per-trace monotonic chunk index.
    chunk_index: u32,
    /// Producer-spooled source NUT, relinked into the trace's chunks dir when
    /// its batch starts encoding.
    spool_nut: PathBuf,
    /// Size of the spooled NUT file in bytes.
    byte_count: u64,
    /// Number of frames in the chunk.
    frame_count: u32,
    /// Per-frame `timestamp_s` values in capture order. The first entry also
    /// anchors the batch's inter-chunk duration spans.
    frame_timestamps_s: Vec<f64>,
    /// Original dtype of every frame in this chunk. A batch spans one dtype.
    dtype: FrameDtype,
}

/// One successfully encoded batch of one or more chunks, keyed by its first
/// chunk index, ready to feed into the finalise concat.
struct CompletedChunk {
    /// `chunk_NNNN_lossy.mp4` segment path (first index of the batch).
    lossy_segment: PathBuf,
    /// `chunk_NNNN_lossless.mp4` segment path (first index of the batch).
    lossless_segment: PathBuf,
    /// Sum of both segments' on-disk byte counts.
    bytes: u64,
    /// Per-frame `timestamp_s` values, the in-order concatenation of the
    /// batch's per-chunk vectors, applied to the metadata accumulator at
    /// finalise in chunk-index order.
    frame_timestamps_s: Vec<f64>,
    /// Total frames covered by this entry: the sum over the batch's chunks.
    frame_count: u32,
    /// The segment's real mp4 content extent, as the batch concat list
    /// dictated it to ffmpeg. The finalise span floors on this so the next
    /// segment never starts inside this one's content, which a replay of the
    /// announced stamps cannot guarantee for a synthesized-PTS batch.
    content_extent_us: i64,
    /// The batch's original frame dtype — stored per batch (not once for the
    /// whole trace) so each batch's metadata entries get their own dtype even
    /// if it somehow differs between chunks of one trace.
    dtype: FrameDtype,
}

/// Outcome of one background encode worker.
enum EncodeWorkerOutcome {
    /// One batch encoded; `chunk_index` is the batch's first index.
    Completed {
        chunk_index: u32,
        completed: CompletedChunk,
    },
    /// The worker found the queue empty: an earlier worker's batch already
    /// covered its chunk. Both reaping paths reap this silently.
    NoOp,
    /// The batch failed; the reaping path logs the error and marks the trace
    /// failed. The batch's relinked NUTs stay on disk for the recovery sweep.
    Failed {
        chunk_index: u32,
        error: VideoEncodeError,
    },
}

/// Run the per-trace actor until the dispatcher closes the inbox or sends a
/// terminal message (`WindowClosing` / `Cancel`).
pub async fn run(
    context: Arc<TraceActorContext>,
    identity: TraceIdentity,
    mut inbox: mpsc::Receiver<TraceActorMessage>,
) {
    let mut state = ActorState::new(identity);
    // Fire-and-forget the row creation as the actor's first write. The batcher
    // inserts it on its next flush — so the boundary's spawn burst is one
    // batched insert, and the actor starts draining its inbox immediately
    // instead of blocking on a synchronous `create_trace`.
    state.send_create(&context);

    while let Some(message) = inbox.recv().await {
        match message {
            TraceActorMessage::Data {
                timestamp_ns,
                timestamp_s,
                payload,
            } => {
                state
                    .handle_data(&context, timestamp_ns, timestamp_s, payload)
                    .await;
            }
            TraceActorMessage::Video {
                chunk_index,
                spool_nut,
                width,
                height,
                byte_count,
                frame_count,
                frame_timestamps_s,
                dtype,
            } => {
                state
                    .handle_video(
                        &context,
                        chunk_index,
                        spool_nut,
                        width,
                        height,
                        byte_count,
                        frame_count,
                        frame_timestamps_s,
                        dtype,
                    )
                    .await;
            }
            TraceActorMessage::WindowClosing => {
                state.finalise_trace(&context).await;
                return;
            }
            TraceActorMessage::Cancel => {
                tracing::info!(
                    trace_id = state.identity.trace_id,
                    "cancel received by actor"
                );
                state.handle_cancel(&context).await;
                return;
            }
        }
    }

    // Inbox closed without a WindowClosing nor a Cancel — typically a daemon
    // shutdown. Mark the trace failed so its lifecycle is observable from the
    // DB and the registration coordinator doesn't pick it up.
    state.handle_shutdown_without_end(&context).await;
}

/// Per-actor mutable bookkeeping. Pulled out of `run` so the message handlers
/// can be tested with synthetic messages against a clean state object.
struct ActorState {
    identity: TraceIdentity,
    writer: TraceWriterKind,
    frame_count: u64,
    bytes_on_disk: u64,
    /// Last `bytes_written` value flushed to the DB. Used by the debouncer to
    /// avoid issuing a no-op UPDATE when the writer's on-disk size hasn't
    /// changed since the last flush.
    last_db_bytes: i64,
    /// Running count of frames the storage budget refused. Logged
    /// periodically so a runaway producer with no disk left doesn't drown
    /// the daemon log in identical warnings.
    dropped_over_budget: u64,
}

impl ActorState {
    fn new(identity: TraceIdentity) -> Self {
        Self {
            identity,
            writer: TraceWriterKind::Pending,
            frame_count: 0,
            bytes_on_disk: 0,
            last_db_bytes: 0,
            dropped_over_budget: 0,
        }
    }

    /// Enqueue the trace's row creation through the write-behind. Idempotent on
    /// `trace_id` (the batched insert is `ON CONFLICT DO NOTHING`).
    fn send_create(&self, context: &Arc<TraceActorContext>) {
        let key = &self.identity.key;
        context.trace_writer.create(
            &self.identity.trace_id,
            key.recording_index,
            Some(&key.data_type),
            key.sensor_name.as_deref(),
        );
    }

    async fn handle_data(
        &mut self,
        context: &Arc<TraceActorContext>,
        timestamp_ns: i64,
        _timestamp_s: Option<f64>,
        payload: Vec<u8>,
    ) {
        if !self.budget_allows_frame(&context.storage_budget, payload.len()) {
            return;
        }

        self.ensure_writer_open(context);

        // Try to mark `writing` exactly once. Subsequent frames don't need an
        // UPDATE for this field; the bytes-written debouncer covers the rest.
        let bumped_status = self.frame_count == 0;

        if let Err(error) = self.append_frame(context, timestamp_ns, payload) {
            tracing::warn!(
                %error,
                trace_id = self.identity.trace_id,
                "failed to append frame; marking trace failed"
            );
            self.mark_failed(context);
            return;
        }

        self.frame_count = self.frame_count.saturating_add(1);

        let bytes_changed = self.bytes_on_disk as i64 != self.last_db_bytes;
        let debounce_due = self
            .frame_count
            .is_multiple_of(BYTES_WRITTEN_DEBOUNCE_FRAMES);
        // Fire-and-forget into the coalescing write-behind (see module docs):
        // the first frame bumps `writing` and the debounced byte count rides
        // along.
        if bumped_status || (debounce_due && bytes_changed) {
            if bumped_status {
                context.trace_writer.mark_writing(&self.identity.trace_id);
            }
            if bytes_changed {
                context
                    .trace_writer
                    .progress(&self.identity.trace_id, self.bytes_on_disk as i64);
                self.last_db_bytes = self.bytes_on_disk as i64;
            }
        }
    }

    /// Reserve `payload_len` bytes against the storage budget before the frame
    /// is written. Uses `reserve` (not `check`) so the in-tree usage estimate is
    /// actually incremented on the write path — otherwise the cap only ever
    /// moved via the periodic rescan and `release`, letting the estimate drift
    /// low between scans (see `StorageBudget` docs). `reserve` only increments
    /// when the result is `Available`, so a refused frame doesn't over-count.
    fn budget_allows_frame(&mut self, budget: &Arc<StorageBudget>, payload_len: usize) -> bool {
        match budget.reserve(payload_len as u64) {
            Ok(check) if check.is_available() => true,
            Ok(check) => {
                self.dropped_over_budget = self.dropped_over_budget.saturating_add(1);
                if self.dropped_over_budget == 1 || self.dropped_over_budget.is_multiple_of(256) {
                    tracing::warn!(
                        trace_id = self.identity.trace_id,
                        dropped = self.dropped_over_budget,
                        ?check,
                        "storage budget refused frame; dropping"
                    );
                }
                false
            }
            Err(error) => {
                tracing::warn!(
                    %error,
                    trace_id = self.identity.trace_id,
                    "storage budget query failed; allowing frame through"
                );
                true
            }
        }
    }

    /// Lazily open the JSON writer for scalar traces. Video traces do not open
    /// a writer on the data path — they wait for the first `Video` message to
    /// allocate the video writer.
    ///
    /// The open is dispatched to the write-behind thread fire-and-forget; an
    /// open failure (e.g. disk full) is surfaced when the trace finalises,
    /// keeping this hot-path call non-blocking.
    fn ensure_writer_open(&mut self, context: &Arc<TraceActorContext>) {
        if !matches!(self.writer, TraceWriterKind::Pending) {
            return;
        }

        let trace_dir = self.trace_directory(context);
        context.json_writer.open(&self.identity.trace_id, trace_dir);
        self.bytes_on_disk = 0;
        self.writer = TraceWriterKind::Json;
    }

    fn append_frame(
        &mut self,
        context: &Arc<TraceActorContext>,
        timestamp_ns: i64,
        payload: Vec<u8>,
    ) -> Result<(), FrameAppendError> {
        match &self.writer {
            TraceWriterKind::Pending => Err(FrameAppendError::WriterNotOpen),
            TraceWriterKind::Json => {
                // Hand the entry to the write-behind thread, which preserves the
                // producer's bit-exact JSON formatting on the verbatim path and
                // wraps non-JSON payloads in a fallback object. Any write error
                // is deferred to finalise. `bytes_on_disk` is tracked as a
                // running estimate from the raw payload sizes — exact only at
                // finalise (the thread returns the true total there) — which is
                // ample for the debounced progress reports.
                self.bytes_on_disk = self.bytes_on_disk.saturating_add(payload.len() as u64);
                context
                    .json_writer
                    .append(&self.identity.trace_id, timestamp_ns, payload);
                Ok(())
            }
            TraceWriterKind::Video { .. } => {
                // Video traces no longer receive standalone data samples —
                // pixel data flows via `Video` messages. A stray sample for a
                // video trace is a producer bug; log it and ignore.
                tracing::warn!(
                    trace_id = self.identity.trace_id,
                    "video trace received standalone Data; ignoring"
                );
                Ok(())
            }
        }
    }

    /// Handle one finished NUT chunk: enqueue it and spawn a worker. The
    /// worker that wins an ffmpeg permit drains up to
    /// [`ENCODE_BATCH_MAX_CHUNKS`] queued chunks and transcodes them as one
    /// batch, then unlinks the source NUTs. When the encoder keeps pace the
    /// queue holds one chunk, so batches form only under backlog.
    #[allow(clippy::too_many_arguments)]
    async fn handle_video(
        &mut self,
        context: &Arc<TraceActorContext>,
        chunk_index: u32,
        spool_nut: PathBuf,
        width: u32,
        height: u32,
        byte_count: u64,
        frame_count: u32,
        frame_timestamps_s: Vec<f64>,
        dtype: FrameDtype,
    ) {
        let trace_dir = self.trace_directory(context);
        let chunks_dir = trace_dir.join(paths::CHUNKS_DIRNAME);

        // Allocate the video writer on the first chunk and mark the trace
        // `writing` so the registration coordinator can observe lifecycle
        // progress. The mark happens once per trace.
        let bumped_status = matches!(self.writer, TraceWriterKind::Pending);
        if bumped_status {
            // Resolve the lossy codec once, at the trace's first chunk, so every
            // chunk and the finalise concat agree even if the config changes
            // mid-recording. Read from the in-memory config the watcher keeps
            // current (env override + active profile); the RGB-only gate lives
            // in `LossyVideoCodec::for_trace`.
            let codec = LossyVideoCodec::for_trace(
                &self.identity.key.data_type,
                context.config_rx.borrow().video_codec.as_deref(),
            );
            self.writer = TraceWriterKind::Video {
                width,
                height,
                codec,
                completed_chunks: BTreeMap::new(),
                pending_encodes: JoinSet::new(),
                encode_queue: Arc::new(Mutex::new(VecDeque::new())),
                unencoded_chunks: Arc::new(AtomicUsize::new(0)),
            };
        }

        // Drain any background encodes that finished while we were idle.
        if self.drain_completed_encodes(context) {
            // A previous chunk's encode failed; mark_failed already ran, no
            // point spawning more work.
            return;
        }

        // Sanity-warn on resolution drift — the on-disk sidecar uses the
        // first-chunk values, so a producer bug shipping a different
        // resolution mid-trace would lose pixels silently.
        if let TraceWriterKind::Video {
            width: stored_width,
            height: stored_height,
            ..
        } = &self.writer
        {
            if (*stored_width, *stored_height) != (width, height) {
                tracing::warn!(
                    trace_id = self.identity.trace_id,
                    chunk_index,
                    stored = ?(*stored_width, *stored_height),
                    arrived = ?(width, height),
                    "video chunk resolution disagrees with first-chunk resolution"
                );
            }
        }

        let TraceWriterKind::Video {
            pending_encodes,
            codec,
            encode_queue,
            unencoded_chunks,
            ..
        } = &mut self.writer
        else {
            // Should be unreachable — we just allocated the writer above.
            return;
        };

        // Enqueue the chunk, then spawn a worker as a background task. The
        // actor returns to the inbox immediately so a slow ffmpeg invocation
        // cannot back-pressure unrelated joint / scalar publishers sharing the
        // commands service. A worker is not tied to "its" chunk: whichever one
        // wins a permit drains the queue front, so it may encode several
        // chunks or none.
        encode_queue
            .lock()
            .expect("encode queue lock")
            .push_back(QueuedChunk {
                chunk_index,
                spool_nut,
                byte_count,
                frame_count,
                frame_timestamps_s,
                dtype,
            });
        unencoded_chunks.fetch_add(1, Ordering::Relaxed);

        let worker = EncodeWorker {
            permits: context.ffmpeg_permits.clone(),
            permit_total: context.ffmpeg_permit_total,
            encode_cores: context.encode_cores,
            encoder: context.video_encoder.clone(),
            trace_id: self.identity.trace_id.clone(),
            trace_dir,
            chunks_dir,
            codec: *codec,
            queue: Arc::clone(encode_queue),
            unencoded_chunks: Arc::clone(unencoded_chunks),
        };
        pending_encodes.spawn(worker.run(chunk_index));

        // Stamp `writing` on the first chunk so the registration coordinator
        // sees the trace's lifecycle moving forward without waiting for the
        // first encode to complete.
        if bumped_status {
            context.trace_writer.mark_writing(&self.identity.trace_id);
        }
    }

    /// Drain every background encode that has already finished. On encode
    /// failure marks the trace failed and returns `true`; otherwise returns
    /// `false`. Caller-side use: gate further work on the return value.
    fn drain_completed_encodes(&mut self, context: &Arc<TraceActorContext>) -> bool {
        let TraceWriterKind::Video {
            completed_chunks,
            pending_encodes,
            ..
        } = &mut self.writer
        else {
            return false;
        };
        let mut any_failure = false;
        let mut new_bytes: u64 = 0;
        let mut new_frames: u64 = 0;
        while let Some(joined) = pending_encodes.try_join_next() {
            match joined {
                Ok(EncodeWorkerOutcome::Completed {
                    chunk_index,
                    completed,
                }) => {
                    new_bytes = new_bytes.saturating_add(completed.bytes);
                    new_frames = new_frames.saturating_add(completed.frame_count as u64);
                    completed_chunks.insert(chunk_index, completed);
                }
                Ok(EncodeWorkerOutcome::NoOp) => {}
                Ok(EncodeWorkerOutcome::Failed { chunk_index, error }) => {
                    tracing::warn!(
                        %error,
                        trace_id = self.identity.trace_id,
                        chunk_index,
                        "failed to encode video chunk batch"
                    );
                    any_failure = true;
                }
                Err(join_error) => {
                    tracing::warn!(
                        %join_error,
                        trace_id = self.identity.trace_id,
                        "video encode task join failed"
                    );
                    any_failure = true;
                }
            }
        }
        if new_bytes > 0 || new_frames > 0 {
            self.bytes_on_disk = self.bytes_on_disk.saturating_add(new_bytes);
            self.frame_count = self.frame_count.saturating_add(new_frames);
            let bytes_changed = self.bytes_on_disk as i64 != self.last_db_bytes;
            if bytes_changed {
                context
                    .trace_writer
                    .progress(&self.identity.trace_id, self.bytes_on_disk as i64);
                self.last_db_bytes = self.bytes_on_disk as i64;
            }
        }
        if any_failure {
            self.mark_failed(context);
        }
        any_failure
    }

    async fn finalise_trace(&mut self, context: &Arc<TraceActorContext>) {
        let started = Instant::now();
        crate::perf_events::emit(
            "trace_finalization",
            "started",
            Some(self.identity.key.recording_index),
            Some(&self.identity.trace_id),
            None,
            serde_json::json!({
                "data_type": self.identity.key.data_type,
                "frame_count": self.frame_count,
            }),
        );
        let writer = std::mem::replace(&mut self.writer, TraceWriterKind::Pending);
        let finalise = self.finalise_writer(writer, context).await;
        match finalise {
            Ok(total_bytes) => {
                self.bytes_on_disk = total_bytes;
                context
                    .trace_writer
                    .finalise(&self.identity.trace_id, total_bytes as i64);
                tracing::info!(
                    trace_id = self.identity.trace_id,
                    recording_index = self.identity.key.recording_index,
                    frame_count = self.frame_count,
                    dropped_over_budget = self.dropped_over_budget,
                    total_bytes,
                    "trace finalised"
                );
                crate::perf_events::emit(
                    "trace_finalization",
                    "completed",
                    Some(self.identity.key.recording_index),
                    Some(&self.identity.trace_id),
                    Some(started.elapsed()),
                    serde_json::json!({
                        "data_type": self.identity.key.data_type,
                        "frame_count": self.frame_count,
                        "total_bytes": total_bytes,
                        "outcome": "ok",
                    }),
                );
                if let Some(bus) = context.event_bus.as_ref() {
                    bus.publish(crate::state::DaemonEvent::TraceWritten {
                        trace_id: self.identity.trace_id.clone(),
                        recording_index: self.identity.key.recording_index,
                    });
                }
            }
            Err(error) => {
                tracing::warn!(
                    %error,
                    trace_id = self.identity.trace_id,
                    "failed to finalise trace artefacts"
                );
                crate::perf_events::emit(
                    "trace_finalization",
                    "failed",
                    Some(self.identity.key.recording_index),
                    Some(&self.identity.trace_id),
                    Some(started.elapsed()),
                    serde_json::json!({
                        "data_type": self.identity.key.data_type,
                        "outcome": "error",
                        "error_kind": error.to_string(),
                    }),
                );
                self.mark_failed(context);
            }
        }
    }

    async fn finalise_writer(
        &mut self,
        writer: TraceWriterKind,
        context: &Arc<TraceActorContext>,
    ) -> Result<u64, FrameAppendError> {
        match writer {
            TraceWriterKind::Pending => {
                // Empty trace — no encoder was ever opened. Leave a single
                // empty `trace.json` behind so the artefact set is complete:
                // open it on the write-behind thread then finalise it.
                let trace_dir = self.trace_directory(context);
                context.json_writer.open(&self.identity.trace_id, trace_dir);
                Ok(context.json_writer.finish(&self.identity.trace_id).await?)
            }
            TraceWriterKind::Json => {
                Ok(context.json_writer.finish(&self.identity.trace_id).await?)
            }
            TraceWriterKind::Video {
                width,
                height,
                codec,
                mut completed_chunks,
                mut pending_encodes,
                encode_queue: _,
                unencoded_chunks,
            } => {
                let encode_started = Instant::now();
                // Un-encoded chunks, not worker tasks: the queue length plus
                // the chunks covered by in-flight batches.
                let pending_encode_count = unencoded_chunks.load(Ordering::Relaxed);
                crate::perf_events::emit(
                    "video_encoding",
                    "started",
                    Some(self.identity.key.recording_index),
                    Some(&self.identity.trace_id),
                    None,
                    serde_json::json!({
                        "scope": "post_close_pending_encode_drain",
                        "pending_encode_count": pending_encode_count,
                        "already_completed_chunks": completed_chunks.len(),
                    }),
                );
                // Drain every still-running encode worker; a no-op worker
                // reaps silently. A failure here is terminal — without a
                // complete chunk set the concat would produce a video with a
                // missing range, which is worse than marking the trace failed.
                while let Some(joined) = pending_encodes.join_next().await {
                    match joined {
                        Ok(EncodeWorkerOutcome::Completed {
                            chunk_index,
                            completed,
                        }) => {
                            completed_chunks.insert(chunk_index, completed);
                        }
                        Ok(EncodeWorkerOutcome::NoOp) => {}
                        Ok(EncodeWorkerOutcome::Failed { error, .. }) => {
                            return Err(error.into());
                        }
                        Err(join_error) => {
                            return Err(FrameAppendError::VideoEncode(VideoEncodeError::Spawn {
                                binary: std::ffi::OsString::from("ffmpeg"),
                                source: std::io::Error::other(format!(
                                    "video encode task join failed: {join_error}"
                                )),
                            }))
                        }
                    }
                }
                // The running total is only bumped by `drain_completed_encodes`,
                // which misses encodes that finish here, at window close.
                self.frame_count = completed_chunks
                    .values()
                    .map(|chunk| chunk.frame_count as u64)
                    .sum();
                crate::perf_events::emit(
                    "video_encoding",
                    "completed",
                    Some(self.identity.key.recording_index),
                    Some(&self.identity.trace_id),
                    Some(encode_started.elapsed()),
                    serde_json::json!({
                        "scope": "post_close_pending_encode_drain",
                        "pending_encode_count": pending_encode_count,
                        "completed_chunks": completed_chunks.len(),
                        "outcome": "ok",
                    }),
                );

                if completed_chunks.is_empty() {
                    // The trace allocated a Video writer but every chunk
                    // failed (or none ever landed) — fall back to the empty
                    // trace.json path so the artefact set isn't missing a
                    // sidecar entirely.
                    let trace_dir = self.trace_directory(context);
                    context.json_writer.open(&self.identity.trace_id, trace_dir);
                    return Ok(context.json_writer.finish(&self.identity.trace_id).await?);
                }

                // In lossy-only mode (nc.Codec.H264_MEDIUM) no lossless archive
                // is produced — only lossy.mp4 is stitched and uploaded.
                let lossy_only = codec.is_lossy_only();

                let trace_dir = self.trace_directory(context);
                let lossy_out = trace_dir.join(paths::LOSSY_VIDEO_FILENAME);
                let lossless_out = trace_dir.join(paths::LOSSLESS_VIDEO_FILENAME);

                // BTreeMap iteration is sorted by chunk_index, so the concat
                // segment lists are guaranteed in producer-arrival order
                // regardless of encode completion order.
                let lossy_segments: Vec<PathBuf> = completed_chunks
                    .values()
                    .map(|chunk| chunk.lossy_segment.clone())
                    .collect();
                let lossless_segments: Vec<PathBuf> = if lossy_only {
                    Vec::new()
                } else {
                    completed_chunks
                        .values()
                        .map(|chunk| chunk.lossless_segment.clone())
                        .collect()
                };

                // Each segment except the last declares its capture span to
                // the next, so the concat lands every frame on its
                // trace-relative capture timestamp instead of accumulating
                // per-segment probe drift.
                let segment_spans_us: Vec<i64> = completed_chunks
                    .values()
                    .zip(completed_chunks.values().skip(1))
                    .map(|(segment, next)| declared_finalise_span_us(segment, next))
                    .collect();

                // Build the metadata accumulator in the same chunk-index
                // order so per-frame entries appear in capture order. Each
                // chunk applies its own stored dtype (not just the first
                // chunk's) to its frame entries — a depth chunk's entries gain
                // a `"dtype"` field carrying the canonical `trace.json` string
                // ("float16" / "float32"); RGB chunks add nothing, keeping the
                // existing RGB `trace.json` schema byte-for-byte unchanged.
                let mut metadata = VideoMetadataAccumulator::new();
                for chunk in completed_chunks.values() {
                    for timestamp_s in &chunk.frame_timestamps_s {
                        let mut entry = serde_json::Map::new();
                        entry.insert("timestamp".to_string(), Value::from(*timestamp_s));
                        entry.insert("width".to_string(), Value::from(width as u64));
                        entry.insert("height".to_string(), Value::from(height as u64));
                        if let Some(dtype_label) = chunk.dtype.depth_label() {
                            entry.insert("dtype".to_string(), Value::from(dtype_label));
                        }
                        metadata.record_frame(entry);
                    }
                }

                // Concat is stream-copy: cheap relative to encode but still
                // bounded by an ffmpeg permit so a tail-stitch storm
                // doesn't fork-bomb the host.
                let concat_started = Instant::now();
                // `chunks` counts encoded segments, one per batch, which
                // equals the trace's chunk count only when the encoder kept
                // pace. Integration reporting reads the field by this name.
                crate::perf_events::emit(
                    "video_concatenation",
                    "started",
                    Some(self.identity.key.recording_index),
                    Some(&self.identity.trace_id),
                    None,
                    serde_json::json!({
                        "chunks": completed_chunks.len(),
                        "lossy_only": lossy_only,
                    }),
                );
                let permit = context
                    .ffmpeg_permits
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| FrameAppendError::FfmpegPermits)?;
                let lossy_outcome = context
                    .video_encoder
                    .concat_segments(&lossy_segments, &segment_spans_us, &lossy_out)
                    .await?;
                let lossless_bytes = if lossy_only {
                    0
                } else {
                    context
                        .video_encoder
                        .concat_segments(&lossless_segments, &segment_spans_us, &lossless_out)
                        .await?
                        .bytes
                };
                drop(permit);

                // Unlink per-chunk segments now that the final outputs are
                // sealed. Best-effort: a leftover segment is wasted disk
                // space, not a correctness problem.
                for segment in lossy_segments.iter().chain(lossless_segments.iter()) {
                    if let Err(error) = std::fs::remove_file(segment) {
                        if error.kind() != std::io::ErrorKind::NotFound {
                            tracing::warn!(
                                %error,
                                trace_id = self.identity.trace_id,
                                path = %segment.display(),
                                "failed to remove encoded chunk segment after concat"
                            );
                        }
                    }
                }

                // Sidecar metadata is the *last* thing on disk so a partial
                // transcode failure leaves a recognisable "no sidecar"
                // signature for the recovery sweep.
                let metadata_bytes = flush_metadata_blocking(metadata, trace_dir.clone()).await?;

                let total_bytes = lossy_outcome
                    .bytes
                    .saturating_add(lossless_bytes)
                    .saturating_add(metadata_bytes);
                crate::perf_events::emit(
                    "video_concatenation",
                    "completed",
                    Some(self.identity.key.recording_index),
                    Some(&self.identity.trace_id),
                    Some(concat_started.elapsed()),
                    serde_json::json!({
                        "chunks": completed_chunks.len(),
                        "total_bytes": total_bytes,
                        "outcome": "ok",
                    }),
                );

                tracing::debug!(
                    trace_id = self.identity.trace_id,
                    chunks_encoded = completed_chunks.len(),
                    "video trace concatenated"
                );

                Ok(total_bytes)
            }
        }
    }

    async fn handle_shutdown_without_end(&mut self, context: &Arc<TraceActorContext>) {
        self.mark_failed(context);
    }

    /// Enqueue a `failed` write for this trace, preserving the latest byte
    /// count. Fire-and-forget through the coalescing batcher; the terminal
    /// guard in `apply_trace_writes` keeps it from clobbering an
    /// already-`written` row.
    fn mark_failed(&mut self, context: &Arc<TraceActorContext>) {
        context
            .trace_writer
            .fail(&self.identity.trace_id, self.bytes_on_disk as i64);
    }

    /// Tear down the writer and release the trace's disk budget.
    ///
    /// Called when the parent recording is cancelled. The on-disk artefacts are
    /// *not* removed here: the recording reaper deletes the whole recording
    /// directory (and the DB rows) together once the cancel has been durably
    /// notified to the backend, so it is the single owner of cancelled-recording
    /// file removal. The DB row's `write_status` is left untouched here — the
    /// dispatcher issues a single `cancel_recording` transaction once every
    /// actor has exited.
    async fn handle_cancel(&mut self, context: &Arc<TraceActorContext>) {
        // Discard any open JSON write-behind writer (no-op for video / unopened
        // traces) so its file handle is released without finalising, then drop
        // the actor-side writer marker.
        // Only scalar/JSON traces reserve against the budget on the write path
        // (via `budget_allows_frame`). A video trace's `bytes_on_disk` is encoder
        // output that was never reserved, so releasing it here would drive the
        // estimate below true usage; release only what a JSON trace reserved.
        let reserved_json_bytes = matches!(self.writer, TraceWriterKind::Json);
        context.json_writer.drop_trace(&self.identity.trace_id);
        self.writer = TraceWriterKind::Pending;
        if reserved_json_bytes && self.bytes_on_disk > 0 {
            context.storage_budget.release(self.bytes_on_disk);
        }
        self.bytes_on_disk = 0;
        self.last_db_bytes = 0;
    }

    /// Build the on-disk directory for this trace:
    /// `{recordings_root}/{recording_index}/{data_type}/{trace_id}/`.
    fn trace_directory(&self, context: &Arc<TraceActorContext>) -> std::path::PathBuf {
        TracePath::new(
            self.identity.key.recording_index.to_string(),
            self.identity.key.data_type.clone(),
            self.identity.trace_id.clone(),
        )
        .directory(context.recordings_root.as_path())
    }
}

/// Everything one spawned encode worker carries: the shared permit pool and
/// chunk queue plus the per-trace paths and codec its batch encode needs.
struct EncodeWorker {
    permits: Arc<Semaphore>,
    permit_total: usize,
    encode_cores: usize,
    encoder: VideoEncoder,
    trace_id: String,
    trace_dir: PathBuf,
    chunks_dir: PathBuf,
    codec: LossyVideoCodec,
    queue: Arc<Mutex<VecDeque<QueuedChunk>>>,
    unencoded_chunks: Arc<AtomicUsize>,
}

impl EncodeWorker {
    /// Wait for an ffmpeg permit, then drain and encode one batch from the
    /// queue front. `arrival_index` is the chunk whose arrival spawned this
    /// worker; it is the `chunk_index` reported when the permit pool closes
    /// before any batch is drained.
    async fn run(self, arrival_index: u32) -> EncodeWorkerOutcome {
        // Acquire a permit before relinking + encoding. Gating the relink
        // (which frees the producer's on-disk spool) on a permit is
        // deliberate: it makes the fixed spool cap the backstop that bounds
        // the un-encoded backlog. When the encoder can't keep up, chunks stay
        // in the spool until a permit frees, the spool fills, and the
        // *producer* back-pressures — rather than the daemon-side chunks dir
        // growing without bound. Adaptive threading (below) keeps the encoder
        // fast enough that this backstop rarely engages. Clone the `Arc` for
        // the (consuming) acquire so `permits` stays live for the
        // `available_permits()` read below.
        let permit = match self.permits.clone().acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                return EncodeWorkerOutcome::Failed {
                    chunk_index: arrival_index,
                    error: VideoEncodeError::Spawn {
                        binary: std::ffi::OsString::from("ffmpeg"),
                        source: std::io::Error::other("ffmpeg permit pool closed"),
                    },
                };
            }
        };
        let batch = drain_encode_batch(&mut self.queue.lock().expect("encode queue lock"));
        if batch.is_empty() {
            // An earlier worker's batch already covered this arrival's chunk.
            return EncodeWorkerOutcome::NoOp;
        }
        let batch_len = batch.len();
        let outcome = self.encode_batch(batch).await;
        // The drained chunks stop counting as un-encoded once their batch
        // has an outcome, success or failure alike.
        self.unencoded_chunks
            .fetch_sub(batch_len, Ordering::Relaxed);
        drop(permit);
        outcome
    }

    /// Relink and encode one drained batch, then unlink its NUTs on success.
    /// A failed batch leaves every relinked NUT on disk for the recovery
    /// sweep.
    async fn encode_batch(&self, batch: Vec<QueuedChunk>) -> EncodeWorkerOutcome {
        let first_index = batch[0].chunk_index;
        let dtype = batch[0].dtype;
        let raw_nuts: Vec<PathBuf> = batch
            .iter()
            .map(|chunk| {
                self.chunks_dir
                    .join(paths::chunk_filename(chunk.chunk_index))
            })
            .collect();

        // Relink every producer-spooled NUT into the recording's chunks dir
        // here rather than on the dispatcher's routing path. The `rename`
        // (and `mkdir`) are filesystem metadata ops that can stall behind an
        // ext4 journal commit on the shared spool, so we run them on a
        // blocking thread — off both the dispatcher and the runtime workers.
        let relink = {
            let spools: Vec<PathBuf> = batch.iter().map(|chunk| chunk.spool_nut.clone()).collect();
            let destinations = raw_nuts.clone();
            let chunks_dir = self.chunks_dir.clone();
            tokio::task::spawn_blocking(move || -> Result<(), (PathBuf, std::io::Error)> {
                for (spool, dest) in spools.into_iter().zip(&destinations) {
                    relink_nut(&spool, &chunks_dir, dest).map_err(|source| (spool, source))?;
                }
                Ok(())
            })
            .await
        };
        match relink {
            Ok(Ok(())) => {}
            Ok(Err((spool, source))) => {
                return EncodeWorkerOutcome::Failed {
                    chunk_index: first_index,
                    error: VideoEncodeError::Io {
                        path: spool,
                        source,
                    },
                };
            }
            Err(join_error) => {
                return EncodeWorkerOutcome::Failed {
                    chunk_index: first_index,
                    error: VideoEncodeError::Spawn {
                        binary: std::ffi::OsString::from("relink"),
                        source: std::io::Error::other(format!(
                            "relink task join failed: {join_error}"
                        )),
                    },
                };
            }
        }

        // Each entry except the last declares the capture span to the next
        // chunk's first frame, floored at the chunk's own PTS extent, so the
        // concat demuxer lands every frame on its batch-relative capture
        // timestamp and never rewinds at an overlapping announcement.
        let spans_to_next_us: Vec<i64> = batch
            .windows(2)
            .map(|pair| {
                declared_segment_span_us(&pair[0].frame_timestamps_s, &pair[1].frame_timestamps_s)
            })
            .collect();
        // The placement those duration lines dictate is the segment's real
        // content extent, which the finalise span floors on.
        let content_extent_us = batch_content_extent_us(
            &spans_to_next_us,
            &batch
                .last()
                .expect("drained batch is never empty")
                .frame_timestamps_s,
        );
        let inputs: Vec<BatchNutInput> = batch
            .iter()
            .zip(&raw_nuts)
            .enumerate()
            .map(|(position, (chunk, raw_nut))| BatchNutInput {
                raw_nut: raw_nut.clone(),
                // One entry per non-last chunk, so the last gets no line.
                span_to_next_us: spans_to_next_us.get(position).copied(),
                frame_count: chunk.frame_count,
            })
            .collect();
        let request = BatchEncodeRequest {
            inputs,
            lossy_out: self
                .trace_dir
                .join(paths::chunk_lossy_filename(first_index)),
            lossless_out: self
                .trace_dir
                .join(paths::chunk_lossless_filename(first_index)),
            codec: self.codec,
        };

        // Size this encode's thread pool to the cores the rest of the fleet
        // isn't using: with the permit held, `available_permits()` is the
        // idle capacity, so `total - available` is how many encodes (incl.
        // this one) are running now. Read after acquiring so we count
        // ourselves.
        let active_encodes = self
            .permit_total
            .saturating_sub(self.permits.available_permits());
        let encode_threads = adaptive_encode_threads(self.encode_cores, active_encodes);
        match self
            .encoder
            .encode_chunk_batch(&request, encode_threads)
            .await
        {
            Ok(encode) => {
                // Drop the source NUT chunks now that the segments are sealed
                // and verified non-empty. Failure to unlink leaves a file for
                // the recovery sweep to collect.
                for raw_nut in &raw_nuts {
                    if let Err(error) = std::fs::remove_file(raw_nut) {
                        if error.kind() != std::io::ErrorKind::NotFound {
                            tracing::warn!(
                                %error,
                                trace_id = %self.trace_id,
                                path = %raw_nut.display(),
                                "failed to remove source NUT chunk after encode"
                            );
                        }
                    }
                }
                let frame_count = batch.iter().map(|chunk| chunk.frame_count).sum::<u32>();
                tracing::debug!(
                    trace_id = %self.trace_id,
                    first_chunk_index = first_index,
                    batch_len = batch.len(),
                    frame_count,
                    byte_count = batch.iter().map(|chunk| chunk.byte_count).sum::<u64>(),
                    lossy_bytes = encode.lossy_bytes,
                    lossless_bytes = encode.lossless_bytes,
                    "video chunk batch encoded"
                );
                let frame_timestamps_s: Vec<f64> = batch
                    .into_iter()
                    .flat_map(|chunk| chunk.frame_timestamps_s)
                    .collect();
                EncodeWorkerOutcome::Completed {
                    chunk_index: first_index,
                    completed: CompletedChunk {
                        lossy_segment: request.lossy_out,
                        lossless_segment: request.lossless_out,
                        bytes: encode.lossy_bytes.saturating_add(encode.lossless_bytes),
                        frame_timestamps_s,
                        frame_count,
                        content_extent_us,
                        dtype,
                    },
                }
            }
            Err(error) => EncodeWorkerOutcome::Failed {
                chunk_index: first_index,
                error,
            },
        }
    }
}

/// The declared concat span from one chunk's announced frames to the next's,
/// used by the batch worker and the drain span cap. A next chunk with no
/// announced frames borrows this chunk's first stamp, so the span floors to
/// this chunk's extent plus 1 us instead of indexing an empty vector.
fn declared_segment_span_us(frame_timestamps_s: &[f64], next_frame_timestamps_s: &[f64]) -> i64 {
    let next_first_timestamp_s = next_frame_timestamps_s
        .first()
        .or_else(|| frame_timestamps_s.first())
        .copied()
        .unwrap_or(0.0);
    declared_batch_span_us(frame_timestamps_s, next_first_timestamp_s)
}

/// The declared finalise span from one encoded segment to the next. Flooring
/// on the carried content extent, rather than a replay of the announced
/// stamps, is what guarantees the next segment never starts inside this
/// segment's real mp4 content.
fn declared_finalise_span_us(segment: &CompletedChunk, next: &CompletedChunk) -> i64 {
    let next_first_timestamp_s = next
        .frame_timestamps_s
        .first()
        .or_else(|| segment.frame_timestamps_s.first())
        .copied()
        .unwrap_or(0.0);
    declared_span_with_extent_us(
        &segment.frame_timestamps_s,
        next_first_timestamp_s,
        segment.content_extent_us,
    )
}

/// Take up to [`ENCODE_BATCH_MAX_CHUNKS`] descriptors from the queue front,
/// stopping early at a dtype change and before a chunk whose declared span
/// exceeds [`ENCODE_BATCH_MAX_SPAN_US`]. A stopped-at chunk stays at the
/// front for the worker its own arrival spawned.
fn drain_encode_batch(queue: &mut VecDeque<QueuedChunk>) -> Vec<QueuedChunk> {
    let mut batch: Vec<QueuedChunk> = Vec::new();
    while batch.len() < ENCODE_BATCH_MAX_CHUNKS {
        let Some(front) = queue.front() else { break };
        if batch
            .first()
            .is_some_and(|first| first.dtype != front.dtype)
        {
            break;
        }
        if batch.last().is_some_and(|last| {
            declared_segment_span_us(&last.frame_timestamps_s, &front.frame_timestamps_s)
                > ENCODE_BATCH_MAX_SPAN_US
        }) {
            break;
        }
        let chunk = queue.pop_front().expect("front exists");
        batch.push(chunk);
    }
    batch
}

/// Errors that can surface while appending or finalising a frame. The variants
/// are unified so `handle_data` / `finalise_trace` can log + mark-failed in
/// one place regardless of which writer raised.
#[derive(Debug, thiserror::Error)]
enum FrameAppendError {
    #[error("trace writer not open")]
    WriterNotOpen,
    #[error("ffmpeg permit pool closed before transcode could start")]
    FfmpegPermits,
    #[error(transparent)]
    Json(#[from] JsonTraceError),
    #[error(transparent)]
    VideoEncode(#[from] VideoEncodeError),
    #[error(transparent)]
    Metadata(#[from] MetadataError),
}

/// Flush the in-memory metadata accumulator to `trace.json` on a blocking
/// thread.
async fn flush_metadata_blocking(
    metadata: VideoMetadataAccumulator,
    output_dir: std::path::PathBuf,
) -> Result<u64, FrameAppendError> {
    let path_for_error = output_dir.clone();
    let handle = task::spawn_blocking(move || metadata.finish(&output_dir));
    match handle.await {
        Ok(result) => Ok(result?),
        Err(join_error) => Err(FrameAppendError::Metadata(MetadataError::Write {
            path: path_for_error,
            source: std::io::Error::other(format!("metadata flush join failed: {join_error}")),
        })),
    }
}

/// Relink a producer-spooled NUT into the recording's chunks directory.
/// Prefers an atomic rename (same filesystem); falls back to copy + remove.
/// Blocking — the actor runs it via `spawn_blocking` inside the background
/// encode task so the rename can't stall the dispatcher or a runtime worker.
fn relink_nut(
    src: &std::path::Path,
    chunks_dir: &std::path::Path,
    dest: &std::path::Path,
) -> std::io::Result<()> {
    std::fs::create_dir_all(chunks_dir)?;
    match std::fs::rename(src, dest) {
        Ok(()) => Ok(()),
        Err(_) => {
            std::fs::copy(src, dest)?;
            let _ = std::fs::remove_file(src);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{SqliteStateStore, StateStore, TraceWriteStatus};
    use crate::storage::budget::StoragePolicy;
    use serde_json::json;
    use std::time::Duration;
    use tempfile::TempDir;

    /// Build an actor context whose write-behind flushes into `store`. The
    /// [`TraceEventDatabaseWriter`] owner is dropped — the spawned task stays alive while the
    /// handle in the returned context lives (dropping its `JoinHandle` detaches,
    /// not cancels). Tests call `context.trace_writer.flush().await` before
    /// asserting on the DB, since actor writes are now fire-and-forget.
    fn test_context(
        root: &std::path::Path,
        store: Arc<SqliteStateStore>,
    ) -> Arc<TraceActorContext> {
        test_context_with_permits(
            root,
            store,
            Arc::new(Semaphore::new(default_ffmpeg_concurrency())),
        )
    }

    /// As [`test_context`] but with an explicit ffmpeg permit pool, so the
    /// batching tests can hold the only permit to force a queue backlog.
    fn test_context_with_permits(
        root: &std::path::Path,
        store: Arc<SqliteStateStore>,
        ffmpeg_permits: Arc<Semaphore>,
    ) -> Arc<TraceActorContext> {
        let policy = StoragePolicy {
            storage_limit_bytes: None,
            min_free_disk_bytes: 0,
            refresh_interval: Duration::from_secs(60),
        };
        let budget = Arc::new(StorageBudget::new(root, policy));
        let (trace_writer, _writer_owner) = crate::state::trace_event_database_writer::spawn(store);
        let (json_writer, _json_owner) = crate::pipeline::json_writer::spawn();
        Arc::new(TraceActorContext::with_ffmpeg_permits(
            root.to_path_buf(),
            budget,
            VideoEncoder::new(),
            ffmpeg_permits,
            trace_writer,
            json_writer,
        ))
    }

    fn identity(recording_index: i64, trace_id: &str, data_type: &str) -> TraceIdentity {
        TraceIdentity {
            trace_id: trace_id.to_string(),
            key: TraceKey {
                recording_index,
                data_type: data_type.to_string(),
                sensor_name: None,
            },
        }
    }

    fn ffmpeg_available() -> bool {
        binary_available("ffmpeg")
    }

    fn binary_available(name: &str) -> bool {
        std::process::Command::new(name)
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    fn ffprobe_available() -> bool {
        binary_available("ffprobe")
    }

    /// Decode `video` with ffprobe and collect frame PTS in presentation
    /// order, as the cloud guard walks them.
    fn decoded_frame_pts(video: &std::path::Path) -> Vec<i64> {
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=pts,pkt_pts",
                "-of",
                "default=noprint_wrappers=1",
            ])
            .arg(video)
            .output()
            .expect("spawn ffprobe");
        assert!(probe.status.success());
        String::from_utf8_lossy(&probe.stdout)
            .lines()
            .filter_map(|line| {
                line.strip_prefix("pts=")
                    .or_else(|| line.strip_prefix("pkt_pts="))
            })
            .filter_map(|value| value.parse().ok())
            .collect()
    }

    #[test]
    fn scalar_fallback_entry_wraps_non_json_payload() {
        let entry = crate::pipeline::json_writer::scalar_fallback_entry(123, &[0xFF, 0xFE]);
        assert_eq!(entry, json!({"timestamp_ns": 123, "payload_len": 2}));
    }

    #[test]
    fn adaptive_threads_fill_idle_cores_and_collapse_at_full_load() {
        let cores = 14;
        // Pool full (cores / 2 encodes running): each collapses to the floor,
        // so total encoder threads stay ~= 2 * cores — the original behaviour.
        let full = default_ffmpeg_concurrency_for(cores);
        assert_eq!(
            adaptive_encode_threads(cores, full),
            ENCODE_THREADS_PER_OUTPUT
        );
        // One encode on an otherwise-idle host takes the whole box; two split it
        // — total encoder threads still ~= 2 * cores at any concurrency.
        assert_eq!(adaptive_encode_threads(cores, 1), cores);
        assert_eq!(adaptive_encode_threads(cores, 2), cores / 2);
        // Never below the floor even if the count is somehow over-subscribed,
        // and degenerate inputs don't panic or divide by zero.
        assert_eq!(
            adaptive_encode_threads(cores, cores * 4),
            ENCODE_THREADS_PER_OUTPUT
        );
        assert_eq!(adaptive_encode_threads(0, 0), ENCODE_THREADS_PER_OUTPUT);
    }

    /// Mirror of [`default_ffmpeg_concurrency`] for an explicit core count, so
    /// the adaptive-threads test can assert the full-load boundary without
    /// depending on the host's real parallelism.
    fn default_ffmpeg_concurrency_for(cores: usize) -> usize {
        (cores / ENCODE_THREADS_PER_OUTPUT).max(2)
    }

    #[tokio::test]
    async fn json_trace_writes_array_on_finalise() {
        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let store_arc = Arc::new(store.clone());
        let context = test_context(&tempdir.path().join("recordings"), store_arc.clone());

        let mut state = ActorState::new(identity(7, "trace-1", "joints"));
        state.send_create(&context);
        for index in 0..3i64 {
            let payload = serde_json::to_vec(&json!({"i": index})).unwrap();
            state
                .handle_data(&context, index * 1_000_000, None, payload)
                .await;
        }
        state.finalise_trace(&context).await;
        context.trace_writer.flush().await;

        let trace_dir =
            TracePath::new("7", "joints", "trace-1").directory(context.recordings_root.as_path());
        let bytes = std::fs::read(trace_dir.join("trace.json")).unwrap();
        let parsed: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed, json!([{"i": 0}, {"i": 1}, {"i": 2}]));

        let trace = store
            .get_trace("trace-1")
            .await
            .expect("get trace")
            .expect("trace exists");
        assert_eq!(trace.write_status, TraceWriteStatus::Written);
        assert_eq!(trace.recording_index, 7);
        assert_eq!(trace.total_bytes as u64, bytes.len() as u64);
    }

    #[tokio::test]
    async fn empty_trace_still_produces_valid_json_array() {
        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let store_arc = Arc::new(store.clone());
        let context = test_context(&tempdir.path().join("recordings"), store_arc.clone());

        let mut state = ActorState::new(identity(1, "trace-1", "joints"));
        state.send_create(&context);
        state.finalise_trace(&context).await;
        context.trace_writer.flush().await;

        let trace_dir =
            TracePath::new("1", "joints", "trace-1").directory(context.recordings_root.as_path());
        let bytes = std::fs::read(trace_dir.join("trace.json")).unwrap();
        assert_eq!(bytes, b"[]");

        let trace = store
            .get_trace("trace-1")
            .await
            .expect("get trace")
            .expect("trace exists");
        assert_eq!(trace.write_status, TraceWriteStatus::Written);
    }

    #[tokio::test]
    async fn video_chunks_concat_on_finalise() {
        if !ffmpeg_available() {
            eprintln!("ffmpeg not on PATH — skipping video trace_actor test.");
            return;
        }

        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let store_arc = Arc::new(store.clone());
        let context = test_context(&tempdir.path().join("recordings"), store_arc.clone());

        let mut state = ActorState::new(identity(1, "trace-vid", "RGB"));
        state.send_create(&context);

        // Build two NUT chunks via ffmpeg testsrc in a spool location; the actor
        // relinks each into the recording's chunks dir before transcoding, just
        // as it does for a producer-spooled chunk in production.
        let trace_dir =
            TracePath::new("1", "RGB", "trace-vid").directory(context.recordings_root.as_path());
        let chunks_dir = trace_dir.join(paths::CHUNKS_DIRNAME);
        let spool_dir = tempdir.path().join("spool");
        std::fs::create_dir_all(&spool_dir).unwrap();

        for chunk_index in 0..2u32 {
            let spool_nut = spool_dir.join(format!("chunk_{chunk_index}.nut"));
            let status = std::process::Command::new("ffmpeg")
                .args([
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                ])
                .arg("testsrc=duration=4:size=16x16:rate=1")
                .args(["-c:v", "rawvideo", "-pix_fmt", "rgb24", "-f", "nut"])
                .arg(&spool_nut)
                .status()
                .expect("synth status");
            assert!(status.success(), "synth NUT failed");

            let byte_count = spool_nut.metadata().unwrap().len();
            let frame_timestamps_s: Vec<f64> =
                (0..4u32).map(|i| (chunk_index * 4 + i) as f64).collect();
            state
                .handle_video(
                    &context,
                    chunk_index,
                    spool_nut,
                    16,
                    16,
                    byte_count,
                    4,
                    frame_timestamps_s,
                    FrameDtype::Rgb8,
                )
                .await;
        }

        state.finalise_trace(&context).await;
        context.trace_writer.flush().await;

        assert!(trace_dir.join(paths::LOSSY_VIDEO_FILENAME).exists());
        assert!(trace_dir.join(paths::LOSSLESS_VIDEO_FILENAME).exists());
        assert!(trace_dir.join(paths::TRACE_JSON_FILENAME).exists());
        for chunk_index in 0..2u32 {
            assert!(!chunks_dir.join(paths::chunk_filename(chunk_index)).exists());
        }

        let trace = store
            .get_trace("trace-vid")
            .await
            .expect("get trace")
            .expect("trace exists");
        assert_eq!(trace.write_status, TraceWriteStatus::Written);
        assert!(trace.total_bytes > 0);
        assert_eq!(
            state.frame_count, 8,
            "finalised frame count must cover every chunk, not just those \
             drained before close"
        );

        // RGB metadata entries must stay exactly as before: no `dtype` field.
        let sidecar: Value = serde_json::from_slice(
            &std::fs::read(trace_dir.join(paths::TRACE_JSON_FILENAME)).unwrap(),
        )
        .unwrap();
        for entry in sidecar.as_array().unwrap() {
            assert!(
                entry.as_object().unwrap().get("dtype").is_none(),
                "RGB trace.json entries must not gain a dtype field: {entry}"
            );
        }
    }

    #[tokio::test]
    async fn depth_chunks_record_their_own_dtype_in_metadata_sidecar() {
        // Depth frame entries must carry "dtype": "float16" / "float32" per
        // chunk — and a later chunk with a different depth dtype must apply
        // its *own* dtype to its own entries, not the first chunk's.
        if !ffmpeg_available() {
            eprintln!("ffmpeg not on PATH — skipping depth metadata sidecar test.");
            return;
        }

        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let store_arc = Arc::new(store.clone());
        let context = test_context(&tempdir.path().join("recordings"), store_arc.clone());

        let mut state = ActorState::new(identity(1, "trace-depth", "DEPTH_IMAGES"));
        state.send_create(&context);

        let trace_dir = TracePath::new("1", "DEPTH_IMAGES", "trace-depth")
            .directory(context.recordings_root.as_path());
        let spool_dir = tempdir.path().join("spool");
        std::fs::create_dir_all(&spool_dir).unwrap();

        let dtypes = [FrameDtype::DepthF16, FrameDtype::DepthF32];
        for (chunk_index, dtype) in dtypes.into_iter().enumerate() {
            let chunk_index = chunk_index as u32;
            let spool_nut = spool_dir.join(format!("chunk_{chunk_index}.nut"));
            let status = std::process::Command::new("ffmpeg")
                .args([
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                ])
                .arg("testsrc=duration=2:size=16x16:rate=1")
                .args(["-c:v", "rawvideo", "-pix_fmt", "rgb24", "-f", "nut"])
                .arg(&spool_nut)
                .status()
                .expect("synth status");
            assert!(status.success(), "synth NUT failed");

            let byte_count = spool_nut.metadata().unwrap().len();
            let frame_timestamps_s: Vec<f64> =
                (0..2u32).map(|i| (chunk_index * 2 + i) as f64).collect();
            state
                .handle_video(
                    &context,
                    chunk_index,
                    spool_nut,
                    16,
                    16,
                    byte_count,
                    2,
                    frame_timestamps_s,
                    dtype,
                )
                .await;
        }

        state.finalise_trace(&context).await;
        context.trace_writer.flush().await;

        let sidecar: Value = serde_json::from_slice(
            &std::fs::read(trace_dir.join(paths::TRACE_JSON_FILENAME)).unwrap(),
        )
        .unwrap();
        let entries = sidecar.as_array().unwrap();
        assert_eq!(entries.len(), 4, "two chunks of two frames each");
        // First chunk's two entries carry float16; second chunk's carry float32.
        for entry in &entries[0..2] {
            assert_eq!(entry["dtype"], json!("float16"));
        }
        for entry in &entries[2..4] {
            assert_eq!(entry["dtype"], json!("float32"));
        }
        assert_eq!(entries[0]["width"], json!(16));
        assert_eq!(entries[0]["height"], json!(16));
    }

    #[tokio::test]
    async fn frames_over_storage_budget_are_dropped_not_written() {
        // Under storage pressure the actor must refuse frames (counting the
        // drops) and keep writing the ones that fit, rather than blow past the
        // cap or wedge — a silent-data-loss path with no other guard.
        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let store_arc = Arc::new(store.clone());
        // A tight 25-byte cap: the first 20-byte frame fits; later ones don't.
        let policy = StoragePolicy {
            storage_limit_bytes: Some(25),
            min_free_disk_bytes: 0,
            refresh_interval: Duration::from_secs(60),
        };
        let recordings_root = tempdir.path().join("recordings");
        let budget = Arc::new(StorageBudget::new(&recordings_root, policy));
        let (trace_writer, _writer_owner) =
            crate::state::trace_event_database_writer::spawn(store_arc.clone());
        let (json_writer, _json_owner) = crate::pipeline::json_writer::spawn();
        let context = Arc::new(TraceActorContext::new(
            recordings_root,
            budget,
            VideoEncoder::new(),
            trace_writer,
            json_writer,
        ));

        let mut state = ActorState::new(identity(1, "trace-1", "joints"));
        state.send_create(&context);
        for _ in 0..3 {
            state.handle_data(&context, 0, None, vec![0u8; 20]).await;
        }
        state.finalise_trace(&context).await;
        context.trace_writer.flush().await;

        assert_eq!(state.frame_count, 1, "only the first frame fits the budget");
        assert_eq!(
            state.dropped_over_budget, 2,
            "the two over-budget frames are counted as dropped"
        );

        let trace_dir =
            TracePath::new("1", "joints", "trace-1").directory(context.recordings_root.as_path());
        let bytes = std::fs::read(trace_dir.join("trace.json")).unwrap();
        let parsed: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            parsed.as_array().unwrap().len(),
            1,
            "only the accepted frame lands on disk"
        );
    }

    #[tokio::test]
    async fn cancel_discards_writer_and_releases_budget_without_finalising() {
        // A cancel mid-trace must drop the open writer (release its file handle
        // without finalising) and give back the budget it reserved — never
        // marking the trace Written.
        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let store_arc = Arc::new(store.clone());
        let context = test_context(&tempdir.path().join("recordings"), store_arc.clone());

        let mut state = ActorState::new(identity(1, "trace-1", "joints"));
        state.send_create(&context);
        state
            .handle_data(
                &context,
                0,
                None,
                serde_json::to_vec(&json!({"i": 0})).unwrap(),
            )
            .await;
        assert!(state.bytes_on_disk > 0, "the frame was accounted on disk");

        state.handle_cancel(&context).await;
        context.trace_writer.flush().await;

        assert!(
            matches!(state.writer, TraceWriterKind::Pending),
            "cancel discards the open writer"
        );
        assert_eq!(
            state.bytes_on_disk, 0,
            "cancel releases the byte accounting"
        );

        let trace = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_ne!(
            trace.write_status,
            TraceWriteStatus::Written,
            "a cancelled trace is never finalised as Written"
        );
    }

    /// Write a spool NUT chunk with the real producer NUT writer: one 16x16
    /// RGB frame per entry of `frame_pts_us` (chunk-relative microsecond
    /// ticks; the producer re-anchors every chunk's first frame near 0).
    fn write_nut_chunk(path: &std::path::Path, frame_pts_us: &[u64]) {
        use data_daemon_bridge::nut_writer::{NutVideoConfig, NutWriter};
        let rgb = vec![128u8; 16 * 16 * 3];
        let mut writer = NutWriter::create(
            path,
            NutVideoConfig {
                width: 16,
                height: 16,
                time_base_num: 1,
                time_base_den: 1_000_000,
            },
        )
        .expect("create NUT");
        for pts in frame_pts_us {
            writer.write_frame(*pts, &rgb).expect("write frame");
        }
        writer.finish().expect("finish NUT");
    }

    /// Spool one chunk whose frames sit at the given batch-absolute capture
    /// microseconds, then hand it to the actor as a `Video` message.
    async fn send_video_chunk(
        state: &mut ActorState,
        context: &Arc<TraceActorContext>,
        spool_dir: &std::path::Path,
        chunk_index: u32,
        capture_us: &[i64],
        dtype: FrameDtype,
    ) {
        send_cut_video_chunk(
            state,
            context,
            spool_dir,
            chunk_index,
            capture_us,
            capture_us.len() as u32,
            dtype,
        )
        .await;
    }

    /// As [`send_video_chunk`], for a chunk the dispatcher cut at a recording
    /// boundary: the NUT holds every frame in `capture_us` but the
    /// announcement owns only the leading `owned_frames` of them.
    async fn send_cut_video_chunk(
        state: &mut ActorState,
        context: &Arc<TraceActorContext>,
        spool_dir: &std::path::Path,
        chunk_index: u32,
        capture_us: &[i64],
        owned_frames: u32,
        dtype: FrameDtype,
    ) {
        let chunk_origin = capture_us[0];
        let relative_pts: Vec<u64> = capture_us
            .iter()
            .map(|us| (us - chunk_origin) as u64)
            .collect();
        let frame_timestamps_s: Vec<f64> = capture_us
            .iter()
            .take(owned_frames as usize)
            .map(|us| *us as f64 / 1e6)
            .collect();
        send_video_chunk_with_nut_pts(
            state,
            context,
            spool_dir,
            chunk_index,
            &relative_pts,
            owned_frames,
            frame_timestamps_s,
            dtype,
        )
        .await;
    }

    /// Spool one chunk whose NUT PTS may differ from its announced capture
    /// stamps (the writer's synthesized-PTS class) and send it to the actor.
    /// `frame_count` is what the announcement owns, which is every NUT frame
    /// unless the dispatcher cut the chunk.
    #[allow(clippy::too_many_arguments)]
    async fn send_video_chunk_with_nut_pts(
        state: &mut ActorState,
        context: &Arc<TraceActorContext>,
        spool_dir: &std::path::Path,
        chunk_index: u32,
        nut_pts_us: &[u64],
        frame_count: u32,
        frame_timestamps_s: Vec<f64>,
        dtype: FrameDtype,
    ) {
        let spool_nut = spool_dir.join(format!("spool_chunk_{chunk_index}.nut"));
        write_nut_chunk(&spool_nut, nut_pts_us);
        let byte_count = spool_nut.metadata().unwrap().len();
        state
            .handle_video(
                context,
                chunk_index,
                spool_nut,
                16,
                16,
                byte_count,
                frame_count,
                frame_timestamps_s,
                dtype,
            )
            .await;
    }

    /// Poll `drain_completed_encodes` until every spawned worker is reaped.
    /// Returns whether any worker reported a failure.
    async fn drain_all_pending(state: &mut ActorState, context: &Arc<TraceActorContext>) -> bool {
        let mut any_failure = false;
        for _ in 0..600 {
            any_failure |= state.drain_completed_encodes(context);
            let remaining = match &state.writer {
                TraceWriterKind::Video {
                    pending_encodes, ..
                } => pending_encodes.len(),
                _ => 0,
            };
            if remaining == 0 {
                return any_failure;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("encode workers did not finish in time");
    }

    #[test]
    fn drain_stops_at_dtype_change_and_batch_cap() {
        fn queued(chunk_index: u32, dtype: FrameDtype) -> QueuedChunk {
            QueuedChunk {
                chunk_index,
                spool_nut: PathBuf::from("unused.nut"),
                byte_count: 0,
                frame_count: 1,
                frame_timestamps_s: vec![0.0],
                dtype,
            }
        }
        fn indices(batch: &[QueuedChunk]) -> Vec<u32> {
            batch.iter().map(|chunk| chunk.chunk_index).collect()
        }

        // A dtype change stops the drain so one batch spans one dtype.
        let mut queue: VecDeque<QueuedChunk> = [
            queued(0, FrameDtype::Rgb8),
            queued(1, FrameDtype::Rgb8),
            queued(2, FrameDtype::DepthF16),
            queued(3, FrameDtype::Rgb8),
        ]
        .into_iter()
        .collect();
        assert_eq!(indices(&drain_encode_batch(&mut queue)), vec![0, 1]);
        assert_eq!(indices(&drain_encode_batch(&mut queue)), vec![2]);
        assert_eq!(indices(&drain_encode_batch(&mut queue)), vec![3]);
        assert!(queue.is_empty());

        // The batch cap bounds a same-dtype drain.
        let mut queue: VecDeque<QueuedChunk> = (0..10)
            .map(|index| queued(index, FrameDtype::Rgb8))
            .collect();
        let batch = drain_encode_batch(&mut queue);
        assert_eq!(batch.len(), ENCODE_BATCH_MAX_CHUNKS);
        assert_eq!(indices(&batch), (0..8).collect::<Vec<u32>>());
        assert_eq!(queue.len(), 2);

        // An empty queue yields an empty (no-op) batch.
        assert!(drain_encode_batch(&mut VecDeque::new()).is_empty());
    }

    #[test]
    fn drain_stops_before_a_chunk_past_the_span_cap() {
        fn queued(chunk_index: u32, frame_capture_us: &[i64]) -> QueuedChunk {
            QueuedChunk {
                chunk_index,
                spool_nut: PathBuf::from("unused.nut"),
                byte_count: 0,
                frame_count: frame_capture_us.len() as u32,
                frame_timestamps_s: frame_capture_us.iter().map(|us| *us as f64 / 1e6).collect(),
                dtype: FrameDtype::Rgb8,
            }
        }
        fn indices(batch: &[QueuedChunk]) -> Vec<u32> {
            batch.iter().map(|chunk| chunk.chunk_index).collect()
        }

        // Chunk 1 starts exactly one cap after chunk 0 (span == cap stays in
        // the batch). Chunk 2's declared span from chunk 1 exceeds the cap
        // by 1 us, so it stays at the queue front for its own worker; the
        // drain resumes normally from it.
        let cap_us = ENCODE_BATCH_MAX_SPAN_US;
        let mut queue: VecDeque<QueuedChunk> = [
            queued(0, &[0, 16_683]),
            queued(1, &[cap_us, cap_us + 16_683]),
            queued(2, &[2 * cap_us + 1]),
            queued(3, &[2 * cap_us + 1 + 16_683]),
        ]
        .into_iter()
        .collect();
        assert_eq!(indices(&drain_encode_batch(&mut queue)), vec![0, 1]);
        assert_eq!(indices(&drain_encode_batch(&mut queue)), vec![2, 3]);
        assert!(queue.is_empty());

        // A chunk that announced no frames contributes extent zero: its span
        // floors to 1 us, so it never splits the batch and never panics.
        let mut queue: VecDeque<QueuedChunk> = [
            queued(0, &[0, 16_683]),
            queued(1, &[]),
            queued(2, &[33_366]),
        ]
        .into_iter()
        .collect();
        assert_eq!(indices(&drain_encode_batch(&mut queue)), vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn batched_backlog_produces_one_completed_chunk_keyed_by_first_index() {
        if !ffmpeg_available() {
            eprintln!("ffmpeg not on PATH — skipping batch accounting test.");
            return;
        }

        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let store_arc = Arc::new(store.clone());
        let permits = Arc::new(Semaphore::new(1));
        let context = test_context_with_permits(
            &tempdir.path().join("recordings"),
            store_arc.clone(),
            permits.clone(),
        );
        let spool_dir = tempdir.path().join("spool");
        std::fs::create_dir_all(&spool_dir).unwrap();

        let mut state = ActorState::new(identity(1, "trace-batch", "RGB_IMAGES"));
        state.send_create(&context);

        // Hold the only permit while three chunks arrive so every worker
        // blocks, then release: the first worker drains all three as one
        // batch and the other two must no-op.
        let gate = permits.clone().acquire_owned().await.unwrap();
        let chunks: [Vec<i64>; 3] = [vec![0, 16_683], vec![33_366, 50_049], vec![66_732, 83_415]];
        for (chunk_index, capture_us) in chunks.iter().enumerate() {
            send_video_chunk(
                &mut state,
                &context,
                &spool_dir,
                chunk_index as u32,
                capture_us,
                FrameDtype::Rgb8,
            )
            .await;
        }
        drop(gate);

        let any_failure = drain_all_pending(&mut state, &context).await;
        assert!(!any_failure, "the batch must encode cleanly");

        let trace_dir = TracePath::new("1", "RGB_IMAGES", "trace-batch")
            .directory(context.recordings_root.as_path());
        let chunks_dir = trace_dir.join(paths::CHUNKS_DIRNAME);
        {
            let TraceWriterKind::Video {
                completed_chunks,
                unencoded_chunks,
                ..
            } = &state.writer
            else {
                panic!("video writer expected");
            };
            assert_eq!(completed_chunks.len(), 1, "one CompletedChunk per batch");
            let (first_index, completed) = completed_chunks.iter().next().unwrap();
            assert_eq!(*first_index, 0, "the batch is keyed by its first index");
            assert_eq!(completed.frame_count, 6, "frame_count is the batch sum");
            let expected_timestamps: Vec<f64> =
                chunks.iter().flatten().map(|us| *us as f64 / 1e6).collect();
            assert_eq!(
                completed.frame_timestamps_s, expected_timestamps,
                "timestamps concatenate in chunk order"
            );
            assert_eq!(
                completed.content_extent_us, 83_415,
                "the carried extent stacks the duration lines plus the last chunk's replayed extent"
            );
            assert_eq!(
                completed.lossy_segment,
                trace_dir.join(paths::chunk_lossy_filename(0))
            );
            assert!(completed.lossy_segment.exists());
            assert!(completed.lossless_segment.exists());
            assert_eq!(
                unencoded_chunks.load(Ordering::Relaxed),
                0,
                "no chunk is left un-encoded"
            );
        }
        // The covered indices produce no segments of their own, and every
        // batched NUT is unlinked after the outputs verified non-empty.
        assert!(!trace_dir.join(paths::chunk_lossy_filename(1)).exists());
        assert!(!trace_dir.join(paths::chunk_lossy_filename(2)).exists());
        for chunk_index in 0..3u32 {
            assert!(!chunks_dir.join(paths::chunk_filename(chunk_index)).exists());
        }

        // The finalise concat consumes the batch segment unchanged.
        state.finalise_trace(&context).await;
        context.trace_writer.flush().await;
        assert!(trace_dir.join(paths::LOSSY_VIDEO_FILENAME).exists());
        let sidecar: Value = serde_json::from_slice(
            &std::fs::read(trace_dir.join(paths::TRACE_JSON_FILENAME)).unwrap(),
        )
        .unwrap();
        assert_eq!(sidecar.as_array().unwrap().len(), 6);
        let trace = store.get_trace("trace-batch").await.unwrap().unwrap();
        assert_eq!(trace.write_status, TraceWriteStatus::Written);
    }

    #[tokio::test]
    async fn batch_stops_at_the_frames_the_recording_owns() {
        if !ffmpeg_available() || !ffprobe_available() {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping batch frame-cap test.");
            return;
        }

        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let store_arc = Arc::new(store.clone());
        let permits = Arc::new(Semaphore::new(1));
        let context = test_context_with_permits(
            &tempdir.path().join("recordings"),
            store_arc.clone(),
            permits.clone(),
        );
        let spool_dir = tempdir.path().join("spool");
        std::fs::create_dir_all(&spool_dir).unwrap();

        let mut state = ActorState::new(identity(1, "trace-cut", "RGB_IMAGES"));
        state.send_create(&context);

        // The dispatcher cuts the trace's last chunk at the recording stop, so
        // the batch must encode two frames of chunk 0 and one of chunk 1.
        let gate = permits.clone().acquire_owned().await.unwrap();
        send_video_chunk(
            &mut state,
            &context,
            &spool_dir,
            0,
            &[0, 16_683],
            FrameDtype::Rgb8,
        )
        .await;
        send_cut_video_chunk(
            &mut state,
            &context,
            &spool_dir,
            1,
            &[33_366, 50_049, 66_732],
            1,
            FrameDtype::Rgb8,
        )
        .await;
        drop(gate);

        let any_failure = drain_all_pending(&mut state, &context).await;
        assert!(!any_failure, "the batch must encode cleanly");

        let trace_dir = TracePath::new("1", "RGB_IMAGES", "trace-cut")
            .directory(context.recordings_root.as_path());
        let TraceWriterKind::Video {
            completed_chunks, ..
        } = &state.writer
        else {
            panic!("video writer expected");
        };
        let completed = completed_chunks.values().next().expect("one batch");
        assert_eq!(
            completed.frame_count, 3,
            "the cut chunk contributes only the frames it owns"
        );
        assert_eq!(
            decoded_frame_pts(&completed.lossy_segment).len(),
            3,
            "the lossy segment must not outrun the sidecar"
        );
        assert_eq!(
            decoded_frame_pts(&completed.lossless_segment).len(),
            3,
            "the lossless segment must stop at the same cut"
        );
        assert!(trace_dir.exists());
    }

    #[tokio::test]
    async fn racing_worker_noop_is_reaped_silently_at_finalise() {
        if !ffmpeg_available() {
            eprintln!("ffmpeg not on PATH — skipping worker racing test.");
            return;
        }

        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let store_arc = Arc::new(store.clone());
        let permits = Arc::new(Semaphore::new(1));
        let context = test_context_with_permits(
            &tempdir.path().join("recordings"),
            store_arc.clone(),
            permits.clone(),
        );
        let spool_dir = tempdir.path().join("spool");
        std::fs::create_dir_all(&spool_dir).unwrap();

        let mut state = ActorState::new(identity(1, "trace-race", "RGB_IMAGES"));
        state.send_create(&context);

        // Two workers race for one queued backlog: the first drains both
        // chunks, the second finds the queue empty and must no-op. Finalise
        // reaps both without treating the no-op as an error or a chunk.
        let gate = permits.clone().acquire_owned().await.unwrap();
        send_video_chunk(
            &mut state,
            &context,
            &spool_dir,
            0,
            &[0, 16_683],
            FrameDtype::Rgb8,
        )
        .await;
        send_video_chunk(
            &mut state,
            &context,
            &spool_dir,
            1,
            &[33_366, 50_049],
            FrameDtype::Rgb8,
        )
        .await;
        drop(gate);
        state.finalise_trace(&context).await;
        context.trace_writer.flush().await;

        let trace_dir = TracePath::new("1", "RGB_IMAGES", "trace-race")
            .directory(context.recordings_root.as_path());
        assert!(trace_dir.join(paths::LOSSY_VIDEO_FILENAME).exists());
        // Four sidecar entries: each frame encoded exactly once.
        let sidecar: Value = serde_json::from_slice(
            &std::fs::read(trace_dir.join(paths::TRACE_JSON_FILENAME)).unwrap(),
        )
        .unwrap();
        assert_eq!(sidecar.as_array().unwrap().len(), 4);
        let trace = store.get_trace("trace-race").await.unwrap().unwrap();
        assert_eq!(
            trace.write_status,
            TraceWriteStatus::Written,
            "a no-op worker must not fail the trace"
        );
    }

    #[tokio::test]
    async fn corrupt_nut_in_batch_marks_trace_failed_and_leaves_nuts() {
        if !ffmpeg_available() {
            eprintln!("ffmpeg not on PATH — skipping batch failure test.");
            return;
        }

        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let store_arc = Arc::new(store.clone());
        let permits = Arc::new(Semaphore::new(1));
        let context = test_context_with_permits(
            &tempdir.path().join("recordings"),
            store_arc.clone(),
            permits.clone(),
        );
        let spool_dir = tempdir.path().join("spool");
        std::fs::create_dir_all(&spool_dir).unwrap();

        let mut state = ActorState::new(identity(1, "trace-bad", "RGB_IMAGES"));
        state.send_create(&context);

        let gate = permits.clone().acquire_owned().await.unwrap();
        send_video_chunk(
            &mut state,
            &context,
            &spool_dir,
            0,
            &[0, 16_683],
            FrameDtype::Rgb8,
        )
        .await;
        // A corrupt second chunk poisons the whole batch invocation.
        let corrupt_nut = spool_dir.join("spool_chunk_1.nut");
        std::fs::write(&corrupt_nut, b"not a nut container").unwrap();
        state
            .handle_video(
                &context,
                1,
                corrupt_nut,
                16,
                16,
                19,
                2,
                vec![0.1, 0.116683],
                FrameDtype::Rgb8,
            )
            .await;
        drop(gate);

        let any_failure = drain_all_pending(&mut state, &context).await;
        assert!(any_failure, "the corrupt batch must surface as a failure");
        context.trace_writer.flush().await;

        let trace = store.get_trace("trace-bad").await.unwrap().unwrap();
        assert_eq!(trace.write_status, TraceWriteStatus::Failed);

        // Both relinked NUTs stay on disk for the recovery sweep.
        let trace_dir = TracePath::new("1", "RGB_IMAGES", "trace-bad")
            .directory(context.recordings_root.as_path());
        let chunks_dir = trace_dir.join(paths::CHUNKS_DIRNAME);
        assert!(chunks_dir.join(paths::chunk_filename(0)).exists());
        assert!(chunks_dir.join(paths::chunk_filename(1)).exists());
    }

    #[tokio::test]
    async fn single_chunk_batch_matches_todays_completed_chunk() {
        if !ffmpeg_available() {
            eprintln!("ffmpeg not on PATH — skipping single-chunk batch test.");
            return;
        }

        for lossy_only in [false, true] {
            let tempdir = TempDir::new().unwrap();
            let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
                .await
                .expect("open store");
            let store_arc = Arc::new(store.clone());
            let mut context = test_context(&tempdir.path().join("recordings"), store_arc.clone());
            let _config_tx;
            if lossy_only {
                let (config_tx, config_rx) = watch::channel(DaemonConfig {
                    video_codec: Some("h264_medium".to_string()),
                    ..DaemonConfig::default()
                });
                _config_tx = config_tx;
                context = Arc::new((*context).clone().with_config_rx(config_rx));
            }
            let spool_dir = tempdir.path().join("spool");
            std::fs::create_dir_all(&spool_dir).unwrap();

            let mut state = ActorState::new(identity(1, "trace-single", "RGB_IMAGES"));
            state.send_create(&context);
            let capture_us = [0i64, 16_683];
            send_video_chunk(
                &mut state,
                &context,
                &spool_dir,
                0,
                &capture_us,
                FrameDtype::Rgb8,
            )
            .await;

            let any_failure = drain_all_pending(&mut state, &context).await;
            assert!(!any_failure, "the single-chunk batch must encode cleanly");

            let trace_dir = TracePath::new("1", "RGB_IMAGES", "trace-single")
                .directory(context.recordings_root.as_path());
            {
                let TraceWriterKind::Video {
                    completed_chunks, ..
                } = &state.writer
                else {
                    panic!("video writer expected");
                };
                assert_eq!(completed_chunks.len(), 1);
                let completed = &completed_chunks[&0];
                assert_eq!(completed.frame_count, 2);
                assert_eq!(completed.frame_timestamps_s, vec![0.0, 0.016683]);
                assert_eq!(
                    completed.content_extent_us, 16_683,
                    "a batch of one carries its chunk's replayed extent"
                );
                assert_eq!(
                    completed.lossy_segment,
                    trace_dir.join(paths::chunk_lossy_filename(0)),
                    "segment names keep today's shape (lossy_only={lossy_only})"
                );
                assert!(completed.lossy_segment.exists());
                assert_eq!(
                    completed.lossless_segment.exists(),
                    !lossy_only,
                    "a lossless segment exists exactly when not lossy-only"
                );
            }

            state.finalise_trace(&context).await;
            context.trace_writer.flush().await;
            let trace = store.get_trace("trace-single").await.unwrap().unwrap();
            assert_eq!(trace.write_status, TraceWriteStatus::Written);
        }
    }

    #[tokio::test]
    async fn finalise_after_synthesized_batch_never_rewinds_into_its_content() {
        if !ffmpeg_available() || !ffprobe_available() {
            eprintln!("ffmpeg/ffprobe not on PATH — skipping synthesized-batch finalise gate.");
            return;
        }

        // The undershoot counterexample. c1's first stamp lands 1 ms past
        // c0's last frame but its second regresses, so the writer synthesized
        // c1's NUT ladder from the 16000 us gap carried out of c0 and the
        // batch places frames at [0, 16000, 17000, 33000]. A span replayed
        // from the announced stamps reads the 1000 us boundary delta as the
        // healthy gap, covers only 18000 us and starts the next segment
        // inside the real content, which decodes backwards on preset medium.
        for lossy_only in [false, true] {
            let tempdir = TempDir::new().unwrap();
            let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
                .await
                .expect("open store");
            let store_arc = Arc::new(store.clone());
            let permits = Arc::new(Semaphore::new(1));
            let mut context = test_context_with_permits(
                &tempdir.path().join("recordings"),
                store_arc.clone(),
                permits.clone(),
            );
            let _config_tx;
            if lossy_only {
                let (config_tx, config_rx) = watch::channel(DaemonConfig {
                    video_codec: Some("h264_medium".to_string()),
                    ..DaemonConfig::default()
                });
                _config_tx = config_tx;
                context = Arc::new((*context).clone().with_config_rx(config_rx));
            }
            let spool_dir = tempdir.path().join("spool");
            std::fs::create_dir_all(&spool_dir).unwrap();

            let mut state = ActorState::new(identity(1, "trace-synth", "RGB_IMAGES"));
            state.send_create(&context);

            // Hold the only permit while c0 and c1 arrive so one worker takes
            // both as a single batch.
            let gate = permits.clone().acquire_owned().await.unwrap();
            send_video_chunk_with_nut_pts(
                &mut state,
                &context,
                &spool_dir,
                0,
                &[0, 16_000],
                2,
                vec![0.0, 0.016],
                FrameDtype::Rgb8,
            )
            .await;
            send_video_chunk_with_nut_pts(
                &mut state,
                &context,
                &spool_dir,
                1,
                &[0, 16_000],
                2,
                vec![0.017, 0.0165],
                FrameDtype::Rgb8,
            )
            .await;
            drop(gate);
            assert!(
                !drain_all_pending(&mut state, &context).await,
                "the batch must encode cleanly (lossy_only={lossy_only})"
            );

            // Arrives after the batch completed, so it is its own segment.
            send_video_chunk_with_nut_pts(
                &mut state,
                &context,
                &spool_dir,
                2,
                &[0, 16_000, 32_000],
                3,
                vec![0.019, 0.035, 0.051],
                FrameDtype::Rgb8,
            )
            .await;
            assert!(
                !drain_all_pending(&mut state, &context).await,
                "the following segment must encode cleanly (lossy_only={lossy_only})"
            );

            {
                let TraceWriterKind::Video {
                    completed_chunks, ..
                } = &state.writer
                else {
                    panic!("video writer expected");
                };
                assert_eq!(completed_chunks.len(), 2);
                // The carried extent stacks the 17000 us duration line with
                // the last chunk's replayed extent, which seeds the unknown
                // carried gap with the 100 ms ceiling: it may overshoot the
                // real 33000 us placement but never undershoot it.
                assert_eq!(completed_chunks[&0].content_extent_us, 117_000);
                assert_eq!(completed_chunks[&2].content_extent_us, 32_000);
            }

            state.finalise_trace(&context).await;
            context.trace_writer.flush().await;
            let trace = store.get_trace("trace-synth").await.unwrap().unwrap();
            assert_eq!(trace.write_status, TraceWriteStatus::Written);

            let trace_dir = TracePath::new("1", "RGB_IMAGES", "trace-synth")
                .directory(context.recordings_root.as_path());
            let mut outputs = vec![trace_dir.join(paths::LOSSY_VIDEO_FILENAME)];
            if !lossy_only {
                outputs.push(trace_dir.join(paths::LOSSLESS_VIDEO_FILENAME));
            }
            for video in &outputs {
                let pts_values = decoded_frame_pts(video);
                assert_eq!(
                    pts_values.len(),
                    7,
                    "{} must keep every frame (lossy_only={lossy_only})",
                    video.display()
                );
                assert!(
                    pts_values.windows(2).all(|pair| pair[1] > pair[0]),
                    "{} must decode strictly monotonic PTS, got {pts_values:?}",
                    video.display()
                );
                assert_eq!(
                    &pts_values[..4],
                    &[0, 16_000, 17_000, 33_000],
                    "{} must keep the batch segment's real ladder",
                    video.display()
                );
                assert!(
                    pts_values[4] > 33_000,
                    "{} must start the next segment after the batch content, got {pts_values:?}",
                    video.display()
                );
                assert_eq!(
                    (pts_values[5] - pts_values[4], pts_values[6] - pts_values[5]),
                    (16_000, 16_000),
                    "{} must keep the next segment's capture deltas, got {pts_values:?}",
                    video.display()
                );
            }
        }
    }
}
