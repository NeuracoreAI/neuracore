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
//! daemon-relinked NUT chunks for ffmpeg-side transcoding into per-chunk MP4
//! segments, then on finalise stitch the segments into the final `lossy.mp4` /
//! `lossless.mp4` and flush the [`VideoMetadataAccumulator`] sidecar.
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

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::{mpsc, watch, Semaphore};
use tokio::task::{self, JoinSet};

use crate::cloud::ConfigRx;
use crate::config::DaemonConfig;
use crate::encoding::json_trace::JsonTraceError;
use crate::encoding::metadata::{MetadataError, VideoMetadataAccumulator};
use crate::encoding::video_encoder::{
    ChunkEncodeRequest, LossyVideoCodec, VideoEncodeError, VideoEncoder, ENCODE_THREADS_PER_OUTPUT,
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

/// Cap on concurrent ffmpeg transcodes.
///
/// Each per-chunk encode bounds its libx264 thread pool to
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
    /// Encoder used to transcode per-chunk NUT files into MP4 segments and
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
    /// Video trace whose chunk encodes run concurrently as background tasks.
    Video {
        /// Frame width in pixels (recorded from the first chunk message).
        width: u32,
        /// Frame height in pixels.
        height: u32,
        /// Lossy codec for this trace, resolved once at the first chunk and
        /// applied uniformly to every chunk encode and the finalise concat.
        codec: LossyVideoCodec,
        /// Encodes completed so far, keyed by `chunk_index` so the finalise
        /// concat can iterate in order regardless of completion order.
        completed_chunks: BTreeMap<u32, CompletedChunk>,
        /// Spawned chunk-encode tasks still running.
        pending_encodes: JoinSet<ChunkEncodeJobResult>,
    },
}

/// One successfully encoded chunk, ready to feed into the finalise concat.
struct CompletedChunk {
    /// `chunk_NNNN_lossy.mp4` segment path.
    lossy_segment: PathBuf,
    /// `chunk_NNNN_lossless.mp4` segment path.
    lossless_segment: PathBuf,
    /// Sum of both segments' on-disk byte counts.
    bytes: u64,
    /// Per-frame `timestamp_s` values from the chunk message, applied to the
    /// metadata accumulator at finalise in chunk-index order.
    frame_timestamps_s: Vec<f64>,
    /// Frame count carried by the chunk message.
    frame_count: u32,
}

/// Outcome of one background chunk-encode task.
struct ChunkEncodeJobResult {
    chunk_index: u32,
    /// `Ok(CompletedChunk)` on success; `Err` is logged and the trace marked
    /// failed by the polling path.
    outcome: Result<CompletedChunk, VideoEncodeError>,
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

    /// Handle one finished NUT chunk: transcode it to per-chunk MP4 segments,
    /// append the segment paths to the pending list for the finalise concat,
    /// and unlink the source NUT.
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
    ) {
        let trace_dir = self.trace_directory(context);
        let chunks_dir = trace_dir.join(paths::CHUNKS_DIRNAME);
        let raw_nut = chunks_dir.join(paths::chunk_filename(chunk_index));
        let lossy_segment = trace_dir.join(paths::chunk_lossy_filename(chunk_index));
        let lossless_segment = trace_dir.join(paths::chunk_lossless_filename(chunk_index));

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
            ..
        } = &mut self.writer
        else {
            // Should be unreachable — we just allocated the writer above.
            return;
        };
        let codec = *codec;

        // Spawn the encode as a background task. The actor returns to the
        // inbox immediately so a slow ffmpeg invocation cannot back-pressure
        // unrelated joint / scalar publishers sharing the commands service.
        let permits = context.ffmpeg_permits.clone();
        let permit_total = context.ffmpeg_permit_total;
        let encode_cores = context.encode_cores;
        let encoder = context.video_encoder.clone();
        let trace_id = self.identity.trace_id.clone();
        let chunks_dir_for_task = chunks_dir.clone();
        let request = ChunkEncodeRequest {
            raw_nut: raw_nut.clone(),
            lossy_out: lossy_segment.clone(),
            lossless_out: lossless_segment.clone(),
            codec,
        };
        pending_encodes.spawn(async move {
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
            let permit = match permits.clone().acquire_owned().await {
                Ok(permit) => permit,
                Err(_) => {
                    return ChunkEncodeJobResult {
                        chunk_index,
                        outcome: Err(VideoEncodeError::Spawn {
                            binary: std::ffi::OsString::from("ffmpeg"),
                            source: std::io::Error::other("ffmpeg permit pool closed"),
                        }),
                    };
                }
            };
            // Relink the producer-spooled NUT into the recording's chunks dir
            // here rather than on the dispatcher's routing path. The `rename`
            // (and `mkdir`) are filesystem metadata ops that can stall behind an
            // ext4 journal commit on the shared spool, so we run them on a
            // blocking thread — off both the dispatcher and the runtime workers.
            let relink = {
                let spool = spool_nut.clone();
                let dest = request.raw_nut.clone();
                let chunks = chunks_dir_for_task.clone();
                tokio::task::spawn_blocking(move || relink_nut(&spool, &chunks, &dest)).await
            };
            match relink {
                Ok(Ok(())) => {}
                Ok(Err(source)) => {
                    return ChunkEncodeJobResult {
                        chunk_index,
                        outcome: Err(VideoEncodeError::Io {
                            path: spool_nut.clone(),
                            source,
                        }),
                    };
                }
                Err(join_error) => {
                    return ChunkEncodeJobResult {
                        chunk_index,
                        outcome: Err(VideoEncodeError::Spawn {
                            binary: std::ffi::OsString::from("relink"),
                            source: std::io::Error::other(format!(
                                "relink task join failed: {join_error}"
                            )),
                        }),
                    };
                }
            }
            // Size this encode's thread pool to the cores the rest of the fleet
            // isn't using: with the permit held, `available_permits()` is the
            // idle capacity, so `total - available` is how many encodes (incl.
            // this one) are running now. Read after acquiring so we count
            // ourselves.
            let active_encodes = permit_total.saturating_sub(permits.available_permits());
            let encode_threads = adaptive_encode_threads(encode_cores, active_encodes);
            let outcome = encoder.encode_chunk(&request, encode_threads).await;
            drop(permit);
            match outcome {
                Ok(encode) => {
                    // Drop the source NUT chunk now that both segments are
                    // sealed. Failure to unlink leaves the file for the
                    // recovery sweep to collect.
                    if let Err(error) = std::fs::remove_file(&request.raw_nut) {
                        if error.kind() != std::io::ErrorKind::NotFound {
                            tracing::warn!(
                                %error,
                                trace_id = %trace_id,
                                chunk_index,
                                path = %request.raw_nut.display(),
                                "failed to remove source NUT chunk after encode"
                            );
                        }
                    }
                    let segment_bytes = encode.lossy_bytes.saturating_add(encode.lossless_bytes);
                    tracing::debug!(
                        trace_id = %trace_id,
                        chunk_index,
                        frame_count,
                        byte_count,
                        lossy_bytes = encode.lossy_bytes,
                        lossless_bytes = encode.lossless_bytes,
                        "video chunk encoded"
                    );
                    ChunkEncodeJobResult {
                        chunk_index,
                        outcome: Ok(CompletedChunk {
                            lossy_segment: request.lossy_out,
                            lossless_segment: request.lossless_out,
                            bytes: segment_bytes,
                            frame_timestamps_s,
                            frame_count,
                        }),
                    }
                }
                Err(error) => ChunkEncodeJobResult {
                    chunk_index,
                    outcome: Err(error),
                },
            }
        });

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
                Ok(result) => match result.outcome {
                    Ok(completed) => {
                        new_bytes = new_bytes.saturating_add(completed.bytes);
                        new_frames = new_frames.saturating_add(completed.frame_count as u64);
                        completed_chunks.insert(result.chunk_index, completed);
                    }
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            trace_id = self.identity.trace_id,
                            chunk_index = result.chunk_index,
                            "failed to encode video chunk"
                        );
                        any_failure = true;
                    }
                },
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
                self.mark_failed(context);
            }
        }
    }

    async fn finalise_writer(
        &self,
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
            } => {
                // Drain every still-running encode. A failure here is
                // terminal — without a complete chunk set the concat would
                // produce a video with a missing range, which is worse than
                // marking the trace failed.
                while let Some(joined) = pending_encodes.join_next().await {
                    let result = match joined {
                        Ok(result) => result,
                        Err(join_error) => {
                            return Err(FrameAppendError::VideoEncode(VideoEncodeError::Spawn {
                                binary: std::ffi::OsString::from("ffmpeg"),
                                source: std::io::Error::other(format!(
                                    "video encode task join failed: {join_error}"
                                )),
                            }))
                        }
                    };
                    let completed = result.outcome?;
                    completed_chunks.insert(result.chunk_index, completed);
                }

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

                // Build the metadata accumulator in the same chunk-index
                // order so per-frame entries appear in capture order.
                let mut metadata = VideoMetadataAccumulator::new();
                for chunk in completed_chunks.values() {
                    for timestamp_s in &chunk.frame_timestamps_s {
                        let mut entry = serde_json::Map::new();
                        entry.insert("timestamp".to_string(), Value::from(*timestamp_s));
                        entry.insert("width".to_string(), Value::from(width as u64));
                        entry.insert("height".to_string(), Value::from(height as u64));
                        metadata.record_frame(entry);
                    }
                }

                // Concat is stream-copy: cheap relative to encode but still
                // bounded by an ffmpeg permit so a tail-stitch storm
                // doesn't fork-bomb the host.
                let permit = context
                    .ffmpeg_permits
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| FrameAppendError::FfmpegPermits)?;
                let lossy_outcome = context
                    .video_encoder
                    .concat_segments(&lossy_segments, &lossy_out)
                    .await?;
                let lossless_bytes = if lossy_only {
                    0
                } else {
                    context
                        .video_encoder
                        .concat_segments(&lossless_segments, &lossless_out)
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

                tracing::debug!(
                    trace_id = self.identity.trace_id,
                    chunks_encoded = completed_chunks.len(),
                    "video trace concatenated"
                );

                Ok(lossy_outcome
                    .bytes
                    .saturating_add(lossless_bytes)
                    .saturating_add(metadata_bytes))
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
        let policy = StoragePolicy {
            storage_limit_bytes: None,
            min_free_disk_bytes: 0,
            refresh_interval: Duration::from_secs(60),
        };
        let budget = Arc::new(StorageBudget::new(root, policy));
        let (trace_writer, _writer_owner) = crate::state::trace_event_database_writer::spawn(store);
        let (json_writer, _json_owner) = crate::pipeline::json_writer::spawn();
        Arc::new(TraceActorContext::new(
            root.to_path_buf(),
            budget,
            VideoEncoder::new(),
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
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
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
}
