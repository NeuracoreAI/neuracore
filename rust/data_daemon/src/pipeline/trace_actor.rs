//! Per-trace actor task.
//!
//! Owns the SQLite lifecycle and the on-disk encoders for one trace: scalar /
//! sensor traces stream into a [`JsonTraceWriter`]; video traces consume
//! [`Envelope::VideoChunkReady`] notifications that hand off producer-spooled
//! NUT chunks for ffmpeg-side transcoding into per-chunk MP4 segments, then
//! on `EndTrace` stitch the segments into the final `lossy.mp4` /
//! `lossless.mp4` and flush the [`VideoMetadataAccumulator`] sidecar.
//!
//! Database writes are debounced — a per-frame `bytes_written` UPDATE is
//! wasteful at 200+ Hz scalar ingestion, so a counter is flushed every
//! [`BYTES_WRITTEN_DEBOUNCE_FRAMES`] frames during ingestion and always on
//! finalise. The encoding tail of the state machine — `initializing → writing
//! → pending_metadata → written` — is driven from this actor; the
//! registration coordinator keys off the terminal `written` state.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use data_daemon_ipc::Envelope;
use serde_json::Value;
use tokio::sync::{mpsc, Semaphore};
use tokio::task::{self, JoinSet};

use crate::encoding::json_trace::{JsonTraceError, JsonTraceWriter};
use crate::encoding::metadata::{MetadataError, VideoMetadataAccumulator};
use crate::encoding::video_encoder::{ChunkEncodeRequest, VideoEncodeError, VideoEncoder};
use crate::state::store::TraceUpdate;
use crate::state::{SqliteStateStore, StateStore, TraceWriteStatus};
use crate::storage::budget::StorageBudget;
use crate::storage::paths::{self, TracePath};

/// Key identifying one per-trace actor.
///
/// Frame, VideoChunkReady, and EndTrace envelopes only carry `trace_id` on
/// the wire, so the dispatcher routes by `trace_id` alone. The actor learns
/// its `recording_id` and `data_type` from the first `StartTrace` it sees.
pub type TraceKey = String;

/// Flush `bytes_written` to the DB every N frames instead of every frame.
///
/// At 30 fps video and 200 Hz scalars this keeps the SQLite write rate well
/// under 10 Hz per trace, which the WAL handles comfortably while still giving
/// the upload coordinator a recent enough byte count for its progress reports.
/// A finalise always issues a fresh UPDATE so the terminal row is exact.
const BYTES_WRITTEN_DEBOUNCE_FRAMES: u64 = 32;

/// Cap on concurrent ffmpeg transcodes. Each ffmpeg child happily saturates a
/// CPU core; without a cap N simultaneous per-chunk encode invocations starve
/// the rest of the daemon. Set to 8 so a single-camera 8-context workload can
/// transcode chunks in parallel; raise further if multi-camera setups
/// exhibit the same queueing bottleneck.
pub(crate) const DEFAULT_FFMPEG_CONCURRENCY: usize = 8;

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
    /// `EndTrace`. Cloning a [`VideoEncoder`] is cheap (it carries only the
    /// configured ffmpeg binary path).
    pub video_encoder: VideoEncoder,
    /// Bounds concurrent ffmpeg children. Shared across actors so the
    /// integration matrix's parallel encode storms don't fork-bomb the
    /// transcoder.
    pub ffmpeg_permits: Arc<Semaphore>,
    /// Optional daemon event bus. When present, the trace actor publishes a
    /// [`crate::state::DaemonEvent::TraceWritten`] on finalise so the
    /// registration coordinator can wake immediately. Optional so unit tests
    /// can exercise the actor without standing up a bus.
    pub event_bus: Option<crate::state::EventBus>,
}

impl TraceActorContext {
    /// Build a context with the default ffmpeg concurrency cap. Suitable for
    /// production wiring; tests that need a deterministic transcode order may
    /// prefer [`TraceActorContext::with_ffmpeg_permits`].
    pub fn new(
        recordings_root: impl Into<std::path::PathBuf>,
        storage_budget: Arc<StorageBudget>,
        video_encoder: VideoEncoder,
    ) -> Self {
        Self::with_ffmpeg_permits(
            recordings_root,
            storage_budget,
            video_encoder,
            Arc::new(Semaphore::new(DEFAULT_FFMPEG_CONCURRENCY)),
        )
    }

    /// Build a context with an externally-provided ffmpeg permit pool.
    pub fn with_ffmpeg_permits(
        recordings_root: impl Into<std::path::PathBuf>,
        storage_budget: Arc<StorageBudget>,
        video_encoder: VideoEncoder,
        ffmpeg_permits: Arc<Semaphore>,
    ) -> Self {
        Self {
            recordings_root: Arc::new(recordings_root.into()),
            storage_budget,
            video_encoder,
            ffmpeg_permits,
            event_bus: None,
        }
    }

    /// Attach a daemon event bus to this context. Returns `self` so it
    /// composes cleanly with [`new`] / [`with_ffmpeg_permits`].
    pub fn with_event_bus(mut self, bus: crate::state::EventBus) -> Self {
        self.event_bus = Some(bus);
        self
    }
}

/// Derive the per-actor routing key from an envelope.
pub fn trace_key(envelope: &Envelope) -> Option<TraceKey> {
    match envelope {
        Envelope::StartTrace { trace_id, .. }
        | Envelope::Frame { trace_id, .. }
        | Envelope::EndTrace { trace_id, .. }
        | Envelope::VideoChunkReady { trace_id, .. } => Some(trace_id.clone()),
        // `BatchedFrames` spans multiple traces and is expanded into
        // per-trace `Frame`s by the IPC listener — it never reaches routing.
        Envelope::BatchedFrames { .. }
        | Envelope::StartRecording { .. }
        | Envelope::StopRecording { .. }
        | Envelope::CancelRecording { .. } => None,
    }
}

/// Message accepted by a per-trace actor.
#[derive(Debug)]
pub enum TraceActorMessage {
    /// A producer-originated envelope routed to this trace.
    Envelope(Envelope),
    /// Drop the in-flight writer and delete the on-disk artefacts. Sent by
    /// the dispatcher when the parent recording is cancelled.
    Cancel,
}

/// Internal state of a per-trace actor.
///
/// Encoders are opened lazily: a scalar trace doesn't need a `trace.json` file
/// until the first frame arrives, and a video trace's segment / metadata
/// state is allocated when the first `VideoChunkReady` lands.
enum TraceWriter {
    /// No frames yet observed; the writer is decided on the first frame or
    /// chunk envelope.
    Pending,
    /// Scalar trace streaming into a single `trace.json` array.
    Json(JsonTraceWriter),
    /// Video trace whose chunk encodes run concurrently as background tasks.
    ///
    /// Each `VideoChunkReady` envelope spawns one encode task; the per-trace
    /// actor returns to the inbox immediately so a slow ffmpeg invocation
    /// can't back-pressure unrelated joint/scalar publishers sharing the
    /// `commands` service. Completed encodes land in `completed_chunks`
    /// keyed by `chunk_index`; `EndTrace` awaits any still-running encode
    /// and concatenates the segments in `chunk_index` order. The
    /// `ffmpeg_permits` semaphore on `TraceActorContext` still caps total
    /// concurrent ffmpeg children across every trace, so per-trace
    /// parallelism is bounded by CPU rather than the actor's serialisation.
    Video {
        /// Frame width in pixels (recorded from the first chunk envelope).
        width: u32,
        /// Frame height in pixels.
        height: u32,
        /// Encodes completed so far, keyed by `chunk_index` so the EndTrace
        /// concat can iterate in order regardless of completion order.
        completed_chunks: BTreeMap<u32, CompletedChunk>,
        /// Spawned chunk-encode tasks still running.
        pending_encodes: JoinSet<ChunkEncodeJobResult>,
    },
}

/// One successfully encoded chunk, ready to feed into the EndTrace concat.
struct CompletedChunk {
    /// `chunk_NNNN_lossy.mp4` segment path.
    lossy_segment: PathBuf,
    /// `chunk_NNNN_lossless.mp4` segment path.
    lossless_segment: PathBuf,
    /// Sum of both segments' on-disk byte counts.
    bytes: u64,
    /// Per-frame `timestamp_s` values from the chunk envelope, applied to
    /// the metadata accumulator at EndTrace in chunk-index order.
    frame_timestamps_s: Vec<f64>,
    /// Frame count carried by the chunk envelope.
    frame_count: u32,
}

/// Outcome of one background chunk-encode task.
struct ChunkEncodeJobResult {
    chunk_index: u32,
    /// `Ok(CompletedChunk)` on success; `Err` is logged and the trace marked
    /// failed by the polling path.
    outcome: Result<CompletedChunk, VideoEncodeError>,
}

/// Whether the actor should keep running after handling an envelope.
enum Flow {
    /// Continue receiving envelopes.
    Continue,
    /// Terminal envelope handled (`EndTrace`) — the actor should return.
    Stop,
}

/// Run the per-trace actor until the dispatcher closes the inbox.
pub async fn run(
    store: Arc<SqliteStateStore>,
    context: Arc<TraceActorContext>,
    trace_id: TraceKey,
    mut inbox: mpsc::Receiver<TraceActorMessage>,
) {
    let mut state = ActorState::new(trace_id.clone());

    while let Some(message) = inbox.recv().await {
        match message {
            TraceActorMessage::Envelope(envelope) => {
                if let Flow::Stop = state.accept_envelope(&store, &context, envelope).await {
                    // After end-of-trace there is nothing more for this
                    // actor to do; returning drops the receiver so the
                    // dispatcher's shutdown sweep is free of dangling
                    // senders.
                    return;
                }
            }
            TraceActorMessage::Cancel => {
                tracing::info!(trace_id = state.trace_id, "cancel received by actor");
                state.handle_cancel(&store, &context).await;
                return;
            }
        }
    }

    // Inbox closed without an explicit `EndTrace` or `Cancel` — typically a
    // shutdown or a stop-recording cancel. Mark the trace as failed so the
    // lifecycle is observable from the DB and the registration coordinator
    // doesn't pick it up. Skip when no StartTrace has been seen — there is
    // no row yet, and writing one in `failed` state would be misleading.
    state.handle_shutdown_without_end(&store).await;
}

/// Per-actor mutable bookkeeping. Pulled out of `run` so the message handlers
/// can be tested with synthetic envelopes against a clean state object.
struct ActorState {
    trace_id: String,
    recording_id: Option<String>,
    data_type: Option<String>,
    writer: TraceWriter,
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
    /// Envelopes received before `StartTrace`. Both scalar `Frame`s and
    /// `VideoChunkReady` envelopes can in principle arrive before
    /// `StartTrace` is drained — although both ride the same `commands`
    /// service today, the per-trace pre-`StartTrace` buffer is kept as a
    /// safety net against any future reordering or producer-side race.
    pending: Vec<Envelope>,
    /// Count of envelopes dropped because [`Self::pending`] hit its cap before
    /// `StartTrace` arrived. Rate-limits the associated warning.
    pending_overflow: u64,
}

impl ActorState {
    fn new(trace_id: String) -> Self {
        Self {
            trace_id,
            recording_id: None,
            data_type: None,
            writer: TraceWriter::Pending,
            frame_count: 0,
            bytes_on_disk: 0,
            last_db_bytes: 0,
            dropped_over_budget: 0,
            pending: Vec::new(),
            pending_overflow: 0,
        }
    }

    /// Route one inbound envelope.
    ///
    /// Until `StartTrace` has been observed every other envelope is buffered
    /// (see [`Self::pending`]); `StartTrace` itself is applied immediately and
    /// then drains the buffer in arrival order. Once the trace is known every
    /// envelope is applied directly.
    async fn accept_envelope(
        &mut self,
        store: &Arc<SqliteStateStore>,
        context: &Arc<TraceActorContext>,
        envelope: Envelope,
    ) -> Flow {
        let is_start_trace = matches!(envelope, Envelope::StartTrace { .. });
        if self.recording_id.is_none() && !is_start_trace {
            self.buffer_pending(envelope);
            return Flow::Continue;
        }
        let flow = self.apply_envelope(store, context, envelope).await;
        if is_start_trace && matches!(flow, Flow::Continue) {
            return self.replay_pending(store, context).await;
        }
        flow
    }

    /// Apply one envelope to the writer/state machine. Assumes ordering
    /// constraints are already satisfied — callers gate pre-`StartTrace`
    /// envelopes via [`Self::accept_envelope`].
    async fn apply_envelope(
        &mut self,
        store: &Arc<SqliteStateStore>,
        context: &Arc<TraceActorContext>,
        envelope: Envelope,
    ) -> Flow {
        match envelope {
            Envelope::StartTrace {
                recording_id,
                data_type,
                data_type_name,
                ..
            } => {
                self.handle_start_trace(store, recording_id, data_type, data_type_name)
                    .await;
                Flow::Continue
            }
            Envelope::Frame {
                timestamp_ns,
                timestamp_s,
                payload,
                ..
            } => {
                self.handle_frame(store, context, timestamp_ns, timestamp_s, payload)
                    .await;
                Flow::Continue
            }
            Envelope::VideoChunkReady {
                chunk_index,
                width,
                height,
                byte_count,
                frame_count,
                frame_timestamps_s,
                ..
            } => {
                self.handle_video_chunk_ready(
                    store,
                    context,
                    chunk_index,
                    width,
                    height,
                    byte_count,
                    frame_count,
                    frame_timestamps_s,
                )
                .await;
                Flow::Continue
            }
            Envelope::EndTrace { trace_id } => {
                tracing::info!(trace_id, "end_trace envelope received by actor");
                self.handle_end_trace(store, context).await;
                Flow::Stop
            }
            Envelope::StartRecording { .. }
            | Envelope::StopRecording { .. }
            | Envelope::CancelRecording { .. }
            | Envelope::BatchedFrames { .. } => {
                // Recording-scoped envelopes never carry a trace_id, so the
                // dispatcher routes them through its own branch — they can't
                // reach a per-trace actor. `BatchedFrames` is likewise never
                // seen here: the IPC listener expands it into per-trace
                // `Frame`s before dispatch. The match arm exists so adding a
                // new non-trace envelope variant is a compile error here
                // rather than a silent ignore.
                unreachable!("non-trace envelopes are filtered by trace_key");
            }
        }
    }

    /// Buffer an envelope that arrived before `StartTrace`.
    ///
    /// Capped so a trace whose `StartTrace` is lost (or pathologically
    /// delayed) cannot grow the buffer without bound; the oldest envelope is
    /// dropped on overflow, which for an in-order frame stream means the
    /// earliest frames — the least costly to lose.
    fn buffer_pending(&mut self, envelope: Envelope) {
        /// Upper bound on buffered pre-`StartTrace` envelopes. Generous
        /// relative to the sub-tick `StartTrace` skew it exists to absorb.
        const MAX_PENDING: usize = 512;
        if self.pending.len() >= MAX_PENDING {
            self.pending.remove(0);
            self.pending_overflow = self.pending_overflow.saturating_add(1);
            if self.pending_overflow == 1 || self.pending_overflow.is_multiple_of(256) {
                tracing::warn!(
                    trace_id = self.trace_id,
                    dropped = self.pending_overflow,
                    "pre-start_trace buffer full; dropping oldest envelope",
                );
            }
        }
        self.pending.push(envelope);
    }

    /// Replay buffered pre-`StartTrace` envelopes in arrival order. Called
    /// immediately after `StartTrace` is applied.
    async fn replay_pending(
        &mut self,
        store: &Arc<SqliteStateStore>,
        context: &Arc<TraceActorContext>,
    ) -> Flow {
        let pending = std::mem::take(&mut self.pending);
        if !pending.is_empty() {
            tracing::debug!(
                trace_id = self.trace_id,
                buffered = pending.len(),
                "replaying envelopes buffered before start_trace",
            );
        }
        for envelope in pending {
            if let Flow::Stop = self.apply_envelope(store, context, envelope).await {
                return Flow::Stop;
            }
        }
        Flow::Continue
    }

    async fn handle_start_trace(
        &mut self,
        store: &Arc<SqliteStateStore>,
        recording_id: String,
        data_type: String,
        data_type_name: Option<String>,
    ) {
        self.recording_id = Some(recording_id.clone());
        self.data_type = Some(data_type.clone());
        match store
            .create_trace(
                &recording_id,
                &self.trace_id,
                Some(&data_type),
                data_type_name.as_deref(),
            )
            .await
        {
            Ok(_) => tracing::debug!(
                trace_id = self.trace_id,
                recording_id,
                data_type,
                data_type_name = data_type_name.as_deref(),
                "trace initialised"
            ),
            Err(error) => tracing::warn!(
                %error,
                trace_id = self.trace_id,
                recording_id,
                "failed to create trace row"
            ),
        }
    }

    async fn handle_frame(
        &mut self,
        store: &Arc<SqliteStateStore>,
        context: &Arc<TraceActorContext>,
        timestamp_ns: i64,
        _timestamp_s: Option<f64>,
        payload: Vec<u8>,
    ) {
        // Trace metadata must exist by the time the first frame arrives —
        // `StartTrace` is the first envelope routed by the dispatcher.
        let (Some(recording_id), Some(data_type)) =
            (self.recording_id.clone(), self.data_type.clone())
        else {
            tracing::warn!(
                trace_id = self.trace_id,
                "dropping frame received before start_trace"
            );
            return;
        };

        if !self.budget_allows_frame(&context.storage_budget, payload.len()) {
            return;
        }

        if !self.ensure_writer_open(context, &recording_id, &data_type) {
            return;
        }

        // Try to mark `writing` exactly once. Subsequent frames don't need an
        // UPDATE for this field; the bytes-written debouncer covers the rest.
        let bumped_status = self.frame_count == 0;

        if let Err(error) = self.append_frame(timestamp_ns, &payload).await {
            tracing::warn!(
                %error,
                trace_id = self.trace_id,
                "failed to append frame; marking trace failed"
            );
            self.mark_failed(store).await;
            return;
        }

        self.frame_count = self.frame_count.saturating_add(1);

        let bytes_changed = self.bytes_on_disk as i64 != self.last_db_bytes;
        let debounce_due = self
            .frame_count
            .is_multiple_of(BYTES_WRITTEN_DEBOUNCE_FRAMES);
        if bumped_status || (debounce_due && bytes_changed) {
            let update = TraceUpdate {
                write_status: bumped_status.then_some(TraceWriteStatus::Writing),
                bytes_written: bytes_changed.then_some(self.bytes_on_disk as i64),
                ..TraceUpdate::default()
            };
            if let Err(error) = store.update_trace(&self.trace_id, update).await {
                tracing::warn!(
                    %error,
                    trace_id = self.trace_id,
                    "failed to update trace progress"
                );
            } else if bytes_changed {
                self.last_db_bytes = self.bytes_on_disk as i64;
            }
        }
    }

    /// Ask the storage budget whether `payload_len` bytes may be written
    /// against the currently open writer. See the original comment block in
    /// the prior implementation for the fail-open rationale.
    fn budget_allows_frame(&mut self, budget: &Arc<StorageBudget>, payload_len: usize) -> bool {
        match budget.check(payload_len as u64) {
            Ok(check) if check.is_available() => true,
            Ok(check) => {
                self.dropped_over_budget = self.dropped_over_budget.saturating_add(1);
                if self.dropped_over_budget == 1 || self.dropped_over_budget.is_multiple_of(256) {
                    tracing::warn!(
                        trace_id = self.trace_id,
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
                    trace_id = self.trace_id,
                    "storage budget query failed; allowing frame through"
                );
                true
            }
        }
    }

    /// Lazily open the JSON writer for scalar traces. Video traces no
    /// longer open a writer on the frame path — they wait for the first
    /// `VideoChunkReady` to allocate the video writer.
    fn ensure_writer_open(
        &mut self,
        context: &Arc<TraceActorContext>,
        recording_id: &str,
        data_type: &str,
    ) -> bool {
        if !matches!(self.writer, TraceWriter::Pending) {
            return true;
        }

        let trace_dir = self.trace_directory(recording_id, data_type, context);
        match JsonTraceWriter::open(&trace_dir) {
            Ok(json_writer) => {
                self.bytes_on_disk = json_writer.bytes_on_disk();
                self.writer = TraceWriter::Json(json_writer);
                true
            }
            Err(error) => {
                tracing::warn!(
                    %error,
                    trace_id = self.trace_id,
                    path = %trace_dir.display(),
                    "failed to open JSON trace"
                );
                false
            }
        }
    }

    async fn append_frame(
        &mut self,
        timestamp_ns: i64,
        payload: &[u8],
    ) -> Result<(), FrameAppendError> {
        match &mut self.writer {
            TraceWriter::Pending => Err(FrameAppendError::WriterNotOpen),
            TraceWriter::Json(writer) => {
                // Prefer writing the producer's JSON payload verbatim so
                // float-precision is preserved bit-for-bit — re-serialising
                // through serde_json can flip the last decimal digit for
                // values like `7/60` and break the integration matrix's
                // exact-match timestamp assertion. A single parse decides
                // both branches: a successful parse means the payload is
                // already valid JSON and goes through verbatim; anything
                // else is wrapped in a small fallback object.
                match serde_json::from_slice::<serde::de::IgnoredAny>(payload) {
                    Ok(_) => writer.add_raw_entry(payload)?,
                    Err(_) => {
                        let entry = scalar_fallback_entry(timestamp_ns, payload);
                        writer.add_entry(&entry)?;
                    }
                }
                self.bytes_on_disk = writer.bytes_on_disk();
                Ok(())
            }
            TraceWriter::Video { .. } => {
                // Video traces no longer receive standalone `Frame`
                // envelopes — pixel data flows via `VideoChunkReady`. A
                // stray frame for a video trace is a producer bug; log it
                // and ignore.
                tracing::warn!(
                    trace_id = self.trace_id,
                    "video trace received standalone Frame; ignoring"
                );
                Ok(())
            }
        }
    }

    /// Handle one finished NUT chunk: transcode it to per-chunk MP4 segments,
    /// append the segment paths to the pending list for the EndTrace concat,
    /// and unlink the source NUT.
    #[allow(clippy::too_many_arguments)]
    async fn handle_video_chunk_ready(
        &mut self,
        store: &Arc<SqliteStateStore>,
        context: &Arc<TraceActorContext>,
        chunk_index: u32,
        width: u32,
        height: u32,
        byte_count: u64,
        frame_count: u32,
        frame_timestamps_s: Vec<f64>,
    ) {
        let (Some(recording_id), Some(data_type)) =
            (self.recording_id.clone(), self.data_type.clone())
        else {
            tracing::warn!(
                trace_id = self.trace_id,
                chunk_index,
                "dropping video chunk received before start_trace"
            );
            return;
        };

        let trace_dir = self.trace_directory(&recording_id, &data_type, context);
        let chunks_dir = trace_dir.join(paths::CHUNKS_DIRNAME);
        let raw_nut = chunks_dir.join(paths::chunk_filename(chunk_index));
        let lossy_segment = trace_dir.join(paths::chunk_lossy_filename(chunk_index));
        let lossless_segment = trace_dir.join(paths::chunk_lossless_filename(chunk_index));

        // Allocate the video writer on the first chunk and mark the trace
        // `writing` so the registration coordinator can observe lifecycle
        // progress. The mark happens once per trace.
        let bumped_status = matches!(self.writer, TraceWriter::Pending);
        if bumped_status {
            self.writer = TraceWriter::Video {
                width,
                height,
                completed_chunks: BTreeMap::new(),
                pending_encodes: JoinSet::new(),
            };
        }

        // Drain any background encodes that finished while we were idle.
        // Doing this here (rather than only at EndTrace) keeps
        // `bytes_on_disk` / `frame_count` fresh enough for the debounced DB
        // progress UPDATE.
        if self.drain_completed_encodes(store).await {
            // A previous chunk's encode failed; mark_failed already ran, no
            // point spawning more work.
            return;
        }

        // Sanity-warn on resolution drift — the on-disk sidecar uses the
        // first-chunk values, so a producer bug shipping a different
        // resolution mid-trace would lose pixels silently.
        if let TraceWriter::Video {
            width: stored_width,
            height: stored_height,
            ..
        } = &self.writer
        {
            if (*stored_width, *stored_height) != (width, height) {
                tracing::warn!(
                    trace_id = self.trace_id,
                    chunk_index,
                    stored = ?(*stored_width, *stored_height),
                    arrived = ?(width, height),
                    "video chunk resolution disagrees with first-chunk resolution"
                );
            }
        }

        let TraceWriter::Video {
            pending_encodes, ..
        } = &mut self.writer
        else {
            // Should be unreachable — we just allocated the writer above.
            return;
        };

        // Spawn the encode as a background task. The actor returns to the
        // inbox immediately so a slow ffmpeg invocation cannot back-pressure
        // unrelated joint / scalar publishers sharing the commands service.
        // `ffmpeg_permits` still caps total concurrent ffmpeg children
        // across every trace.
        let permits = context.ffmpeg_permits.clone();
        let encoder = context.video_encoder.clone();
        let trace_id = self.trace_id.clone();
        let request = ChunkEncodeRequest {
            raw_nut: raw_nut.clone(),
            lossy_out: lossy_segment.clone(),
            lossless_out: lossless_segment.clone(),
        };
        pending_encodes.spawn(async move {
            // Acquire a permit, then encode. The permit lives only inside
            // the task — dropping it releases the slot, even on panic.
            let permit = match permits.acquire_owned().await {
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
            let outcome = encoder.encode_chunk(&request).await;
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
            let update = TraceUpdate {
                write_status: Some(TraceWriteStatus::Writing),
                ..TraceUpdate::default()
            };
            if let Err(error) = store.update_trace(&self.trace_id, update).await {
                tracing::warn!(
                    %error,
                    trace_id = self.trace_id,
                    "failed to mark trace writing"
                );
            }
        }
    }

    /// Drain every background encode that has already finished. On encode
    /// failure marks the trace failed and returns `true`; otherwise returns
    /// `false`. Caller-side use: gate further work on the return value.
    async fn drain_completed_encodes(&mut self, store: &Arc<SqliteStateStore>) -> bool {
        let TraceWriter::Video {
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
                            trace_id = self.trace_id,
                            chunk_index = result.chunk_index,
                            "failed to encode video chunk"
                        );
                        any_failure = true;
                    }
                },
                Err(join_error) => {
                    tracing::warn!(
                        %join_error,
                        trace_id = self.trace_id,
                        "video encode task join failed"
                    );
                    any_failure = true;
                }
            }
        }
        if new_bytes > 0 || new_frames > 0 {
            self.bytes_on_disk = self.bytes_on_disk.saturating_add(new_bytes);
            self.frame_count = self.frame_count.saturating_add(new_frames);
            // Debounce the DB UPDATE the same way the scalar path does — a
            // chunk landing every ~second is rare enough that we can flush
            // every time, but only when there's been an actual change.
            let bytes_changed = self.bytes_on_disk as i64 != self.last_db_bytes;
            if bytes_changed {
                let update = TraceUpdate {
                    bytes_written: Some(self.bytes_on_disk as i64),
                    ..TraceUpdate::default()
                };
                if let Err(error) = store.update_trace(&self.trace_id, update).await {
                    tracing::warn!(
                        %error,
                        trace_id = self.trace_id,
                        "failed to update trace progress"
                    );
                } else {
                    self.last_db_bytes = self.bytes_on_disk as i64;
                }
            }
        }
        if any_failure {
            self.mark_failed(store).await;
        }
        any_failure
    }

    async fn handle_end_trace(
        &mut self,
        store: &Arc<SqliteStateStore>,
        context: &Arc<TraceActorContext>,
    ) {
        let (Some(recording_id), Some(data_type)) =
            (self.recording_id.clone(), self.data_type.clone())
        else {
            tracing::warn!(
                trace_id = self.trace_id,
                "received end_trace before start_trace; dropping"
            );
            return;
        };

        let writer = std::mem::replace(&mut self.writer, TraceWriter::Pending);
        let finalise = self
            .finalise_writer(writer, context, &recording_id, &data_type)
            .await;
        match finalise {
            Ok(total_bytes) => {
                self.bytes_on_disk = total_bytes;
                let update = TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    total_bytes: Some(total_bytes as i64),
                    bytes_written: Some(total_bytes as i64),
                    ..TraceUpdate::default()
                };
                if let Err(error) = store.update_trace(&self.trace_id, update).await {
                    tracing::warn!(
                        %error,
                        trace_id = self.trace_id,
                        "failed to mark trace as written"
                    );
                }
                tracing::info!(
                    trace_id = self.trace_id,
                    recording_id,
                    frame_count = self.frame_count,
                    dropped_over_budget = self.dropped_over_budget,
                    total_bytes,
                    "trace finalised"
                );
                if let Some(bus) = context.event_bus.as_ref() {
                    bus.publish(crate::state::DaemonEvent::TraceWritten {
                        trace_id: self.trace_id.clone(),
                        recording_id: recording_id.clone(),
                    });
                }
            }
            Err(error) => {
                tracing::warn!(
                    %error,
                    trace_id = self.trace_id,
                    "failed to finalise trace artefacts"
                );
                self.mark_failed(store).await;
            }
        }
    }

    async fn finalise_writer(
        &self,
        writer: TraceWriter,
        context: &Arc<TraceActorContext>,
        recording_id: &str,
        data_type: &str,
    ) -> Result<u64, FrameAppendError> {
        match writer {
            TraceWriter::Pending => {
                // Empty trace — no encoder was ever opened. Leave a single
                // empty `trace.json` behind so the artefact set is complete.
                let trace_dir = self.trace_directory(recording_id, data_type, context);
                let json = JsonTraceWriter::open(&trace_dir)?;
                let total = json.finish()?;
                Ok(total)
            }
            TraceWriter::Json(writer) => Ok(writer.finish()?),
            TraceWriter::Video {
                width,
                height,
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
                    let trace_dir = self.trace_directory(recording_id, data_type, context);
                    let json = JsonTraceWriter::open(&trace_dir)?;
                    return Ok(json.finish()?);
                }

                let trace_dir = self.trace_directory(recording_id, data_type, context);
                let lossy_out = trace_dir.join(paths::LOSSY_VIDEO_FILENAME);
                let lossless_out = trace_dir.join(paths::LOSSLESS_VIDEO_FILENAME);

                // BTreeMap iteration is sorted by chunk_index, so the concat
                // segment lists are guaranteed in producer-arrival order
                // regardless of encode completion order.
                let lossy_segments: Vec<PathBuf> = completed_chunks
                    .values()
                    .map(|chunk| chunk.lossy_segment.clone())
                    .collect();
                let lossless_segments: Vec<PathBuf> = completed_chunks
                    .values()
                    .map(|chunk| chunk.lossless_segment.clone())
                    .collect();

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
                let lossless_outcome = context
                    .video_encoder
                    .concat_segments(&lossless_segments, &lossless_out)
                    .await?;
                drop(permit);

                // Unlink per-chunk segments now that the final outputs are
                // sealed. Best-effort: a leftover segment is wasted disk
                // space, not a correctness problem.
                for segment in lossy_segments.iter().chain(lossless_segments.iter()) {
                    if let Err(error) = std::fs::remove_file(segment) {
                        if error.kind() != std::io::ErrorKind::NotFound {
                            tracing::warn!(
                                %error,
                                trace_id = self.trace_id,
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
                    trace_id = self.trace_id,
                    chunks_encoded = completed_chunks.len(),
                    "video trace concatenated"
                );

                Ok(lossy_outcome
                    .bytes
                    .saturating_add(lossless_outcome.bytes)
                    .saturating_add(metadata_bytes))
            }
        }
    }

    async fn handle_shutdown_without_end(&mut self, store: &Arc<SqliteStateStore>) {
        if self.recording_id.is_none() {
            return;
        }
        self.mark_failed(store).await;
    }

    async fn mark_failed(&mut self, store: &Arc<SqliteStateStore>) {
        let update = TraceUpdate {
            write_status: Some(TraceWriteStatus::Failed),
            bytes_written: Some(self.bytes_on_disk as i64),
            ..TraceUpdate::default()
        };
        if let Err(error) = store.update_trace(&self.trace_id, update).await {
            tracing::warn!(
                %error,
                trace_id = self.trace_id,
                "failed to mark trace as failed"
            );
        }
    }

    /// Tear down the writer and delete the on-disk trace directory.
    ///
    /// Called when the parent recording is cancelled. The DB row's
    /// `write_status` is left untouched here — the dispatcher will issue a
    /// single `cancel_recording` transaction once every actor has exited
    /// (the trace's terminal state is `Failed` with
    /// `error_code = recording_cancelled`).
    async fn handle_cancel(
        &mut self,
        _store: &Arc<SqliteStateStore>,
        context: &Arc<TraceActorContext>,
    ) {
        // Drop the writer first so any BufWriter inside releases its file
        // handle before we unlink the directory.
        self.writer = TraceWriter::Pending;

        let (Some(recording_id), Some(data_type)) =
            (self.recording_id.clone(), self.data_type.clone())
        else {
            // No StartTrace observed — nothing on disk to clean up.
            return;
        };
        let trace_dir = self.trace_directory(&recording_id, &data_type, context);
        if let Err(error) = std::fs::remove_dir_all(&trace_dir) {
            if error.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(
                    %error,
                    trace_id = self.trace_id,
                    path = %trace_dir.display(),
                    "failed to remove cancelled trace directory"
                );
            }
        }
        if self.bytes_on_disk > 0 {
            context.storage_budget.release(self.bytes_on_disk);
            self.bytes_on_disk = 0;
            self.last_db_bytes = 0;
        }
    }

    /// Build the on-disk directory for this trace. Callers must have already
    /// confirmed via `StartTrace` that `recording_id` and `data_type` are
    /// known, so the invariant is enforced at the type system (no
    /// `"unknown"` fallback that could silently shunt misrouted traces into
    /// a shared bucket).
    fn trace_directory(
        &self,
        recording_id: &str,
        data_type: &str,
        context: &Arc<TraceActorContext>,
    ) -> std::path::PathBuf {
        TracePath::new(recording_id, data_type, self.trace_id.clone())
            .directory(context.recordings_root.as_path())
    }
}

/// Errors that can surface while appending or finalising a frame. The variants
/// are unified so `handle_frame` / `handle_end_trace` can log + mark-failed in
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

/// Wrap a non-JSON scalar payload in a minimal object so the on-disk
/// `trace.json` array stays parseable. The verbatim path in
/// [`ActorState::append_frame`] only reaches this helper after a structural
/// JSON parse has already failed, so this never re-parses the bytes.
fn scalar_fallback_entry(timestamp_ns: i64, payload: &[u8]) -> Value {
    let mut map = serde_json::Map::new();
    map.insert("timestamp_ns".to_string(), Value::from(timestamp_ns));
    map.insert("payload_len".to_string(), Value::from(payload.len() as u64));
    Value::Object(map)
}

/// Flush the in-memory metadata accumulator to `trace.json` on a blocking
/// thread. The accumulator is small (one dict per video frame) but the write
/// is synchronous, so we hop to a blocking thread to avoid stalling the
/// runtime worker on disks under load.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::budget::StoragePolicy;
    use serde_json::json;
    use std::time::Duration;
    use tempfile::TempDir;

    fn test_context(root: &std::path::Path) -> Arc<TraceActorContext> {
        let policy = StoragePolicy {
            storage_limit_bytes: None,
            min_free_disk_bytes: 0,
            refresh_interval: Duration::from_secs(60),
        };
        let budget = Arc::new(StorageBudget::new(root, policy));
        Arc::new(TraceActorContext::new(
            root.to_path_buf(),
            budget,
            VideoEncoder::new(),
        ))
    }

    #[test]
    fn scalar_fallback_entry_wraps_non_json_payload() {
        let entry = scalar_fallback_entry(123, &[0xFF, 0xFE]);
        assert_eq!(entry, json!({"timestamp_ns": 123, "payload_len": 2}));
    }

    #[tokio::test]
    async fn json_trace_writes_array_on_finalise() {
        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let context = test_context(&tempdir.path().join("recordings"));
        let store_arc = Arc::new(store.clone());

        let mut state = ActorState::new("trace-1".to_string());
        state
            .handle_start_trace(&store_arc, "rec-1".to_string(), "joints".to_string(), None)
            .await;
        for index in 0..3i64 {
            let payload = serde_json::to_vec(&json!({"i": index})).unwrap();
            state
                .handle_frame(&store_arc, &context, index * 1_000_000, None, payload)
                .await;
        }
        state.handle_end_trace(&store_arc, &context).await;

        let trace_dir = TracePath::new("rec-1", "joints", "trace-1")
            .directory(context.recordings_root.as_path());
        let bytes = std::fs::read(trace_dir.join("trace.json")).unwrap();
        let parsed: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed, json!([{"i": 0}, {"i": 1}, {"i": 2}]));

        let trace = store
            .get_trace("trace-1")
            .await
            .expect("get trace")
            .expect("trace exists");
        assert_eq!(trace.write_status, TraceWriteStatus::Written);
        assert_eq!(trace.total_bytes as u64, bytes.len() as u64);
    }

    #[tokio::test]
    async fn empty_trace_still_produces_valid_json_array() {
        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let context = test_context(&tempdir.path().join("recordings"));
        let store_arc = Arc::new(store.clone());

        let mut state = ActorState::new("trace-1".to_string());
        state
            .handle_start_trace(&store_arc, "rec-1".to_string(), "joints".to_string(), None)
            .await;
        state.handle_end_trace(&store_arc, &context).await;

        let trace_dir = TracePath::new("rec-1", "joints", "trace-1")
            .directory(context.recordings_root.as_path());
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
    async fn video_chunk_ready_appends_segment_and_concats_on_end() {
        if !ffmpeg_available() {
            eprintln!("ffmpeg not on PATH — skipping video trace_actor test.");
            return;
        }

        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let context = test_context(&tempdir.path().join("recordings"));
        let store_arc = Arc::new(store.clone());

        let mut state = ActorState::new("trace-vid".to_string());
        state
            .handle_start_trace(&store_arc, "rec-1".to_string(), "RGB".to_string(), None)
            .await;

        // Build two NUT chunks via ffmpeg testsrc and place them where the
        // producer would have spooled them.
        let trace_dir = TracePath::new("rec-1", "RGB", "trace-vid")
            .directory(context.recordings_root.as_path());
        let chunks_dir = trace_dir.join(paths::CHUNKS_DIRNAME);
        std::fs::create_dir_all(&chunks_dir).unwrap();

        for chunk_index in 0..2u32 {
            let chunk_path = chunks_dir.join(paths::chunk_filename(chunk_index));
            let duration = "4";
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
                .arg(format!("testsrc=duration={duration}:size=16x16:rate=1"))
                .args(["-c:v", "rawvideo", "-pix_fmt", "rgb24", "-f", "nut"])
                .arg(&chunk_path)
                .status()
                .expect("synth status");
            assert!(status.success(), "synth NUT failed");

            let frame_timestamps_s: Vec<f64> =
                (0..4u32).map(|i| (chunk_index * 4 + i) as f64).collect();
            state
                .handle_video_chunk_ready(
                    &store_arc,
                    &context,
                    chunk_index,
                    16,
                    16,
                    chunk_path.metadata().unwrap().len(),
                    4,
                    frame_timestamps_s,
                )
                .await;
        }

        state.handle_end_trace(&store_arc, &context).await;

        assert!(trace_dir.join(paths::LOSSY_VIDEO_FILENAME).exists());
        assert!(trace_dir.join(paths::LOSSLESS_VIDEO_FILENAME).exists());
        assert!(trace_dir.join(paths::TRACE_JSON_FILENAME).exists());
        // Per-chunk segments and source NUTs should have been cleaned up.
        for chunk_index in 0..2u32 {
            assert!(!chunks_dir.join(paths::chunk_filename(chunk_index)).exists());
            assert!(!trace_dir
                .join(paths::chunk_lossy_filename(chunk_index))
                .exists());
            assert!(!trace_dir
                .join(paths::chunk_lossless_filename(chunk_index))
                .exists());
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
    async fn handle_frame_without_start_trace_is_a_noop() {
        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let context = test_context(&tempdir.path().join("recordings"));
        let store_arc = Arc::new(store.clone());

        let mut state = ActorState::new("trace-1".to_string());
        state
            .handle_frame(&store_arc, &context, 0, None, b"raw".to_vec())
            .await;
        assert_eq!(state.frame_count, 0);
        assert!(store.get_trace("trace-1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn envelopes_buffered_before_start_trace_replay_in_order() {
        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let context = test_context(&tempdir.path().join("recordings"));
        let store_arc = Arc::new(store.clone());

        let mut state = ActorState::new("trace-1".to_string());

        for index in 0..2i64 {
            let payload = serde_json::to_vec(&json!({"i": index})).unwrap();
            let flow = state
                .accept_envelope(
                    &store_arc,
                    &context,
                    Envelope::frame("trace-1".into(), index, None, payload),
                )
                .await;
            assert!(matches!(flow, Flow::Continue));
        }
        let flow = state
            .accept_envelope(
                &store_arc,
                &context,
                Envelope::EndTrace {
                    trace_id: "trace-1".into(),
                },
            )
            .await;
        assert!(
            matches!(flow, Flow::Continue),
            "EndTrace before StartTrace is buffered, not terminal"
        );
        assert_eq!(state.pending.len(), 3);
        assert_eq!(
            state.frame_count, 0,
            "buffered envelopes are not applied yet"
        );

        let flow = state
            .accept_envelope(
                &store_arc,
                &context,
                Envelope::StartTrace {
                    recording_id: "rec-1".into(),
                    trace_id: "trace-1".into(),
                    data_type: "joints".into(),
                    data_type_name: None,
                },
            )
            .await;
        assert!(
            matches!(flow, Flow::Stop),
            "buffered EndTrace finalises the trace during replay"
        );
        assert!(state.pending.is_empty(), "buffer drained on start_trace");

        let trace = store
            .get_trace("trace-1")
            .await
            .expect("get trace")
            .expect("trace exists");
        assert_eq!(trace.write_status, TraceWriteStatus::Written);
        let trace_dir = TracePath::new("rec-1", "joints", "trace-1")
            .directory(context.recordings_root.as_path());
        let bytes = std::fs::read(trace_dir.join("trace.json")).unwrap();
        let parsed: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed, json!([{"i": 0}, {"i": 1}]));
    }

    #[tokio::test]
    async fn handle_frame_drops_when_storage_budget_exhausted() {
        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let root = tempdir.path().join("recordings");
        let policy = StoragePolicy {
            storage_limit_bytes: None,
            min_free_disk_bytes: u64::MAX,
            refresh_interval: Duration::from_secs(60),
        };
        let budget = Arc::new(crate::storage::budget::StorageBudget::new(&root, policy));
        let context = Arc::new(TraceActorContext::new(
            root.clone(),
            budget,
            VideoEncoder::new(),
        ));
        let store_arc = Arc::new(store.clone());

        let mut state = ActorState::new("trace-1".to_string());
        state
            .handle_start_trace(&store_arc, "rec-1".to_string(), "joints".to_string(), None)
            .await;
        for index in 0..3i64 {
            let payload = serde_json::to_vec(&json!({"i": index})).unwrap();
            state
                .handle_frame(&store_arc, &context, index, None, payload)
                .await;
        }
        assert_eq!(state.frame_count, 0, "all frames must have been dropped");
        assert_eq!(state.dropped_over_budget, 3);
        let trace_dir = TracePath::new("rec-1", "joints", "trace-1")
            .directory(context.recordings_root.as_path());
        assert!(!trace_dir.join("trace.json").exists());
    }

    #[tokio::test]
    async fn handle_cancel_removes_disk_dir_and_resets_writer() {
        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let context = test_context(&tempdir.path().join("recordings"));
        let store_arc = Arc::new(store.clone());

        let mut state = ActorState::new("trace-1".to_string());
        state
            .handle_start_trace(&store_arc, "rec-1".to_string(), "joints".to_string(), None)
            .await;
        let payload = serde_json::to_vec(&json!({"i": 0})).unwrap();
        state
            .handle_frame(&store_arc, &context, 0, None, payload)
            .await;

        let trace_dir = TracePath::new("rec-1", "joints", "trace-1")
            .directory(context.recordings_root.as_path());
        assert!(trace_dir.exists(), "writer opens directory on first frame");

        state.handle_cancel(&store_arc, &context).await;
        assert!(!trace_dir.exists(), "cancel must remove the trace dir");
        assert!(matches!(state.writer, TraceWriter::Pending));
        assert_eq!(state.bytes_on_disk, 0);
    }

    #[tokio::test]
    async fn shutdown_without_end_marks_trace_failed() {
        let tempdir = TempDir::new().unwrap();
        let store = SqliteStateStore::open(&tempdir.path().join("state.db"))
            .await
            .expect("open store");
        let store_arc = Arc::new(store.clone());

        let mut state = ActorState::new("trace-1".to_string());
        state
            .handle_start_trace(&store_arc, "rec-1".to_string(), "joints".to_string(), None)
            .await;
        state.handle_shutdown_without_end(&store_arc).await;

        let trace = store
            .get_trace("trace-1")
            .await
            .expect("get trace")
            .expect("trace exists");
        assert_eq!(trace.write_status, TraceWriteStatus::Failed);
    }

    fn ffmpeg_available() -> bool {
        std::process::Command::new("which")
            .arg("ffmpeg")
            .output()
            .map(|output| output.status.success() && !output.stdout.is_empty())
            .unwrap_or(false)
    }
}
