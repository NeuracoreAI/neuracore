//! Per-trace actor task.
//!
//! Owns the SQLite lifecycle and the on-disk encoders for one trace. Sub-phase
//! 5f wires the encoders in: scalar / sensor traces stream into a
//! [`JsonTraceWriter`]; video traces (signalled by an [`Envelope::OpenFrameStream`])
//! spool raw RGB frames through a [`NutWriter`], then on `EndTrace` shell out
//! to [`VideoEncoder`] for the dual-output transcode and flush the
//! [`VideoMetadataAccumulator`] sidecar.
//!
//! Database writes are debounced — the per-frame `bytes_written` UPDATE that
//! Phase 4 left in place was useful for proving the wiring but is wasteful at
//! 200+ Hz scalar ingestion. We now flush a counter every
//! [`BYTES_WRITTEN_DEBOUNCE_FRAMES`] frames during ingestion and always on
//! finalise, matching the policy described in `docs/data-daemon-rewrite.md`
//! §5f. The encoding tail of the state machine — `initializing → writing →
//! pending_metadata → written` — is still driven from this actor; the
//! registration coordinator (Phase 6) keys off the terminal `written` state.

use std::sync::Arc;

use data_daemon_ipc::Envelope;
use serde_json::Value;
use tokio::sync::{mpsc, Semaphore};
use tokio::task;

use crate::encoding::json_trace::{JsonTraceError, JsonTraceWriter};
use crate::encoding::metadata::{MetadataError, VideoMetadataAccumulator};
use crate::encoding::nut_writer::{NutError, NutVideoConfig, NutWriter};
use crate::encoding::video_encoder::{VideoEncodeError, VideoEncodeRequest, VideoEncoder};
use crate::state::store::TraceUpdate;
use crate::state::{SqliteStateStore, StateStore, TraceWriteStatus};
use crate::storage::budget::StorageBudget;
use crate::storage::paths::{self, TracePath};

/// Key identifying one per-trace actor.
///
/// Frame and EndTrace envelopes only carry `trace_id` on the wire, so the
/// dispatcher routes by `trace_id` alone. The actor learns its
/// `recording_id` and `data_type` from the first `StartTrace` it sees.
pub type TraceKey = String;

/// Flush `bytes_written` to the DB every N frames instead of every frame.
///
/// At 30 fps video and 200 Hz scalars this keeps the SQLite write rate well
/// under 10 Hz per trace, which the WAL handles comfortably while still giving
/// the upload coordinator a recent enough byte count for its progress reports.
/// A finalise always issues a fresh UPDATE so the terminal row is exact.
const BYTES_WRITTEN_DEBOUNCE_FRAMES: u64 = 32;

/// Cap on concurrent ffmpeg transcodes. Each ffmpeg child happily saturates a
/// CPU core; without a cap N simultaneous `EndTrace` envelopes (the
/// integration matrix's `parallel_contexts=8` × 3-camera setup hits 24
/// trivially) starve the rest of the daemon. A small fixed value keeps the
/// transcode tail responsive without pulling in `num_cpus`.
pub(crate) const DEFAULT_FFMPEG_CONCURRENCY: usize = 4;

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
    /// Encoder used to transcode the per-trace NUT spool into the dual mp4
    /// outputs on `EndTrace`. Cloning a [`VideoEncoder`] is cheap (it carries
    /// only the configured ffmpeg binary path).
    pub video_encoder: VideoEncoder,
    /// Bounds concurrent ffmpeg children. Shared across actors so the
    /// integration matrix's parallel `EndTrace` storms don't fork-bomb the
    /// transcoder.
    pub ffmpeg_permits: Arc<Semaphore>,
    /// Optional daemon event bus. When present, the trace actor publishes a
    /// [`crate::state::DaemonEvent::TraceWritten`] on finalise so the
    /// registration coordinator can wake immediately. Optional to keep the
    /// pre-Phase-6 unit tests, which exercise the actor without a bus,
    /// running unchanged.
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
        | Envelope::OpenFrameStream { trace_id, .. } => Some(trace_id.clone()),
        Envelope::StartRecording { .. } | Envelope::StopRecording { .. } => None,
    }
}

/// Message accepted by a per-trace actor.
#[derive(Debug)]
pub enum TraceActorMessage {
    /// A producer-originated envelope routed to this trace.
    Envelope(Envelope),
}

/// Internal state of a per-trace actor.
///
/// Encoders are opened lazily: a scalar trace doesn't need a `trace.json` file
/// until the first frame arrives, and a video trace doesn't need a `raw.nut`
/// until both the resolution (`OpenFrameStream`) *and* a frame have been seen.
/// Keeping the writers as `Option` is what lets the actor uniformly mark an
/// empty trace as `written` on `EndTrace` without producing a stub file.
enum TraceWriter {
    /// No frames yet observed; the writer is decided on the first frame.
    Pending,
    /// Scalar trace streaming into a single `trace.json` array.
    Json(JsonTraceWriter),
    /// Video trace spooling RGB frames into `raw.nut` while the sidecar
    /// metadata buffers in memory.
    ///
    /// PTS is derived from `timestamp_ns` (microsecond ticks, relative to the
    /// first frame), matching the Python encoder's PTS contract in
    /// `disk_video_encoder.py::_compute_pts`. The container's time-base is
    /// `1/1_000_000`; sliding `pts_origin_us` to the first frame keeps PTS
    /// values small (avoids ffmpeg quirks with absolute nanosecond PTS) while
    /// preserving inter-frame spacing exactly. `last_pts` enforces strict
    /// monotonicity for the rare duplicate-timestamp case.
    Video {
        nut_writer: NutWriter,
        metadata: VideoMetadataAccumulator,
        frame_index: u64,
        pts_origin_us: Option<i64>,
        last_pts: Option<u64>,
    },
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
        let TraceActorMessage::Envelope(envelope) = message;
        match envelope {
            Envelope::StartTrace {
                recording_id,
                data_type,
                data_type_name,
                ..
            } => {
                state
                    .handle_start_trace(&store, recording_id, data_type, data_type_name)
                    .await
            }
            Envelope::OpenFrameStream { width, height, .. } => {
                state.handle_open_frame_stream(width, height);
            }
            Envelope::Frame {
                timestamp_ns,
                timestamp_s,
                payload,
                ..
            } => {
                state
                    .handle_frame(&store, &context, timestamp_ns, timestamp_s, payload)
                    .await
            }
            Envelope::EndTrace { trace_id } => {
                tracing::info!(trace_id, "end_trace envelope received by actor");
                state.handle_end_trace(&store, &context).await;
                // After end-of-trace there is nothing more for this actor to
                // do; returning drops the receiver so the dispatcher's
                // shutdown sweep is free of dangling senders.
                return;
            }
            Envelope::StartRecording { .. } | Envelope::StopRecording { .. } => {
                // Recording-scoped envelopes never carry a trace_id, so the
                // dispatcher routes them through its own branch — they can't
                // reach a per-trace actor. The match arm exists so adding a
                // new recording-scoped variant is a compile error here rather
                // than a silent ignore.
                unreachable!("recording-scoped envelopes are filtered by trace_key");
            }
        }
    }

    // Inbox closed without an explicit `EndTrace` — typically a shutdown or a
    // stop-recording cancel. Mark the trace as failed so the lifecycle is
    // observable from the DB and the registration coordinator doesn't pick it
    // up. Skip when no StartTrace has been seen — there is no row yet, and
    // writing one in `failed` state would be misleading.
    state.handle_shutdown_without_end(&store).await;
}

/// Per-actor mutable bookkeeping. Pulled out of `run` so the message handlers
/// can be tested with synthetic envelopes against a clean state object.
struct ActorState {
    trace_id: String,
    recording_id: Option<String>,
    data_type: Option<String>,
    /// Resolution announced by [`Envelope::OpenFrameStream`]. When set, the
    /// next `Frame` opens the video pipeline; when unset, the next `Frame`
    /// opens the JSON pipeline.
    video_config: Option<(u32, u32)>,
    writer: TraceWriter,
    frame_count: u64,
    bytes_on_disk: u64,
    /// Last `bytes_written` value flushed to the DB. Used by the debouncer to
    /// avoid issuing a no-op UPDATE when the writer's on-disk size hasn't
    /// changed since the last flush (e.g. when the JSON writer is still
    /// buffering).
    last_db_bytes: i64,
}

impl ActorState {
    fn new(trace_id: String) -> Self {
        Self {
            trace_id,
            recording_id: None,
            data_type: None,
            video_config: None,
            writer: TraceWriter::Pending,
            frame_count: 0,
            bytes_on_disk: 0,
            last_db_bytes: 0,
        }
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

    fn handle_open_frame_stream(&mut self, width: u32, height: u32) {
        // Reject zero / overflowing geometry now rather than at the first frame
        // — the NUT writer would reject it anyway, but failing here keeps the
        // pipeline closer to the producer's intent and avoids opening a stale
        // JSON writer for what is actually a video trace.
        if width == 0 || height == 0 {
            tracing::warn!(
                trace_id = self.trace_id,
                width,
                height,
                "ignoring open_frame_stream with zero dimension"
            );
            return;
        }
        self.video_config = Some((width, height));
        tracing::debug!(
            trace_id = self.trace_id,
            width,
            height,
            "video resolution announced"
        );
    }

    async fn handle_frame(
        &mut self,
        store: &Arc<SqliteStateStore>,
        context: &Arc<TraceActorContext>,
        timestamp_ns: i64,
        timestamp_s: Option<f64>,
        payload: Vec<u8>,
    ) {
        // Trace metadata must exist by the time the first frame arrives —
        // `StartTrace` is the first envelope routed by the dispatcher. Skip
        // the body otherwise so a misbehaving producer can't silently
        // accumulate frames against a nonexistent DB row. Checking before
        // `ensure_writer_open` also avoids creating an orphan trace file.
        let (Some(recording_id), Some(data_type)) =
            (self.recording_id.clone(), self.data_type.clone())
        else {
            tracing::warn!(
                trace_id = self.trace_id,
                "dropping frame received before start_trace"
            );
            return;
        };

        if !self.ensure_writer_open(context, &recording_id, &data_type, &payload) {
            return;
        }

        // Try to mark `writing` exactly once. Subsequent frames don't need an
        // UPDATE for this field; the bytes-written debouncer covers the rest.
        let bumped_status = self.frame_count == 0;

        if let Err(error) = self.append_frame(timestamp_ns, timestamp_s, &payload).await {
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

    /// Lazily open the JSON or video writer. Returns `false` when the storage
    /// budget refuses the write so the caller can skip the frame without
    /// touching the DB row.
    fn ensure_writer_open(
        &mut self,
        context: &Arc<TraceActorContext>,
        recording_id: &str,
        data_type: &str,
        payload: &[u8],
    ) -> bool {
        if !matches!(self.writer, TraceWriter::Pending) {
            return true;
        }

        // The budget tracker's check is best-effort — we ask for `payload.len()`
        // even though the JSON path expands the buffer and the video path may
        // spool an entire frame-plus-header. That mirrors the Python encoder
        // manager's pre-flight check, which also looks at the *incoming*
        // payload size, not the eventual on-disk footprint.
        let requested = payload.len() as u64;
        match context.storage_budget.check(requested) {
            Ok(check) if check.is_available() => {}
            Ok(check) => {
                tracing::warn!(
                    trace_id = self.trace_id,
                    ?check,
                    "storage budget refused write; dropping frame"
                );
                return false;
            }
            Err(error) => {
                tracing::warn!(
                    %error,
                    trace_id = self.trace_id,
                    "storage budget query failed; allowing write to proceed"
                );
            }
        }

        let trace_dir = self.trace_directory(recording_id, data_type, context);

        if let Some((width, height)) = self.video_config {
            // Time-base 1/1_000_000 (microsecond ticks) matches the Python
            // encoder's `PTS_FRACT = 1_000_000` so the on-disk PTS contract
            // is the same across both implementations. Frame PTS is derived
            // in `append_frame` as `(timestamp_ns - origin_ns) / 1000`.
            let config = NutVideoConfig {
                width,
                height,
                time_base_num: 1,
                time_base_den: 1_000_000,
            };
            let raw_nut = trace_dir.join(paths::RAW_NUT_FILENAME);
            match NutWriter::create(&raw_nut, config) {
                Ok(nut_writer) => {
                    self.bytes_on_disk = nut_writer.bytes_written();
                    self.writer = TraceWriter::Video {
                        nut_writer,
                        metadata: VideoMetadataAccumulator::new(),
                        frame_index: 0,
                        pts_origin_us: None,
                        last_pts: None,
                    };
                }
                Err(error) => {
                    tracing::warn!(
                        %error,
                        trace_id = self.trace_id,
                        path = %raw_nut.display(),
                        "failed to open NUT spool"
                    );
                    return false;
                }
            }
        } else {
            match JsonTraceWriter::open(&trace_dir) {
                Ok(json_writer) => {
                    self.bytes_on_disk = json_writer.bytes_on_disk();
                    self.writer = TraceWriter::Json(json_writer);
                }
                Err(error) => {
                    tracing::warn!(
                        %error,
                        trace_id = self.trace_id,
                        path = %trace_dir.display(),
                        "failed to open JSON trace"
                    );
                    return false;
                }
            }
        }
        true
    }

    async fn append_frame(
        &mut self,
        timestamp_ns: i64,
        timestamp_s: Option<f64>,
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
            TraceWriter::Video {
                nut_writer,
                metadata,
                frame_index,
                pts_origin_us,
                last_pts,
            } => {
                let origin_us = *pts_origin_us.get_or_insert(timestamp_ns / 1_000);
                // Microsecond ticks relative to the first frame. Clamp at 0 so
                // a producer that ships a non-monotonic earlier timestamp does
                // not produce a negative PTS; the `last_pts` monotonicity
                // guard below catches the rest.
                let relative_us = (timestamp_ns / 1_000).saturating_sub(origin_us).max(0);
                let mut pts = relative_us as u64;
                if let Some(previous) = *last_pts {
                    if pts <= previous {
                        pts = previous.saturating_add(1);
                    }
                }
                let (width, height) = self
                    .video_config
                    .expect("video writer implies video_config is set");
                // Decode the 16-byte big-endian test marker the integration
                // test producer stamps into the top-left 4×4 R-channel grid
                // (see `tests/.../shared/test_case/build_test_case_context.py`
                // `encode_frame_number`). The decode is essentially free (16
                // byte reads); we log it on every frame so a downstream test
                // failure has a daemon-side timeline of which source-frame
                // marker arrived at which receive position. Frames produced
                // outside the test harness decode to a deterministic noisy
                // value, which is harmless.
                let marker = decode_test_marker(payload, width);
                tracing::debug!(
                    trace_id = self.trace_id,
                    frame_index = *frame_index,
                    pts,
                    timestamp_ns,
                    payload_len = payload.len(),
                    marker,
                    "video frame received"
                );
                nut_writer.write_frame(pts, payload)?;
                *last_pts = Some(pts);
                let mut entry = serde_json::Map::new();
                // Match the Python writer's per-frame metadata shape:
                // `timestamp` is seconds (float), `width`/`height` set when
                // present. Prefer the producer-supplied `timestamp_s` (a real
                // f64 carrying the SDK's intended capture time) over the
                // nanosecond integer round-trip, which would truncate to
                // microsecond granularity and break the manual-timestamp
                // assertion (see Envelope::Frame docs).
                let timestamp =
                    timestamp_s.unwrap_or_else(|| timestamp_ns as f64 / 1_000_000_000.0);
                entry.insert("timestamp".to_string(), Value::from(timestamp));
                entry.insert("width".to_string(), Value::from(width as u64));
                entry.insert("height".to_string(), Value::from(height as u64));
                metadata.record_frame(entry);
                self.bytes_on_disk = nut_writer.bytes_written();
                *frame_index = frame_index.saturating_add(1);
                Ok(())
            }
        }
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
                // Empty trace — no encoder was ever opened. Match the Python
                // writers, which leave a single empty `trace.json` behind.
                let trace_dir = self.trace_directory(recording_id, data_type, context);
                let json = JsonTraceWriter::open(&trace_dir)?;
                let total = json.finish()?;
                Ok(total)
            }
            TraceWriter::Json(writer) => Ok(writer.finish()?),
            TraceWriter::Video {
                nut_writer,
                metadata,
                ..
            } => {
                // Drop the NUT writer first so its `BufWriter` flushes the
                // last frame before ffmpeg attaches. The spool path is
                // recovered from the trace directory because the writer
                // doesn't survive `finish`.
                let trace_dir = self.trace_directory(recording_id, data_type, context);
                let raw_nut = trace_dir.join(paths::RAW_NUT_FILENAME);
                let _final_nut_bytes = nut_writer.finish()?;

                let lossy = trace_dir.join(paths::LOSSY_VIDEO_FILENAME);
                let lossless = trace_dir.join(paths::LOSSLESS_VIDEO_FILENAME);
                let request = VideoEncodeRequest {
                    raw_nut,
                    lossy_mp4: lossy,
                    lossless_mp4: lossless,
                };

                // Cap concurrent ffmpeg children across all per-trace actors.
                // The permit is released on drop, so a panic or `?` short-
                // circuit inside `run` cannot leak the slot.
                let _ffmpeg_permit = context
                    .ffmpeg_permits
                    .acquire()
                    .await
                    .map_err(|_| FrameAppendError::FfmpegPermits)?;
                let outcome = context.video_encoder.run(&request).await?;
                drop(_ffmpeg_permit);

                // Sidecar metadata is the *last* thing on disk so a partial
                // transcode failure leaves a recognisable "no sidecar"
                // signature for the recovery sweep that lands in Phase 7.
                let metadata_bytes = flush_metadata_blocking(metadata, trace_dir.clone()).await?;

                Ok(outcome
                    .lossy_bytes
                    .saturating_add(outcome.lossless_bytes)
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
    Nut(#[from] NutError),
    #[error(transparent)]
    VideoEncode(#[from] VideoEncodeError),
    #[error(transparent)]
    Metadata(#[from] MetadataError),
}

/// Decode the integration test producer's 16-byte big-endian frame marker.
///
/// The test harness stamps the source frame index into the R-channel of the
/// top-left 4×4 pixel grid (see `tests/.../shared/test_case/constants.py`
/// `FRAME_BYTE_LENGTH=16`, `FRAME_GRID_SIZE=4`). Pixel `(row, col)` in that
/// grid carries marker byte `row * 4 + col`, with the R channel at byte
/// offset `(row * width + col) * 3` in a packed RGB24 buffer.
///
/// Returns `None` if the payload is too short for the 4×4 grid at the given
/// `width`. The first 16 marker bytes are returned as the low bytes of a
/// `u128` so the integer round-trips losslessly through `tracing::debug!`.
fn decode_test_marker(payload: &[u8], width: u32) -> Option<u128> {
    let stride = (width as usize).checked_mul(3)?;
    // Need bytes through row 3, column 3 inclusive — i.e. offset
    // `3 * stride + 3 * 3` must exist in `payload`.
    let last_byte = stride.checked_mul(3)?.checked_add(3 * 3)?;
    if payload.len() <= last_byte {
        return None;
    }
    let mut marker: u128 = 0;
    for row in 0..4 {
        for col in 0..4 {
            let byte = payload[row * stride + col * 3];
            marker = (marker << 8) | u128::from(byte);
        }
    }
    Some(marker)
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
        Err(join_error) => {
            // A join failure inside `spawn_blocking` means the runtime is
            // shutting down or the task panicked — fold both into the
            // metadata-error arm so callers see a uniform `FrameAppendError`.
            Err(FrameAppendError::Metadata(MetadataError::Write {
                path: path_for_error,
                source: std::io::Error::other(format!("metadata flush join failed: {join_error}")),
            }))
        }
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

    #[test]
    fn decode_test_marker_round_trips_known_indices() {
        // Build a 64×4 RGB24 buffer, stamp marker `value` into the top-left
        // 4×4 R-channel grid the way the integration-test producer does, then
        // verify `decode_test_marker` reads it back.
        let width: u32 = 64;
        let stride = (width as usize) * 3;
        let buffer_len = stride * 4;
        for value in [0u128, 1, 254, 255, 256, 257, 599, 1 << 32] {
            let mut payload = vec![100u8; buffer_len];
            for byte_index in 0..16usize {
                let shift = (15 - byte_index) * 8;
                let byte = ((value >> shift) & 0xFF) as u8;
                let row = byte_index / 4;
                let col = byte_index % 4;
                payload[row * stride + col * 3] = byte;
            }
            let decoded = decode_test_marker(&payload, width).expect("decode");
            assert_eq!(decoded, value, "round-trip failed for {value}");
        }
    }

    #[test]
    fn decode_test_marker_returns_none_for_short_payload() {
        // 4-pixel-wide buffer with only 3 rows is missing the row-3 marker
        // bytes; the decoder must refuse rather than read out-of-bounds.
        let width: u32 = 4;
        let payload = vec![0u8; (width as usize) * 3 * 3];
        assert!(decode_test_marker(&payload, width).is_none());
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
    async fn video_trace_produces_mp4_outputs_when_ffmpeg_available() {
        // Same skip pattern as `encoding::video_encoder` tests so the suite
        // stays green in sandboxes that lack FFmpeg.
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
        state.handle_open_frame_stream(16, 16);
        for index in 0..4u64 {
            let mut pixels = vec![0u8; 16 * 16 * 3];
            for (pixel_index, chunk) in pixels.chunks_mut(3).enumerate() {
                chunk[0] = ((pixel_index + index as usize) & 0xFF) as u8;
                chunk[1] = ((pixel_index * 3 + index as usize) & 0xFF) as u8;
                chunk[2] = ((pixel_index * 5 + index as usize) & 0xFF) as u8;
            }
            state
                .handle_frame(
                    &store_arc,
                    &context,
                    index as i64 * 33_000_000,
                    None,
                    pixels,
                )
                .await;
        }
        state.handle_end_trace(&store_arc, &context).await;

        let trace_dir = TracePath::new("rec-1", "RGB", "trace-vid")
            .directory(context.recordings_root.as_path());
        assert!(trace_dir.join("lossy.mp4").exists());
        assert!(trace_dir.join("lossless.mp4").exists());
        assert!(trace_dir.join("trace.json").exists());
        assert!(
            !trace_dir.join("raw.nut").exists(),
            "raw.nut should be unlinked on successful transcode"
        );

        let trace = store
            .get_trace("trace-vid")
            .await
            .expect("get trace")
            .expect("trace exists");
        assert_eq!(trace.write_status, TraceWriteStatus::Written);
        assert!(trace.total_bytes > 0);
    }

    #[tokio::test]
    async fn frame_received_before_start_trace_is_dropped() {
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
        // No `StartTrace` ⇒ no row was created.
        assert!(store.get_trace("trace-1").await.unwrap().is_none());
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
