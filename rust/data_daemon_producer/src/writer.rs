//! Background video-writer thread and the in-progress video-chunk registry.
//!
//! `log_frame` must never block the caller on disk I/O. The producer spool lives
//! on the same filesystem as the daemon's SQLite WAL and the ffmpeg transcode
//! outputs, so on ext4 (`data=ordered`) a frame `write()` can stall for hundreds
//! of ms behind an unrelated `fsync`/journal commit — and because `log_frame`
//! holds the GIL across that write, the stall freezes the whole producer. To
//! keep the robot-facing ingest path latency-bounded, `log_frame` copies the
//! frame and hands it to a dedicated per-process writer thread, returning at
//! once. The writer owns *every* NUT write, chunk seal, and
//! `VideoChunkReady`/`StopRecording`/`CancelRecording` publish for video, so a
//! disk stall blocks only the writer — never a `log_*` caller.
//!
//! Publishing the tail chunks *and* the `StopRecording`/`CancelRecording` from
//! the one writer thread (its own iceoryx2 publisher).
//!
//! Lifecycle envelopes (`StartRecording` / `StopRecording` / `CancelRecording`)
//! are emphatically NOT routed through the writer: they stay on the *calling*
//! thread's publisher, the same port `StartRecording` uses, so consecutive
//! recordings' start/stop boundaries keep their strict in-order delivery. (Were
//! `StopRecording` published from the writer's port instead, it could be
//! reordered against the next recording's `StartRecording` on the main port —
//! the daemon then sees a start while the prior window is still live and drops
//! the overlapping window's data.) The stop/cancel paths only *barrier* on the
//! writer — seal + announce (or drop) the source's tail chunks, ack — and then
//! publish the lifecycle envelope themselves. Chunk-before-stop ordering is not
//! a same-port guarantee here but is safe anyway: the daemon holds every
//! `VideoChunkReady` back (`NCD_HOLDBACK_MS`, default 500 ms) and retains a
//! just-closed window, so a tail chunk announced just before the stop still
//! routes into the (closing) window by its in-window open timestamp.
//!
//! ## Fork safety
//!
//! The process-wide [`VIDEO_CHUNKS`] registry stores the owning PID and wipes
//! itself on the first access from a process whose PID no longer matches, so a
//! forked `multiprocessing` worker never inherits stale parent state. The
//! [`VIDEO_WRITER`] handle heals the same way: the parent's writer thread does
//! not survive into a forked child, so the child re-spawns one on first use.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, LazyLock, Mutex};

use data_daemon_ipc::service_name::MAX_VIDEO_CHUNK_FRAMES;
use data_daemon_ipc::Envelope;

use crate::nut_writer::{NutVideoConfig, NutWriter};
use crate::paths::{source_prefix, split_stream_key, spool_chunk_filename, spool_dir, stream_key};
use crate::publisher::{now_ns, publisher_tx, ProducerError, PublishMsg};

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

/// Backpressure cap for the writer's frame queue. A transient disk stall is
/// absorbed by buffering frames up to this many bytes before `log_frame`
/// blocks; only a *sustained* overload (the writer genuinely can't keep up)
/// propagates backpressure to the caller. 64 MiB holds ~a second of a
/// multi-camera 256×256@30 workload while staying small next to a worker's RSS.
const WRITER_QUEUE_MAX_BYTES: usize = 64 * 1024 * 1024;

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

/// One frame handed to the background writer. Owns its pixel bytes (copied out
/// of the caller's buffer under the GIL) so the caller can return immediately.
pub(crate) struct FrameJob {
    pub(crate) robot_id: String,
    pub(crate) robot_instance: i64,
    pub(crate) data_type: String,
    pub(crate) sensor_name: String,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) timestamp_ns: i64,
    pub(crate) timestamp_s: f64,
    pub(crate) data: Vec<u8>,
}

/// Work item for the writer thread.
pub(crate) enum WriterMsg {
    /// Append one frame to its `(source, sensor)` in-progress chunk.
    Frame(FrameJob),
    /// Stop barrier: drain every frame queued ahead for the source (FIFO), seal
    /// and announce its open chunks, then acknowledge. The caller publishes
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
pub(crate) struct FrameQueue {
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
    pub(crate) fn push(&self, msg: WriterMsg) {
        let add = msg.queue_bytes();
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if add > 0 {
            while inner.bytes > 0 && inner.bytes + add > WRITER_QUEUE_MAX_BYTES {
                inner = self.not_full.wait(inner).unwrap_or_else(|p| p.into_inner());
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
pub(crate) fn writer_queue() -> Arc<FrameQueue> {
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

/// Whether the in-progress chunk should be sealed now, checked after each
/// appended frame. A chunk is rolled at the **lower** of two bounds:
///
/// * [`CHUNK_FLUSH_BYTES`] — keeps the daemon's per-chunk encode cost amortised.
/// * [`MAX_VIDEO_CHUNK_FRAMES`] — keeps the chunk's `VideoChunkReady`
///   announcement within one `COMMANDS_MAX_PAYLOAD_BYTES` sample. Small frames
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
