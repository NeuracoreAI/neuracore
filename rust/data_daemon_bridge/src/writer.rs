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

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, LazyLock, Mutex, Once};
use std::time::{Duration, Instant};

use data_daemon_shared::service_name::MAX_VIDEO_CHUNK_FRAMES;
use data_daemon_shared::Envelope;

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
///
/// One chunk's worth is also the in-RAM writer-queue ceiling
/// ([`WRITER_QUEUE_MAX_BYTES`]): changing this size moves both the chunk
/// granularity *and* the producer's backpressure headroom.
const CHUNK_FLUSH_BYTES: u64 = 256 * 1024 * 1024;

/// Backpressure cap for the writer's frame queue. `log_frame` copies a frame in
/// and returns; the background writer thread drains the queue to disk, so the
/// caller blocks only once the queue is *full*. The cap is therefore sized by
/// the writer-side stall it lets the caller ride out before `log_rgb` slows.
///
/// Those stalls are the kernel's system-wide `balance_dirty_pages` throttle —
/// hundreds of ms once dirty pages cross `vm.dirty_ratio`, driven by *any*
/// process on the host (the daemon's ffmpeg transcodes, other tenants) and not
/// preventable by the producer. So size by drain time: at the heaviest workload
/// the suite runs — 1080p RGB @ 60 fps ≈ 356 MiB/s per camera — one chunk's
/// 256 MiB buys ≈ 0.7 s, enough to absorb the ~0.6-0.8 s stalls seen in
/// practice. The old 64 MiB bought only ~0.17 s and was overrun.
///
/// The queue holds *raw* RGB ([`FrameJob::data`]) — PNG-encoding happens after
/// dequeue — so PNG shrinks what hits the disk but not the queue-fill rate
/// during a stall; the raw sizing stands. One chunk is also the deliberate
/// ceiling: this anonymous RAM competes with the page cache, so a larger queue
/// would shrink the dirty headroom and worsen the very throttle it absorbs.
/// Sustained overload is bounded by the on-disk spool cap (`spool_max`, default
/// 2 GiB).
const WRITER_QUEUE_MAX_BYTES: usize = CHUNK_FLUSH_BYTES as usize;

/// How often the writer rescans its spool inbox to refresh the on-disk backlog
/// estimate and release frame-admission backpressure. Also bounds how long a
/// producer stays blocked after the daemon drains a chunk (≤ this interval).
const SPOOL_SCAN_INTERVAL: Duration = Duration::from_millis(250);

/// How long a video frame may wait for spool-backlog headroom before
/// `log_frame` gives up and raises. The spool drains only as the *daemon*
/// transcodes chunks, so a dead or wedged daemon must surface as a logging
/// error rather than block the caller's thread forever. One second is far
/// longer than any healthy transcode stall, yet short enough that the caller
/// learns promptly instead of silently losing frames.
const FRAME_ADMISSION_TIMEOUT: Duration = Duration::from_secs(1);

/// Resolve the producer's spool-backlog cap (bytes) from the daemon profile
/// config (`spool_limit`: `NCD_SPOOL_LIMIT` → active profile → default).
fn resolved_spool_max_bytes() -> u64 {
    floor_spool_max(data_daemon_shared::config::resolve_spool_limit_bytes())
}

/// Apply the cap's safety floor. A configured value of `0` (or any non-positive)
/// disables the bound; any positive value is floored to two chunk sizes so there
/// is always room for the in-progress chunk plus a sealed one — a cap below the
/// chunk size would wedge the writer (the open chunk alone exceeds it, so every
/// frame blocks and the chunk never seals).
fn floor_spool_max(configured: i64) -> u64 {
    if configured <= 0 {
        return 0;
    }
    (configured as u64).max(2 * CHUNK_FLUSH_BYTES)
}

/// Rescan the spool inbox and publish the fresh backlog estimate to `queue`,
/// releasing any producer blocked on the spool cap. No-op when the bound is
/// disabled, so a disabled bound never pays for a directory walk.
fn refresh_spool_backlog(queue: &FrameQueue) {
    if queue.spool_max == 0 {
        return;
    }
    let scanned = crate::paths::spool_root()
        .map(|root| data_daemon_shared::paths::directory_bytes(&root))
        .unwrap_or(0);
    queue.set_spool_bytes(scanned);
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

/// Returned by [`FrameQueue::push`] when a video frame cannot be admitted
/// within [`FRAME_ADMISSION_TIMEOUT`] because the on-disk spool backlog is stuck
/// at its cap. The producer maps this to a Python exception so a stalled daemon
/// surfaces as a logging error instead of a silently dropped frame.
#[derive(Debug)]
pub(crate) struct LoggingStalled;

/// Byte-bounded MPSC queue between the logging threads and the writer thread.
///
/// Two independent backpressure limits gate *frame* admission (control
/// messages always pass so a stop/cancel can drain even a full queue):
///
/// - [`WRITER_QUEUE_MAX_BYTES`] caps the in-memory frames awaiting a disk write.
/// - `spool_max` caps the producer's on-disk NUT backlog: when the daemon can't
///   transcode spooled chunks fast enough the inbox would otherwise grow
///   unbounded and fill the disk, stalling `stop_recording`'s tail-chunk flush
///   for seconds. The writer thread refreshes `spool_bytes` from a periodic
///   directory scan ([`refresh_spool_backlog`]).
pub(crate) struct FrameQueue {
    inner: Mutex<FrameQueueInner>,
    not_full: Condvar,
    not_empty: Condvar,
    /// In-memory frame-buffer cap. Drained by the local writer thread, which
    /// always makes progress, so a frame blocked on it waits unbounded.
    /// [`WRITER_QUEUE_MAX_BYTES`] in production; tests shrink it to exercise the
    /// path with small frames.
    memory_max: usize,
    /// On-disk spool-backlog cap. Drained only as the daemon transcodes chunks,
    /// so a frame blocked on it is time-limited (see `block_timeout`). `0`
    /// disables the bound.
    spool_max: u64,
    /// How long a frame may wait on the **spool** cap before [`push`] rejects it
    /// with [`LoggingStalled`]. Does not apply to the in-memory cap. Always
    /// [`FRAME_ADMISSION_TIMEOUT`] in production; tests shorten it.
    ///
    /// [`push`]: FrameQueue::push
    block_timeout: Duration,
}

struct FrameQueueInner {
    msgs: VecDeque<WriterMsg>,
    bytes: usize,
    spool_bytes: u64,
}

impl FrameQueue {
    fn new() -> Self {
        Self::build(
            WRITER_QUEUE_MAX_BYTES,
            resolved_spool_max_bytes(),
            FRAME_ADMISSION_TIMEOUT,
        )
    }

    /// Assemble a queue from explicit caps — the one place the fields are
    /// initialised, so the production and test constructors can't drift.
    fn build(memory_max: usize, spool_max: u64, block_timeout: Duration) -> Self {
        FrameQueue {
            inner: Mutex::new(FrameQueueInner {
                msgs: VecDeque::new(),
                bytes: 0,
                spool_bytes: 0,
            }),
            not_full: Condvar::new(),
            not_empty: Condvar::new(),
            memory_max,
            spool_max,
            block_timeout,
        }
    }

    /// Build a queue with an explicit spool cap and stall timeout (keeping the
    /// production in-memory cap), bypassing the profile/env config read so
    /// backpressure can be tested deterministically.
    #[cfg(test)]
    fn with_caps(spool_max: u64, block_timeout: Duration) -> Self {
        Self::build(WRITER_QUEUE_MAX_BYTES, spool_max, block_timeout)
    }

    /// Build a queue with an explicit in-memory cap too, so the in-memory
    /// backpressure path can be exercised with small frames.
    #[cfg(test)]
    fn with_memory_cap(memory_max: usize, spool_max: u64, block_timeout: Duration) -> Self {
        Self::build(memory_max, spool_max, block_timeout)
    }

    /// Enqueue a message, blocking the caller only while a *frame* would exceed
    /// the in-memory cap **or** the on-disk spool backlog is at its cap (control
    /// messages never block, so a stop/cancel drains even a full queue). A lone
    /// frame larger than the in-memory cap is still admitted once the queue is
    /// empty, so forward progress is always possible.
    ///
    /// The two caps wait differently because they drain differently:
    ///
    /// - The in-memory cap drains as the local writer thread writes frames to
    ///   disk, which always makes progress, so a frame blocked *only* on it
    ///   waits unbounded — exactly as before the spool cap existed.
    /// - The spool cap drains only as the daemon transcodes chunks, so a frame
    ///   blocked on it gives up after [`FRAME_ADMISSION_TIMEOUT`] and returns
    ///   [`LoggingStalled`] rather than block a logging thread forever behind a
    ///   dead daemon.
    pub(crate) fn push(&self, msg: WriterMsg) -> Result<(), LoggingStalled> {
        let add = msg.queue_bytes();
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        if add > 0 {
            // `spool_deadline` tracks *continuous* time the frame has spent
            // blocked on the spool cap. It is armed when the spool is the
            // blocker and cleared whenever the spool falls back below its cap, so
            // a slow-but-draining daemon (which keeps dipping under the cap)
            // never trips the timeout — only a daemon that is wedged at the cap
            // does. A frame blocked solely on the in-memory cap waits unbounded.
            let mut spool_deadline: Option<Instant> = None;
            loop {
                let over_spool = self.over_spool_cap(&inner);
                if !over_spool && !self.over_memory_cap(&inner, add) {
                    break;
                }
                inner = if over_spool {
                    let deadline =
                        *spool_deadline.get_or_insert_with(|| Instant::now() + self.block_timeout);
                    let remaining = match deadline.checked_duration_since(Instant::now()) {
                        Some(remaining) if !remaining.is_zero() => remaining,
                        _ => return Err(LoggingStalled),
                    };
                    self.not_full
                        .wait_timeout(inner, remaining)
                        .unwrap_or_else(|p| p.into_inner())
                        .0
                } else {
                    spool_deadline = None;
                    self.not_full.wait(inner).unwrap_or_else(|p| p.into_inner())
                };
            }
        }
        inner.bytes += add;
        inner.msgs.push_back(msg);
        self.not_empty.notify_one();
        Ok(())
    }

    /// Whether admitting an `add`-byte frame would breach the in-memory cap.
    /// Yields once the queue is empty, so an oversized lone frame still makes
    /// progress. The local writer thread always drains this, so a frame blocked
    /// here waits unbounded.
    fn over_memory_cap(&self, inner: &FrameQueueInner, add: usize) -> bool {
        inner.bytes > 0 && inner.bytes + add > self.memory_max
    }

    /// Whether the on-disk spool backlog is at its cap. Disabled when
    /// `spool_max == 0`. This clears only when the daemon drains the inbox, so a
    /// frame blocked here is bounded by [`FRAME_ADMISSION_TIMEOUT`].
    fn over_spool_cap(&self, inner: &FrameQueueInner) -> bool {
        self.spool_max > 0 && inner.spool_bytes >= self.spool_max
    }

    /// Publish the latest scanned spool-backlog size and wake every producer
    /// blocked on the spool cap so they re-evaluate admission.
    fn set_spool_bytes(&self, scanned: u64) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.spool_bytes = scanned;
        self.not_full.notify_all();
    }

    /// Block indefinitely until a message is available, then pop it (FIFO).
    /// Used when the spool cap is disabled, where the writer has no reason to
    /// wake on a timer.
    fn pop(&self) -> WriterMsg {
        self.pop_inner(None)
            .expect("an unbounded wait never times out")
    }

    /// Pop the next message (FIFO), blocking up to `timeout`; `None` on timeout.
    ///
    /// The timeout lets the writer thread wake to rescan the spool even when
    /// frame admission is fully blocked (no new frames arriving) — that rescan
    /// is what releases the backpressure as the daemon drains the inbox, so the
    /// spool bound can never deadlock.
    fn pop_timeout(&self, timeout: Duration) -> Option<WriterMsg> {
        self.pop_inner(Some(timeout))
    }

    /// Shared pop body: a `None` timeout waits forever, `Some` bounds each wait.
    fn pop_inner(&self, timeout: Option<Duration>) -> Option<WriterMsg> {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(msg) = inner.msgs.pop_front() {
                inner.bytes -= msg.queue_bytes();
                self.not_full.notify_one();
                return Some(msg);
            }
            match timeout {
                None => {
                    inner = self
                        .not_empty
                        .wait(inner)
                        .unwrap_or_else(|p| p.into_inner());
                }
                Some(timeout) => {
                    let (guard, result) = self
                        .not_empty
                        .wait_timeout(inner, timeout)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    inner = guard;
                    if result.timed_out() && inner.msgs.is_empty() {
                        return None;
                    }
                }
            }
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

thread_local! {
    /// Per-thread cache of the process video-writer queue. The hot `log_frame`
    /// path hits this slot — a plain TLS load — instead of taking the global
    /// `VIDEO_WRITER` mutex and a `getpid()` syscall on every frame. Cleared by
    /// the fork child handler so a forked child rebuilds.
    static WRITER_QUEUE: RefCell<Option<Arc<FrameQueue>>> = const { RefCell::new(None) };
}

/// Return this process's writer queue. Fast path: the thread-local cache (no
/// lock, no syscall). Slow path: heal/spawn under the global lock and cache.
pub(crate) fn writer_queue() -> Arc<FrameQueue> {
    if let Some(queue) = WRITER_QUEUE.with(|cell| cell.borrow().clone()) {
        return queue;
    }
    let queue = writer_queue_global();
    WRITER_QUEUE.with(|cell| *cell.borrow_mut() = Some(queue.clone()));
    queue
}

/// Heal/spawn the process writer thread under the global lock, returning its
/// queue. On the (near-impossible) spawn failure we log and return a detached
/// queue *without* recording it, so the next call retries rather than the caller
/// blocking forever on a consumer-less queue.
fn writer_queue_global() -> Arc<FrameQueue> {
    ensure_writer_fork_handler_registered();
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
        .spawn(move || {
            // The compression pool lives on (and is owned by) the writer thread,
            // so a forked child rebuilds both together through the heal path.
            let pool = CompressPool::new(compress_pool_size());
            writer_loop(&worker_queue, &pool);
        }) {
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

/// Install a `pthread_atfork` child handler (once) that clears this thread's
/// cached [`WRITER_QUEUE`]. The global `VIDEO_WRITER` self-heals via its
/// `owner_pid`, but the per-thread cache would otherwise hand a forked child a
/// stale queue whose writer thread didn't survive the fork.
fn ensure_writer_fork_handler_registered() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // SAFETY: standard libc fork-callback registration. `clear_queue_cache`
        // is `extern "C"` and only drops a const-initialised TLS `Arc`.
        let result = unsafe { libc::pthread_atfork(None, None, Some(clear_queue_cache)) };
        if result != 0 {
            tracing::warn!(
                errno = result,
                "pthread_atfork registration failed; video writer-queue cache relies on PID heal",
            );
        }
    });
}

/// `pthread_atfork` child callback: drop the surviving thread's cached writer
/// queue so the next [`writer_queue`] rebuilds through the PID-keyed heal path.
extern "C" fn clear_queue_cache() {
    WRITER_QUEUE.with(|cell| {
        cell.borrow_mut().take();
    });
}

/// Cap on frames whose compression may be in flight on the pool ahead of the
/// writer's in-order muxing. Bounds the pipeline's extra memory and, because the
/// writer stops draining the queue once this many are outstanding, preserves the
/// queue's admission backpressure (the writer never outruns the pool).
const MAX_FRAMES_IN_FLIGHT: usize = 8;

/// PNG-compression worker threads feeding the writer. Compression (~24 ms for a
/// detailed 720p frame) — not the NUT muxing — is the pipeline's dominant cost,
/// so a single thread can't keep up with two 30 fps cameras; a small pool can,
/// while the writer thread still owns all chunk state and ordering. Capped so it
/// doesn't oversubscribe a small host against the daemon's transcode fleet.
fn compress_pool_size() -> usize {
    std::thread::available_parallelism()
        .map(|cores| (cores.get() / 2).clamp(2, 4))
        .unwrap_or(2)
}

/// One-shot slot a pool worker fills with a frame's compressed PNG and the
/// writer thread collects, in submission order. Hand-rolled (Mutex + Condvar)
/// rather than a channel so the writer can both block on the next frame and poll
/// whether it is ready yet.
struct FrameResult {
    png: Mutex<Option<Vec<u8>>>,
    ready: Condvar,
}

impl FrameResult {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            png: Mutex::new(None),
            ready: Condvar::new(),
        })
    }

    /// Store the compressed frame and wake a writer blocked in [`wait`](Self::wait).
    fn set(&self, png: Vec<u8>) {
        *self.png.lock().unwrap_or_else(|p| p.into_inner()) = Some(png);
        self.ready.notify_one();
    }

    /// Take the compressed frame if the worker has finished, without blocking.
    fn try_take(&self) -> Option<Vec<u8>> {
        self.png.lock().unwrap_or_else(|p| p.into_inner()).take()
    }

    /// Block until the worker has compressed the frame, then take it.
    fn wait(&self) -> Vec<u8> {
        let mut guard = self.png.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(png) = guard.take() {
                return png;
            }
            guard = self.ready.wait(guard).unwrap_or_else(|p| p.into_inner());
        }
    }
}

/// One frame submitted to the compression pool.
struct CompressJob {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
    result: Arc<FrameResult>,
}

/// Fixed pool of PNG-compression worker threads shared by the writer loop.
struct CompressPool {
    work: Arc<(Mutex<VecDeque<CompressJob>>, Condvar)>,
}

impl CompressPool {
    /// Spawn `workers` compression threads. They live for the process (like the
    /// writer thread); a forked child re-creates the pool via the writer heal
    /// path, so the parent's detached workers are never inherited.
    fn new(workers: usize) -> Self {
        let work = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        for index in 0..workers {
            let work = work.clone();
            if let Err(error) = std::thread::Builder::new()
                .name(format!("nc-png-compress-{index}"))
                .spawn(move || compress_worker(&work))
            {
                tracing::warn!(%error, "failed to spawn PNG compression worker");
            }
        }
        Self { work }
    }

    /// Queue a frame for compression and wake one idle worker.
    fn submit(&self, job: CompressJob) {
        let (lock, cond) = &*self.work;
        lock.lock()
            .unwrap_or_else(|p| p.into_inner())
            .push_back(job);
        cond.notify_one();
    }
}

/// Compression worker: pull frames FIFO and compress each to PNG, publishing the
/// result into its one-shot slot. Compression is stateless per frame, so any
/// worker can take any frame — the writer restores per-stream order on collect.
fn compress_worker(work: &(Mutex<VecDeque<CompressJob>>, Condvar)) {
    let (lock, cond) = work;
    loop {
        let job = {
            let mut queue = lock.lock().unwrap_or_else(|p| p.into_inner());
            loop {
                if let Some(job) = queue.pop_front() {
                    break job;
                }
                queue = cond.wait(queue).unwrap_or_else(|p| p.into_inner());
            }
        };
        let png = crate::nut_writer::encode_png_frame(job.width, job.height, &job.rgb);
        job.result.set(png);
    }
}

/// A frame whose compression is in flight, awaiting in-order muxing by the
/// writer thread. Carries the routing metadata the write phase needs (its raw
/// pixels already moved into the pool job) plus the slot the pool fills.
struct PendingFrame {
    job: FrameJob,
    result: Arc<FrameResult>,
}

/// Validate one frame and submit it to the compression pool, recording it in the
/// in-order pipeline. A frame whose byte length disagrees with its geometry is
/// dropped here (the pre-split inline encode rejected it the same way) rather
/// than handed to a worker.
fn submit_frame(pool: &CompressPool, in_flight: &mut VecDeque<PendingFrame>, mut job: FrameJob) {
    let expected = (job.width as usize)
        .checked_mul(job.height as usize)
        .and_then(|pixels| pixels.checked_mul(3));
    if expected != Some(job.data.len()) {
        tracing::warn!(
            sensor_name = job.sensor_name,
            ?expected,
            actual = job.data.len(),
            "video frame size disagrees with geometry; dropping frame"
        );
        return;
    }
    let result = FrameResult::new();
    // Move the raw pixels into the compression job; the pending frame keeps only
    // the routing metadata (its `data` is now empty and never read again).
    let rgb = std::mem::take(&mut job.data);
    pool.submit(CompressJob {
        width: job.width,
        height: job.height,
        rgb,
        result: result.clone(),
    });
    in_flight.push_back(PendingFrame { job, result });
}

/// Mux one already-compressed pending frame into its chunk (writer thread).
fn write_pending_frame(pending: PendingFrame, png: Vec<u8>) {
    if let Err(error) = record_video_frame(
        &pending.job.robot_id,
        pending.job.robot_instance,
        &pending.job.data_type,
        &pending.job.sensor_name,
        pending.job.width,
        pending.job.height,
        &png,
        pending.job.timestamp_ns,
        pending.job.timestamp_s,
    ) {
        tracing::warn!(%error, sensor_name = pending.job.sensor_name, "failed to spool video frame");
    }
}

/// Mux every frame whose compression has already finished, front to back,
/// stopping at the first still-compressing frame so per-stream order is kept.
fn drain_ready(in_flight: &mut VecDeque<PendingFrame>) {
    while let Some(front) = in_flight.front() {
        match front.result.try_take() {
            Some(png) => {
                let pending = in_flight.pop_front().expect("front just observed");
                write_pending_frame(pending, png);
            }
            None => break,
        }
    }
}

/// Block until the front frame is compressed, then mux it — used to make room
/// when the pipeline is full and to drain it fully at a stop/cancel barrier.
fn write_front(in_flight: &mut VecDeque<PendingFrame>) {
    if let Some(pending) = in_flight.pop_front() {
        let png = pending.result.wait();
        write_pending_frame(pending, png);
    }
}

/// The writer thread's run loop. Sole accessor of the in-progress chunk state
/// and sole publisher of video chunk + stop/cancel envelopes for this process.
///
/// Frame compression is offloaded to `pool`; this thread submits each frame then
/// muxes the results back into their chunks in strict submission order (which is
/// per-stream FIFO). It only ever muxes — the expensive PNG encode runs on the
/// pool — so it keeps up with multi-camera capture the single-thread inline
/// encode could not.
fn writer_loop(queue: &FrameQueue, pool: &CompressPool) {
    // With the spool cap disabled there is nothing to scan, so block on each
    // message indefinitely rather than waking on a timer.
    let bounded = queue.spool_max > 0;
    if bounded {
        // Prime the backlog estimate so the first frames see a real spool size.
        refresh_spool_backlog(queue);
    }
    let mut in_flight: VecDeque<PendingFrame> = VecDeque::new();
    let mut last_scan = Instant::now();
    loop {
        let next = if bounded {
            queue.pop_timeout(SPOOL_SCAN_INTERVAL)
        } else {
            Some(queue.pop())
        };
        match next {
            Some(WriterMsg::Frame(job)) => {
                submit_frame(pool, &mut in_flight, job);
                // Cap outstanding compressions: block-mux the oldest until the
                // pipeline has room. This doubles as the queue's backpressure —
                // the writer stops popping while it waits here.
                while in_flight.len() >= MAX_FRAMES_IN_FLIGHT {
                    write_front(&mut in_flight);
                }
            }
            Some(WriterMsg::FlushSource {
                robot_id,
                robot_instance,
                ack,
            }) => {
                // The stop barrier must seal chunks that include every frame
                // submitted before it, so finish all outstanding compressions and
                // mux them before flushing the tail.
                while !in_flight.is_empty() {
                    write_front(&mut in_flight);
                }
                if let Err(error) = flush_source_chunks(&robot_id, robot_instance) {
                    tracing::warn!(%error, "failed to flush tail video chunks on stop");
                }
                let _ = ack.send(());
            }
            Some(WriterMsg::DropSource {
                robot_id,
                robot_instance,
                ack,
            }) => {
                // Mux outstanding frames before dropping so the pool never holds a
                // slot for a stream we are about to remove; the dropped chunks are
                // reclaimed by the daemon's cancel + recovery sweep regardless.
                while !in_flight.is_empty() {
                    write_front(&mut in_flight);
                }
                let prefix = source_prefix(&robot_id, robot_instance);
                with_video_chunks(|streams| {
                    streams.retain(|key, _| !key.starts_with(&prefix));
                });
                let _ = ack.send(());
            }
            // pop_timeout elapsed with no message: fall through to the rescan.
            None => {}
        }
        // Mux whatever finished compressing, keeping spooling latency low without
        // ever blocking on a slow frame.
        drain_ready(&mut in_flight);
        // Refresh the backlog estimate on a coarse cadence so frame-admission
        // backpressure tracks the daemon draining the spool inbox.
        if bounded && last_scan.elapsed() >= SPOOL_SCAN_INTERVAL {
            refresh_spool_backlog(queue);
            last_scan = Instant::now();
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

/// OS thread id of the calling thread. Used to disambiguate a video chunk's
/// spool filename across producer threads and as a breadcrumb when inspecting
/// the spool directory. Only needs to be stable and per-thread-unique within a
/// process, so any OS-native thread id will do.
#[cfg(target_os = "linux")]
fn current_thread_id() -> i64 {
    // SAFETY: `gettid` takes no arguments and cannot fail.
    unsafe { libc::gettid() as i64 }
}

/// macOS has no `gettid`; `pthread_threadid_np` yields the same kind of
/// kernel-assigned, per-thread-unique 64-bit id.
#[cfg(not(target_os = "linux"))]
fn current_thread_id() -> i64 {
    let mut tid: u64 = 0;
    // SAFETY: writes the id of the calling thread (`pthread_self()`) into `tid`;
    // it cannot fail for a valid, live thread.
    unsafe { libc::pthread_threadid_np(libc::pthread_self(), &mut tid) };
    tid as i64
}

/// Whether the in-progress chunk should be sealed now, checked after each
/// appended frame. A chunk is rolled at the **lower** of two bounds:
///
/// * [`CHUNK_FLUSH_BYTES`] — measured against *logical* (decoded-equivalent)
///   bytes, not the compressed on-disk size, so chunk granularity (and the
///   daemon's per-chunk transcode cost) stays stable however well the PNG
///   frames compress. Keying off the on-disk size would pack thousands of
///   tiny PNG frames into one chunk and balloon that transcode unit.
/// * [`MAX_VIDEO_CHUNK_FRAMES`] — keeps the chunk's `VideoChunkReady`
///   announcement within one `COMMANDS_MAX_PAYLOAD_BYTES` sample. Small frames
///   never reach the byte threshold mid-recording, so without the frame cap a
///   long recording accumulates one ever-growing chunk whose per-frame
///   timestamp vectors eventually overflow the commands slice — the
///   announcement then fails to publish and the recording's video is lost.
fn should_flush_chunk(logical_chunk_bytes: u64, frame_count: u32) -> bool {
    logical_chunk_bytes >= CHUNK_FLUSH_BYTES || frame_count >= MAX_VIDEO_CHUNK_FRAMES
}

/// Append one frame to the `(source, sensor)` in-progress NUT chunk, opening
/// the chunk lazily, enforcing PTS monotonicity, and flushing once the chunk
/// crosses [`CHUNK_FLUSH_BYTES`] or [`MAX_VIDEO_CHUNK_FRAMES`]. Best-effort:
/// NUT-write errors are logged and the frame dropped, never propagated to
/// Python.
///
/// `png_payload` is the frame already compressed to a per-frame PNG by a pool
/// worker (see [`submit_frame`]); this only muxes it into the chunk.
#[allow(clippy::too_many_arguments)]
fn record_video_frame(
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    sensor_name: &str,
    width: u32,
    height: u32,
    png_payload: &[u8],
    timestamp_ns: i64,
    timestamp_s: f64,
) -> Result<(), ProducerError> {
    let key = stream_key(robot_id, robot_instance, data_type, sensor_name);
    // Resolve the slot, building the per-stream state (and its spool dir) only
    // on the FIRST frame of a stream. The spool-dir path build is several
    // allocations, so doing it per frame (as an earlier revision did) added
    // allocation churn to the writer's hot path and backed up the frame queue
    // at high frame rates. The recordings root is pre-validated on the GIL in
    // `log_frame`, so `spool_dir` only returns `None` on a genuine
    // misconfiguration — drop the frame (never panic on the writer thread).
    let slot: VideoChunkSlot = match with_video_chunks(|streams| {
        if let Some(slot) = streams.get(&key) {
            return Some(slot.clone());
        }
        let spool = spool_dir(robot_id, robot_instance, data_type, sensor_name)?;
        let slot = Arc::new(Mutex::new(VideoChunkState {
            width,
            height,
            spool_dir: spool,
            nut_writer: None,
            chunk_publish_ns: 0,
            chunk_thread_id: 0,
            frame_count: 0,
            pts_origin_us: None,
            last_pts_us: None,
            frame_timestamps_ns: Vec::new(),
            frame_timestamps_s: Vec::new(),
        }));
        streams.insert(key.clone(), slot.clone());
        Some(slot)
    }) {
        Some(slot) => slot,
        None => {
            tracing::error!(
                sensor_name,
                "recordings root unresolved on writer thread; dropping video frame"
            );
            return Ok(());
        }
    };

    // The announcements are built under the per-stream lock but published
    // outside it — `publish()` blocks the calling thread when the daemon falls
    // behind, and holding the mutex across that block would stall this
    // camera's next frame.
    let announcements = {
        let mut state = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        append_frame_locked(
            &mut state,
            robot_id,
            robot_instance,
            data_type,
            sensor_name,
            width,
            height,
            png_payload,
            timestamp_ns,
            timestamp_s,
        )
    };

    for envelope in announcements {
        // Hand each sealed chunk's announcement to the publisher thread rather
        // than publishing inline: this runs on the writer thread, which must
        // never block on an IPC publish (the stop/cancel barrier waits on it).
        let _ = publisher_tx().send(PublishMsg::Announce(envelope));
    }
    Ok(())
}

/// Append one frame to the locked per-stream chunk `state`, returning every
/// chunk announcement produced this call: a geometry-change seal and/or a
/// size/frame-cap flush (so a single call can yield two). Pure with respect to
/// IPC — the caller publishes the returned envelopes outside the lock — which
/// also makes the open/seal/roll logic unit-testable without a live daemon.
/// Best-effort: a NUT open/write error logs and drops the frame.
///
/// `png_payload` is the frame already compressed to a per-frame PNG; this muxes
/// it straight into the chunk.
#[allow(clippy::too_many_arguments)]
fn append_frame_locked(
    state: &mut VideoChunkState,
    robot_id: &str,
    robot_instance: i64,
    data_type: &str,
    sensor_name: &str,
    width: u32,
    height: u32,
    png_payload: &[u8],
    timestamp_ns: i64,
    timestamp_s: f64,
) -> Vec<Envelope> {
    let mut announcements: Vec<Envelope> = Vec::new();

    // A mid-stream resolution change can't share a chunk with the prior
    // geometry: the NUT header advertises the opening frame's size, so a
    // differently-sized frame fails the writer's size check and is silently
    // dropped (or, on a coincidental `w*h*3` match, corrupts the encode). Seal
    // the open chunk (announced below) and reopen a fresh one with the new
    // geometry rather than dropping every later frame.
    if state.nut_writer.is_some() && (state.width != width || state.height != height) {
        tracing::warn!(
            sensor_name,
            old_width = state.width,
            old_height = state.height,
            new_width = width,
            new_height = height,
            "video frame geometry changed mid-stream; sealing chunk and reopening"
        );
        if let Some(envelope) =
            flush_chunk_locked(robot_id, robot_instance, data_type, sensor_name, state)
        {
            announcements.push(envelope);
        }
    }
    state.width = width;
    state.height = height;

    // Each fresh chunk opens with a header syncpoint at global_key_pts=0, so
    // reset the PTS origin whenever a chunk is (re)opened — after a geometry
    // seal above, or a size/frame-cap roll on the previous call. Every chunk's
    // frames then start near PTS 0 rather than carrying the whole stream's
    // elapsed time, keeping each chunk's frame PTS consistent with its header.
    if state.nut_writer.is_none() {
        state.pts_origin_us = None;
        state.last_pts_us = None;
    }
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
                return announcements;
            }
        }
    }

    // Roll on decoded-equivalent volume, not the on-disk byte count — see
    // [`should_flush_chunk`] for why.
    let logical_bytes_after_write = {
        let writer = state.nut_writer.as_mut().expect("opened immediately above");
        if let Err(error) = writer.write_frame_precompressed(pts, png_payload) {
            tracing::warn!(
                %error,
                sensor_name,
                "failed to write video frame to NUT chunk; dropping frame"
            );
            return announcements;
        }
        writer.logical_bytes()
    };
    state.last_pts_us = Some(pts);
    state.frame_count = state.frame_count.saturating_add(1);
    state.frame_timestamps_ns.push(timestamp_ns);
    state.frame_timestamps_s.push(timestamp_s);

    if should_flush_chunk(logical_bytes_after_write, state.frame_count) {
        if let Some(envelope) =
            flush_chunk_locked(robot_id, robot_instance, data_type, sensor_name, state)
        {
            announcements.push(envelope);
        }
    }
    announcements
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
    fn frame_result_is_ready_only_after_set_and_taken_once() {
        let result = FrameResult::new();
        assert!(result.try_take().is_none(), "empty slot must not be ready");
        result.set(vec![1, 2, 3]);
        assert_eq!(result.try_take(), Some(vec![1, 2, 3]));
        assert!(
            result.try_take().is_none(),
            "the compressed frame is collected exactly once"
        );
    }

    #[test]
    fn frame_result_wait_wakes_on_a_late_set() {
        // The writer thread blocks in `wait` until a pool worker publishes the
        // compressed frame — the ordering guarantee the pipeline relies on.
        let result = FrameResult::new();
        let worker = result.clone();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            worker.set(vec![7, 7, 7]);
        });
        assert_eq!(result.wait(), vec![7, 7, 7]);
        handle.join().unwrap();
    }

    #[test]
    fn flush_is_the_lower_of_the_two_bounds() {
        // A chunk of tiny frames is sealed by the frame cap with bytes still far
        // under the byte threshold — i.e. whichever bound is hit first wins.
        assert!(should_flush_chunk(1, MAX_VIDEO_CHUNK_FRAMES));
        // ...and a byte-heavy chunk is sealed before the frame cap.
        assert!(should_flush_chunk(CHUNK_FLUSH_BYTES, 1));
    }

    /// Build a fresh, empty per-stream chunk state rooted at `spool_dir`.
    fn fresh_state(spool_dir: PathBuf, width: u32, height: u32) -> VideoChunkState {
        VideoChunkState {
            width,
            height,
            spool_dir,
            nut_writer: None,
            chunk_publish_ns: 0,
            chunk_thread_id: 0,
            frame_count: 0,
            pts_origin_us: None,
            last_pts_us: None,
            frame_timestamps_ns: Vec::new(),
            frame_timestamps_s: Vec::new(),
        }
    }

    #[test]
    fn geometry_change_seals_chunk_and_reopens_at_new_size() {
        // M11 regression: a mid-stream resolution change must seal the open
        // chunk (so its frames aren't lost) and reopen at the new geometry,
        // rather than silently dropping every later, differently-sized frame.
        let dir = tempfile::tempdir().unwrap();
        let mut state = fresh_state(dir.path().to_path_buf(), 2, 2);

        // First frame at 2x2 (rgb24 = 2*2*3 bytes) just opens a chunk.
        let frame_2x2 = vec![0u8; 2 * 2 * 3];
        let opened = append_frame_locked(
            &mut state, "r", 0, "RGB", "cam", 2, 2, &frame_2x2, 1_000, 0.0,
        );
        assert!(
            opened.is_empty(),
            "opening the first chunk emits no announcement"
        );
        assert!(state.nut_writer.is_some());
        assert_eq!(state.frame_count, 1);

        // Second frame at 4x4 must seal the 2x2 chunk and reopen at 4x4.
        let frame_4x4 = vec![0u8; 4 * 4 * 3];
        let sealed = append_frame_locked(
            &mut state, "r", 0, "RGB", "cam", 4, 4, &frame_4x4, 2_000, 0.001,
        );
        assert_eq!(sealed.len(), 1, "the geometry change seals the prior chunk");
        match &sealed[0] {
            Envelope::VideoChunkReady {
                width,
                height,
                frame_count,
                ..
            } => {
                assert_eq!(
                    (*width, *height),
                    (2, 2),
                    "the sealed chunk keeps the original geometry"
                );
                assert_eq!(*frame_count, 1, "it carries the single 2x2 frame");
            }
            other => panic!("expected VideoChunkReady, got {other:?}"),
        }
        assert_eq!(
            (state.width, state.height),
            (4, 4),
            "state adopts the new geometry"
        );
        assert!(
            state.nut_writer.is_some(),
            "a fresh chunk is reopened at the new geometry"
        );
        assert_eq!(state.frame_count, 1, "the new chunk holds the 4x4 frame");
    }

    #[test]
    fn flush_seals_chunk_with_populated_announcement_and_resets_state() {
        // The normal seal path the daemon routes on (size/frame-cap roll or the
        // stop barrier). The announcement must carry every frame's identity in
        // order, and the state must reset so the next frame opens a fresh chunk
        // rather than re-announcing these frames.
        let dir = tempfile::tempdir().unwrap();
        let mut state = fresh_state(dir.path().to_path_buf(), 2, 2);
        let frame = vec![0u8; 2 * 2 * 3];

        for (timestamp_ns, timestamp_s) in [(1_000, 0.0), (2_000, 0.001), (3_000, 0.002)] {
            let announcements = append_frame_locked(
                &mut state,
                "r",
                0,
                "RGB",
                "cam",
                2,
                2,
                &frame,
                timestamp_ns,
                timestamp_s,
            );
            assert!(
                announcements.is_empty(),
                "frames below both bounds accumulate without sealing"
            );
        }
        assert_eq!(state.frame_count, 3);

        let envelope = flush_chunk_locked("r", 0, "RGB", "cam", &mut state)
            .expect("an open chunk seals into an announcement");
        match envelope {
            Envelope::VideoChunkReady {
                sensor_name,
                width,
                height,
                frame_count,
                byte_count,
                frame_timestamps_ns,
                frame_timestamps_s,
                ..
            } => {
                assert_eq!(sensor_name.as_deref(), Some("cam"));
                assert_eq!((width, height), (2, 2));
                assert_eq!(frame_count, 3);
                assert!(byte_count > 0, "a sealed chunk has a non-zero NUT file");
                assert_eq!(frame_timestamps_ns, vec![1_000, 2_000, 3_000]);
                assert_eq!(frame_timestamps_s, vec![0.0, 0.001, 0.002]);
            }
            other => panic!("expected VideoChunkReady, got {other:?}"),
        }

        // The seal takes the writer and clears the counters / per-frame vectors,
        // so the next frame opens a brand-new chunk.
        assert!(state.nut_writer.is_none());
        assert_eq!(state.frame_count, 0);
        assert!(state.frame_timestamps_ns.is_empty());
        assert!(state.frame_timestamps_s.is_empty());
    }

    #[test]
    fn flush_without_an_open_chunk_announces_nothing() {
        // No frames written since the last seal → nothing to announce. Keeps the
        // stop barrier's empty-source case from emitting a bogus zero-frame
        // chunk.
        let dir = tempfile::tempdir().unwrap();
        let mut state = fresh_state(dir.path().to_path_buf(), 2, 2);
        assert!(flush_chunk_locked("r", 0, "RGB", "cam", &mut state).is_none());
    }

    #[test]
    fn non_monotonic_timestamps_do_not_drop_frames() {
        // Capture timestamps can repeat or go backwards (clock coalescing,
        // batched logging). Every frame must still be recorded: the writer bumps
        // each PTS to stay strictly increasing rather than dropping the
        // colliding frames.
        let dir = tempfile::tempdir().unwrap();
        let mut state = fresh_state(dir.path().to_path_buf(), 2, 2);
        let frame = vec![0u8; 2 * 2 * 3];

        // Three frames at the same instant, then one that goes backwards.
        for timestamp_ns in [5_000, 5_000, 5_000, 4_000] {
            let _ = append_frame_locked(
                &mut state,
                "r",
                0,
                "RGB",
                "cam",
                2,
                2,
                &frame,
                timestamp_ns,
                0.0,
            );
        }
        assert_eq!(
            state.frame_count, 4,
            "every frame is written despite non-monotonic capture timestamps"
        );

        match flush_chunk_locked("r", 0, "RGB", "cam", &mut state).expect("seal") {
            Envelope::VideoChunkReady { frame_count, .. } => assert_eq!(frame_count, 4),
            other => panic!("expected VideoChunkReady, got {other:?}"),
        }
    }

    /// A minimal video-frame message carrying `bytes` of payload.
    fn frame_msg(bytes: usize) -> WriterMsg {
        WriterMsg::Frame(FrameJob {
            robot_id: "robot".to_string(),
            robot_instance: 0,
            data_type: "RGB_IMAGES".to_string(),
            sensor_name: "camera".to_string(),
            width: 0,
            height: 0,
            timestamp_ns: 0,
            timestamp_s: 0.0,
            data: vec![0u8; bytes],
        })
    }

    #[test]
    fn floor_spool_max_disables_on_non_positive() {
        assert_eq!(floor_spool_max(0), 0);
        assert_eq!(floor_spool_max(-1), 0);
    }

    #[test]
    fn floor_spool_max_raises_sub_chunk_caps_to_two_chunks() {
        // A sub-chunk cap would wedge the writer, so it is floored.
        assert_eq!(floor_spool_max(1), 2 * CHUNK_FLUSH_BYTES);
        // A comfortably large cap is honoured verbatim.
        let large = 8 * CHUNK_FLUSH_BYTES as i64;
        assert_eq!(floor_spool_max(large), large as u64);
    }

    #[test]
    fn frame_admitted_when_spool_below_cap() {
        let queue = FrameQueue::with_caps(4 * CHUNK_FLUSH_BYTES, FRAME_ADMISSION_TIMEOUT);
        assert!(queue.push(frame_msg(1024)).is_ok());
    }

    #[test]
    fn frame_rejected_when_spool_stuck_at_cap() {
        let queue = FrameQueue::with_caps(CHUNK_FLUSH_BYTES, Duration::from_millis(50));
        // At the cap with nothing draining: a frame must time out and reject
        // rather than block the caller forever.
        queue.set_spool_bytes(CHUNK_FLUSH_BYTES);
        let started = Instant::now();
        assert!(matches!(queue.push(frame_msg(1024)), Err(LoggingStalled)));
        assert!(
            started.elapsed() >= Duration::from_millis(50),
            "it should wait out the stall window before rejecting"
        );
    }

    #[test]
    fn control_messages_bypass_a_full_spool() {
        let queue = FrameQueue::with_caps(CHUNK_FLUSH_BYTES, Duration::from_millis(50));
        queue.set_spool_bytes(CHUNK_FLUSH_BYTES);
        let (ack_tx, _ack_rx) = std::sync::mpsc::channel();
        // A flush/cancel must enqueue immediately even while frames are blocked.
        assert!(queue
            .push(WriterMsg::FlushSource {
                robot_id: "robot".to_string(),
                robot_instance: 0,
                ack: ack_tx,
            })
            .is_ok());
    }

    #[test]
    fn disabled_cap_never_applies_spool_backpressure() {
        let queue = FrameQueue::with_caps(0, Duration::from_millis(50));
        // Even a huge reported backlog cannot block when the bound is disabled.
        queue.set_spool_bytes(u64::MAX);
        assert!(queue.push(frame_msg(1024)).is_ok());
    }

    #[test]
    fn draining_the_spool_unblocks_a_waiting_frame() {
        // A long stall window so only the drain — not a timeout — can release it.
        let queue = Arc::new(FrameQueue::with_caps(
            CHUNK_FLUSH_BYTES,
            Duration::from_secs(10),
        ));
        queue.set_spool_bytes(CHUNK_FLUSH_BYTES);
        let pusher = {
            let queue = Arc::clone(&queue);
            std::thread::spawn(move || queue.push(frame_msg(1024)))
        };
        // Let the pusher reach its wait, then report the spool as drained.
        std::thread::sleep(Duration::from_millis(50));
        queue.set_spool_bytes(0);
        assert!(pusher.join().unwrap().is_ok());
    }

    #[test]
    fn in_memory_backpressure_is_never_time_limited() {
        // The stall timeout guards against a wedged *daemon* (the spool cap). A
        // frame blocked purely on the in-memory cap — with the spool well below
        // its cap — must wait for the local writer to drain it, never reject like
        // a spool stall, even long past the (short) stall window.
        let queue = Arc::new(FrameQueue::with_memory_cap(
            1024,
            4 * CHUNK_FLUSH_BYTES,
            Duration::from_millis(50),
        ));
        // First frame is admitted despite exceeding the cap (queue was empty),
        // leaving the buffer full so the next frame must block.
        assert!(queue.push(frame_msg(2048)).is_ok());

        let pusher = {
            let queue = Arc::clone(&queue);
            std::thread::spawn(move || queue.push(frame_msg(2048)))
        };
        // Well beyond the stall window: a spool stall would have rejected by now,
        // but the in-memory wait must still be blocking.
        std::thread::sleep(Duration::from_millis(150));
        assert!(
            !pusher.is_finished(),
            "in-memory backpressure must not be subject to the spool stall timeout"
        );

        // Draining one frame frees headroom and admits the waiter.
        assert!(matches!(queue.pop(), WriterMsg::Frame(_)));
        assert!(pusher.join().unwrap().is_ok());
    }
}
