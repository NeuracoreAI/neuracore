//! Routes source/sensor-tagged data into recording windows and on to per-trace
//! actors.
//!
//! The producer is a thin shipper: it publishes lifecycle events
//! (`StartRecording` / `StopRecording` / `CancelRecording`) and
//! source/sensor/timestamp-tagged data, knowing nothing about recordings. This
//! single-owner dispatcher task decides which recording each datum belongs to:
//!
//! - **Lifecycle events are applied immediately**, mutating the per-source
//!   active-window map. `StartRecording` allocates a `recording_index` and
//!   opens a window; `StopRecording` closes it (begins the drain); `Cancel`
//!   tears it down.
//! - **Data is held for a fixed holdback** in a per-source-ordered queue, then
//!   routed by its `publish_timestamp_ns` (a wall-clock instant stamped by the
//!   producer at publish, on the same clock as the lifecycle bounds) into the
//!   window whose `[started_at_ns, stopped_at_ns)` contains it. The holdback
//!   absorbs the cross-publisher arrival skew that the old per-frame
//!   `sequence_number` machinery used to reconcile.
//!
//! Membership is decided by the *publish timestamp*, never arrival time, and is
//! decoupled from the data's own capture clock — so cross-publisher reorder
//! cannot change which recording a datum belongs to, only when it is observed,
//! which the holdback + a closing-window retention of `2·HOLDBACK` absorb. A
//! just-closed window stays resolvable until every legitimately-held datum has
//! been released; finalisation is then a single `WindowClosing` signal to each
//! actor (no sequence counting).
//!
//! A video chunk is the one envelope carrying more than one datum, and the same
//! rule decides it *per frame*, at both bounds: a chunk is a container, not a
//! member. The producer seals chunks on its own caps and knows nothing of
//! windows, so one chunk can hold frames from before a recording, inside it, and
//! inside the next one — and each window claims exactly the run of frames it
//! owns (see [`claims_for_chunk`] and [`frame_range_in_window`], with
//! `data_daemon_shared::video_boundary` for where a boundary lands). Deciding
//! membership by the chunk's *open stamp* instead would hand a window every
//! frame or none, and none is what a chunk opened before the window got — taking
//! its in-window frames with it. That is how a camera process that never calls
//! `start_recording` lost its whole trace.
//!
//! **Tail video chunks** escape the holdback: the producer seals them after it
//! published the stop, so no retention constant bounds their lateness. A stopped
//! window waits for that producer's `SourceFlushed` marker instead, capped by
//! [`FLUSH_MARKER_WAIT_CAP`] — leaving retention load-bearing only for windows
//! closed without a stop, which get no marker at all.
//!
//! A marker speaks only for its own process, so a window must know whose markers
//! it is owed: the producer claims the source from its logging thread
//! ([`Envelope::VideoProducerActive`]), since its first sealed chunk would be
//! late by exactly the backlog this exists to tolerate.
//!
//! Everything here is owned by one tokio task, so the window map and holdback
//! queue need no locks — total ordering through the `select!` loop is what
//! makes the routing decisions provable.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use data_daemon_shared::{video_boundary, BatchedDataItem, Envelope, FrameDtype};
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio::time::sleep;
use uuid::Uuid;

use crate::lifecycle::shutdown::ShutdownSignal;
use crate::pipeline::trace_actor::{
    self, TraceActorContext, TraceActorMessage, TraceIdentity, TraceKey,
};
use crate::state::{DaemonEvent, NewRecording, SqliteStateStore, StateStore};
use crate::storage::paths;

/// Default holdback: each data envelope waits this long after daemon receipt
/// before it is routed. Tunable via `NCD_HOLDBACK_MS`.
///
/// Lifecycle is applied on arrival while data is held, so a `StartRecording`
/// racing its own data still opens the window first.
const DEFAULT_HOLDBACK_MS: u64 = 500;

/// Environment override for the holdback, in milliseconds.
const HOLDBACK_ENV: &str = "NCD_HOLDBACK_MS";

/// Hard bound on how long a stopped window waits for its producer's
/// [`Envelope::SourceFlushed`] marker before being evicted anyway.
const FLUSH_MARKER_WAIT_CAP: Duration = Duration::from_secs(30);

/// A source silent (no data, no lifecycle) for this long has its open window
/// force-closed as a crash backstop, so a producer that died without a Stop
/// still finalises (or is swept). Distinct from the restart sweep, which
/// handles a daemon that itself died.
const IDLE_REAP: Duration = Duration::from_secs(30);

/// How long an active source is polled for due releases / evictions. A fully
/// idle daemon (no held data, no closing windows) sleeps [`IDLE_REAP`] instead.
/// The coarse cadence adds at most this much jitter to a release deadline,
/// negligible against the holdback.
const HOUSEKEEP_INTERVAL: Duration = Duration::from_millis(25);

/// Upper bound on how long a `RefreshConfig` may park the dispatcher loop
/// waiting for the watcher's ack. The refresh is a fast `spawn_blocking` profile
/// read, so this only guards against a stalled watcher wedging the hot path; on
/// timeout we proceed and let the periodic poll pick the change up.
const REFRESH_CONFIG_ACK_TIMEOUT: Duration = Duration::from_secs(5);

/// Bounded per-trace queue size. A smaller cap acts as a forced flush throttle;
/// 256 absorbs the high-dimensionality burst at the cost of ~10 KiB of message
/// headers per trace.
const TRACE_QUEUE_CAPACITY: usize = 256;

/// Bounded listener → dispatcher channel.
const DISPATCHER_INBOX_CAPACITY: usize = 1024;

/// Source identity: `(robot_id, robot_instance)`.
type Source = (String, i64);

/// Resolve the configured holdback, honouring the `NCD_HOLDBACK_MS` override.
fn configured_holdback() -> Duration {
    let millis = std::env::var(HOLDBACK_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_HOLDBACK_MS);
    Duration::from_millis(millis)
}

/// Handle owned by the daemon main loop. Drop it on shutdown to close every
/// per-trace actor.
pub struct DispatcherHandle {
    join: JoinHandle<()>,
}

impl DispatcherHandle {
    /// Wait for the dispatcher to finish processing in-flight messages and the
    /// per-trace actors to terminate.
    pub async fn shutdown(self) {
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "dispatcher task join failed during shutdown");
        }
    }
}

/// Optional runtime context passed to the dispatcher.
#[derive(Clone, Default)]
pub struct DispatcherContext {
    /// Daemon event bus, used to publish recording/trace lifecycle events.
    pub event_bus: Option<crate::state::EventBus>,
    /// Refresh-request sender to the config watcher, used to service a
    /// `RefreshConfig` command (see [`Dispatcher::handle_refresh_config`]).
    /// `None` in tests / when no watcher is wired, where `RefreshConfig` is a
    /// no-op.
    pub config_refresh_tx: Option<tokio::sync::mpsc::Sender<crate::cloud::ConfigRefreshRequest>>,
}

/// Spawn the dispatcher task and return its inbound `mpsc::Sender`.
///
/// Test-only convenience over [`spawn_with_context`] with a default context.
#[cfg(test)]
pub fn spawn(
    store: SqliteStateStore,
    actor_context: Arc<TraceActorContext>,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> (mpsc::Sender<Envelope>, DispatcherHandle) {
    spawn_with_context(
        store,
        actor_context,
        DispatcherContext::default(),
        shutdown_rx,
    )
}

/// Spawn the dispatcher with an explicit [`DispatcherContext`].
pub fn spawn_with_context(
    store: SqliteStateStore,
    actor_context: Arc<TraceActorContext>,
    context: DispatcherContext,
    shutdown_rx: broadcast::Receiver<ShutdownSignal>,
) -> (mpsc::Sender<Envelope>, DispatcherHandle) {
    let (tx, rx) = mpsc::channel::<Envelope>(DISPATCHER_INBOX_CAPACITY);
    let join = tokio::spawn(async move {
        let mut dispatcher = Dispatcher::new(store, actor_context, context);
        dispatcher.run(rx, shutdown_rx).await;
    });
    (tx, DispatcherHandle { join })
}

/// A per-trace actor's routing handle, stored inside its window.
struct TraceHandle {
    sender: mpsc::Sender<TraceActorMessage>,
    /// Daemon-assigned, per-trace monotonic video chunk index.
    next_video_chunk: u32,
}

/// One recording window for a source.
///
/// Membership is decided by the producer **publish-clock** boundaries
/// `[started_at_ns, stopped_at_ns)`. Every data envelope carries a
/// `publish_timestamp_ns` stamped at publish on the same wall clock the
/// lifecycle `started_at_ns` / `stopped_at_ns` use, so routing never depends
/// on the data's own (possibly custom) capture clock — it depends only on when
/// the producer published, which is exactly "which recording was active then".
struct ActiveWindow {
    recording_index: i64,
    /// Inclusive lower bound — the lifecycle publish time of the start.
    started_at_ns: i64,
    /// Exclusive upper bound — the lifecycle publish time of the stop. `None`
    /// while live (open above).
    stopped_at_ns: Option<i64>,
    /// Daemon clock at which the window closed — drives the eviction deadline.
    stop_recv_at: Option<Instant>,
    /// Closed by a `StopRecording`, so its producer still owes a
    /// [`Envelope::SourceFlushed`] marker.
    awaiting_flush: bool,
    /// OS process ids that have video in this window; it is not flushed until
    /// every one of them also appears in `flushed_producers`.
    ///
    /// Needs both populating envelopes: a [`Envelope::VideoChunkReady`] arrives
    /// only once a chunk has sealed, while a [`Envelope::VideoProducerActive`]
    /// claim lands inside the window even when that first chunk will not.
    video_producers: HashSet<u32>,
    /// OS process ids whose `SourceFlushed` marker has left the holdback queue,
    /// so every tail chunk ordered behind it has routed.
    flushed_producers: HashSet<u32>,
    /// Per-trace actors spawned within this window.
    traces: HashMap<TraceKey, TraceHandle>,
}

impl ActiveWindow {
    /// Does this window's `[started_at_ns, stopped_at_ns)` contain `ts`?
    fn contains(&self, ts: i64) -> bool {
        ts >= self.started_at_ns && self.stopped_at_ns.is_none_or(|stop| ts < stop)
    }

    /// Time elapsed since stop once this window may be evicted: retention has
    /// passed *and* either every owed flush marker is in or
    /// [`FLUSH_MARKER_WAIT_CAP`] has expired. `None` while it must be kept.
    fn eviction_elapsed(&self, now: Instant, retention: Duration) -> Option<Duration> {
        let since_stop = self.stop_recv_at.map(|at| now.duration_since(at))?;
        let retained = since_stop >= retention;
        let flush_settled = !self.awaiting_flush
            || self.flush_markers_settled()
            || since_stop >= FLUSH_MARKER_WAIT_CAP;
        (retained && flush_settled).then_some(since_stop)
    }

    /// True once every producer with video here has sent its flush marker.
    fn flush_markers_settled(&self) -> bool {
        !self.flushed_producers.is_empty()
            && self.video_producers.is_subset(&self.flushed_producers)
    }
}

/// All windows currently tracked for one source: at most one live, plus
/// recently-closed windows retained until their late data has drained.
#[derive(Default)]
struct WindowsForSource {
    live: Option<ActiveWindow>,
    closing: Vec<ActiveWindow>,
    /// Daemon clock of the last envelope seen for this source — drives the
    /// idle reaper.
    last_seen: Option<Instant>,
}

impl WindowsForSource {
    /// Nothing live, nothing retained, quiet past `IDLE_REAP`: droppable.
    fn is_empty(&self, now: Instant) -> bool {
        self.live.is_none()
            && self.closing.is_empty()
            && self
                .last_seen
                .is_none_or(|at| now.duration_since(at) >= IDLE_REAP)
    }
}

/// One held data envelope awaiting its holdback release.
struct Held {
    source: Source,
    release_at: Instant,
    /// Producer publish time — the window-membership key, decided at release.
    publish_timestamp_ns: i64,
    payload: HeldPayload,
}

/// The data carried by a held envelope. `timestamp_ns` / `timestamp_s` here are
/// the data's *own* capture clock (content), never routing.
enum HeldPayload {
    Data {
        data_type: String,
        sensor_name: Option<String>,
        timestamp_ns: i64,
        timestamp_s: Option<f64>,
        payload: Vec<u8>,
    },
    Batch {
        data_type: String,
        timestamp_ns: i64,
        timestamp_s: Option<f64>,
        items: Vec<BatchedDataItem>,
    },
    Video {
        data_type: String,
        sensor_name: Option<String>,
        thread_id: i64,
        producer_pid: u32,
        width: u32,
        height: u32,
        byte_count: u64,
        frame_count: u32,
        frame_timestamps_s: Vec<f64>,
        dtype: FrameDtype,
        /// Per-frame publish time as µs after the chunk's open stamp
        frame_publish_offsets_us: Vec<u32>,
    },
    SourceFlushed {
        producer_pid: u32,
    },
    VideoProducerActive {
        producer_pid: u32,
    },
}

/// The dispatcher's task-local state.
struct Dispatcher {
    store: SqliteStateStore,
    actor_context: Arc<TraceActorContext>,
    context: DispatcherContext,
    holdback: Duration,
    /// Per-source window map.
    windows: HashMap<Source, WindowsForSource>,
    /// Holdback queue, monotonic in `release_at` (fixed offset + arrival
    /// order).
    held: VecDeque<Held>,
    /// Join handles for every spawned actor, awaited on shutdown.
    actor_handles: Vec<JoinHandle<()>>,
    /// Rate-limited orphan-drop counter (data outside any window).
    orphan_drops: u64,
    /// When the eviction + idle-reap scans last ran. Those scans are throttled
    /// to [`HOUSEKEEP_INTERVAL`] so a data stream arriving faster than that
    /// doesn't re-run the two full window scans (and their `Vec` allocations)
    /// on every inbound envelope — only the cheap holdback release does.
    last_housekeep: Instant,
}

impl Dispatcher {
    fn new(
        store: SqliteStateStore,
        actor_context: Arc<TraceActorContext>,
        context: DispatcherContext,
    ) -> Self {
        Self {
            store,
            actor_context,
            context,
            holdback: configured_holdback(),
            windows: HashMap::new(),
            held: VecDeque::new(),
            actor_handles: Vec::new(),
            orphan_drops: 0,
            last_housekeep: Instant::now(),
        }
    }

    async fn run(
        &mut self,
        mut rx: mpsc::Receiver<Envelope>,
        mut shutdown_rx: broadcast::Receiver<ShutdownSignal>,
    ) {
        tracing::info!(
            holdback_ms = self.holdback.as_millis(),
            "dispatcher started"
        );

        loop {
            // When there is in-flight work, poll frequently for due releases /
            // evictions; otherwise sleep until the next idle-reap horizon.
            let housekeep_after = if self.held.is_empty() && !self.any_closing() {
                IDLE_REAP
            } else {
                HOUSEKEEP_INTERVAL
            };

            tokio::select! {
                biased;
                signal = shutdown_rx.recv() => {
                    tracing::debug!(?signal, "dispatcher shutting down");
                    break;
                }
                envelope = rx.recv() => {
                    let Some(envelope) = envelope else {
                        tracing::debug!("dispatcher inbox closed; exiting");
                        break;
                    };
                    self.handle_inbound(envelope, Instant::now()).await;
                }
                _ = sleep(housekeep_after) => {}
            }

            // Holdback releases run on every wake-up; the housekeeping scans
            // are throttled to HOUSEKEEP_INTERVAL (see `last_housekeep`).
            let now = Instant::now();
            self.release_due_holdback(now).await;
            if now.duration_since(self.last_housekeep) >= HOUSEKEEP_INTERVAL {
                self.housekeep(now).await;
                self.last_housekeep = now;
            }
        }

        self.shutdown().await;
    }

    /// Apply one inbound envelope. Lifecycle events take effect immediately;
    /// data envelopes enter the holdback queue.
    async fn handle_inbound(&mut self, envelope: Envelope, recv_at: Instant) {
        match envelope {
            Envelope::StartRecording {
                robot_id,
                robot_instance,
                dataset_id,
                publish_timestamp_ns,
                timestamp_ns,
                cloud_recording_id,
                ..
            } => {
                self.handle_start(
                    (robot_id, robot_instance),
                    dataset_id,
                    cloud_recording_id,
                    publish_timestamp_ns,
                    timestamp_ns,
                    recv_at,
                )
                .await;
            }
            Envelope::StopRecording {
                robot_id,
                robot_instance,
                publish_timestamp_ns,
                timestamp_ns,
            } => {
                self.handle_stop(
                    (robot_id, robot_instance),
                    publish_timestamp_ns,
                    timestamp_ns,
                    recv_at,
                )
                .await;
            }
            Envelope::CancelRecording {
                robot_id,
                robot_instance,
                timestamp_ns,
            } => {
                self.handle_cancel((robot_id, robot_instance), timestamp_ns)
                    .await;
            }
            Envelope::DiscardRecording {
                recording_id,
                timestamp_ns,
            } => {
                self.handle_discard(&recording_id, timestamp_ns).await;
            }
            Envelope::Data {
                robot_id,
                robot_instance,
                data_type,
                sensor_name,
                publish_timestamp_ns,
                timestamp_ns,
                timestamp_s,
                payload,
            } => {
                let source = (robot_id, robot_instance);
                self.touch_source(&source, recv_at);
                self.held.push_back(Held {
                    source,
                    release_at: recv_at + self.holdback,
                    publish_timestamp_ns,
                    payload: HeldPayload::Data {
                        data_type,
                        sensor_name,
                        timestamp_ns,
                        timestamp_s,
                        payload,
                    },
                });
            }
            Envelope::BatchedData {
                robot_id,
                robot_instance,
                data_type,
                publish_timestamp_ns,
                timestamp_ns,
                timestamp_s,
                items,
            } => {
                let source = (robot_id, robot_instance);
                self.touch_source(&source, recv_at);
                self.held.push_back(Held {
                    source,
                    release_at: recv_at + self.holdback,
                    publish_timestamp_ns,
                    payload: HeldPayload::Batch {
                        data_type,
                        timestamp_ns,
                        timestamp_s,
                        items,
                    },
                });
            }
            Envelope::VideoChunkReady {
                robot_id,
                robot_instance,
                data_type,
                sensor_name,
                publish_timestamp_ns,
                thread_id,
                producer_pid,
                width,
                height,
                byte_count,
                frame_count,
                frame_timestamps_ns,
                frame_timestamps_s,
                dtype,
                frame_publish_offsets_us,
            } => {
                let source = (robot_id, robot_instance);
                self.touch_source(&source, recv_at);
                let _ = frame_timestamps_ns; // capture-clock content, not routing
                self.held.push_back(Held {
                    source,
                    release_at: recv_at + self.holdback,
                    publish_timestamp_ns,
                    payload: HeldPayload::Video {
                        data_type,
                        sensor_name,
                        thread_id,
                        producer_pid,
                        width,
                        height,
                        byte_count,
                        frame_count,
                        frame_timestamps_s,
                        dtype,
                        frame_publish_offsets_us,
                    },
                });
            }
            Envelope::SourceFlushed {
                robot_id,
                robot_instance,
                publish_timestamp_ns,
                producer_pid,
            } => {
                let source = (robot_id, robot_instance);
                self.touch_source(&source, recv_at);
                self.held.push_back(Held {
                    source,
                    release_at: recv_at + self.holdback,
                    publish_timestamp_ns,
                    payload: HeldPayload::SourceFlushed { producer_pid },
                });
            }
            Envelope::VideoProducerActive {
                robot_id,
                robot_instance,
                publish_timestamp_ns,
                producer_pid,
            } => {
                let source = (robot_id, robot_instance);
                self.touch_source(&source, recv_at);
                // Held for the same reason data is: the claim rides the
                // producer's port, `StartRecording` the calling thread's.
                self.held.push_back(Held {
                    source,
                    release_at: recv_at + self.holdback,
                    publish_timestamp_ns,
                    payload: HeldPayload::VideoProducerActive { producer_pid },
                });
            }
            Envelope::RefreshConfig {} => self.handle_refresh_config().await,
        }
    }

    /// Force the config watcher to re-resolve the profile and wait for it to
    /// finish before handling the next envelope, so the SDK's ordered
    /// `set_video_encoding_options → start_recording` sequence never races the
    /// async refresh (see [`crate::cloud::watchers::config_watcher`] for the
    /// full rationale). The wait is bounded by [`REFRESH_CONFIG_ACK_TIMEOUT`] so
    /// a stalled watcher can't wedge the routing loop. A missing sender (tests /
    /// no watcher) or a closed channel (watcher gone at shutdown) is a no-op.
    async fn handle_refresh_config(&self) {
        let Some(refresh_tx) = self.context.config_refresh_tx.as_ref() else {
            return;
        };
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        if refresh_tx.send(ack_tx).await.is_err() {
            tracing::debug!("config watcher gone; ignoring RefreshConfig");
            return;
        }
        if tokio::time::timeout(REFRESH_CONFIG_ACK_TIMEOUT, ack_rx)
            .await
            .is_err()
        {
            tracing::warn!("config refresh ack timed out; proceeding (poll will catch up)");
        }
    }

    fn touch_source(&mut self, source: &Source, recv_at: Instant) {
        // Hot path: probe with `get_mut` and only clone the key on insert.
        if let Some(window) = self.windows.get_mut(source) {
            window.last_seen = Some(recv_at);
        } else {
            self.windows.entry(source.clone()).or_default().last_seen = Some(recv_at);
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn handle_start(
        &mut self,
        source: Source,
        dataset_id: Option<String>,
        cloud_recording_id: Option<String>,
        publish_timestamp_ns: i64,
        timestamp_ns: i64,
        recv_at: Instant,
    ) {
        // A `StartRecording` carrying a known cloud id can reach this daemon
        // more than once for the very same recording: every local process
        // connected to the source learns about a web-started recording
        // independently, and each may try to open it. Only the first must
        // create a row and open a window.
        if let Some(cloud_recording_id) = cloud_recording_id.as_deref() {
            match self
                .store
                .recording_index_for_cloud_id(cloud_recording_id)
                .await
            {
                Ok(Some(existing_index)) => {
                    tracing::debug!(
                        recording_index = existing_index,
                        cloud_recording_id,
                        robot_id = source.0,
                        "recording already open for this cloud id; ignoring duplicate start"
                    );
                    return;
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(
                        %error,
                        cloud_recording_id,
                        robot_id = source.0,
                        "failed to check for an existing recording; proceeding"
                    );
                }
            }
        }

        // Insert the recording row synchronously: cloud notifiers react to the
        // `RecordingStarted` event by reading this row, and `cancel_recording`
        // burns it by index, so the row must exist before either runs. After the
        // create_trace burst was folded into the write-behind (the actors no
        // longer create rows here), this is a single uncontended write.
        //
        // The row's `start_timestamp_ns` is the caller's *capture* time (→
        // backend `start_time`); the window opens on the *publish* clock below.
        let new = NewRecording {
            robot_id: Some(&source.0),
            robot_instance: Some(source.1),
            dataset_id: dataset_id.as_deref(),
            start_timestamp_ns: timestamp_ns,
        };
        let recording_index = match self.store.create_recording(new).await {
            Ok(row) => row.recording_index,
            Err(error) => {
                tracing::warn!(%error, robot_id = source.0, "failed to create recording row");
                return;
            }
        };
        tracing::info!(recording_index, robot_id = source.0, "recording started");

        let entry = self.windows.entry(source).or_default();
        entry.last_seen = Some(recv_at);
        // An idle-reaped window sits in `closing` with an open upper bound
        // (`i64::MAX`) to catch stragglers; clamp any such window to this new
        // start so a restarted recording's data cannot be mis-routed into it
        // (`window_for_mut` checks `closing` before the live window).
        for closing in entry.closing.iter_mut() {
            let open_past_start = closing
                .stopped_at_ns
                .is_none_or(|stop| stop >= publish_timestamp_ns);
            if open_past_start {
                closing.stopped_at_ns = Some(publish_timestamp_ns);
                if closing.stop_recv_at.is_none() {
                    closing.stop_recv_at = Some(recv_at);
                }
            }
        }
        // A well-behaved producer stops before starting; if a live window is
        // somehow still open, retire it to `closing` bounded at the new start's
        // publish time so it stops catching data published after this point.
        // This also happens under an inverted start/stop pair (a slow stop
        // reaching the daemon after the next start), so persist the retired
        // recording's stop here — using the new start's publish time as the
        // exclusive upper bound of its membership range — rather than leaving
        // its row open. A later stolen stop refines the exact boundary.
        let mut retired: Option<i64> = None;
        if let Some(mut previous) = entry.live.take() {
            if previous.stopped_at_ns.is_none() {
                previous.stopped_at_ns = Some(publish_timestamp_ns);
                previous.stop_recv_at = Some(recv_at);
                retired = Some(previous.recording_index);
            }
            entry.closing.push(previous);
        }
        entry.live = Some(ActiveWindow {
            recording_index,
            started_at_ns: publish_timestamp_ns,
            stopped_at_ns: None,
            stop_recv_at: None,
            awaiting_flush: false,
            video_producers: HashSet::new(),
            flushed_producers: HashSet::new(),
            traces: HashMap::new(),
        });

        if let Some(retired_index) = retired {
            tracing::warn!(
                recording_index = retired_index,
                successor_index = recording_index,
                "start arrived with prior recording still live; closing it"
            );
            // Persist the retired recording's stop so it reaches a terminal,
            // notifiable state even if its own (late) stop never arrives.
            if let Err(error) = self
                .store
                .mark_recording_stopped(retired_index, publish_timestamp_ns)
                .await
            {
                tracing::warn!(%error, recording_index = retired_index, "failed to mark superseded recording stopped");
            }
        }

        match cloud_recording_id {
            None => {
                if let Some(bus) = self.context.event_bus.as_ref() {
                    bus.publish(DaemonEvent::RecordingStarted { recording_index });
                }
            }
            Some(cloud_recording_id) => {
                // The backend already created this id — skip the notifier's
                // POST and wake its waiters directly.
                if let Err(error) = self
                    .store
                    .mark_recording_start_notified(recording_index, &cloud_recording_id)
                    .await
                {
                    tracing::warn!(
                        %error,
                        recording_index,
                        cloud_recording_id,
                        "failed to persist recording id"
                    );
                }
                if let Some(bus) = self.context.event_bus.as_ref() {
                    bus.publish(DaemonEvent::RecordingCloudIdAssigned { recording_index });
                }
            }
        }
    }

    async fn handle_stop(
        &mut self,
        source: Source,
        publish_timestamp_ns: i64,
        timestamp_ns: i64,
        recv_at: Instant,
    ) {
        let Some(entry) = self.windows.get_mut(&source) else {
            tracing::debug!(robot_id = source.0, "stop for unknown source; ignoring");
            return;
        };
        entry.last_seen = Some(recv_at);

        // A stop whose publish time falls inside the live window closes the live
        // recording normally. A stop that predates the live window is a delayed
        // stop for a recording `handle_start` already retired (an inverted
        // start/stop pair) — it is matched against the closing windows instead.
        // Both paths converge on the shared post-persist tail below, so a
        // retired recording also becomes notifiable (fires `RecordingStopped`).
        let recording_index = if entry
            .live
            .as_ref()
            .is_some_and(|window| window.contains(publish_timestamp_ns))
        {
            let mut window = entry
                .live
                .take()
                .expect("live window was checked immediately above");
            // The window closes on the publish clock; the row's
            // `stop_timestamp_ns` (→ backend `end_time`) is the caller's capture
            // time.
            window.stopped_at_ns = Some(publish_timestamp_ns);
            window.stop_recv_at = Some(recv_at);
            window.awaiting_flush = true;
            let recording_index = window.recording_index;
            entry.closing.push(window);
            // Persist `stopped_at` before publishing the event: the cloud
            // stop-notifier reads this row on `RecordingStopped`, so the
            // timestamp must be on disk first.
            if let Err(error) = self
                .store
                .mark_recording_stopped(recording_index, timestamp_ns)
                .await
            {
                tracing::warn!(%error, recording_index, "failed to mark recording stopped");
                return;
            }
            recording_index
        } else if let Some(position) = entry
            .closing
            .iter()
            .rposition(|window| window.contains(publish_timestamp_ns))
        {
            let window = &mut entry.closing[position];
            let recording_index = window.recording_index;
            // Deliberately DO NOT narrow the window's in-memory `stopped_at_ns`
            // to this true (earlier) stop. `handle_start` set it to the
            // successor recording's start time so the two consecutive windows
            // tile the publish-clock line with no gap. Rewinding it to the real
            // stop would re-open a no-window interval `(true stop, successor
            // start)`; a tail video chunk whose NUT open time lands in that gap
            // — the writer is mid-flush at exactly that moment — would then find
            // no window and be dropped as an orphan. Only the retired
            // recording's own producer can stamp data in that interval, so
            // keeping the boundary at the successor start mis-routes nothing.
            //
            // Refresh the closing-retention deadline so the extra late tail data
            // this delayed stop implies still has a window to land in, and wait
            // on this stop's flush marker for the same reason.
            window.stop_recv_at = Some(recv_at);
            window.awaiting_flush = true;
            window.flushed_producers.clear();
            tracing::warn!(
                robot_id = source.0,
                recording_index,
                "stop arrived after a later recording started; refining the retired recording's stop"
            );
            // Refine the row's `stop_timestamp_ns` (→ backend `end_time`) to
            // this true capture stop. `handle_start` already marked the row with
            // the successor start's time, and `mark_recording_stopped` is
            // COALESCE-idempotent, so a plain re-mark would no-op — a forced
            // overwrite is required, and correct because `stop < successor
            // start`.
            if let Err(error) = self
                .store
                .refine_recording_stop(recording_index, timestamp_ns)
                .await
            {
                tracing::warn!(%error, recording_index, "failed to refine retired recording stop");
                return;
            }
            recording_index
        } else {
            tracing::debug!(
                robot_id = source.0,
                publish_timestamp_ns,
                "stop does not belong to any retained recording window; ignoring"
            );
            return;
        };

        // Shared post-persist tail. A retired recording never fired
        // `RecordingStopped` at retire time (only its row was marked), so this
        // is where it — like a normally-closed recording — becomes notifiable.
        tracing::info!(recording_index, "recording stopped");
        crate::perf_events::emit(
            "holdback",
            "started",
            Some(recording_index),
            None,
            None,
            serde_json::json!({
                "configured_holdback_ms": self.holdback.as_secs_f64() * 1_000.0,
                "closing_retention_ms": (self.holdback * 2).as_secs_f64() * 1_000.0,
            }),
        );
        if let Some(bus) = self.context.event_bus.as_ref() {
            bus.publish(DaemonEvent::RecordingStopped { recording_index });
        }
    }

    async fn handle_cancel(&mut self, source: Source, timestamp_ns: i64) {
        let Some(mut entry) = self.windows.remove(&source) else {
            return;
        };
        // Drop any held data for this source — a cancelled recording's data
        // must never reach an actor.
        self.held.retain(|held| held.source != source);

        let mut windows: Vec<ActiveWindow> = Vec::new();
        if let Some(live) = entry.live.take() {
            windows.push(live);
        }
        windows.append(&mut entry.closing);

        for window in windows {
            self.discard_window(window, timestamp_ns).await;
        }
    }

    /// Relay of a backend `DISCARDED`, keyed by the cloud `recording_id`.
    ///
    /// Unlike [`handle_cancel`](Self::handle_cancel) this routinely arrives for
    /// a recording with no window left — stopped a while ago, still uploading —
    /// and then burning the rows and announcing it is the whole job. Idempotent,
    /// and a no-op for an id this daemon never held (the notification is
    /// broadcast org-wide).
    async fn handle_discard(&mut self, recording_id: &str, timestamp_ns: i64) {
        let recording_index = match self.store.recording_index_for_cloud_id(recording_id).await {
            Ok(Some(index)) => index,
            Ok(None) => {
                tracing::debug!(
                    recording_id,
                    "discard notification for a recording this daemon does not hold; ignoring"
                );
                return;
            }
            Err(error) => {
                tracing::warn!(%error, recording_id, "failed to resolve discarded recording");
                return;
            }
        };
        match self.take_window(recording_index) {
            // Still open: tear it down or its trace actors keep writing.
            Some((source, window)) => {
                self.held.retain(|held| held.source != source);
                self.discard_window(window, timestamp_ns).await;
            }
            None => self.burn_recording(recording_index, timestamp_ns).await,
        }
    }

    /// Remove the window owning `recording_index` from whichever source holds
    /// it — a discard is keyed by cloud id, so the source is not known up
    /// front. `None` once no window is left, the common case for a discard.
    fn take_window(&mut self, recording_index: i64) -> Option<(Source, ActiveWindow)> {
        for (source, entry) in self.windows.iter_mut() {
            if entry
                .live
                .as_ref()
                .is_some_and(|window| window.recording_index == recording_index)
            {
                return entry.live.take().map(|window| (source.clone(), window));
            }
            if let Some(position) = entry
                .closing
                .iter()
                .position(|window| window.recording_index == recording_index)
            {
                return Some((source.clone(), entry.closing.remove(position)));
            }
        }
        None
    }

    /// Tear one window down: stop its trace actors, then burn the recording.
    async fn discard_window(&mut self, window: ActiveWindow, timestamp_ns: i64) {
        let recording_index = window.recording_index;
        for (_, handle) in window.traces {
            let _ = handle.sender.send(TraceActorMessage::Cancel).await;
        }
        self.burn_recording(recording_index, timestamp_ns).await;
    }

    /// Stamp a recording cancelled — burning every non-terminal trace row —
    /// and announce it so the uploader abandons its work. Idempotent.
    async fn burn_recording(&mut self, recording_index: i64, timestamp_ns: i64) {
        // Purge unflushed trace creates *before* burning the rows: a create
        // still in the write-behind batcher would otherwise commit afterwards
        // as a non-terminal row and be uploaded for a recording that is gone.
        self.actor_context
            .trace_writer
            .drop_recording(recording_index)
            .await;
        // The cancel's capture timestamp becomes the row's
        // `stop_timestamp_ns` (→ backend `end_time`), exactly as a stop.
        match self
            .store
            .cancel_recording(recording_index, timestamp_ns)
            .await
        {
            Ok((_, touched)) => {
                tracing::info!(
                    recording_index,
                    trace_rows_touched = touched,
                    "recording cancelled"
                );
                if let Some(bus) = self.context.event_bus.as_ref() {
                    bus.publish(DaemonEvent::RecordingCancelled { recording_index });
                }
            }
            Err(error) => {
                tracing::warn!(%error, recording_index, "failed to mark recording cancelled");
            }
        }
    }

    /// True when any source has a retained closing window.
    fn any_closing(&self) -> bool {
        self.windows.values().any(|entry| !entry.closing.is_empty())
    }

    /// Release every held envelope whose hold has elapsed. Cheap — pops only
    /// what is due — and runs on every dispatcher wake-up. Kept strictly ahead
    /// of [`housekeep`](Self::housekeep)'s evictions so a datum releasing in
    /// this tick still finds its (possibly closing) window.
    async fn release_due_holdback(&mut self, now: Instant) {
        while self.held.front().is_some_and(|held| held.release_at <= now) {
            let held = self.held.pop_front().expect("front checked");
            self.route(held).await;
        }
    }

    /// Evict windows past their retention and force-close idle sources. Two full
    /// window scans, so throttled to [`HOUSEKEEP_INTERVAL`] by the caller rather
    /// than run per inbound envelope.
    async fn housekeep(&mut self, now: Instant) {
        let retention = self.holdback * 2;
        let (closing_actors, empty_sources) = self.evict_closing_windows(now, retention);

        for handle in closing_actors {
            let _ = handle.sender.send(TraceActorMessage::WindowClosing).await;
        }
        for source in empty_sources {
            self.windows.remove(&source);
        }

        // Idle reaper: force-close a live window whose source has gone
        // silent (producer crashed without a Stop).
        self.reap_idle(now).await;
    }

    /// Evict every closing window past retention and collect the sources left
    /// with nothing to track. Returns the evicted windows' trace actors, still
    /// owed a `WindowClosing`, and the now-empty source keys.
    fn evict_closing_windows(
        &mut self,
        now: Instant,
        retention: Duration,
    ) -> (Vec<TraceHandle>, Vec<Source>) {
        let mut closing_actors: Vec<TraceHandle> = Vec::new();
        let mut empty_sources: Vec<Source> = Vec::new();
        for (source, entry) in self.windows.iter_mut() {
            entry.closing.retain_mut(|window| {
                let Some(elapsed) = window.eviction_elapsed(now, retention) else {
                    return true;
                };
                Self::finish_eviction(source, window, elapsed, retention, &mut closing_actors);
                false
            });
            if entry.is_empty(now) {
                empty_sources.push(source.clone());
            }
        }
        (closing_actors, empty_sources)
    }

    /// Emit one evicted window's completion event and drain its trace actors
    /// into `closing_actors`.
    fn finish_eviction(
        source: &Source,
        window: &mut ActiveWindow,
        elapsed: Duration,
        retention: Duration,
        closing_actors: &mut Vec<TraceHandle>,
    ) {
        if window.awaiting_flush && !window.flush_markers_settled() {
            tracing::warn!(
                recording_index = window.recording_index,
                robot_id = source.0,
                elapsed_s = elapsed.as_secs_f64(),
                outstanding_producers = window
                    .video_producers
                    .difference(&window.flushed_producers)
                    .count(),
                "no producer flush marker before the cap; retiring the \
                 window anyway — any tail video chunk still in flight \
                 will be dropped as an orphan"
            );
        }
        crate::perf_events::emit(
            "holdback",
            "completed",
            Some(window.recording_index),
            None,
            Some(elapsed),
            serde_json::json!({
                "trace_actor_count": window.traces.len(),
                "configured_retention_ms": retention.as_secs_f64() * 1_000.0,
                "awaited_flush_marker": window.awaiting_flush,
                "flush_marker_seen": window.flush_markers_settled(),
            }),
        );
        for (_, handle) in window.traces.drain() {
            closing_actors.push(handle);
        }
    }

    /// Force-close any live window whose source has been silent past
    /// [`IDLE_REAP`], giving it an open upper bound (`i64::MAX`) so any
    /// straggler data still routes to it before eviction; the row's capture
    /// stop time is the reap moment, so the recording reaches a terminal,
    /// notifiable state.
    async fn reap_idle(&mut self, now: Instant) {
        let stale: Vec<Source> = self
            .windows
            .iter()
            .filter(|(_, entry)| {
                entry.live.is_some()
                    && entry
                        .last_seen
                        .is_some_and(|at| now.duration_since(at) >= IDLE_REAP)
            })
            .map(|(source, _)| source.clone())
            .collect();
        for source in stale {
            tracing::warn!(
                robot_id = source.0,
                "source idle past reap horizon; force-closing window"
            );
            let Some(entry) = self.windows.get_mut(&source) else {
                continue;
            };
            let Some(mut window) = entry.live.take() else {
                continue;
            };
            // The producer crashed without a Stop, so there is no next
            // recording to partition against — keep the window's publish upper
            // bound open (`i64::MAX`) to catch any straggler data before
            // eviction. The row's capture stop time (→ backend `end_time`) is
            // the reap moment, so the backend reports a finite end rather than
            // the year-2262 the `i64::MAX` window sentinel would imply.
            window.stopped_at_ns = Some(i64::MAX);
            window.stop_recv_at = Some(now);
            let recording_index = window.recording_index;
            entry.closing.push(window);
            let stop_capture_ns = Utc::now().timestamp_nanos_opt().unwrap_or(i64::MAX);
            if let Err(error) = self
                .store
                .mark_recording_stopped(recording_index, stop_capture_ns)
                .await
            {
                tracing::warn!(%error, recording_index, "failed to mark idle recording stopped");
            } else if let Some(bus) = self.context.event_bus.as_ref() {
                bus.publish(DaemonEvent::RecordingStopped { recording_index });
            }
        }
    }

    /// Route one released held envelope into its window's actors, using its
    /// `publish_timestamp_ns` as the membership key.
    async fn route(&mut self, held: Held) {
        let publish_ts = held.publish_timestamp_ns;
        match held.payload {
            HeldPayload::Data {
                data_type,
                sensor_name,
                timestamp_ns,
                timestamp_s,
                payload,
            } => {
                self.route_data(
                    &held.source,
                    publish_ts,
                    data_type,
                    sensor_name,
                    timestamp_ns,
                    timestamp_s,
                    payload,
                )
                .await;
            }
            HeldPayload::Batch {
                data_type,
                timestamp_ns,
                timestamp_s,
                items,
            } => {
                for item in items {
                    self.route_data(
                        &held.source,
                        publish_ts,
                        data_type.clone(),
                        item.sensor_name,
                        timestamp_ns,
                        timestamp_s,
                        item.payload,
                    )
                    .await;
                }
            }
            HeldPayload::Video {
                data_type,
                sensor_name,
                thread_id,
                producer_pid,
                width,
                height,
                byte_count,
                frame_count,
                frame_timestamps_s,
                dtype,
                frame_publish_offsets_us,
            } => {
                self.route_video(
                    &held.source,
                    publish_ts,
                    data_type,
                    sensor_name,
                    thread_id,
                    producer_pid,
                    width,
                    height,
                    byte_count,
                    frame_count,
                    frame_timestamps_s,
                    dtype,
                    frame_publish_offsets_us,
                )
                .await;
            }
            HeldPayload::SourceFlushed { producer_pid } => {
                self.mark_source_flushed(&held.source, producer_pid)
            }
            HeldPayload::VideoProducerActive { producer_pid } => {
                self.note_video_producer(&held.source, publish_ts, producer_pid)
            }
        }
    }

    /// Attribute the window containing `publish_ts` to `producer_pid`, before
    /// any of its video has sealed into a chunk.
    fn note_video_producer(&mut self, source: &Source, publish_ts: i64, producer_pid: u32) {
        let Some(entry) = self.windows.get_mut(source) else {
            return;
        };
        let Some(window) = Self::window_for_mut(entry, publish_ts) else {
            return;
        };
        if window.video_producers.insert(producer_pid) {
            tracing::debug!(
                recording_index = window.recording_index,
                producer_pid,
                "producer claimed video for window; its own flush marker is now \
                 required before the window can retire"
            );
        }
    }

    /// Credit one producer's drained flush barrier to *every* closing window
    /// still owed a marker from that `producer_pid`.
    ///
    /// Every window, not the oldest one: a barrier drains everything the
    /// producer holds for the source, so one marker speaks for every window its
    /// chunks reached. A process that publishes no stop of its own flushes once,
    /// at its own exit — and one of its chunks can carry several recordings —
    /// so crediting a single window would leave the others waiting out
    /// [`FLUSH_MARKER_WAIT_CAP`] for a marker that is never coming.
    ///
    /// Safe for the owner-driven case too, where each stop produces its own
    /// marker: a window already credited is skipped, and a window whose chunks
    /// this producer never reached is not owed one (it is not in
    /// `video_producers`, so it is not waiting).
    fn mark_source_flushed(&mut self, source: &Source, producer_pid: u32) {
        let Some(entry) = self.windows.get_mut(source) else {
            return;
        };
        for window in entry.closing.iter_mut().filter(|window| {
            window.awaiting_flush && !window.flushed_producers.contains(&producer_pid)
        }) {
            window.flushed_producers.insert(producer_pid);
            tracing::debug!(
                recording_index = window.recording_index,
                producer_pid,
                "producer flush barrier drained; window may retire once every \
                 video-contributing producer has reported"
            );
        }
    }

    /// Find the window for `source` containing `ts`. Closing windows are
    /// bounded on both sides and are checked first (newest-first); the live
    /// window is an unbounded-above catch-all, so it must be the last resort or
    /// it would steal data belonging to a just-closed window.
    fn window_for_mut(entry: &mut WindowsForSource, ts: i64) -> Option<&mut ActiveWindow> {
        if let Some(pos) = entry.closing.iter().rposition(|window| window.contains(ts)) {
            return entry.closing.get_mut(pos);
        }
        if entry
            .live
            .as_ref()
            .is_some_and(|window| window.contains(ts))
        {
            return entry.live.as_mut();
        }
        None
    }

    /// Borrow the window a [`ChunkClaim`] addresses.
    fn window_at_mut(entry: &mut WindowsForSource, slot: WindowSlot) -> Option<&mut ActiveWindow> {
        match slot {
            WindowSlot::Live => entry.live.as_mut(),
            WindowSlot::Closing(index) => entry.closing.get_mut(index),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn route_data(
        &mut self,
        source: &Source,
        publish_ts: i64,
        data_type: String,
        sensor_name: Option<String>,
        timestamp_ns: i64,
        timestamp_s: Option<f64>,
        payload: Vec<u8>,
    ) {
        let Some(entry) = self.windows.get_mut(source) else {
            self.note_orphan();
            return;
        };
        let Some(window) = Self::window_for_mut(entry, publish_ts) else {
            self.note_orphan();
            return;
        };
        let sender = Self::ensure_actor(
            window,
            &self.actor_context,
            data_type,
            sensor_name,
            &mut self.actor_handles,
        )
        .sender
        .clone();
        if sender
            .send(TraceActorMessage::Data {
                timestamp_ns,
                timestamp_s,
                payload,
            })
            .await
            .is_err()
        {
            tracing::warn!("trace actor inbox closed; dropping data");
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn route_video(
        &mut self,
        source: &Source,
        publish_ts: i64,
        data_type: String,
        sensor_name: Option<String>,
        thread_id: i64,
        producer_pid: u32,
        width: u32,
        height: u32,
        byte_count: u64,
        frame_count: u32,
        frame_timestamps_s: Vec<f64>,
        dtype: FrameDtype,
        frame_publish_offsets_us: Vec<u32>,
    ) {
        let recordings_root = self.actor_context.recordings_root.clone();
        // The chunk's `publish_timestamp_ns` (its open time) keys both the
        // spool filename and the window routing below.
        let spool_nut = paths::spool_chunk_path(
            recordings_root.as_path(),
            &source.0,
            source.1,
            &data_type,
            sensor_name.as_deref(),
            publish_ts,
            thread_id,
        );

        let Some(entry) = self.windows.get_mut(source) else {
            remove_spool_nut(&spool_nut);
            self.note_orphan();
            return;
        };

        // Every window any of this chunk's frames belongs to, and which frames.
        // A chunk is a container, not a member: the producer seals it on its own
        // caps and knows nothing of windows, so one chunk can hold frames from
        // before a recording, inside it, and inside the next one.
        let claims = claims_for_chunk(entry, publish_ts, &frame_publish_offsets_us, frame_count);
        if claims.is_empty() {
            tracing::debug!(
                frame_count,
                "video chunk has no frame published inside any window; dropping it"
            );
            remove_spool_nut(&spool_nut);
            self.note_orphan();
            return;
        }

        // One NUT can feed several recordings, so each claim needs its own copy
        // of the source to relink and unlink. Hard links keep that to a metadata
        // op and one inode; the last claim consumes the spool file itself.
        let sources = claim_sources(&spool_nut, claims.len());

        for (claim, source_nut) in claims.iter().zip(sources) {
            let Some(source_nut) = source_nut else {
                // Could not give this claim its own name; the others still route.
                continue;
            };
            let Some(window) = Self::window_at_mut(entry, claim.slot) else {
                remove_spool_nut(&source_nut);
                continue;
            };
            window.video_producers.insert(producer_pid);
            if claim.skip > 0 || claim.count < frame_count {
                tracing::debug!(
                    recording_index = window.recording_index,
                    skipped = claim.skip,
                    kept = claim.count,
                    frame_count,
                    "video chunk spans this window's boundary; keeping the frames \
                     published inside it"
                );
            }
            let claimed_timestamps = claim.timestamps(&frame_timestamps_s);

            let recording_index = window.recording_index;
            let handle = Self::ensure_actor(
                window,
                &self.actor_context,
                data_type.clone(),
                sensor_name.clone(),
                &mut self.actor_handles,
            );
            let chunk_index = handle.next_video_chunk;
            handle.next_video_chunk = handle.next_video_chunk.saturating_add(1);
            let sender = handle.sender.clone();

            // The actor relinks the spooled NUT into the recording itself — on a
            // blocking thread inside its background encode task — so the
            // rename's possible journal-commit stall never lands on this routing
            // path. The dispatcher only hands over the source spool path.
            if sender
                .send(TraceActorMessage::Video {
                    chunk_index,
                    spool_nut: source_nut.clone(),
                    width,
                    height,
                    byte_count,
                    frame_count: claim.count,
                    skip_frames: claim.skip,
                    frame_timestamps_s: claimed_timestamps,
                    dtype,
                })
                .await
                .is_err()
            {
                tracing::warn!(
                    recording_index,
                    "video trace actor inbox closed; dropping chunk"
                );
                remove_spool_nut(&source_nut);
            }
        }
    }

    /// Look up or spawn the per-trace actor for `(window, data_type,
    /// sensor_name)`, returning its routing handle.
    fn ensure_actor<'a>(
        window: &'a mut ActiveWindow,
        actor_context: &Arc<TraceActorContext>,
        data_type: String,
        sensor_name: Option<String>,
        actor_handles: &mut Vec<JoinHandle<()>>,
    ) -> &'a mut TraceHandle {
        let key = TraceKey {
            recording_index: window.recording_index,
            data_type,
            sensor_name,
        };
        window.traces.entry(key.clone()).or_insert_with(|| {
            let identity = TraceIdentity {
                trace_id: Uuid::new_v4().to_string(),
                key,
            };
            let (tx, actor_rx) = mpsc::channel(TRACE_QUEUE_CAPACITY);
            let actor_context = Arc::clone(actor_context);
            let join = tokio::spawn(async move {
                trace_actor::run(actor_context, identity, actor_rx).await;
            });
            actor_handles.push(join);
            TraceHandle {
                sender: tx,
                next_video_chunk: 0,
            }
        })
    }

    fn note_orphan(&mut self) {
        self.orphan_drops = self.orphan_drops.saturating_add(1);
        if self.orphan_drops == 1 || self.orphan_drops.is_multiple_of(1024) {
            tracing::warn!(
                dropped = self.orphan_drops,
                "dropped datum outside any recording window"
            );
        }
    }

    /// Clean shutdown: flush every held datum against the current windows, then
    /// signal `WindowClosing` to every actor so in-flight recordings finalise.
    async fn shutdown(&mut self) {
        let held: Vec<Held> = self.held.drain(..).collect();
        for item in held {
            self.route(item).await;
        }
        let windows = std::mem::take(&mut self.windows);
        for (_, mut entry) in windows {
            let mut all: Vec<ActiveWindow> = Vec::new();
            if let Some(live) = entry.live.take() {
                all.push(live);
            }
            all.append(&mut entry.closing);
            for window in all {
                for (_, handle) in window.traces {
                    let _ = handle.sender.send(TraceActorMessage::WindowClosing).await;
                }
            }
        }
        let handles = std::mem::take(&mut self.actor_handles);
        for handle in handles {
            if let Err(error) = handle.await {
                tracing::warn!(?error, "trace actor join failed during shutdown");
            }
        }
        // Every actor has exited, so all their finalise/failed writes are now
        // queued in the write-behind. Flush it here so that by the time
        // `DispatcherHandle::shutdown` returns the trace rows are durable —
        // callers (and tests) can read final state without a separate barrier.
        self.actor_context.trace_writer.flush().await;
        tracing::info!("dispatcher stopped");
    }
}

/// Addresses one window inside a [`WindowsForSource`] without borrowing it, so
/// a chunk's claims can be resolved in one immutable pass and acted on in a
/// second, mutable one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowSlot {
    Live,
    Closing(usize),
}

/// One window's claim on a chunk: which window, and which run of frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChunkClaim {
    slot: WindowSlot,
    /// Leading frames this window does not own (published before it opened).
    skip: u32,
    /// Frames it does own, counted from `skip`.
    count: u32,
}

impl ChunkClaim {
    /// This claim's slice of the chunk's per-frame capture timestamps.
    fn timestamps(&self, frame_timestamps_s: &[f64]) -> Vec<f64> {
        let start = (self.skip as usize).min(frame_timestamps_s.len());
        let end = start
            .saturating_add(self.count as usize)
            .min(frame_timestamps_s.len());
        frame_timestamps_s[start..end].to_vec()
    }
}

/// The run of a chunk's frames that falls inside `[started_at_ns,
/// stopped_at_ns)`, as `(skip, count)`.
///
/// Both ends are cut, and for the same reason: the producer seals chunks on its
/// own caps, so a boundary can fall anywhere inside one. Membership by the
/// chunk's open stamp would give a window either every frame or none of them —
/// and none is what a chunk opened before the window got, taking its in-window
/// frames with it.
fn frame_range_in_window(
    offsets_us: &[u32],
    chunk_open_ns: i64,
    started_at_ns: i64,
    stopped_at_ns: Option<i64>,
    frame_count: u32,
) -> (u32, u32) {
    if offsets_us.is_empty() {
        // No per-frame data to cut on, so the open stamp is all there is: an
        // all-or-nothing decision for the whole chunk.
        let inside =
            chunk_open_ns >= started_at_ns && stopped_at_ns.is_none_or(|stop| chunk_open_ns < stop);
        return if inside { (0, frame_count) } else { (0, 0) };
    }
    let bounded = |bound_ns| {
        let index = video_boundary::frames_before_boundary(offsets_us, chunk_open_ns, bound_ns);
        u32::try_from(index).unwrap_or(u32::MAX).min(frame_count)
    };
    let skip = bounded(started_at_ns);
    let end = stopped_at_ns.map_or(frame_count, bounded);
    (skip, end.saturating_sub(skip))
}

/// Resolve which of a source's windows own which of a chunk's frames.
///
/// Windows never overlap on the publish clock (a new start clamps any window
/// still closing), so the claims are disjoint, and they come out oldest-first:
/// closing windows in order, then the live one.
fn claims_for_chunk(
    entry: &WindowsForSource,
    chunk_open_ns: i64,
    offsets_us: &[u32],
    frame_count: u32,
) -> Vec<ChunkClaim> {
    let candidates = entry
        .closing
        .iter()
        .enumerate()
        .map(|(index, window)| (WindowSlot::Closing(index), window))
        .chain(entry.live.as_ref().map(|window| (WindowSlot::Live, window)));
    candidates
        .filter_map(|(slot, window)| {
            let (skip, count) = frame_range_in_window(
                offsets_us,
                chunk_open_ns,
                window.started_at_ns,
                window.stopped_at_ns,
                frame_count,
            );
            (count > 0).then_some(ChunkClaim { slot, skip, count })
        })
        .collect()
}

/// One source path per claim, so each can be relinked and unlinked
/// independently.
///
/// The last claim takes the spool file itself; earlier ones get a hard link to
/// it, which is a metadata op against one inode rather than a copy of the
/// pixels. `None` for a claim whose link could not be made — its frames are lost
/// but the other claims still route.
fn claim_sources(spool_nut: &std::path::Path, claims: usize) -> Vec<Option<std::path::PathBuf>> {
    (0..claims)
        .map(|index| {
            if index + 1 == claims {
                return Some(spool_nut.to_path_buf());
            }
            let link = claim_link_path(spool_nut, index);
            match std::fs::hard_link(spool_nut, &link) {
                Ok(()) => Some(link),
                Err(error) => {
                    // Fall back to a copy: a spool on a filesystem without hard
                    // links still splits, just not for free.
                    match std::fs::copy(spool_nut, &link) {
                        Ok(_) => Some(link),
                        Err(copy_error) => {
                            tracing::warn!(
                                %error,
                                %copy_error,
                                path = %link.display(),
                                "could not give a chunk claim its own source NUT; \
                                 dropping that recording's share of the chunk"
                            );
                            None
                        }
                    }
                }
            }
        })
        .collect()
}

/// Sibling path for one claim's hard link to a shared spool NUT.
fn claim_link_path(spool_nut: &std::path::Path, index: usize) -> std::path::PathBuf {
    let mut name = spool_nut.as_os_str().to_os_string();
    name.push(format!(".claim{index}.nut"));
    std::path::PathBuf::from(name)
}

fn remove_spool_nut(path: &std::path::Path) {
    if let Err(error) = std::fs::remove_file(path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            tracing::debug!(%error, path = %path.display(), "failed to remove orphan spool NUT");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud::ConfigRefreshRequest;
    use crate::encoding::video_encoder::VideoEncoder;
    use crate::state::{SqliteStateStore, TraceWriteStatus};
    use crate::storage::budget::{StorageBudget, StoragePolicy};
    use crate::storage::paths::TracePath;
    use std::path::PathBuf;
    use tempfile::TempDir;
    use tokio::sync::{broadcast, mpsc};
    use tokio::time::{timeout, Duration};

    /// A live window for a source, as `handle_start` would have built it.
    fn test_window(recording_index: i64, started_at_ns: i64) -> ActiveWindow {
        ActiveWindow {
            recording_index,
            started_at_ns,
            stopped_at_ns: None,
            stop_recv_at: None,
            awaiting_flush: false,
            video_producers: HashSet::new(),
            flushed_producers: HashSet::new(),
            traces: HashMap::new(),
        }
    }

    async fn open_store() -> (SqliteStateStore, TempDir) {
        let dir = TempDir::new().expect("tempdir");
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .expect("open store");
        (store, dir)
    }

    fn test_context(recordings_root: PathBuf, store: SqliteStateStore) -> Arc<TraceActorContext> {
        let policy = StoragePolicy {
            storage_limit_bytes: None,
            min_free_disk_bytes: 0,
            refresh_interval: Duration::from_secs(60),
        };
        let budget = Arc::new(StorageBudget::new(&recordings_root, policy));
        // The writer owner is dropped: the spawned task lives while the handle
        // inside the context does. The dispatcher flushes it on shutdown, so
        // tests see durable trace state after `handle.shutdown().await`.
        let (trace_writer, _writer_owner) =
            crate::state::trace_event_database_writer::spawn(Arc::new(store));
        let (json_writer, _json_owner) = crate::pipeline::json_writer::spawn();
        Arc::new(TraceActorContext::new(
            recordings_root,
            budget,
            VideoEncoder::new(),
            trace_writer,
            json_writer,
        ))
    }

    // Tests exercise window membership, which is keyed on the publish clock, so
    // the helper sets the capture `timestamp_ns` to the same value.
    fn start(robot: &str, publish_timestamp_ns: i64) -> Envelope {
        Envelope::StartRecording {
            robot_id: robot.into(),
            robot_instance: 0,
            robot_name: None,
            dataset_id: None,
            dataset_name: None,
            publish_timestamp_ns,
            timestamp_ns: publish_timestamp_ns,
            cloud_recording_id: None,
        }
    }

    fn stop(robot: &str, publish_timestamp_ns: i64) -> Envelope {
        Envelope::StopRecording {
            robot_id: robot.into(),
            robot_instance: 0,
            publish_timestamp_ns,
            timestamp_ns: publish_timestamp_ns,
        }
    }

    /// The producer's end-of-barrier marker: no more tail chunks are coming for
    /// this source's just-stopped window.
    fn source_flushed(robot: &str, publish_timestamp_ns: i64, producer_pid: u32) -> Envelope {
        Envelope::SourceFlushed {
            robot_id: robot.into(),
            robot_instance: 0,
            publish_timestamp_ns,
            producer_pid,
        }
    }

    /// A datum published at `publish_ts` with `content_ts` as its own
    /// (decoupled) capture timestamp.
    fn datum_full(robot: &str, publish_ts: i64, content_ts: i64, value: i64) -> Envelope {
        Envelope::Data {
            robot_id: robot.into(),
            robot_instance: 0,
            data_type: "joints".into(),
            sensor_name: Some("waist".into()),
            publish_timestamp_ns: publish_ts,
            timestamp_ns: content_ts,
            timestamp_s: None,
            payload: serde_json::to_vec(&serde_json::json!({ "i": value })).unwrap(),
        }
    }

    /// A datum whose publish time and capture time coincide.
    fn datum(robot: &str, publish_ts: i64, value: i64) -> Envelope {
        datum_full(robot, publish_ts, publish_ts, value)
    }

    /// A short holdback keeps the tests fast.
    fn fast_holdback() {
        std::env::set_var(HOLDBACK_ENV, "60");
    }

    #[tokio::test]
    async fn refresh_config_forwards_to_watcher_and_awaits_ack() {
        // The RefreshConfig arm must hand a refresh request to the config
        // watcher and await its ack. Because the commands channel is in-order
        // and the dispatcher processes envelopes sequentially, awaiting here is
        // what guarantees the in-memory config is updated before a following
        // StartRecording / VideoChunkReady resolves its codec. A stand-in
        // watcher acks the request; the test proves the request is delivered and
        // the dispatcher keeps running after the ack.
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let (refresh_tx, mut refresh_rx) = mpsc::channel::<ConfigRefreshRequest>(4);
        let dispatcher_context = DispatcherContext {
            event_bus: None,
            config_refresh_tx: Some(refresh_tx),
        };
        let (tx, handle) =
            spawn_with_context(store.clone(), context, dispatcher_context, shutdown_rx);

        // Stand-in watcher: ack the first refresh request it receives.
        let watcher = tokio::spawn(async move {
            match refresh_rx.recv().await {
                Some(ack) => ack.send(()).is_ok(),
                None => false,
            }
        });

        tx.send(Envelope::RefreshConfig {}).await.unwrap();
        let acked = timeout(Duration::from_secs(5), watcher)
            .await
            .expect("watcher observed the refresh within 5s")
            .expect("watcher task joined");
        assert!(
            acked,
            "dispatcher must forward RefreshConfig to the watcher"
        );

        drop(tx);
        timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");
    }

    #[tokio::test]
    async fn refresh_config_without_watcher_is_noop() {
        // With no `config_refresh_tx` wired (tests / no watcher) a RefreshConfig
        // must be a harmless no-op: the dispatcher neither blocks nor dies, and
        // keeps routing afterwards.
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let dispatcher_context = DispatcherContext {
            event_bus: None,
            config_refresh_tx: None,
        };
        let (tx, handle) =
            spawn_with_context(store.clone(), context, dispatcher_context, shutdown_rx);

        timeout(Duration::from_secs(5), tx.send(Envelope::RefreshConfig {}))
            .await
            .expect("send did not block")
            .expect("dispatcher accepted the envelope");

        drop(tx);
        timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");
    }

    #[tokio::test]
    async fn routes_data_into_its_window_by_timestamp() {
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let bus = crate::state::EventBus::new();
        let dispatcher_context = DispatcherContext {
            event_bus: Some(bus.clone()),
            config_refresh_tx: None,
        };
        let (tx, handle) = spawn_with_context(
            store.clone(),
            context.clone(),
            dispatcher_context,
            shutdown_rx,
        );

        tx.send(start("robot-1", 100)).await.unwrap();
        for index in 0..3i64 {
            tx.send(datum("robot-1", 100 + index, index)).await.unwrap();
        }
        tx.send(stop("robot-1", 200)).await.unwrap();

        drop(tx);
        timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        // Exactly one recording (index 1) with one written trace.
        let recordings = store.recordings_for_source("robot-1", 0).await.unwrap();
        assert_eq!(recordings.len(), 1);
        let recording_index = recordings[0].recording_index;
        let traces = store
            .list_traces_for_recording(recording_index)
            .await
            .unwrap();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].write_status, TraceWriteStatus::Written);

        let trace_dir = TracePath::new(
            recording_index.to_string(),
            "joints",
            traces[0].trace_id.clone(),
        )
        .directory(context.recordings_root.as_path());
        let bytes = std::fs::read(trace_dir.join("trace.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed, serde_json::json!([{"i": 0}, {"i": 1}, {"i": 2}]));
    }

    #[tokio::test]
    async fn back_to_back_recordings_route_by_publish_timestamp() {
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let (tx, handle) = spawn(store.clone(), context.clone(), shutdown_rx);

        // Recording A: [100, 200). Recording B: [200, 300).
        tx.send(start("robot-1", 100)).await.unwrap();
        tx.send(stop("robot-1", 200)).await.unwrap();
        tx.send(start("robot-1", 200)).await.unwrap();
        tx.send(stop("robot-1", 300)).await.unwrap();
        // A datum published inside A's window but delivered after B opened
        // still lands in A by its publish timestamp.
        tx.send(datum("robot-1", 150, 1)).await.unwrap();
        tx.send(datum("robot-1", 250, 2)).await.unwrap();

        drop(tx);
        timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        let recordings = store.recordings_for_source("robot-1", 0).await.unwrap();
        assert_eq!(recordings.len(), 2);
        let first = recordings[0].recording_index;
        let second = recordings[1].recording_index;

        let first_traces = store.list_traces_for_recording(first).await.unwrap();
        let second_traces = store.list_traces_for_recording(second).await.unwrap();
        assert_eq!(first_traces.len(), 1, "ts=150 routes to recording A");
        assert_eq!(second_traces.len(), 1, "ts=250 routes to recording B");

        let a_dir = TracePath::new(
            first.to_string(),
            "joints",
            first_traces[0].trace_id.clone(),
        )
        .directory(context.recordings_root.as_path());
        let a: serde_json::Value =
            serde_json::from_slice(&std::fs::read(a_dir.join("trace.json")).unwrap()).unwrap();
        assert_eq!(a, serde_json::json!([{"i": 1}]));
    }

    #[tokio::test]
    async fn inverted_stop_after_next_start_preserves_both_recordings() {
        // A slow stop can reach the daemon after the next recording's start
        // (start/stop inversion). Listener order here is:
        //   start(A, t1=100) -> data(A) -> start(B, t3=200)
        //   -> stop(t2=150) -> data(B) -> stop(t4=300)
        // with t1 < t2 < t3 < t4. The dispatcher must: close A (stopped_at set,
        // refined to the true stop 150) with its data; keep B alive through
        // t2's stolen stop, closing it only at t4 (300) with its data; drop
        // nothing as an orphan (both traces exist); and fire RecordingStopped
        // for BOTH recordings (the retired one must still become notifiable).
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let bus = crate::state::EventBus::new();
        let mut sub = bus.subscribe();
        let dispatcher_context = DispatcherContext {
            event_bus: Some(bus.clone()),
            config_refresh_tx: None,
        };
        let (tx, handle) = spawn_with_context(
            store.clone(),
            context.clone(),
            dispatcher_context,
            shutdown_rx,
        );

        tx.send(start("robot-1", 100)).await.unwrap();
        tx.send(datum("robot-1", 110, 1)).await.unwrap();
        tx.send(start("robot-1", 200)).await.unwrap();
        // The stolen stop: its publish time (150) predates B's open (200).
        tx.send(stop("robot-1", 150)).await.unwrap();
        tx.send(datum("robot-1", 210, 2)).await.unwrap();
        tx.send(stop("robot-1", 300)).await.unwrap();

        drop(tx);
        timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        let recordings = store.recordings_for_source("robot-1", 0).await.unwrap();
        assert_eq!(recordings.len(), 2);
        let recording_a = &recordings[0];
        let recording_b = &recordings[1];

        assert!(
            recording_a.stopped_at.is_some(),
            "recording A must be closed when the next start supersedes it"
        );
        assert_eq!(
            recording_a.stop_timestamp_ns,
            Some(150),
            "recording A's stop must be refined to the true (earlier) stop"
        );
        assert!(
            recording_b.stopped_at.is_some(),
            "recording B must be closed by its own stop"
        );
        assert_eq!(
            recording_b.stop_timestamp_ns,
            Some(300),
            "the stolen stop at 150 must not close B; only t4 does"
        );

        let a_traces = store
            .list_traces_for_recording(recording_a.recording_index)
            .await
            .unwrap();
        let b_traces = store
            .list_traces_for_recording(recording_b.recording_index)
            .await
            .unwrap();
        assert_eq!(a_traces.len(), 1, "A keeps its datum (ts=110)");
        assert_eq!(
            b_traces.len(),
            1,
            "B keeps its datum (ts=210), not orphaned"
        );

        let a_dir = TracePath::new(
            recording_a.recording_index.to_string(),
            "joints",
            a_traces[0].trace_id.clone(),
        )
        .directory(context.recordings_root.as_path());
        let a: serde_json::Value =
            serde_json::from_slice(&std::fs::read(a_dir.join("trace.json")).unwrap()).unwrap();
        assert_eq!(a, serde_json::json!([{"i": 1}]));

        let b_dir = TracePath::new(
            recording_b.recording_index.to_string(),
            "joints",
            b_traces[0].trace_id.clone(),
        )
        .directory(context.recordings_root.as_path());
        let b: serde_json::Value =
            serde_json::from_slice(&std::fs::read(b_dir.join("trace.json")).unwrap()).unwrap();
        assert_eq!(b, serde_json::json!([{"i": 2}]));

        // Both recordings must fire RecordingStopped — the retired recording A
        // (via the delayed-stop fall-through) as well as the normally-closed B —
        // so the cloud stop-notifier fires for each.
        let mut stopped: Vec<i64> = Vec::new();
        while let Ok(event) = sub.try_recv() {
            if let DaemonEvent::RecordingStopped { recording_index } = event {
                stopped.push(recording_index);
            }
        }
        assert!(
            stopped.contains(&recording_a.recording_index),
            "RecordingStopped must fire for the retired recording A"
        );
        assert!(
            stopped.contains(&recording_b.recording_index),
            "RecordingStopped must fire for the normally-closed recording B"
        );
    }

    #[test]
    fn frame_range_cuts_at_the_stop_boundary() {
        // `video_boundary` tests the cut; this pins the argument order.
        let open_ns = 1_000_000_000;
        let stop_ns = open_ns + 25 * 1_000;
        let offsets = [0, 10, 20, 30, 40];
        assert_eq!(
            frame_range_in_window(&offsets, open_ns, open_ns, Some(stop_ns), 5),
            (0, 3)
        );
    }

    #[test]
    fn frame_range_cuts_the_frames_published_before_the_window_opened() {
        // The case a chunk-open-stamp membership rule could not express: a
        // camera process that never calls start_recording has a chunk open when
        // the window arrives, so its first frames predate the recording and the
        // rest belong to it.
        let open_ns = 1_000_000_000;
        let started_at_ns = open_ns + 25 * 1_000;
        let offsets = [0, 10, 20, 30, 40];
        assert_eq!(
            frame_range_in_window(&offsets, open_ns, started_at_ns, None, 5),
            (3, 2)
        );
    }

    #[test]
    fn frame_range_cuts_both_ends_of_a_chunk_that_spans_a_whole_window() {
        // A producer sealing only on its own caps can hold an entire recording
        // inside one chunk, with frames either side of it.
        let open_ns = 1_000_000_000;
        let offsets = [0, 10, 20, 30, 40, 50];
        assert_eq!(
            frame_range_in_window(
                &offsets,
                open_ns,
                open_ns + 15 * 1_000,
                Some(open_ns + 45 * 1_000),
                6
            ),
            (2, 3)
        );
    }

    #[test]
    fn frame_range_without_per_frame_data_is_all_or_nothing() {
        // No offsets to cut on, so the open stamp decides the whole chunk.
        let open_ns = 1_000_000_000;
        assert_eq!(
            frame_range_in_window(&[], open_ns, open_ns, None, 7),
            (0, 7)
        );
        assert_eq!(
            frame_range_in_window(&[], open_ns, open_ns + 1, None, 7),
            (0, 0)
        );
    }

    #[test]
    fn frame_range_clamps_to_the_announced_frame_count() {
        // The pipeline sizes itself from `frame_count`, so the cut cannot
        // exceed it.
        let open_ns = 1_000_000_000;
        let offsets = [0, 10, 20, 30];
        assert_eq!(
            frame_range_in_window(&offsets, open_ns, open_ns, Some(open_ns + 60 * 1_000), 2),
            (0, 2)
        );
    }

    #[test]
    fn a_chunk_spanning_two_recordings_is_claimed_by_both() {
        // The defect this replaced: one chunk held both recordings' frames and
        // was dropped whole, because its open stamp sat before the first window.
        let open_ns = 1_000_000_000;
        let us = |n: i64| open_ns + n * 1_000;
        let mut entry = WindowsForSource::default();
        let mut first = test_window(1, us(10));
        first.stopped_at_ns = Some(us(30));
        entry.closing.push(first);
        entry.live = Some(test_window(2, us(40)));

        // Frames at 0..50 µs: before recording 1, inside it, between, inside 2.
        let claims = claims_for_chunk(&entry, open_ns, &[0, 10, 20, 30, 40, 50], 6);

        assert_eq!(
            claims,
            vec![
                ChunkClaim {
                    slot: WindowSlot::Closing(0),
                    skip: 1,
                    count: 2,
                },
                ChunkClaim {
                    slot: WindowSlot::Live,
                    skip: 4,
                    count: 2,
                },
            ]
        );
    }

    #[test]
    fn a_claim_slices_the_sidecar_to_the_frames_it_owns() {
        // The sidecar indexes the encoded mp4, so its stamps must be the same
        // run the encode keeps: `skip` off the head, `count` off the tail.
        let stamps = [0.0, 0.1, 0.2, 0.3, 0.4];
        let claim = |skip, count| {
            ChunkClaim {
                slot: WindowSlot::Live,
                skip,
                count,
            }
            .timestamps(&stamps)
        };

        assert_eq!(claim(0, 5), stamps, "an uncut claim keeps every stamp");
        assert_eq!(claim(0, 3), vec![0.0, 0.1, 0.2], "tail cut");
        assert_eq!(claim(2, 3), vec![0.2, 0.3, 0.4], "head cut");
        assert_eq!(claim(1, 2), vec![0.1, 0.2], "cut at both ends");
        // A claim can only ever be clamped to `frame_count`, but the slice
        // must not panic if the stamps run short of it.
        assert_eq!(claim(3, 9), vec![0.3, 0.4]);
        assert!(claim(9, 2).is_empty());
    }

    #[tokio::test]
    async fn one_flush_marker_credits_every_window_that_producer_reached() {
        // A camera process publishes no stop, so it flushes once — at its own
        // exit — and one of its chunks can carry several recordings. Crediting
        // only the oldest window leaves the rest waiting out the flush-marker
        // cap for a marker that is never coming.
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let mut dispatcher = Dispatcher::new(store, context, DispatcherContext::default());

        let source = ("robot-1".to_string(), 0);
        let entry = dispatcher.windows.entry(source.clone()).or_default();
        for (recording_index, started_at_ns) in [(1, 100), (2, 200)] {
            let mut window = test_window(recording_index, started_at_ns);
            window.stopped_at_ns = Some(started_at_ns + 50);
            window.awaiting_flush = true;
            window.video_producers.insert(7);
            entry.closing.push(window);
        }

        dispatcher.mark_source_flushed(&source, 7);

        let entry = &dispatcher.windows[&source];
        assert!(
            entry
                .closing
                .iter()
                .all(|window| window.flush_markers_settled()),
            "both windows the producer's chunk reached must be settled by its \
             single marker"
        );
    }

    #[test]
    fn a_chunk_with_no_in_window_frame_claims_nothing() {
        let open_ns = 1_000_000_000;
        let entry = WindowsForSource {
            live: Some(test_window(1, open_ns + 100 * 1_000)),
            ..Default::default()
        };
        assert!(claims_for_chunk(&entry, open_ns, &[0, 10, 20], 3).is_empty());
    }

    #[test]
    fn every_claim_but_the_last_gets_its_own_hard_link() {
        // Each claim is relinked and unlinked by its own trace actor, so they
        // cannot share one path.
        let dir = tempfile::tempdir().unwrap();
        let spool = dir.path().join("chunk_1_2.nut");
        std::fs::write(&spool, b"nut").unwrap();

        let sources = claim_sources(&spool, 3);

        let paths: Vec<_> = sources.into_iter().map(|s| s.unwrap()).collect();
        assert_eq!(paths.len(), 3);
        assert_eq!(paths[2], spool, "the last claim consumes the spool file");
        for path in &paths {
            assert_eq!(std::fs::read(path).unwrap(), b"nut");
        }
        // Distinct names over one inode: unlinking one leaves the others.
        std::fs::remove_file(&paths[0]).unwrap();
        assert!(paths[1].exists() && paths[2].exists());
    }

    #[test]
    fn a_single_claim_never_copies_the_spool_file() {
        let dir = tempfile::tempdir().unwrap();
        let spool = dir.path().join("chunk_9_9.nut");
        std::fs::write(&spool, b"nut").unwrap();
        assert_eq!(claim_sources(&spool, 1), vec![Some(spool)]);
    }

    /// Announce a finished single-frame chunk opened at `publish_ts`. The
    /// caller must have spooled the matching NUT first.
    fn video_chunk(robot: &str, publish_ts: i64, thread_id: i64, producer_pid: u32) -> Envelope {
        video_chunk_frames(robot, publish_ts, thread_id, producer_pid, &[0])
    }

    /// As [`video_chunk`], with one frame per entry in `publish_offsets_us`.
    fn video_chunk_frames(
        robot: &str,
        publish_ts: i64,
        thread_id: i64,
        producer_pid: u32,
        publish_offsets_us: &[u32],
    ) -> Envelope {
        let capture_stamps: Vec<i64> = publish_offsets_us
            .iter()
            .map(|offset| publish_ts + i64::from(*offset) * 1_000)
            .collect();
        Envelope::VideoChunkReady {
            robot_id: robot.into(),
            robot_instance: 0,
            data_type: "RGB_IMAGES".into(),
            sensor_name: Some("camera_0".into()),
            publish_timestamp_ns: publish_ts,
            thread_id,
            producer_pid,
            width: 64,
            height: 64,
            byte_count: 9,
            frame_count: publish_offsets_us.len() as u32,
            frame_timestamps_s: capture_stamps.iter().map(|ns| *ns as f64 / 1e9).collect(),
            frame_timestamps_ns: capture_stamps,
            dtype: FrameDtype::Rgb8,
            frame_publish_offsets_us: publish_offsets_us.to_vec(),
        }
    }

    /// Spool a placeholder NUT at the path the producer would have written, so
    /// the dispatcher's relink has a file to move.
    fn spool_placeholder_nut(recordings_root: &std::path::Path, publish_ts: i64, thread_id: i64) {
        let path = paths::spool_chunk_path(
            recordings_root,
            "robot-1",
            0,
            "RGB_IMAGES",
            Some("camera_0"),
            publish_ts,
            thread_id,
        );
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"nut-bytes").unwrap();
    }

    #[tokio::test]
    async fn video_chunk_routes_by_open_time_into_its_window() {
        // A video chunk's `publish_timestamp_ns` is its *open* time — strictly
        // inside the recording — so a recording's tail chunk (announced just
        // before the stop) routes by a timestamp before the stop boundary and
        // lands in the recording rather than being dropped at the boundary.
        fast_holdback();
        let (store, dir) = open_store().await;
        let recordings_root = dir.path().join("recordings");
        let context = test_context(recordings_root.clone(), store.clone());
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let (tx, handle) = spawn(store.clone(), context.clone(), shutdown_rx);

        let (publish_ts, thread_id) = (150, 7);
        spool_placeholder_nut(&recordings_root, publish_ts, thread_id);

        // Window [100, 200); the chunk (open ts 150) is announced before stop.
        tx.send(start("robot-1", 100)).await.unwrap();
        tx.send(video_chunk("robot-1", publish_ts, thread_id, 1))
            .await
            .unwrap();
        tx.send(stop("robot-1", 200)).await.unwrap();

        drop(tx);
        timeout(Duration::from_secs(10), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        let recordings = store.recordings_for_source("robot-1", 0).await.unwrap();
        assert_eq!(recordings.len(), 1);
        let traces = store
            .list_traces_for_recording(recordings[0].recording_index)
            .await
            .unwrap();
        assert!(
            traces
                .iter()
                .any(|trace| trace.data_type.as_deref() == Some("RGB_IMAGES")),
            "the in-window video chunk must route to a video trace, not be dropped"
        );
        let spool_path = paths::spool_chunk_path(
            &recordings_root,
            "robot-1",
            0,
            "RGB_IMAGES",
            Some("camera_0"),
            publish_ts,
            thread_id,
        );
        assert!(
            !spool_path.exists(),
            "the spooled NUT must be relinked out of the spool dir"
        );
    }

    #[tokio::test]
    async fn straddling_video_chunk_keeps_its_in_window_prefix() {
        // The frames published after the stop are cut off, not the chunk:
        // dropping it loses the recording's video, keeping it whole leaves the
        // recording holding video published after it closed.
        fast_holdback();
        let (store, dir) = open_store().await;
        let recordings_root = dir.path().join("recordings");
        let context = test_context(recordings_root.clone(), store.clone());
        let mut dispatcher = Dispatcher::new(store.clone(), context, DispatcherContext::default());

        let source = ("robot-1".to_string(), 0);
        let opened_at = Instant::now();
        // The chunk opens inside the window and its frames run past the stop.
        let stop_ns = 150 + 25 * 1_000;
        dispatcher
            .handle_start(source.clone(), None, None, 100, 100, opened_at)
            .await;
        dispatcher
            .handle_stop(source.clone(), stop_ns, stop_ns, opened_at)
            .await;

        let (publish_ts, thread_id) = (150, 7);
        spool_placeholder_nut(&recordings_root, publish_ts, thread_id);
        dispatcher
            .handle_inbound(
                video_chunk_frames("robot-1", publish_ts, thread_id, 1, &[0, 10, 20, 30, 40]),
                opened_at,
            )
            .await;
        dispatcher
            .release_due_holdback(opened_at + dispatcher.holdback + Duration::from_millis(1))
            .await;

        let entry = dispatcher.windows.get(&source).unwrap();
        assert_eq!(
            entry.closing[0].traces.len(),
            1,
            "the chunk's in-window frames must still route to a video trace"
        );
        assert_eq!(
            dispatcher.orphan_drops, 0,
            "cutting a chunk's tail is not an orphan drop"
        );
    }

    #[tokio::test]
    async fn video_chunk_with_no_in_window_frame_is_dropped() {
        // Defensive branch: an empty video trace would leave the recording
        // advertising a video with no frames.
        fast_holdback();
        let (store, dir) = open_store().await;
        let recordings_root = dir.path().join("recordings");
        let context = test_context(recordings_root.clone(), store.clone());
        let mut dispatcher = Dispatcher::new(store.clone(), context, DispatcherContext::default());

        let source = ("robot-1".to_string(), 0);
        let opened_at = Instant::now();
        let stop_ns = 150 + 25 * 1_000;
        dispatcher
            .handle_start(source.clone(), None, None, 100, 100, opened_at)
            .await;
        dispatcher
            .handle_stop(source.clone(), stop_ns, stop_ns, opened_at)
            .await;

        let (publish_ts, thread_id) = (150, 7);
        spool_placeholder_nut(&recordings_root, publish_ts, thread_id);
        dispatcher
            .handle_inbound(
                video_chunk_frames("robot-1", publish_ts, thread_id, 1, &[50, 60]),
                opened_at,
            )
            .await;
        dispatcher
            .release_due_holdback(opened_at + dispatcher.holdback + Duration::from_millis(1))
            .await;

        let entry = dispatcher.windows.get(&source).unwrap();
        assert!(
            entry.closing[0].traces.is_empty(),
            "a chunk with no in-window frame must not register a video trace"
        );
        assert_eq!(dispatcher.orphan_drops, 1);
        let spool_path = paths::spool_chunk_path(
            &recordings_root,
            "robot-1",
            0,
            "RGB_IMAGES",
            Some("camera_0"),
            publish_ts,
            thread_id,
        );
        assert!(
            !spool_path.exists(),
            "the dropped chunk's spooled NUT must be removed"
        );
    }

    #[tokio::test]
    async fn video_chunk_published_after_stop_is_dropped() {
        // A chunk whose open time falls after the window closed belongs to no
        // window and is dropped — the contrast that proves routing is by the
        // chunk's own timestamp, not by arrival order.
        fast_holdback();
        let (store, dir) = open_store().await;
        let recordings_root = dir.path().join("recordings");
        let context = test_context(recordings_root.clone(), store.clone());
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let (tx, handle) = spawn(store.clone(), context.clone(), shutdown_rx);

        let (publish_ts, thread_id) = (250, 7); // after the window's stop
        spool_placeholder_nut(&recordings_root, publish_ts, thread_id);

        tx.send(start("robot-1", 100)).await.unwrap();
        tx.send(stop("robot-1", 200)).await.unwrap();
        tx.send(video_chunk("robot-1", publish_ts, thread_id, 1))
            .await
            .unwrap();

        drop(tx);
        timeout(Duration::from_secs(10), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        let recordings = store.recordings_for_source("robot-1", 0).await.unwrap();
        assert_eq!(recordings.len(), 1);
        let traces = store
            .list_traces_for_recording(recordings[0].recording_index)
            .await
            .unwrap();
        assert!(
            !traces
                .iter()
                .any(|trace| trace.data_type.as_deref() == Some("RGB_IMAGES")),
            "a chunk published after the window closed has no window and is dropped"
        );
    }

    #[tokio::test]
    async fn routing_is_decoupled_from_the_provided_timestamp() {
        // The integration matrix's manual timestamp mode logs data with
        // 0-based capture timestamps, NOT wall clock. Routing uses the
        // publish timestamp (wall clock, in the window), so the data lands
        // correctly while its own 0-based timestamp is preserved as content.
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let (tx, handle) = spawn(store.clone(), context.clone(), shutdown_rx);

        let base = 1_700_000_000_000_000_000i64; // wall-clock publish window
        tx.send(start("robot-1", base)).await.unwrap();
        for index in 0..3i64 {
            // publish ts in-window; content ts 0-based.
            tx.send(datum_full("robot-1", base + index, index, index))
                .await
                .unwrap();
        }
        tx.send(stop("robot-1", base + 1000)).await.unwrap();

        drop(tx);
        timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        let recordings = store.recordings_for_source("robot-1", 0).await.unwrap();
        assert_eq!(recordings.len(), 1);
        let traces = store
            .list_traces_for_recording(recordings[0].recording_index)
            .await
            .unwrap();
        assert_eq!(
            traces.len(),
            1,
            "0-based-content data must route into the window"
        );
        assert_eq!(traces[0].write_status, TraceWriteStatus::Written);
    }

    #[tokio::test]
    async fn data_outside_any_window_is_dropped() {
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let (tx, handle) = spawn(store.clone(), context.clone(), shutdown_rx);

        // No StartRecording — the datum belongs to no window.
        tx.send(datum("robot-1", 100, 1)).await.unwrap();

        drop(tx);
        timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        let recordings = store.recordings_for_source("robot-1", 0).await.unwrap();
        assert!(recordings.is_empty(), "no recording should be created");
    }

    /// The case a cancel cannot reach: a recording that already stopped and is
    /// still uploading, so no window is left to key off.
    #[tokio::test]
    async fn discard_burns_a_stopped_recording_with_no_window_left() {
        let (store, dir) = open_store().await;
        // The state a recording is in while the daemon drains it after the
        // producer has gone.
        let row = store
            .create_recording(crate::state::NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                start_timestamp_ns: 100,
                ..crate::state::NewRecording::default()
            })
            .await
            .expect("create recording");
        let recording_index = row.recording_index;
        store
            .mark_recording_start_notified(recording_index, "cloud-rec-1")
            .await
            .expect("stamp cloud id");
        store
            .mark_recording_stopped(recording_index, 200)
            .await
            .expect("stop");
        store
            .create_trace(
                recording_index,
                "trace-1",
                Some("JOINT_POSITIONS"),
                Some("arm"),
            )
            .await
            .expect("create trace");

        let context = test_context(dir.path().join("recordings"), store.clone());
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let bus = crate::state::EventBus::new();
        let mut sub = bus.subscribe();
        let dispatcher_context = DispatcherContext {
            event_bus: Some(bus.clone()),
            config_refresh_tx: None,
        };
        let (tx, handle) = spawn_with_context(
            store.clone(),
            context.clone(),
            dispatcher_context,
            shutdown_rx,
        );

        tx.send(Envelope::DiscardRecording {
            recording_id: "cloud-rec-1".into(),
            timestamp_ns: 300,
        })
        .await
        .unwrap();

        drop(tx);
        timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        let recording = store
            .get_recording(recording_index)
            .await
            .unwrap()
            .expect("recording exists");
        assert!(
            recording.cancelled_at.is_some(),
            "a discard must burn the recording even with no window open"
        );
        let traces = store
            .list_traces_for_recording(recording_index)
            .await
            .unwrap();
        assert_eq!(
            traces[0].upload_status,
            crate::state::TraceUploadStatus::Failed,
            "queued upload work must be dropped"
        );

        // The announcement is what aborts the uploads already in flight.
        let mut saw_cancel = false;
        while let Ok(event) = sub.try_recv() {
            if matches!(
                event,
                DaemonEvent::RecordingCancelled {
                    recording_index: index
                } if index == recording_index
            ) {
                saw_cancel = true;
            }
        }
        assert!(saw_cancel, "RecordingCancelled must be published");
    }

    /// A discard must purge the write-behind batcher too, not just the rows it
    /// can see: a queued create would otherwise commit after the burn and
    /// resurrect uploadable work.
    #[tokio::test]
    async fn discard_purges_unflushed_trace_creates() {
        let (store, dir) = open_store().await;
        let row = store
            .create_recording(crate::state::NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(0),
                start_timestamp_ns: 100,
                ..crate::state::NewRecording::default()
            })
            .await
            .expect("create recording");
        let recording_index = row.recording_index;
        store
            .mark_recording_start_notified(recording_index, "cloud-rec-1")
            .await
            .expect("stamp cloud id");
        store
            .mark_recording_stopped(recording_index, 200)
            .await
            .expect("stop");

        let context = test_context(dir.path().join("recordings"), store.clone());
        // Unflushed, the way a still-finalising trace sits mid-drain.
        context
            .trace_writer
            .create("late-trace", recording_index, Some("JOINT_POSITIONS"), None);

        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let bus = crate::state::EventBus::new();
        let dispatcher_context = DispatcherContext {
            event_bus: Some(bus.clone()),
            config_refresh_tx: None,
        };
        let (tx, handle) = spawn_with_context(
            store.clone(),
            context.clone(),
            dispatcher_context,
            shutdown_rx,
        );

        tx.send(Envelope::DiscardRecording {
            recording_id: "cloud-rec-1".into(),
            timestamp_ns: 300,
        })
        .await
        .unwrap();

        drop(tx);
        timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        // Whether the create was purged or had already committed and was then
        // burned depends on whether the batcher's flush tick beat the discard,
        // so assert the invariant both orderings must satisfy.
        let traces = store
            .list_traces_for_recording(recording_index)
            .await
            .unwrap();
        for trace in &traces {
            assert_eq!(
                trace.upload_status,
                crate::state::TraceUploadStatus::Failed,
                "a surviving row must be terminal, found {trace:?}"
            );
        }
        assert!(
            store.traces_ready_for_upload().await.unwrap().is_empty(),
            "nothing may be left uploadable for a discarded recording"
        );
    }

    /// `DISCARDED` is broadcast to the whole org, so most arrive for recordings
    /// this daemon never held.
    #[tokio::test]
    async fn discard_of_an_unknown_recording_is_ignored() {
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let bus = crate::state::EventBus::new();
        let mut sub = bus.subscribe();
        let dispatcher_context = DispatcherContext {
            event_bus: Some(bus.clone()),
            config_refresh_tx: None,
        };
        let (tx, handle) = spawn_with_context(
            store.clone(),
            context.clone(),
            dispatcher_context,
            shutdown_rx,
        );

        tx.send(Envelope::DiscardRecording {
            recording_id: "someone-elses-recording".into(),
            timestamp_ns: 300,
        })
        .await
        .unwrap();

        drop(tx);
        timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        assert!(
            !matches!(sub.try_recv(), Ok(DaemonEvent::RecordingCancelled { .. })),
            "an unknown recording id must not cancel anything"
        );
    }

    #[tokio::test]
    async fn cancel_purges_held_data_and_marks_cancelled() {
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let bus = crate::state::EventBus::new();
        let mut sub = bus.subscribe();
        let dispatcher_context = DispatcherContext {
            event_bus: Some(bus.clone()),
            config_refresh_tx: None,
        };
        let (tx, handle) = spawn_with_context(
            store.clone(),
            context.clone(),
            dispatcher_context,
            shutdown_rx,
        );

        tx.send(start("robot-1", 100)).await.unwrap();
        tx.send(datum("robot-1", 110, 1)).await.unwrap();
        tx.send(Envelope::CancelRecording {
            robot_id: "robot-1".into(),
            robot_instance: 0,
            timestamp_ns: 120,
        })
        .await
        .unwrap();

        drop(tx);
        timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        let recordings = store.recordings_for_source("robot-1", 0).await.unwrap();
        assert_eq!(recordings.len(), 1);
        assert!(recordings[0].cancelled_at.is_some());

        let mut saw_cancel = false;
        while let Ok(event) = sub.try_recv() {
            if matches!(event, DaemonEvent::RecordingCancelled { .. }) {
                saw_cancel = true;
            }
        }
        assert!(saw_cancel, "RecordingCancelled must be published");
    }

    #[tokio::test]
    async fn reap_idle_force_closes_a_silent_live_window() {
        // A producer that crashes without a Stop leaves a live window open. The
        // idle reaper must force-close it (open upper bound, so straggler data
        // still routes) and mark the recording stopped so it reaches a terminal,
        // notifiable state — otherwise the recording leaks forever.
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let bus = crate::state::EventBus::new();
        let mut sub = bus.subscribe();
        let mut dispatcher = Dispatcher::new(
            store.clone(),
            context,
            DispatcherContext {
                event_bus: Some(bus.clone()),
                config_refresh_tx: None,
            },
        );

        let source = ("robot-1".to_string(), 0);
        let opened_at = Instant::now();
        dispatcher
            .handle_start(source.clone(), None, None, 100, 100, opened_at)
            .await;
        assert!(dispatcher.windows.get(&source).unwrap().live.is_some());

        // Advance past the idle horizon (a future instant — no real waiting).
        let now = opened_at + IDLE_REAP + Duration::from_secs(1);
        dispatcher.reap_idle(now).await;

        let entry = dispatcher.windows.get(&source).unwrap();
        assert!(entry.live.is_none(), "the idle live window is force-closed");
        assert_eq!(entry.closing.len(), 1);
        assert_eq!(
            entry.closing[0].stopped_at_ns,
            Some(i64::MAX),
            "the reaped window keeps an open upper bound for stragglers"
        );

        let recordings = store.recordings_for_source("robot-1", 0).await.unwrap();
        assert_eq!(recordings.len(), 1);
        assert!(
            recordings[0].stopped_at.is_some(),
            "the recording row is marked stopped at the reap moment"
        );

        let mut saw_stop = false;
        while let Ok(event) = sub.try_recv() {
            if matches!(event, DaemonEvent::RecordingStopped { .. }) {
                saw_stop = true;
            }
        }
        assert!(
            saw_stop,
            "RecordingStopped is published for the reaped window"
        );
    }

    #[tokio::test]
    async fn reap_idle_leaves_a_recently_active_window_open() {
        // A window whose source was seen within the idle horizon must NOT be
        // reaped — the guard against force-closing a still-live recording.
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let mut dispatcher = Dispatcher::new(store.clone(), context, DispatcherContext::default());

        let source = ("robot-1".to_string(), 0);
        let opened_at = Instant::now();
        dispatcher
            .handle_start(source.clone(), None, None, 100, 100, opened_at)
            .await;

        // Only a short time has passed — well within the idle horizon.
        dispatcher
            .reap_idle(opened_at + Duration::from_millis(5))
            .await;

        assert!(
            dispatcher.windows.get(&source).unwrap().live.is_some(),
            "a recently-active window must stay live"
        );
    }

    #[tokio::test]
    async fn housekeep_evicts_a_closing_window_past_retention() {
        // A closing window is retained for 2·holdback (so its in-window data has
        // released) and then evicted; without this the window map — and the
        // actor handles it holds — leak for the daemon's lifetime.
        //
        // Retention is the floor, not the whole gate: the flush marker is
        // settled first so the deadline itself is what is tested.
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let mut dispatcher = Dispatcher::new(store.clone(), context, DispatcherContext::default());

        let source = ("robot-1".to_string(), 0);
        let opened_at = Instant::now();
        dispatcher
            .handle_start(source.clone(), None, None, 100, 100, opened_at)
            .await;
        let stopped_at = opened_at + Duration::from_millis(1);
        dispatcher
            .handle_stop(source.clone(), 200, 200, stopped_at)
            .await;
        assert_eq!(
            dispatcher.windows.get(&source).unwrap().closing.len(),
            1,
            "the stopped window is retained as closing"
        );
        dispatcher
            .handle_inbound(source_flushed("robot-1", 900, 1), stopped_at)
            .await;
        dispatcher
            .release_due_holdback(stopped_at + dispatcher.holdback + Duration::from_millis(1))
            .await;

        // Just past the 2·holdback retention window.
        let retention = dispatcher.holdback * 2;
        let now = stopped_at + retention + Duration::from_millis(1);
        dispatcher.housekeep(now).await;

        let closing = dispatcher
            .windows
            .get(&source)
            .map_or(0, |entry| entry.closing.len());
        assert_eq!(closing, 0, "a closing window past 2·holdback is evicted");
    }

    #[tokio::test]
    async fn housekeep_holds_a_stopped_window_until_the_flush_marker() {
        // The stop is published before the writer's flush barrier, so the tail
        // chunks it seals arrive after their own stop by an unbounded backlog.
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let mut dispatcher = Dispatcher::new(store.clone(), context, DispatcherContext::default());

        let source = ("robot-1".to_string(), 0);
        let opened_at = Instant::now();
        dispatcher
            .handle_start(source.clone(), None, None, 100, 100, opened_at)
            .await;
        let stopped_at = opened_at + Duration::from_millis(1);
        dispatcher
            .handle_stop(source.clone(), 200, 200, stopped_at)
            .await;

        // Well past the retention deadline.
        let past_retention = stopped_at + dispatcher.holdback * 2 + Duration::from_millis(1);
        dispatcher.housekeep(past_retention).await;
        assert_eq!(
            dispatcher.windows.get(&source).unwrap().closing.len(),
            1,
            "a stopped window still owed a flush marker outlives the retention deadline"
        );

        // The barrier drains and announces its marker.
        dispatcher
            .handle_inbound(source_flushed("robot-1", 900, 1), past_retention)
            .await;
        let released_at = past_retention + dispatcher.holdback + Duration::from_millis(1);
        dispatcher.release_due_holdback(released_at).await;
        dispatcher.housekeep(released_at).await;

        let closing = dispatcher
            .windows
            .get(&source)
            .map_or(0, |entry| entry.closing.len());
        assert_eq!(
            closing, 0,
            "the window retires once the marker has released"
        );
    }

    #[tokio::test]
    async fn housekeep_evicts_a_stopped_window_when_the_flush_marker_never_comes() {
        // The marker is best-effort: a producer that never sends one must not
        // pin the window open forever.
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let mut dispatcher = Dispatcher::new(store.clone(), context, DispatcherContext::default());

        let source = ("robot-1".to_string(), 0);
        let opened_at = Instant::now();
        dispatcher
            .handle_start(source.clone(), None, None, 100, 100, opened_at)
            .await;
        let stopped_at = opened_at + Duration::from_millis(1);
        dispatcher
            .handle_stop(source.clone(), 200, 200, stopped_at)
            .await;

        let at_cap = stopped_at + FLUSH_MARKER_WAIT_CAP + Duration::from_millis(1);
        dispatcher.housekeep(at_cap).await;

        let closing = dispatcher
            .windows
            .get(&source)
            .map_or(0, |entry| entry.closing.len());
        assert_eq!(
            closing, 0,
            "a marker that never arrives is capped, not waited on forever"
        );
    }

    /// A producer's claim that it is logging video, before any chunk sealed.
    fn video_producer_active(robot: &str, publish_ts: i64, producer_pid: u32) -> Envelope {
        Envelope::VideoProducerActive {
            robot_id: robot.into(),
            robot_instance: 0,
            publish_timestamp_ns: publish_ts,
            producer_pid,
        }
    }

    #[tokio::test]
    async fn video_claim_holds_the_window_for_a_producer_whose_chunk_has_not_sealed() {
        // In real arrival order: the lifecycle process's instant marker is
        // processed while the camera process is still deep in its backlog, so
        // chunk-driven attribution is empty and the window would retire on a
        // marker vouching for nothing. The claim is what holds it.
        fast_holdback();
        let (store, dir) = open_store().await;
        let recordings_root = dir.path().join("recordings");
        let context = test_context(recordings_root.clone(), store.clone());
        let mut dispatcher = Dispatcher::new(store.clone(), context, DispatcherContext::default());

        let source = ("robot-1".to_string(), 0);
        let opened_at = Instant::now();
        dispatcher
            .handle_start(source.clone(), None, None, 100, 100, opened_at)
            .await;

        // The camera process claims the source as it logs, before any chunk.
        dispatcher
            .handle_inbound(video_producer_active("robot-1", 150, 1), opened_at)
            .await;
        dispatcher
            .release_due_holdback(opened_at + dispatcher.holdback + Duration::from_millis(1))
            .await;

        let stopped_at = opened_at + Duration::from_millis(2);
        dispatcher
            .handle_stop(source.clone(), 200, 200, stopped_at)
            .await;

        // The lifecycle process owns no video, so its marker lands at once.
        dispatcher
            .handle_inbound(source_flushed("robot-1", 300, 2), stopped_at)
            .await;
        let retention = dispatcher.holdback * 2;
        let past_retention = stopped_at + retention + Duration::from_millis(1);
        dispatcher.release_due_holdback(past_retention).await;
        dispatcher.housekeep(past_retention).await;
        assert_eq!(
            dispatcher.windows.get(&source).unwrap().closing.len(),
            1,
            "a claim from a producer that has not announced a chunk yet must \
             still hold the window against another producer's marker"
        );

        // Its chunk is announced long past retention and must still route.
        let (publish_ts, thread_id) = (150, 7);
        spool_placeholder_nut(&recordings_root, publish_ts, thread_id);
        let drained_at = past_retention + Duration::from_millis(1);
        dispatcher
            .handle_inbound(video_chunk("robot-1", publish_ts, thread_id, 1), drained_at)
            .await;
        dispatcher
            .handle_inbound(source_flushed("robot-1", 400, 1), drained_at)
            .await;
        let released_at = drained_at + dispatcher.holdback + Duration::from_millis(1);
        dispatcher.release_due_holdback(released_at).await;
        assert_eq!(
            dispatcher.orphan_drops, 0,
            "the tail chunk must route into the window the claim held open"
        );

        // The claim delays eviction, it does not deadlock it.
        dispatcher.housekeep(released_at).await;
        let closing = dispatcher
            .windows
            .get(&source)
            .map_or(0, |entry| entry.closing.len());
        assert_eq!(
            closing, 0,
            "the window retires once the claiming producer's own marker arrives"
        );
    }

    #[tokio::test]
    async fn video_claim_outside_every_window_is_dropped_without_counting_an_orphan() {
        // Routine, not data loss: it must not inflate the orphan counter.
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"), store.clone());
        let mut dispatcher = Dispatcher::new(store.clone(), context, DispatcherContext::default());

        let source = ("robot-1".to_string(), 0);
        let opened_at = Instant::now();
        dispatcher
            .handle_start(source.clone(), None, None, 100, 100, opened_at)
            .await;
        dispatcher
            .handle_stop(source.clone(), 200, 200, opened_at)
            .await;

        // Published after the stop: past every window's upper bound.
        dispatcher
            .handle_inbound(video_producer_active("robot-1", 250, 1), opened_at)
            .await;
        dispatcher
            .release_due_holdback(opened_at + dispatcher.holdback + Duration::from_millis(1))
            .await;

        assert_eq!(dispatcher.orphan_drops, 0);
        let window = &dispatcher.windows.get(&source).unwrap().closing[0];
        assert!(
            window.video_producers.is_empty(),
            "a claim outside the window must not make the window wait on that \
             producer's marker"
        );
    }

    #[tokio::test]
    async fn source_flushed_from_one_producer_must_not_vouch_for_another_producers_chunk() {
        // `SourceFlushed` is per-process, so a video-less process's marker says
        // nothing about a camera process's still-open barrier. Retiring on the
        // first marker from any producer orphans the camera's tail chunk.
        fast_holdback();
        let (store, dir) = open_store().await;
        let recordings_root = dir.path().join("recordings");
        let context = test_context(recordings_root.clone(), store.clone());
        let mut dispatcher = Dispatcher::new(store.clone(), context, DispatcherContext::default());

        let source = ("robot-1".to_string(), 0);
        let opened_at = Instant::now();
        dispatcher
            .handle_start(source.clone(), None, None, 100, 100, opened_at)
            .await;

        // Producer A seals a chunk mid-recording, so the window knows of it.
        let (publish_ts, thread_id) = (150, 7);
        spool_placeholder_nut(&recordings_root, publish_ts, thread_id);
        dispatcher
            .handle_inbound(video_chunk("robot-1", publish_ts, thread_id, 1), opened_at)
            .await;
        dispatcher
            .release_due_holdback(opened_at + dispatcher.holdback + Duration::from_millis(1))
            .await;

        let stopped_at = opened_at + Duration::from_millis(2);
        dispatcher
            .handle_stop(source.clone(), 200, 200, stopped_at)
            .await;

        // Producer B owns no video, so its marker lands immediately.
        dispatcher
            .handle_inbound(source_flushed("robot-1", 300, 2), stopped_at)
            .await;
        dispatcher
            .release_due_holdback(stopped_at + dispatcher.holdback + Duration::from_millis(1))
            .await;

        // Well past retention, but producer A has not sent its marker.
        let retention = dispatcher.holdback * 2;
        let past_retention = stopped_at + retention + Duration::from_millis(1);
        dispatcher.housekeep(past_retention).await;
        assert_eq!(
            dispatcher.windows.get(&source).unwrap().closing.len(),
            1,
            "producer B's marker must not vouch for producer A's still-open \
             flush barrier"
        );

        // Producer A's barrier never drains, so the cap has to release it.
        let at_cap = stopped_at + FLUSH_MARKER_WAIT_CAP + Duration::from_millis(1);
        dispatcher.housekeep(at_cap).await;
        let closing = dispatcher
            .windows
            .get(&source)
            .map_or(0, |entry| entry.closing.len());
        assert_eq!(
            closing, 0,
            "an unmatched producer still gives way to the cap"
        );
    }

    #[tokio::test]
    async fn tail_video_chunk_announced_past_the_retention_deadline_still_routes() {
        // A burst producer is still draining when `stop_recording` returns, so
        // its in-window tail chunk is announced past the retention deadline;
        // without the marker gating eviction it is orphan-dropped.
        fast_holdback();
        let (store, dir) = open_store().await;
        let recordings_root = dir.path().join("recordings");
        let context = test_context(recordings_root.clone(), store.clone());
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let (tx, handle) = spawn(store.clone(), context.clone(), shutdown_rx);

        let (publish_ts, thread_id) = (150, 7); // opened inside window [100, 200)
        spool_placeholder_nut(&recordings_root, publish_ts, thread_id);

        tx.send(start("robot-1", 100)).await.unwrap();
        tx.send(stop("robot-1", 200)).await.unwrap();

        // Stand in for a slow flush barrier, well past the retention.
        tokio::time::sleep(Duration::from_millis(400)).await;
        tx.send(video_chunk("robot-1", publish_ts, thread_id, 1))
            .await
            .unwrap();
        tx.send(source_flushed("robot-1", 500, 1)).await.unwrap();

        drop(tx);
        timeout(Duration::from_secs(10), handle.shutdown())
            .await
            .expect("dispatcher shut down in time");

        let recordings = store.recordings_for_source("robot-1", 0).await.unwrap();
        assert_eq!(recordings.len(), 1);
        let traces = store
            .list_traces_for_recording(recordings[0].recording_index)
            .await
            .unwrap();
        assert!(
            traces
                .iter()
                .any(|trace| trace.data_type.as_deref() == Some("RGB_IMAGES")),
            "a tail chunk announced after the retention deadline must still route"
        );
    }
}
