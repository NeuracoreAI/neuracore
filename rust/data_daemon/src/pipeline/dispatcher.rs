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
//! Everything here is owned by one tokio task, so the window map and holdback
//! queue need no locks — total ordering through the `select!` loop is what
//! makes the routing decisions provable.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use data_daemon_ipc::{BatchedDataItem, Envelope};
use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio::time::sleep;
use uuid::Uuid;

use crate::lifecycle::signals::ShutdownSignal;
use crate::pipeline::trace_actor::{
    self, TraceActorContext, TraceActorMessage, TraceIdentity, TraceKey,
};
use crate::state::{DaemonEvent, NewRecording, SqliteStateStore, StateStore};
use crate::storage::paths::{self, TracePath};

/// Default holdback: each data envelope waits this long after daemon receipt
/// before it is routed. Tunable via `NCD_HOLDBACK_MS`. A generous default is
/// safe — joint/scalar data is sparse, so even a 1 s holdback retains only a
/// few thousand small envelopes per source. Completeness (catching a datum
/// whose publisher was preempted between capture and publish) scales directly
/// with this value.
const DEFAULT_HOLDBACK_MS: u64 = 500;

/// Environment override for the holdback, in milliseconds.
const HOLDBACK_ENV: &str = "NCD_HOLDBACK_MS";

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
    /// Org identifier the daemon stamps on every new recording row. Left as
    /// `None` when the daemon is running offline.
    pub org_id: Option<String>,
    /// Daemon event bus, used to publish recording/trace lifecycle events.
    pub event_bus: Option<crate::state::EventBus>,
}

/// Spawn the dispatcher task and return its inbound `mpsc::Sender`.
#[allow(dead_code)]
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
    trace_id: String,
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
    /// Per-trace actors spawned within this window.
    traces: HashMap<TraceKey, TraceHandle>,
}

impl ActiveWindow {
    /// Does this window's `[started_at_ns, stopped_at_ns)` contain `ts`?
    fn contains(&self, ts: i64) -> bool {
        ts >= self.started_at_ns && self.stopped_at_ns.is_none_or(|stop| ts < stop)
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
        timestamp_ns: i64,
        timestamp_s: Option<f64>,
        items: Vec<BatchedDataItem>,
    },
    Video {
        data_type: String,
        sensor_name: Option<String>,
        spool_seq: u64,
        width: u32,
        height: u32,
        byte_count: u64,
        frame_count: u32,
        frame_timestamps_s: Vec<f64>,
    },
}

/// The dispatcher's task-local state.
struct Dispatcher {
    store: Arc<SqliteStateStore>,
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
}

impl Dispatcher {
    fn new(
        store: SqliteStateStore,
        actor_context: Arc<TraceActorContext>,
        context: DispatcherContext,
    ) -> Self {
        Self {
            store: Arc::new(store),
            actor_context,
            context,
            holdback: configured_holdback(),
            windows: HashMap::new(),
            held: VecDeque::new(),
            actor_handles: Vec::new(),
            orphan_drops: 0,
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

            self.process_due(Instant::now()).await;
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
                robot_name,
                dataset_id,
                dataset_name,
                started_at_ns,
            } => {
                self.handle_start(
                    (robot_id, robot_instance),
                    robot_name,
                    dataset_id,
                    dataset_name,
                    started_at_ns,
                    recv_at,
                )
                .await;
            }
            Envelope::StopRecording {
                robot_id,
                robot_instance,
                stopped_at_ns,
            } => {
                self.handle_stop((robot_id, robot_instance), stopped_at_ns, recv_at)
                    .await;
            }
            Envelope::CancelRecording {
                robot_id,
                robot_instance,
                ..
            } => {
                self.handle_cancel((robot_id, robot_instance)).await;
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
                spool_seq,
                width,
                height,
                byte_count,
                frame_count,
                frame_timestamps_ns,
                frame_timestamps_s,
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
                        spool_seq,
                        width,
                        height,
                        byte_count,
                        frame_count,
                        frame_timestamps_s,
                    },
                });
            }
        }
    }

    fn touch_source(&mut self, source: &Source, recv_at: Instant) {
        self.windows.entry(source.clone()).or_default().last_seen = Some(recv_at);
    }

    async fn handle_start(
        &mut self,
        source: Source,
        robot_name: Option<String>,
        dataset_id: Option<String>,
        dataset_name: Option<String>,
        started_at_ns: i64,
        recv_at: Instant,
    ) {
        let new = NewRecording {
            robot_id: Some(&source.0),
            robot_instance: Some(source.1),
            robot_name: robot_name.as_deref(),
            dataset_id: dataset_id.as_deref(),
            dataset_name: dataset_name.as_deref(),
            org_id: self.context.org_id.as_deref(),
            start_timestamp_ns: started_at_ns,
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
        // A well-behaved producer stops before starting; if a live window is
        // somehow still open, retire it to `closing` bounded at the new start
        // so it stops catching data published after this point.
        if let Some(mut previous) = entry.live.take() {
            if previous.stopped_at_ns.is_none() {
                previous.stopped_at_ns = Some(started_at_ns);
                previous.stop_recv_at = Some(recv_at);
            }
            entry.closing.push(previous);
        }
        entry.live = Some(ActiveWindow {
            recording_index,
            started_at_ns,
            stopped_at_ns: None,
            stop_recv_at: None,
            traces: HashMap::new(),
        });

        if let Some(bus) = self.context.event_bus.as_ref() {
            bus.publish(DaemonEvent::RecordingStarted { recording_index });
        }
    }

    async fn handle_stop(&mut self, source: Source, stopped_at_ns: i64, recv_at: Instant) {
        let Some(entry) = self.windows.get_mut(&source) else {
            tracing::debug!(robot_id = source.0, "stop for unknown source; ignoring");
            return;
        };
        entry.last_seen = Some(recv_at);
        let Some(mut window) = entry.live.take() else {
            tracing::debug!(
                robot_id = source.0,
                "stop with no active recording; ignoring"
            );
            return;
        };
        window.stopped_at_ns = Some(stopped_at_ns);
        window.stop_recv_at = Some(recv_at);
        let recording_index = window.recording_index;
        entry.closing.push(window);

        if let Err(error) = self
            .store
            .mark_recording_stopped(recording_index, stopped_at_ns)
            .await
        {
            tracing::warn!(%error, recording_index, "failed to mark recording stopped");
        } else {
            tracing::info!(recording_index, "recording stopped");
            if let Some(bus) = self.context.event_bus.as_ref() {
                bus.publish(DaemonEvent::RecordingStopped { recording_index });
            }
        }
    }

    async fn handle_cancel(&mut self, source: Source) {
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
            let recording_index = window.recording_index;
            for (_, handle) in window.traces {
                let _ = handle.sender.send(TraceActorMessage::Cancel).await;
            }
            match self.store.cancel_recording(recording_index).await {
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
    }

    /// True when any source has a retained closing window.
    fn any_closing(&self) -> bool {
        self.windows.values().any(|entry| !entry.closing.is_empty())
    }

    /// Release every due held envelope, evict windows past their retention,
    /// and force-close idle sources.
    async fn process_due(&mut self, now: Instant) {
        // 1. Holdback releases — process strictly before evictions so a datum
        //    releasing in this tick still finds its (possibly closing) window.
        while self.held.front().is_some_and(|held| held.release_at <= now) {
            let held = self.held.pop_front().expect("front checked");
            self.route(held).await;
        }

        // 2. Window evictions: a closing window is retained for 2·HOLDBACK
        //    after its stop, by which point all in-window data has released.
        let retention = self.holdback * 2;
        let mut closing_actors: Vec<TraceHandle> = Vec::new();
        let mut empty_sources: Vec<Source> = Vec::new();
        for (source, entry) in self.windows.iter_mut() {
            entry.closing.retain_mut(|window| {
                let evict = window
                    .stop_recv_at
                    .is_some_and(|at| now.duration_since(at) >= retention);
                if evict {
                    for (_, handle) in window.traces.drain() {
                        closing_actors.push(handle);
                    }
                    false
                } else {
                    true
                }
            });
            if entry.live.is_none()
                && entry.closing.is_empty()
                && entry
                    .last_seen
                    .is_none_or(|at| now.duration_since(at) >= IDLE_REAP)
            {
                empty_sources.push(source.clone());
            }
        }
        // Send WindowClosing to every actor of an evicted window. Their senders
        // drop after, so each actor finalises and exits.
        for handle in closing_actors {
            let _ = handle.sender.send(TraceActorMessage::WindowClosing).await;
        }
        for source in empty_sources {
            self.windows.remove(&source);
        }

        // 3. Idle reaper: force-close a live window whose source has gone
        //    silent (producer crashed without a Stop).
        self.reap_idle(now).await;
    }

    /// Force-close any live window whose source has been silent past
    /// [`IDLE_REAP`]. Treated as if stopped at the window's start so the
    /// recording still reaches a terminal, notifiable state.
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
            // recording to partition against — keep the window open above to
            // catch any straggler data before eviction.
            window.stopped_at_ns = Some(i64::MAX);
            window.stop_recv_at = Some(now);
            let recording_index = window.recording_index;
            entry.closing.push(window);
            if let Err(error) = self
                .store
                .mark_recording_stopped(recording_index, i64::MAX)
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
                timestamp_ns,
                timestamp_s,
                items,
            } => {
                for item in items {
                    self.route_data(
                        &held.source,
                        publish_ts,
                        item.data_type,
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
                spool_seq,
                width,
                height,
                byte_count,
                frame_count,
                frame_timestamps_s,
            } => {
                self.route_video(
                    &held.source,
                    publish_ts,
                    data_type,
                    sensor_name,
                    spool_seq,
                    width,
                    height,
                    byte_count,
                    frame_count,
                    frame_timestamps_s,
                )
                .await;
            }
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
            &self.store,
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
        spool_seq: u64,
        width: u32,
        height: u32,
        byte_count: u64,
        frame_count: u32,
        frame_timestamps_s: Vec<f64>,
    ) {
        let recordings_root = self.actor_context.recordings_root.clone();
        let inbox_nut = paths::inbox_chunk_path(
            recordings_root.as_path(),
            &source.0,
            source.1,
            &data_type,
            sensor_name.as_deref(),
            spool_seq,
        );

        // The whole chunk routes by its announce (publish) time. The producer
        // rolls a chunk on every lifecycle event, so a chunk is announced
        // inside exactly one recording window.
        let Some(entry) = self.windows.get_mut(source) else {
            remove_inbox_nut(&inbox_nut);
            self.note_orphan();
            return;
        };
        let Some(window) = Self::window_for_mut(entry, publish_ts) else {
            remove_inbox_nut(&inbox_nut);
            self.note_orphan();
            return;
        };

        let recording_index = window.recording_index;
        let handle = Self::ensure_actor(
            window,
            &self.store,
            &self.actor_context,
            data_type.clone(),
            sensor_name.clone(),
            &mut self.actor_handles,
        );
        let chunk_index = handle.next_video_chunk;
        handle.next_video_chunk = handle.next_video_chunk.saturating_add(1);
        let sender = handle.sender.clone();
        let trace_id = handle.trace_id.clone();

        // Relink the inbox NUT under the recording's trace directory at the
        // path the actor expects: `{recording_index}/{data_type}/{trace_id}/chunks/chunk_NNNN.nut`.
        let chunks_dir = TracePath::new(recording_index.to_string(), data_type, trace_id)
            .chunks_dir(recordings_root.as_path());
        let dest = chunks_dir.join(paths::chunk_filename(chunk_index));
        if let Err(error) = relink_nut(&inbox_nut, &chunks_dir, &dest) {
            tracing::warn!(
                %error,
                recording_index,
                src = %inbox_nut.display(),
                dst = %dest.display(),
                "failed to relink video chunk into recording; dropping"
            );
            remove_inbox_nut(&inbox_nut);
            return;
        }

        if sender
            .send(TraceActorMessage::Video {
                chunk_index,
                width,
                height,
                byte_count,
                frame_count,
                frame_timestamps_s,
            })
            .await
            .is_err()
        {
            tracing::warn!(
                recording_index,
                "video trace actor inbox closed; dropping chunk"
            );
        }
    }

    /// Look up or spawn the per-trace actor for `(window, data_type,
    /// sensor_name)`, returning its routing handle.
    fn ensure_actor<'a>(
        window: &'a mut ActiveWindow,
        store: &Arc<SqliteStateStore>,
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
            let trace_id = Uuid::new_v4().to_string();
            let identity = TraceIdentity {
                trace_id: trace_id.clone(),
                key,
            };
            let (tx, actor_rx) = mpsc::channel(TRACE_QUEUE_CAPACITY);
            let actor_store = Arc::clone(store);
            let actor_context = Arc::clone(actor_context);
            let join = tokio::spawn(async move {
                trace_actor::run(actor_store, actor_context, identity, actor_rx).await;
            });
            actor_handles.push(join);
            TraceHandle {
                sender: tx,
                trace_id,
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
        tracing::info!("dispatcher stopped");
    }
}

/// Relink a producer-spooled NUT into the recording's chunks directory.
/// Prefers an atomic rename (same filesystem); falls back to copy + remove.
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

fn remove_inbox_nut(path: &std::path::Path) {
    if let Err(error) = std::fs::remove_file(path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            tracing::debug!(%error, path = %path.display(), "failed to remove orphan inbox NUT");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::video_encoder::VideoEncoder;
    use crate::state::{SqliteStateStore, TraceWriteStatus};
    use crate::storage::budget::{StorageBudget, StoragePolicy};
    use std::path::PathBuf;
    use tempfile::TempDir;
    use tokio::sync::broadcast;
    use tokio::time::{timeout, Duration};

    async fn open_store() -> (SqliteStateStore, TempDir) {
        let dir = TempDir::new().expect("tempdir");
        let store = SqliteStateStore::open(&dir.path().join("state.db"))
            .await
            .expect("open store");
        (store, dir)
    }

    fn test_context(recordings_root: PathBuf) -> Arc<TraceActorContext> {
        let policy = StoragePolicy {
            storage_limit_bytes: None,
            min_free_disk_bytes: 0,
            refresh_interval: Duration::from_secs(60),
        };
        let budget = Arc::new(StorageBudget::new(&recordings_root, policy));
        Arc::new(TraceActorContext::new(
            recordings_root,
            budget,
            VideoEncoder::new(),
        ))
    }

    fn start(robot: &str, started_at_ns: i64) -> Envelope {
        Envelope::StartRecording {
            robot_id: robot.into(),
            robot_instance: 0,
            robot_name: None,
            dataset_id: None,
            dataset_name: None,
            started_at_ns,
        }
    }

    fn stop(robot: &str, stopped_at_ns: i64) -> Envelope {
        Envelope::StopRecording {
            robot_id: robot.into(),
            robot_instance: 0,
            stopped_at_ns,
        }
    }

    /// A datum published at `publish_ts` with `content_ts` as its own
    /// (decoupled) capture timestamp.
    fn datum_full(robot: &str, publish_ts: i64, content_ts: i64, value: i64) -> Envelope {
        Envelope::data(
            robot.into(),
            0,
            "joints".into(),
            Some("waist".into()),
            publish_ts,
            content_ts,
            None,
            serde_json::to_vec(&serde_json::json!({ "i": value })).unwrap(),
        )
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
    async fn routes_data_into_its_window_by_timestamp() {
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"));
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let bus = crate::state::EventBus::new();
        let dispatcher_context = DispatcherContext {
            org_id: None,
            event_bus: Some(bus.clone()),
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
        let context = test_context(dir.path().join("recordings"));
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
    async fn routing_is_decoupled_from_the_provided_timestamp() {
        // The integration matrix's manual timestamp mode logs data with
        // 0-based capture timestamps, NOT wall clock. Routing uses the
        // publish timestamp (wall clock, in the window), so the data lands
        // correctly while its own 0-based timestamp is preserved as content.
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"));
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
        let context = test_context(dir.path().join("recordings"));
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

    #[tokio::test]
    async fn cancel_purges_held_data_and_marks_cancelled() {
        fast_holdback();
        let (store, dir) = open_store().await;
        let context = test_context(dir.path().join("recordings"));
        let (_shutdown_tx, shutdown_rx) = broadcast::channel(8);
        let bus = crate::state::EventBus::new();
        let mut sub = bus.subscribe();
        let dispatcher_context = DispatcherContext {
            org_id: None,
            event_bus: Some(bus.clone()),
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
            cancelled_at_ns: 120,
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
}
