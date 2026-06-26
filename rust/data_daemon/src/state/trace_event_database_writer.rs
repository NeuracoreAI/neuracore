//! Coalescing + batching write-behind for per-trace actor writes.
//!
//! Per-trace actors fire-and-forget partial column updates (a `writing` bump, a
//! debounced `bytes_written`, the finalise `written` + `total_bytes`, or a
//! `failed`) without ever awaiting a transaction. This task coalesces
//! consecutive ops for the same trace last-writer-wins per column (a burst of
//! `bytes_written` collapses to one row write) and flushes the pending set in a
//! single batched transaction ([`SqliteStateStore::apply_trace_writes`]) on a
//! short timer or once the pending set grows past a cap.
//!
//! Terminal-state monotonicity (a late progress write can't resurrect a
//! cancelled row) lives in `apply_trace_writes`'s `WHERE` guard, so the writer
//! needs no coordination with the cancel path.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::{interval, Duration, MissedTickBehavior};

use crate::state::schema::{TraceErrorCode, TraceWriteStatus};
use crate::state::store::{CoalescedTraceWrite, SqliteStateStore, TraceCreate};

/// How often pending writes are flushed. Short enough that finalised traces
/// become visible promptly, long enough that a burst of progress updates
/// coalesces into one row write per flush.
const FLUSH_INTERVAL: Duration = Duration::from_millis(25);

/// Flush eagerly once this many distinct traces are pending, so a wide
/// fan-out (many traces updated within one interval) doesn't grow an
/// unbounded batch before the timer fires.
const MAX_PENDING_TRACES: usize = 512;

/// Control + data messages accepted by the writer task.
enum Message {
    /// A partial column update for one trace, merged into the pending set.
    Write(CoalescedTraceWrite),
    /// Discard every *pending create* for a recording, then acknowledge. Sent by
    /// the dispatcher's cancel path before `cancel_recording` so a not-yet-
    /// flushed trace can't be inserted as an orphan row after the recording is
    /// burned. Pending update-only writes are left — the terminal-state guard in
    /// `apply_trace_writes` already makes them no-ops against the failed row.
    DropRecording {
        recording_index: i64,
        ack: oneshot::Sender<()>,
    },
    /// Flush everything pending now and acknowledge (tests + shutdown).
    Flush(oneshot::Sender<()>),
    /// Drain, flush, acknowledge, and exit.
    Shutdown(oneshot::Sender<()>),
}

/// Cloneable handle the per-trace actors use to enqueue writes. Every method is
/// synchronous and non-blocking: the actor fires an update and moves on.
#[derive(Clone)]
pub struct TraceWriteHandle {
    tx: mpsc::UnboundedSender<Message>,
}

impl TraceWriteHandle {
    /// Create the trace row (fire-and-forget). Sent once, as the actor's first
    /// write, so the row is inserted by the next batched flush instead of the
    /// actor blocking on a synchronous `create_trace`. Works at any point in a
    /// recording, including a sensor that starts logging midway.
    pub fn create(
        &self,
        trace_id: &str,
        recording_index: i64,
        data_type: Option<&str>,
        data_type_name: Option<&str>,
    ) {
        self.enqueue(CoalescedTraceWrite {
            trace_id: trace_id.to_string(),
            create: Some(TraceCreate {
                recording_index,
                data_type: data_type.map(str::to_string),
                data_type_name: data_type_name.map(str::to_string),
            }),
            ..Default::default()
        });
    }

    /// Mark the trace `writing` (first frame / first video chunk).
    pub fn mark_writing(&self, trace_id: &str) {
        self.enqueue(CoalescedTraceWrite {
            trace_id: trace_id.to_string(),
            write_status: Some(TraceWriteStatus::Writing),
            ..Default::default()
        });
    }

    /// Record the latest absolute on-disk byte count.
    pub fn progress(&self, trace_id: &str, bytes_written: i64) {
        self.enqueue(CoalescedTraceWrite {
            trace_id: trace_id.to_string(),
            bytes_written: Some(bytes_written),
            ..Default::default()
        });
    }

    /// Record the latest rolling upload offset (advisory progress). Coalesced
    /// like the write-phase progress so the uploader's per-64-MiB checkpoint
    /// across many concurrent uploads collapses to one batched row write
    /// instead of a synchronous transaction each. Resume correctness comes from
    /// the server's 308 offset, not this row, so a coalesced/late value is
    /// harmless; the store skips it once the upload has settled.
    pub fn upload_progress(&self, trace_id: &str, bytes_uploaded: i64) {
        self.enqueue(CoalescedTraceWrite {
            trace_id: trace_id.to_string(),
            bytes_uploaded: Some(bytes_uploaded),
            ..Default::default()
        });
    }

    /// Finalise the trace: `written`, with the final byte total.
    pub fn finalise(&self, trace_id: &str, total_bytes: i64) {
        self.enqueue(CoalescedTraceWrite {
            trace_id: trace_id.to_string(),
            write_status: Some(TraceWriteStatus::Written),
            total_bytes: Some(total_bytes),
            bytes_written: Some(total_bytes),
            ..Default::default()
        });
    }

    /// Mark the trace `failed`, preserving the latest byte count.
    pub fn fail(&self, trace_id: &str, bytes_written: i64) {
        self.enqueue(CoalescedTraceWrite {
            trace_id: trace_id.to_string(),
            write_status: Some(TraceWriteStatus::Failed),
            bytes_written: Some(bytes_written),
            ..Default::default()
        });
    }

    /// Mark the trace `failed` with a write-phase error code + message.
    #[allow(dead_code)]
    pub fn fail_with(
        &self,
        trace_id: &str,
        bytes_written: i64,
        error_code: TraceErrorCode,
        error_message: impl Into<String>,
    ) {
        self.enqueue(CoalescedTraceWrite {
            trace_id: trace_id.to_string(),
            write_status: Some(TraceWriteStatus::Failed),
            bytes_written: Some(bytes_written),
            error_code: Some(error_code),
            error_message: Some(error_message.into()),
            ..Default::default()
        });
    }

    /// Flush all pending writes and wait for the batch to commit. Used by
    /// tests and by callers that need a happens-before with the DB.
    pub async fn flush(&self) {
        let (ack, ack_rx) = oneshot::channel();
        if self.tx.send(Message::Flush(ack)).is_ok() {
            let _ = ack_rx.await;
        }
    }

    /// Discard pending creates for a recording and wait for the purge to
    /// complete. The dispatcher calls this before `cancel_recording` so a
    /// not-yet-flushed trace of a cancelled recording can't land as an orphan
    /// row after the cancel has burned the recording's existing traces.
    pub async fn drop_recording(&self, recording_index: i64) {
        let (ack, ack_rx) = oneshot::channel();
        if self
            .tx
            .send(Message::DropRecording {
                recording_index,
                ack,
            })
            .is_ok()
        {
            let _ = ack_rx.await;
        }
    }

    fn enqueue(&self, write: CoalescedTraceWrite) {
        // The channel only closes once the writer task has exited (daemon
        // shutdown). A drop here means we're past the point where writes
        // matter, so swallow it rather than propagate to the actor.
        let _ = self.tx.send(Message::Write(write));
    }
}

/// Owns the writer task's lifetime. Held by the daemon main loop; dropping it
/// does not stop the task (clones of the handle keep the channel open) — call
/// [`TraceEventDatabaseWriter::shutdown`] to drain, flush, and join.
pub struct TraceEventDatabaseWriter {
    tx: mpsc::UnboundedSender<Message>,
    join: JoinHandle<()>,
}

impl TraceEventDatabaseWriter {
    /// Drain every queued write, flush a final batch, and join the task. Call
    /// after the dispatcher (and therefore every actor) has shut down, so no
    /// further writes can be produced, and before the store is closed.
    pub async fn shutdown(self) {
        let (ack, ack_rx) = oneshot::channel();
        if self.tx.send(Message::Shutdown(ack)).is_ok() {
            let _ = ack_rx.await;
        }
        if let Err(error) = self.join.await {
            tracing::warn!(?error, "trace-writer task join failed during shutdown");
        }
    }
}

/// Spawn the writer task and return a cloneable [`TraceWriteHandle`] for the
/// actors plus the [`TraceEventDatabaseWriter`] owner for shutdown.
pub fn spawn(store: Arc<SqliteStateStore>) -> (TraceWriteHandle, TraceEventDatabaseWriter) {
    let (tx, rx) = mpsc::unbounded_channel();
    let join = tokio::spawn(run(store, rx));
    (
        TraceWriteHandle { tx: tx.clone() },
        TraceEventDatabaseWriter { tx, join },
    )
}

/// Merge one partial update into the pending set, last-writer-wins per column.
fn merge(pending: &mut HashMap<String, CoalescedTraceWrite>, write: CoalescedTraceWrite) {
    let entry = pending
        .entry(write.trace_id.clone())
        .or_insert_with(|| CoalescedTraceWrite {
            trace_id: write.trace_id.clone(),
            ..Default::default()
        });
    // `create` is set-once — it arrives on the first write and is immutable
    // thereafter (the row identity never changes).
    if write.create.is_some() && entry.create.is_none() {
        entry.create = write.create;
    }
    if write.write_status.is_some() {
        entry.write_status = write.write_status;
    }
    if write.bytes_written.is_some() {
        entry.bytes_written = write.bytes_written;
    }
    if write.total_bytes.is_some() {
        entry.total_bytes = write.total_bytes;
    }
    if write.bytes_uploaded.is_some() {
        entry.bytes_uploaded = write.bytes_uploaded;
    }
    // `error_code`/`error_message` are only ever set by `fail`, which is
    // mutually exclusive with `finalise` (a trace either fails or finalises, not
    // both), so a `written` status never coalesces with a stale error in the
    // same entry.
    if write.error_code.is_some() {
        entry.error_code = write.error_code;
    }
    if write.error_message.is_some() {
        entry.error_message = write.error_message;
    }
}

/// Discard pending entries that would *insert* a row for `recording_index`.
/// Update-only entries (whose create already flushed) are left: the terminal
/// guard in `apply_trace_writes` makes them no-ops against the cancelled row.
fn drop_recording_creates(
    pending: &mut HashMap<String, CoalescedTraceWrite>,
    recording_index: i64,
) {
    pending.retain(|_, write| {
        write
            .create
            .as_ref()
            .is_none_or(|create| create.recording_index != recording_index)
    });
}

/// Flush the pending set in one batched transaction, clearing it only on a
/// successful commit.
///
/// On error the drained batch is **re-merged** into `pending` rather than
/// discarded: dropping it loses a `finalise`/`failed`, which wedges the trace
/// in `writing` and retains its parent recording forever. Re-merging keeps the
/// updates for the next tick's retry and, because the merge is keyed by
/// `trace_id`, coalesces with any writes that arrived since — so a persistent
/// failure can't grow `pending` past the live trace count.
async fn flush(store: &SqliteStateStore, pending: &mut HashMap<String, CoalescedTraceWrite>) {
    if pending.is_empty() {
        return;
    }
    let batch: Vec<CoalescedTraceWrite> = pending.drain().map(|(_, write)| write).collect();
    if let Err(error) = store.apply_trace_writes(&batch).await {
        tracing::warn!(
            %error,
            rows = batch.len(),
            "trace-writer batch flush failed; re-queueing batch for retry"
        );
        for write in batch {
            merge(pending, write);
        }
    }
}

async fn run(store: Arc<SqliteStateStore>, mut rx: mpsc::UnboundedReceiver<Message>) {
    let mut pending: HashMap<String, CoalescedTraceWrite> = HashMap::new();
    let mut ticker = interval(FLUSH_INTERVAL);
    // A flush that runs long must not fire a backlog of catch-up ticks.
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            message = rx.recv() => match message {
                Some(Message::Write(write)) => {
                    merge(&mut pending, write);
                    if pending.len() >= MAX_PENDING_TRACES {
                        flush(&store, &mut pending).await;
                    }
                }
                Some(Message::DropRecording { recording_index, ack }) => {
                    drop_recording_creates(&mut pending, recording_index);
                    let _ = ack.send(());
                }
                Some(Message::Flush(ack)) => {
                    flush(&store, &mut pending).await;
                    let _ = ack.send(());
                }
                Some(Message::Shutdown(ack)) => {
                    // Drain anything already queued behind the Shutdown so no
                    // finalise is lost, then flush a last batch.
                    while let Ok(message) = rx.try_recv() {
                        match message {
                            Message::Write(write) => merge(&mut pending, write),
                            Message::DropRecording { recording_index, ack: inner } => {
                                drop_recording_creates(&mut pending, recording_index);
                                let _ = inner.send(());
                            }
                            Message::Flush(inner) => {
                                flush(&store, &mut pending).await;
                                let _ = inner.send(());
                            }
                            Message::Shutdown(inner) => {
                                let _ = inner.send(());
                            }
                        }
                    }
                    flush(&store, &mut pending).await;
                    let _ = ack.send(());
                    return;
                }
                // All handles dropped without an explicit shutdown — flush
                // whatever's left so a finalise isn't lost on an abrupt exit.
                None => {
                    flush(&store, &mut pending).await;
                    return;
                }
            },
            _ = ticker.tick() => {
                flush(&store, &mut pending).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::schema::TraceWriteStatus;
    use crate::state::store::NewRecording;
    use crate::state::StateStore;
    use tempfile::TempDir;

    async fn store_with_trace() -> (Arc<SqliteStateStore>, TempDir, String) {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(
            SqliteStateStore::open(&dir.path().join("state.db"))
                .await
                .unwrap(),
        );
        let rec = store
            .create_recording(NewRecording {
                robot_id: Some("r"),
                robot_instance: Some(0),
                start_timestamp_ns: 1,
                ..Default::default()
            })
            .await
            .unwrap()
            .recording_index;
        store
            .create_trace(rec, "t1", Some("J"), Some("j"))
            .await
            .unwrap();
        (store, dir, "t1".to_string())
    }

    #[tokio::test]
    async fn coalesces_progress_and_finalises() {
        let (store, _dir, trace_id) = store_with_trace().await;
        let (handle, writer) = spawn(store.clone());

        handle.mark_writing(&trace_id);
        for bytes in [10, 20, 30, 40] {
            handle.progress(&trace_id, bytes);
        }
        handle.finalise(&trace_id, 100);
        handle.flush().await;

        let trace = store.get_trace(&trace_id).await.unwrap().unwrap();
        assert_eq!(trace.write_status, TraceWriteStatus::Written);
        assert_eq!(trace.total_bytes, 100);
        assert_eq!(trace.bytes_written, 100);

        writer.shutdown().await;
    }

    #[tokio::test]
    async fn progress_does_not_resurrect_a_failed_row() {
        let (store, _dir, trace_id) = store_with_trace().await;
        let (handle, writer) = spawn(store.clone());

        // Simulate cancel burning the row to `failed` out of band.
        store
            .update_trace(
                &trace_id,
                crate::state::store::TraceUpdate {
                    write_status: Some(TraceWriteStatus::Failed),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // A late coalesced progress write must NOT move it back to writing.
        handle.progress(&trace_id, 999);
        handle.mark_writing(&trace_id);
        handle.flush().await;

        let trace = store.get_trace(&trace_id).await.unwrap().unwrap();
        assert_eq!(trace.write_status, TraceWriteStatus::Failed);

        writer.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_flushes_queued_writes() {
        let (store, _dir, trace_id) = store_with_trace().await;
        let (handle, writer) = spawn(store.clone());

        handle.finalise(&trace_id, 42);
        // No explicit flush — shutdown must drain and persist it.
        writer.shutdown().await;

        let trace = store.get_trace(&trace_id).await.unwrap().unwrap();
        assert_eq!(trace.write_status, TraceWriteStatus::Written);
        assert_eq!(trace.total_bytes, 42);
    }

    #[tokio::test]
    async fn flush_retains_batch_when_apply_fails() {
        let (store, _dir, trace_id) = store_with_trace().await;
        let mut pending = HashMap::new();
        merge(
            &mut pending,
            CoalescedTraceWrite {
                trace_id: trace_id.clone(),
                write_status: Some(TraceWriteStatus::Written),
                total_bytes: Some(99),
                ..Default::default()
            },
        );

        // Force apply_trace_writes to fail by closing the write connection.
        store.write_pool().close().await;
        flush(&store, &mut pending).await;

        // Regression guard for H2: a failed flush must NOT silently drop the
        // batch — a lost `finalise` would wedge the trace in `writing` and
        // retain its parent recording forever.
        assert_eq!(pending.len(), 1, "failed flush must retain the batch");
        let retained = pending.get(&trace_id).expect("batch retained for retry");
        assert_eq!(retained.write_status, Some(TraceWriteStatus::Written));
        assert_eq!(retained.total_bytes, Some(99));
    }

    async fn store_with_recording() -> (Arc<SqliteStateStore>, TempDir, i64) {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(
            SqliteStateStore::open(&dir.path().join("state.db"))
                .await
                .unwrap(),
        );
        let rec = store
            .create_recording(NewRecording {
                robot_id: Some("r"),
                robot_instance: Some(0),
                start_timestamp_ns: 1,
                ..Default::default()
            })
            .await
            .unwrap()
            .recording_index;
        (store, dir, rec)
    }

    #[tokio::test]
    async fn batched_create_inserts_then_finalises() {
        let (store, _dir, rec) = store_with_recording().await;
        let (handle, writer) = spawn(store.clone());

        // No synchronous create_trace — the row is born from the batch.
        handle.create("t-new", rec, Some("J"), Some("j"));
        handle.mark_writing("t-new");
        handle.progress("t-new", 64);
        handle.finalise("t-new", 128);
        handle.flush().await;

        let trace = store.get_trace("t-new").await.unwrap().unwrap();
        assert_eq!(trace.recording_index, rec);
        assert_eq!(trace.data_type.as_deref(), Some("J"));
        assert_eq!(trace.write_status, TraceWriteStatus::Written);
        assert_eq!(trace.total_bytes, 128);

        writer.shutdown().await;
    }

    #[tokio::test]
    async fn create_only_write_inserts_initializing_row() {
        let (store, _dir, rec) = store_with_recording().await;
        let (handle, writer) = spawn(store.clone());

        // A sensor that starts logging mid-recording: actor spawns, sends the
        // create, but no data has been appended before the flush.
        handle.create("t-mid", rec, Some("RGB"), Some("cam"));
        handle.flush().await;

        let trace = store.get_trace("t-mid").await.unwrap().unwrap();
        assert_eq!(trace.write_status, TraceWriteStatus::Initializing);
        assert_eq!(trace.recording_index, rec);

        writer.shutdown().await;
    }

    #[tokio::test]
    async fn drop_recording_discards_unflushed_create() {
        let (store, _dir, rec) = store_with_recording().await;
        let (handle, writer) = spawn(store.clone());

        // Create queued but NOT flushed, then the recording is cancelled.
        handle.create("t-cancel", rec, Some("J"), Some("j"));
        handle.mark_writing("t-cancel");
        handle.drop_recording(rec).await;
        handle.flush().await;

        // The orphan row must never have been inserted.
        assert!(store.get_trace("t-cancel").await.unwrap().is_none());

        writer.shutdown().await;
    }
}
