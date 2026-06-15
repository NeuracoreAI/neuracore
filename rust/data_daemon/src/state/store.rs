//! SQLite-backed implementation of the daemon's [`StateStore`].
//!
//! The persistence layer: schema migration via `sqlx`, WAL pragmas on every
//! connection, and the CRUD operations the per-trace actors and registration
//! coordinator rely on.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use chrono::Utc;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::{ConnectOptions, SqliteConnection, SqlitePool};
use thiserror::Error;

use crate::state::schema::{
    ProgressReportStatus, RecordingRow, TraceErrorCode, TraceRecord, TraceRegistrationStatus,
    TraceUploadStatus, TraceWriteStatus,
};

/// Embedded migrations, applied on every [`SqliteStateStore::open`].
static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Busy timeout (`PRAGMA busy_timeout`) applied to every connection.
///
/// Raised from 1 s: with the read/write pool split the daemon's own writers
/// can no longer collide (only one connection ever writes), but a background
/// WAL checkpoint or an external reader can still briefly hold the database
/// lock, and a 1 s ceiling under load surfaced as spurious `SQLITE_BUSY`
/// errors. 5 s comfortably absorbs those without masking a genuine deadlock.
const BUSY_TIMEOUT_MS: u32 = 5000;

/// Connections in the read pool. Only the cloud coordinators (a handful of
/// periodic tasks) read concurrently, so a small pool is ample; the writer
/// lives on its own single-connection pool and never competes for these.
const READ_POOL_CONNECTIONS: u32 = 4;

/// Errors surfaced by [`StateStore`] operations.
#[derive(Debug, Error)]
pub enum StateStoreError {
    /// Wrapped `sqlx` error from a query or migration.
    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),
    /// Wrapped migration error.
    #[error(transparent)]
    Migration(#[from] sqlx::migrate::MigrateError),
    /// Failed to create the SQLite parent directory.
    #[error("failed to prepare state directory {path}: {source}")]
    Io {
        /// Directory whose creation failed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Parameters for inserting a new recording row.
///
/// The daemon supplies the source identity and metadata from the
/// `StartRecording` envelope; the store allocates the `recording_index`.
#[derive(Debug, Clone, Default)]
pub struct NewRecording<'a> {
    /// Robot identifier — first half of the source key.
    pub robot_id: Option<&'a str>,
    /// Robot instance — second half of the source key.
    pub robot_instance: Option<i64>,
    /// Dataset identifier.
    pub dataset_id: Option<&'a str>,
    /// Producer capture-clock window lower bound (Unix nanoseconds).
    pub start_timestamp_ns: i64,
}

/// Persistence interface for daemon state.
///
/// Covers the operations the dispatcher, per-trace actors, and cloud
/// coordinators need: recording / trace lifecycle transitions, upload
/// bookkeeping, and reconciliation queries. Recordings are keyed by the local
/// `recording_index` the store allocates; the cloud `recording_id` is a
/// separate, nullable column filled asynchronously.
#[async_trait]
pub trait StateStore: Send + Sync {
    /// Insert a new recording row, allocating its `recording_index`, and
    /// return it. Each `StartRecording` envelope opens a distinct recording,
    /// so this always inserts (never upserts).
    async fn create_recording(
        &self,
        new: NewRecording<'_>,
    ) -> Result<RecordingRow, StateStoreError>;

    /// Fetch a recording by its local index, returning `None` when absent.
    async fn get_recording(
        &self,
        recording_index: i64,
    ) -> Result<Option<RecordingRow>, StateStoreError>;

    /// Return the most recently created recordings for a source, ordered by
    /// `recording_index` ascending. Used by the recovery sweep and the
    /// integration tests to correlate a recorded session to its DB rows.
    async fn recordings_for_source(
        &self,
        robot_id: &str,
        robot_instance: i64,
    ) -> Result<Vec<RecordingRow>, StateStoreError>;

    /// Stamp the cloud `recording_id` **and** `backend_start_notified_at`
    /// after the recording-start notifier successfully POSTed
    /// `/recording/start`. Idempotent.
    async fn mark_recording_start_notified(
        &self,
        recording_index: i64,
        recording_id: &str,
    ) -> Result<Option<RecordingRow>, StateStoreError>;

    /// List recordings whose `/recording/start` POST has not yet succeeded:
    /// `recording_id IS NULL`, `backend_start_notified_at IS NULL`, and the
    /// recording is not cancelled. The start notifier's startup sweep.
    async fn recordings_pending_start_notify(&self) -> Result<Vec<RecordingRow>, StateStoreError>;

    /// Insert a trace row in the [`TraceWriteStatus::Initializing`] state under
    /// an existing recording. Idempotent on `trace_id`.
    async fn create_trace(
        &self,
        recording_index: i64,
        trace_id: &str,
        data_type: Option<&str>,
        data_type_name: Option<&str>,
    ) -> Result<TraceRecord, StateStoreError>;

    /// Apply a partial update to an existing trace.
    ///
    /// Only set fields are written; unset fields preserve their existing value.
    /// Returns `Ok(())` whether or not a row matched — no caller consumes the
    /// updated row, so the previous mandatory read-back `SELECT` (a second
    /// round-trip on the contended write path) was pure waste.
    async fn update_trace(
        &self,
        trace_id: &str,
        update: TraceUpdate,
    ) -> Result<(), StateStoreError>;

    /// Fetch a trace by ID, returning `None` when absent.
    async fn get_trace(&self, trace_id: &str) -> Result<Option<TraceRecord>, StateStoreError>;

    /// Return all traces for the given recording, ordered by `created_at`.
    async fn list_traces_for_recording(
        &self,
        recording_index: i64,
    ) -> Result<Vec<TraceRecord>, StateStoreError>;

    /// Claim up to `limit` traces in [`TraceWriteStatus::Written`] /
    /// [`TraceRegistrationStatus::Pending`] for registration.
    ///
    /// Traces are eligible immediately when at least `limit` are ready (size
    /// trigger) or when their `last_updated` is older than `max_wait_secs`
    /// (age trigger) — the registration coordinator's debounce policy.
    ///
    /// Claimed rows are transitioned to
    /// [`TraceRegistrationStatus::Registering`] atomically inside a single
    /// transaction so two coordinators cannot double-claim.
    async fn claim_traces_for_registration(
        &self,
        limit: usize,
        max_wait_secs: f64,
    ) -> Result<Vec<TraceRecord>, StateStoreError>;

    /// Mark a recording as stopped, setting `stopped_at` (wall clock) and
    /// `stop_timestamp_ns` (producer capture clock).
    ///
    /// Idempotent: re-stopping a recording that already has a `stopped_at`
    /// leaves the existing timestamps untouched so a duplicate `StopRecording`
    /// envelope does not slide the window forward.
    async fn mark_recording_stopped(
        &self,
        recording_index: i64,
        stop_timestamp_ns: i64,
    ) -> Result<RecordingRow, StateStoreError>;

    /// Stamp `backend_stop_notified_at = now` after the recording-stop
    /// notifier successfully POSTed `/recording/stop`. Idempotent: a second
    /// call leaves the existing timestamp untouched.
    async fn mark_recording_stop_notified(
        &self,
        recording_index: i64,
    ) -> Result<Option<RecordingRow>, StateStoreError>;

    /// Stamp `backend_cancel_notified_at = now` after the recording-cancel
    /// notifier successfully POSTed `/recording/cancel`. Idempotent.
    async fn mark_recording_cancel_notified(
        &self,
        recording_index: i64,
    ) -> Result<Option<RecordingRow>, StateStoreError>;

    /// List recordings that have a cloud `recording_id` AND `cancelled_at IS
    /// NOT NULL` but whose backend cancel notification has not yet been
    /// delivered. Used by the recording-cancel notifier's startup sweep.
    async fn recordings_pending_cancel_notify(&self) -> Result<Vec<RecordingRow>, StateStoreError>;

    /// List recordings that have been stopped, have a cloud `recording_id`,
    /// but whose backend `/recording/stop` notification has not yet been
    /// delivered. Skips cancelled recordings and recordings whose `/start` was
    /// never notified (a NULL `recording_id` means there is nothing to stop
    /// server-side; the start notifier fills it first, then this sweep fires).
    async fn recordings_pending_stop_notify(&self) -> Result<Vec<RecordingRow>, StateStoreError>;

    /// List earlier recordings for `(robot_id, robot_instance)` that are still
    /// *pending* on the backend: they have a cloud `recording_id` (so they were
    /// opened server-side), are resolved locally (`cancelled_at` or `stopped_at`
    /// is set), but have not yet had their backend cancel/stop notification
    /// delivered. The backend dedupes pending recordings per robot instance —
    /// it returns the existing pending recording for an instance instead of
    /// minting a new one — so these must be closed server-side before the next
    /// recording's `/recording/start`, or that start reuses their cloud id.
    /// Restricted to `recording_index < before_index`, ordered oldest first.
    async fn recordings_pending_backend_resolution_for_source(
        &self,
        robot_id: &str,
        robot_instance: i64,
        before_index: i64,
    ) -> Result<Vec<RecordingRow>, StateStoreError>;

    /// List every recording row currently in the DB.
    ///
    /// Used by the progress reporter to discover stopped recordings whose
    /// traces have all finished uploading. Returned in `created_at` order.
    async fn list_recordings(&self) -> Result<Vec<RecordingRow>, StateStoreError>;

    /// List the `trace_id`s of every trace currently ready to upload
    /// (`upload_status` is `queued` or `retrying`).
    ///
    /// Server-side filtered via `idx_traces_upload_status` so the uploader's
    /// post-completion rescan does not walk every recording's full trace set
    /// (an O(recordings × traces) N+1 scan after each completed upload).
    async fn traces_ready_for_upload(&self) -> Result<Vec<String>, StateStoreError>;

    /// List recordings the progress reporter still has work for: stopped, not
    /// cancelled, with a cloud id, and not yet fully reported (either the
    /// progress report or the expected-trace-count PUT is still outstanding).
    ///
    /// Server-side filtered so fully-settled recordings drop out of the
    /// reporter's periodic sweep instead of being scanned (and their traces
    /// re-fetched) forever. Returned in `created_at` order.
    async fn recordings_pending_progress(&self) -> Result<Vec<RecordingRow>, StateStoreError>;

    /// List recordings the reaper can reclaim *now*: either cancelled with the
    /// backend cancel notified, or stopped + stop-notified + progress-reported
    /// with every declared trace uploaded (expected count met, none non-
    /// `uploaded`). The per-trace "all uploaded" test is folded into the query
    /// so the reaper neither walks every recording nor re-fetches the traces of
    /// a recording stuck on a permanently-failed upload on every 60 s sweep.
    /// Returned in `created_at` order.
    async fn recordings_pending_reclaim(&self) -> Result<Vec<RecordingRow>, StateStoreError>;

    /// Resolve the cloud `recording_id` for the recording identified by
    /// `(robot_id, robot_instance, start_timestamp_ns)`.
    ///
    /// Backs the `queries` IPC service: the SDK asks the daemon for the id
    /// instead of reading this DB directly. Matches `start_timestamp_ns`
    /// exactly (the producer's capture marker, stored verbatim) and excludes
    /// cancelled recordings, mirroring the previous client-side query. Returns
    /// `None` when no such recording exists or its cloud id has not been minted
    /// yet.
    async fn resolve_recording_id_for_marker(
        &self,
        robot_id: &str,
        robot_instance: i64,
        start_timestamp_ns: i64,
    ) -> Result<Option<String>, StateStoreError>;

    /// Atomically transition `progress_reported` for `recording_id`.
    ///
    /// `expected` is the status the caller observed before the request — if
    /// the row no longer matches (e.g. another tick already advanced it) the
    /// update is a no-op and the current row is returned. Returns `None` when
    /// the recording is not present.
    async fn set_progress_report_status(
        &self,
        recording_index: i64,
        expected: ProgressReportStatus,
        next: ProgressReportStatus,
    ) -> Result<Option<RecordingRow>, StateStoreError>;

    /// Stamp the recording's `expected_trace_count` once the producer-side
    /// trace set is known to be final. Idempotent: the value is only written
    /// when currently NULL so two reporters cannot race each other into
    /// inconsistent state.
    async fn set_expected_trace_count(
        &self,
        recording_index: i64,
        expected_trace_count: i64,
    ) -> Result<Option<RecordingRow>, StateStoreError>;

    /// Stamp the recording's `expected_trace_count_reported` to `count` once
    /// the backend has acknowledged the `expected-trace-count` PUT. Stored as
    /// a non-zero integer so the reporter can use a single column to mean
    /// both "reported" (non-zero) and "what we told the backend".
    async fn mark_expected_trace_count_reported(
        &self,
        recording_index: i64,
        count: i64,
    ) -> Result<Option<RecordingRow>, StateStoreError>;

    /// Re-arm pipeline rows that were mid-flight when the daemon last
    /// stopped. Mirrors `reset_retrying_to_written` in
    /// `state_store_sqlite.py`. Called on startup so a SIGKILL or panic does
    /// not leave traces wedged in transient `registering` / `uploading`
    /// states the coordinators no longer scan.
    ///
    /// Returns the number of trace rows rewritten.
    async fn reset_stale_pipeline_states(&self) -> Result<u64, StateStoreError>;

    /// Mark trace rows whose writer-side state is stale (`writing` /
    /// `initializing` / `pending_metadata`) and whose `last_updated` is older
    /// than `stale_threshold_secs` as `failed`.
    ///
    /// On startup these rows belong to a previous daemon process — by
    /// definition no current actor is touching them — and leaving them in a
    /// transient state would forever block their parent recording from
    /// reaching the "all traces written" gate the progress reporter waits on.
    /// The age threshold is a defence against accidentally clobbering a row
    /// that the current daemon has just begun writing (the row's
    /// `last_updated` is touched on creation, so a fresh row will not be
    /// caught by the sweep).
    ///
    /// Returns the number of trace rows rewritten.
    async fn mark_stale_writing_traces_failed(
        &self,
        stale_threshold_secs: i64,
    ) -> Result<u64, StateStoreError>;

    /// Atomically mark a recording as cancelled and burn every non-terminal
    /// trace it owns to terminal states the cloud coordinators ignore
    /// (`write_status = failed`, `upload_status = failed`,
    /// `registration_status = failed` if not already `registered`).
    ///
    /// A cancel is a recording stop that discards data, so it also stamps
    /// `stop_timestamp_ns` (the cancel's capture time, → backend `end_time`)
    /// just like [`mark_recording_stopped`](Self::mark_recording_stopped).
    ///
    /// Idempotent: re-cancelling a recording that already has a `cancelled_at`
    /// leaves both timestamps untouched. Returns the recording row after the
    /// update and the number of trace rows touched.
    async fn cancel_recording(
        &self,
        recording_index: i64,
        stop_timestamp_ns: i64,
    ) -> Result<(RecordingRow, u64), StateStoreError>;

    /// Delete a recording and all of its trace rows in a single transaction.
    ///
    /// Called by the recording reaper once a recording is fully settled (every
    /// trace uploaded and the backend notified), after its on-disk artefacts
    /// have been removed. Returns the number of trace rows deleted.
    async fn delete_recording_cascade(&self, recording_index: i64) -> Result<u64, StateStoreError>;
}

/// Optional fields to update on a trace row.
///
/// Fields left as `None` are not written. Use `Default::default()` and set
/// only the fields the caller intends to change.
#[derive(Debug, Clone, Default)]
pub struct TraceUpdate {
    /// New write lifecycle state.
    pub write_status: Option<TraceWriteStatus>,
    /// New registration lifecycle state.
    pub registration_status: Option<TraceRegistrationStatus>,
    /// New upload lifecycle state.
    pub upload_status: Option<TraceUploadStatus>,
    /// On-disk artefact path.
    pub path: Option<String>,
    /// Bytes written so far.
    pub bytes_written: Option<i64>,
    /// Final byte total (set on finalise).
    pub total_bytes: Option<i64>,
    /// Bytes uploaded so far.
    pub bytes_uploaded: Option<i64>,
    /// JSON-encoded `{filepath: session_uri}` map persisted by the
    /// registration coordinator.
    pub upload_session_uris: Option<String>,
    /// Set the latest error code (use `Some(None)` to clear, `None` to leave
    /// untouched).
    pub error_code: Option<Option<TraceErrorCode>>,
    /// Set the latest error message (use `Some(None)` to clear, `None` to
    /// leave untouched).
    pub error_message: Option<Option<String>>,
}

impl TraceUpdate {
    /// True when every field is unset and no SQL write is needed.
    fn is_empty(&self) -> bool {
        self.write_status.is_none()
            && self.registration_status.is_none()
            && self.upload_status.is_none()
            && self.path.is_none()
            && self.bytes_written.is_none()
            && self.total_bytes.is_none()
            && self.bytes_uploaded.is_none()
            && self.upload_session_uris.is_none()
            && self.error_code.is_none()
            && self.error_message.is_none()
    }
}

/// Row-creation fields for a trace, carried on the *first* coalesced write so
/// the batcher can insert the row instead of the actor blocking on a
/// synchronous `create_trace`. A trace is created the moment its actor spawns —
/// which can happen at any point in a recording (a sensor that starts logging
/// midway), not only at the boundary — so this rides every actor's first write.
#[derive(Debug, Clone)]
pub struct TraceCreate {
    /// Parent recording's local index.
    pub recording_index: i64,
    /// Wire data-type label.
    pub data_type: Option<String>,
    /// Per-stream sensor label (persisted to `data_type_name`).
    pub data_type_name: Option<String>,
}

/// One coalesced set of actor-owned column updates for a single trace — the
/// unit the [`crate::state::trace_writer::TraceWriter`] batcher flushes.
///
/// The per-trace actor owns the write-phase columns (`write_status`, the
/// on-disk byte counter, `total_bytes`, the write-phase error). The uploader's
/// rolling `bytes_uploaded` checkpoint is also coalesced here — it is the one
/// cloud-owned column that is hot (one write per 64 MiB per concurrent upload),
/// purely advisory (resume correctness comes from the server's 308 offset, not
/// this row), and last-writer-wins, so it fits the write-behind exactly. The
/// remaining cloud columns (`upload_status`, `registration_status`,
/// `upload_session_uris`, `path`) stay on the synchronous write path: they gate
/// reads (e.g. `traces_ready_for_upload`) and must be ordered against the bus
/// events the coordinators publish, so deferring them would reintroduce
/// read-after-write races. `None` fields are left untouched
/// (`COALESCE`-preserved). `create` is set only on the trace's first write and
/// triggers an idempotent row insert.
#[derive(Debug, Clone, Default)]
pub struct CoalescedTraceWrite {
    /// Target trace row.
    pub trace_id: String,
    /// Row-creation fields, present only on the trace's first coalesced write.
    pub create: Option<TraceCreate>,
    /// New write-lifecycle state, if it changed.
    pub write_status: Option<TraceWriteStatus>,
    /// Latest absolute on-disk byte count.
    pub bytes_written: Option<i64>,
    /// Final byte total (set on finalise).
    pub total_bytes: Option<i64>,
    /// Latest rolling upload offset (advisory progress checkpoint).
    pub bytes_uploaded: Option<i64>,
    /// Write-phase error code.
    pub error_code: Option<TraceErrorCode>,
    /// Write-phase error message.
    pub error_message: Option<String>,
}

impl CoalescedTraceWrite {
    /// True when a write-phase column changed and the guarded write-phase
    /// UPDATE must run. A create-only write (and an upload-progress-only write)
    /// skips it; `bytes_uploaded` is applied by its own statement because it
    /// updates legitimately *after* the row has reached the terminal
    /// `written` state, which the write-phase guard forbids.
    fn has_write_column_update(&self) -> bool {
        self.write_status.is_some()
            || self.bytes_written.is_some()
            || self.total_bytes.is_some()
            || self.error_code.is_some()
            || self.error_message.is_some()
    }
}

/// SQLite-backed [`StateStore`].
///
/// Writers serialise on a dedicated single-connection [`write_pool`], so the
/// daemon needs no app-level write mutex: only one connection ever writes, so
/// WAL never returns `SQLITE_BUSY` between the daemon's own writers and nothing
/// is held across a transaction's `.await`s. The earlier `Arc<Mutex<()>>`
/// guard (mirroring `state_store_sqlite.py`'s `asyncio.Semaphore(1)`) was held
/// across `begin..commit` and dominated the DB-layer profile (~96% of time was
/// spent waiting on it under load); letting the single write connection
/// serialise writers removes that wait entirely.
///
/// Reads run on a separate multi-connection [`read_pool`] so they never queue
/// behind the writer — under WAL a reader observes a consistent snapshot while
/// the writer commits.
///
/// [`write_pool`]: SqliteStateStore::write_pool
/// [`read_pool`]: SqliteStateStore::read_pool
#[derive(Clone)]
pub struct SqliteStateStore {
    read_pool: SqlitePool,
    write_pool: SqlitePool,
}

impl SqliteStateStore {
    /// Open the SQLite database at `db_path`, creating it (and the parent
    /// directory) if missing, applying WAL pragmas, and running pending
    /// migrations.
    pub async fn open(db_path: &Path) -> Result<Self, StateStoreError> {
        if let Some(parent) = db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|source| StateStoreError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
        }

        let options = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS as u64))
            // sqlx prints every statement at INFO by default; quiet that down so
            // the daemon's tracing output isn't drowned out by the same SQL on
            // every trace write.
            .log_statements(tracing::log::LevelFilter::Debug);

        // Single writer: serialises commits without an app-level mutex and makes
        // SQLITE_BUSY between the daemon's own writers impossible.
        let write_pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options.clone())
            .await?;
        // Reads run concurrently on their own pool so they never wait behind the
        // writer.
        let read_pool = SqlitePoolOptions::new()
            .max_connections(READ_POOL_CONNECTIONS)
            .connect_with(options)
            .await?;

        // Migrations are writes — run them on the write connection.
        MIGRATOR.run(&write_pool).await?;

        Ok(SqliteStateStore {
            read_pool,
            write_pool,
        })
    }

    /// Borrow the read pool, e.g. for diagnostics in tests.
    pub fn pool(&self) -> &SqlitePool {
        &self.read_pool
    }

    /// Borrow the write pool. Test-only: production code reaches the write
    /// connection exclusively through the typed write methods.
    #[cfg(test)]
    pub(crate) fn write_pool(&self) -> &SqlitePool {
        &self.write_pool
    }

    /// Close both pools, draining outstanding connections.
    pub async fn close(self) {
        self.write_pool.close().await;
        self.read_pool.close().await;
    }

    /// Apply a batch of coalesced per-trace writes in a single transaction.
    ///
    /// The [`crate::state::trace_writer::TraceWriter`] batcher coalesces many
    /// actors' per-frame progress / status / finalise updates (last-writer-wins
    /// per trace) and flushes them together here, so the per-transaction cost
    /// and the `write_guard` acquisition are amortised across the whole batch
    /// instead of paid per row. Each SET clause is a fixed `COALESCE(?, col)`
    /// form (only supplied columns change; statement stays prepared-cache
    /// friendly) and there is no read-back `SELECT`.
    ///
    /// The write-phase `WHERE` guard keys off the *target* lifecycle state so a
    /// flush is monotonic w.r.t. terminal states — a late coalesced progress
    /// write can never resurrect a row a concurrent
    /// [`StateStore::cancel_recording`] already burned to `failed`, nor
    /// un-finish a `written` row. That invariant is what lets the batcher run
    /// without tight coordination with the cancel path. The three guard
    /// variants are fixed `const` statements (not `format!`-ed per row) so they
    /// stay friendly to sqlx's prepared-statement cache.
    pub async fn apply_trace_writes(
        &self,
        writes: &[CoalescedTraceWrite],
    ) -> Result<(), StateStoreError> {
        // The write-phase UPDATE, one fixed statement per terminal guard so
        // sqlx's prepared-statement cache sees the same SQL every flush. Bind
        // order is identical across the three: ?1 write_status, ?2
        // bytes_written, ?3 total_bytes, ?4 error_code, ?5 error_message, ?6
        // last_updated, ?7 trace_id; only the trailing guard differs.
        //
        // Finalise: apply unless cancel already burned the row.
        const SQL_WRITE_FINALISE: &str = "UPDATE traces SET \
             write_status = COALESCE(?1, write_status), \
             bytes_written = COALESCE(?2, bytes_written), \
             total_bytes = COALESCE(?3, total_bytes), \
             error_code = COALESCE(?4, error_code), \
             error_message = COALESCE(?5, error_message), \
             last_updated = ?6 \
           WHERE trace_id = ?7 AND write_status != 'failed'";
        // Fail: apply unless the trace already finished writing.
        const SQL_WRITE_FAIL: &str = "UPDATE traces SET \
             write_status = COALESCE(?1, write_status), \
             bytes_written = COALESCE(?2, bytes_written), \
             total_bytes = COALESCE(?3, total_bytes), \
             error_code = COALESCE(?4, error_code), \
             error_message = COALESCE(?5, error_message), \
             last_updated = ?6 \
           WHERE trace_id = ?7 AND write_status != 'written'";
        // Progress / Writing / byte-only: live (non-terminal) rows only.
        const SQL_WRITE_PROGRESS: &str = "UPDATE traces SET \
             write_status = COALESCE(?1, write_status), \
             bytes_written = COALESCE(?2, bytes_written), \
             total_bytes = COALESCE(?3, total_bytes), \
             error_code = COALESCE(?4, error_code), \
             error_message = COALESCE(?5, error_message), \
             last_updated = ?6 \
           WHERE trace_id = ?7 AND write_status NOT IN ('written', 'failed')";
        // The advisory upload checkpoint. Updates while the upload is live;
        // skipped once it has settled (`uploaded`/`failed`) so a late flush
        // can't touch a terminal row. Bind order: ?1 bytes_uploaded,
        // ?2 last_updated, ?3 trace_id.
        const SQL_UPLOAD_PROGRESS: &str = "UPDATE traces SET \
             bytes_uploaded = ?1, last_updated = ?2 \
           WHERE trace_id = ?3 AND upload_status NOT IN ('uploaded', 'failed')";

        if writes.is_empty() {
            return Ok(());
        }
        let mut tx = self.write_pool.begin().await?;
        let now = Utc::now().naive_utc();
        for write in writes {
            // First write for a trace carries its create fields: insert the row
            // (idempotent) before applying any column update. This folds the
            // per-actor `create_trace` into the batch, so the boundary's spawn
            // burst becomes one batched transaction instead of N.
            if let Some(create) = &write.create {
                sqlx::query(
                    "INSERT INTO traces (trace_id, recording_index, write_status, \
                                         data_type, data_type_name, created_at, last_updated) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6) \
                     ON CONFLICT(trace_id) DO NOTHING",
                )
                .bind(&write.trace_id)
                .bind(create.recording_index)
                .bind(TraceWriteStatus::Initializing.as_str())
                .bind(create.data_type.as_deref())
                .bind(create.data_type_name.as_deref())
                .bind(now)
                .execute(&mut *tx)
                .await?;
            }

            if write.has_write_column_update() {
                let sql = match write.write_status {
                    Some(TraceWriteStatus::Written) => SQL_WRITE_FINALISE,
                    Some(TraceWriteStatus::Failed) => SQL_WRITE_FAIL,
                    _ => SQL_WRITE_PROGRESS,
                };
                sqlx::query(sql)
                    .bind(write.write_status.as_ref().map(|status| status.as_str()))
                    .bind(write.bytes_written)
                    .bind(write.total_bytes)
                    .bind(write.error_code.as_ref().map(|code| code.as_str()))
                    .bind(write.error_message.as_deref())
                    .bind(now)
                    .bind(&write.trace_id)
                    .execute(&mut *tx)
                    .await?;
            }

            if let Some(bytes_uploaded) = write.bytes_uploaded {
                sqlx::query(SQL_UPLOAD_PROGRESS)
                    .bind(bytes_uploaded)
                    .bind(now)
                    .bind(&write.trace_id)
                    .execute(&mut *tx)
                    .await?;
            }
        }
        tx.commit().await?;
        Ok(())
    }

    /// Fetch a recording row by its local index inside an open connection.
    async fn fetch_recording_locked(
        conn: &mut SqliteConnection,
        recording_index: i64,
    ) -> Result<Option<RecordingRow>, sqlx::Error> {
        let row = sqlx::query("SELECT * FROM recordings WHERE recording_index = ?1")
            .bind(recording_index)
            .fetch_optional(&mut *conn)
            .await?;
        match row {
            Some(row) => Ok(Some(RecordingRow::from_row(&row)?)),
            None => Ok(None),
        }
    }
}

#[async_trait]
impl StateStore for SqliteStateStore {
    async fn create_recording(
        &self,
        new: NewRecording<'_>,
    ) -> Result<RecordingRow, StateStoreError> {
        let mut tx = self.write_pool.begin().await?;
        let now = Utc::now().naive_utc();
        let result = sqlx::query(
            "INSERT INTO recordings ( \
                 robot_id, robot_instance, dataset_id, \
                 start_timestamp_ns, created_at, last_updated \
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
        )
        .bind(new.robot_id)
        .bind(new.robot_instance)
        .bind(new.dataset_id)
        .bind(new.start_timestamp_ns)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        let recording_index = result.last_insert_rowid();
        let row = Self::fetch_recording_locked(&mut tx, recording_index)
            .await?
            .ok_or_else(|| sqlx::Error::RowNotFound)?;
        tx.commit().await?;
        Ok(row)
    }

    async fn get_recording(
        &self,
        recording_index: i64,
    ) -> Result<Option<RecordingRow>, StateStoreError> {
        let row = sqlx::query("SELECT * FROM recordings WHERE recording_index = ?1")
            .bind(recording_index)
            .fetch_optional(&self.read_pool)
            .await?;
        Ok(match row {
            Some(row) => Some(RecordingRow::from_row(&row)?),
            None => None,
        })
    }

    async fn recordings_for_source(
        &self,
        robot_id: &str,
        robot_instance: i64,
    ) -> Result<Vec<RecordingRow>, StateStoreError> {
        let rows = sqlx::query(
            "SELECT * FROM recordings \
              WHERE robot_id = ?1 AND robot_instance = ?2 \
           ORDER BY recording_index ASC",
        )
        .bind(robot_id)
        .bind(robot_instance)
        .fetch_all(&self.read_pool)
        .await?;
        rows.iter()
            .map(RecordingRow::from_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    async fn mark_recording_start_notified(
        &self,
        recording_index: i64,
        recording_id: &str,
    ) -> Result<Option<RecordingRow>, StateStoreError> {
        let mut tx = self.write_pool.begin().await?;
        let now = Utc::now().naive_utc();
        sqlx::query(
            "UPDATE recordings \
                SET recording_id = COALESCE(recording_id, ?1), \
                    backend_start_notified_at = COALESCE(backend_start_notified_at, ?2), \
                    last_updated = ?2 \
              WHERE recording_index = ?3",
        )
        .bind(recording_id)
        .bind(now)
        .bind(recording_index)
        .execute(&mut *tx)
        .await?;
        let row = Self::fetch_recording_locked(&mut tx, recording_index).await?;
        tx.commit().await?;
        Ok(row)
    }

    async fn recordings_pending_start_notify(&self) -> Result<Vec<RecordingRow>, StateStoreError> {
        let rows = sqlx::query(
            "SELECT * FROM recordings \
              WHERE recording_id IS NULL \
                AND backend_start_notified_at IS NULL \
                AND cancelled_at IS NULL \
           ORDER BY recording_index ASC",
        )
        .fetch_all(&self.read_pool)
        .await?;
        rows.iter()
            .map(RecordingRow::from_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    async fn create_trace(
        &self,
        recording_index: i64,
        trace_id: &str,
        data_type: Option<&str>,
        data_type_name: Option<&str>,
    ) -> Result<TraceRecord, StateStoreError> {
        let mut tx = self.write_pool.begin().await?;

        let now = Utc::now().naive_utc();
        sqlx::query(
            "INSERT INTO traces (trace_id, recording_index, write_status, data_type, \
                                 data_type_name, created_at, last_updated) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6) \
             ON CONFLICT(trace_id) DO NOTHING",
        )
        .bind(trace_id)
        .bind(recording_index)
        .bind(TraceWriteStatus::Initializing.as_str())
        .bind(data_type)
        .bind(data_type_name)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        let row = sqlx::query("SELECT * FROM traces WHERE trace_id = ?1")
            .bind(trace_id)
            .fetch_one(&mut *tx)
            .await?;
        let record = TraceRecord::from_row(&row)?;

        tx.commit().await?;
        Ok(record)
    }

    async fn update_trace(
        &self,
        trace_id: &str,
        update: TraceUpdate,
    ) -> Result<(), StateStoreError> {
        if update.is_empty() {
            return Ok(());
        }

        let mut tx = self.write_pool.begin().await?;

        // Build the UPDATE dynamically so we only touch fields the caller set.
        // Always bump `last_updated` so the registration coordinator's
        // age-based claim policy sees fresh timestamps.
        let now = Utc::now().naive_utc();
        let mut assignments: Vec<&'static str> = Vec::new();
        if update.write_status.is_some() {
            assignments.push("write_status = ?");
        }
        if update.registration_status.is_some() {
            assignments.push("registration_status = ?");
        }
        if update.upload_status.is_some() {
            assignments.push("upload_status = ?");
        }
        if update.path.is_some() {
            assignments.push("path = ?");
        }
        if update.bytes_written.is_some() {
            assignments.push("bytes_written = ?");
        }
        if update.total_bytes.is_some() {
            assignments.push("total_bytes = ?");
        }
        if update.bytes_uploaded.is_some() {
            assignments.push("bytes_uploaded = ?");
        }
        if update.upload_session_uris.is_some() {
            assignments.push("upload_session_uris = ?");
        }
        if update.error_code.is_some() {
            assignments.push("error_code = ?");
        }
        if update.error_message.is_some() {
            assignments.push("error_message = ?");
        }
        assignments.push("last_updated = ?");

        let sql = format!(
            "UPDATE traces SET {} WHERE trace_id = ?",
            assignments.join(", ")
        );
        let mut query = sqlx::query(&sql);
        if let Some(status) = update.write_status {
            query = query.bind(status.as_str());
        }
        if let Some(status) = update.registration_status {
            query = query.bind(status.as_str());
        }
        if let Some(status) = update.upload_status {
            query = query.bind(status.as_str());
        }
        if let Some(path) = update.path {
            query = query.bind(path);
        }
        if let Some(bytes) = update.bytes_written {
            query = query.bind(bytes);
        }
        if let Some(bytes) = update.total_bytes {
            query = query.bind(bytes);
        }
        if let Some(bytes) = update.bytes_uploaded {
            query = query.bind(bytes);
        }
        if let Some(uris) = update.upload_session_uris {
            query = query.bind(uris);
        }
        if let Some(code) = update.error_code {
            query = query.bind(code.map(|value| value.as_str().to_string()));
        }
        if let Some(message) = update.error_message {
            query = query.bind(message);
        }
        query = query.bind(now).bind(trace_id);

        // No read-back: a non-matching `trace_id` is a harmless no-op UPDATE
        // (0 rows affected) and no caller distinguishes it from a hit.
        query.execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(())
    }

    async fn get_trace(&self, trace_id: &str) -> Result<Option<TraceRecord>, StateStoreError> {
        let row = sqlx::query("SELECT * FROM traces WHERE trace_id = ?1")
            .bind(trace_id)
            .fetch_optional(&self.read_pool)
            .await?;
        Ok(match row {
            Some(row) => Some(TraceRecord::from_row(&row)?),
            None => None,
        })
    }

    async fn list_traces_for_recording(
        &self,
        recording_index: i64,
    ) -> Result<Vec<TraceRecord>, StateStoreError> {
        let rows = sqlx::query(
            "SELECT * FROM traces WHERE recording_index = ?1 ORDER BY created_at ASC, trace_id ASC",
        )
        .bind(recording_index)
        .fetch_all(&self.read_pool)
        .await?;
        rows.iter()
            .map(TraceRecord::from_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    async fn claim_traces_for_registration(
        &self,
        limit: usize,
        max_wait_secs: f64,
    ) -> Result<Vec<TraceRecord>, StateStoreError> {
        let limit_i64 = limit as i64;
        if limit_i64 <= 0 {
            return Ok(Vec::new());
        }
        let mut tx = self.write_pool.begin().await?;

        // Count ready traces first so the size-vs-age policy stays explicit.
        // SQLite's transactional snapshot means this count is stable across the
        // subsequent claim.
        let ready_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM traces \
              WHERE write_status = ?1 AND registration_status = ?2",
        )
        .bind(TraceWriteStatus::Written.as_str())
        .bind(TraceRegistrationStatus::Pending.as_str())
        .fetch_one(&mut *tx)
        .await?;

        // Select the eligible trace_ids (oldest first, capped at `limit`) and
        // flip them to `registering` in a single `UPDATE … RETURNING`, so two
        // coordinators can't double-claim and there's neither a separate SELECT
        // round-trip nor a per-row UPDATE loop. The `IN (subquery)` form is used
        // because SQLite's stock build doesn't support `LIMIT` directly on
        // `UPDATE`; the subquery materialises before the UPDATE runs, so the
        // claim sees a stable candidate set.
        let now = Utc::now().naive_utc();
        let rows = if ready_count >= limit_i64 {
            // Size trigger: enough are ready — claim the oldest `limit`
            // regardless of age.
            sqlx::query(
                "UPDATE traces SET registration_status = ?1, last_updated = ?2 \
                  WHERE trace_id IN ( \
                      SELECT trace_id FROM traces \
                       WHERE write_status = ?3 AND registration_status = ?4 \
                    ORDER BY last_updated ASC LIMIT ?5 \
                  ) \
                RETURNING *",
            )
            .bind(TraceRegistrationStatus::Registering.as_str())
            .bind(now)
            .bind(TraceWriteStatus::Written.as_str())
            .bind(TraceRegistrationStatus::Pending.as_str())
            .bind(limit_i64)
            .fetch_all(&mut *tx)
            .await?
        } else {
            // Age trigger: claim only those whose `last_updated` is older than
            // the debounce cutoff.
            let cutoff = now - chrono::Duration::milliseconds((max_wait_secs * 1000.0) as i64);
            sqlx::query(
                "UPDATE traces SET registration_status = ?1, last_updated = ?2 \
                  WHERE trace_id IN ( \
                      SELECT trace_id FROM traces \
                       WHERE write_status = ?3 AND registration_status = ?4 \
                         AND last_updated <= ?5 \
                    ORDER BY last_updated ASC LIMIT ?6 \
                  ) \
                RETURNING *",
            )
            .bind(TraceRegistrationStatus::Registering.as_str())
            .bind(now)
            .bind(TraceWriteStatus::Written.as_str())
            .bind(TraceRegistrationStatus::Pending.as_str())
            .bind(cutoff)
            .bind(limit_i64)
            .fetch_all(&mut *tx)
            .await?
        };

        tx.commit().await?;
        rows.iter()
            .map(TraceRecord::from_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    async fn mark_recording_stopped(
        &self,
        recording_index: i64,
        stop_timestamp_ns: i64,
    ) -> Result<RecordingRow, StateStoreError> {
        let mut tx = self.write_pool.begin().await?;

        let now = Utc::now().naive_utc();
        sqlx::query(
            "UPDATE recordings \
                SET stopped_at = COALESCE(stopped_at, ?2), \
                    stop_timestamp_ns = COALESCE(stop_timestamp_ns, ?3), \
                    last_updated = ?2 \
              WHERE recording_index = ?1",
        )
        .bind(recording_index)
        .bind(now)
        .bind(stop_timestamp_ns)
        .execute(&mut *tx)
        .await?;

        let record = Self::fetch_recording_locked(&mut tx, recording_index)
            .await?
            .ok_or(sqlx::Error::RowNotFound)?;

        tx.commit().await?;
        Ok(record)
    }

    async fn list_recordings(&self) -> Result<Vec<RecordingRow>, StateStoreError> {
        let rows = sqlx::query("SELECT * FROM recordings ORDER BY created_at ASC")
            .fetch_all(&self.read_pool)
            .await?;
        rows.iter()
            .map(RecordingRow::from_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    async fn traces_ready_for_upload(&self) -> Result<Vec<String>, StateStoreError> {
        let ids = sqlx::query_scalar::<_, String>(
            "SELECT trace_id FROM traces WHERE upload_status IN ('queued', 'retrying')",
        )
        .fetch_all(&self.read_pool)
        .await?;
        Ok(ids)
    }

    async fn recordings_pending_progress(&self) -> Result<Vec<RecordingRow>, StateStoreError> {
        let rows = sqlx::query(
            "SELECT * FROM recordings \
              WHERE stopped_at IS NOT NULL \
                AND cancelled_at IS NULL \
                AND recording_id IS NOT NULL \
                AND (progress_reported != 'reported' OR expected_trace_count_reported = 0) \
           ORDER BY created_at ASC",
        )
        .fetch_all(&self.read_pool)
        .await?;
        rows.iter()
            .map(RecordingRow::from_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    async fn recordings_pending_reclaim(&self) -> Result<Vec<RecordingRow>, StateStoreError> {
        let rows = sqlx::query(
            "SELECT r.* FROM recordings r \
              WHERE (r.cancelled_at IS NOT NULL AND r.backend_cancel_notified_at IS NOT NULL) \
                 OR ( \
                     r.cancelled_at IS NULL \
                     AND r.stopped_at IS NOT NULL \
                     AND r.backend_stop_notified_at IS NOT NULL \
                     AND r.progress_reported = 'reported' \
                     AND r.expected_trace_count IS NOT NULL \
                     AND r.expected_trace_count = \
                         (SELECT COUNT(*) FROM traces t WHERE t.recording_index = r.recording_index) \
                     AND EXISTS \
                         (SELECT 1 FROM traces t WHERE t.recording_index = r.recording_index) \
                     AND NOT EXISTS \
                         (SELECT 1 FROM traces t \
                           WHERE t.recording_index = r.recording_index \
                             AND t.upload_status != 'uploaded') \
                 ) \
           ORDER BY r.created_at ASC",
        )
        .fetch_all(&self.read_pool)
        .await?;
        rows.iter()
            .map(RecordingRow::from_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    async fn resolve_recording_id_for_marker(
        &self,
        robot_id: &str,
        robot_instance: i64,
        start_timestamp_ns: i64,
    ) -> Result<Option<String>, StateStoreError> {
        let recording_id = sqlx::query_scalar::<_, Option<String>>(
            "SELECT recording_id FROM recordings \
              WHERE robot_id = ?1 AND robot_instance = ?2 AND start_timestamp_ns = ?3 \
                AND cancelled_at IS NULL \
           ORDER BY recording_index DESC LIMIT 1",
        )
        .bind(robot_id)
        .bind(robot_instance)
        .bind(start_timestamp_ns)
        .fetch_optional(&self.read_pool)
        .await?;
        // Outer `Option` = row present; inner = the nullable column. A matching
        // row whose cloud id has not been minted yet flattens to `None`.
        Ok(recording_id.flatten())
    }

    async fn mark_recording_stop_notified(
        &self,
        recording_index: i64,
    ) -> Result<Option<RecordingRow>, StateStoreError> {
        let mut tx = self.write_pool.begin().await?;
        let now = Utc::now().naive_utc();
        sqlx::query(
            "UPDATE recordings \
                SET backend_stop_notified_at = COALESCE(backend_stop_notified_at, ?2), \
                    last_updated = ?2 \
              WHERE recording_index = ?1",
        )
        .bind(recording_index)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        let record = Self::fetch_recording_locked(&mut tx, recording_index).await?;
        tx.commit().await?;
        Ok(record)
    }

    async fn mark_recording_cancel_notified(
        &self,
        recording_index: i64,
    ) -> Result<Option<RecordingRow>, StateStoreError> {
        let mut tx = self.write_pool.begin().await?;
        let now = Utc::now().naive_utc();
        sqlx::query(
            "UPDATE recordings \
                SET backend_cancel_notified_at = COALESCE(backend_cancel_notified_at, ?2), \
                    last_updated = ?2 \
              WHERE recording_index = ?1",
        )
        .bind(recording_index)
        .bind(now)
        .execute(&mut *tx)
        .await?;
        let record = Self::fetch_recording_locked(&mut tx, recording_index).await?;
        tx.commit().await?;
        Ok(record)
    }

    async fn recordings_pending_cancel_notify(&self) -> Result<Vec<RecordingRow>, StateStoreError> {
        let rows = sqlx::query(
            "SELECT * FROM recordings \
              WHERE cancelled_at IS NOT NULL \
                AND recording_id IS NOT NULL \
                AND backend_cancel_notified_at IS NULL \
           ORDER BY cancelled_at ASC",
        )
        .fetch_all(&self.read_pool)
        .await?;
        rows.iter()
            .map(RecordingRow::from_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    async fn recordings_pending_stop_notify(&self) -> Result<Vec<RecordingRow>, StateStoreError> {
        let rows = sqlx::query(
            "SELECT * FROM recordings \
              WHERE stopped_at IS NOT NULL \
                AND recording_id IS NOT NULL \
                AND backend_stop_notified_at IS NULL \
                AND cancelled_at IS NULL \
           ORDER BY stopped_at ASC",
        )
        .fetch_all(&self.read_pool)
        .await?;
        rows.iter()
            .map(RecordingRow::from_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    async fn recordings_pending_backend_resolution_for_source(
        &self,
        robot_id: &str,
        robot_instance: i64,
        before_index: i64,
    ) -> Result<Vec<RecordingRow>, StateStoreError> {
        let rows = sqlx::query(
            "SELECT * FROM recordings \
              WHERE robot_id = ? \
                AND robot_instance = ? \
                AND recording_index < ? \
                AND recording_id IS NOT NULL \
                AND backend_stop_notified_at IS NULL \
                AND backend_cancel_notified_at IS NULL \
                AND (cancelled_at IS NOT NULL OR stopped_at IS NOT NULL) \
           ORDER BY recording_index ASC",
        )
        .bind(robot_id)
        .bind(robot_instance)
        .bind(before_index)
        .fetch_all(&self.read_pool)
        .await?;
        rows.iter()
            .map(RecordingRow::from_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    async fn set_progress_report_status(
        &self,
        recording_index: i64,
        expected: ProgressReportStatus,
        next: ProgressReportStatus,
    ) -> Result<Option<RecordingRow>, StateStoreError> {
        let mut tx = self.write_pool.begin().await?;
        let now = Utc::now().naive_utc();
        sqlx::query(
            "UPDATE recordings \
                SET progress_reported = ?1, last_updated = ?2 \
              WHERE recording_index = ?3 AND progress_reported = ?4",
        )
        .bind(next.as_str())
        .bind(now)
        .bind(recording_index)
        .bind(expected.as_str())
        .execute(&mut *tx)
        .await?;

        let record = Self::fetch_recording_locked(&mut tx, recording_index).await?;
        tx.commit().await?;
        Ok(record)
    }

    async fn set_expected_trace_count(
        &self,
        recording_index: i64,
        expected_trace_count: i64,
    ) -> Result<Option<RecordingRow>, StateStoreError> {
        let mut tx = self.write_pool.begin().await?;
        let now = Utc::now().naive_utc();
        sqlx::query(
            "UPDATE recordings \
                SET expected_trace_count = COALESCE(expected_trace_count, ?1), \
                    last_updated = ?2 \
              WHERE recording_index = ?3",
        )
        .bind(expected_trace_count)
        .bind(now)
        .bind(recording_index)
        .execute(&mut *tx)
        .await?;
        let record = Self::fetch_recording_locked(&mut tx, recording_index).await?;
        tx.commit().await?;
        Ok(record)
    }

    async fn mark_expected_trace_count_reported(
        &self,
        recording_index: i64,
        count: i64,
    ) -> Result<Option<RecordingRow>, StateStoreError> {
        let mut tx = self.write_pool.begin().await?;
        let now = Utc::now().naive_utc();
        sqlx::query(
            "UPDATE recordings \
                SET expected_trace_count_reported = ?1, \
                    last_updated = ?2 \
              WHERE recording_index = ?3",
        )
        .bind(count)
        .bind(now)
        .bind(recording_index)
        .execute(&mut *tx)
        .await?;
        let record = Self::fetch_recording_locked(&mut tx, recording_index).await?;
        tx.commit().await?;
        Ok(record)
    }

    async fn reset_stale_pipeline_states(&self) -> Result<u64, StateStoreError> {
        let mut tx = self.write_pool.begin().await?;
        let now = Utc::now().naive_utc();
        // `registering` → `pending` so the registration coordinator's claim
        // query sees the row again on the next tick.
        let reg_result = sqlx::query(
            "UPDATE traces \
                SET registration_status = ?1, last_updated = ?2 \
              WHERE registration_status = ?3",
        )
        .bind(TraceRegistrationStatus::Pending.as_str())
        .bind(now)
        .bind(TraceRegistrationStatus::Registering.as_str())
        .execute(&mut *tx)
        .await?;
        // `uploading` → `retrying` so the uploader's drain (which filters on
        // `Queued | Retrying`) re-picks it up. Stays inside the registered
        // half of the pipeline because the session URI is still valid
        // (`registration_status` is preserved by definition).
        let upload_result = sqlx::query(
            "UPDATE traces \
                SET upload_status = ?1, last_updated = ?2 \
              WHERE upload_status = ?3",
        )
        .bind(TraceUploadStatus::Retrying.as_str())
        .bind(now)
        .bind(TraceUploadStatus::Uploading.as_str())
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(reg_result.rows_affected() + upload_result.rows_affected())
    }

    async fn mark_stale_writing_traces_failed(
        &self,
        stale_threshold_secs: i64,
    ) -> Result<u64, StateStoreError> {
        let mut tx = self.write_pool.begin().await?;
        let now = Utc::now().naive_utc();
        let cutoff = now - chrono::Duration::seconds(stale_threshold_secs);
        let result = sqlx::query(
            "UPDATE traces \
                SET write_status = ?1, \
                    error_code = COALESCE(error_code, ?2), \
                    error_message = COALESCE(error_message, ?3), \
                    last_updated = ?4 \
              WHERE write_status IN (?5, ?6, ?7) \
                AND last_updated <= ?8",
        )
        .bind(TraceWriteStatus::Failed.as_str())
        .bind(TraceErrorCode::WriteFailed.as_str())
        .bind("daemon exited before encoding finished")
        .bind(now)
        .bind(TraceWriteStatus::Writing.as_str())
        .bind(TraceWriteStatus::Initializing.as_str())
        .bind(TraceWriteStatus::PendingMetadata.as_str())
        .bind(cutoff)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(result.rows_affected())
    }

    async fn cancel_recording(
        &self,
        recording_index: i64,
        stop_timestamp_ns: i64,
    ) -> Result<(RecordingRow, u64), StateStoreError> {
        let mut tx = self.write_pool.begin().await?;

        let now = Utc::now().naive_utc();
        sqlx::query(
            "UPDATE recordings \
                SET cancelled_at = COALESCE(cancelled_at, ?2), \
                    stop_timestamp_ns = COALESCE(stop_timestamp_ns, ?4), \
                    progress_reported = ?3, \
                    last_updated = ?2 \
              WHERE recording_index = ?1",
        )
        .bind(recording_index)
        .bind(now)
        .bind(ProgressReportStatus::Reported.as_str())
        .bind(stop_timestamp_ns)
        .execute(&mut *tx)
        .await?;

        // Burn every non-terminal trace so the registration / upload /
        // progress coordinators ignore them. `failed` is the existing
        // terminal label for all three pipelines; tagging with the
        // recording-cancelled error code lets operators distinguish a
        // user-cancel from an actual write or upload failure.
        let write_result = sqlx::query(
            "UPDATE traces \
                SET write_status = ?1, \
                    error_code = ?2, \
                    error_message = COALESCE(error_message, ?3), \
                    last_updated = ?4 \
              WHERE recording_index = ?5 \
                AND write_status NOT IN (?6, ?1)",
        )
        .bind(TraceWriteStatus::Failed.as_str())
        .bind(TraceErrorCode::RecordingCancelled.as_str())
        .bind("recording cancelled by producer")
        .bind(now)
        .bind(recording_index)
        .bind(TraceWriteStatus::Written.as_str())
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "UPDATE traces \
                SET upload_status = ?1, last_updated = ?2 \
              WHERE recording_index = ?3 \
                AND upload_status NOT IN (?1, ?4)",
        )
        .bind(TraceUploadStatus::Failed.as_str())
        .bind(now)
        .bind(recording_index)
        .bind(TraceUploadStatus::Uploaded.as_str())
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "UPDATE traces \
                SET registration_status = ?1, last_updated = ?2 \
              WHERE recording_index = ?3 \
                AND registration_status NOT IN (?1, ?4)",
        )
        .bind(TraceRegistrationStatus::Failed.as_str())
        .bind(now)
        .bind(recording_index)
        .bind(TraceRegistrationStatus::Registered.as_str())
        .execute(&mut *tx)
        .await?;

        let record = Self::fetch_recording_locked(&mut tx, recording_index)
            .await?
            .ok_or(sqlx::Error::RowNotFound)?;

        tx.commit().await?;
        Ok((record, write_result.rows_affected()))
    }

    async fn delete_recording_cascade(&self, recording_index: i64) -> Result<u64, StateStoreError> {
        let mut tx = self.write_pool.begin().await?;
        let traces_deleted = sqlx::query("DELETE FROM traces WHERE recording_index = ?1")
            .bind(recording_index)
            .execute(&mut *tx)
            .await?
            .rows_affected();
        sqlx::query("DELETE FROM recordings WHERE recording_index = ?1")
            .bind(recording_index)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(traces_deleted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn open_store() -> (SqliteStateStore, TempDir) {
        let tempdir = TempDir::new().expect("tempdir");
        let path = tempdir.path().join("state.db");
        let store = SqliteStateStore::open(&path).await.expect("open store");
        (store, tempdir)
    }

    /// Insert a recording for `(robot-1, instance)` and return its index.
    async fn seed_recording(store: &SqliteStateStore, instance: i64) -> i64 {
        store
            .create_recording(NewRecording {
                robot_id: Some("robot-1"),
                robot_instance: Some(instance),
                dataset_id: Some("ds-1"),
                start_timestamp_ns: 1_700_000_000_000_000_000,
            })
            .await
            .expect("create_recording")
            .recording_index
    }

    #[tokio::test]
    async fn create_recording_allocates_increasing_indices() {
        let (store, _tempdir) = open_store().await;
        let first = seed_recording(&store, 0).await;
        let second = seed_recording(&store, 1).await;
        assert!(
            second > first,
            "recording_index must increase: {first} {second}"
        );

        let row = store.get_recording(first).await.unwrap().unwrap();
        assert_eq!(row.recording_index, first);
        assert_eq!(row.recording_id, None, "cloud id starts NULL");
        assert_eq!(row.robot_id.as_deref(), Some("robot-1"));
        assert_eq!(row.robot_instance, Some(0));
    }

    #[tokio::test]
    async fn recordings_for_source_orders_by_index() {
        let (store, _tempdir) = open_store().await;
        let first = seed_recording(&store, 0).await;
        let second = seed_recording(&store, 0).await;
        // A different instance must not be returned.
        seed_recording(&store, 9).await;

        let rows = store.recordings_for_source("robot-1", 0).await.unwrap();
        let indices: Vec<i64> = rows.iter().map(|row| row.recording_index).collect();
        assert_eq!(indices, vec![first, second]);
    }

    #[tokio::test]
    async fn cloud_id_and_start_notify_lifecycle() {
        let (store, _tempdir) = open_store().await;
        let index = seed_recording(&store, 0).await;

        // Pending until notified/failed.
        let pending = store.recordings_pending_start_notify().await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].recording_index, index);

        // Start notify stamps both the cloud id and the notified timestamp,
        // and is idempotent: a second call cannot clobber the persisted id.
        let row = store
            .mark_recording_start_notified(index, "cloud-rec-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.recording_id.as_deref(), Some("cloud-rec-1"));
        assert!(row.backend_start_notified_at.is_some());
        assert!(store
            .recordings_pending_start_notify()
            .await
            .unwrap()
            .is_empty());

        let row = store
            .mark_recording_start_notified(index, "other-id")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.recording_id.as_deref(), Some("cloud-rec-1"));
    }

    #[tokio::test]
    async fn stop_notify_sweep_requires_a_cloud_id() {
        let (store, _tempdir) = open_store().await;
        let index = seed_recording(&store, 0).await;
        store.mark_recording_stopped(index, 2).await.unwrap();
        // Stopped but no cloud id yet → not eligible for the stop sweep.
        assert!(store
            .recordings_pending_stop_notify()
            .await
            .unwrap()
            .is_empty());

        store
            .mark_recording_start_notified(index, "cloud-rec-1")
            .await
            .unwrap();
        let pending = store.recordings_pending_stop_notify().await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].recording_index, index);
    }

    #[tokio::test]
    async fn open_creates_schema_and_applies_wal() {
        let (store, _tempdir) = open_store().await;

        let journal_mode: String = sqlx::query_scalar("PRAGMA journal_mode")
            .fetch_one(store.pool())
            .await
            .expect("journal_mode");
        assert_eq!(journal_mode.to_lowercase(), "wal");

        // `synchronous=NORMAL` is the numeric `1` from SQLite's PRAGMA result.
        let synchronous: i64 = sqlx::query_scalar("PRAGMA synchronous")
            .fetch_one(store.pool())
            .await
            .expect("synchronous");
        assert_eq!(synchronous, 1);

        let tables: Vec<String> =
            sqlx::query_scalar("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
                .fetch_all(store.pool())
                .await
                .expect("tables");
        assert!(tables.contains(&"recordings".to_string()));
        assert!(tables.contains(&"traces".to_string()));
    }

    #[tokio::test]
    async fn create_trace_inserts_trace_rows() {
        let (store, _tempdir) = open_store().await;
        let index = seed_recording(&store, 0).await;

        let trace = store
            .create_trace(index, "trace-1", Some("video"), None)
            .await
            .expect("create_trace");
        assert_eq!(trace.trace_id, "trace-1");
        assert_eq!(trace.recording_index, index);
        assert_eq!(trace.write_status, TraceWriteStatus::Initializing);
        assert_eq!(trace.data_type.as_deref(), Some("video"));

        // Creating the same trace twice is a no-op (write_status preserved).
        let again = store
            .create_trace(index, "trace-1", Some("video"), None)
            .await
            .expect("idempotent create_trace");
        assert_eq!(again.trace_id, "trace-1");
        let traces = store
            .list_traces_for_recording(index)
            .await
            .expect("list_traces");
        assert_eq!(traces.len(), 1);
    }

    #[tokio::test]
    async fn update_trace_overwrites_only_set_fields() {
        let (store, _tempdir) = open_store().await;
        let index = seed_recording(&store, 0).await;
        store
            .create_trace(index, "trace-1", None, None)
            .await
            .expect("create_trace");

        store
            .update_trace(
                "trace-1",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Writing),
                    bytes_written: Some(2048),
                    ..TraceUpdate::default()
                },
            )
            .await
            .expect("update_trace");
        let updated = store.get_trace("trace-1").await.unwrap().unwrap();
        assert_eq!(updated.write_status, TraceWriteStatus::Writing);
        assert_eq!(updated.bytes_written, 2048);
        // Unset fields keep their prior values.
        assert_eq!(updated.bytes_uploaded, 0);
        assert_eq!(updated.upload_status, TraceUploadStatus::Pending);

        // Updating an unknown trace is a harmless no-op (no row created).
        store
            .update_trace(
                "unknown-trace",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Failed),
                    ..TraceUpdate::default()
                },
            )
            .await
            .expect("update_trace");
        assert!(store.get_trace("unknown-trace").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn claim_for_registration_respects_size_trigger() {
        let (store, _tempdir) = open_store().await;
        let recording_index = seed_recording(&store, 0).await;
        for index in 0..5 {
            let trace_id = format!("trace-{index}");
            store
                .create_trace(recording_index, &trace_id, None, None)
                .await
                .expect("create_trace");
            store
                .update_trace(
                    &trace_id,
                    TraceUpdate {
                        write_status: Some(TraceWriteStatus::Written),
                        ..TraceUpdate::default()
                    },
                )
                .await
                .expect("update_trace");
        }

        // Size trigger: 5 ≥ limit, so all 3 are claimed even though
        // last_updated is fresh.
        let claimed = store
            .claim_traces_for_registration(3, 60.0)
            .await
            .expect("claim_traces");
        assert_eq!(claimed.len(), 3);
        for trace in &claimed {
            assert_eq!(
                trace.registration_status,
                TraceRegistrationStatus::Registering
            );
        }

        // The remaining two are too fresh for the age trigger so are not
        // claimed on a second call with a smaller limit.
        let second = store
            .claim_traces_for_registration(50, 60.0)
            .await
            .expect("claim_traces");
        assert!(
            second.is_empty(),
            "expected no age-eligible traces, got {second:?}"
        );
    }

    #[tokio::test]
    async fn reset_stale_pipeline_states_rearms_registering_and_uploading() {
        let (store, _tempdir) = open_store().await;
        // Three recordings to make sure the sweep doesn't leak across rows.
        for (instance, trace_id, reg, upload) in [
            (
                0,
                "trace-reg",
                TraceRegistrationStatus::Registering,
                TraceUploadStatus::Pending,
            ),
            (
                1,
                "trace-up",
                TraceRegistrationStatus::Registered,
                TraceUploadStatus::Uploading,
            ),
            (
                2,
                "trace-clean",
                TraceRegistrationStatus::Registered,
                TraceUploadStatus::Queued,
            ),
        ] {
            let recording_index = seed_recording(&store, instance).await;
            store
                .create_trace(
                    recording_index,
                    trace_id,
                    Some("JOINT_POSITIONS"),
                    Some("arm"),
                )
                .await
                .unwrap();
            store
                .update_trace(
                    trace_id,
                    TraceUpdate {
                        write_status: Some(TraceWriteStatus::Written),
                        registration_status: Some(reg),
                        upload_status: Some(upload),
                        ..TraceUpdate::default()
                    },
                )
                .await
                .unwrap();
        }

        let touched = store.reset_stale_pipeline_states().await.unwrap();
        assert_eq!(
            touched, 2,
            "registering + uploading rows should be re-armed"
        );

        let reg = store.get_trace("trace-reg").await.unwrap().unwrap();
        assert_eq!(reg.registration_status, TraceRegistrationStatus::Pending);
        let up = store.get_trace("trace-up").await.unwrap().unwrap();
        assert_eq!(up.upload_status, TraceUploadStatus::Retrying);
        let clean = store.get_trace("trace-clean").await.unwrap().unwrap();
        // Untouched rows keep their state — the sweep is targeted.
        assert_eq!(
            clean.registration_status,
            TraceRegistrationStatus::Registered
        );
        assert_eq!(clean.upload_status, TraceUploadStatus::Queued);
    }

    #[tokio::test]
    async fn mark_stale_writing_traces_failed_burns_old_rows_only() {
        let (store, _tempdir) = open_store().await;
        let recording_index = seed_recording(&store, 0).await;
        for (trace_id, write_status) in [
            ("fresh-writing", TraceWriteStatus::Writing),
            ("stale-writing", TraceWriteStatus::Writing),
            ("stale-initializing", TraceWriteStatus::Initializing),
            ("stale-pending-meta", TraceWriteStatus::PendingMetadata),
            ("done", TraceWriteStatus::Written),
            ("failed", TraceWriteStatus::Failed),
        ] {
            store
                .create_trace(
                    recording_index,
                    trace_id,
                    Some("JOINT_POSITIONS"),
                    Some("arm"),
                )
                .await
                .unwrap();
            store
                .update_trace(
                    trace_id,
                    TraceUpdate {
                        write_status: Some(write_status),
                        ..TraceUpdate::default()
                    },
                )
                .await
                .unwrap();
        }
        // Backdate the three "stale-*" rows to ~5 minutes ago by stamping
        // last_updated directly. SQLite stores `DATETIME` as ISO8601 text,
        // which `NaiveDateTime` serialises automatically.
        let stale_at = Utc::now().naive_utc() - chrono::Duration::seconds(300);
        for trace_id in ["stale-writing", "stale-initializing", "stale-pending-meta"] {
            sqlx::query("UPDATE traces SET last_updated = ?1 WHERE trace_id = ?2")
                .bind(stale_at)
                .bind(trace_id)
                .execute(store.write_pool())
                .await
                .unwrap();
        }

        let touched = store.mark_stale_writing_traces_failed(30).await.unwrap();
        assert_eq!(touched, 3, "only stale writing-side rows should be touched");

        for trace_id in ["stale-writing", "stale-initializing", "stale-pending-meta"] {
            let row = store.get_trace(trace_id).await.unwrap().unwrap();
            assert_eq!(row.write_status, TraceWriteStatus::Failed);
            assert_eq!(row.error_code, Some(TraceErrorCode::WriteFailed));
        }
        // Fresh + already-terminal rows are not touched.
        let fresh = store.get_trace("fresh-writing").await.unwrap().unwrap();
        assert_eq!(fresh.write_status, TraceWriteStatus::Writing);
        let done = store.get_trace("done").await.unwrap().unwrap();
        assert_eq!(done.write_status, TraceWriteStatus::Written);
        let failed = store.get_trace("failed").await.unwrap().unwrap();
        assert_eq!(failed.error_code, None, "pre-existing rows untouched");
    }

    #[tokio::test]
    async fn cancel_recording_burns_traces_and_stamps_cancelled_at() {
        let (store, _tempdir) = open_store().await;
        let recording_index = seed_recording(&store, 0).await;
        for (trace_id, write, upload, reg) in [
            (
                "in-flight",
                TraceWriteStatus::Writing,
                TraceUploadStatus::Pending,
                TraceRegistrationStatus::Pending,
            ),
            (
                "registered-queued",
                TraceWriteStatus::Written,
                TraceUploadStatus::Queued,
                TraceRegistrationStatus::Registered,
            ),
            (
                "already-uploaded",
                TraceWriteStatus::Written,
                TraceUploadStatus::Uploaded,
                TraceRegistrationStatus::Registered,
            ),
        ] {
            store
                .create_trace(recording_index, trace_id, Some("JOINT_POSITIONS"), None)
                .await
                .unwrap();
            store
                .update_trace(
                    trace_id,
                    TraceUpdate {
                        write_status: Some(write),
                        upload_status: Some(upload),
                        registration_status: Some(reg),
                        ..TraceUpdate::default()
                    },
                )
                .await
                .unwrap();
        }
        // A trace belonging to another recording must not be touched.
        let other_index = seed_recording(&store, 9).await;
        store
            .create_trace(other_index, "untouched", Some("JOINT_POSITIONS"), None)
            .await
            .unwrap();

        let (row, touched) = store
            .cancel_recording(recording_index, 5_000_000_000)
            .await
            .unwrap();
        assert!(row.cancelled_at.is_some(), "cancelled_at must be stamped");
        assert_eq!(
            row.stop_timestamp_ns,
            Some(5_000_000_000),
            "a cancel stamps stop_timestamp_ns like a stop"
        );
        assert_eq!(row.progress_reported, ProgressReportStatus::Reported);
        assert_eq!(touched, 1, "only the non-Written trace's write was touched");

        let in_flight = store.get_trace("in-flight").await.unwrap().unwrap();
        assert_eq!(in_flight.write_status, TraceWriteStatus::Failed);
        assert_eq!(
            in_flight.error_code,
            Some(TraceErrorCode::RecordingCancelled)
        );
        assert_eq!(in_flight.upload_status, TraceUploadStatus::Failed);

        let queued = store.get_trace("registered-queued").await.unwrap().unwrap();
        assert_eq!(queued.upload_status, TraceUploadStatus::Failed);
        assert_eq!(queued.write_status, TraceWriteStatus::Written);

        let uploaded = store.get_trace("already-uploaded").await.unwrap().unwrap();
        assert_eq!(uploaded.upload_status, TraceUploadStatus::Uploaded);
        assert_eq!(
            uploaded.registration_status,
            TraceRegistrationStatus::Registered
        );

        let other = store.get_trace("untouched").await.unwrap().unwrap();
        assert_eq!(other.write_status, TraceWriteStatus::Initializing);
        assert_eq!(other.upload_status, TraceUploadStatus::Pending);
    }

    #[tokio::test]
    async fn cancel_recording_is_idempotent() {
        let (store, _tempdir) = open_store().await;
        let recording_index = seed_recording(&store, 0).await;
        store
            .create_trace(recording_index, "trace-1", Some("JOINT_POSITIONS"), None)
            .await
            .unwrap();

        let (first, _) = store
            .cancel_recording(recording_index, 5_000_000_000)
            .await
            .unwrap();
        let first_at = first.cancelled_at.expect("cancelled_at set");
        // Sleep across a clock tick to make a date change observable.
        std::thread::sleep(std::time::Duration::from_millis(10));
        let (second, _) = store
            .cancel_recording(recording_index, 9_000_000_000)
            .await
            .unwrap();
        assert_eq!(
            second.stop_timestamp_ns,
            Some(5_000_000_000),
            "subsequent cancels must not slide stop_timestamp_ns forward"
        );
        assert_eq!(
            second.cancelled_at,
            Some(first_at),
            "subsequent cancels must not slide cancelled_at forward"
        );
    }

    #[tokio::test]
    async fn claim_for_registration_respects_age_trigger() {
        let (store, _tempdir) = open_store().await;
        let recording_index = seed_recording(&store, 0).await;
        store
            .create_trace(recording_index, "trace-1", None, None)
            .await
            .expect("create_trace");
        store
            .update_trace(
                "trace-1",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    ..TraceUpdate::default()
                },
            )
            .await
            .expect("update_trace");

        // Age trigger: max_wait_secs = 0 ⇒ any row at or before "now" is
        // eligible. We pass a generous limit so the size trigger doesn't fire.
        let claimed = store
            .claim_traces_for_registration(50, 0.0)
            .await
            .expect("claim_traces");
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].trace_id, "trace-1");
        assert_eq!(
            claimed[0].registration_status,
            TraceRegistrationStatus::Registering
        );
    }

    #[tokio::test]
    async fn apply_trace_writes_records_upload_progress_until_settled() {
        let (store, _tempdir) = open_store().await;
        let index = seed_recording(&store, 0).await;
        store
            .create_trace(index, "trace-up", Some("RGB"), None)
            .await
            .unwrap();
        // Write phase done, upload phase begins.
        store
            .update_trace(
                "trace-up",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Written),
                    upload_status: Some(TraceUploadStatus::Uploading),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();

        // A rolling checkpoint applies while the upload is live — and it does so
        // even though the row is already `written`, which the write-phase guard
        // forbids (so it must travel on its own statement).
        store
            .apply_trace_writes(&[CoalescedTraceWrite {
                trace_id: "trace-up".to_string(),
                bytes_uploaded: Some(4_000_000),
                ..Default::default()
            }])
            .await
            .unwrap();
        assert_eq!(
            store
                .get_trace("trace-up")
                .await
                .unwrap()
                .unwrap()
                .bytes_uploaded,
            4_000_000
        );

        // Once the upload settles, a late/duplicate coalesced checkpoint must
        // not touch the terminal row (it would otherwise rewind the final
        // byte count the synchronous finalise wrote).
        store
            .update_trace(
                "trace-up",
                TraceUpdate {
                    upload_status: Some(TraceUploadStatus::Uploaded),
                    bytes_uploaded: Some(8_000_000),
                    ..TraceUpdate::default()
                },
            )
            .await
            .unwrap();
        store
            .apply_trace_writes(&[CoalescedTraceWrite {
                trace_id: "trace-up".to_string(),
                bytes_uploaded: Some(1),
                ..Default::default()
            }])
            .await
            .unwrap();
        assert_eq!(
            store
                .get_trace("trace-up")
                .await
                .unwrap()
                .unwrap()
                .bytes_uploaded,
            8_000_000,
            "a late checkpoint must not touch a settled upload"
        );
    }

    #[tokio::test]
    async fn recordings_pending_reclaim_only_returns_settled_recordings() {
        let (store, _tempdir) = open_store().await;

        // Helper: drive a stopped recording with one trace at `upload` status
        // through the full notify/progress gates.
        async fn stopped_with_trace(
            store: &SqliteStateStore,
            instance: i64,
            trace_id: &str,
            upload: TraceUploadStatus,
        ) -> i64 {
            let index = seed_recording(store, instance).await;
            store
                .mark_recording_start_notified(index, &format!("cloud-{instance}"))
                .await
                .unwrap();
            store
                .create_trace(index, trace_id, Some("J"), None)
                .await
                .unwrap();
            store
                .update_trace(
                    trace_id,
                    TraceUpdate {
                        write_status: Some(TraceWriteStatus::Written),
                        upload_status: Some(upload),
                        ..TraceUpdate::default()
                    },
                )
                .await
                .unwrap();
            store.mark_recording_stopped(index, 1).await.unwrap();
            store.mark_recording_stop_notified(index).await.unwrap();
            store.set_expected_trace_count(index, 1).await.unwrap();
            store
                .set_progress_report_status(
                    index,
                    ProgressReportStatus::Pending,
                    ProgressReportStatus::Reported,
                )
                .await
                .unwrap();
            index
        }

        // A: stopped + fully uploaded → reclaimable.
        let uploaded = stopped_with_trace(&store, 0, "a1", TraceUploadStatus::Uploaded).await;
        // B: stopped, settled at the recording level, but one trace failed to
        // upload → must NOT be reclaimable (and must not be re-scanned forever).
        let failed = stopped_with_trace(&store, 1, "b1", TraceUploadStatus::Failed).await;
        // C: cancelled + backend notified → reclaimable with no trace scan.
        let cancelled = seed_recording(&store, 2).await;
        store
            .mark_recording_start_notified(cancelled, "cloud-c")
            .await
            .unwrap();
        store.cancel_recording(cancelled, 1).await.unwrap();
        store
            .mark_recording_cancel_notified(cancelled)
            .await
            .unwrap();
        // D: live (never stopped) → not reclaimable.
        let live = seed_recording(&store, 3).await;

        let reclaimable: Vec<i64> = store
            .recordings_pending_reclaim()
            .await
            .unwrap()
            .iter()
            .map(|row| row.recording_index)
            .collect();
        assert!(
            reclaimable.contains(&uploaded),
            "fully-uploaded stopped recording reclaims"
        );
        assert!(
            reclaimable.contains(&cancelled),
            "cancelled+notified recording reclaims"
        );
        assert!(
            !reclaimable.contains(&failed),
            "a permanently-failed trace blocks reclaim (M10: no perpetual re-scan)"
        );
        assert!(
            !reclaimable.contains(&live),
            "a live recording never reclaims"
        );
    }

    #[tokio::test]
    async fn delete_recording_cascade_removes_recording_and_its_traces_only() {
        let (store, _tempdir) = open_store().await;
        let index = seed_recording(&store, 0).await;
        for trace_id in ["t-1", "t-2"] {
            store
                .create_trace(index, trace_id, Some("JOINT_POSITIONS"), None)
                .await
                .unwrap();
        }
        // A sibling recording + trace must survive the cascade.
        let other = seed_recording(&store, 9).await;
        store
            .create_trace(other, "keep", Some("JOINT_POSITIONS"), None)
            .await
            .unwrap();

        let deleted = store.delete_recording_cascade(index).await.unwrap();
        assert_eq!(deleted, 2, "both traces of the recording are deleted");

        assert!(store.get_recording(index).await.unwrap().is_none());
        assert!(store.get_trace("t-1").await.unwrap().is_none());
        assert!(store.get_trace("t-2").await.unwrap().is_none());

        // Sibling untouched.
        assert!(store.get_recording(other).await.unwrap().is_some());
        assert!(store.get_trace("keep").await.unwrap().is_some());
    }
}
